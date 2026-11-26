import re
import pickle
import numpy as np
from scipy import sparse
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = "best_mercari_v2.pt"
RIDGE_PATH = "ridge_v2.pkl"
ARTIFACTS_PATH = "artifacts_v2.pkl"
VAL_RMSLE = 0.3950
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="PriceLens Multimodal v2 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BUNDLE_RE = re.compile(r"\b(?:lot of|bundle|set of|pack of|pair of|\d+\s*pack|\d+\s*pcs)\b", re.IGNORECASE)
RETAIL_RE = re.compile(r"\[rm\]|retail|msrp", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = text.lower().replace("[rm]", " price_removed ")
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class MercariMultimodalV2(nn.Module):
    def __init__(self, title_dim: int, desc_dim: int, vocab_sizes: dict, cat_cols: list, num_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList()
        total_emb_dim = 0
        for col in cat_cols:
            num_classes = vocab_sizes[col]
            emb_dim = min(32, max(4, (num_classes + 1) // 2))
            self.embeddings.append(nn.Embedding(num_classes, emb_dim))
            total_emb_dim += emb_dim
        self.emb_drop = nn.Dropout(0.20)

        self.title_net = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(title_dim, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.50),
        )
        self.desc_net = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(desc_dim, 192),
            nn.BatchNorm1d(192),
            nn.SiLU(),
            nn.Dropout(0.50),
        )
        self.num_net = nn.Sequential(
            nn.Linear(num_dim, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )

        fusion_dim = 192 + 192 + total_emb_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 160),
            nn.BatchNorm1d(160),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(160, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_title, x_desc, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_feat = self.emb_drop(torch.cat(emb_outs, dim=1))
        fused = torch.cat(
            [self.title_net(x_title), self.desc_net(x_desc), cat_feat, self.num_net(x_num)],
            dim=1,
        )
        return self.head(fused).squeeze(1)


print("Loading v2 artifacts, Ridge model, and PyTorch weights...")
with open(ARTIFACTS_PATH, "rb") as f:
    art = pickle.load(f)

vec_title_word = art["vec_title_word"]
vec_title_char = art["vec_title_char"]
vec_desc = art["vec_desc"]
valid_brands_set = art["valid_brands"]
vocab_maps = art["vocab_maps"]
vocab_sizes = art["vocab_sizes"]
ohe = art["ohe"]
te_maps = art["te_maps"]
global_mean = art["global_mean"]
scaler_num = art["scaler_num"]
cat_cols = art["cat_cols"]
blend_w = art["blend_weight"]

with open(RIDGE_PATH, "rb") as f:
    ridge = pickle.load(f)

valid_brands = sorted(list(valid_brands_set - {"rare_brand", "missing"}))

category_tree: dict[str, dict[str, list[str]]] = {}
for full_cat in vocab_maps["category_name"].keys():
    if full_cat == "missing/missing/missing":
        continue
    parts = (full_cat.split("/", 2) + ["missing", "missing"])[:3]
    c0, c1, c2 = parts
    category_tree.setdefault(c0, {}).setdefault(c1, []).append(c2)

for c0 in category_tree:
    for c1 in category_tree[c0]:
        category_tree[c0][c1] = sorted(list(set(category_tree[c0][c1])))

title_dim = len(vec_title_word.vocabulary_) + len(vec_title_char.vocabulary_)
desc_dim = len(vec_desc.vocabulary_)
num_dim = len(art["num_cols"])

model = MercariMultimodalV2(title_dim, desc_dim, vocab_sizes, cat_cols, num_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print(f"PriceLens v2 Ensemble ({VAL_RMSLE:.4f} RMSLE) loaded on {DEVICE} and ready!")


class ListingInput(BaseModel):
    name: str
    brand: str = ""
    cat_main: str
    cat_sub1: str
    cat_sub2: str
    condition_id: int = 1
    shipping: int = 0


@app.get("/options")
def get_options():
    return {
        "category_tree": category_tree,
        "brands": valid_brands,
    }


@app.post("/predict")
def predict(item: ListingInput):
    raw_name = item.name.strip()

    has_brand = 1.0 if item.brand.strip() else 0.0
    has_desc = 0.0  # Fixed to missing prior (no free-text description input)

    brand_str = item.brand.strip() if has_brand else "missing"
    brand_clean = brand_str if brand_str in valid_brands_set else "rare_brand"

    cat_str = f"{item.cat_main}/{item.cat_sub1}/{item.cat_sub2}"
    c_name = clean_text(raw_name if raw_name else "missing")
    c_desc = "missing"

    # 1. Domain numerical features (exact order from train_v2.py)
    name_len = float(len(raw_name))
    desc_len = 0.0
    name_words = float(len(raw_name.split()))
    desc_words = 0.0
    name_upper_ratio = sum(1 for c in raw_name if c.isupper()) / (name_len + 1.0)

    is_bundle = 1.0 if BUNDLE_RE.search(raw_name) else 0.0
    mentions_retail = 1.0 if RETAIL_RE.search(raw_name) else 0.0

    brand_te = float(te_maps["brand_clean"].get(brand_clean, global_mean))
    cat_te = float(te_maps["category_name"].get(cat_str, global_mean))

    raw_num = np.array([[
        has_brand, has_desc, name_len, desc_len, name_words,
        desc_words, name_upper_ratio, is_bundle, mentions_retail,
        brand_te, cat_te
    ]], dtype=np.float32)
    x_num_np = scaler_num.transform(raw_num).astype(np.float32)

    # 2. Separate Title (word + char) and fixed missing Description TF-IDF
    title_ctx = f"{c_name} | {brand_clean.lower()} | {item.cat_sub2.lower()}"
    x_title_sp = sparse.hstack(
        [vec_title_word.transform([title_ctx]), vec_title_char.transform([c_name])],
        format="csr",
    )
    x_desc_sp = vec_desc.transform([c_desc]).tocsr()

    # 3. Categorical indices
    cat_vals = np.array([[
        vocab_maps["brand_clean"].get(brand_clean, 0),
        vocab_maps["category_name"].get(cat_str, 0),
        vocab_maps["cat_main"].get(item.cat_main, 0),
        vocab_maps["cat_sub1"].get(item.cat_sub1, 0),
        vocab_maps["cat_sub2"].get(item.cat_sub2, 0),
        vocab_maps["item_condition_id"].get(item.condition_id, 0),
        vocab_maps["shipping"].get(item.shipping, 0),
    ]], dtype=np.int64)

    # 4. Neural Network forward pass
    x_title_t = torch.from_numpy(x_title_sp.toarray()).to(DEVICE)
    x_desc_t = torch.from_numpy(x_desc_sp.toarray()).to(DEVICE)
    x_cat_t = torch.from_numpy(cat_vals).to(DEVICE)
    x_num_t = torch.from_numpy(x_num_np).to(DEVICE)

    with torch.no_grad():
        nn_log_pred = max(0.0, model(x_title_t, x_desc_t, x_cat_t, x_num_t).item())

    # 5. Ridge Regression prediction + Ensemble Blend
    x_ridge_sp = sparse.hstack(
        [x_title_sp, x_desc_sp, ohe.transform(cat_vals), sparse.csr_matrix(x_num_np)],
        format="csr",
    )
    ridge_log_pred = max(0.0, float(ridge.predict(x_ridge_sp)[0]))

    log_pred = blend_w * nn_log_pred + (1.0 - blend_w) * ridge_log_pred

    half_sigma = 0.5 * VAL_RMSLE
    point_price = float(np.expm1(log_pred))
    low_price = float(max(1.0, np.expm1(max(0.0, log_pred - half_sigma))))
    high_price = float(np.expm1(log_pred + half_sigma))

    return {
        "estimated_price": round(point_price, 2),
        "range_low": round(low_price, 2),
        "range_high": round(high_price, 2),
        "recognized_brand": brand_clean not in ("missing", "rare_brand"),
    }