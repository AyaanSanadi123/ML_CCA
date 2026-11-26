import gc
import re
import time
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn

DATA_PATH = "train.tsv"
MODEL_PATH = "best_mercari_v2.pt"
RIDGE_PATH = "ridge_v2.pkl"
ARTIFACTS_OUT = "artifacts_v2.pkl"
MIN_BRAND_FREQ = 5
BLEND_W = 0.82
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text: str) -> str:
    text = text.lower().replace("[rm]", " price_removed ")
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# ==============================================================================
# 1. REPRODUCE DETERMINISTIC PREPROCESSING & FIT ARTIFACTS
# ==============================================================================
t0 = time.time()
print("[1/3] Reconstructing deterministic v2 feature transformers...")
df = pd.read_csv(DATA_PATH, sep="\t")
df = df[df["price"] > 0].reset_index(drop=True)
df["log_price"] = np.log1p(df["price"].values).astype(np.float32)

df["has_brand"] = df["brand_name"].notna().astype(np.float32)
no_desc = df["item_description"].isna() | (df["item_description"] == "No description yet")
df["has_desc"] = (~no_desc).astype(np.float32)

raw_name = df["name"].fillna("")
raw_desc = df["item_description"].fillna("").replace("No description yet", "")

df["name_len"] = raw_name.str.len().astype(np.float32)
df["desc_len"] = raw_desc.str.len().astype(np.float32)
df["name_words"] = raw_name.str.split().str.len().astype(np.float32)
df["desc_words"] = raw_desc.str.split().str.len().astype(np.float32)
df["name_upper_ratio"] = (
    raw_name.apply(lambda s: sum(1 for c in s if c.isupper())) / (df["name_len"] + 1.0)
).astype(np.float32)

combined_raw_lower = (raw_name + " " + raw_desc).str.lower()
df["is_bundle"] = combined_raw_lower.str.contains(
    r"\b(?:lot of|bundle|set of|pack of|pair of|\d+\s*pack|\d+\s*pcs)\b", regex=True
).astype(np.float32)
df["mentions_retail"] = combined_raw_lower.str.contains(r"\[rm\]|retail|msrp", regex=True).astype(np.float32)

df["brand_name"] = df["brand_name"].fillna("missing")
df["category_name"] = df["category_name"].fillna("missing/missing/missing")
cat_parts = df["category_name"].str.split("/", n=2, expand=True).fillna("missing")
df["cat_main"], df["cat_sub1"], df["cat_sub2"] = cat_parts[0], cat_parts[1], cat_parts[2]

df["clean_name"] = raw_name.replace("", "missing").apply(clean_text)
df["clean_desc"] = raw_desc.replace("", "missing").apply(clean_text)

del raw_name, raw_desc, combined_raw_lower, cat_parts, no_desc
gc.collect()

train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
del df
gc.collect()
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

brand_counts = train_df["brand_name"].value_counts()
valid_brands = set(brand_counts[brand_counts >= MIN_BRAND_FREQ].index)
for split_df in (train_df, val_df):
    split_df["brand_clean"] = split_df["brand_name"].where(split_df["brand_name"].isin(valid_brands), "rare_brand")

# Target encoding lookup maps
global_mean = float(train_df["log_price"].mean())
SMOOTHING = 20.0
te_maps = {}
for col in ["brand_clean", "category_name"]:
    full_stats = train_df.groupby(col)["log_price"].agg(["count", "mean"])
    smooth = (full_stats["count"] * full_stats["mean"] + SMOOTHING * global_mean) / (full_stats["count"] + SMOOTHING)
    te_maps[col] = smooth.to_dict()
    val_df[f"{col}_te"] = val_df[col].map(te_maps[col]).fillna(global_mean).astype(np.float32).values

# Fit TF-IDF vectorizers
train_df["title_ctx"] = (
    train_df["clean_name"] + " | " + train_df["brand_clean"].str.lower() + " | " + train_df["cat_sub2"].str.lower()
)
val_df["title_ctx"] = (
    val_df["clean_name"] + " | " + val_df["brand_clean"].str.lower() + " | " + val_df["cat_sub2"].str.lower()
)

vec_title_word = TfidfVectorizer(max_features=25_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)
vec_title_char = TfidfVectorizer(max_features=15_000, analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, dtype=np.float32)
vec_desc = TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)

vec_title_word.fit(train_df["title_ctx"])
X_va_tw = vec_title_word.transform(val_df["title_ctx"])

vec_title_char.fit(train_df["clean_name"])
X_va_tc = vec_title_char.transform(val_df["clean_name"])

X_val_title = sparse.hstack([X_va_tw, X_va_tc], format="csr")
del X_va_tw, X_va_tc

vec_desc.fit(train_df["clean_desc"])
X_val_desc = vec_desc.transform(val_df["clean_desc"]).tocsr()

# Categorical mappings & OneHotEncoder
cat_cols = ["brand_clean", "category_name", "cat_main", "cat_sub1", "cat_sub2", "item_condition_id", "shipping"]
vocab_maps, vocab_sizes = {}, {}
X_train_cat = np.zeros((len(train_df), len(cat_cols)), dtype=np.int64)
X_val_cat = np.zeros((len(val_df), len(cat_cols)), dtype=np.int64)

for i, col in enumerate(cat_cols):
    unique_vals = train_df[col].unique().tolist()
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    vocab_maps[col] = mapping
    vocab_sizes[col] = len(mapping) + 1
    X_train_cat[:, i] = train_df[col].map(mapping).fillna(0).values
    X_val_cat[:, i] = val_df[col].map(mapping).fillna(0).values

ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
ohe.fit(X_train_cat)
X_va_ohe = ohe.transform(X_val_cat)

# Fit StandardScaler (need OOF TE on train_df just to fit the exact scaler mean/std)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for col in ["brand_clean", "category_name"]:
    oof_mean = np.full(len(train_df), global_mean, dtype=np.float32)
    for tr_idx, oof_idx in kf.split(train_df):
        sub = train_df.iloc[tr_idx]
        stats = sub.groupby(col)["log_price"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + SMOOTHING * global_mean) / (stats["count"] + SMOOTHING)
        oof_mean[oof_idx] = train_df.iloc[oof_idx][col].map(smooth).fillna(global_mean).values
    train_df[f"{col}_te"] = oof_mean

num_cols = [
    "has_brand", "has_desc", "name_len", "desc_len", "name_words",
    "desc_words", "name_upper_ratio", "is_bundle", "mentions_retail",
    "brand_clean_te", "category_name_te"
]
scaler_num = StandardScaler()
scaler_num.fit(train_df[num_cols].values)
X_val_num = scaler_num.transform(val_df[num_cols].values).astype(np.float32)
y_val = val_df["log_price"].values.copy()

del train_df, val_df, X_train_cat
gc.collect()

# Save unified artifact bundle
artifacts = {
    "vec_title_word": vec_title_word,
    "vec_title_char": vec_title_char,
    "vec_desc": vec_desc,
    "valid_brands": valid_brands,
    "vocab_maps": vocab_maps,
    "vocab_sizes": vocab_sizes,
    "ohe": ohe,
    "te_maps": te_maps,
    "global_mean": global_mean,
    "scaler_num": scaler_num,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "blend_weight": BLEND_W,
}
with open(ARTIFACTS_OUT, "wb") as f:
    pickle.dump(artifacts, f)
print(f"[2/3] Saved all v2 preprocessing artifacts to '{ARTIFACTS_OUT}' in {time.time() - t0:.1f}s\n")

# ==============================================================================
# 2. VERIFY SAVED MODELS & COMPUTE FULL DOLLAR METRICS
# ==============================================================================
print("[3/3] Verifying saved NN + Ridge ensemble on validation set...")

class MercariMultimodalV2(nn.Module):
    def __init__(self, title_dim: int, desc_dim: int, vocab_sizes: dict, num_dim: int):
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
            nn.Dropout(0.25), nn.Linear(title_dim, 192), nn.BatchNorm1d(192), nn.SiLU(), nn.Dropout(0.50)
        )
        self.desc_net = nn.Sequential(
            nn.Dropout(0.25), nn.Linear(desc_dim, 192), nn.BatchNorm1d(192), nn.SiLU(), nn.Dropout(0.50)
        )
        self.num_net = nn.Sequential(nn.Linear(num_dim, 32), nn.BatchNorm1d(32), nn.SiLU())
        fusion_dim = 192 + 192 + total_emb_dim + 32
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 160), nn.BatchNorm1d(160), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(160, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def forward(self, x_title, x_desc, x_cat, x_num):
        emb_outs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_feat = self.emb_drop(torch.cat(emb_outs, dim=1))
        fused = torch.cat([self.title_net(x_title), self.desc_net(x_desc), cat_feat, self.num_net(x_num)], dim=1)
        return self.head(fused).squeeze(1)

model = MercariMultimodalV2(40_000, 40_000, vocab_sizes, len(num_cols)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

nn_preds = []
with torch.no_grad():
    for start in range(0, len(y_val), 2048):
        end = start + 2048
        xt = torch.from_numpy(X_val_title[start:end].toarray()).to(DEVICE)
        xd = torch.from_numpy(X_val_desc[start:end].toarray()).to(DEVICE)
        xc = torch.from_numpy(X_val_cat[start:end]).to(DEVICE)
        xn = torch.from_numpy(X_val_num[start:end]).to(DEVICE)
        nn_preds.append(model(xt, xd, xc, xn).cpu().numpy())
nn_preds = np.clip(np.concatenate(nn_preds), 0.0, None)

with open(RIDGE_PATH, "rb") as f:
    ridge = pickle.load(f)
X_va_ridge = sparse.hstack([X_val_title, X_val_desc, X_va_ohe, sparse.csr_matrix(X_val_num)], format="csr")
ridge_preds = np.clip(ridge.predict(X_va_ridge), 0.0, None)

blend_log = BLEND_W * nn_preds + (1.0 - BLEND_W) * ridge_preds
rmsle = np.sqrt(np.mean((blend_log - y_val) ** 2))

# Convert log-space back to actual USD ($)
actual_usd = np.expm1(y_val)
pred_usd = np.expm1(blend_log)
abs_err = np.abs(actual_usd - pred_usd)

print("\n================ FINAL DOLLAR-SPACE EVALUATION ================")
print(f"  Verified Validation RMSLE:       {rmsle:.4f}")
print(f"  Log-Space R² Score:              {r2_score(y_val, blend_log):.4f}")
print(f"  Dollar Mean Absolute Error (MAE): ${mean_absolute_error(actual_usd, pred_usd):.2f}")
print(f"  Median Dollar Error:             ${np.median(abs_err):.2f}")
print(f"  Predictions within ±$5.00:       {(abs_err <= 5.0).mean():.1%}")
print(f"  Predictions within ±$10.00:      {(abs_err <= 10.0).mean():.1%}")
print("===============================================================")