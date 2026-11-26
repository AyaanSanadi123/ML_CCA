import gc
import re
import time
import pickle
import queue
import threading
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn

# --- CONFIGURATION ---
DATA_PATH = "train.tsv"
MODEL_OUT = "best_mercari_v2.pt"
RIDGE_OUT = "ridge_v2.pkl"
MIN_BRAND_FREQ = 5

BATCH_SIZE = 1024        # Safe for WSL2 system RAM (~327 MB per dense batch)
VAL_BATCH_SIZE = 2048
MAX_EPOCHS = 200
PATIENCE = 7
INIT_LR = 2e-3
WEIGHT_DECAY = 1e-3

assert torch.cuda.is_available(), "CUDA GPU not detected!"
DEVICE = torch.device("cuda")
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
USE_BF16 = torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

torch.manual_seed(42)
np.random.seed(42)

print(f"GPU: {torch.cuda.get_device_name(0)} | Precision: {'bfloat16' if USE_BF16 else 'float16'} AMP + TF32\n")

# ==============================================================================
# 1. LOAD & ENGINEER DOMAIN FEATURES
# ==============================================================================
t_start = time.time()
print("[1/4] Loading raw dataset & engineering domain features...")
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
df["cat_main"] = cat_parts[0]
df["cat_sub1"] = cat_parts[1]
df["cat_sub2"] = cat_parts[2]

def clean_text(text: str) -> str:
    text = text.lower().replace("[rm]", " price_removed ")
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean_name"] = raw_name.replace("", "missing").apply(clean_text)
df["clean_desc"] = raw_desc.replace("", "missing").apply(clean_text)

# Free raw string series immediately
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

global_mean = float(train_df["log_price"].mean())
SMOOTHING = 20.0
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for col in ["brand_clean", "category_name"]:
    oof_mean = np.full(len(train_df), global_mean, dtype=np.float32)
    for tr_idx, oof_idx in kf.split(train_df):
        sub = train_df.iloc[tr_idx]
        stats = sub.groupby(col)["log_price"].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + SMOOTHING * global_mean) / (stats["count"] + SMOOTHING)
        oof_mean[oof_idx] = train_df.iloc[oof_idx][col].map(smooth).fillna(global_mean).values
    train_df[f"{col}_te"] = oof_mean

    full_stats = train_df.groupby(col)["log_price"].agg(["count", "mean"])
    full_smooth = (full_stats["count"] * full_stats["mean"] + SMOOTHING * global_mean) / (full_stats["count"] + SMOOTHING)
    val_df[f"{col}_te"] = val_df[col].map(full_smooth).fillna(global_mean).astype(np.float32).values

# ==============================================================================
# 2. VECTORIZE SEPARATE TITLE (WORD+CHAR), DESCRIPTION, CATEGORICALS & NUMERICALS
# ==============================================================================
print("[2/4] Vectorizing separate Title (word + char), Description, and Tabular features...")

for split_df in (train_df, val_df):
    split_df["title_ctx"] = (
        split_df["clean_name"] + " | " +
        split_df["brand_clean"].str.lower() + " | " +
        split_df["cat_sub2"].str.lower()
    )

vec_title_word = TfidfVectorizer(max_features=25_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)
vec_title_char = TfidfVectorizer(max_features=15_000, analyzer="char_wb", ngram_range=(3, 5), min_df=5, sublinear_tf=True, dtype=np.float32)
vec_desc = TfidfVectorizer(max_features=40_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True, dtype=np.float32)

X_tr_tw = vec_title_word.fit_transform(train_df["title_ctx"])
X_va_tw = vec_title_word.transform(val_df["title_ctx"])

X_tr_tc = vec_title_char.fit_transform(train_df["clean_name"])
X_va_tc = vec_title_char.transform(val_df["clean_name"])

X_train_title = sparse.hstack([X_tr_tw, X_tr_tc], format="csr")
X_val_title = sparse.hstack([X_va_tw, X_va_tc], format="csr")
del X_tr_tw, X_va_tw, X_tr_tc, X_va_tc
gc.collect()

X_train_desc = vec_desc.fit_transform(train_df["clean_desc"]).tocsr()
X_val_desc = vec_desc.transform(val_df["clean_desc"]).tocsr()

cat_cols = ["brand_clean", "category_name", "cat_main", "cat_sub1", "cat_sub2", "item_condition_id", "shipping"]
vocab_sizes = {}
X_train_cat = np.zeros((len(train_df), len(cat_cols)), dtype=np.int64)
X_val_cat = np.zeros((len(val_df), len(cat_cols)), dtype=np.int64)

for i, col in enumerate(cat_cols):
    unique_vals = train_df[col].unique().tolist()
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    vocab_sizes[col] = len(mapping) + 1
    X_train_cat[:, i] = train_df[col].map(mapping).fillna(0).values
    X_val_cat[:, i] = val_df[col].map(mapping).fillna(0).values

num_cols = [
    "has_brand", "has_desc", "name_len", "desc_len", "name_words",
    "desc_words", "name_upper_ratio", "is_bundle", "mentions_retail",
    "brand_clean_te", "category_name_te"
]
scaler_num = StandardScaler()
X_train_num = scaler_num.fit_transform(train_df[num_cols].values).astype(np.float32)
X_val_num = scaler_num.transform(val_df[num_cols].values).astype(np.float32)

y_train = train_df["log_price"].values.copy()
y_val = val_df["log_price"].values.copy()

# Free the DataFrames completely before Ridge & PyTorch training
del train_df, val_df
gc.collect()

print(f"Features ready in {time.time() - t_start:.1f}s -> Title TF-IDF: {X_train_title.shape[1]:,} | Desc TF-IDF: {X_train_desc.shape[1]:,} | Num: {X_train_num.shape[1]}\n")

# ==============================================================================
# 3. FIT LINEAR RIDGE BASELINE (FOR ENSEMBLE BLEND)
# ==============================================================================
print("[3/4] Fitting Sparse Ridge Regression baseline on CPU...")
t_ridge = time.time()
ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
X_tr_ohe = ohe.fit_transform(X_train_cat)
X_va_ohe = ohe.transform(X_val_cat)

X_tr_ridge = sparse.hstack([X_train_title, X_train_desc, X_tr_ohe, sparse.csr_matrix(X_train_num)], format="csr")
X_va_ridge = sparse.hstack([X_val_title, X_val_desc, X_va_ohe, sparse.csr_matrix(X_val_num)], format="csr")

ridge = Ridge(alpha=3.0, solver="lsqr", tol=1e-4, max_iter=1000)
ridge.fit(X_tr_ridge, y_train)
ridge_val_preds = np.clip(ridge.predict(X_va_ridge), 0.0, None)
ridge_rmsle = np.sqrt(np.mean((ridge_val_preds - y_val) ** 2))
print(f"Ridge Regression done in {time.time() - t_ridge:.1f}s | Val RMSLE: {ridge_rmsle:.4f}\n")

with open(RIDGE_OUT, "wb") as f:
    pickle.dump(ridge, f)

del X_tr_ridge, X_va_ridge, X_tr_ohe, X_va_ohe, ridge
gc.collect()

# ==============================================================================
# 4. TRAIN REGULARIZED FOUR-BRANCH MULTIMODAL NEURAL NETWORK
# ==============================================================================
print("[4/4] Training Regularized 4-Branch Multimodal Neural Network on GPU...")

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
        t_feat = self.title_net(x_title)
        d_feat = self.desc_net(x_desc)
        n_feat = self.num_net(x_num)
        fused = torch.cat([t_feat, d_feat, cat_feat, n_feat], dim=1)
        return self.head(fused).squeeze(1)

def prefetched_batches(X_t, X_d, X_c, X_n, y, batch_size, shuffle=True, prefetch=1):
    n_samples = len(y)
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    q = queue.Queue(maxsize=prefetch)
    sentinel = object()

    def producer():
        for start in range(0, n_samples, batch_size):
            idx = indices[start : start + batch_size]
            batch = (
                torch.from_numpy(X_t[idx].toarray()),
                torch.from_numpy(X_d[idx].toarray()),
                torch.from_numpy(X_c[idx]),
                torch.from_numpy(X_n[idx]),
                torch.from_numpy(y[idx]),
            )
            q.put(batch)
        q.put(sentinel)

    threading.Thread(target=producer, daemon=True).start()
    while True:
        item = q.get()
        if item is sentinel:
            break
        yield item

model = MercariMultimodalV2(
    title_dim=X_train_title.shape[1],
    desc_dim=X_train_desc.shape[1],
    vocab_sizes=vocab_sizes,
    num_dim=X_train_num.shape[1],
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=(not USE_BF16))

best_nn_rmsle = float("inf")
best_epoch = 0
epochs_no_improve = 0

for epoch in range(1, MAX_EPOCHS + 1):
    t0 = time.time()
    model.train()
    train_loss_sum = 0.0

    for x_t, x_d, x_c, x_n, target in prefetched_batches(
        X_train_title, X_train_desc, X_train_cat, X_train_num, y_train, BATCH_SIZE, shuffle=True
    ):
        x_t = x_t.to(DEVICE)
        x_d = x_d.to(DEVICE)
        x_c = x_c.to(DEVICE)
        x_n = x_n.to(DEVICE)
        target = target.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            preds = model(x_t, x_d, x_c, x_n)
            loss = criterion(preds, target)

        if USE_BF16:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss_sum += loss.item() * len(target)

    train_rmsle = np.sqrt(train_loss_sum / len(y_train))

    model.eval()
    val_preds = []
    with torch.no_grad():
        for x_t, x_d, x_c, x_n, _ in prefetched_batches(
            X_val_title, X_val_desc, X_val_cat, X_val_num, y_val, VAL_BATCH_SIZE, shuffle=False
        ):
            x_t = x_t.to(DEVICE)
            x_d = x_d.to(DEVICE)
            x_c = x_c.to(DEVICE)
            x_n = x_n.to(DEVICE)
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                val_preds.append(model(x_t, x_d, x_c, x_n).float().cpu().numpy())

    val_preds = np.clip(np.concatenate(val_preds), 0.0, None)
    val_rmsle = np.sqrt(np.mean((val_preds - y_val) ** 2))
    scheduler.step(val_rmsle)

    elapsed = time.time() - t0
    current_lr = optimizer.param_groups[0]["lr"]
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    if val_rmsle < best_nn_rmsle - 1e-5:
        best_nn_rmsle = val_rmsle
        best_epoch = epoch
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_OUT)
        status = f"  <-- BEST SAVED ({MODEL_OUT})"
    else:
        epochs_no_improve += 1
        status = f"  (no improve: {epochs_no_improve}/{PATIENCE})"

    print(
        f"Epoch {epoch:03d}/{MAX_EPOCHS} [{elapsed:4.1f}s | VRAM {peak_vram:.2f}GB | LR {current_lr:.1e}] | "
        f"Train RMSLE: {train_rmsle:.4f} | Val RMSLE: {val_rmsle:.4f}{status}"
    )

    if epochs_no_improve >= PATIENCE:
        print(f"\nEarly stopping triggered after {PATIENCE} epochs without improvement.")
        break

# ==============================================================================
# 5. FIND OPTIMAL ENSEMBLE BLEND (NEURAL NET + RIDGE)
# ==============================================================================
model.load_state_dict(torch.load(MODEL_OUT, weights_only=True))
model.eval()
best_nn_preds = []
with torch.no_grad():
    for x_t, x_d, x_c, x_n, _ in prefetched_batches(
        X_val_title, X_val_desc, X_val_cat, X_val_num, y_val, VAL_BATCH_SIZE, shuffle=False
    ):
        x_t = x_t.to(DEVICE)
        x_d = x_d.to(DEVICE)
        x_c = x_c.to(DEVICE)
        x_n = x_n.to(DEVICE)
        with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
            best_nn_preds.append(model(x_t, x_d, x_c, x_n).float().cpu().numpy())

best_nn_preds = np.clip(np.concatenate(best_nn_preds), 0.0, None)

best_blend_rmsle = float("inf")
best_w = 1.0
for w in np.linspace(0.0, 1.0, 101):
    blend = w * best_nn_preds + (1.0 - w) * ridge_val_preds
    score = np.sqrt(np.mean((blend - y_val) ** 2))
    if score < best_blend_rmsle:
        best_blend_rmsle = score
        best_w = w

print("\n================ FINAL BENCHMARK SUMMARY ================")
print(f"  v1 Baseline Neural Net RMSLE: 0.4263")
print(f"  v2 Ridge Regression RMSLE:    {ridge_rmsle:.4f}")
print(f"  v2 Neural Net RMSLE (Ep {best_epoch:02d}):  {best_nn_rmsle:.4f}")
print(f"  v2 Optimal Blend ({best_w:.2f} NN + {1-best_w:.2f} Ridge): {best_blend_rmsle:.4f}")
print("=========================================================")