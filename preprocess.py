import re
import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "train.tsv"  # On Kaggle: "/kaggle/working/train.tsv"
OUT_DIR = "processed_data"
MIN_BRAND_FREQ = 5
SAMPLE_SIZE = None  # Set to 200_000 for fast prototyping, or None for full 1.48M rows

os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(text: str) -> str:
    text = text.lower()
    # Preserve Mercari's price-removed token before stripping punctuation
    text = text.replace("[rm]", " price_removed ")
    # Keep alphanumeric characters and decimals/slashes (e.g., 8oz, 16gb, 1/2)
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("1. Loading and filtering raw data...")
df = pd.read_csv(DATA_PATH, sep="\t")
df = df[df["price"] > 0].reset_index(drop=True)

if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Target log1p transformation (reduces skew from 11.39 -> 0.70)
df["log_price"] = np.log1p(df["price"].values).astype(np.float32)

print("2. Engineering missingness flags & splitting category hierarchy...")
df["has_brand"] = df["brand_name"].notna().astype(np.float32)
no_desc_mask = df["item_description"].isna() | (df["item_description"] == "No description yet")
df["has_desc"] = (~no_desc_mask).astype(np.float32)

df["brand_name"] = df["brand_name"].fillna("missing")
df["category_name"] = df["category_name"].fillna("missing/missing/missing")
df["name"] = df["name"].fillna("missing")
df["item_description"] = df["item_description"].fillna("missing").replace("No description yet", "missing")

# Split category into 3 tiers + keep full category path
cat_parts = df["category_name"].str.split("/", n=2, expand=True).fillna("missing")
df["cat_main"] = cat_parts[0]
df["cat_sub1"] = cat_parts[1]
df["cat_sub2"] = cat_parts[2]

print("3. Cleaning text fields...")
df["clean_name"] = df["name"].apply(clean_text)
df["clean_desc"] = df["item_description"].apply(clean_text)
# Combine structured text context for TF-IDF
df["combined_text"] = (
    df["clean_name"] + " | " +
    df["brand_name"].str.lower() + " | " +
    df["category_name"].str.lower().str.replace("/", " ") + " | " +
    df["clean_desc"]
)

# Train / Validation split (90% / 10%) BEFORE fitting vocabulary/TF-IDF to prevent leakage
train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("4. Encoding categorical features (pruning rare brands < 5 occurrences)...")
brand_counts = train_df["brand_name"].value_counts()
valid_brands = set(brand_counts[brand_counts >= MIN_BRAND_FREQ].index)

for split_df in (train_df, val_df):
    split_df["brand_clean"] = split_df["brand_name"].where(split_df["brand_name"].isin(valid_brands), "rare_brand")

cat_cols = ["brand_clean", "category_name", "cat_main", "cat_sub1", "cat_sub2", "item_condition_id", "shipping"]
vocab_maps = {}
vocab_sizes = {}

for col in cat_cols:
    # Reserve index 0 for unknown/unseen validation categories
    unique_vals = train_df[col].unique().tolist()
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)}
    vocab_maps[col] = mapping
    vocab_sizes[col] = len(mapping) + 1
    
    train_df[f"{col}_idx"] = train_df[col].map(mapping).fillna(0).astype(np.int64)
    val_df[f"{col}_idx"] = val_df[col].map(mapping).fillna(0).astype(np.int64)

print("   Embedding vocab sizes:", vocab_sizes)

print("5. Vectorizing text with sublinear TF-IDF (1-2 ngrams)...")
tfidf = TfidfVectorizer(
    max_features=60_000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    dtype=np.float32
)
X_train_tfidf = tfidf.fit_transform(train_df["combined_text"])
X_val_tfidf = tfidf.transform(val_df["combined_text"])

# Extract dense tabular arrays
idx_cols = [f"{c}_idx" for c in cat_cols]
num_cols = ["has_brand", "has_desc"]

X_train_cat = train_df[idx_cols].values
X_val_cat = val_df[idx_cols].values
X_train_num = train_df[num_cols].values
X_val_num = val_df[num_cols].values
y_train = train_df["log_price"].values
y_val = val_df["log_price"].values

# Save artifacts
sparse.save_npz(f"{OUT_DIR}/X_train_tfidf.npz", X_train_tfidf)
sparse.save_npz(f"{OUT_DIR}/X_val_tfidf.npz", X_val_tfidf)
np.savez(
    f"{OUT_DIR}/tabular_data.npz",
    X_train_cat=X_train_cat, X_val_cat=X_val_cat,
    X_train_num=X_train_num, X_val_num=X_val_num,
    y_train=y_train, y_val=y_val
)
with open(f"{OUT_DIR}/metadata.pkl", "wb") as f:
    pickle.dump({"vocab_sizes": vocab_sizes, "vocab_maps": vocab_maps, "tfidf": tfidf}, f)

print(f"\nDone! Saved preprocessed datasets to '{OUT_DIR}/'")
print(f"Train shape -> TF-IDF: {X_train_tfidf.shape} | Cats: {X_train_cat.shape} | Num: {X_train_num.shape}")
print(f"Val shape   -> TF-IDF: {X_val_tfidf.shape} | Cats: {X_val_cat.shape} | Num: {X_val_num.shape}")