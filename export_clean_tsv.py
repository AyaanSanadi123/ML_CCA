import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "train.tsv"
META_PATH = "processed_data/metadata.pkl"
OUT_TSV = "processed_data/cleaned_reference.tsv"
MAX_ROWS = None  # Set to e.g. 50_000 for a lightweight preview file, or None for all 1.48M rows

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace("[rm]", " price_removed ")
    text = re.sub(r"[^a-z0-9\s\.]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

print("Loading raw data and saved metadata...")
df = pd.read_csv(DATA_PATH, sep="\t")
df = df[df["price"] > 0].reset_index(drop=True)

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
valid_brands = set(meta["vocab_maps"]["brand_clean"].keys()) - {"rare_brand"}

# Apply exact preprocessing transformations
df["log_price"] = np.log1p(df["price"].values).round(4)
df["has_brand"] = df["brand_name"].notna().astype(int)
no_desc_mask = df["item_description"].isna() | (df["item_description"] == "No description yet")
df["has_desc"] = (~no_desc_mask).astype(int)

df["brand_name"] = df["brand_name"].fillna("missing")
df["category_name"] = df["category_name"].fillna("missing/missing/missing")
df["name"] = df["name"].fillna("missing")
df["item_description"] = df["item_description"].fillna("missing").replace("No description yet", "missing")

cat_parts = df["category_name"].str.split("/", n=2, expand=True).fillna("missing")
df["cat_main"] = cat_parts[0]
df["cat_sub1"] = cat_parts[1]
df["cat_sub2"] = cat_parts[2]

df["clean_name"] = df["name"].apply(clean_text)
df["clean_desc"] = df["item_description"].apply(clean_text)

# Reproduce the exact 90/10 train/val split
train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
train_df = train_df.copy()
val_df = val_df.copy()
train_df["split"] = "train"
val_df["split"] = "val"

full_clean = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
full_clean["brand_clean"] = full_clean["brand_name"].where(
    full_clean["brand_name"].isin(valid_brands), "rare_brand"
)

# Select the exact human-readable columns represented in the .npz files
ref_cols = [
    "split",
    "price",
    "log_price",
    "brand_clean",
    "cat_main",
    "cat_sub1",
    "cat_sub2",
    "item_condition_id",
    "shipping",
    "has_brand",
    "has_desc",
    "clean_name",
    "clean_desc",
]
full_clean = full_clean[ref_cols]

if MAX_ROWS is not None:
    full_clean = full_clean.head(MAX_ROWS)

full_clean.to_csv(OUT_TSV, sep="\t", index=False)
print(f"Saved {len(full_clean):,} cleaned rows to '{OUT_TSV}'")
print("\nPreview of first 5 rows:")
print(full_clean.head(5).to_string())