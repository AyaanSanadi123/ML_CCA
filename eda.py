import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("train.tsv", sep="\t")

print("=== 1. SHAPE, TYPES & TARGET (PRICE) DISTRIBUTION ===")
print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
print(df.dtypes, "\n")

zero_prices = (df["price"] <= 0).sum()
print(f"Zero or negative prices: {zero_prices:,}")
print(f"Raw price -> Mean: ${df['price'].mean():.2f} | Median: ${df['price'].median():.2f} | Skew: {df['price'].skew():.2f}")
percentiles = [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0]
print("Price Percentiles:")
print(df["price"].quantile(percentiles), "\n")

valid_df = df[df["price"] > 0].copy()
valid_df["log_price"] = np.log1p(valid_df["price"])
print(f"Log(price + 1) -> Mean: {valid_df['log_price'].mean():.4f} | Median: {valid_df['log_price'].median():.4f} | Skew: {valid_df['log_price'].skew():.4f}\n")

print("=== 2. MISSINGNESS, PLACEHOLDERS & TARGET SIGNAL ===")
missing = valid_df.isna().sum()
print("NaN counts per column:")
print(missing[missing > 0])

placeholder_desc = (valid_df["item_description"] == "No description yet").sum()
print(f"\nLiteral 'No description yet': {placeholder_desc:,} ({placeholder_desc / len(valid_df):.2%})")

has_brand = valid_df["brand_name"].notna()
no_desc = valid_df["item_description"].isna() | (valid_df["item_description"] == "No description yet")
print(f"Median price WITH brand:          ${valid_df.loc[has_brand, 'price'].median():.2f}")
print(f"Median price WITHOUT brand:       ${valid_df.loc[~has_brand, 'price'].median():.2f}")
print(f"Median price WITH description:    ${valid_df.loc[~no_desc, 'price'].median():.2f}")
print(f"Median price WITHOUT description: ${valid_df.loc[no_desc, 'price'].median():.2f}\n")

print("=== 3. CATEGORICAL CARDINALITY & TARGET VARIANCE ===")
for col in ["category_name", "brand_name", "item_condition_id", "shipping"]:
    vc = valid_df[col].value_counts()
    rare = (vc < 5).sum()
    print(f"{col:20s} -> Unique: {valid_df[col].nunique():6,d} | Appearing <5 times: {rare:,}")

print("\nCategory slash-depth counts (number of '/' characters):")
print(valid_df["category_name"].dropna().str.count("/").value_counts())

print("\nPrice by Condition ID (1=New -> 5=Poor):")
print(valid_df.groupby("item_condition_id")["price"].agg(["count", "median", "mean"]))

print("\nPrice by Shipping (0=Buyer pays, 1=Seller pays):")
print(valid_df.groupby("shipping")["price"].agg(["count", "median", "mean"]), "\n")

print("=== 4. TEXT LENGTH & NOISE PROFILE ===")
valid_df["name_words"] = valid_df["name"].fillna("").str.split().str.len()
valid_df["desc_words"] = valid_df["item_description"].fillna("").str.split().str.len()

print("Title word count quantiles (50%, 95%, 99%, max):")
print(valid_df["name_words"].quantile([0.5, 0.95, 0.99, 1.0]).to_dict())
print("Description word count quantiles (50%, 95%, 99%, max):")
print(valid_df["desc_words"].quantile([0.5, 0.95, 0.99, 1.0]).to_dict())

corr_name = valid_df["name_words"].corr(valid_df["log_price"])
corr_desc = valid_df["desc_words"].corr(valid_df["log_price"])
print(f"Correlation with log_price -> Name words: {corr_name:.4f} | Desc words: {corr_desc:.4f}")

print("\n5 Random Raw Descriptions (check for HTML, symbols, or masked tokens):")
for sample in valid_df["item_description"].dropna().sample(5, random_state=42):
    print(" ->", repr(sample[:180]))

print("\n=== 5. DUPLICATES & IRREDUCIBLE NOISE ===")
feature_cols = ["name", "item_description", "brand_name", "category_name", "item_condition_id", "shipping"]
exact_dups = valid_df.duplicated(subset=feature_cols + ["price"]).sum()
feature_dups = valid_df[valid_df.duplicated(subset=feature_cols, keep=False)]
price_std_on_dups = feature_dups.groupby(["name", "brand_name", "item_condition_id"], dropna=False)["log_price"].std().dropna()

print(f"Exact full-row duplicates (same features + same price): {exact_dups:,}")
print(f"Identical feature rows (potentially different prices):  {len(feature_dups):,}")
print(f"Mean log_price std dev across identical listings:       {price_std_on_dups.mean():.4f}")

# --- DIAGNOSTIC PLOTS ---
valid_df["main_cat"] = valid_df["category_name"].fillna("Missing").apply(lambda x: x.split("/")[0])

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Mercari Dataset — Complete Diagnostic EDA", fontsize=14, fontweight="bold")

# Plot 1: Raw Price (<= 99th percentile)
p99 = valid_df["price"].quantile(0.99)
axes[0, 0].hist(valid_df[valid_df["price"] <= p99]["price"], bins=60, color="#3b82f6", edgecolor="black", alpha=0.8)
axes[0, 0].set_title(f"1. Raw Price Distribution (<= 99th pct: ${p99:.0f})")
axes[0, 0].set_xlabel("Price ($)")
axes[0, 0].set_ylabel("Count")

# Plot 2: Log-Transformed Price
axes[0, 1].hist(valid_df["log_price"], bins=60, color="#10b981", edgecolor="black", alpha=0.8)
axes[0, 1].set_title("2. Log-Transformed Price: log(price + 1)")
axes[0, 1].set_xlabel("log(price + 1)")
axes[0, 1].set_ylabel("Count")

# Plot 3: Log Price by Top-Level Category
top_cats = valid_df["main_cat"].value_counts().head(10).index
cat_data = [valid_df[valid_df["main_cat"] == c]["log_price"].values for c in top_cats]
axes[1, 0].boxplot(cat_data, tick_labels=top_cats, vert=False)
axes[1, 0].set_title("3. Log Price Distribution by Main Category")
axes[1, 0].set_xlabel("log(price + 1)")

# Plot 4: Description Word Count (<= 99th percentile)
d99 = valid_df["desc_words"].quantile(0.99)
axes[1, 1].hist(valid_df[valid_df["desc_words"] <= d99]["desc_words"], bins=50, color="#f59e0b", edgecolor="black", alpha=0.8)
axes[1, 1].set_title(f"4. Description Word Count (<= 99th pct: {d99:.0f} words)")
axes[1, 1].set_xlabel("Word Count")
axes[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
print("\nSaved diagnostic plots to eda_plots.png")