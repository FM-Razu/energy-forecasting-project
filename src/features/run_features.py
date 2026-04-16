"""Run the feature engineering pipeline: create features then select them."""

import sys
import time

sys.path.insert(0, "src")

import pandas as pd
from features.engineering import create_features, select_features

t0 = time.time()

# Load cleaned data
print("Loading data/cleaned.csv ...")
df = pd.read_csv("data/cleaned.csv")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")

# Create features
print("\nCreating features ...")
df_features = create_features(df)
print(f"  Shape after create_features: {df_features.shape}")

# Select features
print("\nSelecting features ...")
selected, df_selected = select_features(df_features)

# Save
print("\nSaving ...")
df_features.to_csv("data/features.csv", index=False)
print(f"  Saved full features to data/features.csv")

elapsed = time.time() - t0

# Summary
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Before feature engineering: {df.shape}")
print(f"After create_features:      {df_features.shape}")
print(f"After select_features:      {df_selected.shape}")
print(f"\nKept features ({len(selected)}):")
for f in selected:
    print(f"  - {f}")
print(f"\nElapsed time: {elapsed:.2f}s")
