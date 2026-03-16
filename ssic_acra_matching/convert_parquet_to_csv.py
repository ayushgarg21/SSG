import pandas as pd

# Read parquet file
print("Reading parquet file...")
df = pd.read_parquet("../data/jan-2026-jobs-data-matched-acra-ssic.parquet")

print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

# Write to CSV
output_file = "../data/jan-2026-jobs-data-matched-acra-ssic.csv"
print(f"Writing to {output_file}...")
df.to_csv(output_file, index=False)

print(f"✅ Done! Saved to {output_file}")
