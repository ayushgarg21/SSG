"""
Convert partitioned Parquet folder to single CSV file
Optionally sample n rows with random state
"""
import pandas as pd
from pathlib import Path
import time

# Start timing
start_time = time.time()

# Configuration - update these paths as needed
input_path = "/Users/eugene/Downloads/sampled_jobs_spark.parquet"  # Folder with part files
output_path = "/Users/eugene/Downloads/sampled_jobs.csv"

# Sampling configuration
n_rows = 10000  # Set to a number to sample (e.g., 10000), or None to use all rows
random_state = 42  # For reproducible sampling

print("="*60)
print("Parquet to CSV Converter")
print("="*60)
print(f"Input:  {input_path}")
print(f"Output: {output_path}")
if n_rows:
    print(f"Sample: {n_rows:,} rows (random_state={random_state})")
else:
    print(f"Sample: All rows")
print()

# Check if input exists
if not Path(input_path).exists():
    print(f"❌ Error: Input path does not exist: {input_path}")
    print("\nMake sure you've downloaded the Parquet folder from MinIO first!")
    exit(1)

# Read partitioned Parquet (pandas handles the folder automatically)
print("📖 Reading Parquet files...")
df = pd.read_parquet(input_path)

print(f"   Total rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# Sample if n_rows is specified
if n_rows:
    if n_rows >= len(df):
        print(f"   ⚠️  Requested {n_rows:,} rows but only {len(df):,} available - using all rows")
    else:
        print(f"   🎲 Sampling {n_rows:,} random rows...")
        df = df.sample(n=n_rows, random_state=random_state)
        print(f"   ✓ Sampled {len(df):,} rows")

print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print()

# Show schema
print("Schema:")
print(df.dtypes)
print()

# Save to CSV
print("💾 Writing CSV file...")
df.to_csv(output_path, index=False)

# Get file size
output_size = Path(output_path).stat().st_size / 1024**2

print(f"✅ Done! Saved {len(df):,} rows to {output_path}")
print(f"   File size: {output_size:.1f} MB")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

print()
print("="*60)
print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
print("="*60)
