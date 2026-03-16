"""
Main Pipeline: Data Sampling -> CSV Export -> Richness Analysis
Orchestrates the complete workflow for job data processing
"""
import subprocess
import sys
from pathlib import Path
import time

def run_command(description, command, cwd=None):
    """Run a shell command and handle errors"""
    print("\n" + "="*60)
    print(f"📍 {description}")
    print("="*60)

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=cwd,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    start_time = time.time()

    print("="*60)
    print("JOB DATA PROCESSING PIPELINE")
    print("="*60)
    print("Steps:")
    print("  1. Sample data from MinIO (stratified sampling)")
    print("  2. Export 10k rows to CSV for viewing")
    print("  3. Run richness analysis on full sample")
    print("="*60)

    # Get the directory where this script is located
    script_dir = Path(__file__).parent

    # Step 1: Run sampling
    # step1_success = run_command(
    #     "Step 1: Sampling data from MinIO",
    #     f"python {script_dir}/sample_data_spark.py",
    #     cwd=script_dir
    # )

    # if not step1_success:
    #     print("\n❌ Pipeline failed at Step 1: Sampling")
    #     sys.exit(1)

    # Step 2: Export to CSV (10k rows for viewing)
    # print("\n" + "="*60)
    # print("📍 Step 2: Exporting 10k rows to CSV")
    # print("="*60)

    # # Import here to avoid issues if script hasn't run yet
    # try:
    #     import pandas as pd
    #     import s3fs

    #     # MinIO configuration
    #     minio_endpoint = "192.168.18.51:9000"
    #     minio_access_key = "minioadmin"
    #     minio_secret_key = "minioadmin123"
    #     input_path = "ssg-sg-jobs/sampled_jobs_spark.parquet"
    #     output_path = script_dir / "sampled_jobs_10k.csv"

    #     print("🔌 Connecting to MinIO...")
    #     fs = s3fs.S3FileSystem(
    #         key=minio_access_key,
    #         secret=minio_secret_key,
    #         client_kwargs={'endpoint_url': f'http://{minio_endpoint}'},
    #         use_ssl=False
    #     )

    #     print(f"📖 Reading Parquet from MinIO: s3://{input_path}")
    #     df = pd.read_parquet(f"s3://{input_path}", filesystem=fs)

    #     print(f"   Total rows available: {len(df):,}")

    #     # Sample 10k rows
    #     n_rows = min(10000, len(df))
    #     print(f"   🎲 Sampling {n_rows:,} random rows...")
    #     df_sample = df.sample(n=n_rows, random_state=42)

    #     print(f"💾 Writing to {output_path}")
    #     df_sample.to_csv(output_path, index=False)

    #     file_size = output_path.stat().st_size / 1024**2
    #     print(f"✅ Step 2 completed: {n_rows:,} rows saved ({file_size:.1f} MB)")

    # except Exception as e:
    #     print(f"❌ Step 2 failed: {e}")
    #     print("\nNote: Make sure s3fs is installed: pip install s3fs")
    #     sys.exit(1)

    # Step 3: Run richness analysis
    step3_success = run_command(
        "Step 3: Running richness analysis",
        f"python {script_dir}/analyze_job_richness.py",
        cwd=script_dir
    )
    if not step3_success:
        print("\n❌ Pipeline failed at Step 3: Richness Analysis")
        sys.exit(1)

    # Pipeline complete
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"⏱️  Total pipeline time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print("="*60)

if __name__ == "__main__":
    main()
