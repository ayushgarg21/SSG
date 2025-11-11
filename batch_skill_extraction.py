"""
Batch Skill Extraction for JobTech Dataset
Extracts skills from all job descriptions in the CSV file and saves results with position information.
"""

import pandas as pd
import subprocess
import time
import sys
from pathlib import Path

# Add the skill_extractor_streamlit directory to path
sys.path.insert(0, str(Path(__file__).parent / "skill_extractor_streamlit copy"))

from lib.skillsExtractor3 import SkillExtractor

# Configuration
JAR_FILE = "./skill_extractor_streamlit copy/lib/talented-utility-server-1.1.3.jar"
SEDC_FILE = "./skill_extractor_streamlit copy/lib/sedc-obj/jobtech-sedc_dbs_new_241024.obj"
INPUT_CSV = "./jobtech_job_export_Q3_2025_20251110_203838.csv"
OUTPUT_CSV = "./jobtech_skills_extracted.csv"
BATCH_SIZE = 1000  # Process 1000 jobs at a time
CHECKPOINT_INTERVAL = 5000  # Save checkpoint every 5000 jobs

def start_java_server():
    """Start the Java server with the SEDC object"""
    print(f"Starting Java server with SEDC: {SEDC_FILE}")
    args = ["-sedc", SEDC_FILE]
    process = subprocess.Popen(
        ["java", "-jar", JAR_FILE] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    print("Waiting for Java server to initialize...")

    # Wait for server to be ready
    time.sleep(10)
    print("Java server started!")
    return process

def extract_skills_batch(df_batch, batch_num, total_batches):
    """Extract skills from a batch of job descriptions"""
    print(f"\n{'='*80}")
    print(f"Processing Batch {batch_num}/{total_batches} ({len(df_batch)} jobs)")
    print(f"{'='*80}")

    start_time = time.time()

    try:
        # Initialize skill extractor
        extractor = SkillExtractor(
            df_batch,
            title_col='job_title',
            desc_col='job_description',
            job_id_col='job_id',
            isJob=True
        )
        extractor.process_skills()
        skills_df = extractor.df

        # Keep important columns with position information
        output_columns = [
            'job_id', 'job_title', 'job_description',
            'skillId', 'skill', 'skillType', 'skillCategory',
            'f1', 'skill_count', 'weight',
            'start', 'end'  # Position in text where skill was found
        ]

        skills_df = skills_df[output_columns].copy()

        # Rename for clarity
        skills_df = skills_df.rename(columns={
            'skillId': 'skill_id',
            'skillType': 'skill_type',
            'skillCategory': 'skill_category',
            'f1': 'is_core_skill'
        })

        elapsed = time.time() - start_time
        print(f"✓ Batch {batch_num} completed in {elapsed:.2f}s")
        print(f"  - Extracted {len(skills_df)} skill instances from {len(df_batch)} jobs")

        return skills_df

    except Exception as e:
        print(f"✗ Error processing batch {batch_num}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def main():
    """Main processing function"""
    java_process = None

    try:
        # Start Java server
        java_process = start_java_server()

        # Load the CSV
        print(f"\nLoading CSV: {INPUT_CSV}")
        df = pd.read_csv(INPUT_CSV)
        print(f"✓ Loaded {len(df):,} job postings")

        # Select only necessary columns for processing
        required_cols = ['job_id', 'job_title', 'job_description']
        df_process = df[required_cols].copy()

        # Fill NaN values
        df_process['job_title'] = df_process['job_title'].fillna('')
        df_process['job_description'] = df_process['job_description'].fillna('')

        # Remove rows with empty descriptions
        df_process = df_process[df_process['job_description'].str.strip() != '']
        print(f"✓ {len(df_process):,} jobs with valid descriptions")

        # Calculate batches
        total_jobs = len(df_process)
        total_batches = (total_jobs + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING PLAN")
        print(f"{'='*80}")
        print(f"Total jobs: {total_jobs:,}")
        print(f"Batch size: {BATCH_SIZE:,}")
        print(f"Total batches: {total_batches:,}")
        print(f"Output file: {OUTPUT_CSV}")
        print(f"{'='*80}\n")

        # Ask for confirmation - DISABLED FOR UNATTENDED RUNS
        # response = input("Proceed with extraction? (yes/no): ").strip().lower()
        # if response != 'yes':
        #     print("Extraction cancelled.")
        #     return

        # Process in batches
        all_results = []
        overall_start = time.time()

        for i in range(0, total_jobs, BATCH_SIZE):
            batch_num = (i // BATCH_SIZE) + 1
            batch_df = df_process.iloc[i:i+BATCH_SIZE].copy()

            # Extract skills
            skills_batch = extract_skills_batch(batch_df, batch_num, total_batches)

            if not skills_batch.empty:
                all_results.append(skills_batch)

            # Save checkpoint
            if batch_num % (CHECKPOINT_INTERVAL // BATCH_SIZE) == 0 and all_results:
                checkpoint_file = f"{OUTPUT_CSV}.checkpoint_{batch_num}"
                checkpoint_df = pd.concat(all_results, ignore_index=True)
                checkpoint_df.to_csv(checkpoint_file, index=False)
                print(f"  💾 Checkpoint saved: {checkpoint_file} ({len(checkpoint_df):,} records)")

        # Combine all results
        if all_results:
            print(f"\n{'='*80}")
            print("COMBINING RESULTS")
            print(f"{'='*80}")

            final_df = pd.concat(all_results, ignore_index=True)

            # Save final results
            print(f"Saving to {OUTPUT_CSV}...")
            final_df.to_csv(OUTPUT_CSV, index=False)

            # Print summary
            total_time = time.time() - overall_start
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)

            print(f"\n{'='*80}")
            print("EXTRACTION COMPLETE!")
            print(f"{'='*80}")
            print(f"Total processing time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
            print(f"Total jobs processed: {total_jobs:,}")
            print(f"Total skill instances: {len(final_df):,}")
            print(f"Unique skills: {final_df['skill'].nunique():,}")
            print(f"Output file: {OUTPUT_CSV}")
            print(f"File size: {Path(OUTPUT_CSV).stat().st_size / 1024 / 1024:.2f} MB")
            print(f"\nSkill Type Distribution:")
            print(final_df['skill_type'].value_counts())
            print(f"{'='*80}\n")

        else:
            print("No results extracted.")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        # Try to save partial results
        if 'all_results' in locals() and all_results:
            partial_file = f"{OUTPUT_CSV}.partial"
            print(f"Saving partial results to {partial_file}...")
            partial_df = pd.concat(all_results, ignore_index=True)
            partial_df.to_csv(partial_file, index=False)
            print(f"✓ Partial results saved ({len(partial_df):,} records)")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Shutdown Java server
        if java_process:
            print("\nShutting down Java server...")
            java_process.terminate()
            try:
                java_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                java_process.kill()
            print("Java server stopped.")

if __name__ == "__main__":
    main()
