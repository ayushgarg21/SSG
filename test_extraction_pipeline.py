"""
Test Extraction Pipeline
Tests skill extraction and sentence extraction on a small sample.
"""

import pandas as pd
import subprocess
import time
import sys
from pathlib import Path

# Add the skill_extractor_streamlit directory to path
sys.path.insert(0, str(Path(__file__).parent / "skill_extractor_streamlit copy"))

from lib.skillsExtractor3 import SkillExtractor
from extract_skill_sentences import clean_html, extract_sentence

# Configuration
JAR_FILE = "./skill_extractor_streamlit copy/lib/talented-utility-server-1.1.3.jar"
SEDC_FILE = "./skill_extractor_streamlit copy/lib/sedc-obj/jobtech-sedc_dbs_new_241024.obj"
INPUT_CSV = "./jobtech_job_export_Q3_2025_20251110_203838.csv"
TEST_SAMPLE_SIZE = 10  # Test with 10 jobs

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
    time.sleep(10)
    print("✓ Java server started!\n")
    return process

def test_skill_extraction():
    """Test skill extraction on a small sample"""
    print(f"{'='*80}")
    print("STEP 1: SKILL EXTRACTION TEST")
    print(f"{'='*80}\n")

    # Load sample data
    print(f"Loading {TEST_SAMPLE_SIZE} jobs from CSV...")
    df = pd.read_csv(INPUT_CSV, nrows=TEST_SAMPLE_SIZE)
    print(f"✓ Loaded {len(df)} jobs\n")

    # Prepare data
    required_cols = ['job_id', 'job_title', 'job_description']
    df_test = df[required_cols].copy()
    df_test['job_title'] = df_test['job_title'].fillna('')
    df_test['job_description'] = df_test['job_description'].fillna('')
    df_test = df_test[df_test['job_description'].str.strip() != '']

    print(f"Processing {len(df_test)} jobs with valid descriptions...")

    # Extract skills
    extractor = SkillExtractor(
        df_test,
        title_col='job_title',
        desc_col='job_description',
        job_id_col='job_id',
        isJob=True
    )
    extractor.process_skills()
    skills_df = extractor.df

    # Format output
    output_columns = [
        'job_id', 'job_title', 'job_description',
        'skillId', 'skill', 'skillType', 'skillCategory',
        'f1', 'skill_count', 'weight',
        'start', 'end'
    ]

    skills_df = skills_df[output_columns].copy()
    skills_df = skills_df.rename(columns={
        'skillId': 'skill_id',
        'skillType': 'skill_type',
        'skillCategory': 'skill_category',
        'f1': 'is_core_skill'
    })

    # Print results
    print(f"\n✓ Skill extraction completed!")
    print(f"  - Total skill instances: {len(skills_df)}")
    print(f"  - Unique skills: {skills_df['skill'].nunique()}")
    print(f"  - Jobs processed: {skills_df['job_id'].nunique()}")

    print(f"\nSkill Type Distribution:")
    print(skills_df['skill_type'].value_counts())

    print(f"\nSample Skills Extracted:")
    print(skills_df[['job_id', 'skill', 'skill_type', 'is_core_skill', 'start', 'end']].head(20).to_string(index=False))

    return skills_df

def test_sentence_extraction(skills_df):
    """Test context extraction (10 words before/after) on the extracted skills"""
    print(f"\n{'='*80}")
    print("STEP 2: SKILL CONTEXT EXTRACTION TEST")
    print(f"{'='*80}\n")

    print(f"Extracting context (10 words before + skill + 10 words after) for {len(skills_df)} skill instances...")

    # Extract context
    skills_df['skill_context'] = skills_df.apply(
        lambda row: extract_sentence(
            row['job_description'],
            int(row['start']) if pd.notna(row['start']) else -1,
            int(row['end']) if pd.notna(row['end']) else -1
        ),
        axis=1
    )

    skills_df['job_description_clean'] = skills_df['job_description'].apply(clean_html)
    skills_df['context_length'] = skills_df['skill_context'].str.len()
    skills_df['context_word_count'] = skills_df['skill_context'].str.split().str.len()

    # Print results
    print(f"✓ Context extraction completed!")
    print(f"  - Contexts extracted: {(skills_df['skill_context'] != '').sum()}")

    print(f"\nContext Word Count Statistics:")
    print(skills_df['context_word_count'].describe())

    print(f"\nContext Length (characters) Statistics:")
    print(skills_df['context_length'].describe())

    print(f"\n{'='*80}")
    print("SAMPLE EXTRACTED CONTEXTS")
    print(f"{'='*80}\n")

    # Show examples
    samples = skills_df[skills_df['skill_context'] != ''].head(10)
    for idx, row in samples.iterrows():
        print(f"Job ID: {row['job_id']}")
        print(f"Job Title: {row['job_title']}")
        print(f"Skill: {row['skill']} ({row['skill_type']})")
        print(f"Core Skill: {row['is_core_skill']}")
        print(f"Position: {row['start']}-{row['end']}")
        print(f"Context: {row['skill_context']}")
        print(f"Word count: {row['context_word_count']}")
        print(f"{'-'*80}\n")

    return skills_df

def save_test_results(skills_df):
    """Save test results to CSV"""
    output_file = "test_results_skills_and_context.csv"

    # Select final columns
    final_columns = [
        'job_id',
        'job_title',
        'skill_id',
        'skill',
        'skill_type',
        'skill_category',
        'is_core_skill',
        'skill_count',
        'weight',
        'skill_context',
        'context_length',
        'context_word_count',
        'start',
        'end',
        'job_description_clean',
        'job_description'
    ]

    df_output = skills_df[final_columns].copy()

    print(f"\n{'='*80}")
    print(f"Saving test results to: {output_file}")
    df_output.to_csv(output_file, index=False)
    print(f"✓ Saved {len(df_output)} records")
    print(f"File size: {Path(output_file).stat().st_size / 1024:.2f} KB")
    print(f"{'='*80}\n")

def main():
    """Main test function"""
    java_process = None

    try:
        # Start Java server
        java_process = start_java_server()

        # Test skill extraction
        skills_df = test_skill_extraction()

        # Test sentence extraction
        skills_with_sentences = test_sentence_extraction(skills_df)

        # Save results
        save_test_results(skills_with_sentences)

        print(f"\n{'='*80}")
        print("TEST COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print("\nYou can now run the full extraction with:")
        print("  1. python batch_skill_extraction.py    (extracts all skills)")
        print("  2. python extract_skill_sentences.py  (extracts 10 words before/after each skill)")
        print("\nOr run both automatically:")
        print("  caffeinate -i python run_full_extraction.py")
        print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")

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
