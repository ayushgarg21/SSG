#!/usr/bin/env python3
"""
Full Skills Extraction Pipeline
Runs both skill extraction and sentence extraction in sequence
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

def run_command(script_name, log_file, description):
    """Run a Python script and log output"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"Logging to: {log_file}")
    print(f"{'='*80}\n")

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Run the script
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end='')
            log.write(line)
            log.flush()

        process.wait()

    return process.returncode

def main():
    """Main pipeline execution"""
    start_time = datetime.now()

    print("="*80)
    print("Starting Full Skills Extraction Pipeline")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Step 1: Extract skills
    print("\nSTEP 1/2: Extracting skills from all job descriptions...")
    print("This will take approximately 6-10 hours")

    result1 = run_command(
        "batch_skill_extraction.py",
        "logs/skill_extraction.log",
        "Skill Extraction"
    )

    if result1 != 0:
        print("\n✗ Step 1 FAILED - Skill extraction encountered an error")
        print("Check logs/skill_extraction.log for details")
        sys.exit(1)

    print("\n✓ Step 1 completed successfully!")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 2: Extract sentences
    print("\nSTEP 2/2: Extracting sentences for each skill...")
    print("This will take approximately 30-60 minutes")

    result2 = run_command(
        "extract_skill_sentences.py",
        "logs/sentence_extraction.log",
        "Sentence Extraction"
    )

    if result2 != 0:
        print("\n✗ Step 2 FAILED - Sentence extraction encountered an error")
        print("Check logs/sentence_extraction.log for details")
        sys.exit(1)

    # Success!
    end_time = datetime.now()
    duration = end_time - start_time
    hours = int(duration.total_seconds() // 3600)
    minutes = int((duration.total_seconds() % 3600) // 60)

    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {hours}h {minutes}m")
    print("="*80)
    print("\nOutput files:")
    print("  - jobtech_skills_extracted.csv")
    print("  - jobtech_skills_with_sentences.csv")
    print("\nLogs saved in:")
    print("  - logs/skill_extraction.log")
    print("  - logs/sentence_extraction.log")
    print("\n" + "="*80)
    print("All done! You can now analyze: jobtech_skills_with_sentences.csv")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
