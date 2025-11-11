#!/bin/bash

# Full Skills Extraction Pipeline
# Runs both skill extraction and sentence extraction in sequence
# Logs output to files for review

echo "======================================================================"
echo "Starting Full Skills Extraction Pipeline"
echo "Started at: $(date)"
echo "======================================================================"
echo ""

# Create logs directory
mkdir -p logs

# Step 1: Extract skills
echo "STEP 1/2: Extracting skills from all job descriptions..."
echo "This will take approximately 6-10 hours"
echo "Logging to: logs/skill_extraction.log"
echo ""

python batch_skill_extraction.py 2>&1 | tee logs/skill_extraction.log

# Check if skill extraction succeeded
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✓ Step 1 completed successfully!"
    echo "Completed at: $(date)"
    echo ""

    # Step 2: Extract sentences
    echo "STEP 2/2: Extracting sentences for each skill..."
    echo "This will take approximately 30-60 minutes"
    echo "Logging to: logs/sentence_extraction.log"
    echo ""

    python extract_skill_sentences.py 2>&1 | tee logs/sentence_extraction.log

    # Check if sentence extraction succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ PIPELINE COMPLETED SUCCESSFULLY!"
        echo "Finished at: $(date)"
        echo "======================================================================"
        echo ""
        echo "Output files:"
        echo "  - jobtech_skills_extracted.csv"
        echo "  - jobtech_skills_with_sentences.csv"
        echo ""
        echo "Logs saved in:"
        echo "  - logs/skill_extraction.log"
        echo "  - logs/sentence_extraction.log"
        echo ""
    else
        echo ""
        echo "✗ Step 2 FAILED - Sentence extraction encountered an error"
        echo "Check logs/sentence_extraction.log for details"
        exit 1
    fi
else
    echo ""
    echo "✗ Step 1 FAILED - Skill extraction encountered an error"
    echo "Check logs/skill_extraction.log for details"
    exit 1
fi

echo "======================================================================"
echo "All done! You can now analyze: jobtech_skills_with_sentences.csv"
echo "======================================================================"
