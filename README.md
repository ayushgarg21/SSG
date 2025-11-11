# JobTech Skills Extraction Pipeline

This pipeline extracts skills from job descriptions and identifies the specific sentences where those skills are mentioned.

## Overview

The pipeline consists of two main scripts:

1. **`batch_skill_extraction.py`** - Extracts skills from job descriptions using your existing skills extractor
2. **`extract_skill_sentences.py`** - Extracts the complete sentences containing each identified skill

## Files

```
SSG/
├── jobtech_job_export_Q3_2025_20251110_203838.csv  # Input data (598K jobs)
├── skill_extractor_streamlit copy/                 # Your existing skills extractor
├── requirements.txt                                 # Python dependencies
├── test_extraction_pipeline.py                      # Test script (start here!)
├── batch_skill_extraction.py                        # Full skill extraction
├── extract_skill_sentences.py                       # Sentence extraction
└── README.md                                        # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- beautifulsoup4
- py4j (for Java communication)
- lxml

### 2. Verify Java Server

Make sure you have Java installed:
```bash
java -version
```

The pipeline uses:
- JAR file: `skill_extractor_streamlit copy/lib/talented-utility-server-1.1.3.jar`
- SEDC file: `skill_extractor_streamlit copy/lib/sedc-obj/jobtech-sedc_dbs_new_241024.obj`

## Usage

### Option 1: Test First (Recommended)

Test the pipeline on a small sample (10 jobs) before processing the full dataset:

```bash
python test_extraction_pipeline.py
```

This will:
- Start the Java server
- Extract skills from 10 sample jobs
- Extract sentences for those skills
- Save results to `test_results_skills_and_sentences.csv`
- Show you sample output

**Review the test results before proceeding!**

### Option 2: Full Pipeline (Run Overnight)

Once you're satisfied with the test results, run the full extraction.

**RECOMMENDED - Run both scripts automatically:**

```bash
# Option A: Using the master script (recommended)
python run_full_extraction.py

# Option B: Using bash (Mac/Linux)
./run_full_extraction.sh

# Option C: Simple command
python batch_skill_extraction.py && python extract_skill_sentences.py
```

All options will:
- Run skill extraction first (~6-10 hours)
- Then run sentence extraction (~30-60 minutes)
- Save logs to `logs/` directory
- Exit if any step fails

**OR run scripts individually:**

#### Step 1: Extract Skills (Full Dataset)

```bash
python batch_skill_extraction.py
```

**What it does:**
- Processes all 598K jobs from the CSV
- Extracts skills using your Java-based skills extractor
- Saves results with position information (start/end) for each skill
- Processes in batches of 1,000 jobs
- Saves checkpoints every 5,000 jobs
- **Output:** `jobtech_skills_extracted.csv`

**Processing time:** Approximately 6-10 hours for full dataset (depends on your hardware)

**Output columns:**
- `job_id` - Unique job identifier
- `job_title` - Job title
- `job_description` - Original job description (with HTML)
- `skill_id` - Unique skill identifier
- `skill` - Skill name
- `skill_type` - Type (Domain, Hard, Soft, etc.)
- `skill_category` - Category classification
- `is_core_skill` - Whether it's a core/F1 skill
- `skill_count` - How many times this skill appears in the job
- `weight` - Relative importance of this skill in the job
- `start` - Starting position of skill in text
- `end` - Ending position of skill in text

#### Step 2: Extract Sentences

```bash
python extract_skill_sentences.py
```

**What it does:**
- Reads the skills extraction output
- For each skill, extracts the complete sentence where it was found
- Uses the `start` and `end` positions from skill extraction
- Cleans HTML from job descriptions
- **Output:** `jobtech_skills_with_sentences.csv`

**Processing time:** Approximately 30-60 minutes

**Additional output columns:**
- `skill_sentence` - The complete sentence containing the skill
- `sentence_length` - Length of the extracted sentence
- `job_description_clean` - Full job description with HTML removed

## Output Files

### After Full Pipeline:

1. **`jobtech_skills_extracted.csv`** (~300-500 MB)
   - All skills extracted with positions
   - Multiple rows per job (one per skill instance)

2. **`jobtech_skills_with_sentences.csv`** (~500-800 MB)
   - Same as above + extracted sentences
   - Ready for analysis

### Checkpoint Files:

During processing, checkpoint files are created:
- `jobtech_skills_extracted.csv.checkpoint_*` - Periodic saves during skill extraction
- `jobtech_skills_extracted.csv.partial` - If you interrupt the process

You can resume from these if needed.

## Data Structure

### Input Data (Original CSV)
```
job_id, job_title, job_description, ...
```

### After Skill Extraction
```
job_id, job_title, skill, skill_type, start, end, ...
```

### After Sentence Extraction
```
job_id, job_title, skill, skill_type, skill_sentence, ...
```

## Example Output

**Skill Extracted:**
```
job_id: 000008856342...
skill: Python
skill_type: Hard
start: 245
end: 251
```

**Sentence Extracted:**
```
skill_sentence: "Proficient in Python, JavaScript, TypeScript, SQL, React,
                 Node.js, and frameworks like .NET Framework/Core and WPF (MVVM)."
```

## Tips for Large Dataset Processing

1. **Test first** - Always run `test_extraction_pipeline.py` before full extraction
2. **Monitor progress** - Both scripts print progress updates
3. **Checkpoints** - If interrupted, you can resume from checkpoint files
4. **Disk space** - Ensure you have at least 5GB free space
5. **Memory** - Processing 1000 jobs at a time keeps memory usage reasonable (~2-4GB)
6. **Time** - Plan for 8-12 hours total for the full pipeline

## Troubleshooting

### Java Server Won't Start
```bash
# Check if Java is installed
java -version

# Check if port 25333 is available
lsof -i :25333

# Kill any existing Java processes
pkill -f "talented-utility-server"
```

### Out of Memory
- Reduce `BATCH_SIZE` in `batch_skill_extraction.py` (default: 1000)
- Close other applications

### Interrupted Processing
- Look for checkpoint files
- Resume by modifying the batch start index in the script

### Empty Sentences
- Some skills might not extract sentences properly if HTML structure is complex
- Check the `skill_sentence` field - empty means extraction failed for that skill

## Next Steps

After extraction, you can:

1. **Analyze skill frequency** - Which skills appear most often?
2. **Job clustering** - Group jobs by skill profiles
3. **Skill co-occurrence** - Which skills appear together?
4. **Sentence-level analysis** - Study how skills are described in context
5. **Requirements extraction** - Use the sentences to extract job requirements

## Additional Scripts

Also included (optional):

- **`clean_job_descriptions.py`** - Removes contact info, HTML, and boilerplate text
  - Creates structured fields: responsibilities, requirements, company info
  - Run separately if you need cleaned descriptions

## Questions?

The scripts include detailed logging and progress updates. If you encounter issues:

1. Check the console output for error messages
2. Review the test results to verify extraction quality
3. Examine the checkpoint files if processing was interrupted

---

**Summary:**
1. `python test_extraction_pipeline.py` - Test on 10 jobs
2. `python batch_skill_extraction.py` - Extract skills from all 598K jobs
3. `python extract_skill_sentences.py` - Extract sentences for each skill
4. Analyze `jobtech_skills_with_sentences.csv` 🎉
