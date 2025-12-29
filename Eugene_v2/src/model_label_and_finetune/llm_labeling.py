"""
LLM Labeling Script for JoBert Fine-tuning
Uses GPT-4o-mini to label sentences from job descriptions
"""
import pandas as pd
import json
import time
import os
from openai import OpenAI
from tqdm import tqdm
import sys
import ast

from Eugene.ml_classify.jobert_utils import preprocess_line_breaks,split_into_sentences

from Eugene_v2.src.utils import read_dataframe, write_dataframe

# ============================================================================
# Configuration
# ============================================================================
API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
MODEL = "gpt-4o-mini"  # Fast and cheap
BATCH_SIZE = 5  # Jobs per API call
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1  # seconds between calls

INPUT_FILE = "results_data/Sample-extractions_20251127_with_base_min_5.xlsx"

# Generate output and checkpoint file paths based on input file
input_dir = os.path.dirname(INPUT_FILE)
input_basename = os.path.basename(INPUT_FILE)
input_name, input_ext = os.path.splitext(input_basename)

OUTPUT_FILE = os.path.join(input_dir, f"{input_name}_labeled_sentences{input_ext}")
CHECKPOINT_FILE = f"/Users/eugene/Documents/Task/SSG/Eugene_v2/checkpoints/{input_name}_checkpoint.json"

# The 5 categories (same as JoBert)
CATEGORIES = [
    "About the Company",
    "Job Description",
    "Job Requirements",
    "Responsibilities",
    "Benefits"
]

# ============================================================================
# OpenAI Client
# ============================================================================
client = OpenAI(api_key=API_KEY)

# ============================================================================
# Labeling Prompt
# ============================================================================
SYSTEM_PROMPT = """You are a job description classifier. For each sentence or paragraph, classify it into exactly ONE of these categories:

1. About the Company - Information about the company (history, culture, values, size, industry)
2. Job Description - General overview of the role/position
3. Job Requirements - Required skills, experience, education, qualifications
4. Responsibilities - Job duties, tasks, what the person will do
5. Benefits - Salary, perks, insurance, leave, working hours, bonuses

If a sentence doesn't clearly fit any category (like location, contact info, boilerplate), classify it as "Other".


Respond ONLY with a JSON array of labels, one for each sentence. Example:
["Responsibilities", "Job Requirements", "Benefits", "About the Company"]"""

def create_labeling_prompt(sentences):
    """Create the prompt for labeling sentences"""
    numbered = "\n".join([f"{i+1}. {s}" for i, s in enumerate(sentences)])
    return f"Classify each sentence:\n\n{numbered}"

def label_sentences_batch(sentences):
    """Label a batch of sentences using GPT-4o-mini"""
    if not sentences:
        return []

    for attempt in range(MAX_RETRIES):
        try:
            prompt=create_labeling_prompt (sentences)
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content":prompt }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=2000,
                timeout=60
            )
            # response = client.responses.create(
            # model=MODEL,
            # input=[
            #         {"role": "system", "content": SYSTEM_PROMPT},
            #         {"role": "user", "content": create_labeling_prompt(sentences)}
            #     ],   # messages or plain text both OK
            #         temperature=0.1,  # Low temperature for consistency

            # timeout=60
        # )

            # Parse response
            content = response.choices[0].message.content.strip()

            # content = response.output_text.strip()
            # Try to extract JSON array
            if content.startswith("["):
                labels = json.loads(content)

            else:
                # Try to find JSON in response
                start = content.find("[")
                end = content.rfind("]") + 1
                if start >= 0 and end > start:
                    labels = json.loads(content[start:end])
                else:
                    print(f"Warning: Could not parse response: {content[:100]}")
                    labels = ["Other"] * len(sentences)

            # Validate labels
            validated = []
            for label in labels:
                if label in CATEGORIES:
                    validated.append(label)
                elif label == "Other":
                    validated.append("Other")
                else:
                    # Try to match partial
                    matched = False
                    for cat in CATEGORIES:
                        if cat.lower() in label.lower() or label.lower() in cat.lower():
                            validated.append(cat)
                            matched = True
                            break
                    if not matched:
                        validated.append("Other")

            # Ensure we have the right number of labels
            while len(validated) < len(sentences):
                validated.append("Other")
            validated = validated[:len(sentences)]

            return validated

        except Exception as e:
            print(f"Error (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return ["Other"] * len(sentences)

    return ["Other"] * len(sentences)

def load_checkpoint():
    """Load progress from checkpoint file"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {"processed_jobs": [], "results": []}

def save_checkpoint(checkpoint):
    """Save progress to checkpoint file"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

# ============================================================================
# Main Processing
# ============================================================================
def main():
    print("="*80)
    print("LLM LABELING FOR JOBERT FINE-TUNING")
    print("="*80)

    # Load data
    print(f"\nLoading data from {INPUT_FILE}...")
    # df = pd.read_parquet(INPUT_FILE)
    df = read_dataframe(INPUT_FILE)
    print(f"Loaded {len(df)} jobs")

    # Load checkpoint
    checkpoint = load_checkpoint()
    processed_jobs = set(checkpoint["processed_jobs"])
    results = checkpoint["results"]

    print(f"Resuming from checkpoint: {len(processed_jobs)} jobs already processed")

    # Filter out already processed
    remaining_df = df[~df['jobId'].isin(processed_jobs)]
    print(f"Jobs remaining: {len(remaining_df)}")

    if len(remaining_df) == 0:
        print("All jobs already processed!")
    else:
        # Process jobs
        total_sentences = 0
        total_cost_estimate = 0

        """Special; case"""
        remaining_df['neu_chunks'] = remaining_df['neu_chunks'].apply(ast.literal_eval)

        for idx, row in tqdm(remaining_df.iterrows(), total=len(remaining_df), desc="Labeling"):
            job_id = row['jobId']
            # job_desc = row['neu_chunks']

            # Preprocess and split
            # cleaned = preprocess_line_breaks(job_desc)
            # sentences = split_into_sentences(cleaned, smart_split=True)
            sentences = row['neu_chunks']
            if not sentences:
                processed_jobs.add(job_id)
                continue

            # Label sentences in batches
            all_labels = []
            for i in range(0, len(sentences), 20):  # 20 sentences per API call
                batch = sentences[i:i+20]
                labels = label_sentences_batch(batch)
                all_labels.extend(labels)
                time.sleep(RATE_LIMIT_DELAY)

            # Store results
            for sentence, label in zip(sentences, all_labels):
                results.append({
                    "job_id": job_id,
                    "sentence": sentence,
                    "label": label
                })

            processed_jobs.add(job_id)
            total_sentences += len(sentences)

            # Save checkpoint every 10 jobs
            if len(processed_jobs) % 10 == 0:
                checkpoint = {
                    "processed_jobs": list(processed_jobs),
                    "results": results
                }
                save_checkpoint(checkpoint)
                print(f"\n  Checkpoint saved: {len(processed_jobs)} jobs, {len(results)} sentences")

    # Final save
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    results_df = pd.DataFrame(results)
    write_dataframe(results_df, OUTPUT_FILE)
    # results_df.to_parquet(OUTPUT_FILE, index=False)
    # print(f"Saved to: {OUTPUT_FILE}")

    # # Also save as Excel for review
    # excel_file = OUTPUT_FILE.replace('.parquet', '.xlsx')
    # results_df.to_excel(excel_file, index=False)
    # print(f"Saved to: {excel_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total jobs processed: {len(processed_jobs)}")
    print(f"Total sentences labeled: {len(results_df)}")
    print(f"\nLabel distribution:")
    print(results_df['label'].value_counts())

    # Estimate cost
    total_tokens = len(results_df) * 50  # rough estimate
    cost = (total_tokens / 1_000_000) * 0.15 + (total_tokens / 1_000_000) * 0.60
    print(f"\nEstimated cost: ${cost:.2f}")

if __name__ == "__main__":
    main()
