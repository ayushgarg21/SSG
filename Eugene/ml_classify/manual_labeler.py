"""
Manual sentence labeling tool
Press 1-5 to label, s to skip, q to quit
Progress is saved automatically
"""
import pandas as pd
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.dirname(SCRIPT_DIR)

# Files
INPUT_FILE = os.path.join(BASE_PATH, "diverse_sample_1k.parquet")
OUTPUT_FILE = os.path.join(BASE_PATH, "manual_labels.parquet")
CHECKPOINT_FILE = os.path.join(SCRIPT_DIR, "manual_labeling_checkpoint.json")

# Import sentence splitter
from sentence_splitter_v2 import split_into_sentences_v2

# Labels
LABELS = {
    '1': 'About the Company',
    '2': 'Job Description',
    '3': 'Job Requirements',
    '4': 'Responsibilities',
    '5': 'Benefits',
}

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'job_idx': 0, 'sentence_idx': 0}

def save_checkpoint(job_idx, sentence_idx):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'job_idx': job_idx, 'sentence_idx': sentence_idx}, f)

def load_existing_labels():
    if os.path.exists(OUTPUT_FILE):
        return pd.read_parquet(OUTPUT_FILE).to_dict('records')
    return []

def save_labels(labels):
    if labels:
        df = pd.DataFrame(labels)
        df.to_parquet(OUTPUT_FILE, index=False)

def main():
    print("="*80)
    print("MANUAL LABELING TOOL")
    print("="*80)
    print("\nKeys:")
    for key, label in LABELS.items():
        print(f"  {key} = {label}")
    print("  s = skip this sentence")
    print("  b = go back one sentence")
    print("  q = quit and save")
    print("="*80)

    # Load data
    df = pd.read_parquet(INPUT_FILE)
    checkpoint = load_checkpoint()
    labels = load_existing_labels()

    print(f"\nLoaded {len(labels)} existing labels")
    print(f"Resuming from job {checkpoint['job_idx']}, sentence {checkpoint['sentence_idx']}")
    print("\nPress Enter to start...")
    input()

    job_idx = checkpoint['job_idx']
    sentence_idx = checkpoint['sentence_idx']

    # Build list of all sentences
    all_sentences = []
    for idx, row in df.iterrows():
        job_id = row['job_id']
        sentences = split_into_sentences_v2(str(row['job_description']))
        for sent in sentences:
            if sent.strip():
                all_sentences.append({'job_id': job_id, 'sentence': sent.strip()})

    print(f"\nTotal sentences to label: {len(all_sentences)}")

    # Start from checkpoint
    current_idx = 0
    for i, row in df.iterrows():
        if i < job_idx:
            sentences = split_into_sentences_v2(str(row['job_description']))
            current_idx += len([s for s in sentences if s.strip()])
        elif i == job_idx:
            current_idx += sentence_idx
            break

    # Label loop
    while current_idx < len(all_sentences):
        item = all_sentences[current_idx]

        # Clear screen
        print("\n" + "="*80)
        print(f"Progress: {len(labels)} labeled | {current_idx + 1}/{len(all_sentences)} ({100*(current_idx+1)/len(all_sentences):.1f}%)")
        print("="*80)
        print(f"\nJob ID: {item['job_id']}")
        print("-"*40)
        print(f"\n{item['sentence']}\n")
        print("-"*40)
        print("\n[1] About Company  [2] Job Desc  [3] Requirements  [4] Responsibilities  [5] Benefits")
        print("[s] skip  [b] back  [q] quit")

        choice = input("\nLabel: ").strip().lower()

        if choice == 'q':
            # Find job_idx and sentence_idx for checkpoint
            count = 0
            for i, row in df.iterrows():
                sentences = split_into_sentences_v2(str(row['job_description']))
                num_sents = len([s for s in sentences if s.strip()])
                if count + num_sents > current_idx:
                    save_checkpoint(i, current_idx - count)
                    break
                count += num_sents
            save_labels(labels)
            print(f"\nSaved {len(labels)} labels. Goodbye!")
            break
        elif choice == 'b':
            if current_idx > 0:
                current_idx -= 1
                # Remove last label if exists
                if labels and labels[-1]['sentence'] == all_sentences[current_idx]['sentence']:
                    labels.pop()
            continue
        elif choice == 's':
            current_idx += 1
            continue
        elif choice in LABELS:
            labels.append({
                'job_id': item['job_id'],
                'sentence': item['sentence'],
                'label': LABELS[choice]
            })
            current_idx += 1

            # Auto-save every 50 labels
            if len(labels) % 50 == 0:
                save_labels(labels)
                print(f"\n[Auto-saved {len(labels)} labels]")
        else:
            print("Invalid input. Use 1-5, s, b, or q")

    if current_idx >= len(all_sentences):
        save_labels(labels)
        print(f"\nDone! Labeled all {len(labels)} sentences.")

if __name__ == "__main__":
    main()
