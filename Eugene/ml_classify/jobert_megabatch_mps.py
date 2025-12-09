"""
ULTRA-OPTIMIZED VERSION FOR 16M ROWS
Strategy: Process all sentences from multiple rows in mega-batches
This is 10-20x faster than processing row-by-row
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import html
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pyarrow.parquet as pq
import sys
import logging
import time
from datetime import datetime, timedelta
import gc

sys.path.insert(0, "/Users/eugene/Documents/py_utils")
from jt_golden_utility import setup_logger

from jobert_utils import preprocess_line_breaks, split_into_sentences

# Setup logger
logger = setup_logger(log_dir='Eugene/ml-classify/logs', log_file='megabatch_mps', retention_days=30)
logger.setLevel(logging.INFO)

# Configuration
MEGA_BATCH_SIZE = 512  # Process 512 sentences at once on GPU (increase for better GPU utilization)
CHUNK_SIZE = 100  # Read 1000 rows at a time from disk
SAVE_EVERY = 100  # Save intermediate results every 1k rows

# File paths
INPUT_FILE = "Eugene/sampled_jobs_final.parquet"  # Input parquet file
OUTPUT_DIR = "Eugene/ml-classify"  # Base output directory
CHUNKS_DIR = f"{OUTPUT_DIR}/chunks"  # Directory for intermediate chunk files
OUTPUT_PARQUET = f"{OUTPUT_DIR}/categorized_jobs_megabatch.parquet"  # Final output parquet
EXCEL_CHECKPOINT_PREFIX = f"{OUTPUT_DIR}/categorized_jobs_megabatch"  # Prefix for checkpoint Excel files
EXCEL_FINAL = f"{OUTPUT_DIR}/categorized_jobs_megabatch_final.xlsx"  # Final Excel sample

# Use first 5 labels only
label_names = ['About the Company', 'Job Description', 'Job Requirements', 'Responsibilities', 'Benefits']

print("Loading models...")
logger.info("=" * 70)
logger.info("Starting JoBert Megabatch Classification")
logger.info("=" * 70)

# Load transformer model
model_load_start = time.time()
tokenizer = AutoTokenizer.from_pretrained("AhmedBou/JoBert")
model = AutoModelForSequenceClassification.from_pretrained("AhmedBou/JoBert")
model.eval()

# Setup device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using Apple Silicon GPU (MPS)")
    logger.info("Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA GPU")
    logger.info("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("⚠ Using CPU")
    logger.warning("Using CPU (slow performance expected)")

model = model.to(device)
model_load_time = time.time() - model_load_start
print(f"✓ Models loaded in {model_load_time:.2f}s")
logger.info(f"Models loaded in {model_load_time:.2f}s")

def classify_mega_batch(sentences, batch_size=MEGA_BATCH_SIZE):
    """Classify many sentences in large batches"""
    if not sentences:
        return []

    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            try:
                inputs = tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,  # Reduced from 512 for faster inference
                    padding=True
                )
                inputs = {key: val.to(device) for key, val in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

                for pred in predictions:
                    if pred < len(label_names):
                        all_predictions.append(label_names[pred])
                    else:
                        all_predictions.append(None)

                # Explicit cleanup
                del inputs, outputs, logits, predictions

            except Exception as e:
                print(f"Error in batch: {e}")
                all_predictions.extend([None] * len(batch))

    return all_predictions

def process_chunk_megabatch(chunk_df):
    """
    OPTIMIZED: Process entire chunk's sentences in mega-batches
    Instead of row-by-row, collect all sentences first, classify in mega-batches,
    then distribute results back to rows
    """
    # Step 1: Preprocess all job descriptions
    chunk_df['job_description'] = chunk_df['job_description'].apply(preprocess_line_breaks)

    # Step 2: Split all into sentences and track which row each sentence belongs to
    all_sentences = []
    sentence_to_row_idx = []  # Maps sentence index to row index

    for idx, row in chunk_df.iterrows():
        sentences = split_into_sentences(row['job_description'])
        all_sentences.extend(sentences)
        sentence_to_row_idx.extend([idx] * len(sentences))

    # Step 3: Classify ALL sentences in mega-batches (FAST!)
    print(f"  Classifying {len(all_sentences):,} sentences in mega-batches...")
    all_labels = classify_mega_batch(all_sentences, batch_size=MEGA_BATCH_SIZE)

    # Step 4: Distribute results back to rows
    row_data = {idx: {
        'sentences': [],
        'labels': [],
        'label_sentences': defaultdict(list)
    } for idx in chunk_df.index}

    for sentence, label, row_idx in zip(all_sentences, all_labels, sentence_to_row_idx):
        row_data[row_idx]['sentences'].append(sentence)
        row_data[row_idx]['labels'].append(label)
        if label:
            row_data[row_idx]['label_sentences'][label].append(sentence)

    # Step 5: Build result dataframe
    results = []
    for idx, row in chunk_df.iterrows():
        data = row_data[idx]
        result = {
            'job_id': row['job_id'],
            'job_description': row['job_description'],
            'sentences': data['sentences'],
            'sentence_count': len(data['sentences']),
            **{label: " ".join(data['label_sentences'][label]) for label in label_names}
        }
        results.append(result)

    # Clear GPU cache and force garbage collection
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()

    gc.collect()

    return pd.DataFrame(results)

def clean_for_excel(text):
    """Clean text for Excel compatibility"""
    if isinstance(text, list):
        return str(text)[:32767]
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text[:32767]

def save_excel_sample(df, output_path, sample_size=1000):
    """Save a sample of the dataframe to Excel"""
    try:
        df_sample = df.sample(n=min(sample_size, len(df)), random_state=42).copy()

        for col in df_sample.columns:
            if df_sample[col].dtype == 'object':
                df_sample[col] = df_sample[col].apply(clean_for_excel)

        df_sample.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  📊 Excel sample saved: {output_path}")
        logger.info(f"Excel sample saved: {output_path}")
    except Exception as e:
        print(f"  ⚠️  Failed to save Excel: {e}")
        logger.warning(f"Failed to save Excel: {e}")

# Main processing
print("\nLoading data...")
logger.info("Starting data processing")

# For 16M rows, use chunked reading to avoid memory issues
print(f"Processing file: {INPUT_FILE}")
print(f"Chunk size: {CHUNK_SIZE:,} rows")
print(f"Mega-batch size: {MEGA_BATCH_SIZE} sentences")
logger.info(f"Configuration: CHUNK_SIZE={CHUNK_SIZE:,}, MEGA_BATCH_SIZE={MEGA_BATCH_SIZE}, SAVE_EVERY={SAVE_EVERY:,}")

# Get total row count
df_info = pd.read_parquet(INPUT_FILE, columns=['job_id'])
total_rows = len(df_info)
print(f"Total rows to process: {total_rows:,}")
logger.info(f"Total rows to process: {total_rows:,}")
del df_info

# Start overall timer
overall_start_time = time.time()

# Create chunks directory if it doesn't exist
import os
os.makedirs(CHUNKS_DIR, exist_ok=True)
print(f"✓ Chunk directory ready: {CHUNKS_DIR}")

# Process in chunks
processed_count = 0
chunk_files = []  # Track individual chunk files for merging later

# Read and process in chunks using PyArrow
parquet_file = pq.ParquetFile(INPUT_FILE)
chunk_num = 0

for batch in parquet_file.iter_batches(batch_size=CHUNK_SIZE, columns=["job_id", "job_description"]):
    chunk_start_time = time.time()
    chunk_df = batch.to_pandas()

    print(f"\n[Chunk {chunk_num + 1}] Processing rows {processed_count:,} to {processed_count + len(chunk_df):,}")
    logger.info(f"[Chunk {chunk_num + 1}] Processing rows {processed_count:,} to {processed_count + len(chunk_df):,}")

    # Process chunk
    result_df = process_chunk_megabatch(chunk_df)

    # Save this chunk to its own file (O(1) operation)
    chunk_output_file = f"{CHUNKS_DIR}/chunk_{chunk_num:06d}.parquet"
    result_df.to_parquet(chunk_output_file, index=False)
    chunk_files.append(chunk_output_file)

    processed_count += len(chunk_df)
    chunk_time = time.time() - chunk_start_time
    rows_per_sec = len(chunk_df) / chunk_time
    print(f"  ⏱️  Chunk time: {chunk_time:.2f}s ({rows_per_sec:.0f} rows/sec)")
    print(f"  💾 Saved chunk to: {chunk_output_file}")
    logger.info(f"[Chunk {chunk_num + 1}] Completed in {chunk_time:.2f}s ({rows_per_sec:.0f} rows/sec)")
    logger.info(f"Chunk saved to: {chunk_output_file}")

    # Save Excel sample at milestones (every 1000 rows)
    if processed_count % 1000 == 0:
        print(f"  📊 Creating Excel sample at {processed_count:,} rows...")
        # Read just the latest chunk for sampling (not all data)
        save_excel_sample(result_df, f"{EXCEL_CHECKPOINT_PREFIX}_{processed_count}.xlsx", sample_size=min(100, len(result_df)))

    # Explicit cleanup
    del result_df, chunk_df
    gc.collect()

    # Progress update with ETA
    progress_pct = (processed_count / total_rows) * 100
    elapsed_time = time.time() - overall_start_time
    if processed_count > 0:
        estimated_total_time = elapsed_time * (total_rows / processed_count)
        eta_seconds = estimated_total_time - elapsed_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        print(f"  Progress: {progress_pct:.1f}% ({processed_count:,}/{total_rows:,}) | ETA: {eta_str}")
        logger.info(f"Progress: {progress_pct:.1f}% ({processed_count:,}/{total_rows:,}) | ETA: {eta_str}")

    chunk_num += 1

# Merge all chunk files into final output
print("\n" + "="*70)
print("💾 Merging all chunk files into final output...")
logger.info("Starting merge of chunk files")

merge_start = time.time()
print(f"  Merging {len(chunk_files)} chunk files...")

# Read and concatenate all chunk files
chunk_dfs = []
for i, chunk_file in enumerate(chunk_files):
    print(f"  Reading chunk {i+1}/{len(chunk_files)}...", end='\r')
    chunk_dfs.append(pd.read_parquet(chunk_file))

print(f"  Concatenating {len(chunk_dfs)} chunks...                  ")
final_df = pd.concat(chunk_dfs, ignore_index=True)

print(f"  Writing final parquet...")
final_df.to_parquet(OUTPUT_PARQUET, index=False)

merge_time = time.time() - merge_start
print(f"✓ Merge complete in {merge_time:.2f}s")
logger.info(f"Merge completed in {merge_time:.2f}s")

# Save final Excel sample
save_excel_sample(final_df, EXCEL_FINAL, sample_size=1000)

# Cleanup chunk dataframes
del chunk_dfs
gc.collect()

# Calculate final statistics
total_time = time.time() - overall_start_time
total_time_str = str(timedelta(seconds=int(total_time)))
avg_rows_per_sec = processed_count / total_time

print("\n" + "="*70)
print("✓ PROCESSING COMPLETE!")
print(f"Total rows processed: {processed_count:,}")
print(f"Total time: {total_time_str} ({total_time:.2f}s)")
print(f"Average speed: {avg_rows_per_sec:.0f} rows/sec")
print(f"Output file: {OUTPUT_PARQUET}")

logger.info("=" * 70)
logger.info("PROCESSING COMPLETE!")
logger.info(f"Total rows processed: {processed_count:,}")
logger.info(f"Total time: {total_time_str} ({total_time:.2f}s)")
logger.info(f"Average speed: {avg_rows_per_sec:.0f} rows/sec")
logger.info(f"Output file: {OUTPUT_PARQUET}")

# Summary of final result
print(f"Final dataset shape: {final_df.shape}")
print(f"Columns: {list(final_df.columns)}")
logger.info(f"Final dataset shape: {final_df.shape}")

print("="*70)
print("\n🎉 All done!")
logger.info("="*70)
logger.info("All processing complete!")
