import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import html
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from jobert_utils import preprocess_line_breaks, split_into_sentences

# file_path="Eugene/sampled_jobs_unsorted.xlsx"
file_path="Eugene/sampled_jobs_final.parquet"
df = pd.read_parquet(file_path)
print(f"Original dataframe shape: {df.shape}")
print(df.columns)

# Sample 1k random rows
df_sample = df.sample(n=min(100, len(df)), random_state=42)
df_sample = df_sample[["job_id","job_description"]].copy()
print(f"Sampled dataframe shape: {df_sample.shape}")

# Use first 5 labels only
label_names = ['About the Company', 'Job Description', 'Job Requirements', 'Responsibilities', 'Benefits']

# Global variables will be initialized per worker
model = None
tokenizer = None

def init_worker():
    """Initialize heavy models once per worker process"""
    global model, tokenizer

    print(f"Initializing worker {multiprocessing.current_process().name}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("AhmedBou/JoBert")
    model = AutoModelForSequenceClassification.from_pretrained("AhmedBou/JoBert")
    model.eval()

def classify_sentences_batch(sentences, batch_size=32):
    """Classify multiple sentences in batches"""
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
                    max_length=512,
                    padding=True
                )

                outputs = model(**inputs)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()

                for pred in predictions:
                    if pred < len(label_names):
                        all_predictions.append(label_names[pred])
                    else:
                        all_predictions.append(None)

            except Exception as e:
                print(f"Error classifying batch: {e}")
                all_predictions.extend([None] * len(batch))

    return all_predictions

def process_single_job(row):
    """Process a single job description - runs in parallel"""
    job_id = row['job_id']
    job_desc = preprocess_line_breaks(row['job_description'])

    # Split into sentences
    sentences = split_into_sentences(job_desc, smart_split=True)

    # Classify sentences in batches
    labels = classify_sentences_batch(sentences, batch_size=32)

    # Organize by label
    label_sentences = defaultdict(list)
    for sentence, label in zip(sentences, labels):
        if label:
            label_sentences[label].append(sentence)

    # Build result dictionary
    result = {
        'job_id': job_id,
        'job_description': job_desc,
        'sentences': sentences,
        'sentence_count': len(sentences)
    }

    # Add categorized sentences
    for label in label_names:
        result[label] = " ".join(label_sentences[label])

    return result

# Determine number of workers
n_workers = max(1, multiprocessing.cpu_count() - 1)
print(f"\nUsing {n_workers} worker processes")

# Process in parallel
print("\nProcessing job descriptions in parallel...")
results = Parallel(n_jobs=n_workers, backend='multiprocessing')(
    delayed(process_single_job)(row)
    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample))
)

# Convert results to DataFrame
df_result = pd.DataFrame(results)

print("\nCategorization complete!")

# Save to parquet
output_parquet = "Eugene/ml-classify/categorized_jobs_multiprocessing.parquet"
df_result.to_parquet(output_parquet, index=False)
print(f"Saved to parquet: {output_parquet}")

# Prepare for Excel: remove control characters and unescape HTML
def clean_for_excel(text):
    """Remove control characters and unescape HTML"""
    if isinstance(text, list):
        return [clean_for_excel(item) for item in text]
    if not isinstance(text, str):
        return text
    text = html.unescape(text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text

# Clean all text columns
df_excel = df_result.copy()
for col in df_excel.columns:
    if df_excel[col].dtype == 'object':
        df_excel[col] = df_excel[col].apply(clean_for_excel)

# Save to Excel
output_excel = "Eugene/ml-classify/categorized_jobs_multiprocessing.xlsx"
df_excel.to_excel(output_excel, index=False, engine='openpyxl')
print(f"Saved to Excel: {output_excel}")

print("\nDone! Summary:")
print(f"Total rows processed: {len(df_result)}")
print(f"Columns: {list(df_result.columns)}")
