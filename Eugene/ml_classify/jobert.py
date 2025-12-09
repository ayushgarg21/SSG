import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import html
from collections import defaultdict
from tqdm import tqdm
from jobert_utils import preprocess_line_breaks, split_into_sentences
tqdm.pandas()

# file_path="Eugene/sampled_jobs_unsorted.xlsx"
file_path="Eugene/sampled_jobs_final.parquet"
df = pd.read_parquet(file_path)
print(f"Original dataframe shape: {df.shape}")
print(df.columns)

# Sample 1k random rows
df_sample = df.sample(n=min(100, len(df)), random_state=42)
df_sample = df_sample[["job_id","job_description"]].copy()
print(f"Sampled dataframe shape: {df_sample.shape}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("AhmedBou/JoBert")
model = AutoModelForSequenceClassification.from_pretrained("AhmedBou/JoBert")

# Use first 5 labels only
label_names = ['About the Company', 'Job Description', 'Job Requirements', 'Responsibilities', 'Benefits']

def classify_sentence(sentence, model, tokenizer, label_names):
    """Classify a single sentence"""
    try:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, max_length=512)
        inputs = {key: val for key, val in inputs.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits).item()
        # Only return if prediction is within our 5 labels
        if prediction < len(label_names):
            return label_names[prediction]
        return None
    except Exception as e:
        print(f"Error classifying sentence: {e}")
        return None

# Preprocess job descriptions
print("\nPreprocessing job descriptions...")
df_sample['job_description'] = df_sample['job_description'].progress_apply(preprocess_line_breaks)

# Initialize columns for each label
for label in label_names:
    df_sample[label] = ""

# Add columns for sentences list and count
df_sample['sentences'] = None
df_sample['sentence_count'] = 0

# Process each row
print("\nProcessing and categorizing sentences...")
for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    job_desc = row['job_description']
    sentences = split_into_sentences(job_desc, smart_split=True)

    # Store the sentences list and count
    df_sample.at[idx, 'sentences'] = sentences
    df_sample.at[idx, 'sentence_count'] = len(sentences)

    # Dictionary to collect sentences for each label
    label_sentences = defaultdict(list)

    for sentence in sentences:
        label = classify_sentence(sentence, model, tokenizer, label_names)
        if label:
            label_sentences[label].append(sentence)

    # Assign categorized sentences to columns
    for label in label_names:
        df_sample.at[idx, label] = " ".join(label_sentences[label])

print("\nCategorization complete!")

# Save to parquet
output_parquet = "Eugene/ml-classify/categorized_jobs.parquet"
df_sample.to_parquet(output_parquet, index=False)
print(f"Saved to parquet: {output_parquet}")

# Prepare for Excel: remove control characters and unescape HTML
def clean_for_excel(text):
    """Remove control characters and unescape HTML"""
    # Handle lists (like the sentences column)
    if isinstance(text, list):
        return [clean_for_excel(item) for item in text]
    # Handle strings
    if not isinstance(text, str):
        return text
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove control characters (keep newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return text

# Clean all text columns
df_excel = df_sample.copy()
for col in df_excel.columns:
    if df_excel[col].dtype == 'object':
        df_excel[col] = df_excel[col].apply(clean_for_excel)

# Save to Excel
output_excel = "Eugene/ml-classify/categorized_jobs.xlsx"
df_excel.to_excel(output_excel, index=False, engine='openpyxl')
print(f"Saved to Excel: {output_excel}")

print("\nDone! Summary:")
print(f"Total rows processed: {len(df_sample)}")
print(f"Columns: {list(df_sample.columns)}")
