import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
import html
from collections import defaultdict
from tqdm import tqdm
from Eugene.ml_classify.jobert_utils import preprocess_line_breaks, split_into_sentences
tqdm.pandas()

# Use first 5 labels only
LABEL_NAMES = ['About the Company', 'Job Description', 'Job Requirements', 'Responsibilities', 'Benefits','Others']


def load_model():
    """Load the JoBert model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("AhmedBou/JoBert")
    model = AutoModelForSequenceClassification.from_pretrained("AhmedBou/JoBert")
    return model, tokenizer


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


def process_job_descriptions(df, model, tokenizer, label_names):
    """Process and categorize job descriptions"""
    # Preprocess job descriptions
    print("\nPreprocessing job descriptions...")
    df['job_description'] = df['job_description'].progress_apply(preprocess_line_breaks)

    # Initialize columns for each label
    for label in label_names:
        df[label] = ""

    # Add columns for sentences list and count
    df['sentences'] = None
    df['sentence_count'] = 0

    # Process each row
    print("\nProcessing and categorizing sentences...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        job_desc = row['job_description']
        sentences = split_into_sentences(job_desc, smart_split=True)

        # Store the sentences list and count
        df.at[idx, 'sentences'] = sentences
        df.at[idx, 'sentence_count'] = len(sentences)

        # Dictionary to collect sentences for each label
        label_sentences = defaultdict(list)

        for sentence in sentences:
            label = classify_sentence(sentence, model, tokenizer, label_names)
            if label:
                label_sentences[label].append(sentence)

        # Assign categorized sentences to columns
        for label in label_names:
            df.at[idx, label] = " ".join(label_sentences[label])

    print("\nCategorization complete!")
    return df


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


def save_output(df, output_path, output_format='both'):
    """
    Save the processed dataframe to the specified format

    Args:
        df: DataFrame to save
        output_path: Base path for output file (without extension)
        output_format: 'excel', 'parquet', or 'both'
    """
    if output_format not in ['excel', 'parquet', 'both']:
        raise ValueError("output_format must be 'excel', 'parquet', or 'both'")

    if output_format in ['parquet', 'both']:
        parquet_path = f"{output_path}.parquet"
        df.to_parquet(parquet_path, index=False)
        print(f"Saved to parquet: {parquet_path}")

    if output_format in ['excel', 'both']:
        # Clean all text columns for Excel
        df_excel = df.copy()
        for col in df_excel.columns:
            if df_excel[col].dtype == 'object':
                df_excel[col] = df_excel[col].apply(clean_for_excel)

        excel_path = f"{output_path}.xlsx"
        df_excel.to_excel(excel_path, index=False, engine='openpyxl')
        print(f"Saved to Excel: {excel_path}")


def main(input_path, output_path, sample_size=None, random_state=42, output_format='both'):
    """
    Main function to process job descriptions and categorize them

    Args:
        input_path: Path to input file (Excel or parquet)
        output_path: Base path for output file (without extension)
        sample_size: Number of rows to sample (None for all rows)
        random_state: Random seed for sampling
        output_format: 'excel', 'parquet', or 'both'
    """
    # Read input file
    print(f"Reading input file: {input_path}")
    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_excel(input_path)

    print(f"Original dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Sample rows if needed
    if sample_size is not None and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=random_state)
    else:
        df_sample = df.copy()

    df_sample = df_sample[["job_id", "job_description"]].copy()
    print(f"Processing dataframe shape: {df_sample.shape}")

    # Load model and tokenizer
    print("\nLoading model...")
    model, tokenizer = load_model()

    # Process job descriptions
    df_result = process_job_descriptions(df_sample, model, tokenizer, LABEL_NAMES)

    # Save output
    save_output(df_result, output_path, output_format)

    print("\nDone! Summary:")
    print(f"Total rows processed: {len(df_result)}")
    print(f"Columns: {list(df_result.columns)}")


if __name__ == "__main__":
    # Configuration
    input_file = "v2/Sample-extractions.xlsx"
    output_base = "v2/categorized_jobs"
    model,tokenizer=load_model()
    sentence=' Competitive, based on experience + benefit.'
 
    result=classify_sentence(sentence,model,tokenizer,LABEL_NAMES)
    print(result)
    # Run main function
    # main(
    #     input_path=input_file,
    #     output_path=output_base,
    #     # sample_size=100,
    #     # random_state=42,
    #     output_format='excel'  # Options: 'excel', 'parquet', or 'both'
    # )
