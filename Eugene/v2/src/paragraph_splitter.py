# paragraph_splitter.py
import pandas as pd
from pathlib import Path
from .utils import html_to_text, split_sentences, group_paragraphs_from_text, extract_job_paragraphs
from tqdm import tqdm
import argparse


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read dataframe from CSV, Excel, or Parquet based on file extension"""
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext in ['.parquet', '.pq']:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, .xls, .parquet, or .pq")


def write_dataframe(df: pd.DataFrame, file_path: str):
    """Write dataframe to CSV, Excel, or Parquet based on file extension"""
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        df.to_csv(file_path, index=False)
    elif ext in ['.xlsx', '.xls']:
        df.to_excel(file_path, index=False)
    elif ext in ['.parquet', '.pq']:
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, .xls, .parquet, or .pq")


def paragraphs_pysbd_only(text: str):
    """Uses only pysbd sentence segmentation (no heuristics)"""
    if not text:
        return []
    return split_sentences(text)


def df_to_paragraphs(df: pd.DataFrame, html_col: str = "raw_html"):
    """Adds multiple columns for comparison (only generates missing columns):
    - paragraphs_2way: 2-way method (double newlines + pysbd with heuristics)
    - paragraphs_sentences: Just pysbd sentence segmentation
    - paragraphs_semantic_0.3: Semantic merge (threshold 0.3 - loose)
    - paragraphs_semantic_0.5: Semantic merge (threshold 0.5 - moderate)
    - paragraphs_semantic_0.75: Semantic merge (threshold 0.75 - strict)
    - paragraphs_semantic_0.9: Semantic merge (threshold 0.9 - very strict)
    """
    # Check which columns already exist
    need_2way = "paragraphs_2way" not in df.columns
    need_sentences = "paragraphs_sentences" not in df.columns
    need_sem_03 = "paragraphs_semantic_0.3" not in df.columns
    need_sem_05 = "paragraphs_semantic_0.5" not in df.columns
    need_sem_075 = "paragraphs_semantic_0.75" not in df.columns
    need_sem_09 = "paragraphs_semantic_0.9" not in df.columns

    # If all columns exist, nothing to do
    if not (need_2way or need_sentences or need_sem_03 or need_sem_05 or need_sem_075 or need_sem_09):
        print("All paragraph columns already exist. Skipping processing.")
        return df

    # Initialize lists for all columns
    paragraphs_2way = []
    paragraphs_sentences = []
    paragraphs_sem_03 = []
    paragraphs_sem_05 = []
    paragraphs_sem_075 = []
    paragraphs_sem_09 = []

    # Generate only missing columns
    for html in tqdm(df[html_col].fillna("").astype(str), desc="Splitting paragraphs"):
        text = html_to_text(html)

        if need_2way:
            paras_2way = group_paragraphs_from_text(text)
            paragraphs_2way.append(paras_2way)

        if need_sentences:
            paras_sentences = paragraphs_pysbd_only(text)
            paragraphs_sentences.append(paras_sentences)

        if need_sem_03:
            paras = extract_job_paragraphs(html, similarity_threshold=0.3)
            paragraphs_sem_03.append(paras)

        if need_sem_05:
            paras = extract_job_paragraphs(html, similarity_threshold=0.5)
            paragraphs_sem_05.append(paras)

        if need_sem_075:
            paras = extract_job_paragraphs(html, similarity_threshold=0.75)
            paragraphs_sem_075.append(paras)

        if need_sem_09:
            paras = extract_job_paragraphs(html, similarity_threshold=0.9)
            paragraphs_sem_09.append(paras)

    # Add only the columns we generated
    if need_2way:
        df["paragraphs_2way"] = paragraphs_2way
    if need_sentences:
        df["paragraphs_sentences"] = paragraphs_sentences
    if need_sem_03:
        df["paragraphs_semantic_0.3"] = paragraphs_sem_03
    if need_sem_05:
        df["paragraphs_semantic_0.5"] = paragraphs_sem_05
    if need_sem_075:
        df["paragraphs_semantic_0.75"] = paragraphs_sem_075
    if need_sem_09:
        df["paragraphs_semantic_0.9"] = paragraphs_sem_09

    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Input file (.csv, .xlsx, .xls, .parquet, .pq)")
    parser.add_argument("--html_col", default="raw_html")
    parser.add_argument("--output_file", required=True, help="Output file (.csv, .xlsx, .xls, .parquet, .pq)")
    parser.add_argument("--sample", type=int, help="Process only first N rows (for testing)")
    args = parser.parse_args()

    df = read_dataframe(args.input_file)

    # Sample if requested
    if args.sample:
        print(f"Processing sample of {args.sample} rows (out of {len(df)} total)")
        df = df.head(args.sample)

    df = df_to_paragraphs(df, html_col=args.html_col)
    write_dataframe(df, args.output_file)
    print("Wrote paragraphized dataframe to", args.output_file)

