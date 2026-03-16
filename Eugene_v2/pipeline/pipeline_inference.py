"""
Pipeline 1: Inference Pipeline
Chains: Paragraph Splitter -> Inference/Prediction -> Extract Primary Activities

This pipeline uses a pre-trained model for inference without retraining.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Add Eugene_v2 to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Eugene_v2.src.utils import read_dataframe, write_dataframe
from Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2 import df_to_paragraphs as split_paragraphs
from Eugene_v2.src.inference_prediction.inference import infer_job_sections_from_df
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd


def run_paragraph_splitting(input_file, html_col, sample=None):
    """Stage 1: Split paragraphs into semantic chunks"""
    print(f"\n{'='*60}")
    print("STAGE 1: PARAGRAPH SPLITTING")
    print(f"{'='*60}")
    print(f"Reading input file: {input_file}")

    df = read_dataframe(input_file)

    if sample:
        print(f"Sampling first {sample} rows")
        df = df.head(sample)

    print(f"Processing {len(df)} rows...")
    df_chunks = split_paragraphs(df, html_col=html_col)

    print(f"✓ Completed paragraph splitting")
    print(f"  - Added columns: neu_chunks, num_neu_chunks")
    return df_chunks


def run_inference(df, model_dir, text_col="neu_chunks", batch_size=32, max_length=128):
    """Stage 2: Classify chunks into job description sections"""
    print(f"\n{'='*60}")
    print("STAGE 2: INFERENCE/PREDICTION")
    print(f"{'='*60}")
    print(f"Loading model from: {model_dir}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    # Get label mapping from model config
    id2label = model.config.id2label

    print(f"Predicting sections for {len(df)} jobs...")
    df_predicted = infer_job_sections_from_df(
        df,
        model,
        tokenizer,
        id2label,
        text_col=text_col,
        batch_size=batch_size,
        max_length=max_length,
        col_suffix="_predicted",
        join_results=True  # Join results into strings
    )

    print(f"✓ Completed inference")
    print(f"  - Added predicted columns for each category")
    return df_predicted


def run_activity_extraction(df, backend="openai", model=None, sample=None):
    """Stage 3: Extract primary activities from company descriptions"""
    print(f"\n{'='*60}")
    print("STAGE 3: EXTRACT PRIMARY ACTIVITIES")
    print(f"{'='*60}")

    if sample:
        print(f"Processing sample of {sample} rows")
        df = df.head(sample)

    # Import and configure based on backend
    if backend == "openai":
        from Eugene_v2.src.extract_primary_activities.extraction import extract_with_openai
        from openai import OpenAI

        model_name = model or "gpt-4o-mini"
        print(f"Using OpenAI backend: {model_name}")

        # Check for API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        client = OpenAI(api_key=api_key)

        print(f"Extracting activities for {len(df)} jobs...")
        df = extract_with_openai(
            df,
            input_col="About the Company_predicted",
            output_col="primary_activities",
            client=client,
            model_name=model_name
        )

    elif backend == "ollama":
        from Eugene_v2.src.extract_primary_activities.extraction import extract_with_ollama

        model_name = model or "qwen2.5:7b-instruct"
        print(f"Using Ollama backend: {model_name}")
        print("Make sure Ollama is running: ollama serve")

        print(f"Extracting activities for {len(df)} jobs...")
        df = extract_with_ollama(
            df,
            input_col="About the Company_predicted",
            output_col="primary_activities",
            model_name=model_name
        )
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'openai' or 'ollama'")

    print(f"✓ Completed activity extraction")
    print(f"  - Added column: primary_activities")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 1: Inference Pipeline (Splitter -> Inference -> Activities)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m Eugene_v3.pipeline_inference \\
    --input_file data/jobs.xlsx \\
    --html_col jobDescription \\
    --output_file results/jobs_with_activities.xlsx \\
    --model_dir Eugene_v2/model_finetuned/base_min_5_with_other \\
    --sample 100

  python -m Eugene_v3.pipeline_inference \\
    --input_file data/jobs.csv \\
    --html_col raw_html \\
    --output_file results/output.csv \\
    --backend ollama \\
    --extraction_model qwen2.5:7b-instruct
        """
    )

    # Required arguments
    parser.add_argument("--input_file", required=True,
                        help="Input file path (.csv, .xlsx, .xls, .parquet)")
    parser.add_argument("--output_file", required=True,
                        help="Output file path")

    # Input configuration
    parser.add_argument("--html_col", default="jobDescription",
                        help="Column name containing job description HTML/text (default: jobDescription)")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N rows (for testing)")

    # Model configuration
    parser.add_argument("--model_dir", default="Eugene_v2/model_finetuned/base_min_5_with_other",
                        help="Path to fine-tuned JoBert model directory")
    parser.add_argument("--text_col", default="neu_chunks",
                        help="Column with chunks to classify (default: neu_chunks)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length for inference (default: 128)")

    # Activity extraction configuration
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                        help="Backend for activity extraction (default: openai)")
    parser.add_argument("--extraction_model", default=None,
                        help="Model for extraction (default: gpt-4o-mini for openai, qwen2.5:7b-instruct for ollama)")
    parser.add_argument("--extraction_sample", type=int, default=None,
                        help="Extract activities for only first N rows (useful for API cost control)")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    if not os.path.exists(args.model_dir):
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"\n{'#'*60}")
    print("INFERENCE PIPELINE")
    print(f"{'#'*60}")
    print(f"Input:  {args.input_file}")
    print(f"Output: {args.output_file}")
    print(f"Model:  {args.model_dir}")

    try:
        # Stage 1: Paragraph Splitting
        df = run_paragraph_splitting(args.input_file, args.html_col, args.sample)

        # Stage 2: Inference
        df = run_inference(df, args.model_dir, args.text_col, args.batch_size, args.max_length)

        # Stage 3: Extract Activities
        df = run_activity_extraction(df, args.backend, args.extraction_model, args.extraction_sample)

        # Save final output
        print(f"\n{'='*60}")
        print("SAVING RESULTS")
        print(f"{'='*60}")
        write_dataframe(df, args.output_file)
        print(f"✓ Saved results to: {args.output_file}")

        print(f"\n{'#'*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'#'*60}")
        print(f"Total rows processed: {len(df)}")
        print(f"Output columns: {', '.join(df.columns)}")

    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'!'*60}")
        raise


if __name__ == "__main__":
    main()
