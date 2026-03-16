"""
Pipeline 2: Full Training Pipeline
Chains: Paragraph Splitter -> LLM Labeling -> Fine-tune -> Inference -> Extract Activities

This pipeline includes the full training loop, useful when you need to retrain the model.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

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


def run_llm_labeling(df, output_dir, model="gpt-4o-mini", batch_size=5, text_col="neu_chunks"):
    """Stage 2: Label chunks with GPT-4o-mini"""
    print(f"\n{'='*60}")
    print("STAGE 2: LLM LABELING")
    print(f"{'='*60}")

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    print(f"Using model: {model}")
    print(f"Batch size: {batch_size}")

    # Import necessary functions from llm_labeling
    from openai import OpenAI
    import json
    import time
    import ast

    client = OpenAI()

    # Label categories
    CATEGORIES = {
        "0": "About the Company",
        "1": "Job Description",
        "2": "Job Requirements",
        "3": "Responsibilities",
        "4": "Benefits",
        "5": "Other"
    }

    SYSTEM_PROMPT = """You are an expert at categorizing job description sentences.

Given a list of sentences from a job description, classify each sentence into ONE of these categories:
- About the Company (0): Information about the company, its mission, values, culture, size, industry
- Job Description (1): General overview of the job role, position summary
- Job Requirements (2): Required qualifications, skills, education, experience
- Responsibilities (3): What the person will do in this role, day-to-day tasks
- Benefits (4): Compensation, benefits, perks offered
- Other (5): Anything that doesn't fit the above categories

Return a JSON array where each object has "sentence" and "label" (0-5).

Example:
[
  {"sentence": "We are a leading tech company.", "label": "0"},
  {"sentence": "Must have 5 years of Python experience.", "label": "2"}
]"""

    def label_sentences_batch(sentences, model_name, max_retries=3):
        """Call OpenAI to label a batch of sentences"""
        user_prompt = f"Classify these sentences:\n\n{json.dumps(sentences, indent=2)}"

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"}
                )

                content = response.choices[0].message.content
                result = json.loads(content)

                # Handle both array and object with "result" key
                if isinstance(result, list):
                    return result
                elif "result" in result:
                    return result["result"]
                elif "labels" in result:
                    return result["labels"]
                else:
                    # Assume first key contains the array
                    first_key = list(result.keys())[0]
                    return result[first_key]

            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise

        raise Exception("Max retries exceeded")

    # Process jobs
    labeled_data = []
    checkpoint_file = os.path.join(output_dir, "llm_labeling_checkpoint.pkl")

    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from: {checkpoint_file}")
        checkpoint = pd.read_pickle(checkpoint_file)
        labeled_data = checkpoint["labeled_data"]
        start_idx = checkpoint["last_processed_idx"] + 1
        print(f"Resuming from job {start_idx}")
    else:
        start_idx = 0

    # Ensure we have a job ID column
    if "jobId" not in df.columns and "job_id" not in df.columns:
        df["job_id"] = df.index

    job_id_col = "jobId" if "jobId" in df.columns else "job_id"

    print(f"Processing {len(df) - start_idx} jobs...")

    for idx, row in df.iloc[start_idx:].iterrows():
        job_id = row[job_id_col]
        chunks_str = row[text_col]

        # Parse chunks (they're stored as string representation of list)
        try:
            if isinstance(chunks_str, str):
                chunks = ast.literal_eval(chunks_str)
            else:
                chunks = chunks_str
        except:
            print(f"  Warning: Could not parse chunks for job {job_id}, skipping")
            continue

        if not chunks or len(chunks) == 0:
            continue

        # Label in batches
        print(f"  Labeling job {job_id} ({len(chunks)} sentences)...")
        batch_results = label_sentences_batch(chunks, model)

        # Store results
        for result in batch_results:
            labeled_data.append({
                "job_id": job_id,
                "sentence": result["sentence"],
                "label": CATEGORIES[str(result["label"])]
            })

        # Save checkpoint every 10 jobs
        if (idx + 1) % 10 == 0:
            checkpoint = {
                "labeled_data": labeled_data,
                "last_processed_idx": idx
            }
            pd.to_pickle(checkpoint, checkpoint_file)
            print(f"  Checkpoint saved at job {idx + 1}")

        # Rate limiting
        time.sleep(1)

    # Convert to DataFrame
    labeled_df = pd.DataFrame(labeled_data)

    # Save final labeled data
    labeled_file = os.path.join(output_dir, "labeled_sentences.xlsx")
    write_dataframe(labeled_df, labeled_file)

    print(f"✓ Completed LLM labeling")
    print(f"  - Total sentences labeled: {len(labeled_df)}")
    print(f"  - Saved to: {labeled_file}")

    return labeled_df


def run_finetuning(labeled_df, output_model_dir, base_model="AhmedBou/JoBert", epochs=3, batch_size=8, learning_rate=2e-5, sample=None):
    """Stage 3: Fine-tune JoBert on labeled data"""
    print(f"\n{'='*60}")
    print("STAGE 3: FINE-TUNING JOBERT")
    print(f"{'='*60}")

    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import numpy as np

    print(f"Base model: {base_model}")
    print(f"Output directory: {output_model_dir}")

    # Sample if requested
    if sample:
        print(f"Training on sample of {sample} sentences")
        labeled_df = labeled_df.sample(n=min(sample, len(labeled_df)), random_state=42)

    # Prepare data
    label2id = {
        "About the Company": 0,
        "Job Description": 1,
        "Job Requirements": 2,
        "Responsibilities": 3,
        "Benefits": 4,
        "Other": 5
    }
    id2label = {v: k for k, v in label2id.items()}

    labeled_df["label_id"] = labeled_df["label"].map(label2id)

    # Train/test split
    train_df, test_df = train_test_split(labeled_df, test_size=0.15, random_state=42, stratify=labeled_df["label_id"])
    print(f"Train set: {len(train_df)} sentences")
    print(f"Test set: {len(test_df)} sentences")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    # Create datasets
    from datasets import Dataset

    train_dataset = Dataset.from_pandas(train_df[["sentence", "label_id"]])
    test_dataset = Dataset.from_pandas(test_df[["sentence", "label_id"]])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset = train_dataset.rename_column("label_id", "labels")
    test_dataset = test_dataset.rename_column("label_id", "labels")

    # Compute metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_model_dir}/logs",
        logging_steps=100,
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print(f"Starting training for {epochs} epochs...")
    trainer.train()

    # Evaluate
    print("\nEvaluating model...")
    eval_results = trainer.evaluate()

    print(f"✓ Completed fine-tuning")
    print(f"  - Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"  - F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"  - Precision: {eval_results['eval_precision']:.4f}")
    print(f"  - Recall: {eval_results['eval_recall']:.4f}")
    print(f"  - Model saved to: {output_model_dir}")

    return output_model_dir


def run_inference(df, model_dir, text_col="neu_chunks", batch_size=32, max_length=128):
    """Stage 4: Classify chunks into job description sections"""
    print(f"\n{'='*60}")
    print("STAGE 4: INFERENCE/PREDICTION")
    print(f"{'='*60}")
    print(f"Loading model from: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

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
        join_results=True
    )

    print(f"✓ Completed inference")
    return df_predicted


def run_activity_extraction(df, backend="openai", model=None, sample=None):
    """Stage 5: Extract primary activities from company descriptions"""
    print(f"\n{'='*60}")
    print("STAGE 5: EXTRACT PRIMARY ACTIVITIES")
    print(f"{'='*60}")

    if sample:
        print(f"Processing sample of {sample} rows")
        df = df.head(sample)

    if backend == "openai":
        from Eugene_v2.src.extract_primary_activities.extraction import extract_with_openai
        from openai import OpenAI

        model_name = model or "gpt-4o-mini"
        print(f"Using OpenAI backend: {model_name}")

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
        raise ValueError(f"Invalid backend: {backend}")

    print(f"✓ Completed activity extraction")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline 2: Full Training Pipeline (Splitter -> Labeling -> Finetune -> Inference -> Activities)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python -m Eugene_v3.pipeline_full \\
    --input_file data/jobs.xlsx \\
    --html_col jobDescription \\
    --output_file results/jobs_with_activities.xlsx \\
    --output_dir results/training_run_1 \\
    --sample 500

  # With custom training parameters
  python -m Eugene_v3.pipeline_full \\
    --input_file data/jobs.csv \\
    --output_file results/output.csv \\
    --output_dir results/my_model \\
    --epochs 5 \\
    --batch_size 16 \\
    --backend ollama
        """
    )

    # Required arguments
    parser.add_argument("--input_file", required=True,
                        help="Input file path (.csv, .xlsx, .xls, .parquet)")
    parser.add_argument("--output_file", required=True,
                        help="Final output file path")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for intermediate files and trained model")

    # Input configuration
    parser.add_argument("--html_col", default="jobDescription",
                        help="Column name containing job description HTML/text")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N rows (for testing)")

    # LLM Labeling configuration
    parser.add_argument("--labeling_model", default="gpt-4o-mini",
                        help="OpenAI model for labeling (default: gpt-4o-mini)")
    parser.add_argument("--labeling_batch_size", type=int, default=5,
                        help="Batch size for LLM labeling (default: 5)")

    # Fine-tuning configuration
    parser.add_argument("--base_model", default="AhmedBou/JoBert",
                        help="Base model to fine-tune (default: AhmedBou/JoBert)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--training_sample", type=int, default=None,
                        help="Train on sample of N labeled sentences (for quick testing)")

    # Inference configuration
    parser.add_argument("--text_col", default="neu_chunks",
                        help="Column with chunks to classify (default: neu_chunks)")
    parser.add_argument("--inference_batch_size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Max token length for inference (default: 128)")

    # Activity extraction configuration
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai",
                        help="Backend for activity extraction (default: openai)")
    parser.add_argument("--extraction_model", default=None,
                        help="Model for extraction")
    parser.add_argument("--extraction_sample", type=int, default=None,
                        help="Extract activities for only first N rows")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    model_dir = os.path.join(run_dir, "model")

    print(f"\n{'#'*60}")
    print("FULL TRAINING PIPELINE")
    print(f"{'#'*60}")
    print(f"Input:      {args.input_file}")
    print(f"Output:     {args.output_file}")
    print(f"Run dir:    {run_dir}")
    print(f"Model dir:  {model_dir}")

    try:
        # Stage 1: Paragraph Splitting
        df = run_paragraph_splitting(args.input_file, args.html_col, args.sample)

        # Save intermediate result
        chunks_file = os.path.join(run_dir, "01_chunks.xlsx")
        write_dataframe(df, chunks_file)
        print(f"  Saved intermediate: {chunks_file}")

        # Stage 2: LLM Labeling
        labeled_df = run_llm_labeling(df, run_dir, args.labeling_model, args.labeling_batch_size, args.text_col)

        # Stage 3: Fine-tuning
        model_dir = run_finetuning(
            labeled_df, model_dir, args.base_model,
            args.epochs, args.batch_size, args.learning_rate, args.training_sample
        )

        # Stage 4: Inference
        df = run_inference(df, model_dir, args.text_col, args.inference_batch_size, args.max_length)

        # Save intermediate result
        inference_file = os.path.join(run_dir, "02_inference.xlsx")
        write_dataframe(df, inference_file)
        print(f"  Saved intermediate: {inference_file}")

        # Stage 5: Extract Activities
        df = run_activity_extraction(df, args.backend, args.extraction_model, args.extraction_sample)

        # Save final output
        print(f"\n{'='*60}")
        print("SAVING FINAL RESULTS")
        print(f"{'='*60}")
        write_dataframe(df, args.output_file)
        print(f"✓ Saved results to: {args.output_file}")

        # Also save to run directory
        final_file = os.path.join(run_dir, "03_final.xlsx")
        write_dataframe(df, final_file)
        print(f"✓ Saved copy to: {final_file}")

        print(f"\n{'#'*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'#'*60}")
        print(f"Total rows processed: {len(df)}")
        print(f"Output columns: {', '.join(df.columns)}")
        print(f"\nAll outputs saved in: {run_dir}")

    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"ERROR: {str(e)}")
        print(f"{'!'*60}")
        raise


if __name__ == "__main__":
    main()
