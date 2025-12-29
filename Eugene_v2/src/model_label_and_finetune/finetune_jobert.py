"""
Fine-tune JoBert on LLM-labeled job description sentences.
This script trains the model to classify sentences into 5 categories:
- About the Company
- Job Description
- Job Requirements
- Responsibilities
- Benefits
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import warnings
warnings.filterwarnings('ignore')
from Eugene_v2.src.utils import read_dataframe, write_dataframe

# Configuration
INPUT_FILE = "results_data/Sample-extractions_20251127_with_base_min_5_labeled_sentences.xlsx"
OUTPUT_DIR = "Eugene_v2/model_finetuned/base_min_5_with_other"
BASE_MODEL = "AhmedBou/JoBert"

# Label mapping (excluding "Other" since we want to filter those out)
LABELS = [
    "About the Company",
    "Job Description",
    "Job Requirements",
    "Responsibilities",
    "Benefits",
    "Other" ## New
]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Training hyperparameters
EPOCHS = 3
BATCH_SIZE = 8  # Small batch size for MPS memory
LEARNING_RATE = 2e-5
MAX_LENGTH = 128
NUM_WORKERS = 0  # Disable multiprocessing to avoid tokenizer fork issues

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class JobDescriptionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for the model.

    Metrics explanation:
    - Accuracy: Overall correctness (correct predictions / total predictions)
    - Precision: Of all predicted positives, how many were actually positive (TP / (TP + FP))
      → High precision = few false alarms
    - Recall: Of all actual positives, how many did we find (TP / (TP + FN))
      → High recall = we catch most of the true cases
    - F1-Score: Harmonic mean of precision and recall (2 * (precision * recall) / (precision + recall))
      → Balances precision and recall into a single metric

    We use 'macro' averaging: calculate metrics for each class independently, then average.
    This treats all classes equally regardless of their size.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = (predictions == labels).mean()

    # Calculate precision, recall, F1 (macro-averaged across all classes)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',  # Treat all classes equally
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune JoBert on labeled job description sentences")
    parser.add_argument("--sample", type=int, help="Train on a sample of N sentences (for testing)")
    args = parser.parse_args()

    print("=" * 80)
    print("JOBERT FINE-TUNING FOR JOB DESCRIPTION CLASSIFICATION")
    print("=" * 80)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load labeled data
    print(f"\nLoading labeled data from {INPUT_FILE}...")
    # df = pd.read_parquet(INPUT_FILE)
    df=read_dataframe(INPUT_FILE)
    print(f"Total sentences: {len(df):,}")

    # Filter out "Other" category - we only want the 5 main categories
    # df_filtered = df[df['label'] != 'Other'].copy()
    # print(f"After removing 'Other': {len(df_filtered):,} sentences")
    df_filtered = df.copy()
    # Sample if requested (for quick testing)
    if args.sample:
        print(f"\n⚠️  SAMPLING MODE: Using only {args.sample:,} sentences for quick testing")
        df_filtered = df_filtered.sample(n=min(args.sample, len(df_filtered)), random_state=42)
        print(f"Sampled: {len(df_filtered):,} sentences")

    # Map labels to IDs
    df_filtered['label_id'] = df_filtered['label'].map(LABEL2ID)

    # Check distribution
    print("\nLabel distribution:")
    print(df_filtered['label'].value_counts())

    # Split data
    train_df, test_df = train_test_split(
        df_filtered,
        test_size=0.15,
        random_state=42,
        stratify=df_filtered['label_id']
    )

    print(f"\nTraining samples: {len(train_df):,}")
    print(f"Test samples: {len(test_df):,}")

    # Load tokenizer and model
    print(f"\nLoading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True  # Since we're changing the classification head
    )

    # Create datasets
    print("\nPreparing datasets...")
    train_dataset = JobDescriptionDataset(
        train_df['sentence'].tolist(),
        train_df['label_id'].tolist(),
        tokenizer,
        MAX_LENGTH
    )

    test_dataset = JobDescriptionDataset(
        test_df['sentence'].tolist(),
        test_df['label_id'].tolist(),
        tokenizer,
        MAX_LENGTH
    )

    # Training arguments - optimized for MPS memory constraints
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{OUTPUT_DIR}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1 as the best metric (balances precision/recall)
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=NUM_WORKERS,
        use_mps_device=torch.backends.mps.is_available(),
        optim="adamw_torch",
        gradient_accumulation_steps=4,  # Effective batch size = 32
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING...")
    print("=" * 80)
    trainer.train()

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    # Overall metrics
    print("\nOverall Metrics:")
    print("-" * 80)
    test_metrics = predictions.metrics
    print(f"  Accuracy:  {test_metrics['test_accuracy']:.4f}")
    print(f"  Precision: {test_metrics['test_precision']:.4f}")
    print(f"  Recall:    {test_metrics['test_recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['test_f1']:.4f}")
    print()
    print("Metric Explanations:")
    print("  • Accuracy:  % of correct predictions overall")
    print("  • Precision: Of predictions for each class, % that were correct (macro-avg)")
    print("  • Recall:    Of actual examples in each class, % we correctly identified (macro-avg)")
    print("  • F1-Score:  Harmonic mean of precision & recall (macro-avg)")

    # Classification report (per-class metrics)
    print("\n" + "-" * 80)
    print("Per-Class Metrics (Precision, Recall, F1 for each category):")
    print("-" * 80)
    print(classification_report(
        test_df['label_id'].tolist(),
        preds,
        target_names=LABELS
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("(Rows = True labels, Columns = Predicted labels)")
    print("-" * 80)
    cm = confusion_matrix(test_df['label_id'].tolist(), preds)
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS))

    # Save the model
    print(f"\nSaving fine-tuned model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nTo use the fine-tuned model:")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")


if __name__ == "__main__":
    main()
