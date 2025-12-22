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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = "/Users/ayush/Desktop/ssg_eugene/EUGENE/labeled_sentences.parquet"
OUTPUT_DIR = "/Users/ayush/Desktop/ssg_eugene/EUGENE/jobert_finetuned"
BASE_MODEL = "AhmedBou/JoBert"

# Label mapping (excluding "Other" since we want to filter those out)
LABELS = [
    "About the Company",
    "Job Description",
    "Job Requirements",
    "Responsibilities",
    "Benefits"
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
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate accuracy
    accuracy = (predictions == labels).mean()

    return {'accuracy': accuracy}


def main():
    print("=" * 80)
    print("JOBERT FINE-TUNING FOR JOB DESCRIPTION CLASSIFICATION")
    print("=" * 80)

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load labeled data
    print(f"\nLoading labeled data from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"Total sentences: {len(df):,}")

    # Filter out "Other" category - we only want the 5 main categories
    df_filtered = df[df['label'] != 'Other'].copy()
    print(f"After removing 'Other': {len(df_filtered):,} sentences")

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
        metric_for_best_model="accuracy",
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

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_df['label_id'].tolist(),
        preds,
        target_names=LABELS
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
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
