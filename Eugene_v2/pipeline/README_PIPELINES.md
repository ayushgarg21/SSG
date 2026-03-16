# Eugene v3 - Pipeline Documentation

This folder contains two main pipeline scripts that chain together the job description processing workflow.

## Overview

### Pipeline 1: Inference Pipeline (`pipeline_inference.py`)
**Use when**: You already have a trained model and just want to process new data

**Stages**:
1. **Paragraph Splitting** - Split job descriptions into semantic chunks
2. **Inference/Prediction** - Classify chunks using pre-trained JoBert model
3. **Extract Primary Activities** - Extract company activities using LLM

**Speed**: Fast (no training involved)
**Cost**: Low (only activity extraction uses API)

---

### Pipeline 2: Full Training Pipeline (`pipeline_full.py`)
**Use when**: You need to train a new model or retrain with new labeled data

**Stages**:
1. **Paragraph Splitting** - Split job descriptions into semantic chunks
2. **LLM Labeling** - Label chunks with GPT-4o-mini
3. **Fine-tuning** - Train JoBert classifier on labeled data
4. **Inference/Prediction** - Classify chunks using newly trained model
5. **Extract Primary Activities** - Extract company activities using LLM

**Speed**: Slow (includes training)
**Cost**: High (LLM labeling + activity extraction + GPU time)

---

## Pipeline 1: Inference Pipeline

### Quick Start

```bash
# Basic usage - process jobs with existing model
python -m Eugene_v3.pipeline_inference \
  --input_file data/jobs.xlsx \
  --html_col jobDescription \
  --output_file results/jobs_with_activities.xlsx

# With sample for testing
python -m Eugene_v3.pipeline_inference \
  --input_file data/jobs.csv \
  --html_col raw_html \
  --output_file results/test_output.xlsx \
  --sample 100

# Using Ollama instead of OpenAI for activity extraction
python -m Eugene_v3.pipeline_inference \
  --input_file data/jobs.xlsx \
  --html_col jobDescription \
  --output_file results/output.xlsx \
  --backend ollama \
  --extraction_model qwen2.5:7b-instruct
```

### Parameters

#### Required
- `--input_file`: Input CSV/Excel file with job descriptions
- `--output_file`: Where to save final results

#### Input Configuration
- `--html_col`: Column containing job description text (default: `jobDescription`)
- `--sample`: Process only first N rows for testing

#### Model Configuration
- `--model_dir`: Path to trained model (default: `Eugene_v2/model_finetuned/base_min_5_with_other`)
- `--text_col`: Column with text chunks (default: `neu_chunks`)
- `--batch_size`: Inference batch size (default: 32)
- `--max_length`: Max tokens (default: 128)

#### Activity Extraction
- `--backend`: `openai` or `ollama` (default: `openai`)
- `--extraction_model`: Model name (default: `gpt-4o-mini` for OpenAI, `qwen2.5:7b-instruct` for Ollama)
- `--extraction_sample`: Extract for only first N rows (useful to control API costs)

### Prerequisites

**For OpenAI backend** (default):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**For Ollama backend**:
```bash
# Start Ollama server
ollama serve

# Pull model (first time only)
ollama pull qwen2.5:7b-instruct
```

### Output

The output file will contain all input columns plus:
- `neu_chunks`: List of semantic chunks (from Stage 1)
- `num_neu_chunks`: Number of chunks (from Stage 1)
- `About the Company_predicted`: Classified sentences about company (from Stage 2)
- `Job Description_predicted`: Classified job overview sentences (from Stage 2)
- `Job Requirements_predicted`: Classified requirement sentences (from Stage 2)
- `Responsibilities_predicted`: Classified responsibility sentences (from Stage 2)
- `Benefits_predicted`: Classified benefit sentences (from Stage 2)
- `Other_predicted`: Other classified sentences (from Stage 2)
- `primary_activities`: Extracted company activities (from Stage 3)

---

## Pipeline 2: Full Training Pipeline

### Quick Start

```bash
# Basic usage - train new model and process data
python -m Eugene_v3.pipeline_full \
  --input_file data/jobs.xlsx \
  --html_col jobDescription \
  --output_file results/jobs_with_activities.xlsx \
  --output_dir results/my_training_run

# Quick test with small sample
python -m Eugene_v3.pipeline_full \
  --input_file data/jobs.csv \
  --output_file results/test.xlsx \
  --output_dir results/test_run \
  --sample 100 \
  --training_sample 500 \
  --epochs 2

# Full training with custom parameters
python -m Eugene_v3.pipeline_full \
  --input_file data/jobs.xlsx \
  --output_file results/final_output.xlsx \
  --output_dir results/production_model \
  --epochs 5 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --backend ollama
```

### Parameters

#### Required
- `--input_file`: Input CSV/Excel file with job descriptions
- `--output_file`: Where to save final results
- `--output_dir`: Directory for intermediate files and trained model

#### Input Configuration
- `--html_col`: Column containing job description text (default: `jobDescription`)
- `--sample`: Process only first N rows

#### LLM Labeling Configuration
- `--labeling_model`: OpenAI model (default: `gpt-4o-mini`)
- `--labeling_batch_size`: Batch size for labeling (default: 5)

#### Fine-tuning Configuration
- `--base_model`: Base model to fine-tune (default: `AhmedBou/JoBert`)
- `--epochs`: Training epochs (default: 3)
- `--batch_size`: Training batch size (default: 8)
- `--learning_rate`: Learning rate (default: 2e-5)
- `--training_sample`: Train on sample of N labeled sentences (for quick testing)

#### Inference Configuration
- `--text_col`: Column with chunks (default: `neu_chunks`)
- `--inference_batch_size`: Batch size (default: 32)
- `--max_length`: Max tokens (default: 128)

#### Activity Extraction
- `--backend`: `openai` or `ollama` (default: `openai`)
- `--extraction_model`: Model name
- `--extraction_sample`: Extract for only first N rows

### Prerequisites

**Required**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Optional** (for Ollama):
```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

### Output Structure

The pipeline creates a timestamped run directory with:

```
results/my_training_run/
└── run_20260117_143022/
    ├── 01_chunks.xlsx              # After paragraph splitting
    ├── labeled_sentences.xlsx       # After LLM labeling
    ├── llm_labeling_checkpoint.pkl  # Resume checkpoint (auto-saved)
    ├── 02_inference.xlsx            # After inference
    ├── 03_final.xlsx                # Final output (copy)
    └── model/                       # Trained JoBert model
        ├── config.json
        ├── pytorch_model.bin
        ├── tokenizer_config.json
        └── ...
```

Plus the final output at your specified `--output_file` location.

### Resume from Checkpoint

If LLM labeling fails partway through, the pipeline automatically saves checkpoints. Simply re-run the same command and it will resume from the last checkpoint.

---

## Comparison Table

| Feature | Inference Pipeline | Full Training Pipeline |
|---------|-------------------|----------------------|
| **Speed** | Fast (~5 min for 1000 jobs*) | Slow (~2-3 hours for 1000 jobs*) |
| **API Cost** | Low (only activity extraction) | High (labeling + extraction) |
| **GPU Required** | Optional (faster with GPU) | Recommended for training |
| **When to Use** | Process new data with existing model | Need to train/retrain model |
| **Resumable** | No | Yes (LLM labeling stage) |
| **Output Model** | Uses existing model | Creates new fine-tuned model |

*Estimates assuming GPU and OpenAI API

---

## Common Use Cases

### Use Case 1: Process New Job Postings Weekly
```bash
# Use inference pipeline with existing model
python -m Eugene_v3.pipeline_inference \
  --input_file data/new_jobs_week_$(date +%Y%m%d).xlsx \
  --output_file results/processed_$(date +%Y%m%d).xlsx
```

### Use Case 2: Initial Model Training
```bash
# Use full pipeline to train first model
python -m Eugene_v3.pipeline_full \
  --input_file data/training_data.xlsx \
  --output_file results/initial_training.xlsx \
  --output_dir models/v1 \
  --epochs 5
```

### Use Case 3: Quick Testing on Sample
```bash
# Test on 50 jobs with fast inference
python -m Eugene_v3.pipeline_inference \
  --input_file data/jobs.xlsx \
  --output_file results/test.xlsx \
  --sample 50 \
  --backend ollama \
  --extraction_sample 10
```

### Use Case 4: Retrain with More Data
```bash
# Retrain when you have new labeled data
python -m Eugene_v3.pipeline_full \
  --input_file data/all_jobs_combined.xlsx \
  --output_file results/retrained.xlsx \
  --output_dir models/v2 \
  --epochs 4
```

---

## Troubleshooting

### Error: OPENAI_API_KEY not set
```bash
export OPENAI_API_KEY="sk-..."
```

### Error: Model directory not found
Make sure you've either:
1. Run the full training pipeline first, OR
2. Specify correct path to existing model:
```bash
--model_dir path/to/your/model
```

### Ollama connection error
```bash
# Start Ollama server first
ollama serve

# In another terminal, check it's running
ollama list
```

### Out of memory during training
Reduce batch size:
```bash
--batch_size 4 --inference_batch_size 16
```

### LLM labeling is expensive
Use smaller sample first:
```bash
--sample 500 --training_sample 500
```

---

## Tips for Best Results

1. **Start Small**: Always test with `--sample 100` first
2. **Cost Control**: Use `--extraction_sample` to limit API calls during testing
3. **GPU Recommended**: Training and inference are much faster with GPU
4. **Checkpoints**: Full pipeline auto-saves checkpoints every 10 jobs during labeling
5. **Model Reuse**: Save your trained models and use inference pipeline for new data
6. **Ollama for Cost**: Use Ollama backend for activity extraction to avoid API costs

---

## File Formats Supported

Both pipelines support:
- `.csv` - Comma-separated values
- `.xlsx` - Excel (modern)
- `.xls` - Excel (legacy)
- `.parquet` / `.pq` - Parquet (efficient for large datasets)

Input and output formats are auto-detected from file extension.

---

## Dependencies

Make sure you have installed:
```bash
pip install pandas openpyxl torch transformers datasets scikit-learn openai
```

For Ollama support:
```bash
# Install ollama from https://ollama.ai
# Then pull models:
ollama pull qwen2.5:7b-instruct
```

---

## Next Steps

1. **Test inference pipeline** on a small sample (100 rows)
2. **Review outputs** to ensure quality
3. **Run full pipeline** if you need to train a custom model
4. **Use inference pipeline** for ongoing data processing

For questions or issues, check the main Eugene_v2 README or raise an issue.
