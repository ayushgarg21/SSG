from .inference import infer_job_sections_from_df
from Eugene_v2.src.utils import read_dataframe, write_dataframe
from Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2 import df_to_paragraphs
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Eugene.ml_classify.jobert_utils import preprocess_line_breaks, split_into_sentences

OTHER_MODEL_DIR = "Eugene_v2/model_finetuned/base_min_5_with_other"
MODEL_DIR = "Eugene_v2/model_finetuned/base_min_5"
BASE_MODEL_DIR = "AhmedBou/JoBert"  # Original JoBert from Hugging Face

BATCH_SIZE = 32
MAX_LENGTH = 256
ID2LABEL = {
    0: "About the Company",
    1: "Job Description",
    2: "Job Requirements",
    3: "Responsibilities",
    4: "Benefits",
    5: "Other"
}

if __name__ == "__main__":
    # Load fine-tuned model and tokenizer
    print(f"Loading fine-tuned model from {MODEL_DIR}...")
    tokenizer_finetuned = AutoTokenizer.from_pretrained(MODEL_DIR)
    model_finetuned = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    print("Fine-tuned model loaded successfully!")

    # Load base JoBert model from Hugging Face
    print(f"\nLoading base JoBert model from {BASE_MODEL_DIR}...")
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)
    model_base = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_DIR)
    print("Base JoBert model loaded successfully!")

    print(f"\nLoading base OTHER_MODEL_DIR model from {OTHER_MODEL_DIR}...")
    tokenizer_other = AutoTokenizer.from_pretrained(OTHER_MODEL_DIR)
    model_other = AutoModelForSequenceClassification.from_pretrained(OTHER_MODEL_DIR)
    print("OTHER_MODEL_DIR odloaded successfully!")

    ## Read data
    df = read_dataframe("data/nov_25_jobs_final.parquet")
    df = df[['job_id', 'job_title', 'job_description', 'company_name', 'industry_id']]

    # Sample random rows for testing
    SAMPLE_SIZE =300  # Change this to desired sample size
    print(f"\nSampling {SAMPLE_SIZE} random jobs from {len(df)} total jobs...")
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=422)
    print(f"Sampled {len(df)} jobs")

    # Split job descriptions into paragraphs and sentences
    ## Paragraph
    df = df_to_paragraphs(df, html_col="job_description", output_col="splited_jd")
    # df[html_col] = df[job_description].progress_apply(preprocess_line_breaks)  # No need as preprocess line break done above
    
    df["sentences"] = df["job_description"].progress_apply(lambda x: split_into_sentences(x))
    
    
    # Run inference with fine-tuned model
    print("\nRunning inference with fine-tuned model...")
    results_df = infer_job_sections_from_df(
        df,
        model=model_finetuned,
        tokenizer=tokenizer_finetuned,
        model_label=ID2LABEL,
        text_col="splited_jd",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        col_suffix="_finetuned"
    )
    print("\nRunning inference with fine-tuned model with other...")
    results_df = infer_job_sections_from_df(
        results_df,
        model=model_other,
        tokenizer=tokenizer_other,
        model_label=ID2LABEL,
        text_col="splited_jd",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        col_suffix="_other"
    )


    # Run inference with base JoBert model
    print("\nRunning inference with base JoBert model...")
    results_df = infer_job_sections_from_df(
        results_df,
        model=model_base,
        tokenizer=tokenizer_base,
        model_label=ID2LABEL,
        text_col="splited_jd",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        col_suffix="_base"
    )

    # Run inference with fine-tuned model
    print("\n[SENTENCES]Running inference with fine-tuned model...")
    results_df = infer_job_sections_from_df(
        results_df,
        model=model_finetuned,
        tokenizer=tokenizer_finetuned,
        model_label=ID2LABEL,
        text_col="sentences",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        col_suffix="_finetuned_SENT"
    )
    print("\n[SENTENCES]Running inference with fine-tuned model with other...")
    results_df = infer_job_sections_from_df(
        results_df,
        model=model_other,
        tokenizer=tokenizer_other,
        model_label=ID2LABEL,
        text_col="sentences",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        col_suffix="_other_SENT"
    )


    # Run inference with base JoBert model
    print("\n[SENTENCES]Running inference with base JoBert model...")
    results_df = infer_job_sections_from_df(
        results_df,
        model=model_base,
        tokenizer=tokenizer_base,
        model_label=ID2LABEL,
        text_col="sentences",
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        col_suffix="_base_SENT"
    )

   

    result_name=f"results_data/compar_{SAMPLE_SIZE}.xlsx"
    write_dataframe(results_df,result_name)