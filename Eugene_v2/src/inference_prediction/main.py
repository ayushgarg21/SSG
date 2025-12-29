from .inference import infer_job_sections_from_df
from Eugene_v2.src.utils import read_dataframe, write_dataframe

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "Eugene_v2/model_finetuned/base_min_5_with_other"

ID2LABEL = {
0: "About the Company",
1: "Job Description",
2: "Job Requirements",
3: "Responsibilities",
4: "Benefits",
5: "Other"
}

tokenizer_other = AutoTokenizer.from_pretrained(MODEL_DIR)
model_other = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
if __name__ == "__main__":
    # df = read_dataframe("results_data/Sample-extractions_20251127_with_base_min_5.xlsx")
    # results_df= infer_job_sections_from_df(df,MODEL_DIR,ID2LABEL,text_col="neu_chunks",batch_size=16,max_length=256)
    # write_dataframe(results_df,"results_data/SE_with_base_min_5_inferenced.xlsx")
    df = read_dataframe("results_data/Sample-extractions_20251127_with_base_min_5.xlsx")
    
    print(df.columns)
    results_df= infer_job_sections_from_df(df,model_other,tokenizer_other,ID2LABEL,text_col="neu_chunks",batch_size=16,max_length=256,join_results=True)
    write_dataframe(results_df,"results_data/SE_with_base_min_5_others_inferenced_BASELINE.xlsx")