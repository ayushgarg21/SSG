    python paragraph_splitter.py --input_file data.parquet --html_col --output_file result.xlsx


v2
python -m Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2 --input_file 'data/nov_25_jobs_final.parquet' --html_col job_description --output_file 'Eugene/results_data/nov_25_jobs_final_break_paragraph_v2.xlsx' --sample 1000

python -m Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2 --input_file 'data/Sample-extractions_20251127.xlsx' --html_col jobDescription --output_file 'Eugene/results_data/Sample-extractions_20251127.xlsx' --sample 1000



## User large bert model, reduced minium character to 1 - not using custom tokenizer, tokenzier should macth the model

python -m Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2 --input_file 'data/Sample-extractions_20251127.xlsx' --html_col jobDescription --output_file 'Eugene/results_data/Sample-extractions_20251127_with_tokenizer.xlsx' --sample 1000



python -m Eugene_v2.src.model_label_and_finetune.llm_labeling 

python -m Eugene_v2.src.model_label_and_finetune.finetune_jobert 

 python -m Eugene_v2.src.inference_prediction.main
  python -m Eugene_v2.src.inference_prediction.compare


python -m Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2_slum

python -m Eugene_v2.src.paragraph_splitter.paragraph_splitter_v2_slum --input_file 'data/Sample-extractions_20251127.xlsx' --html_col jobDescription --output_file 'results_data/slum_vs_neu_min5.xlsx' --sample 5

