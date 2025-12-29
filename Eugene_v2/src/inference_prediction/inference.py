

import ast
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from Eugene_v2.src.utils import read_dataframe, write_dataframe
from tqdm.auto import tqdm

def infer_job_sections_from_df(
    df: pd.DataFrame,
    model,
    tokenizer,
    model_label:dict,
    text_col: str,
    batch_size: int = 32,
    max_length: int = 128,
    col_suffix:str="_predicted",
    join_results=False

) -> pd.DataFrame:
    """
    Run inference on a dataframe of job descriptions.

    Args:
        df: input dataframe
        model: Pre-loaded transformers model (AutoModelForSequenceClassification)
        tokenizer: Pre-loaded transformers tokenizer (AutoTokenizer)
        model_label: dict mapping label IDs to category names (e.g., {0: "Job Description", 1: "Benefits", ...})
        text_col: column containing stringified list of sentences
        batch_size: inference batch size
        max_length: tokenizer max length
        col_suffix: suffix to add to output column names (default: "_predicted")
        join_results: if true, we join the list of str to a single str
    Returns:
        DataFrame with original columns + one column per category with suffix
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    model.eval()

    # --- Step 1: Parse list column safely ---
    df = df.copy()
    df = df.reset_index(drop=True)  # Reset index to ensure 0-based indexing
    df["_sentences"] = df[text_col].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else (x if isinstance(x, list) else [])
    )

    # --- Step 2: Prepare output columns ---
    for cat in model_label.values():
        col_name = f"{cat}{col_suffix}" if col_suffix else cat
        df[col_name] = [[] for _ in range(len(df))]

    # --- Step 3: Flatten all sentences with row index ---
    all_sentences = []
    row_mapping = []  # keeps track of which row each sentence came from

    for row_idx, sentences in enumerate(df["_sentences"]):
        for sent in sentences:
            all_sentences.append(str(sent))
            row_mapping.append(row_idx)

    if not all_sentences:
        df.drop(columns=["_sentences"], inplace=True)
        return df

    # --- Step 4: Batch inference ---
    for i in tqdm(
        range(0, len(all_sentences), batch_size),
        desc="Running JoBert inference",
        total=(len(all_sentences) + batch_size - 1) // batch_size
    ):
        batch_sents = all_sentences[i:i + batch_size]
        batch_rows = row_mapping[i:i + batch_size]

        inputs = tokenizer(
            batch_sents,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

        # --- Step 5: Assign predictions back to rows ---
        for sent, row_idx, pred_id in zip(batch_sents, batch_rows, preds):
            label = model_label[pred_id]
            col_name = f"{label}{col_suffix}" if col_suffix else label
            df.at[row_idx, col_name].append(sent)

    # --- Step 6: Cleanup ---
    df.drop(columns=["_sentences"], inplace=True)

    # --- Step 7: Join results if requested ---
    if join_results:
        # Get all prediction columns (those with the suffix)
        pred_cols = [col for col in df.columns if col.endswith(col_suffix)]
        for col in pred_cols:
            # Strip each item (including trailing periods) and join with newline
            df[col] = df[col].apply(lambda x: "\n".join([str(item).strip().lstrip('. ') for item in x]) if isinstance(x, list) else "")

    return df