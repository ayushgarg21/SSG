"""Utility functions for data I/O operations"""
import pandas as pd
from pathlib import Path


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


""" From v1 clean"""
import re
from bs4 import BeautifulSoup
import pysbd
from sentence_transformers import SentenceTransformer, util
import torch

# --------------------------
# DEVICE & MODEL SETUP
# --------------------------
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

_seg = pysbd.Segmenter(language="en", clean=True)
_semantic_model = None

def get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
    return _semantic_model

def split_sentences(text):
    if not text:
        return []
    return _seg.segment(text)

# --------------------------
# HTML CLEANING
# --------------------------
def html_to_text(html: str) -> str:
    if not isinstance(html, str):
        return ""
    # Normalize <br> tags
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()

# --------------------------
# HEADER DETECTION
# --------------------------
HEADER_PATTERNS = [
    re.compile(r"^\s*(Position|Key|Responsibilities|Summary|About|Qualifications|Requirements)[\w\s]*[:]?$",
               re.IGNORECASE),
]

BULLET_MARKER = re.compile(r"^\s*[-*•\d\)]\s+")

def is_header(line: str) -> bool:
    line = line.strip()
    return any(p.search(line) for p in HEADER_PATTERNS)

# --------------------------
# 2-WAY METHOD (OLD APPROACH)
# --------------------------
def group_paragraphs_from_text(text: str, min_sentences=1):
    """
    Returns a list of paragraph texts. Heuristics:
    - Split on double newlines
    - If no double newlines, split by groups of sentences, merging based on header detection and bullets
    """
    if not text:
        return []

    # Primary fast split: double newline separates paragraphs
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    paragraphs = []

    # If blocks already look like paragraphs, return them after cleanup
    if len(blocks) > 1:
        for b in blocks:
            # rejoin lines within blocks if they are broken by single newlines
            lines = [ln.strip() for ln in b.splitlines() if ln.strip()]
            para = " ".join(lines)
            paragraphs.append(para)
        return paragraphs

    # Fallback: sentence segmentation + heuristic grouping
    sents = split_sentences(text)
    cur = []
    for s in sents:
        # If sentence looks like a header, flush current
        if any(p.search(s.strip()) for p in HEADER_PATTERNS):
            if cur:
                paragraphs.append(" ".join(cur).strip())
                cur = []
            paragraphs.append(s.strip())
            continue

        # Bullet starts -> start new paragraph
        if BULLET_MARKER.search(s.strip()):
            if cur:
                paragraphs.append(" ".join(cur).strip())
                cur = []

        cur.append(s)

    # Flush remaining
    if cur:
        paragraphs.append(" ".join(cur).strip())

    # Filter out very short paragraphs
    filtered = [p for p in paragraphs if len(p.split()) >= min_sentences]
    return filtered

# --------------------------
# HEURISTIC BLOCK SPLIT
# --------------------------
def split_into_blocks(text: str):
    """
    Split text into alternating header + text blocks.
    If no header is detected, all text is one block.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    blocks = []
    current_header = None
    current_text = []

    for line in lines:
        if is_header(line):
            # Save previous block
            if current_header or current_text:
                blocks.append(current_header or "")
                blocks.append(" ".join(current_text).strip())
                current_text = []

            current_header = line
        else:
            current_text.append(line)

    # Flush last block
    if current_header or current_text:
        blocks.append(current_header or "")
        blocks.append(" ".join(current_text).strip())

    # Remove empty blocks
    blocks = [b for b in blocks if b.strip()]
    return blocks

# --------------------------
# SEMANTIC MERGE OF HEADER + TEXT
# --------------------------
def merge_header_with_text_semantic(blocks, similarity_threshold=0.55):
    """
    Merge each header + its text block into cohesive paragraphs using semantic similarity.
    """
    model = get_semantic_model()
    merged_paragraphs = []

    i = 0
    while i < len(blocks):
        header = blocks[i].strip()
        text_block = blocks[i + 1].strip() if i + 1 < len(blocks) else ""

        sentences = split_sentences(text_block)
        if not sentences:
            merged_paragraphs.append(header)
            i += 2
            continue

        # Merge header with first sentence
        current_para = [header + " " + sentences[0]]

        if len(sentences) > 1:
            embeddings = model.encode([current_para[0]] + sentences[1:], convert_to_tensor=True, device=DEVICE)
            para_sents = [sentences[0]]

            for idx in range(1, len(sentences)):
                sim = util.cos_sim(embeddings[idx], embeddings[idx - 1]).item()
                if sim >= similarity_threshold:
                    para_sents.append(sentences[idx])
                else:
                    merged_paragraphs.append(header + " " + " ".join(para_sents))
                    para_sents = [sentences[idx]]

            if para_sents:
                merged_paragraphs.append(header + " " + " ".join(para_sents))
        else:
            merged_paragraphs.append(current_para[0])

        i += 2

    return merged_paragraphs

# --------------------------
# SEMANTIC SENTENCE GROUPING
# --------------------------
def semantic_merge_sentences(text, similarity_threshold=0.55):
    """
    Take all text, split into sentences, merge semantically into paragraphs.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    model = get_semantic_model()
    embeddings = model.encode(sentences, convert_to_tensor=True, device=DEVICE)

    paragraphs = []
    current_para = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()

        if sim >= similarity_threshold:
            # Similar to previous - add to current paragraph
            current_para.append(sentences[i])
        else:
            # Topic shift - flush current paragraph and start new one
            paragraphs.append(" ".join(current_para))
            current_para = [sentences[i]]

    # Flush remaining
    if current_para:
        paragraphs.append(" ".join(current_para))

    return paragraphs

# --------------------------
# FULL PIPELINE FUNCTION
# --------------------------
def extract_job_paragraphs(html_text, similarity_threshold=0.55):
    """
    Input: raw HTML job description
    Output: list of clean paragraphs (semantically grouped sentences)
    """
    text_clean = html_to_text(html_text)
    paragraphs = semantic_merge_sentences(text_clean, similarity_threshold=similarity_threshold)
    return paragraphs
