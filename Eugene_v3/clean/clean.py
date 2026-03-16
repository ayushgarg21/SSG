import re
import emoji
import html
from utils import read_dataframe, write_dataframe
from utils import html_to_text, split_sentences, group_paragraphs_from_text, extract_job_paragraphs

import spacy
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
df=read_dataframe("data/Sample-extractions_v2.xlsx")
# ============================================================
# Job Description Cleaning — Stage-by-Stage (Single Script)
# ============================================================


# --------------------------
# Setup
# ----------------------------
nlp = spacy.load("en_core_web_md", disable=["parser", "lemmatizer"])


def df_to_paragraphs(df: pd.DataFrame, html_col: str = "raw_html"):
    """Adds multiple columns for comparison (only generates missing columns):
    - paragraphs_2way: 2-way method (double newlines + pysbd with heuristics)
    - paragraphs_sentences: Just pysbd sentence segmentation
    - paragraphs_semantic_0.3: Semantic merge (threshold 0.3 - loose)
    - paragraphs_semantic_0.5: Semantic merge (threshold 0.5 - moderate)
    - paragraphs_semantic_0.75: Semantic merge (threshold 0.75 - strict)
    - paragraphs_semantic_0.9: Semantic merge (threshold 0.9 - very strict)
    """
    # Check which columns already exist
    need_2way = "paragraphs_2way" not in df.columns
    need_sentences = "paragraphs_sentences" not in df.columns
    need_sem_03 = "paragraphs_semantic_0.3" not in df.columns
    need_sem_05 = "paragraphs_semantic_0.5" not in df.columns
    need_sem_075 = "paragraphs_semantic_0.75" not in df.columns
    need_sem_09 = "paragraphs_semantic_0.9" not in df.columns

    # If all columns exist, nothing to do
    if not (need_2way or need_sentences or need_sem_03 or need_sem_05 or need_sem_075 or need_sem_09):
        print("All paragraph columns already exist. Skipping processing.")
        return df

    # Initialize lists for all columns
    paragraphs_2way = []
    paragraphs_sentences = []
    paragraphs_sem_03 = []
    paragraphs_sem_05 = []
    paragraphs_sem_075 = []
    paragraphs_sem_09 = []

    # Generate only missing columns
    for html in tqdm(df[html_col].fillna("").astype(str), desc="Splitting paragraphs"):
        text = html_to_text(html)

        if need_2way:
            paras_2way = group_paragraphs_from_text(text)
            paragraphs_2way.append(paras_2way)

        if need_sentences:
            paras_sentences = paragraphs_pysbd_only(text)
            paragraphs_sentences.append(paras_sentences)

        if need_sem_03:
            paras = extract_job_paragraphs(html, similarity_threshold=0.3)
            paragraphs_sem_03.append(paras)

        if need_sem_05:
            paras = extract_job_paragraphs(html, similarity_threshold=0.5)
            paragraphs_sem_05.append(paras)

        if need_sem_075:
            paras = extract_job_paragraphs(html, similarity_threshold=0.75)
            paragraphs_sem_075.append(paras)

        if need_sem_09:
            paras = extract_job_paragraphs(html, similarity_threshold=0.9)
            paragraphs_sem_09.append(paras)

    # Add only the columns we generated
    if need_2way:
        df["paragraphs_2way"] = paragraphs_2way
    if need_sentences:
        df["paragraphs_sentences"] = paragraphs_sentences
    if need_sem_03:
        df["paragraphs_semantic_0.3"] = paragraphs_sem_03
    if need_sem_05:
        df["paragraphs_semantic_0.5"] = paragraphs_sem_05
    if need_sem_075:
        df["paragraphs_semantic_0.75"] = paragraphs_sem_075
    if need_sem_09:
        df["paragraphs_semantic_0.9"] = paragraphs_sem_09

    return df


# ============================================================
# STAGE 1 — Remove emojis & special characters
# Input  : raw job description (string)
# Output : cleaned text (string)
# ============================================================
def stage_1_clean_chars(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # First, decode HTML entities (&rsquo; → ', &nbsp; → space, etc.)
    text = html.unescape(text)

    # Convert HTML breaks to newlines (BEFORE removing special chars)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)  # <br>, <br/>, <BR>, etc.
    text = re.sub(r'<p>', '\n', text, flags=re.IGNORECASE)  # <p> tags
    text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)  # </p> tags

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove all other HTML tags

    # Clean special characters
    text = re.sub(r"[^\w\s\.,\-\/\(\)&:+']", " ", text)  # Keep apostrophes for contractions

    # Collapse multiple spaces but PRESERVE newlines
    text = re.sub(r"[ \t]+", " ", text)  # Only collapse spaces/tabs, not newlines
    text = re.sub(r"\n+", "\n", text)     # Collapse multiple newlines to single
    return text.strip()


# ============================================================
# STAGE 2 — Split text by JD section headers
# Input  : cleaned text (string)
# Output : list of section blocks
# ============================================================
SECTION_HEADERS = [
    "job title", "position",
    "company name", "organization",
    "location", "work location", "office", "city", "region",
    "job description", "about the role", "responsibilities", "duties", "tasks",
    "requirements", "qualifications", 
    "education", "academic requirements",
    "employment type", "job type", "contract type",
    "salary", "allowance", "remuneration", "compensation",
    "benefits", "perks", "incentives",
    "working hours", "schedule", "shift",
    "about the company", "company profile", "organization overview",
    "who we are", "company overview",
    "how to apply", "apply now", "application instructions",
    # Common variations found in job descriptions
    "what we offer", "what you'll be doing", "what you'll do", "what you will be doing",
    "what we're looking for", "what we are looking for", "who we're looking for",
    "about us", "why join us", "why work with us",
    "key responsibilities", "your responsibilities", "main duties",
    "required skills", "desired skills", "essential skills",
    "nice to have", "preferred qualifications", "bonus points"
]

def stage_2_split_sections(text: str, debug=False) -> list[str]:
    """
    Split text into sections by detecting job description headers.

    NEW APPROACH: For each line, check if it contains embedded headers.
    If a header is found in the middle of a line, split at that point.
    This handles cases like "...text What we offer More text..."
    → splits into ["...text", "What we offer", "More text..."]

    Header matching is CASE-INSENSITIVE.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    sections = []

    # Split by newlines first
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    for line in lines:
        # Check if this line contains any headers embedded in it
        line_lower = line.lower()

        # Try to find headers in this line
        found_header = None
        header_start = -1
        header_end = -1

        for h in SECTION_HEADERS:
            h_lower = h.lower()

            # Create pattern that handles apostrophe variations
            # "what you'll do" matches both "what you'll do" and "what you ll do"
            # Use word boundaries \b to ensure exact word matching
            pattern_parts = []
            for word in h_lower.split():
                if word.endswith("'ll"):
                    base = word[:-3]
                    pattern_parts.append(r"\b" + re.escape(base) + r"\b\s*['\s]*\bll\b")
                elif word.endswith("'re"):
                    base = word[:-3]
                    pattern_parts.append(r"\b" + re.escape(base) + r"\b\s*['\s]*\bre\b")
                elif word.endswith("'ve"):
                    base = word[:-3]
                    pattern_parts.append(r"\b" + re.escape(base) + r"\b\s*['\s]*\bve\b")
                elif word.endswith("'d"):
                    base = word[:-2]
                    pattern_parts.append(r"\b" + re.escape(base) + r"\b\s*['\s]*\bd\b")
                else:
                    pattern_parts.append(r"\b" + re.escape(word) + r"\b")

            pattern = r'\s+'.join(pattern_parts)

            # Search for this header in the line
            match = re.search(pattern, line_lower)
            if match:
                # Found a header! Record its position
                if header_start == -1 or match.start() < header_start:
                    found_header = h
                    header_start = match.start()
                    header_end = match.end()

        if found_header:
            # Header found! Split and skip the header itself
            before = line[:header_start].strip()
            after = line[header_end:].strip()

            # Add parts in sequence (preserving order), but skip the header itself
            if before:
                sections.append(before)
            if after:
                sections.append(after)
            # If line is ONLY the header (no before, no after), we skip it entirely
        else:
            # No header found - add whole line
            sections.append(line)

    # If no sections, return whole text
    if not sections:
        return [text.strip()]

    # Clean up sections: remove leading ": ", "- ", and special whitespace like \xa0
    cleaned_sections = []
    for section in sections:
        # Remove non-breaking spaces and other special whitespace characters
        section = re.sub(r'[\xa0\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]', ' ', section)

        # Remove leading ": " and "- "
        section = re.sub(r'^:\s+', '', section)
        section = re.sub(r'^-\s+', '', section)

        # Collapse multiple spaces and strip
        section = re.sub(r'\s+', ' ', section).strip()

        if section:  # Only add non-empty sections
            cleaned_sections.append(section)

    return cleaned_sections


# ============================================================
# STAGE 3 — Split sections into bullet-level sentences
# Input  : list of sections
# Output : list of sentences
# ============================================================
def stage_3_split_bullets(sections: list[str]) -> list[str]:
    """
    Split sections on bullet points AND newlines, but NOT on hyphens within words.

    Split on:
    - Bullet characters: •, ∙, ⦿, ⦁, etc.
    - Asterisks used as bullets: * Item
    - Hyphens used as bullets: - Item (with space after)
    - Newlines (in case bullets were on separate lines)

    Do NOT split on hyphens in compound words like "full-time" or "cross-functional"
    """
    sentences = []
    for sec in sections:
        # First check if section has newlines - if so, split by newlines
        if '\n' in sec:
            lines = sec.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 8:
                    sentences.append(line)
        else:
            # No newlines - split on bullet characters
            # Pattern explanation:
            # - [•\u2022\u2023\u25E6\u2043\u2219\*] : actual bullet chars and asterisk
            # - \s*-\s+ : hyphen with whitespace around it (bullet usage, not compound words)
            parts = re.split(r'[•\u2022\u2023\u25E6\u2043\u2219\*]|\s+-\s+', sec)
            sentences.extend(
                [p.strip() for p in parts if len(p.strip()) > 8]
            )
    return sentences


# ============================================================
# STAGE 4 — Remove PII (drop whole sentence)
# Input  : list of sentences
# Output : list of PII-safe sentences
# ============================================================
EMAIL_REGEX = r"r'\b[\w\.-]+@[\w\.-]+\.\w+\b'"
PHONE_REGEX = r'\b(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){6,14}\b'
URL_REGEX = r"http[s]?://\S+|www\.\S+"


def contains_pii(sentence: str) -> bool:
    if re.search(EMAIL_REGEX, sentence):
        return True
    if re.search(PHONE_REGEX, sentence):
        return True
    if re.search(URL_REGEX, sentence):
        return True

    # doc = nlp(sentence)
    # return any(ent.label_ in {"PERSON", "GPE", "LOC"} for ent in doc.ents)
    return False

def stage_3b_semantic_split(sentences: list[str], similarity_threshold: float = 0.75) -> list[str]:
    """
    Further break down each sentence using semantic paragraph splitting.
    Takes each sentence and applies extract_job_paragraphs to break it into smaller chunks.

    Input: list of sentences from stage 3
    Output: list of semantically-split paragraphs (flattened)
    """
    all_paragraphs = []

    for sentence in sentences:
        if not sentence or len(sentence.strip()) < 10:
            continue

        # Apply semantic paragraph extraction to this sentence
        paragraphs = extract_job_paragraphs(sentence, similarity_threshold=similarity_threshold)

        # Add all resulting paragraphs to our list
        if paragraphs:
            all_paragraphs.extend(paragraphs)
        else:
            # If no paragraphs extracted, keep original sentence
            all_paragraphs.append(sentence)

    return all_paragraphs


def stage_4_remove_pii(sentences: list[str]) -> list[str]:
    return [s for s in sentences if not contains_pii(s)]


# ============================================================
# APPLY ALL STAGES — DataFrame (stage-by-stage columns)
# ============================================================
def process_job_descriptions(
    df: pd.DataFrame,
    source_col: str
) -> pd.DataFrame:

    # Stage 1
    print("Stage 1: Cleaning characters and emojis...")
    df["stage_1_clean_text"] = df[source_col].progress_apply(stage_1_clean_chars)

    # Show sample of cleaned text
    if len(df) > 0:
        sample_text = df["stage_1_clean_text"].iloc[0]
        print(f"\n  Sample cleaned text (first 500 chars):")
        print(f"  {sample_text[:500]}")
        print(f"  → Contains {sample_text.count(chr(10))} newlines")

    # Stage 2
    print("\nStage 2: Splitting into sections by headers...")
    df["stage_2_sections"] = df["stage_1_clean_text"].progress_apply(lambda x: stage_2_split_sections(x, debug=False))

    # Check results
    non_empty = df["stage_2_sections"].apply(lambda x: len(x) > 0).sum()
    avg_sections = df["stage_2_sections"].apply(len).mean()
    print(f"  → {non_empty}/{len(df)} descriptions have sections")
    print(f"  → Average sections per description: {avg_sections:.2f}")

    # Stage 3
    print("\nStage 3: Splitting sections into bullet-level sentences...")
    df["stage_3_sentences"] = df["stage_2_sections"].progress_apply(stage_3_split_bullets)

    # Check results
    non_empty = df["stage_3_sentences"].apply(lambda x: len(x) > 0).sum()
    avg_sentences = df["stage_3_sentences"].apply(len).mean()
    print(f"  → {non_empty}/{len(df)} descriptions have sentences")
    print(f"  → Average sentences per description: {avg_sentences:.2f}")

    # Stage 3b
    print("\nStage 3b: Further semantic splitting of sentences (threshold=0.75)...")
    df["stage_3b_semantic_paragraphs"] = df["stage_3_sentences"].progress_apply(
        lambda x: stage_3b_semantic_split(x, similarity_threshold=0.75)
    )

    # Check results
    non_empty = df["stage_3b_semantic_paragraphs"].apply(lambda x: len(x) > 0).sum()
    avg_paragraphs = df["stage_3b_semantic_paragraphs"].apply(len).mean()
    print(f"  → {non_empty}/{len(df)} descriptions have semantic paragraphs")
    print(f"  → Average semantic paragraphs per description: {avg_paragraphs:.2f}")

    # Stage 4
    print("\nStage 4: Removing PII (names, locations, contact info)...")
    df["stage_4_final_sentences"] = df["stage_3b_semantic_paragraphs"].progress_apply(stage_4_remove_pii)

    return df

result_df=process_job_descriptions(df,"jobDescription")
write_dataframe(result_df,"data/Sample-extractions_v2_cleaned.xlsx")
print("\nProcessing complete!")
print(f"Output saved to: data/Sample-extractions_v2_cleaned.xlsx")
print(f"\nFinal DataFrame shape: {result_df.shape}")
print(f"Columns: {list(result_df.columns)}")