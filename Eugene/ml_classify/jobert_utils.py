"""
Shared utilities for JoBert classification preprocessing
"""
import pandas as pd
import html
import re
import spacy
import en_core_web_lg

# Load spacy globally for sentence splitting
nlp = en_core_web_lg.load()
# Disable heavy pipes we don't need
nlp.disable_pipes(['ner', 'parser', 'tagger', 'attribute_ruler', 'lemmatizer'])
# Add lightweight sentencizer for sentence boundary detection
nlp.add_pipe('sentencizer')

# Common section headers to filter out (they're not real sentences)
SECTION_HEADERS_TO_FILTER = {
    "job responsibilities", "responsibilities", "job responsibilities:",
    "requirements", "requirements:", "job requirements", "job requirements:",
    "qualifications", "qualifications:", "qualifications & skills",
    "about the company", "about the company:", "about us", "about us:",
    "job description", "job description:", "job summary", "job summary:",
    "duties", "duties:", "key responsibilities", "key responsibilities:",
    "benefits", "benefits:", "what we offer", "what we offer:",
    "key", "job", "role", "overview", "summary", "description",
    "skills", "experience", "education", "the role", "the role:",
    "competencies", "competencies:", "roles &", "key.",
}

# Minimum word count for a valid sentence
MIN_WORDS = 3


def preprocess_line_breaks(text: str) -> str:
    """Clean HTML and replace line breaks with periods"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g., &amp; -> &, &acirc; -> â)
    text = html.unescape(text)

    # =========================================
    # STEP 1: Remove/Replace HTML tags
    # =========================================
    # Replace <br> tags with newlines (to preserve structure)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = text.replace("</br>", "\n")

    # Replace <p>, <div>, <li> tags with newlines
    text = re.sub(r'</?p[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?div[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?li[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?ul[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?ol[^>]*>', '\n', text, flags=re.IGNORECASE)

    # Replace <tr>, <td> with space/newline
    text = re.sub(r'</?tr[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</?td[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?th[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'</?table[^>]*>', '\n', text, flags=re.IGNORECASE)

    # Replace headers with newlines
    text = re.sub(r'</?h[1-6][^>]*>', '\n', text, flags=re.IGNORECASE)

    # Remove all other HTML tags completely
    text = re.sub(r'<[^>]+>', '', text)

    # Remove &nbsp; and other HTML entities that might remain
    text = re.sub(r'&nbsp;', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)

    # =========================================
    # STEP 2: Clean up line breaks and bullets
    # =========================================
    # Replace newlines with period + space
    text = text.replace("\n", ". ")

    # Clean up bullets - replace with period
    text = re.sub(r'[•·▪►▸‣⁃⦿○●◦]', '.', text)

    # Clean up dashes used as bullets at start of lines
    text = re.sub(r'\.\s*-\s+', '. ', text)

    # =========================================
    # STEP 3: Clean up punctuation
    # =========================================
    text = text.replace("..", ".")
    text = text.replace("!.", "!")
    text = text.replace("?.", "?")
    text = text.replace(":.", ":")
    text = text.replace(" .", ".")
    text = text.replace(".- ", ". ")

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Replace multiple periods with single
    text = re.sub(r'\.+', '.', text)

    # Replace multiple question marks
    text = re.sub(r'\?+', '?', text)

    # Remove leading period if exists
    text = text.strip()
    if text.startswith('.'):
        text = text[1:].strip()

    return text


def is_valid_sentence(sentence: str) -> bool:
    """Check if a sentence is valid (not just a header, not too short)"""
    if not sentence:
        return False

    sentence_clean = sentence.strip()

    # Remove trailing punctuation for comparison
    sentence_lower = sentence_clean.rstrip('.:!?').lower().strip()

    # Filter out section headers
    if sentence_lower in SECTION_HEADERS_TO_FILTER:
        return False

    # Filter out very short sentences (less than MIN_WORDS words)
    words = sentence_clean.split()
    if len(words) < MIN_WORDS:
        return False

    # Filter out sentences that are just punctuation
    if re.match(r'^[\s\.\,\!\?\:\;\-]+$', sentence_clean):
        return False

    # Filter out sentences that are just numbers/IDs
    if re.match(r'^[\d\s\-\(\)\.]+$', sentence_clean):
        return False

    return True


def split_into_sentences(text, smart_split=True):
    """Split text into sentences using spacy with improved filtering"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []

    if smart_split:
        SECTION_HEADERS = [
            "Job Responsibilities",
            "Responsibilities",
            "Job responsibilities",
            "Requirements",
            "Requirement",
            "Qualifications",
            "About the Company",
            "Job Description",
            "Duties",
            "Benefits",
            "Key Responsibilities",
            "Job Summary",
            "What We Offer",
            "About Us",
            "The Role",
            "Skills",
            "Experience",
            "Education",
        ]

        # Insert breaks before section headers like "Responsibilities:"
        pattern = r"(" + "|".join(SECTION_HEADERS) + r")\s*:"
        text = re.sub(pattern, r". \1: ", text, flags=re.IGNORECASE)
        text = text.replace(" .", ".")
        text = re.sub(r'\.+', '.', text)

        # Split when uppercase word starts after lowercase (e.g. "...implementationConduct...")
        text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)

    doc = nlp(text)

    # Extract sentences and filter
    sentences = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        if is_valid_sentence(sentence):
            sentences.append(sentence)

    return sentences


# Keep old function for backwards compatibility
def split_into_sentences_old(text, smart_split=True):
    """Original split function (for comparison)"""
    if pd.isna(text) or not isinstance(text, str) or not text.strip():
        return []
    if smart_split == True:
        SECTION_HEADERS = [
            "Job Responsibilities",
            "Responsibilities",
            "Job responsibilities",
            "Requirements",
            "Requirement",
            "Qualifications",
            "About the Company",
            "Job Description",
            "Duties",
            "Benefits",
        ]

        pattern = r"(" + "|".join(SECTION_HEADERS) + r")\s*:"
        text = re.sub(pattern, r". \1: ", text)
        text = text.replace(" .", ".")
        text = re.sub(r'\.+', '.', text)

        text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences
