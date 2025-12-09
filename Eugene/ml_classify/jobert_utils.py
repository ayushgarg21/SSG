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


def preprocess_line_breaks(text: str) -> str:
    """Replace HTML <br> tags and newlines with periods"""
    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = text.replace("<br>", ". ")
    text = text.replace("\n", ". ")
    text = text.replace("<br/>>", ". ")
    text = text.replace("</br>", ". ")
    text = text.replace("<br/>", ". ")
    text = text.replace(" . ", "")
    text = text.replace("..", ".")
    text = text.replace("!.", "!")
    text = text.replace("• ", "•")
    text = text.replace("· ", "·")
    text = text.replace(":.", ":")
    text = text.replace("•", ".")
    text = text.replace(" - ", ".")
    text = text.replace(" .", ".")


    text = re.sub(r'\s+', ' ', text)
    text=re.sub(r'\?+', '?', text) #Replace any sequence of "?" with a single ?.
    text = re.sub(r'\.+', '.', text) #Replace any sequence of "." with a single .
    
    return text


def split_into_sentences(text, smart_split=True):
    """Split text into sentences using spacy"""
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

        # 2. Insert breaks before section headers like "Responsibilities:"
        pattern = r"(" + "|".join(SECTION_HEADERS) + r")\s*:"
        text = re.sub(pattern, r". \1: ", text)
        text = text.replace(" .", ".")
        text = re.sub(r'\.+', '.', text) #Replace any sequence of "." with a single .

        # 3. Split when uppercase word starts after lowercase (e.g. "...implementationConduct...")
        text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences
