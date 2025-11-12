import sys
import logging
sys.path.insert(0, "/Users/eugene/Documents/py_utils")
from transformers import pipeline
import re
from tqdm import tqdm
tqdm.pandas()
import spacy,nltk
import en_core_web_sm
import html
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

# Import your custom module
from jt_golden_utility import execute_read_query, create_db_engine, execute_insert_update, setup_logger

logger = setup_logger(log_dir='logs', log_file='log', retention_days=30)
logger.setLevel(logging.INFO)

RRdb=create_db_engine("RR",logger,"jobtech_data_singapore2")
nlp = spacy.load("en_core_web_sm")
nlp = en_core_web_sm.load()
# jobtech_job_test_ssg_2025
# UATdb=create_db_engine("REBOOTPROACCCONFIG",logger,"account")

def get_data(RRdb):
    query_with_email="""
    SELECT job_id,source_id,job_url,job_title,job_description,date_posted 
    FROM jobtech_data_singapore2.jobtech_job_test_ssg_2025 where job_description like "%@%.com"
    ORDER BY RAND()
    LIMIT 5000
    """
    data=execute_read_query(RRdb,query_with_email,logger)
    print(data)
    return data

pii_pipe = pipeline("token-classification", model="iiiorg/piiranha-v1-detect-personal-information", aggregation_strategy="simple")
# ner_pipe = pipeline("token-classification", model="your-org/finetuned-job-ner", aggregation_strategy="simple")

def process_mask_text(text:str)->str:
    def mask_contacts(text):
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '<EMAIL>', text)
        text = re.sub(r'\b(?:\+?\d{1,3}[ -]?)?(?:\d[ -]?){6,14}\b', '<PHONE>', text)
           # Mask URLs or domains (http, https, www, or bare domain like example.com)
        text = re.sub(
        r'(?:https?://|www\.|[a-zA-Z0-9.-]+\.(?:com|sg|net|org|io|edu|gov))\S*',
        '<URL>',
        text,
        flags=re.IGNORECASE
    )
        return text

    text_masked = mask_contacts(text)
    pii_entities = pii_pipe(text_masked)
    # ner_entities = ner_pipe(text_masked)
    return text_masked

def process_pii_text(text:str)->str:

    pii_entities = pii_pipe(text)
    # ner_entities = ner_pipe(text_masked)
    return pii_entities

def remove_sentences_with_pii_tokens(sentences: list, ner_tokens: list) -> str:
    """
    Remove sentences containing any of the ner_tokens from a list of sentences.

    Parameters:
    - sentences: list of sentence strings
    - ner_tokens: list of tokens to remove (e.g., ['<EMAIL>', '<PHONE>'])

    Returns:
    - string with sentences joined, excluding those containing any ner_tokens
    """
    clean_sentences = [
        s for s in sentences if not any(token in s for token in ner_tokens)
    ]
    return " ".join(clean_sentences)
def remove_sentences_with_pii_entities(sentences: list, pii_entities: list, threshold: float = 0.9) -> str:
    """
    Remove sentences that contain any PII entities with score above threshold.

    Parameters:
    - sentences: list of sentence strings
    - pii_entities: list of dicts from pii_pipe output
      Each dict should have keys: 'word', 'score', 'entity_group', 'start', 'end'
    - threshold: minimum score to consider for removal

    Returns:
    - string with sentences joined, excluding those containing high-confidence PII
    """
    # Filter entities by score
    high_conf_pii_words = [e['word'] for e in pii_entities if e['score'] > threshold]

    # Remove sentences that contain any high-confidence PII
    clean_sentences = [
        s for s in sentences if not any(pii_word in s for pii_word in high_conf_pii_words)
    ]

    return " ".join(clean_sentences)
def split_into_sentece_with_spacy(text):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def preprocess_line_breaks(text: str) -> str:
    """
    Replace HTML <br> tags and newlines with a period + space
    so that sentence tokenizer recognizes them as sentence boundaries.
    """
    text = html.unescape(text)

    text = text.replace("<br>", ". ")
    text = text.replace("\n", ". ")
    text = text.replace("<br/>>", ". ")
    text = text.replace("</br>", ". ")


    text = text.replace("<br/>", ". ")
    text = text.replace(" . ", "")
    text = text.replace("..", ".")
    text = text.replace("..", ".")
    text = text.replace("!.", "!")
    text = text.replace("!.", "!")
    text = text.replace("• ", "•")
    text = text.replace("· ", "·")
    text = text.replace(":.", ":")

    text = re.sub(r'\s+', ' ', text) #Replace any sequence of whitespace (' ', '\t', '\n', etc.) with a single space.
    re.sub(r'\?+', '?', text) #Replace any sequence of "?" with a single ?.








    return text
data=get_data(RRdb)
data['job_description'] = data['job_description'].progress_apply(preprocess_line_breaks)
data['masked'] = data['job_description'].progress_apply(process_mask_text)
data['pii'] = data['masked'].progress_apply(process_pii_text)

data["nltk_sentences"]=data['masked'].progress_apply(sent_tokenize)
data["spacy_sentences"]=data['masked'].progress_apply((split_into_sentece_with_spacy))

data['masked_remove'] = data['spacy_sentences'].progress_apply(remove_sentences_with_pii_tokens, ner_tokens=['<EMAIL>', '<PHONE>','<URL>'])

# Step 4: Remove sentences containing high-confidence PII entities (>0.9)
data['pii_remove'] = data.progress_apply(
    lambda row: remove_sentences_with_pii_entities(
        row['spacy_sentences'], 
        row['pii'], 
        threshold=0.9
    ),
    axis=1
)
## remove_sentences_with_pii_entities takes in senetence now, so have to break masked_remove into sentences first
data["masked_remove_sentences"]=data['masked_remove'].progress_apply((split_into_sentece_with_spacy))
data['both_remove'] = data.progress_apply(
    lambda row: remove_sentences_with_pii_entities(
        row['masked_remove_sentences'], 
        row['pii'], 
        threshold=0.9
    ),
    axis=1
)
data.to_excel("processed_df.xlsx",index=False)
