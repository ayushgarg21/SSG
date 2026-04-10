"""
One-time script: Generate ACRA entity name embeddings and FAISS index.

Run this once (or whenever ACRA data changes).
Outputs:
  acra_embeddings/vectors.npy    — 2M x 384 float32 vectors
  acra_embeddings/index.faiss    — FAISS index for fast nearest-neighbor search
  acra_embeddings/metadata.pkl   — mapping: index position → ACRA row dict
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import pymysql
import faiss
from sentence_transformers import SentenceTransformer
import re
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

t0 = time.time()

# Config
DB_HOST = os.getenv("DB_HOST", "roadrunner")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME", "jobtech_knowledge_base")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "acra_embeddings")
BATCH_SIZE = 512
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# 1. Load ACRA data
# ---------------------------
print("Loading ACRA data from MariaDB...")
conn = pymysql.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME)
df_acra = pd.read_sql("""
    SELECT uen, entity_name, former_entity_name1, entity_status_description,
           primary_ssic_code, primary_ssic_description,
           secondary_ssic_code, secondary_ssic_description
    FROM acra_directory
    WHERE entity_name IS NOT NULL AND TRIM(entity_name) != ''
""", conn)
conn.close()
print(f"  ACRA loaded: {len(df_acra):,} rows ({time.time() - t0:.1f}s)")

# ---------------------------
# 2. Light clean (same as matching_fast.py)
# ---------------------------
_legal_suffixes = [
    "private limited", "pte ltd", "pte. ltd.", "pte. ltd",
    "(s) pte", "(s)pte", "limited", "ltd", "llc", "inc",
]
_legal_patterns = []
for w in sorted(_legal_suffixes, key=len, reverse=True):
    cw = w.replace(".", "").replace("(", "").replace(")", "").strip()
    if cw:
        _legal_patterns.append(re.compile(r'\b' + re.escape(cw) + r'\b'))

_ws = re.compile(r'\s+')
_non_alpha = re.compile(r'[^a-z0-9 ]')
_url_pattern = re.compile(r'[a-z0-9\-]+\.(com|sg|co|org|net|io|ai|edu|gov)\b.*$')


def clean_light(name):
    if pd.isna(name) or not name:
        return ""
    c = name.lower()
    c = _url_pattern.sub("", c)
    c = _non_alpha.sub(" ", c)
    for p in _legal_patterns:
        c = p.sub("", c)
    c = _ws.sub(" ", c).strip()
    return c


df_acra["company_clean"] = df_acra["entity_name"].map(clean_light)

# Status priority
status_map = {"Live Company": 1, "Live": 1,
              "Live (Receiver or Receiver and Manager appointed)": 2,
              "Under Judicial Management": 2}
df_acra["status_priority"] = df_acra["entity_status_description"].map(status_map).fillna(3).astype(int)

# Dedup: keep best (active, alphabetical) per cleaned name
df_acra = df_acra[df_acra["company_clean"] != ""].copy()
df_acra = df_acra.sort_values(["status_priority", "entity_name"])
df_acra_dedup = df_acra.drop_duplicates(subset="company_clean", keep="first").reset_index(drop=True)

print(f"  Unique ACRA names after dedup: {len(df_acra_dedup):,}")

# ---------------------------
# 3. Generate embeddings
# ---------------------------
print(f"\nLoading model '{MODEL_NAME}'...")
model = SentenceTransformer(MODEL_NAME)
print(f"  Model loaded ({time.time() - t0:.1f}s)")

names = df_acra_dedup["company_clean"].tolist()
print(f"Encoding {len(names):,} names in batches of {BATCH_SIZE}...")

t_enc = time.time()
embeddings = model.encode(names, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings, dtype=np.float32)
print(f"  Encoding done: {embeddings.shape} ({time.time() - t_enc:.1f}s)")

# ---------------------------
# 4. Build FAISS index
# ---------------------------
print("\nBuilding FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity (vectors are normalized)
index.add(embeddings)
print(f"  FAISS index built: {index.ntotal:,} vectors, {dimension}d")

# ---------------------------
# 5. Save everything
# ---------------------------
print("\nSaving to disk...")

vectors_path = os.path.join(OUTPUT_DIR, "vectors.npy")
index_path = os.path.join(OUTPUT_DIR, "index.faiss")
metadata_path = os.path.join(OUTPUT_DIR, "metadata.pkl")

np.save(vectors_path, embeddings)
faiss.write_index(index, index_path)

# Save metadata: list of dicts with ACRA info for each index position
metadata = []
for _, row in df_acra_dedup.iterrows():
    metadata.append({
        "company_clean": row["company_clean"],
        "entity_name": row["entity_name"],
        "uen": row["uen"],
        "primary_ssic_code": row["primary_ssic_code"],
        "primary_ssic_description": row["primary_ssic_description"],
        "secondary_ssic_code": row["secondary_ssic_code"],
        "secondary_ssic_description": row["secondary_ssic_description"],
        "entity_status_description": row["entity_status_description"],
        "status_priority": row["status_priority"],
    })

with open(metadata_path, "wb") as f:
    pickle.dump(metadata, f)

vectors_mb = os.path.getsize(vectors_path) / 1024 / 1024
index_mb = os.path.getsize(index_path) / 1024 / 1024
meta_mb = os.path.getsize(metadata_path) / 1024 / 1024

print(f"  vectors.npy:  {vectors_mb:.1f} MB")
print(f"  index.faiss:  {index_mb:.1f} MB")
print(f"  metadata.pkl: {meta_mb:.1f} MB")
print(f"\nDone! Total time: {time.time() - t0:.1f}s")
