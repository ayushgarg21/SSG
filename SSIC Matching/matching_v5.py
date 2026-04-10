"""
SSIC Matching v5 — Fast pure-Python implementation.
No Spark. Uses pandas + FAISS embeddings for semantic fuzzy matching.

Two-level cleaning:
  - Light: legal suffixes only (for exact/contains phases)
  - Heavy: + business words (not used in v5 — embeddings replace it)

Phases:
  0. UEN direct lookup
  1. Exact match (light clean, current names)
  1b. Exact match (light clean, former names)
  1c. Exact match (light clean, original company_name)
  1d. Token-sorted exact match (light clean)
  1e. Word-boundary contains match (light clean, first-word blocking)
  2. Embedding match (sentence-transformers + FAISS)
"""

import os
import re
import time
import pickle
import pandas as pd
import numpy as np
import pymysql
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz as rfuzz  # only for Levenshtein safety check
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

t0 = time.time()

# ---------------------------
# Config
# ---------------------------
INPUT_FILE = os.getenv("INPUT_FILE", "/Users/ayush/Downloads/jan_26_jobs_final.parquet")
OUTPUT_FILE = "output/job_acra_matches_v5.parquet"
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "acra_embeddings")

DB_HOST = os.getenv("DB_HOST", "roadrunner")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME", "jobtech_knowledge_base")

print("=" * 60)
print("SSIC matching v4 — fast (no Spark)")
print("=" * 60)

# ---------------------------
# 1. Load data
# ---------------------------
print(f"\nLoading jobs from {INPUT_FILE}...")
df_jobs = pd.read_parquet(INPUT_FILE)
print(f"  Jobs loaded: {len(df_jobs):,}")

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
print(f"  ACRA loaded: {len(df_acra):,}")

print(f"  Load time: {time.time() - t0:.1f}s")


# ---------------------------
# 2. Cleaning functions
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

_business_words = [
    "holdings", "holding", "corp", "corporation", "co", "company",
    "singapore", "spore", "international", "intl",
    "group", "enterprise", "enterprises", "trading", "services", "solutions",
]
_business_patterns = []
for w in sorted(_business_words, key=len, reverse=True):
    _business_patterns.append(re.compile(r'\b' + re.escape(w) + r'\b'))

_ws = re.compile(r'\s+')
_non_alpha = re.compile(r'[^a-z0-9 ]')
_url_pattern = re.compile(r'[a-z0-9\-]+\.(com|sg|co|org|net|io|ai|edu|gov)\b.*$')


def clean_light(name):
    if pd.isna(name) or not name:
        return ""
    c = name.lower()
    c = _url_pattern.sub("", c)  # strip appended URLs
    c = _non_alpha.sub(" ", c)
    for p in _legal_patterns:
        c = p.sub("", c)
    c = _ws.sub(" ", c).strip()
    return c


def clean_heavy(name):
    if pd.isna(name) or not name:
        return ""
    c = name.lower()
    c = _non_alpha.sub(" ", c)
    for p in _legal_patterns:
        c = p.sub("", c)
    for p in _business_patterns:
        c = p.sub("", c)
    c = _ws.sub(" ", c).strip()
    return c


def token_sort(name):
    if not name:
        return ""
    return " ".join(sorted(name.split()))


# ---------------------------
# 3. Apply cleaning
# ---------------------------
print("\nCleaning names...")
t1 = time.time()

df_jobs["company_clean"] = df_jobs["company_name_final_norm"].map(clean_light)
has_orig = "company_name" in df_jobs.columns
if has_orig:
    df_jobs["company_name_orig_clean"] = df_jobs["company_name"].map(clean_light)

df_acra["company_clean"] = df_acra["entity_name"].map(clean_light)
df_acra["company_clean_heavy"] = df_acra["entity_name"].map(clean_heavy)
df_acra["former_clean"] = df_acra["former_entity_name1"].map(clean_light)

# Status priority
status_map = {"Live Company": 1, "Live": 1,
              "Live (Receiver or Receiver and Manager appointed)": 2,
              "Under Judicial Management": 2}
df_acra["status_priority"] = df_acra["entity_status_description"].map(status_map).fillna(3).astype(int)

# Filter empty
df_jobs = df_jobs[df_jobs["company_clean"] != ""].copy()
df_acra = df_acra[df_acra["company_clean"] != ""].copy()

print(f"  Cleaning time: {time.time() - t1:.1f}s")

# Unique companies
unique_companies = df_jobs[["company_name_final_norm", "company_clean"]].drop_duplicates("company_clean")
if has_orig:
    orig_map = df_jobs[["company_clean", "company_name_orig_clean"]].drop_duplicates("company_clean")
    unique_companies = unique_companies.merge(orig_map, on="company_clean", how="left")
unique_companies["company_clean_heavy"] = unique_companies["company_name_final_norm"].map(clean_heavy)

print(f"  Unique companies: {len(unique_companies):,}")


# ---------------------------
# 4. Build ACRA lookup indexes
# ---------------------------
print("\nBuilding lookup indexes...")
t2 = time.time()

# Dedup ACRA: for each cleaned name, keep the best (active, alphabetical)
df_acra_sorted = df_acra.sort_values(["status_priority", "entity_name"])

# Index: light clean name → best ACRA row
acra_by_clean = {}
for _, row in df_acra_sorted.iterrows():
    key = row["company_clean"]
    if key and key not in acra_by_clean:
        acra_by_clean[key] = row

# Index: former name (light clean) → best ACRA row
acra_by_former = {}
for _, row in df_acra_sorted.iterrows():
    key = row["former_clean"]
    if key and key not in acra_by_former:
        acra_by_former[key] = row

# Index: UEN → ACRA row
acra_by_uen = {}
for _, row in df_acra_sorted.iterrows():
    uen = str(row["uen"]).strip().upper() if pd.notna(row["uen"]) else ""
    if uen and uen not in acra_by_uen:
        acra_by_uen[uen] = row

# Index: token-sorted light clean → ACRA row
acra_by_sorted = {}
for _, row in df_acra_sorted.iterrows():
    key = token_sort(row["company_clean"])
    if key and key not in acra_by_sorted:
        acra_by_sorted[key] = row

# Index: first word → list of (full_clean_name, ACRA row) for contains matching
acra_by_first_word = defaultdict(list)
for _, row in df_acra_sorted.iterrows():
    words = row["company_clean"].split()
    if words:
        acra_by_first_word[words[0]].append((row["company_clean"], row))

# Index: light clean name → for fuzzy matching (NOT heavy — preserves distinguishing words)
acra_light_names = []
acra_light_lookup = {}
for _, row in df_acra_sorted.iterrows():
    key = row["company_clean"]
    if key and key not in acra_light_lookup:
        acra_light_names.append(key)
        acra_light_lookup[key] = row

print(f"  Index build time: {time.time() - t2:.1f}s")
print(f"  ACRA unique clean names: {len(acra_by_clean):,}")
print(f"  ACRA unique former names: {len(acra_by_former):,}")
print(f"  ACRA unique heavy names: {len(acra_light_names):,}")


# ---------------------------
# 5. Matching — all phases
# ---------------------------
ACRA_COLS = ["entity_name", "primary_ssic_code", "primary_ssic_description",
             "secondary_ssic_code", "secondary_ssic_description",
             "entity_status_description"]

results = {}  # company_clean → {match_type, acra_row, score}


def record_match(company_clean, acra_row, match_type, score=0.0):
    if company_clean not in results:
        results[company_clean] = {
            "match_type": match_type,
            "matched_entity_name": acra_row["entity_name"],
            "primary_ssic_code": acra_row["primary_ssic_code"],
            "primary_ssic_description": acra_row["primary_ssic_description"],
            "secondary_ssic_code": acra_row["secondary_ssic_code"],
            "secondary_ssic_description": acra_row["secondary_ssic_description"],
            "entity_status_description": acra_row["entity_status_description"],
            "normalized_lev": score,
            "JaccardDistance": 0.0,
        }


print("\n" + "=" * 60)
print("Running all matching phases...")
print("=" * 60)

# --- Phase 0: UEN lookup ---
uen_pattern = re.compile(r'^[0-9]{8,9}[A-Za-z]$')
uen_count = 0
for _, row in unique_companies.iterrows():
    name = row["company_name_final_norm"]
    if pd.notna(name) and uen_pattern.match(name.strip()):
        uen_key = name.strip().upper()
        if uen_key in acra_by_uen:
            record_match(row["company_clean"], acra_by_uen[uen_key], "uen_lookup")
            uen_count += 1
print(f"  Phase 0 (UEN lookup):        {uen_count:,}")

# --- Phase 1: Exact match (light clean) ---
exact_count = 0
for _, row in unique_companies.iterrows():
    cc = row["company_clean"]
    if cc not in results and cc in acra_by_clean:
        record_match(cc, acra_by_clean[cc], "exact")
        exact_count += 1
print(f"  Phase 1 (exact):             {exact_count:,}")

# --- Phase 1b: Exact match on former names ---
former_count = 0
for _, row in unique_companies.iterrows():
    cc = row["company_clean"]
    if cc not in results and cc in acra_by_former:
        record_match(cc, acra_by_former[cc], "exact_former")
        former_count += 1
print(f"  Phase 1b (former):           {former_count:,}")

# --- Phase 1c: Exact match on original company_name ---
orig_count = 0
if has_orig:
    for _, row in unique_companies.iterrows():
        cc = row["company_clean"]
        if cc not in results and pd.notna(row.get("company_name_orig_clean")):
            orig_clean = row["company_name_orig_clean"]
            if orig_clean and orig_clean in acra_by_clean:
                record_match(cc, acra_by_clean[orig_clean], "exact_orig_name")
                orig_count += 1
print(f"  Phase 1c (orig name):        {orig_count:,}")

# --- Phase 1d: Token-sorted exact match ---
sorted_count = 0
for _, row in unique_companies.iterrows():
    cc = row["company_clean"]
    if cc not in results:
        ts = token_sort(cc)
        if ts and ts in acra_by_sorted:
            record_match(cc, acra_by_sorted[ts], "exact_token_sorted")
            sorted_count += 1
print(f"  Phase 1d (token-sorted):     {sorted_count:,}")

# --- Phase 1e: Word-boundary contains (first-word blocking) ---
contains_count = 0
for _, row in unique_companies.iterrows():
    cc = row["company_clean"]
    if cc not in results and len(cc) >= 4:
        job_words = cc.split()
        if not job_words:
            continue
        first_word = job_words[0]
        candidates = acra_by_first_word.get(first_word, [])
        best = None
        best_priority = 99
        for acra_clean, acra_row in candidates:
            acra_words = acra_clean.split()
            # Word-prefix check: job words are prefix of ACRA words
            n = min(len(job_words), len(acra_words))
            if job_words[:n] == acra_words[:n]:
                # Word overlap ratio >= 70%
                ratio = n / max(len(job_words), len(acra_words))
                if ratio >= 0.7:
                    if acra_row["status_priority"] < best_priority:
                        best = acra_row
                        best_priority = acra_row["status_priority"]
        if best is not None:
            record_match(cc, best, "contains")
            contains_count += 1
print(f"  Phase 1e (contains):         {contains_count:,}")

# --- Phase 2: Embedding match (FAISS + sentence-transformers) ---
print("  Phase 2 (embedding): loading FAISS index...")
t3 = time.time()

unmatched = [row["company_clean"]
             for _, row in unique_companies.iterrows()
             if row["company_clean"] not in results and row["company_clean"]]

embedding_count = 0

index_path = os.path.join(EMBEDDINGS_DIR, "index.faiss")
metadata_path = os.path.join(EMBEDDINGS_DIR, "metadata.pkl")

if unmatched and os.path.exists(index_path) and os.path.exists(metadata_path):
    # Load FAISS index and metadata
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        acra_metadata = pickle.load(f)
    print(f"    FAISS index loaded: {index.ntotal:,} vectors ({time.time() - t3:.1f}s)")

    # Load the same model used to build the index
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"    Model loaded ({time.time() - t3:.1f}s)")

    # Encode unmatched company names
    print(f"    Encoding {len(unmatched):,} unmatched names...")
    query_embeddings = model.encode(unmatched, batch_size=512, normalize_embeddings=True)
    query_embeddings = np.array(query_embeddings, dtype=np.float32)
    print(f"    Encoding done ({time.time() - t3:.1f}s)")

    # Search: top-5 nearest neighbors per query
    TOP_K = 5
    COSINE_THRESHOLD = 0.85
    LEV_THRESHOLD = 0.4  # safety check: normalized Levenshtein must be < 0.4

    print(f"    Searching top-{TOP_K} neighbors (cosine >= {COSINE_THRESHOLD})...")
    scores, indices = index.search(query_embeddings, TOP_K)

    for i, cc in enumerate(unmatched):
        best_match = None
        best_score = 0.0
        best_priority = 99

        for j in range(TOP_K):
            idx = indices[i][j]
            cosine_score = scores[i][j]

            if idx == -1 or cosine_score < COSINE_THRESHOLD:
                continue

            candidate = acra_metadata[idx]

            # Levenshtein safety check: prevent semantically similar but factually different matches
            lev_score = rfuzz.ratio(cc, candidate["company_clean"]) / 100.0
            norm_lev = 1.0 - lev_score
            if norm_lev > LEV_THRESHOLD:
                continue

            # Prefer higher cosine, then active entities
            if (cosine_score > best_score) or \
               (cosine_score == best_score and candidate["status_priority"] < best_priority):
                best_match = candidate
                best_score = cosine_score
                best_priority = candidate["status_priority"]

        if best_match is not None:
            # Convert metadata dict to something record_match can use
            class RowProxy:
                def __init__(self, d):
                    self._d = d
                def __getitem__(self, key):
                    return self._d[key]
            record_match(cc, RowProxy(best_match), "embedding",
                         score=round(1.0 - best_score, 3))
            embedding_count += 1

    print(f"  Phase 2 (embedding):         {embedding_count:,} ({time.time() - t3:.1f}s)")

    # Phase 2b: rapidfuzz fallback for names that embeddings missed
    # (handles short abbreviations like "DBS", "OCBC" where embeddings fail)
    still_unmatched = [name for name in unmatched if name not in results]
    if still_unmatched:
        print(f"  Phase 2b (rapidfuzz fallback): {len(still_unmatched):,} remaining...")
        t4 = time.time()
        from rapidfuzz import process as rfprocess
        fallback_count = 0
        for cc in still_unmatched:
            result = rfprocess.extractOne(
                cc, acra_light_names,
                scorer=rfuzz.ratio,
                score_cutoff=90
            )
            if result:
                matched_name, score, _ = result
                acra_row = acra_light_lookup[matched_name]
                record_match(cc, acra_row, "fuzzy_fallback", score=round(1.0 - score / 100.0, 3))
                fallback_count += 1
        print(f"  Phase 2b (rapidfuzz fallback): {fallback_count:,} ({time.time() - t4:.1f}s)")
else:
    if not os.path.exists(index_path):
        print("  Phase 2 (embedding): SKIPPED — run build_acra_embeddings.py first")
    else:
        print("  Phase 2 (embedding):         0 (no unmatched companies)")

total_matched = len(results)
total_unmatched = len(unique_companies) - total_matched
print(f"\n  TOTAL matched companies: {total_matched:,} / {len(unique_companies):,}")
print(f"  TOTAL unmatched:         {total_unmatched:,}")


# ---------------------------
# 6. Join results back to jobs
# ---------------------------
print("\nJoining back to jobs...")

results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.index.name = "company_clean"
results_df = results_df.reset_index()

df_out = df_jobs.merge(results_df, on="company_clean", how="left")

matched_jobs = df_out["primary_ssic_code"].notna().sum()
total_jobs = len(df_out)
print(f"\nTotal jobs matched: {matched_jobs:,} out of {total_jobs:,} ({100 * matched_jobs / total_jobs:.1f}%)")

for mt in ["uen_lookup", "exact", "exact_former", "exact_orig_name",
           "exact_token_sorted", "contains", "embedding", "fuzzy_fallback"]:
    ct = (df_out["match_type"] == mt).sum()
    if ct > 0:
        print(f"  - {mt:25s} {ct:>8,} ({100 * ct / total_jobs:.1f}%)")

unmatched_jobs = df_out["primary_ssic_code"].isna().sum()
print(f"  - {'unmatched':25s} {unmatched_jobs:>8,} ({100 * unmatched_jobs / total_jobs:.1f}%)")


# ---------------------------
# 7. Save
# ---------------------------
print(f"\nSaving to {OUTPUT_FILE}...")
df_out.to_parquet(OUTPUT_FILE, index=False)
print(f"Done! Total time: {time.time() - t0:.1f}s")
