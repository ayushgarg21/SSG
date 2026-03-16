from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import Tokenizer, HashingTF, MinHashLSH, NGram

# ---------------------------
# Configuration
# ---------------------------
DEBUG = False  # Set to True to enable debug output

minio_endpoint = "http://192.168.0.57:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin123"
input_file = "s3a://ssg-sg-jobs/jan_26_jobs_final.parquet"

print("="*60)
print("SSIC matching")
print("="*60)
print(f"MinIO Endpoint: {minio_endpoint}")
print(f"Input File: {input_file}")
print()

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("SSIC Matching") \
    .master("spark://192.168.0.57:7077") \
    .config("spark.executor.memory", "6g") \
    .config("spark.num.executors", "3") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.cores", "4") \
    .config("spark.cores.max", "12") \
    .config("spark.driver.host", "192.168.0.121") \
    .config("spark.driver.bindAddress", "192.168.0.121") \
    .config("spark.driver.maxResultSize", "3g") \
    .config("spark.sql.shuffle.partitions", "12") \
    .config("spark.default.parallelism", "12") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.2,com.amazonaws:aws-java-sdk-bundle:1.12.720,org.mariadb.jdbc:mariadb-java-client:3.1.4") \
    .getOrCreate()

# Configure MinIO/S3 access
conf = spark._jsc.hadoopConfiguration()
conf.set("fs.s3a.endpoint", minio_endpoint)
conf.set("fs.s3a.access.key", minio_access_key)
conf.set("fs.s3a.secret.key", minio_secret_key)
conf.set("fs.s3a.path.style.access", "true")
conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("spark.hadoop.fs.s3a.aws.credentials.provider",
         "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

print(f"Spark version: {spark.version}")
print(f"Spark master: {spark.sparkContext.master}")



# ---------------------------
# 2. Read Input DataFrame
# ---------------------------
print(f"Reading data from {input_file}...")
# Read and immediately repartition for parallelization
df_jobs = spark.read.parquet(input_file).repartition(12)
# df_jobs = spark.read.parquet(input_file).limit(100).repartition(12)

total_rows = df_jobs.count()
print(f"Total rows (sampled): {total_rows:,}")

if DEBUG:
    print("\n[DEBUG] Jobs DataFrame columns:", df_jobs.columns)
    print("[DEBUG] Jobs sample data:")
    df_jobs.select("company_name_final_norm").show(5, truncate=False)


# ---------------------------
# 3. Load ACRA data from MariaDB
# ---------------------------
# FIX: Add sessionVariables=sql_mode=ANSI_QUOTES to make MariaDB treat " as identifier quotes
jdbc_url = "jdbc:mariadb://128.106.171.242:7706/jobtech_knowledge_base?sessionVariables=sql_mode=ANSI_QUOTES"
jdbc_properties = {
    "user": "eugene",
    "password": "pR^i2t^#EmLBGQ",
    "driver": "org.mariadb.jdbc.Driver"
}
if DEBUG:
    print("[FIX] JDBC URL configured with ANSI_QUOTES mode")

print("Loading ACRA data from MariaDB...")

# Use a subquery to filter at database level (more efficient)
acra_query = """
(SELECT
    entity_name,
    primary_ssic_code,
    primary_ssic_description,
    CAST(ROW_NUMBER() OVER (ORDER BY entity_name) AS SIGNED) as row_id
FROM acra_directory
WHERE entity_name IS NOT NULL
  AND TRIM(entity_name) != ''
) AS acra_filtered
"""

df_acra = spark.read.jdbc(
    url=jdbc_url,
    table=acra_query,
    column="row_id",           # Partition by this column
    lowerBound=1,              # Start of range
    upperBound=2000000,        # Approximate max rows
    numPartitions=12,          # Create 12 partitions matching shuffle partitions
    properties=jdbc_properties
).select("entity_name", "primary_ssic_code", "primary_ssic_description")

acra_count = df_acra.count()
print(f"ACRA rows loaded: {acra_count:,}")

if DEBUG:
    print("[DEBUG] ACRA sample data:")
    df_acra.show(5, truncate=False)

# ---------------------------
# 4. Clean & normalize company names
# ---------------------------
stopwords = ["pte ltd", "private limited", "ltd", "llc"]

def clean_name(col):
    # Step 1: Convert to lowercase FIRST
    c = F.lower(col)
    # Step 2: Remove special characters (keep only a-z, 0-9, and spaces)
    c = F.regexp_replace(c, "[^a-z0-9 ]", " ")
    # Step 3: Remove stopwords
    for word in stopwords:
        c = F.regexp_replace(c, word, "")
    # Step 4: Collapse multiple spaces into one
    c = F.regexp_replace(c, "\s+", " ")
    # Step 5: Trim leading/trailing spaces
    return F.trim(c)

df_jobs = df_jobs.withColumn("company_clean", clean_name(F.col("company_name_final_norm")))
df_acra = df_acra.withColumn("company_clean", clean_name(F.col("entity_name")))

# FILTER OUT EMPTY NAMES BEFORE EXPENSIVE OPERATIONS (tokenize, hash, etc.)
print("Filtering empty company names...")
df_jobs = df_jobs.filter((F.col("company_clean").isNotNull()) & (F.col("company_clean") != ""))
df_acra = df_acra.filter((F.col("company_clean").isNotNull()) & (F.col("company_clean") != ""))

if DEBUG:
    print(f"[DEBUG] Jobs after filtering: {df_jobs.count():,}")
    print(f"[DEBUG] ACRA after filtering: {df_acra.count():,}")
    print("\n[DEBUG] Cleaned company names (Jobs):")
    df_jobs.select("company_name_final_norm", "company_clean").show(10, truncate=False)
    print("[DEBUG] Cleaned company names (ACRA):")
    df_acra.select("entity_name", "company_clean").show(10, truncate=False)

# ---------------------------
# 5. Extract unique companies for matching
# ---------------------------
print("Extracting unique companies...")
# Get unique company names to reduce matching workload
df_unique_companies = df_jobs.select("company_name_final_norm", "company_clean").distinct()
unique_count = df_unique_companies.count()
print(f"Unique companies: {unique_count:,} (reduced from {df_jobs.count():,})")

# ---------------------------
# 6. Tokenize company names
# ---------------------------
print("Tokenizing company names...")
tokenizer = Tokenizer(inputCol="company_clean", outputCol="tokens")
df_unique_companies = tokenizer.transform(df_unique_companies)
df_acra = tokenizer.transform(df_acra)

if DEBUG:
    print("[DEBUG] Tokenized data (Unique Companies):")
    df_unique_companies.select("company_clean", "tokens").show(5, truncate=False)
    print("[DEBUG] Tokenized data (ACRA):")
    df_acra.select("company_clean", "tokens").show(5, truncate=False)

# ---------------------------
# 7. Hash tokens into vectors
# ---------------------------
print("Creating feature vectors...")
hashingTF = HashingTF(inputCol="tokens", outputCol="features", numFeatures=1024)
df_unique_companies = hashingTF.transform(df_unique_companies)
df_acra = hashingTF.transform(df_acra)

# Persist ACRA data to memory AND disk (spills to disk if memory is full)
print("Persisting ACRA features...")
from pyspark import StorageLevel
df_acra = df_acra.persist(StorageLevel.MEMORY_AND_DISK)
df_acra.count()  # Force persistence

if DEBUG:
    print("[DEBUG] Feature vectors created (Unique Companies):")
    df_unique_companies.select("company_clean", "features").show(5, truncate=False)
    print("[DEBUG] Feature vectors created (ACRA):")
    df_acra.select("company_clean", "features").show(5, truncate=False)

# ---------------------------
# 8. MinHash LSH approximate similarity join (on unique companies)
# ---------------------------
print("Building MinHash LSH model...")
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=6)
model = mh.fit(df_acra)

print(f"Running LSH similarity join on {unique_count:,} unique companies (threshold=0.65)...")
# Match unique companies only to reduce workload
matches = model.approxSimilarityJoin(df_unique_companies, df_acra, 0.65, distCol="JaccardDistance")

match_count = matches.count()
print(f"Total matches found (before filtering): {match_count:,}")

# ---------------------------
# 8.5. Filter false positives using Levenshtein distance
# ---------------------------
print("Filtering matches by Levenshtein distance...")
matches = matches.withColumn(
    "levenshtein_dist",
    F.levenshtein(F.col("datasetA.company_clean"), F.col("datasetB.company_clean"))
)

# Calculate normalized Levenshtein distance (0-1 scale)
matches = matches.withColumn(
    "normalized_lev",
    F.col("levenshtein_dist") / F.greatest(
        F.length(F.col("datasetA.company_clean")),
        F.length(F.col("datasetB.company_clean"))
    )
)

# Keep only matches where normalized Levenshtein < 0.5 (strings are >50% similar)
matches = matches.filter(F.col("normalized_lev") < 0.35)

match_count_filtered = matches.count()
print(f"Total matches after Levenshtein filter: {match_count_filtered:,}")

if DEBUG and match_count_filtered > 0:
    print("[DEBUG] Sample filtered matches:")
    matches.select(
        F.col("datasetA.company_clean").alias("jobs_company"),
        F.col("datasetB.company_clean").alias("acra_company"),
        F.col("JaccardDistance"),
        F.col("normalized_lev")
    ).show(10, truncate=False)

if match_count_filtered == 0:
    print("⚠️ NO MATCHES AFTER FILTERING! Trying more relaxed Levenshtein threshold (0.6)...")
    matches = model.approxSimilarityJoin(df_unique_companies, df_acra, 0.65, distCol="JaccardDistance")
    matches = matches.withColumn(
        "levenshtein_dist",
        F.levenshtein(F.col("datasetA.company_clean"), F.col("datasetB.company_clean"))
    )
    matches = matches.withColumn(
        "normalized_lev",
        F.col("levenshtein_dist") / F.greatest(
            F.length(F.col("datasetA.company_clean")),
            F.length(F.col("datasetB.company_clean"))
        )
    )
    matches = matches.filter(F.col("normalized_lev") < 0.6)
    match_count_filtered = matches.count()
    print(f"Matches with relaxed filter: {match_count_filtered:,}")

# ---------------------------
# 9. Select relevant columns and get best match per unique company
# ---------------------------
matches_clean = matches.select(
    F.col("datasetA.company_name_final_norm"),
    F.col("datasetA.company_clean"),
    F.col("datasetB.primary_ssic_code"),
    F.col("datasetB.primary_ssic_description"),
    F.col("datasetB.entity_name").alias("matched_entity_name"),
    F.col("JaccardDistance"),
    F.col("normalized_lev")
)

# Pick best match per unique company (lowest Levenshtein distance, then Jaccard)
from pyspark.sql.window import Window
window = Window.partitionBy("company_clean").orderBy("normalized_lev", "JaccardDistance")

best_matches_unique = matches_clean.withColumn("rank", F.row_number().over(window)) \
                                   .filter(F.col("rank") == 1) \
                                   .drop("rank")

print(f"Best matches for unique companies: {best_matches_unique.count():,}")

# ---------------------------
# 10. Join matches back to all jobs
# ---------------------------
print("Joining matches back to all jobs...")
best_matches = df_jobs.join(
    best_matches_unique.select(
        "company_clean",
        "primary_ssic_code",
        "primary_ssic_description",
        "matched_entity_name",
        "JaccardDistance",
        "normalized_lev"
    ),
    on="company_clean",
    how="left"
)

# ---------------------------
# 11. Inspect results
# ---------------------------
# Count jobs that got matched (have non-null SSIC codes)
matched_jobs = best_matches.filter(F.col("primary_ssic_code").isNotNull()).count()
print(f"Total jobs matched: {matched_jobs:,} out of {df_jobs.count():,}")

if DEBUG and matched_jobs > 0:
    print("\n[RESULTS] Sample matched jobs:")
    best_matches.filter(F.col("primary_ssic_code").isNotNull()) \
                .select("company_name_final_norm", "matched_entity_name", "primary_ssic_code", "normalized_lev", "JaccardDistance") \
                .show(20, truncate=False)
elif matched_jobs == 0:
    print("⚠️ NO JOBS MATCHED! The pipeline filtered out all results.")

# Optional: save result to S3 / Parquet
if matched_jobs > 0:
    # Save as multiple files (faster, better for large datasets)
    print("\nSaving results as partitioned parquet...")
    best_matches.write.mode("overwrite").parquet("s3a://ssg-sg-jobs/job_acra_matches_partitioned.parquet")

    # Save as single file (convenient for small datasets)
    print("Saving results as single parquet file...")
    best_matches.coalesce(1).write.mode("overwrite").parquet("s3a://ssg-sg-jobs/job_acra_matches_single.parquet")
    print(f"✅ Results saved successfully! ({matched_jobs:,} jobs matched)")
else:
    print("\n⚠️ Skipping save - no matches to write.")