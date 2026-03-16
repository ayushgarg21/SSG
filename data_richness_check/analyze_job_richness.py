"""
Job Description Richness Analysis
Analyzes job descriptions from MinIO for text richness metrics
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import time

# Start timing
start_time = time.time()

# ---------------------------
# Configuration
# ---------------------------
minio_endpoint = "http://192.168.18.51:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin123"
input_file = "s3a://ssg-sg-jobs/sampled_jobs_spark.parquet"

print("="*60)
print("Job Description Richness Analysis")
print("="*60)
print(f"MinIO Endpoint: {minio_endpoint}")
print(f"Input File: {input_file}")
print()

# ---------------------------
# 1. Initialize Spark Session
# ---------------------------
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("JobDescription_Richness") \
    .master("spark://192.168.18.51:7077") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.cores", "4") \
    .config("spark.cores.max", "8") \
    .config("spark.driver.host", "192.168.18.79") \
    .config("spark.driver.bindAddress", "192.168.18.79") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.2,com.amazonaws:aws-java-sdk-bundle:1.12.720") \
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
print()

# ---------------------------
# 2. Read Input DataFrame
# ---------------------------
print(f"Reading data from {input_file}...")
# df = spark.read.parquet(input_file).limit(10000)
df = spark.read.parquet(input_file)

total_rows = df.count()
print(f"Total rows (sampled): {total_rows:,}")
print(f"Columns: {', '.join(df.columns[:10])}...")
print()

# Check for job_description column
if "job_description" not in df.columns:
    print("❌ Error: 'job_description' column not found!")
    print(f"Available columns: {', '.join(df.columns)}")
    spark.stop()
    exit(1)

# ---------------------------
# 3. Text Cleaning + Tokenization
# ---------------------------
print("Step 1: Text cleaning and tokenization...")
df = df.withColumn(
    "clean_text",
    F.lower(F.regexp_replace(F.col("job_description"), "[^a-zA-Z\\s]", " "))
)

df = df.withColumn("tokens", F.split(F.col("clean_text"), "\\s+"))
df = df.withColumn("word_count", F.size("tokens"))

df = df.withColumn("unique_tokens", F.array_distinct("tokens"))
df = df.withColumn("unique_count", F.size("unique_tokens"))

# ---------------------------
# 4. Root TTR (removed)
# ---------------------------

# ---------------------------
# 5. Shannon Entropy
# ---------------------------
print("Step 3: Calculating Shannon Entropy...")
token_df = df.select("job_description", "word_count", "tokens").withColumn("token", F.explode("tokens"))

freq_df = token_df.groupBy("job_description", "word_count", "token").count()
freq_df = freq_df.withColumn("p", F.col("count") / F.col("word_count"))
freq_df = freq_df.withColumn("entropy_component", -F.col("p") * F.log2(F.col("p")))
entropy_df = freq_df.groupBy("job_description").agg(F.sum("entropy_component").alias("entropy"))

df = df.join(entropy_df, on="job_description", how="left")

# ---------------------------
# 6. Composite Richness Score
# ---------------------------
print("Step 4: Computing richness scores...")
df = df.withColumn(
    "length_score", F.col("word_count") / 300
)

df = df.withColumn(
    "richness_score",
    0.5 * F.col("length_score") +
    0.5 * F.col("entropy")
)

# ---------------------------
# 7. Define Useful Threshold
# ---------------------------
print("Step 5: Classifying useful vs not useful...")
df = df.withColumn(
    "is_useful",
    F.when(
        (F.col("word_count") > 150) &
        (F.col("entropy") >= 4.5),
        1
    ).otherwise(0)
)

# ---------------------------
# 8. Compute Statistics
# ---------------------------
print()
print("="*60)
print("RESULTS - Overall Statistics")
print("="*60)

stats = df.agg(
    F.count("*").alias("total_rows"),
    F.sum("is_useful").alias("useful_count"),
    (F.avg("is_useful") * 100).alias("percent_useful"),
    F.avg("word_count").alias("avg_word_count"),
    F.avg("entropy").alias("avg_entropy"),
    F.avg("richness_score").alias("avg_richness_score")
)

stats_result = stats.collect()[0]
print(f"Total Rows:              {stats_result['total_rows']:,}")
print(f"Useful Count:            {stats_result['useful_count']:,}")
print(f"Percent Useful:          {stats_result['percent_useful']:.2f}%")
print(f"Avg Word Count:          {stats_result['avg_word_count']:.1f}")
print(f"Avg Entropy:             {stats_result['avg_entropy']:.2f}")
print(f"Avg Richness Score:      {stats_result['avg_richness_score']:.3f}")
print("="*60)
print()

# ---------------------------
# 9. Inspect Sample
# ---------------------------
print("Sample of Results (Top 10 by Richness Score):")
print("-"*60)
df.select(
    "job_description",
    "word_count",
    "entropy",
    "richness_score",
    "is_useful"
).orderBy(F.desc("richness_score")).show(10, truncate=100)

# ---------------------------
# 10. Distribution Analysis
# ---------------------------
print("\nDistribution by Usefulness:")
print("-"*60)
df.groupBy("is_useful").agg(
    F.count("*").alias("count"),
    F.avg("word_count").alias("avg_word_count"),
    F.avg("entropy").alias("avg_entropy"),
    F.avg("richness_score").alias("avg_richness_score")
).show()

# ---------------------------
# 11. Save Results
# ---------------------------
print("\nSaving enriched data with metrics...")
output_path = "s3a://ssg-sg-jobs/enriched_jobs_with_richness.parquet"

# Select columns to save (original data + all metrics)
output_df = df.select(
    "job_id",
    "source_id",
    "job_url",
    "job_title",
    "job_description",
    "company_id",
    "job_specialisation",
    "raw_location_address",
    "country_code",
    "location_state",
    "location_city",
    "employment_type",
    "word_count",
    "unique_count",
    "entropy",
    "richness_score",
    "is_useful"
)

output_df.write.mode("overwrite").parquet(output_path)
print(f"✅ Saved to: {output_path}")

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

print()
print("="*60)
print(f"⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print("="*60)

# Stop Spark
spark.stop()
