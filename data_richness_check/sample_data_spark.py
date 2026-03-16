from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path
import time

# Start timing
start_time = time.time()

# Configuration
minio_endpoint = "http://192.168.18.51:9000"
minio_access_key = "minioadmin"
minio_secret_key = "minioadmin123"
s3_bucket = "s3a://ssg-sg-jobs"
output_file = "s3a://ssg-sg-jobs/sampled_jobs_spark.parquet"
sample_frac = 0.05  # 5% sample
random_state = 42
group_by_cols = ["source_id", "year"]

print("Initializing Spark Session...")
print(f"Spark Master: spark://192.168.18.51:7077")
print(f"MinIO Endpoint: {minio_endpoint}")
print(f"S3 Bucket: {s3_bucket}")

# Initialize Spark Session with S3A dependencies
# These packages provide the S3A file system implementation for MinIO
spark = SparkSession.builder \
    .appName("StratifiedSampling") \
    .master("spark://192.168.18.51:7077") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.cores", "4") \
    .config("spark.cores.max", "8") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.driver.host", "192.168.18.79") \
    .config("spark.driver.bindAddress", "192.168.18.79") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.parquet.columnarReaderBatchSize", "2048") \
    .config("spark.sql.files.maxPartitionBytes", "134217728") \
    .config("spark.sql.files.openCostInBytes", "134217728") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.4.2,com.amazonaws:aws-java-sdk-bundle:1.12.720") \
    .getOrCreate()

# Configure MinIO/S3 access
print("\nConfiguring MinIO S3 access...")
conf = spark._jsc.hadoopConfiguration()
conf.set("fs.s3a.endpoint", minio_endpoint)
conf.set("fs.s3a.access.key", minio_access_key)
conf.set("fs.s3a.secret.key", minio_secret_key)
conf.set("fs.s3a.path.style.access", "true")
conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
conf.set("spark.hadoop.fs.s3a.aws.credentials.provider",
         "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")

# Connection settings - set as strings to avoid NumberFormatException
conf.set("fs.s3a.connection.timeout", "60000")
conf.set("fs.s3a.connection.establish.timeout", "60000")
conf.set("fs.s3a.attempts.maximum", "3")
conf.set("fs.s3a.retry.limit", "3")
conf.set("fs.s3a.threads.max", "50")

# Memory optimization for S3 reads
conf.set("fs.s3a.readahead.range", "65536")  # 64KB readahead (smaller chunks)
conf.set("fs.s3a.vectored.read.min.seek.size", "4096")  # Reduce vectored read size
conf.set("fs.s3a.vectored.read.max.merged.size", "1048576")  # 1MB max merged reads

print(f"Spark version: {spark.version}")
print(f"Spark master: {spark.sparkContext.master}")

# Read all parquet files from MinIO
print(f"\nReading parquet files from {s3_bucket}...")
df = spark.read.parquet(s3_bucket)

print(f"Total rows: {df.count():,}")
print(f"Partitions: {df.rdd.getNumPartitions()}")

# Add year column if not present
if 'year' not in df.columns and 'date_posted' in df.columns:
    print("Extracting year from date_posted...")
    df = df.withColumn('year', F.year(F.col('date_posted')))

# Show schema
print("\nSchema:")
df.printSchema()

# Create stratification column (concatenate all group_by columns)
print(f"\nPerforming stratified sampling by {group_by_cols}...")
df = df.withColumn(
    '_strata',
    F.concat_ws('_', *[F.col(c).cast('string') for c in group_by_cols])
)

# Calculate sample fractions for each stratum (all same fraction)
strata = df.select('_strata').distinct().collect()
fractions = {row['_strata']: sample_frac for row in strata}

print(f"Found {len(fractions)} unique strata")

# Perform stratified sampling
sampled_df = df.sampleBy('_strata', fractions, seed=random_state)

# Drop the helper column
sampled_df = sampled_df.drop('_strata')

sampled_count = sampled_df.count()
total_count = df.count()

print(f"\n{'='*60}")
print(f"Total rows: {total_count:,}")
print(f"Sampled rows: {sampled_count:,}")
print(f"Sample rate: {sampled_count/total_count*100:.2f}%")
print(f"{'='*60}\n")

# Save to parquet
print(f"Saving to {output_file}...")
sampled_df.write.mode('overwrite').parquet(output_file)

print(f"✅ Done! Saved {sampled_count:,} rows to {output_file}")

# Show sample of results
print("\nSample of results:")
sampled_df.show(10, truncate=False)

# End timing
end_time = time.time()
elapsed_time = end_time - start_time

print(f"\n{'='*60}")
print(f"⏱️  SPARK TOTAL TIME: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
print(f"{'='*60}")

# Stop Spark
spark.stop()
