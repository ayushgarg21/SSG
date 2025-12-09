import pandas as pd
import sys
import logging
import time

sys.path.insert(0, "/Users/eugene/Documents/py_utils")
from jt_golden_utility import create_db_engine, setup_logger

logger = setup_logger(log_dir='logs', log_file='test_query_log', retention_days=30)
logger.setLevel(logging.INFO)

RRdb = create_db_engine("RR", logger, "jobtech_data_singapore2")

# First, let's test if the query returns anything with LIMIT
test_query = """
SELECT COUNT(*) as total_count
FROM jobtech_data_singapore2.jobtech_job
WHERE job_description IS NOT NULL
"""

print("Testing COUNT query...")
start_time = time.time()
try:
    count_df = pd.read_sql(test_query, RRdb)
    elapsed = time.time() - start_time
    print(f"✓ Query completed in {elapsed:.2f} seconds")
    print(f"Total rows with job_description: {count_df['total_count'].iloc[0]:,}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Now test reading a small chunk
print("\nTesting reading first 1000 rows...")
limited_query = """
SELECT job_id, source_id, job_url, job_title, job_description, raw_position, raw_years_experience, company_id, job_specialisation, raw_location_address, location_coordinates, country_code, location_state, location_city, wage_min, wage_max, wage_currency, wage_rate, date_posted, date_posted_actual_assume, date_expiring, date_expiring_actual_assume, date_last_seen, employment_type, date_first_crawled, date_first_scraped, time_updated
FROM jobtech_data_singapore2.jobtech_job
WHERE job_description IS NOT NULL
LIMIT 1000
"""

start_time = time.time()
try:
    test_df = pd.read_sql(limited_query, RRdb)
    elapsed = time.time() - start_time
    print(f"✓ Read {len(test_df):,} rows in {elapsed:.2f} seconds")
    print(f"✓ Columns: {len(test_df.columns)}")
    print(f"✓ Memory usage: {test_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)

# Test chunked reading with small chunksize
print("\nTesting chunked reading (3 chunks of 100 rows)...")
chunk_query = """
SELECT job_id, job_title, job_description
FROM jobtech_data_singapore2.jobtech_job
WHERE job_description IS NOT NULL
LIMIT 300
"""

start_time = time.time()
try:
    chunk_count = 0
    for chunk in pd.read_sql(chunk_query, RRdb, chunksize=100):
        chunk_count += 1
        print(f"  Chunk {chunk_count}: {len(chunk)} rows")
        if chunk_count >= 3:
            break
    elapsed = time.time() - start_time
    print(f"✓ Chunked reading works! ({elapsed:.2f} seconds)")
except Exception as e:
    print(f"✗ Error in chunked reading: {e}")
    exit(1)

print("\n✓ All tests passed! The query and chunked reading work fine.")
print("The issue might be:")
print("  1. Large dataset taking time to start streaming")
print("  2. Multiprocessing creating multiple database connections")
print("  3. Database server being slow to respond")
