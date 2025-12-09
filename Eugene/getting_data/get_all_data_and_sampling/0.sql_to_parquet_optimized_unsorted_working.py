import pandas as pd
import os
import sys
import logging
import gc
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, "/Users/eugene/Documents/py_utils")
from jt_golden_utility import execute_read_query, create_db_engine, execute_insert_update, setup_logger

# Setup logger in main process only
logger = setup_logger(log_dir='logs', log_file='log', retention_days=30)
logger.setLevel(logging.INFO)

# Create engine with connection pooling disabled and timeout settings
RRdb = create_db_engine("RR", logger, "jobtech_data_singapore2")

# Add retry logic for database queries
def read_sql_with_retry(query, engine, max_retries=3, initial_delay=5):
    """Execute SQL query with exponential backoff retry on connection errors"""
    for attempt in range(max_retries):
        try:
            # Test connection before executing query
            try:
                engine.connect().close()
            except:
                engine.dispose()

            return pd.read_sql(query, engine)
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                print(f"⚠️  Database error (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
                print(f"   Retrying in {delay} seconds...")
                logger.warning(f"Database query failed (attempt {attempt + 1}/{max_retries}): {error_msg}")
                time.sleep(delay)
                # Dispose connection pool to force fresh connections
                try:
                    engine.dispose()
                except:
                    pass
            else:
                logger.error(f"Database query failed after {max_retries} attempts: {error_msg}")
                raise

# -----------------------------
# 1. Configuration
# -----------------------------
chunk_size = 100_000        # smaller chunks for better memory management
rows_per_file = 1_000_000  # rows per Parquet file
output_dir = "parquet_jobs"
max_workers = min(4, cpu_count())  # limit to 4 workers or CPU count
max_pending_files = 2      # limit concurrent file writes to prevent memory buildup
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# 2. Worker function to write parquet files
# -----------------------------
def write_parquet_file(args):
    """Write accumulated chunks to a single parquet file in worker process"""
    file_idx, chunks_data, output_dir = args
    parquet_file = os.path.join(output_dir, f"jobs_{file_idx}.parquet")

    try:
        # Concatenate all chunks for this file
        combined_df = pd.concat(chunks_data, ignore_index=True)

        # Fix mixed-type columns that cause parquet conversion errors
        # Convert datetime columns to strings to handle inconsistent types
        date_columns = [
            'date_posted', 'date_posted_actual_assume', 'date_expiring',
            'date_expiring_actual_assume', 'date_last_seen',
            'date_first_crawled', 'date_first_scraped', 'time_updated'
        ]
        for col in date_columns:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype(str)

        # Write to parquet (compression happens here - the CPU-intensive part)
        combined_df.to_parquet(
            parquet_file,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        rows_written = len(combined_df)

        # Clean up memory in worker process
        del combined_df
        del chunks_data
        gc.collect()

        return file_idx, rows_written, "success"
    except Exception as e:
        return file_idx, 0, f"error: {e}"

# -----------------------------
# 3. Main processing with indexed pagination
# -----------------------------
if __name__ == '__main__':
    print(f"Starting parallel export with {max_workers} workers (max {max_pending_files} files queued)...")

    file_counter = 0
    rows_in_current_file = 0
    current_file_chunks = []
    chunk_counter = 0
    total_rows_processed = 0

    # Get min and max job_id to know the range
    print("Getting job_id range...")
    range_query = "SELECT MIN(job_id) as min_id, MAX(job_id) as max_id FROM jobtech_data_singapore2.jobtech_job"
    id_range = read_sql_with_retry(range_query, RRdb)
    min_id = id_range['min_id'].iloc[0]
    max_id = id_range['max_id'].iloc[0]
    print(f"job_id range: {min_id} to {max_id}")

    with Pool(processes=max_workers) as pool:
        pending_tasks = []

        # Paginate using job_id (indexed) instead of OFFSET
        print("Starting to read chunks from database using indexed pagination...")
        last_id = min_id

        first_iteration = True
        while True:
            # Fetch chunk using indexed WHERE clause (much faster than OFFSET)
            if first_iteration:
                where_clause = f"WHERE job_id >= '{last_id}'"
                first_iteration = False
            else:
                where_clause = f"WHERE job_id > '{last_id}'"

            chunk_query = f"""
            SELECT job_id, source_id, job_url, job_title, job_description, raw_position,
                   raw_years_experience, company_id, job_specialisation, raw_location_address,
                   location_coordinates, country_code, location_state, location_city, wage_min,
                   wage_max, wage_currency, wage_rate, date_posted, date_posted_actual_assume,
                   date_expiring, date_expiring_actual_assume, date_last_seen, employment_type,
                   date_first_crawled, date_first_scraped, time_updated
            FROM jobtech_data_singapore2.jobtech_job
            {where_clause}
            ORDER BY job_id
            LIMIT {chunk_size}
            """

            print(f"Fetching chunk starting from job_id {last_id[:20]}...")

            # Time the database read (with retry logic)
            read_start = time.time()
            chunk = read_sql_with_retry(chunk_query, RRdb)
            read_time = time.time() - read_start

            # If no more rows, we're done
            if len(chunk) == 0:
                print("No more rows to fetch.")
                break

            chunk_counter += 1
            total_rows_processed += len(chunk)
            print(f"Chunk {chunk_counter}: {len(chunk):,} rows | Read time: {read_time:.2f}s | Total: {total_rows_processed:,}")

            # Update last_id for next iteration (use '>' to avoid duplicates)
            last_id = chunk['job_id'].iloc[-1]

            # Check if we need to start a new file
            if rows_in_current_file + len(chunk) > rows_per_file and rows_in_current_file > 0:
                # Submit current file to worker pool
                submit_start = time.time()
                task = pool.apply_async(write_parquet_file,
                                       [(file_counter, current_file_chunks, output_dir)])
                pending_tasks.append((file_counter, rows_in_current_file, task, submit_start))

                logger.info(f"Queued file {file_counter} for writing ({rows_in_current_file:,} rows)")

                # Move to next file
                file_counter += 1
                current_file_chunks = []
                rows_in_current_file = 0

                # CRITICAL: Block if too many pending tasks to prevent memory buildup
                while len(pending_tasks) >= max_pending_files:
                    # Wait for oldest task to complete
                    f_idx, f_rows, task, task_start = pending_tasks[0]
                    wait_start = time.time()
                    result = task.get()  # Blocks until complete
                    total_time = time.time() - task_start
                    wait_time = time.time() - wait_start
                    logger.info(f"File {result[0]} completed: {result[1]:,} rows - {result[2]} | Write time: {total_time:.2f}s (waited: {wait_time:.2f}s)")
                    pending_tasks.pop(0)
                    gc.collect()  # Clean up after task completes

            # Add chunk to current file
            current_file_chunks.append(chunk)
            rows_in_current_file += len(chunk)

        # Submit the last file if there are remaining chunks
        if current_file_chunks:
            submit_start = time.time()
            task = pool.apply_async(write_parquet_file,
                                   [(file_counter, current_file_chunks, output_dir)])
            pending_tasks.append((file_counter, rows_in_current_file, task, submit_start))
            logger.info(f"Queued final file {file_counter} for writing ({rows_in_current_file:,} rows)")

        # Wait for all remaining tasks to complete
        print(f"\nWaiting for {len(pending_tasks)} remaining files to finish...")
        for i, (f_idx, f_rows, task, task_start) in enumerate(pending_tasks, 1):
            print(f"Waiting for file {f_idx} ({i}/{len(pending_tasks)})...")
            wait_start = time.time()
            result = task.get()
            total_time = time.time() - task_start
            wait_time = time.time() - wait_start
            logger.info(f"File {result[0]} completed: {result[1]:,} rows - {result[2]} | Write time: {total_time:.2f}s (waited: {wait_time:.2f}s)")

        # Pool cleanup happens automatically with context manager

    print(f"\n✓ Successfully processed {chunk_counter} chunks ({total_rows_processed:,} rows)")
    print(f"✓ Created {file_counter + 1} parquet files in '{output_dir}/' directory")
