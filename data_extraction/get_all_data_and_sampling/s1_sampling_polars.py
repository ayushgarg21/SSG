import sys
import logging
from pathlib import Path
import argparse
import re
import html
import pandas as pd
import polars as pl
from tqdm import tqdm

# keep tqdm pandas compatibility if used elsewhere
tqdm.pandas()

# Import your custom module for logging if available, otherwise fall back
try:
	sys.path.insert(0, "/Users/eugene/Documents/py_utils")
	from jt_golden_utility import setup_logger
	logger = setup_logger(log_dir='logs', log_file='log', retention_days=30)
except Exception:
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger(__name__)


def clean_excel_string(value):
	"""Remove illegal characters for Excel cells.

	First unescapes HTML entities, then removes control characters.
	Excel doesn't allow control characters (ASCII 0-31).
	"""
	if not isinstance(value, str):
		return value

	# First unescape HTML entities (e.g., &lt; becomes <, &#11; becomes vertical tab)
	value = html.unescape(value)

	# Remove control characters (0x00-0x1F) and replace with space
	# This includes vertical tabs, form feeds, etc.
	cleaned = re.sub(r'[\x00-\x1F]+', ' ', value)

	# Collapse multiple spaces
	cleaned = re.sub(r'\s+', ' ', cleaned).strip()

	return cleaned


def safe_to_string(val):
	"""Safely convert value to string, handling encoding errors."""
	if pd.isna(val):
		return ''
	if isinstance(val, bytes):
		# Decode bytes with error handling
		return val.decode('utf-8', errors='replace')
	try:
		return str(val)
	except:
		# If str() fails, return empty string
		return ''


def clean_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
	"""Clean all string columns in dataframe for Excel export."""
	df = df.copy()
	for col in df.columns:
		if df[col].dtype == 'object':
			# Safely convert to string, handling encoding errors
			df[col] = df[col].apply(safe_to_string)
			# First, unescape HTML entities
			df[col] = df[col].str.replace(r'&\w+;|&#\d+;', lambda m: html.unescape(m.group(0)), regex=True)
			# Remove control characters and replace with space
			df[col] = df[col].str.replace(r'[\x00-\x1F]+', ' ', regex=True)
			# Collapse multiple spaces
			df[col] = df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
	return df


def stratified_sample_polars(
	df: pl.DataFrame,
	group_by_cols: list[str],
	sample_frac: float = 0.05,
	random_state: int = 42,
	date_column: str = "date_posted"
) -> pl.DataFrame:
	"""Perform stratified sampling using Polars.

	Args:
		df: Polars DataFrame to sample from
		group_by_cols: List of column names to group by for stratification
		sample_frac: Fraction to sample from each group (default: 0.05 = 5%)
		random_state: Random seed for reproducibility
		date_column: Column name containing dates (default: "date_posted")

	Returns:
		Sampled Polars DataFrame

	Example:
		# Sample 5% from each source_id and year combination
		sampled = stratified_sample_polars(
			df,
			group_by_cols=["source_id", "year"],
			sample_frac=0.05,
			random_state=42
		)
	"""
	# Create a copy to avoid modifying original
	df = df.clone()

	# Add year column if needed and not present
	if 'year' in group_by_cols and 'year' not in df.columns:
		if date_column in df.columns:
			# Check if already datetime type, otherwise parse it
			dtype = df[date_column].dtype
			if dtype in [pl.Date, pl.Datetime]:
				df = df.with_columns(
					pl.col(date_column).dt.year().alias('year')
				)
			else:
				df = df.with_columns(
					pl.col(date_column).str.to_datetime(strict=False).dt.year().alias('year')
				)
		else:
			raise ValueError(f"Cannot create 'year' column: '{date_column}' not found in dataframe")

	# Verify all group_by columns exist
	missing_cols = [col for col in group_by_cols if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Columns not found in dataframe: {missing_cols}")

	# Perform stratified sampling by iterating over groups
	# This is more reliable than map_groups in recent Polars versions
	sampled_frames = []

	for group_values, group_df in df.group_by(group_by_cols, maintain_order=True):
		# Sample from this group
		n_samples = max(1, int(len(group_df) * sample_frac))
		sampled_group = group_df.sample(n=n_samples, seed=random_state, with_replacement=False)
		sampled_frames.append(sampled_group)

	# Concatenate all sampled groups
	if sampled_frames:
		df_sampled = pl.concat(sampled_frames, how='vertical')
	else:
		df_sampled = df.head(0)  # Empty dataframe with same schema

	return df_sampled


def stratified_sample(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
	"""Perform stratified sampling using pandas groupby.

	Behavior:
	- Group by `source_id` and year of `date_posted`.
	- Sample 5% from each group.
	"""
	df = df.copy()
	# ensure date_posted is datetime
	df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
	df['year'] = df['date_posted'].dt.year

	# drop rows where job_description is null/empty
	df['job_description'] = df['job_description'].astype(object)
	mask_non_null = df['job_description'].notna() & (df['job_description'].astype(str).str.strip() != '')
	df = df[mask_non_null]

	if df.empty:
		return df

	sample_frac = 0.05  # 5%
	df_sampled = df.groupby(["source_id", "year"], group_keys=False).apply(
		lambda x: x.sample(frac=sample_frac, random_state=random_state),
		include_groups=False
	).reset_index(drop=True)

	return df_sampled


def process_parquet_dir_polars(
	parquet_dir: Path,
	output_path: Path,
	group_by_cols: list[str],
	sample_frac: float = 0.05,
	random_state: int = 42,
	limit_per_file: int | None = None,
	filter_null_descriptions: bool = True
):
	"""Process parquet files with Polars for better performance.

	Args:
		parquet_dir: Directory containing parquet files
		output_path: Output file path (.parquet, .csv, or .xlsx)
		group_by_cols: Columns to group by for stratified sampling
		sample_frac: Fraction to sample (default: 0.05 = 5%)
		random_state: Random seed for reproducibility
		limit_per_file: Optional row limit per file for testing
		filter_null_descriptions: Whether to filter null/empty job_description
	"""
	files = sorted(parquet_dir.glob('*.parquet'))
	if not files:
		logger.error('No parquet files found in %s', parquet_dir)
		return

	sampled_frames = []
	for p in files:
		logger.info('Reading %s', p)
		try:
			# Use Polars lazy API for efficient reading
			df = pl.scan_parquet(p)

			if limit_per_file:
				df = df.head(limit_per_file)

			# Collect to execute the lazy query
			df = df.collect()

		except Exception as e:
			logger.exception('Failed to read %s: %s', p, e)
			continue

		if 'job_description' not in df.columns:
			logger.warning('file %s has no job_description column; skipping', p)
			continue

		# Filter null/empty descriptions if requested
		if filter_null_descriptions:
			df = df.filter(
				pl.col('job_description').is_not_null() &
				(pl.col('job_description').str.strip_chars() != "")
			)

		if df.is_empty():
			logger.warning('No valid rows after filtering in %s', p)
			continue

		# Add year column if grouping by year
		if 'year' in group_by_cols and 'year' not in df.columns:
			if 'date_posted' in df.columns:
				df = df.with_columns(
					pl.col('date_posted').str.to_datetime(strict=False).dt.year().alias('year')
				)

		# Perform stratified sampling
		sampled = stratified_sample_polars(df, group_by_cols, sample_frac, random_state)
		sampled_frames.append(sampled)

	if not sampled_frames:
		logger.warning('No sampled frames produced')
		return

	# Concatenate all sampled frames
	out_df = pl.concat(sampled_frames, how='vertical')
	logger.info('Writing sampled output (%d rows) to %s', len(out_df), output_path)

	# Write based on file extension
	if output_path.suffix.lower() == '.parquet':
		out_df.write_parquet(output_path)
	elif output_path.suffix.lower() == '.csv':
		out_df.write_csv(output_path)
	else:
		# Convert to pandas for Excel export with cleaning
		logger.info('Converting to pandas for Excel export...')
		out_df_pd = out_df.to_pandas()
		out_df_pd = clean_dataframe_for_excel(out_df_pd)
		out_df_pd.to_excel(output_path, index=False)


def process_parquet_dir(parquet_dir: Path, output_path: Path, limit_per_file: int | None = None):
	"""Legacy pandas-based processing (kept for compatibility)."""
	files = sorted(parquet_dir.glob('*.parquet'))
	if not files:
		logger.error('No parquet files found in %s', parquet_dir)
		return

	sampled_frames = []
	for p in files:
		logger.info('Reading %s', p)
		try:
			df = pd.read_parquet(p)
		except Exception as e:
			logger.exception('Failed to read %s: %s', p, e)
			continue

		if limit_per_file:
			df = df.head(limit_per_file)

		if 'job_description' not in df.columns:
			logger.warning('file %s has no job_description column; skipping', p)
			continue

		sampled = stratified_sample(df)
		sampled_frames.append(sampled)

	if not sampled_frames:
		logger.warning('No sampled frames produced')
		return

	out_df = pd.concat(sampled_frames, ignore_index=True, sort=False)
	logger.info('Writing sampled output (%d rows) to %s', len(out_df), output_path)
	# choose format by suffix
	if output_path.suffix.lower() in ['.parquet']:
		out_df.to_parquet(output_path, index=False)
	elif output_path.suffix.lower() in ['.csv']:
		out_df.to_csv(output_path, index=False, encoding='utf-8', errors='replace')
	else:
		# Clean data before writing to Excel to remove illegal characters
		logger.info('Cleaning data for Excel export...')
		out_df = clean_dataframe_for_excel(out_df)
		out_df.to_excel(output_path, index=False)


def main():
	parser = argparse.ArgumentParser(description='Stratified sampling for parquet jobs')
	parser.add_argument('--parquet-dir', type=str, default='Eugene/parquet_jobs', help='Directory containing parquet files')
	parser.add_argument('--output', type=str, default='sampled_jobs.parquet', help='Output file path (.parquet, .csv, or .xlsx)')
	parser.add_argument('--limit', type=int, default=None, help='Optional: limit rows per file for quick test')
	parser.add_argument('--group-by', type=str, nargs='+', default=['source_id', 'year'], help='Columns to group by for stratification')
	parser.add_argument('--sample-frac', type=float, default=0.05, help='Sample fraction (default: 0.05 = 5%%)')
	parser.add_argument('--random-state', type=int, default=42, help='Random seed for reproducibility')
	parser.add_argument('--use-pandas', action='store_true', help='Use pandas instead of polars (slower)')

	args = parser.parse_args()
	parquet_dir = Path(args.parquet_dir)
	output_path = Path(args.output)

	if args.use_pandas:
		logger.info('Using pandas processing (legacy mode)')
		process_parquet_dir(parquet_dir, output_path, limit_per_file=args.limit)
	else:
		logger.info('Using Polars processing (recommended for large datasets)')
		process_parquet_dir_polars(
			parquet_dir=parquet_dir,
			output_path=output_path,
			group_by_cols=args.group_by,
			sample_frac=args.sample_frac,
			random_state=args.random_state,
			limit_per_file=args.limit
		)


if __name__ == '__main__':
	main()