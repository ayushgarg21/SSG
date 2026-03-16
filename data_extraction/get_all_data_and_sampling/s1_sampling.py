import sys
import logging
from pathlib import Path
import argparse
import re
import html
import pandas as pd
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


def process_parquet_dir(parquet_dir: Path, output_path: Path, limit_per_file: int | None = None):
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
	parser.add_argument('--output', type=str, default='sampled_jobs.xlsx', help='Output file path (csv or parquet)')
	parser.add_argument('--limit', type=int, default=None, help='Optional: limit rows per file for quick test')

	args = parser.parse_args()
	parquet_dir = Path(args.parquet_dir)
	output_path = Path(args.output)

	process_parquet_dir(parquet_dir, output_path, limit_per_file=args.limit)


if __name__ == '__main__':
	main()