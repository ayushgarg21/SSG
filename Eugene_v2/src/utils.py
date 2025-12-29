"""Utility functions for data I/O operations"""
import pandas as pd
from pathlib import Path


def read_dataframe(file_path: str) -> pd.DataFrame:
    """Read dataframe from CSV, Excel, or Parquet based on file extension"""
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(file_path)
    elif ext in ['.parquet', '.pq']:
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, .xls, .parquet, or .pq")


def write_dataframe(df: pd.DataFrame, file_path: str):
    """Write dataframe to CSV, Excel, or Parquet based on file extension"""
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        df.to_csv(file_path, index=False)
    elif ext in ['.xlsx', '.xls']:
        df.to_excel(file_path, index=False)
    elif ext in ['.parquet', '.pq']:
        df.to_parquet(file_path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv, .xlsx, .xls, .parquet, or .pq")
