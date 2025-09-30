#!/usr/bin/env python3

import argparse
import os
import sys
import gc
import gzip
import shutil
import re
from typing import Optional, Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
from urllib.parse import urljoin

CRITEO_PAGE_URL = "https://ailab.criteo.com/criteo-uplift-prediction-dataset/"


def resolve_download_url(url: str) -> str:
	"""Return a direct .csv.gz URL. If `url` is a page, try to find a .csv.gz link."""
	if not url:
		raise ValueError("Empty URL provided")
	if url.lower().endswith(".gz"):
		return url
	# Fetch page and search for a .csv.gz link
	r = requests.get(url, timeout=30)
	r.raise_for_status()
	matches = re.findall(r"href=[\'\"]([^\'\"]+\.csv\.gz)[\'\"]", r.text, flags=re.IGNORECASE)
	if not matches:
		raise ValueError("Could not find a .csv.gz link on the provided page: " + url)
	return urljoin(url, matches[0])


def download_file(url: str, dest_path: str, timeout: int = 30) -> None:
	os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
	with requests.get(url, stream=True, timeout=timeout) as r:
		r.raise_for_status()
		total = int(r.headers.get("content-length", 0))
		progress = tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path))
		with open(dest_path, "wb") as f:
			for chunk in r.iter_content(chunk_size=1024 * 1024):
				if chunk:
					f.write(chunk)
					progress.update(len(chunk))
		progress.close()


def extract_gz(gz_path: str, csv_path: str) -> None:
	os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
	with gzip.open(gz_path, "rb") as f_in, open(csv_path, "wb") as f_out:
		shutil.copyfileobj(f_in, f_out)


def ensure_csv(csv_path: str, gz_path: Optional[str], download_url: Optional[str]) -> str:
	"""Ensure a CSV exists. If missing, try .gz then download.

	Returns the path to the CSV to use.
	"""
	if os.path.exists(csv_path):
		print(f"Found CSV: {csv_path}")
		return csv_path

	# Determine gz path if not provided
	resolved_url: Optional[str] = None
	if not gz_path:
		if download_url and download_url.strip():
			resolved_url = resolve_download_url(download_url.strip())
			filename = os.path.basename(resolved_url)
			if not filename:
				filename = os.path.basename(csv_path) + ".gz"
			gz_path = os.path.join(os.path.dirname(csv_path), filename)
		else:
			gz_path = csv_path + ".gz"

	if os.path.exists(gz_path):
		print(f"Extracting existing .gz: {gz_path} -> {csv_path}")
		extract_gz(gz_path, csv_path)
		return csv_path

	if download_url:
		if not resolved_url:
			resolved_url = resolve_download_url(download_url.strip())
		print(f"Downloading .gz from {resolved_url} -> {gz_path}")
		download_file(resolved_url, gz_path)
		print(f"Extracting: {gz_path} -> {csv_path}")
		extract_gz(gz_path, csv_path)
		return csv_path

	raise FileNotFoundError(
		f"CSV not found: {csv_path}. Provide the CSV, or pass --download-url, or place a .gz at {gz_path}."
	)

	

def infer_and_optimize_dtypes(chunk: pd.DataFrame) -> pd.DataFrame:
	"""Best-effort dtype optimization for a chunk.

	- Try integers where possible
	- Try floats where possible
	- Replace empty strings with NA
	"""
	for col in chunk.columns:
		# If already numeric, try downcast fast path
		if pd.api.types.is_integer_dtype(chunk[col]):
			chunk[col] = pd.to_numeric(chunk[col], downcast="integer")
			continue
		if pd.api.types.is_float_dtype(chunk[col]):
			chunk[col] = pd.to_numeric(chunk[col], downcast="float")
			continue

		# Work as string to test patterns
		series = chunk[col].astype("string")
		# Empty strings to NA
		series = series.replace({"": pd.NA})

		# Try int
		is_int_like = series.str.match(r"^[+-]?\d+$").fillna(False).all()
		if is_int_like:
			chunk[col] = pd.to_numeric(series, downcast="integer")
			continue

		# Try float
		try:
			chunk[col] = pd.to_numeric(series, errors="raise", downcast="float")
		except Exception:
			chunk[col] = series

	return chunk


def convert_csv_to_parquet(csv_path: str, parquet_path: str, chunksize: int = 1_000_000, overwrite: bool = False) -> None:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	if os.path.exists(parquet_path) and not overwrite:
		print(f"Parquet already exists, skipping: {parquet_path}")
		return

	print(f"Converting CSV to Parquet...\n  csv: {csv_path}\n  parquet: {parquet_path}\n  chunksize: {chunksize}")
	os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)

	csv_iter = pd.read_csv(
		csv_path,
		chunksize=chunksize,
		low_memory=True,
		dtype="unicode",
	)

	writer: Optional[pq.ParquetWriter] = None
	total_rows = 0

	for i, chunk in enumerate(csv_iter):
		chunk = infer_and_optimize_dtypes(chunk)
		table = pa.Table.from_pandas(chunk, preserve_index=False)
		if writer is None:
			writer = pq.ParquetWriter(parquet_path, table.schema, compression="snappy")
		writer.write_table(table)
		total_rows += len(chunk)
		print(f"Wrote chunk {i+1:,}, rows so far: {total_rows:,}")
		del chunk, table
		gc.collect()

	if writer is not None:
		writer.close()
		print(f"Done. Total rows written: {total_rows:,}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="CSV → Parquet converter with optional download & extract")
	parser.add_argument("csv", help="Path to input CSV file (will be created if downloaded/extracted)")
	parser.add_argument("parquet", help="Path to output Parquet file")
	parser.add_argument("--chunksize", type=int, default=1_000_000, help="Rows per chunk (default: 1,000,000)")
	parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Parquet file")
	parser.add_argument("--download-url", dest="download_url", default=CRITEO_PAGE_URL, help="URL to .csv.gz or a page containing it (default: Criteo dataset page)")
	parser.add_argument("--gz-path", dest="gz_path", default=None, help="Path to save/read the .gz (defaults near CSV)")
	return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
	args = parse_args(argv)
	csv_path = ensure_csv(args.csv, args.gz_path, args.download_url)
	convert_csv_to_parquet(csv_path, args.parquet, chunksize=args.chunksize, overwrite=args.overwrite)


if __name__ == "__main__":
	main()
