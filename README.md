# Uplift Modeling — Criteo Dataset

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Data source: [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/). Please review the license and cite the dataset/paper accordingly.

### One-time download, extract, and CSV → Parquet conversion
Use the standalone script to download the `.csv.gz` (if not present), extract to CSV, and convert to Parquet.

```bash
# Default: uses the Criteo dataset page to auto-resolve the .csv.gz
python src/etl/prepare_dataset.py \
  data/criteo-uplift-v2.1.csv \
  data/criteo-uplift-v2.1.parquet \
  --chunksize 1000000

# Or specify a direct .csv.gz URL explicitly
python src/etl/prepare_dataset.py \
  data/criteo-uplift-v2.1.csv \
  data/criteo-uplift-v2.1.parquet \
  --download-url "https://example.com/criteo-uplift-v2.1.csv.gz" \
  --chunksize 1000000
```

Notes:
- If you already have `data/criteo-uplift-v2.1.csv.gz`, the script will extract it.
- If you already have the CSV, it will skip download/extract and go straight to conversion.
- Use `--gz-path` to control where the `.gz` is saved/read.
- Use `--overwrite` to rebuild Parquet if it exists.
 - Deduplication:
   - `--drop-duplicates-in-chunks` removes duplicates within each CSV chunk before writing.
   - `--final-dedupe` loads the written Parquet and drops duplicates globally (slower, more thorough).
   - `--dedupe-subset col1,col2` to define duplicates by subset of columns (default: all columns).
   - `--deduped-parquet path.parquet` to write the deduped output to a separate file.

## Notebooks
Open the EDA notebook after conversion:

```bash
jupyter lab
```
Then open `notebooks/01_eda_uplift.ipynb`. It expects Parquet at `data/criteo-uplift-v2.1.parquet`.

## Pipeline

Run steps via `pipeline.py`:

```bash
# Prepare data (download/extract/convert)
python pipeline.py --step prepare

# Balance diagnostics (KS tests, summaries, plots)
python pipeline.py --step balance --log-level INFO

# Full run (no interactive plots; images saved under reports/plots/)
python pipeline.py --step all --log-level INFO

# Optional: downsample after loading Parquet (e.g., 1,000,000 rows)
python pipeline.py --step all --sample-size 1000000 --log-level INFO

# Heterogeneity (CATE) only
python pipeline.py --step heterogeneity --sample-size 1000000 --log-level INFO
```

Artifacts:
- CSVs: `reports/balance/balance_results.csv`, `reports/balance/balance_summary.csv`
- CATE CSV: `reports/heterogeneity/cate_results.csv`
- Plots: `reports/plots/features/*.png`, `reports/plots/balance/*.png`, `reports/plots/heterogeneity/*.png`

## Project Structure
```
├── data/
├── notebooks/
├── reports/
│   ├── balance/
│   │   ├── balance_results.csv
│   │   └── balance_summary.csv
│   └── plots/
│       ├── features/
│       └── balance/
├── src/
│   ├── eda/
│   │   ├── balance.py
│   │   └── visualize.py
│   ├── estimation/
│   │   └── baseline.py
│   └── etl/
│       └── prepare_dataset.py
├── pipeline.py
├── requirements.txt
└── README.md
```

## Citation
If you use this dataset, please cite:

Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). A Large Scale Benchmark for Uplift Modeling. AdKDD 2018.

BibTeX:

```
@inproceedings{Diemert2018,
  author = {{Diemert Eustache, Betlei Artem} and Renaudin, Christophe and Massih-Reza, Amini},
  title = {A Large Scale Benchmark for Uplift Modeling},
  publisher = {ACM},
  booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom, August, 20, 2018},
  year = {2018}
}
```
