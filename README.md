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

# Propensity methods (IPW + Matching)
python pipeline.py --step propensity --sample-size 1000000 --log-level INFO

# Uplift models (T-/X-learner, optional Causal Forest)
python pipeline.py --step uplift --sample-size 1000000 --log-level INFO
```

Artifacts:
- CSVs: `reports/balance/balance_results.csv`, `reports/balance/balance_summary.csv`
- CATE CSV: `reports/heterogeneity/cate_results.csv`
- Propensity CSV: `reports/propensity/propensity_results.csv`
- Uplift: per-user scores `reports/uplift/uplift_scores_<method>.parquet`, summary `reports/uplift/uplift_results.csv`
- Plots: `reports/plots/features/*.png`, `reports/plots/balance/*.png`, `reports/plots/heterogeneity/*.png`, `reports/plots/uplift/*.png`

### Heterogeneity (CATE) details

- Binning: features are split into quartiles (or fewer bins if low cardinality). Numeric bin labels show interval endpoints rounded to 1 decimal (e.g., `(0.1, 2.3]`).
- Estimation: uplift per bin is computed via difference-in-means (probability difference) with 95% Wald confidence intervals.
- Outputs:
  - CSV `reports/heterogeneity/cate_results.csv` includes: `feature`, `bin_label`, `bin_index`, `n_treat`, `n_control`, `p_treat`, `p_control`, `uplift`, `ci_low`, `ci_high` (floats rounded to 4 decimals).
  - Plots under `reports/plots/heterogeneity/` show uplift by bin with error bars (95% CI) and grids enabled.

### Propensity methods

- Propensity score estimation via logistic regression (optionally XGBoost if installed).
- Methods implemented:
  - IPW: inverse probability weighting ATE with robust SE and 95% CI.
  - Matching: nearest-neighbor (k=1 by default) on standardized features, ATE with SE and 95% CI.
- Results saved to `reports/propensity/propensity_results.csv`.

### Uplift modeling

- T-Learner, X-Learner (optionally Causal Forest via CausalML if available).
- Outputs:
  - Per-user uplift scores: `reports/uplift/uplift_scores_<method>.parquet`.
  - Curves: Uplift and Qini under `reports/plots/uplift/`.
  - Summary: `reports/uplift/uplift_results.csv` with:
    - `qini`, `uplift_auc` (model comparison)
    - `incr_conv_top10`, `incr_conv_top20`, `incr_conv_top30` — incremental conversions when targeting the top 10/20/30% by predicted uplift. Computed as `N_treat * (p_treat - p_ctrl)` within each top-k% subgroup.

Business interpretation: “If I only target the top-k% highest uplift users, how many extra conversions do I gain over control?” This provides an actionable KPI alongside Qini/AUC.

## Project Structure
```
├── data/
├── notebooks/
├── reports/
│   ├── balance/
│   │   ├── balance_results.csv
│   │   └── balance_summary.csv
│   ├── heterogeneity/
│   │   └── cate_results.csv
│   └── plots/
│       ├── features/
│       ├── balance/
│       ├── heterogeneity/
│       └── uplift/
├── src/
│   ├── eda/
│   │   ├── balance.py
│   │   └── visualize.py
│   ├── estimation/
│   │   ├── baseline.py
│   │   ├── heterogeneity.py
│   │   ├── propensity.py
│   │   └── uplift.py
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
