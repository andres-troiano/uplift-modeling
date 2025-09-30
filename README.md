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
python src/etl/prepare_dataset.py \
  data/criteo-uplift-v2.1.csv \
  data/criteo-uplift-v2.1.parquet \
  --download-url "<PUT_THE_CSV_GZ_DOWNLOAD_URL_HERE>" \
  --chunksize 1000000
```

Notes:
- If you already have `data/criteo-uplift-v2.1.csv.gz`, the script will extract it.
- If you already have the CSV, it will skip download/extract and go straight to conversion.
- Use `--gz-path` to control where the `.gz` is saved/read.
- Use `--overwrite` to rebuild Parquet if it exists.

## Notebooks
Open the EDA notebook after conversion:

```bash
jupyter lab
```
Then open `notebooks/01_eda_uplift.ipynb`. It expects Parquet at `data/criteo-uplift-v2.1.parquet`.

## Project Structure
```
├── data/
├── notebooks/
├── src/
│   └── etl/
│       └── prepare_dataset.py
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
