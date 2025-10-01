#!/usr/bin/env python3
"""
Uplift Modeling Pipeline
========================

Run this script to execute different steps of the project:

    python pipeline.py --step prepare
    python pipeline.py --step balance
    python pipeline.py --step baseline
    python pipeline.py --step propensity
    python pipeline.py --step uplift
    python pipeline.py --step all
"""

import argparse
import os
import pandas as pd
import logging

# ETL (dataset preparation)
from src.etl.prepare_dataset import ensure_csv, convert_csv_to_parquet

# Balance diagnostics + visualization
from src.eda.balance import check_balance, summarize_balance
from src.eda.visualize import (
    plot_feature_distributions,
    plot_balance_distributions,
)

# Baseline causal estimation
from src.estimation.baseline import diff_in_means, logistic_adjusted_ate
logger = logging.getLogger(__name__)
# Propensity & Uplift (placeholders for now)
# from src.estimation.propensity import run_ipw, run_matching
# from src.estimation.uplift import run_uplift_models


# -----------------------
# Config
# -----------------------
RAW_CSV = "data/criteo-uplift-v2.1.csv"
RAW_PARQUET = "data/criteo-uplift-v2.1.parquet"

TREATMENT = "treatment"
OUTCOME = "conversion"
FEATURES = [f"f{i}" for i in range(12)]


# -----------------------
# Pipeline Steps
# -----------------------

def run_prepare():
    """Download, extract and convert dataset to Parquet."""
    logger.info("📥 Preparing dataset...")
    csv_path = ensure_csv(RAW_CSV, gz_path=None, download_url=None)
    convert_csv_to_parquet(csv_path, RAW_PARQUET, chunksize=1_000_000)
    logger.info("✅ Dataset ready at: %s", RAW_PARQUET)


def run_balance(df):
    """Run balance diagnostics + visualizations."""
    logger.info("🔍 Running balance diagnostics...")
    balance_df = check_balance(df, tr_col=TREATMENT, features=FEATURES)
    summary = summarize_balance(balance_df)
    logger.info("Balance summary (head):\n%s", summary.head().to_string())
    # Persist results under reports/balance/
    out_dir = os.path.join("reports", "balance")
    os.makedirs(out_dir, exist_ok=True)
    balance_results_path = os.path.join(out_dir, "balance_results.csv")
    balance_summary_path = os.path.join(out_dir, "balance_summary.csv")
    # Round numeric results to 4 decimals for readability
    balance_df_rounded = balance_df.copy()
    for col in balance_df_rounded.columns:
        if pd.api.types.is_numeric_dtype(balance_df_rounded[col]):
            balance_df_rounded[col] = balance_df_rounded[col].round(4)
    summary_rounded = summary.copy()
    for col in summary_rounded.columns:
        if pd.api.types.is_numeric_dtype(summary_rounded[col]):
            summary_rounded[col] = summary_rounded[col].round(4)
    balance_df_rounded.to_csv(balance_results_path)
    summary_rounded.to_csv(balance_summary_path)
    logger.info("📄 Saved balance results to %s and summary to %s", balance_results_path, balance_summary_path)
    # Plots: save under reports/plots
    plots_root = os.path.join("reports", "plots")
    features_plots_dir = os.path.join(plots_root, "features")
    balance_plots_dir = os.path.join(plots_root, "balance")
    os.makedirs(features_plots_dir, exist_ok=True)
    os.makedirs(balance_plots_dir, exist_ok=True)
    plot_feature_distributions(df, sample_size=200_000, save_dir=features_plots_dir, show=False)
    logger.info("📊 Saved feature plots under %s", features_plots_dir)
    plot_balance_distributions(df, tr_col=TREATMENT, balance_df=balance_df, features=FEATURES, save_dir=balance_plots_dir, show=False)
    logger.info("📊 Saved balance plots under %s", balance_plots_dir)
    return balance_df, summary


def run_baseline(df):
    """Run baseline causal effect estimation (ATE)."""
    logger.info("📊 Running baseline causal estimation...")
    p_treat, p_ctrl, uplift, ci_low, ci_high = diff_in_means(df[OUTCOME], df[TREATMENT])
    logger.info(
        "Naïve ATE (diff in means): %.5f (p_treat=%.5f, p_ctrl=%.5f), CI95=(%.5f, %.5f)",
        uplift, p_treat, p_ctrl, ci_low, ci_high,
    )

    coef_w, ate_log_odds, ate_prob_diff = logistic_adjusted_ate(
        df, y_col=OUTCOME, w_col=TREATMENT, x_cols=FEATURES
    )
    logger.info(
        "Adjusted ATE (logit): coef_w=%.4f, prob-diff≈%.5f",
        coef_w, ate_prob_diff,
    )
    return ate_prob_diff


def run_propensity(df):
    """Placeholder for propensity score analysis."""
    logger.info("⚖️ Running propensity score analysis... (to be implemented)")
    # run_ipw(df, TREATMENT, OUTCOME, FEATURES)
    # run_matching(df, TREATMENT, OUTCOME, FEATURES)


def run_uplift(df):
    """Placeholder for uplift modeling."""
    logger.info("🌲 Running uplift models... (to be implemented)")
    # run_uplift_models(df, TREATMENT, OUTCOME, FEATURES)


# -----------------------
# Main
# -----------------------

def main(step: str):
    if step == "prepare":
        run_prepare()
        return

    logger.info("📂 Loading data...")
    df = pd.read_parquet(RAW_PARQUET)
    logger.info("✅ Loaded %s rows and %s columns.", f"{len(df):,}", len(df.columns))


    if step == "balance":
        run_balance(df)
    elif step == "baseline":
        run_baseline(df)
    elif step == "propensity":
        run_propensity(df)
    elif step == "uplift":
        run_uplift(df)
    elif step == "all":
        run_balance(df)
        run_baseline(df)
        run_propensity(df)
        run_uplift(df)
    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uplift Modeling Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        help="Step to run: prepare | balance | baseline | propensity | uplift | all")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(args.step)
