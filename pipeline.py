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
from typing import Optional

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
from src.estimation.heterogeneity import run_cate_by_feature_bins
from src.estimation.propensity import run_propensity_methods
from src.estimation.uplift import run_uplift_models
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

# Subsample controls (stratified by treatment)
SUBSAMPLE_SIZES = {
    "balance":       200_000,
    "baseline":    1_000_000,
    "prop_ipw":    1_000_000,
    "prop_match":    100_000,
    "heterogeneity": 300_000,
    "uplift":        300_000,
}
SEED = 42


def stratified_sample(df: pd.DataFrame, n: int, by: str, seed: int = SEED) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    total = len(df)
    parts = []
    for val, g in df.groupby(by):
        take = max(1, int(round(len(g) * n / total)))
        parts.append(g.sample(n=take, random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def maybe_subsample(df: pd.DataFrame, step_key: str, enabled: bool, by: str, seed: int = SEED) -> pd.DataFrame:
    if not enabled:
        return df
    n = SUBSAMPLE_SIZES.get(step_key)
    if n and n < len(df):
        before = len(df)
        df_s = stratified_sample(df, n=n, by=by, seed=seed)
        logger.info("🔬 Subsampled for %s: %s → %s rows", step_key, f"{before:,}", f"{len(df_s):,}")
        return df_s
    return df


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
    plot_feature_distributions(df, sample_size=SUBSAMPLE_SIZES["balance"], save_dir=features_plots_dir, show=False)
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

    logit_res = logistic_adjusted_ate(df, y_col=OUTCOME, w_col=TREATMENT, x_cols=FEATURES)
    logger.info(
        "Adjusted ATE (logit): coef_w=%.4f (se=%.4f, 95%% CI=[%.4f, %.4f]), OR=%.3f, prob-diff≈%.5f, n_treat=%d, n_control=%d",
        logit_res["coef_w"],
        logit_res["se"],
        logit_res["ci_low"],
        logit_res["ci_high"],
        logit_res["odds_ratio"],
        logit_res["ate_prob_diff"],
        logit_res["n_treat"],
        logit_res["n_control"],
    )
    return logit_res["ate_prob_diff"]


def run_uplift(df):
    """Run uplift models (T-Learner, X-Learner, optional Causal Forest)."""
    logger.info("🌲 Running uplift models...")
    out_dir = os.path.join("reports", "uplift")
    plots_dir = os.path.join("reports", "plots", "uplift")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    results_df = run_uplift_models(df, t_col=TREATMENT, y_col=OUTCOME, x_cols=FEATURES, reports_dir=out_dir, plots_dir=plots_dir, sample_size=None, show=False)
    logger.info("📄 Uplift results saved with %d methods.", len(results_df))


def run_heterogeneity(df):
    """Run CATE (heterogeneity) analysis by feature bins and save artifacts."""
    logger.info("🔎 Running CATE (heterogeneity) analysis...")
    cate_df = run_cate_by_feature_bins(
        df,
        y_col=OUTCOME,
        w_col=TREATMENT,
        features=FEATURES,
        n_bins=4,
        binning="quantile",
        reports_dir=os.path.join("reports", "heterogeneity"),
        plots_dir=os.path.join("reports", "plots", "heterogeneity"),
        show=False,
        fit_interactions=False,
    )
    logger.info("📄 CATE analysis complete; %d rows in results.", len(cate_df))
    return cate_df

# -----------------------
# Main
# -----------------------

def main(step: str, subsample: bool, seed: int):
    if step == "prepare":
        run_prepare()
        return

    logger.info("📂 Loading data...")
    df = pd.read_parquet(RAW_PARQUET)
    logger.info("✅ Loaded %s rows and %s columns.", f"{len(df):,}", len(df.columns))


    if step == "balance":
        run_balance(maybe_subsample(df, "balance", subsample, TREATMENT, seed))
    elif step == "baseline":
        run_baseline(maybe_subsample(df, "baseline", subsample, TREATMENT, seed))
    elif step == "heterogeneity":
        run_heterogeneity(maybe_subsample(df, "heterogeneity", subsample, TREATMENT, seed))
    elif step == "propensity":
        # Separate subsamples for IPW and Matching, then delegate to runner
        df_ipw = maybe_subsample(df, "prop_ipw", subsample, TREATMENT, seed)
        df_match = maybe_subsample(df, "prop_match", subsample, TREATMENT, seed)
        out_dir = os.path.join("reports", "propensity")
        os.makedirs(out_dir, exist_ok=True)
        run_propensity_methods(df_ipw, df_match, TREATMENT, OUTCOME, FEATURES, reports_dir=out_dir)
    elif step == "uplift":
        run_uplift(maybe_subsample(df, "uplift", subsample, TREATMENT, seed))
    elif step == "all":
        df_b = maybe_subsample(df, "balance", subsample, TREATMENT, seed)
        run_balance(df_b)

        df_base = maybe_subsample(df, "baseline", subsample, TREATMENT, seed)
        run_baseline(df_base)

        df_het = maybe_subsample(df, "heterogeneity", subsample, TREATMENT, seed)
        run_heterogeneity(df_het)

        df_ipw = maybe_subsample(df, "prop_ipw", subsample, TREATMENT, seed)
        df_match = maybe_subsample(df, "prop_match", subsample, TREATMENT, seed)
        out_dir = os.path.join("reports", "propensity")
        os.makedirs(out_dir, exist_ok=True)
        run_propensity_methods(df_ipw, df_match, TREATMENT, OUTCOME, FEATURES, reports_dir=out_dir)

        df_upl = maybe_subsample(df, "uplift", subsample, TREATMENT, seed)
        run_uplift(df_upl)
    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uplift Modeling Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        help="Step to run: prepare | balance | baseline | heterogeneity | propensity | uplift | all")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level: DEBUG, INFO, WARNING, ERROR")
    parser.add_argument("--subsample", action="store_true", help="If set, use per-step subsample sizes with stratified sampling")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for sampling")
    args = parser.parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main(args.step, args.subsample, args.seed)
