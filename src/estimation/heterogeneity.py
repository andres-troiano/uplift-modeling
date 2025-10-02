import os
import logging
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .baseline import diff_in_means


logger = logging.getLogger(__name__)


def _bin_feature(series: pd.Series, n_bins: int = 4, binning: str = "quantile") -> pd.Series:
    """Bin a feature into labels.

    - quantile: use pd.qcut
    - uniform: use pd.cut on min-max range
    For non-numeric data, fallback to top categories.
    """
    s = series
    # Prefer numeric binning when possible
    if pd.api.types.is_numeric_dtype(s):
        valid = s.dropna()
        if valid.nunique() < n_bins:
            # Fewer unique values than bins; fall back to categories
            return s.astype(str)
        if binning == "uniform":
            binned = pd.cut(s, bins=n_bins, duplicates="drop")
        else:
            binned = pd.qcut(s, q=n_bins, duplicates="drop")
        # Round interval endpoints in labels to 1 decimal for readability
        if pd.api.types.is_categorical_dtype(binned) and hasattr(binned.dtype, "categories"):
            try:
                cats = binned.cat.categories
                new_labels = []
                for iv in cats:
                    left = getattr(iv, "left", None)
                    right = getattr(iv, "right", None)
                    closed = iv.closed if hasattr(iv, "closed") else "right"
                    if left is not None and right is not None:
                        lbl = f"({left:.1f}, {right:.1f}{']' if closed == 'right' else ')'}"
                    else:
                        lbl = str(iv)
                    new_labels.append(lbl)
                binned = binned.cat.rename_categories(new_labels)
            except Exception:
                # If anything goes wrong, keep default labels
                pass
        return binned
    else:
        # Non-numeric: keep as string categories
        return s.astype(str)


def run_cate_by_feature_bins(
    df: pd.DataFrame,
    y_col: str,
    w_col: str,
    features: List[str],
    n_bins: int = 4,
    binning: str = "quantile",
    reports_dir: str = "reports/heterogeneity",
    plots_dir: str = "reports/plots/heterogeneity",
    show: bool = False,
    fit_interactions: bool = False,
) -> pd.DataFrame:
    """Compute per-bin ATE (CATE) for each feature and save CSV + plots.

    Returns a DataFrame with columns:
    feature, bin_label, bin_index, n_treat, n_control, p_treat, p_control, uplift, ci_low, ci_high
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    results: List[Dict[str, Any]] = []

    for feat in features:
        logger.info("CATE: processing feature '%s'", feat)
        try:
            bins = _bin_feature(df[feat], n_bins=n_bins, binning=binning)
        except Exception as e:
            logger.warning("Skipping feature %s due to binning error: %s", feat, e)
            continue

        # If categorical, ensure limited cardinality by collapsing to top categories if too many
        if not pd.api.types.is_categorical_dtype(bins) and bins.dtype == object:
            top = df[feat].astype(str).value_counts().head(n_bins).index
            bins = df[feat].astype(str).where(df[feat].astype(str).isin(top), other="OTHER")

        # Iterate bins (categories or intervals)
        # Create a stable label for x-axis
        if pd.api.types.is_categorical_dtype(bins):
            labels = bins.cat.categories
        else:
            labels = sorted(bins.dropna().unique().tolist())

        for idx, label in enumerate(labels):
            mask = bins == label
            seg = df.loc[mask]
            if seg.empty:
                continue
            p_t, p_c, uplift, ci_l, ci_h = diff_in_means(seg[y_col], seg[w_col])
            results.append({
                "feature": feat,
                "bin_label": str(label),
                "bin_index": idx,
                "n_treat": int((seg[w_col] == 1).sum()),
                "n_control": int((seg[w_col] == 0).sum()),
                "p_treat": round(p_t, 6),
                "p_control": round(p_c, 6),
                "uplift": round(uplift, 6),
                "ci_low": round(ci_l, 6),
                "ci_high": round(ci_h, 6),
            })

        # Plot per-feature uplift by bin
        feat_df = pd.DataFrame([r for r in results if r["feature"] == feat])
        if not feat_df.empty:
            feat_df = feat_df.sort_values("bin_index")
            plt.figure(figsize=(8, 4))
            x = np.arange(len(feat_df))
            y = feat_df["uplift"].values
            yerr = np.vstack((feat_df["uplift"] - feat_df["ci_low"], feat_df["ci_high"] - feat_df["uplift"]))
            plt.errorbar(x, y, yerr=yerr, fmt="o-", color="steelblue", ecolor="gray", capsize=4)
            plt.xticks(x, feat_df["bin_label"].astype(str), rotation=0)
            plt.title(f"CATE by {feat} bins")
            plt.xlabel(feat)
            plt.ylabel("Uplift (p_treat - p_control)")
            plt.grid(True, alpha=0.3)
            out_path = os.path.join(plots_dir, f"cate_{feat}.png")
            plt.tight_layout()
            plt.savefig(out_path, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
            logger.info("Saved CATE plot for %s to %s", feat, out_path)

        # Optional: interaction regressions (placeholder; can be expanded)
        if fit_interactions:
            # e.g., logistic regression with treatment, bin dummies and interactions
            # Implement later if needed; currently placeholder to keep API
            logger.debug("Interaction regression is enabled, but not implemented in detail yet for '%s'", feat)

    out_df = pd.DataFrame(results)
    # Round float columns to 4 decimals
    if not out_df.empty:
        num_cols = out_df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if pd.api.types.is_float_dtype(out_df[col]):
                out_df[col] = out_df[col].round(4)

    out_csv = os.path.join(reports_dir, "cate_results.csv")
    out_df.to_csv(out_csv, index=False)
    logger.info("📄 Saved CATE results to %s", out_csv)

    return out_df
