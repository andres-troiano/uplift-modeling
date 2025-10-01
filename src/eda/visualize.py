import os
import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


logger = logging.getLogger(__name__)

# Consistent color palette across plots
PALETTE: Dict[str, str] = {
    "control": "orange",
    "treatment": "steelblue",
}


def plot_feature_distributions(
    df: pd.DataFrame,
    sample_size: Optional[int] = 200_000,
    K: int = 10,
    random_state: int = 42,
    return_handles: bool = False,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Optional[Dict[str, Any]]:
    """Plot histograms/boxplots for top-K numeric and bar charts for top-K categorical.

    Returns
    -------
    Optional[Dict[str, Any]]
        Dict with handles: {"numeric": (fig, axes) or None, "categorical": [fig, ...]}
    """
    if sample_size is not None and len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=random_state)
    else:
        df_sample = df.copy()

    num_cols = [c for c in df_sample.select_dtypes(include=[np.number]).columns][:K]
    cat_cols = [c for c in df_sample.select_dtypes(exclude=[np.number]).columns][:K]

    handles: Dict[str, Any] = {"numeric": None, "categorical": []}

    if num_cols:
        fig, axes = plt.subplots(len(num_cols), 2, figsize=(10, 4 * len(num_cols)))
        if len(num_cols) == 1:
            axes = np.array([axes])
        for i, c in enumerate(num_cols):
            logger.info("Plotting numeric feature '%s'", c)
            sns.histplot(df_sample[c], bins=50, ax=axes[i, 0], element="step", color=PALETTE.get("treatment", "steelblue"))
            axes[i, 0].set_title(f"{c} distribution")
            sns.boxplot(x=df_sample[c], ax=axes[i, 1], color=PALETTE.get("control", "orange"))
            axes[i, 1].set_title(f"{c} boxplot")
        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, "numeric_distributions.png")
            fig.savefig(out_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        if return_handles:
            handles["numeric"] = (fig, axes)

    if cat_cols:
        for c in cat_cols:
            logger.info("Plotting categorical feature '%s'", c)
            plt.figure(figsize=(8, 3))
            vc = df_sample[c].astype(str).value_counts().head(20)
            sns.barplot(x=vc.values, y=vc.index, color=PALETTE.get("treatment", "steelblue"))
            plt.title(f"Top categories for {c}")
            plt.tight_layout()
            fig = plt.gcf()
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, f"categorical_top_{c}.png")
                fig.savefig(out_path, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            if return_handles:
                handles["categorical"].append(fig)

    return handles if return_handles else None


def plot_balance_distributions(
    df: pd.DataFrame,
    tr_col: str,
    features: List[str],
    balance_df: pd.DataFrame,
    top_k: int = 3,
    bins: int = 50,
    sample_size: int = 200_000,
    random_state: int = 42,
    return_handles: bool = False,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Optional[List[plt.Figure]]:
    """Plot histograms for the top-K most imbalanced features (by KS)."""
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    top_feats = balance_df.sort_values("ks_stat", ascending=False).head(top_k).index.tolist()
    figs: List[plt.Figure] = []

    for f in top_feats:
        logger.info("Plotting balance distributions for %s", f)
        plt.figure(figsize=(8, 4))
        sns.histplot(df.loc[df[tr_col] == 1, f], bins=bins, color=PALETTE.get("treatment", "steelblue"), label="Treatment", stat="density", alpha=0.6)
        sns.histplot(df.loc[df[tr_col] == 0, f], bins=bins, color=PALETTE.get("control", "orange"), label="Control", stat="density", alpha=0.6)
        plt.title(f"Distribution of {f} by treatment status\n(KS={balance_df.loc[f,'ks_stat']:.3f})")
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"distribution_{f}.png")
            fig.savefig(out_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        if return_handles:
            figs.append(fig)

    return figs if return_handles else None


def plot_balance_cdfs(
    df: pd.DataFrame,
    tr_col: str,
    features: List[str],
    balance_df: pd.DataFrame,
    top_k: int = 3,
    sample_size: int = 200_000,
    random_state: int = 42,
    return_handles: bool = False,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Optional[List[plt.Figure]]:
    """Plot CDFs for the top-K most imbalanced features (by KS)."""
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    top_feats = balance_df.sort_values("ks_stat", ascending=False).head(top_k).index.tolist()
    figs: List[plt.Figure] = []

    for f in top_feats:
        logger.info("Plotting balance CDFs for %s", f)
        plt.figure(figsize=(8, 4))
        t_vals = np.sort(df.loc[df[tr_col] == 1, f].dropna())
        t_cdf = np.arange(1, len(t_vals) + 1) / max(1, len(t_vals))
        plt.plot(t_vals, t_cdf, label="Treatment", color=PALETTE.get("treatment", "steelblue"))
        c_vals = np.sort(df.loc[df[tr_col] == 0, f].dropna())
        c_cdf = np.arange(1, len(c_vals) + 1) / max(1, len(c_vals))
        plt.plot(c_vals, c_cdf, label="Control", color=PALETTE.get("control", "orange"))
        plt.title(f"CDF of {f} by treatment status\n(KS={balance_df.loc[f,'ks_stat']:.3f})")
        plt.xlabel(f)
        plt.ylabel("Cumulative probability")
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"cdf_{f}.png")
            fig.savefig(out_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        if return_handles:
            figs.append(fig)

    return figs if return_handles else None


def plot_cdfs_by_group(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    groups: List[int] = [0, 1],
    sample_size: Optional[int] = 200_000,
    random_state: int = 42,
    return_handles: bool = False,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Optional[plt.Figure]:
    """Plot CDFs of a feature for each group value in group_col (e.g., treatment vs control)."""
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)

    logger.info("Plotting CDF for %s by %s", feature, group_col)
    plt.figure(figsize=(8, 4))
    palette = {groups[0]: PALETTE.get("control", "orange"), groups[1]: PALETTE.get("treatment", "steelblue")}
    for g in groups:
        vals = np.sort(df.loc[df[group_col] == g, feature].dropna())
        cdf = np.arange(1, len(vals) + 1) / max(1, len(vals))
        plt.plot(vals, cdf, label=f"{group_col}={g}", color=palette.get(g, None))
    plt.title(f"CDF of {feature} by {group_col}")
    plt.xlabel(feature)
    plt.ylabel("Cumulative probability")
    plt.legend()
    plt.tight_layout()
    fig = plt.gcf()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"cdf_{feature}.png")
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig if return_handles else None


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: List[str],
    return_handles: bool = False,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Tuple[Optional[plt.Figure], pd.DataFrame]:
    """Plot a correlation heatmap and return the figure (optional) and numeric matrix."""
    corr = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature correlations")
    fig = plt.gcf()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "correlation_heatmap.png")
        fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return (fig if return_handles else None, corr) 