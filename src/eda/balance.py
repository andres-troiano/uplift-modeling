import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logger = logging.getLogger(__name__)


def check_balance(
	df: pd.DataFrame,
	tr_col: str,
	features: List[str],
	sample_size: int = 50_000,
	random_state: int = 42,
) -> pd.DataFrame:
	"""Kolmogorov–Smirnov balance tests for features across treatment vs control.

	Returns a DataFrame indexed by feature with columns ["ks_stat", "p_value"].
	"""
	results = []

	# Sample treatment and control groups for speed
	t_df = df[df[tr_col] == 1]
	c_df = df[df[tr_col] == 0]
	if sample_size is not None:
		t_df = t_df.sample(n=min(sample_size, len(t_df)), random_state=random_state)
		c_df = c_df.sample(n=min(sample_size, len(c_df)), random_state=random_state)

	for c in features:
		t_vals = t_df[c].dropna()
		c_vals = c_df[c].dropna()
		if len(t_vals) > 0 and len(c_vals) > 0:
			ks_stat, p_val = stats.ks_2samp(t_vals, c_vals)
			results.append({"feature": c, "ks_stat": ks_stat, "p_value": p_val})

	df_out = pd.DataFrame(results).set_index("feature").sort_values("ks_stat", ascending=False)
	if not df_out.empty:
		top_n = min(5, len(df_out))
		logger.info(
			"check_balance: top %d features by KS: %s | n_features=%d",
			top_n,
			{f: round(v, 4) for f, v in df_out.head(top_n)["ks_stat"].to_dict().items()},
			len(df_out),
		)
	return df_out


def summarize_balance(balance_df: pd.DataFrame, thresholds: Tuple[float, float] = (0.01, 0.05)) -> pd.DataFrame:
	"""Categorize imbalance level based on KS statistic thresholds."""
	t1, t2 = thresholds
	def categorize(ks: float) -> str:
		if ks < t1:
			return "Negligible"
		elif ks < t2:
			return "Moderate"
		else:
			return "Large"

	summary = balance_df.copy()
	summary["imbalance_level"] = summary["ks_stat"].apply(categorize)
	summary = summary.sort_values("ks_stat", ascending=False)
	counts = summary["imbalance_level"].value_counts().to_dict()
	logger.info("summarize_balance: counts per category: %s", counts)
	return summary


def compute_correlations(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
	"""Return correlation matrix for selected columns."""
	return df[columns].corr()


def significant_correlations(corr_df: pd.DataFrame, threshold: float = 0.1) -> List[Tuple[str, str, float]]:
	"""List pairs with absolute correlation >= threshold, sorted by magnitude."""
	results: List[Tuple[str, str, float]] = []
	cols = corr_df.columns
	for i in range(len(cols)):
		for j in range(i + 1, len(cols)):
			val = float(corr_df.iloc[i, j])
			if abs(val) >= threshold:
				results.append((str(cols[i]), str(cols[j]), round(val, 3)))
	results.sort(key=lambda x: abs(x[2]), reverse=True)
	return results


def confounding_risk_table(
	balance_df: pd.DataFrame,
	corr_df: pd.DataFrame,
	y_col: str = "conversion",
	ks_threshold: float = 0.01,
	corr_threshold: float = 0.1,
) -> pd.DataFrame:
	"""Combine KS imbalance and correlation with outcome into a confounding risk table."""
	records = []
	for f in balance_df.index:
		ks = float(balance_df.loc[f, "ks_stat"]) if f in balance_df.index else 0.0
		corr_val = float(corr_df.loc[f, y_col]) if f in corr_df.index and y_col in corr_df.columns else 0.0
		if abs(ks) >= ks_threshold and abs(corr_val) >= corr_threshold:
			risk = "High"
		elif abs(ks) >= ks_threshold or abs(corr_val) >= corr_threshold:
			risk = "Moderate"
		else:
			risk = "Low"
		records.append({"feature": f, "ks_stat": ks, "corr_outcome": corr_val, "confounding_risk": risk})

	df_out = pd.DataFrame(records).set_index("feature")
	risk_order = pd.CategoricalDtype(categories=["High", "Moderate", "Low"], ordered=True)
	df_out["confounding_risk"] = df_out["confounding_risk"].astype(risk_order)
	return df_out.sort_values("confounding_risk")
