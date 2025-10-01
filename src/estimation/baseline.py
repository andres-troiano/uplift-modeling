import math
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def diff_in_means(y: pd.Series, w: pd.Series) -> Tuple[float, float, float, float, float]:
	"""Difference in means (ATE) and Wald 95% CI for binary outcomes.

	Returns (p_treat, p_control, uplift, ci_low, ci_high)
	"""
	y = pd.to_numeric(y, errors="coerce")
	w = pd.to_numeric(w, errors="coerce")
	mask = (~y.isna()) & (~w.isna())
	y = y[mask]
	w = w[mask]

	# Binarize outcome if not already
	if sorted(y.unique()) not in ([0, 1], [0.0, 1.0]):
		y = (y > 0).astype(int)
	w = (w > 0).astype(int)

	p_treat = y[w == 1].mean()
	p_ctrl = y[w == 0].mean()
	uplift = float(p_treat - p_ctrl)
	n_t = int((w == 1).sum())
	n_c = int((w == 0).sum())
	se = math.sqrt(p_treat * (1 - p_treat) / max(1, n_t) + p_ctrl * (1 - p_ctrl) / max(1, n_c))
	z = 1.96
	ci_low = uplift - z * se
	ci_high = uplift + z * se
	return float(p_treat), float(p_ctrl), float(uplift), float(ci_low), float(ci_high)


def logistic_adjusted_ate(
	df: pd.DataFrame,
	y_col: str,
	w_col: str,
	x_cols: List[str],
	max_iter: int = 200,
	C: float = 1.0,
	n_jobs: int = -1,
) -> Tuple[float, float, float]:
	"""Compute adjusted ATE via logistic regression with treatment + covariates.

	Model: logit(y) ~ w + X
	Returns (coef_w, ate_log_odds, ate_prob_diff_approx)
	"""
	y = df[y_col].astype(int).values
	w = df[w_col].astype(int).values
	X = df[x_cols].values
	X_model = np.column_stack([w, X])
	model = LogisticRegression(max_iter=max_iter, C=C, n_jobs=n_jobs)
	model.fit(X_model, y)
	coef_w = float(model.coef_[0][0])
	# Approximate prob-diff using average derivative of logistic at baseline
	p_hat = model.predict_proba(X_model)[:, 1]
	avg_var = float(np.mean(p_hat * (1 - p_hat)))
	ate_prob_diff = float(coef_w * avg_var)
	return coef_w, coef_w, ate_prob_diff


def cuped_adjustment(
	df: pd.DataFrame,
	pre_col: str,
	post_col: str,
	w_col: str,
) -> Tuple[float, float]:
	"""CUPED adjustment: y_adj = y_post - theta * y_pre, theta = cov(y_pre,y_post)/var(y_pre).

	Returns (theta, variance_reduction_pct)
	"""
	pre = pd.to_numeric(df[pre_col], errors="coerce")
	post = pd.to_numeric(df[post_col], errors="coerce")
	mask = (~pre.isna()) & (~post.isna())
	pre = pre[mask]
	post = post[mask]
	cov = float(np.cov(pre, post)[0, 1])
	var_pre = float(np.var(pre, ddof=1))
	theta = 0.0 if var_pre == 0 else cov / var_pre
	# Variance reduction approximated as R^2 of regression on pre
	r2 = 0.0 if var_pre == 0 else (cov ** 2) / (var_pre * float(np.var(post, ddof=1)))
	return theta, float(100.0 * max(0.0, min(1.0, r2)))
