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


import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Dict, Any


def logistic_adjusted_ate(
    df: pd.DataFrame,
    y_col: str,
    w_col: str,
    x_cols: List[str],
) -> Dict[str, Any]:
    """
    Compute adjusted ATE via logistic regression with treatment + covariates.

    Model: logit(y) ~ w + X

    Returns dict with:
        - coef_w: treatment coefficient (log-odds scale)
        - odds_ratio: exp(coef_w)
        - ate_prob_diff: approx probability difference
        - se: standard error of coef_w
        - ci_low, ci_high: 95% CI for coef_w
        - n_treat, n_control: sample sizes
    """

    # Prepare data: coerce to numeric and filter NaNs
    X_df = df[x_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    y_s = pd.to_numeric(df[y_col], errors="coerce")
    w_s = pd.to_numeric(df[w_col], errors="coerce")

    mask = (~y_s.isna()) & (~w_s.isna())
    mask &= (~X_df.isna()).all(axis=1)

    # Binarize outcome/treatment (assume >0 indicates positive class)
    y = (y_s[mask] > 0).astype(int).values
    w = (w_s[mask] > 0).astype(int).values
    X = X_df.loc[mask].values

    # Add constant and treatment as first column
    X_model = sm.add_constant(np.column_stack([w, X]))

    # Fit logistic regression
    model = sm.Logit(y, X_model)
    result = model.fit(disp=False)

    # Extract treatment coefficient (col=1 after constant)
    coef_w = float(result.params[1])
    se_w = float(result.bse[1])
    ci = result.conf_int()
    try:
        # Preferred: DataFrame with iloc
        ci_low = float(ci.iloc[1, 0])
        ci_high = float(ci.iloc[1, 1])
    except Exception:
        # Fallback: numpy array
        ci_low = float(ci[1, 0])
        ci_high = float(ci[1, 1])

    # Odds ratio
    odds_ratio = float(np.exp(coef_w))

    # Approximate probability difference using average derivative trick
    p_hat = result.predict(X_model)
    avg_var = float(np.mean(p_hat * (1 - p_hat)))
    ate_prob_diff = float(coef_w * avg_var)

    # Sample sizes
    n_treat = int(w.sum())
    n_control = int(len(w) - n_treat)

    return {
        "coef_w": coef_w,
        "odds_ratio": odds_ratio,
        "ate_prob_diff": ate_prob_diff,
        "se": se_w,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_treat": n_treat,
        "n_control": n_control,
    }


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
