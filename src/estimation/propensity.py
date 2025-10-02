import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


def _coerce_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.values
    return X


def estimate_propensity_scores(
    df: pd.DataFrame,
    t_col: str,
    x_cols: List[str],
    model: str = "logit",
) -> np.ndarray:
    """Estimate P(T=1|X) using a classifier.

    model: "logit" (default) or "xgb" (if xgboost installed)
    """
    logger.info("Estimating propensity scores with model=%s", model)
    t = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int).values
    X = _coerce_matrix(df, x_cols)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    t = t[mask]

    if model == "xgb":
        try:
            import xgboost as xgb  # type: ignore

            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
            )
        except Exception:
            logger.warning("xgboost not available; falling back to logistic regression")
            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    else:
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)

    clf.fit(X, t)
    e = np.zeros(len(mask), dtype=float)
    e[mask] = clf.predict_proba(X)[:, 1]
    # For rows with NaNs in X, drop them by setting NaN (handled downstream)
    if (~mask).any():
        dropped = int((~mask).sum())
        logger.info("Propensity: dropping %d rows with NaNs in features", dropped)
        e[~mask] = np.nan
    # Trim to avoid division issues
    eps = 1e-3
    e = np.clip(e, eps, 1 - eps)
    return e


def _ipw_ate_core(t: np.ndarray, y: np.ndarray, e: np.ndarray) -> Tuple[float, float, float, float]:
    n = len(y)
    # IPW estimate
    pseudo = t * y / e - (1 - t) * y / (1 - e)
    ate = float(pseudo.mean())
    # Influence-function-based SE
    phi = pseudo - ate
    var = float(np.sum(phi ** 2) / (n * (n - 1))) if n > 1 else 0.0
    se = float(np.sqrt(var))
    z = 1.96
    ci_low = ate - z * se
    ci_high = ate + z * se
    return ate, se, ci_low, ci_high


def ate_ipw(df: pd.DataFrame, t_col: str, y_col: str, x_cols: List[str]) -> Dict[str, Any]:
    logger.info("Running IPW ATE...")
    e = estimate_propensity_scores(df, t_col, x_cols, model="logit")
    t = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int).values
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
    mask = (~np.isnan(e)) & (~np.isnan(t)) & (~np.isnan(y))
    t = (t[mask] > 0).astype(int)
    y = (y[mask] > 0).astype(int)
    e = e[mask]

    ate, se, ci_low, ci_high = _ipw_ate_core(t, y, e)
    n_treat = int(t.sum())
    n_control = int(len(t) - n_treat)
    logger.info("IPW ATE = %.6f (95%% CI [%.6f, %.6f])", ate, ci_low, ci_high)
    return {
        "method": "ipw",
        "ate": ate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_treat": n_treat,
        "n_control": n_control,
    }


def ate_matching(
    df: pd.DataFrame,
    t_col: str,
    y_col: str,
    x_cols: List[str],
    k: int = 1,
) -> Dict[str, Any]:
    logger.info("Running nearest-neighbor matching (k=%d)...", k)
    t = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int).values
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
    X = _coerce_matrix(df, x_cols)
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    t = (t[mask] > 0).astype(int)
    y = (y[mask] > 0).astype(int)

    # Scale features for distance matching
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    treated_idx = np.where(t == 1)[0]
    control_idx = np.where(t == 0)[0]
    if len(treated_idx) == 0 or len(control_idx) == 0:
        return {"method": "matching", "ate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_treat": int((t==1).sum()), "n_control": int((t==0).sum())}

    # Match treated -> controls
    nn_c = NearestNeighbors(n_neighbors=min(k, len(control_idx)), algorithm="auto")
    nn_c.fit(Xs[control_idx])
    dists_tc, idxs_tc = nn_c.kneighbors(Xs[treated_idx])
    matched_y0_for_treated = y[control_idx][idxs_tc].mean(axis=1)
    diffs_t = y[treated_idx] - matched_y0_for_treated

    # Match controls -> treated
    nn_t = NearestNeighbors(n_neighbors=min(k, len(treated_idx)), algorithm="auto")
    nn_t.fit(Xs[treated_idx])
    dists_ct, idxs_ct = nn_t.kneighbors(Xs[control_idx])
    matched_y1_for_controls = y[treated_idx][idxs_ct].mean(axis=1)
    diffs_c = matched_y1_for_controls - y[control_idx]

    diffs = np.concatenate([diffs_t, diffs_c])
    ate = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    z = 1.96
    ci_low = ate - z * se
    ci_high = ate + z * se

    n_treat = int((t == 1).sum())
    n_control = int((t == 0).sum())
    logger.info("Matching ATE = %.6f (95%% CI [%.6f, %.6f])", ate, ci_low, ci_high)
    return {
        "method": "matching",
        "ate": ate,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_treat": n_treat,
        "n_control": n_control,
    }


def run_propensity_methods(
    df: pd.DataFrame,
    t_col: str,
    y_col: str,
    x_cols: List[str],
    reports_dir: str = "reports/propensity",
    sample_size: Optional[int] = None,
) -> pd.DataFrame:
    os.makedirs(reports_dir, exist_ok=True)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        logger.info("🔬 Downsampled to %s rows for propensity methods", f"{len(df):,}")


    res_ipw = ate_ipw(df, t_col, y_col, x_cols)
    res_match = ate_matching(df, t_col, y_col, x_cols)

    results_df = pd.DataFrame([res_ipw, res_match])
    # Round floats for readability
    float_cols = [c for c in ["ate", "se", "ci_low", "ci_high"] if c in results_df.columns]
    results_df[float_cols] = results_df[float_cols].round(4)

    out_csv = os.path.join(reports_dir, "propensity_results.csv")
    results_df.to_csv(out_csv, index=False)
    logger.info("📄 Saved propensity results to %s", out_csv)
    return results_df
