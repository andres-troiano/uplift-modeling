import os
import logging
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import clone

from .propensity import estimate_propensity_scores


logger = logging.getLogger(__name__)


def _coerce_X(df: pd.DataFrame, x_cols: List[str]) -> np.ndarray:
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").astype(float)
    return X.values


def _make_base_model(base_model: Optional[Any] = None, model_hint: str = "logit"):
    if base_model is not None:
        return clone(base_model)
    if model_hint == "xgb":
        try:
            import xgboost as xgb  # type: ignore

            return xgb.XGBClassifier(
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
    return LogisticRegression(max_iter=1000, n_jobs=-1)


def t_learner(
    df: pd.DataFrame,
    t_col: str,
    y_col: str,
    x_cols: List[str],
    base_model: Optional[Any] = None,
    model_hint: str = "logit",
) -> np.ndarray:
    """T-Learner uplift = P(y=1|X,T=1) - P(y=1|X,T=0)."""
    logger.info("Training T-Learner...")
    X = _coerce_X(df, x_cols)
    t = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int).values
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values

    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(t)) & (~np.isnan(y))
    X = X[mask]
    t = t[mask]
    y = y[mask]

    X1 = X[t == 1]
    y1 = y[t == 1]
    X0 = X[t == 0]
    y0 = y[t == 0]

    m1 = _make_base_model(base_model, model_hint)
    m0 = _make_base_model(base_model, model_hint)
    m1.fit(X1, y1)
    m0.fit(X0, y0)

    p1 = m1.predict_proba(X)[:, 1]
    p0 = m0.predict_proba(X)[:, 1]
    uplift = p1 - p0
    # Re-insert into original ordering
    res = np.zeros(len(mask), dtype=float)
    res[:] = np.nan
    res[mask] = uplift
    return res


def x_learner(
    df: pd.DataFrame,
    t_col: str,
    y_col: str,
    x_cols: List[str],
    base_model: Optional[Any] = None,
    model_hint: str = "logit",
) -> np.ndarray:
    """X-Learner with propensity-weighted blending."""
    logger.info("Training X-Learner...")
    X = _coerce_X(df, x_cols)
    t = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int).values
    y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
    mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(t)) & (~np.isnan(y))
    X = X[mask]
    t = t[mask]
    y = y[mask]

    # Step 1: Fit M1, M0
    m1 = _make_base_model(base_model, model_hint)
    m0 = _make_base_model(base_model, model_hint)
    m1.fit(X[t == 1], y[t == 1])
    m0.fit(X[t == 0], y[t == 0])

    # Imputed treatment effects
    D1 = y[t == 1] - m0.predict_proba(X[t == 1])[:, 1]
    D0 = m1.predict_proba(X[t == 0])[:, 1] - y[t == 0]

    # Meta-learners must be regressors, since D1/D0 are continuous
    tau1 = Ridge(alpha=1.0)
    tau0 = Ridge(alpha=1.0)
    tau1.fit(X[t == 1], D1)
    tau0.fit(X[t == 0], D0)

    # Propensity scores for blending
    e_full = estimate_propensity_scores(df.loc[mask], t_col=t_col, x_cols=x_cols, model="logit")

    tau1_pred = tau1.predict_proba(X)[:, 1] if hasattr(tau1, "predict_proba") else tau1.predict(X)
    tau0_pred = tau0.predict_proba(X)[:, 1] if hasattr(tau0, "predict_proba") else tau0.predict(X)

    # Blend: following X-learner suggestion, use propensity as weight
    uplift = e_full * tau1_pred + (1 - e_full) * tau0_pred

    res = np.zeros(len(mask), dtype=float)
    res[:] = np.nan
    res[mask] = uplift
    return res


def causal_forest(
    df: pd.DataFrame,
    t_col: str,
    y_col: str,
    x_cols: List[str],
) -> np.ndarray:
    """Optional Causal Forest wrapper (if econml/causalml present)."""
    try:
        from causalml.inference.tree import UpliftRandomForestClassifier  # type: ignore

        logger.info("Training Causal Forest (CausalML)...")
        X = _coerce_X(df, x_cols)
        t = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(int).values
        y = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int).values
        mask = (~np.isnan(X).any(axis=1)) & (~np.isnan(t)) & (~np.isnan(y))
        X = X[mask]
        t = t[mask]
        y = y[mask]

        # causalml expects treatments as {0,1} and classes
        model = UpliftRandomForestClassifier(control_name=0, n_estimators=200, max_depth=5, min_samples_leaf=200)
        # This API trains with pandas
        X_df = pd.DataFrame(X, columns=x_cols)
        model.fit(X_df, treatment=t, y=y)
        uplift = model.predict(X_df)

        res = np.zeros(len(mask), dtype=float)
        res[:] = np.nan
        res[mask] = uplift
        return res
    except Exception:
        logger.warning("Causal Forest not available; skipping")
        return np.full(len(df), np.nan, dtype=float)


def _binned_groups(uplift_scores: np.ndarray, bins: int) -> np.ndarray:
    order = np.argsort(-uplift_scores)  # descending
    n = len(uplift_scores)
    bin_ids = np.zeros(n, dtype=int)
    # Equal-size bins
    for i, idx in enumerate(order):
        bin_ids[idx] = (i * bins) // n
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    return bin_ids


def compute_uplift_curve(y_true: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    mask = (~np.isnan(y_true)) & (~np.isnan(t)) & (~np.isnan(uplift_scores))
    y = y_true[mask]
    w = t[mask]
    s = uplift_scores[mask]

    bin_ids = _binned_groups(s, bins)
    uplift_perc = []
    x_vals = []
    for b in range(bins):
        sel = bin_ids == b
        if not np.any(sel):
            uplift_perc.append(0.0)
            x_vals.append((b + 1) / bins)
            continue
        y_b = y[sel]
        w_b = w[sel]
        n1 = max(1, int((w_b == 1).sum()))
        n0 = max(1, int((w_b == 0).sum()))
        p1 = float(y_b[w_b == 1].mean()) if n1 > 0 else 0.0
        p0 = float(y_b[w_b == 0].mean()) if n0 > 0 else 0.0
        uplift_perc.append(p1 - p0)
        x_vals.append((b + 1) / bins)
    return np.array(x_vals), np.array(uplift_perc)


def compute_qini_curve(y_true: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    mask = (~np.isnan(y_true)) & (~np.isnan(t)) & (~np.isnan(uplift_scores))
    y = y_true[mask]
    w = t[mask]
    s = uplift_scores[mask]
    order = np.argsort(-s)
    y = y[order]
    w = w[order]
    n = len(y)
    # cumulative by partitions
    parts = np.linspace(0, n, bins + 1, dtype=int)[1:]
    qini_vals = []
    x_vals = []
    for p in parts:
        y_p = y[:p]
        w_p = w[:p]
        n1 = max(1, int((w_p == 1).sum()))
        n0 = max(1, int((w_p == 0).sum()))
        lift = float(y_p[w_p == 1].sum()) - float(y_p[w_p == 0].sum()) * (n1 / max(1, n0))
        qini_vals.append(lift)
        x_vals.append(p / n if n else 0.0)
    return np.array(x_vals), np.array(qini_vals)


def qini_coefficient(y_true: np.ndarray, t: np.ndarray, uplift_scores: np.ndarray, bins: int = 10) -> float:
    x, q = compute_qini_curve(y_true, t, uplift_scores, bins=bins)
    if len(x) < 2:
        return 0.0
    # Trapezoidal area
    area = float(np.trapz(q, x))
    return area


def _save_curve_plot(x: np.ndarray, y: np.ndarray, title: str, out_path: str, show: bool = False) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", color="steelblue")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("Fraction targeted")
    plt.ylabel(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _incremental_conversions_at_fracs(
    y_true: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
    fracs: List[float],
) -> Dict[str, int]:
    """Compute incremental conversions at top-k%% thresholds.

    For a given fraction f, select top f of users by uplift score (descending),
    compute treated and control rates within that subset, and return:
        incr = N_treat * (p_treat - p_ctrl)
    """
    mask = (~np.isnan(y_true)) & (~np.isnan(t)) & (~np.isnan(uplift_scores))
    y = y_true[mask]
    w = t[mask]
    s = uplift_scores[mask]

    order = np.argsort(-s)
    y = y[order]
    w = w[order]
    n = len(y)
    out: Dict[str, int] = {}
    for f in fracs:
        k = int(np.floor(f * n))
        if k <= 0:
            out[f"incr_conv_top{int(f*100)}"] = 0
            continue
        y_top = y[:k]
        w_top = w[:k]
        n_treat = int((w_top == 1).sum())
        n_ctrl = int((w_top == 0).sum())
        p_treat = float(y_top[w_top == 1].mean()) if n_treat > 0 else 0.0
        p_ctrl = float(y_top[w_top == 0].mean()) if n_ctrl > 0 else 0.0
        incr = int(round(n_treat * (p_treat - p_ctrl)))
        out[f"incr_conv_top{int(f*100)}"] = incr
    return out


def run_uplift_models(
    df: pd.DataFrame,
    t_col: str,
    y_col: str,
    x_cols: List[str],
    reports_dir: str = "reports/uplift",
    plots_dir: str = "reports/plots/uplift",
    sample_size: Optional[int] = None,
    base_model: Optional[Any] = None,
    model_hint: str = "logit",
    show: bool = False,
) -> pd.DataFrame:
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    data = df
    if sample_size is not None and len(df) > sample_size:
        data = df.sample(sample_size, random_state=42).reset_index(drop=True)
        logger.info("🔬 Downsampled to %s rows for uplift models", f"{len(data):,}")


    y = pd.to_numeric(data[y_col], errors="coerce").fillna(0).astype(int).values
    t = pd.to_numeric(data[t_col], errors="coerce").fillna(0).astype(int).values

    methods: List[Tuple[str, np.ndarray]] = []

    uplift_t = t_learner(data, t_col, y_col, x_cols, base_model=base_model, model_hint=model_hint)
    methods.append(("t_learner", uplift_t))

    uplift_x = x_learner(data, t_col, y_col, x_cols, base_model=base_model, model_hint=model_hint)
    methods.append(("x_learner", uplift_x))

    uplift_cf = causal_forest(data, t_col, y_col, x_cols)
    if not np.all(np.isnan(uplift_cf)):
        methods.append(("causal_forest", uplift_cf))

    summaries = []
    row_ids = data.index.values
    for name, scores in methods:
        # Save per-user scores
        out_scores = pd.DataFrame({"row_id": row_ids, "uplift_score": scores})
        out_scores_path = os.path.join(reports_dir, f"uplift_scores_{name}.parquet")
        out_scores.to_parquet(out_scores_path, index=False)
        logger.info("Saved uplift scores for %s to %s", name, out_scores_path)

        # Metrics
        x_uplift, y_uplift = compute_uplift_curve(y, t, scores, bins=10)
        x_qini, y_qini = compute_qini_curve(y, t, scores, bins=10)
        qini = qini_coefficient(y, t, scores, bins=10)
        uplift_auc = float(np.trapz(y_uplift, x_uplift)) if len(x_uplift) > 1 else 0.0

        # Incremental conversions at top thresholds
        incr = _incremental_conversions_at_fracs(y_true=y, t=t, uplift_scores=scores, fracs=[0.1, 0.2, 0.3])

        summaries.append({
            "method": name,
            "qini": round(qini, 4),
            "uplift_auc": round(uplift_auc, 4),
            **incr,
        })

        # Plots
        _save_curve_plot(x_uplift, y_uplift, f"Uplift Curve - {name}", os.path.join(plots_dir, f"{name}_uplift_curve.png"), show=show)
        _save_curve_plot(x_qini, y_qini, f"Qini Curve - {name}", os.path.join(plots_dir, f"{name}_qini_curve.png"), show=show)
        logger.info("%s Qini=%.4f, uplift AUC=%.4f, incr@10%%=%s, incr@20%%=%s, incr@30%%=%s",
                    name, qini, uplift_auc,
                    incr.get("incr_conv_top10"), incr.get("incr_conv_top20"), incr.get("incr_conv_top30"))

    results_df = pd.DataFrame(summaries)
    results_path = os.path.join(reports_dir, "uplift_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info("📄 Saved uplift summary to %s", results_path)

    return results_df
