"""Statistical analysis of the risk model's confusion matrix and related diagnostics.

Run:  python -m ihds.cm_analysis

Writes `reports/confusion_matrix_analysis.json` and PNG plots under `reports/`.
Uses the same stratified split (`ihds.data.split`, seed 42, 614/154) and the
same sklearn Pipeline as `ihds.model` (saved `models/risk_model.joblib` when
present, otherwise retrained on the training split). Point estimates at
threshold 0.5 are computed from that model, not copied from
`reports/train_metrics.json`.

This module is not imported by the Streamlit app or the agent graph. Matplotlib
is used only when writing plots and is optional for the numeric report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from . import data, model, stats_analysis
from .config import FEATURES, ROOT, SEED, TARGET

REPORTS_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"

# Age bands match params.yaml (explain.age_bins / age_labels) and reports/fairness.json.
AGE_BINS = [21, 30, 45, 120]
AGE_LABELS = ["21-29", "30-44", "45+"]

DEFAULT_THRESHOLD = 0.5
BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 2000
COEFFICIENT_BOOTSTRAPS = 1000
N_CV_FOLDS = 5
N_CALIBRATION_BINS = 10
THRESHOLD_GRID = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
DECISION_THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.71, 0.05)]
Z95 = stats.norm.ppf(0.975)

EXPECTED_TEST_CM = {"tn": 75, "fp": 25, "fn": 16, "tp": 38}


def _load_settings() -> dict:
    """Defaults, overlaid with params.yaml when that file is present."""
    cfg = {
        "threshold": DEFAULT_THRESHOLD,
        "age_bins": list(AGE_BINS),
        "age_labels": list(AGE_LABELS),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "coefficient_bootstraps": COEFFICIENT_BOOTSTRAPS,
        "seed": SEED,
        "n_cv_folds": N_CV_FOLDS,
        "n_calibration_bins": N_CALIBRATION_BINS,
    }
    path = ROOT / "params.yaml"
    if not path.exists():
        return cfg
    try:
        import yaml

        params = yaml.safe_load(path.read_text()) or {}
        train = params.get("train") or {}
        explain = params.get("explain") or {}
        cm = params.get("cm_analysis") or {}
        if train.get("threshold") is not None:
            cfg["threshold"] = float(train["threshold"])
        if explain.get("age_bins"):
            cfg["age_bins"] = list(explain["age_bins"])
        if explain.get("age_labels"):
            cfg["age_labels"] = list(explain["age_labels"])
        for key in ("bootstrap_resamples", "permutation_resamples", "coefficient_bootstraps",
                    "seed", "n_cv_folds", "n_calibration_bins"):
            if cm.get(key) is not None:
                cfg[key] = int(cm[key])
    except Exception:  # noqa: BLE001 - keep defaults if the optional yaml read fails
        return cfg
    return cfg


def _py(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def _round(v, nd=4):
    v = _py(v)
    return None if v is None else round(v, nd)


def confusion_counts(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def metrics_from_cm(tn: int, fp: int, fn: int, tp: int) -> dict:
    """Point metrics that depend only on a 2x2 confusion matrix."""
    n = tn + fp + fn + tp
    support_0 = tn + fp
    support_1 = tp + fn
    pred_pos = tp + fp
    pred_neg = tn + fn

    def _div(a, b):
        return None if b == 0 else a / b

    acc = _div(tp + tn, n)
    sens = _div(tp, support_1)  # recall / TPR / sensitivity (class 1)
    spec = _div(tn, support_0)
    ppv = _div(tp, pred_pos)  # precision / PPV
    npv = _div(tn, pred_neg)
    fpr = _div(fp, support_0)
    fnr = _div(fn, support_1)
    prec_0 = npv  # precision of class 0
    rec_0 = spec
    f1_1 = None if not (ppv and sens) else _div(2 * ppv * sens, ppv + sens)
    f1_0 = None if not (prec_0 and rec_0) else _div(2 * prec_0 * rec_0, prec_0 + rec_0)
    bal_acc = None if sens is None or spec is None else (sens + spec) / 2
    mcc_den = math.sqrt(pred_pos * support_1 * support_0 * pred_neg) if n else 0
    mcc = None if mcc_den == 0 else (tp * tn - fp * fn) / mcc_den
    pe = None if not n else ((pred_pos / n) * (support_1 / n) + (pred_neg / n) * (support_0 / n))
    kappa = None if pe is None or pe == 1 else (acc - pe) / (1 - pe)
    lr_plus = None if not fpr else _div(sens, fpr)
    lr_minus = None if not spec else _div(fnr, spec)
    return {
        "n": int(n),
        "support_0": int(support_0),
        "support_1": int(support_1),
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": ppv,
        "recall": sens,
        "f1": f1_1,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "mcc": mcc,
        "cohens_kappa": kappa,
        "lr_plus": lr_plus,
        "lr_minus": lr_minus,
        "per_class": {
            "0": {"precision": prec_0, "recall": rec_0, "f1": f1_0, "support": int(support_0)},
            "1": {"precision": ppv, "recall": sens, "f1": f1_1, "support": int(support_1)},
        },
    }


def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None, None
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return float(max(0.0, center - margin)), float(min(1.0, center + margin))


def wilson_for_metrics(tn: int, fp: int, fn: int, tp: int, alpha: float = 0.05) -> dict:
    n = tn + fp + fn + tp
    support_0, support_1 = tn + fp, tp + fn
    pred_pos, pred_neg = tp + fp, tn + fn

    def _item(name, successes, denom):
        lo, hi = wilson_interval(successes, denom, alpha)
        point = None if denom == 0 else successes / denom
        return {
            "metric": name,
            "estimate": _round(point),
            "successes": int(successes),
            "n": int(denom),
            "ci_low": _round(lo),
            "ci_high": _round(hi),
            "method": "wilson",
            "level": 1 - alpha,
        }

    return {
        "sensitivity": _item("sensitivity", tp, support_1),
        "specificity": _item("specificity", tn, support_0),
        "ppv": _item("ppv", tp, pred_pos),
        "npv": _item("npv", tn, pred_neg),
        "accuracy": _item("accuracy", tp + tn, n),
    }


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return np.concatenate([
        rng.choice(pos, size=len(pos), replace=True),
        rng.choice(neg, size=len(neg), replace=True),
    ])


def _safe_auc(y, scores) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


def _safe_ap(y, scores) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, scores))


def stratified_bootstrap_cis(
    y: np.ndarray,
    proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    alpha: float = 0.05,
) -> dict:
    """Percentile CIs from a stratified bootstrap (class counts fixed)."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    keys = ("roc_auc", "average_precision", "f1", "mcc", "balanced_accuracy")
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = _stratified_indices(y, rng)
        yt, pt = y[idx], proba[idx]
        pred = (pt >= threshold).astype(int)
        cm = confusion_counts(yt, pred)
        m = metrics_from_cm(**cm)
        auc = _safe_auc(yt, pt)
        ap = _safe_ap(yt, pt)
        if auc is not None:
            samples["roc_auc"].append(auc)
        if ap is not None:
            samples["average_precision"].append(ap)
        if m["f1"] is not None:
            samples["f1"].append(m["f1"])
        if m["mcc"] is not None:
            samples["mcc"].append(m["mcc"])
        if m["balanced_accuracy"] is not None:
            samples["balanced_accuracy"].append(m["balanced_accuracy"])
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    pred = (proba >= threshold).astype(int)
    cm = confusion_counts(y, pred)
    m = metrics_from_cm(**cm)
    points = {
        "roc_auc": _safe_auc(y, proba),
        "average_precision": _safe_ap(y, proba),
        "f1": m["f1"],
        "mcc": m["mcc"],
        "balanced_accuracy": m["balanced_accuracy"],
    }
    out = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "stratified": True,
        "level": 1 - alpha,
        "method": "percentile, stratified by Outcome",
    }
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        lo, hi = (np.percentile(arr, [lo_q, hi_q]) if len(arr) else (np.nan, np.nan))
        out[k] = {
            "estimate": _round(points[k]),
            "ci_low": _round(float(lo) if len(arr) else None),
            "ci_high": _round(float(hi) if len(arr) else None),
            "n_valid": int(len(arr)),
        }
    return out


def delong_auc_ci(y, scores, alpha: float = 0.05) -> dict:
    """AUC with a DeLong 95% CI (DeLong, DeLong, Clarke-Pearson 1988)."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[y == 1], scores[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "se": None, "method": "deLong"}
    # phi[i, j] = 1 if pos_i > neg_j, 0.5 if tied, else 0.
    diff = pos[:, None] - neg[None, :]
    phi = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
    auc = float(phi.mean())
    v10, v01 = phi.mean(axis=1), phi.mean(axis=0)
    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    var = s10 / m + s01 / n
    se = math.sqrt(max(var, 0.0))
    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = auc - z * se, auc + z * se
    return {
        "estimate": _round(auc, 4),
        "ci_low": _round(max(0.0, lo), 4),
        "ci_high": _round(min(1.0, hi), 4),
        "se": _round(se, 4),
        "n_positive": int(m),
        "n_negative": int(n),
        "method": "deLong",
        "level": 1 - alpha,
    }


def binomial_accuracy_vs_nir(tn: int, fp: int, fn: int, tp: int) -> dict:
    """Exact binomial test of accuracy against the no-information rate (majority class)."""
    n = tn + fp + fn + tp
    correct = tp + tn
    support_0, support_1 = tn + fp, tp + fn
    majority = max(support_0, support_1)
    nir = majority / n if n else None
    acc = correct / n if n else None
    # One-sided: is accuracy greater than the majority-class rate?
    greater = stats.binomtest(correct, n, p=nir, alternative="greater") if n and nir is not None else None
    two_sided = stats.binomtest(correct, n, p=nir, alternative="two-sided") if n and nir is not None else None
    return {
        "n": int(n),
        "correct": int(correct),
        "accuracy": _round(acc),
        "no_information_rate": _round(nir),
        "majority_class": 0 if support_0 >= support_1 else 1,
        "majority_count": int(majority),
        "p_value_greater": _py(greater.pvalue) if greater is not None else None,
        "p_value_two_sided": _py(two_sided.pvalue) if two_sided is not None else None,
        "method": "exact binomial (binomtest)",
    }


def mcnemar_vs_majority(tn: int, fp: int, fn: int, tp: int) -> dict:
    """Exact McNemar test of the model vs the majority-class baseline (always predict 0 if class 0 is majority).

    When the baseline always predicts the majority class 0:
      model correct / baseline wrong = TP (true diabetic cases the model catches)
      model wrong / baseline correct = FP (non-diabetic cases the model flags)
    """
    support_0, support_1 = tn + fp, tp + fn
    if support_0 >= support_1:
        b, c = tp, fp  # baseline predicts 0
        baseline = "always 0 (majority)"
    else:
        b, c = tn, fn
        baseline = "always 1 (majority)"
    n_disc = b + c
    # Exact binomial on discordant pairs, two-sided, p=0.5. Equivalent to McNemar without continuity correction.
    p_two = stats.binomtest(b, n_disc, p=0.5, alternative="two-sided").pvalue if n_disc else None
    p_greater = stats.binomtest(b, n_disc, p=0.5, alternative="greater").pvalue if n_disc else None
    return {
        "baseline": baseline,
        "model_correct_baseline_wrong": int(b),
        "model_wrong_baseline_correct": int(c),
        "discordant": int(n_disc),
        "p_value_two_sided": _py(p_two),
        "p_value_greater": _py(p_greater),
        "method": "exact McNemar (binomial on discordant pairs)",
    }


def threshold_metrics_table(y, proba, thresholds) -> list[dict]:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in thresholds:
        cm = confusion_counts(y, (proba >= t).astype(int))
        m = metrics_from_cm(**cm)
        rows.append({
            "threshold": float(t),
            "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
            "accuracy": _round(m["accuracy"]),
            "recall": _round(m["recall"]),
            "precision": _round(m["precision"]),
            "specificity": _round(m["specificity"]),
            "f1": _round(m["f1"]),
            "mcc": _round(m["mcc"]),
            "balanced_accuracy": _round(m["balanced_accuracy"]),
        })
    return rows


def oof_predict_proba(train: pd.DataFrame, n_folds: int = N_CV_FOLDS, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predicted probabilities on the training split (stratified K-fold)."""
    y = train[TARGET].to_numpy()
    oof = np.zeros(len(train), dtype=float)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(train[FEATURES], y):
        pipe = model.train(train.iloc[tr_idx])
        oof[va_idx] = pipe.predict_proba(train.iloc[va_idx][FEATURES])[:, 1]
    return oof


def youden_threshold(y, proba, grid: np.ndarray | None = None) -> dict:
    """Threshold that maximises Youden's J = sensitivity + specificity - 1."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    if grid is None:
        # Midpoints of unique scores, plus the ends, so the search is determined by the predictions.
        uniq = np.unique(proba)
        if len(uniq) > 1:
            mid = (uniq[:-1] + uniq[1:]) / 2
            grid = np.unique(np.concatenate([[0.0], mid, [1.0], uniq]))
        else:
            grid = np.array([0.0, 0.5, 1.0])
    best_t, best_j, best_m = 0.5, -np.inf, None
    for t in grid:
        cm = confusion_counts(y, (proba >= t).astype(int))
        m = metrics_from_cm(**cm)
        if m["sensitivity"] is None or m["specificity"] is None:
            continue
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j or (j == best_j and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_j, best_m = float(t), float(j), {**cm, "sensitivity": m["sensitivity"],
                                                          "specificity": m["specificity"]}
    return {"threshold": _round(best_t, 4), "youden_j": _round(best_j, 4), "train_oof_metrics": {
        "tn": best_m["tn"], "fp": best_m["fp"], "fn": best_m["fn"], "tp": best_m["tp"],
        "sensitivity": _round(best_m["sensitivity"]), "specificity": _round(best_m["specificity"]),
    } if best_m else None}


def permutation_auc_test(y, scores, n_perm: int = PERMUTATION_RESAMPLES, seed: int = SEED) -> dict:
    """Permutation test of AUC against chance (AUC = 0.5). Labels are shuffled; scores stay fixed."""
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores, dtype=float)
    obs = _safe_auc(y, scores)
    rng = np.random.default_rng(seed)
    if obs is None:
        return {"estimate": None, "n_perm": int(n_perm), "p_value": None, "method": "permutation vs AUC=0.5"}
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        val = _safe_auc(yp, scores)
        if val is not None and val >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {
        "estimate": _round(obs),
        "null_value": 0.5,
        "n_perm": int(n_perm),
        "n_ge_observed": int(count),
        "p_value": _py(p),
        "seed": int(seed),
        "method": "permutation of labels; p = (#{AUC_perm >= AUC_obs} + 1) / (n_perm + 1)",
    }


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def hosmer_lemeshow(y, proba, n_groups: int = N_CALIBRATION_BINS) -> dict:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    frame = pd.DataFrame({"y": y, "p": proba})
    try:
        frame["bin"] = pd.qcut(frame["p"], q=n_groups, duplicates="drop")
    except ValueError:
        frame["bin"] = pd.cut(frame["p"], bins=n_groups)
    rows = []
    hl = 0.0
    for label, g in frame.groupby("bin", observed=True):
        n_g = len(g)
        o_g = float(g["y"].sum())
        e_g = float(g["p"].sum())
        denom = e_g * (1 - e_g / n_g) if n_g and e_g not in (0, n_g) else None
        term = ((o_g - e_g) ** 2 / denom) if denom else 0.0
        hl += term
        rows.append({
            "bin": str(label), "n": int(n_g), "observed": _round(o_g, 4),
            "expected": _round(e_g, 4), "mean_predicted": _round(g["p"].mean()),
            "observed_rate": _round(g["y"].mean()),
        })
    g_eff = len(rows)
    df = max(g_eff - 2, 1)
    p_value = float(stats.chi2.sf(hl, df))
    return {
        "statistic": _round(hl, 4),
        "df": int(df),
        "p_value": _py(p_value),
        "n_groups_requested": int(n_groups),
        "n_groups": int(g_eff),
        "bins": rows,
        "method": "Hosmer-Lemeshow, quantile bins of predicted probability, df = G-2",
    }


def expected_calibration_error(y, proba, n_bins: int = N_CALIBRATION_BINS) -> dict:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    points = []
    n = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (proba >= lo) & (proba <= hi if i == n_bins - 1 else proba < hi)
        if not mask.any():
            continue
        conf = float(proba[mask].mean())
        acc = float(y[mask].mean())
        w = float(mask.mean())
        ece += w * abs(acc - conf)
        points.append({
            "bin": i, "bin_left": _round(lo), "bin_right": _round(hi),
            "n": int(mask.sum()), "mean_predicted": _round(conf), "observed_rate": _round(acc),
        })
    return {"ece": _round(ece, 4), "n_bins": int(n_bins), "scheme": "equal-width on [0, 1]", "bins": points, "n": int(n)}


def calibration_slope_intercept(y, proba) -> dict:
    """Logistic recalibration: logit P(Y=1) = a + b * logit(p_hat). Perfect calibration is a=0, b=1.

    Fit by unregularized maximum likelihood (statsmodels when installed, else Newton on the same likelihood).
    This is a diagnostic on the already-fitted model's scores; it is not how the deployed model is trained.
    """
    y = np.asarray(y).astype(int)
    lp = _logit(proba)
    X = np.column_stack([np.ones(len(lp)), lp])
    intercept = slope = se_a = se_b = p_a = p_b = None
    method = "unregularized logistic MLE"
    try:
        import statsmodels.api as sm  # optional; not a runtime pin

        fit = sm.Logit(y, X).fit(disp=False)
        intercept, slope = float(fit.params[0]), float(fit.params[1])
        se_a, se_b = float(fit.bse[0]), float(fit.bse[1])
        p_a, p_b = float(fit.pvalues[0]), float(fit.pvalues[1])
        method = "statsmodels Logit MLE"
    except Exception:  # noqa: BLE001 - fall back if statsmodels is missing or the fit fails
        # Newton-Raphson for unpenalized logistic regression.
        beta = np.zeros(2)
        ok = False
        for _ in range(50):
            eta = X @ beta
            p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
            w = p * (1 - p)
            grad = X.T @ (y - p)
            hess = X.T @ (X * w[:, None])
            try:
                step = np.linalg.solve(hess + 1e-8 * np.eye(2), grad)
            except np.linalg.LinAlgError:
                break
            beta = beta + step
            if np.abs(step).max() < 1e-8:
                ok = True
                break
        if ok:
            intercept, slope = float(beta[0]), float(beta[1])
            method = "Newton MLE (statsmodels unavailable)"
    citl_mean_diff = float(np.mean(proba) - np.mean(y))
    # Intercept-only recalibration with slope fixed at 1 (calibration-in-the-large).
    offset = lp
    # a_hat = logit(mean y adjusted...) — fit intercept-only with offset logit(p).
    a_citl = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(a_citl + offset, -30, 30)))
        grad = float(np.sum(y - p))
        hess = float(-np.sum(p * (1 - p)))
        if hess == 0:
            break
        step = grad / (-hess) if hess else 0
        a_citl += step
        if abs(step) < 1e-10:
            break
    return {
        "intercept": _round(intercept, 4),
        "slope": _round(slope, 4),
        "intercept_se": _round(se_a, 4),
        "slope_se": _round(se_b, 4),
        "intercept_p_value": _py(p_a),
        "slope_p_value_vs_zero": _py(p_b),
        "calibration_in_the_large_mean_predicted_minus_base_rate": _round(citl_mean_diff, 4),
        "calibration_in_the_large_intercept": _round(a_citl, 4),
        "mean_predicted": _round(float(np.mean(proba))),
        "base_rate": _round(float(np.mean(y))),
        "method": method,
        "note": "Perfect calibration is intercept 0 and slope 1. The deployed model uses class_weight=balanced, which can shift probabilities upward.",
    }


def decision_curve(y, proba, thresholds) -> list[dict]:
    """Net benefit of treating when p >= pt, versus treat-all and treat-none."""
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba, dtype=float)
    n = len(y)
    prev = float(y.mean()) if n else 0.0
    rows = []
    for t in thresholds:
        if t <= 0 or t >= 1:
            continue
        pred = proba >= t
        tp = int(((y == 1) & pred).sum())
        fp = int(((y == 0) & pred).sum())
        odds = t / (1 - t)
        nb = tp / n - fp / n * odds
        nb_all = prev - (1 - prev) * odds
        rows.append({
            "threshold": float(t),
            "net_benefit": _round(nb, 4),
            "treat_all": _round(nb_all, 4),
            "treat_none": 0.0,
            "tp": tp,
            "fp": fp,
        })
    return rows


def age_band_tables(test: pd.DataFrame, proba, threshold: float = DEFAULT_THRESHOLD) -> dict:
    y = test[TARGET].to_numpy().astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    bands = pd.cut(test["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    by = []
    tp_fn = []  # among positives, for the heterogeneity test
    for label in AGE_LABELS:
        mask = bands == label
        cm = confusion_counts(y[mask], pred[mask])
        m = metrics_from_cm(**cm)
        w = wilson_for_metrics(**cm)
        by.append({
            "age_band": label,
            "n": int(mask.sum()),
            "positives": int(y[mask].sum()),
            "confusion_matrix": cm,
            "recall": _round(m["recall"]),
            "precision": _round(m["precision"]),
            "specificity": _round(m["specificity"]),
            "accuracy": _round(m["accuracy"]),
            "recall_wilson": w["sensitivity"],
            "specificity_wilson": w["specificity"],
            "accuracy_wilson": w["accuracy"],
        })
        tp_fn.append([cm["tp"], cm["fn"]])  # detected vs missed among positives
    table = np.asarray(tp_fn, dtype=int)
    hetero = _recall_heterogeneity(table, AGE_LABELS)
    return {
        "bins": AGE_BINS,
        "labels": AGE_LABELS,
        "split": "test",
        "threshold": threshold,
        "note": "Test bands are small (tens of patients, 12–26 positives). A recall of 1.00 in 45+ is 12/12 and is not a precise rate. The 21–29 band has the lowest recall.",
        "by_band": by,
        "recall_heterogeneity": hetero,
    }


def _recall_heterogeneity(table: np.ndarray, labels: list[str], n_perm: int = 5000, seed: int = SEED) -> dict:
    """Does recall differ across age bands? `table` is (n_bands x 2) of [TP, FN] among positives."""
    table = np.asarray(table, dtype=int)
    # Drop bands with no positives.
    keep = table.sum(axis=1) > 0
    tab = table[keep]
    labs = [lb for lb, k in zip(labels, keep) if k]
    if tab.shape[0] < 2 or tab.sum() == 0:
        return {"p_value_chi2": None, "p_value_permutation": None}
    chi2, p_chi2, df, expected = stats.chi2_contingency(tab)
    # Monte Carlo: among positives, shuffle 'detected' and recompute chi-square (bands fixed).
    band_ids = np.concatenate([np.repeat(i, tab[i].sum()) for i in range(len(labs))])
    flags = np.concatenate([np.array([1] * int(tab[i, 0]) + [0] * int(tab[i, 1])) for i in range(len(labs))])
    rng = np.random.default_rng(seed)

    def _chi2_of(f):
        counts = np.zeros_like(tab)
        for i in range(len(labs)):
            sel = band_ids == i
            counts[i, 0] = int(f[sel].sum())
            counts[i, 1] = int(sel.sum() - counts[i, 0])
        # chi2 without p-value; zero expected would be a problem but margins are fixed so expected is stable
        return stats.chi2_contingency(counts)[0]

    count = 0
    for _ in range(n_perm):
        if _chi2_of(rng.permutation(flags)) >= chi2 - 1e-12:
            count += 1
    p_perm = (count + 1) / (n_perm + 1)
    pairwise = []
    pairs = []
    pvals = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            sub = np.array([tab[i], tab[j]])
            if sub.sum() == 0 or (sub[0].sum() == 0) or (sub[1].sum() == 0):
                p = None
            else:
                # fisher_exact wants [[tp_i, fn_i], [tp_j, fn_j]]
                p = float(stats.fisher_exact(sub)[1])
            pairwise.append({"bands": [labs[i], labs[j]], "table": sub.tolist(), "p_value_fisher": p})
            pairs.append((labs[i], labs[j]))
            pvals.append(np.nan if p is None else p)
    valid = [k for k, p in enumerate(pvals) if np.isfinite(p)]
    adj = np.full(len(pvals), np.nan)
    if valid:
        adj_vals = stats_analysis.adjust_pvalues([pvals[k] for k in valid], "bh")
        for k, a in zip(valid, adj_vals):
            adj[k] = a
    for row, a in zip(pairwise, adj):
        row["p_value_fisher_bh"] = None if not np.isfinite(a) else _py(float(a))
    return {
        "table": [{"age_band": lab, "tp": int(r[0]), "fn": int(r[1])} for lab, r in zip(labs, tab)],
        "chi2": _round(float(chi2), 4),
        "df": int(df),
        "p_value_chi2": _py(float(p_chi2)),
        "expected": [[_round(float(x), 2) for x in row] for row in expected],
        "p_value_permutation": _py(p_perm),
        "n_perm": int(n_perm),
        "pairwise_fisher": pairwise,
        "note": "Chi-square can be unreliable with a zero cell (e.g. 12/12 detected in 45+). The permutation p-value and pairwise Fisher tests are the more appropriate summaries here.",
    }


def coefficient_table(train: pd.DataFrame, n_boot: int = COEFFICIENT_BOOTSTRAPS, seed: int = SEED,
                      alpha: float = 0.05) -> dict:
    """Coefficients of the deployed pipeline (standardized features) with bootstrap CIs.

    Optional statsmodels Logit p-values are from an *unregularized, unweighted* MLE on the
    same imputed-and-standardized training matrix. They do not describe the deployed
    LogisticRegression (C=1, class_weight='balanced') and are labelled as such.
    """
    pipe = model.train(train)
    coef = pipe.named_steps["clf"].coef_[0]
    intercept = float(pipe.named_steps["clf"].intercept_[0])
    rng = np.random.default_rng(seed)
    boots = np.zeros((n_boot, len(FEATURES)))
    intercepts = np.zeros(n_boot)
    y = train[TARGET].to_numpy()
    for i in range(n_boot):
        idx = _stratified_indices(y, rng)
        fitted = model.train(train.iloc[idx].reset_index(drop=True))
        boots[i] = fitted.named_steps["clf"].coef_[0]
        intercepts[i] = fitted.named_steps["clf"].intercept_[0]
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    lo = np.percentile(boots, lo_q, axis=0)
    hi = np.percentile(boots, hi_q, axis=0)
    int_lo, int_hi = np.percentile(intercepts, [lo_q, hi_q])

    sm_rows = None
    sm_note = "statsmodels was not used (package missing or fit failed)."
    try:
        import statsmodels.api as sm  # optional

        X = pipe.named_steps["scale"].transform(pipe.named_steps["impute"].transform(train[FEATURES]))
        fit = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=False)
        sm_rows = []
        # params[0] is intercept, 1..k features in FEATURES order
        params = np.asarray(fit.params, dtype=float)
        ci = np.asarray(fit.conf_int(), dtype=float)
        bse = np.asarray(fit.bse, dtype=float)
        pvals = np.asarray(fit.pvalues, dtype=float)
        for i, name in enumerate(["intercept", *FEATURES]):
            sm_rows.append({
                "feature": name,
                "coefficient": _round(float(params[i]), 4),
                "odds_ratio": _round(float(np.exp(params[i])), 4),
                "ci_low": _round(float(ci[i, 0]), 4),
                "ci_high": _round(float(ci[i, 1]), 4),
                "p_value": _py(float(pvals[i])),
                "se": _round(float(bse[i]), 4),
            })
        sm_note = (
            "Unregularized, unweighted Logit MLE on the imputed and standardized training matrix. "
            "This is not the deployed model (which uses C=1 L2 and class_weight='balanced'), so "
            "coefficients and p-values will not match the sklearn / bootstrap columns."
        )
    except Exception as exc:  # noqa: BLE001
        sm_note = f"statsmodels inferential fit skipped: {type(exc).__name__}"

    rows = []
    for i, name in enumerate(FEATURES):
        rows.append({
            "feature": name,
            "coefficient": _round(float(coef[i]), 4),
            "odds_ratio_per_sd": _round(float(np.exp(coef[i])), 4),
            "ci_low": _round(float(lo[i]), 4),
            "ci_high": _round(float(hi[i]), 4),
            "odds_ratio_ci_low": _round(float(np.exp(lo[i])), 4),
            "odds_ratio_ci_high": _round(float(np.exp(hi[i])), 4),
        })
    rows = sorted(rows, key=lambda r: abs(r["coefficient"] or 0), reverse=True)
    return {
        "space": "standardized features (StandardScaler inside the app pipeline)",
        "model": "sklearn LogisticRegression(C=1, class_weight='balanced', max_iter=1000)",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "level": 1 - alpha,
        "method": "percentile CI from a stratified bootstrap of the full training pipeline",
        "intercept": {
            "coefficient": _round(intercept, 4),
            "ci_low": _round(float(int_lo), 4),
            "ci_high": _round(float(int_hi), 4),
        },
        "features": rows,
        "unregularized_mle": {"rows": sm_rows, "note": sm_note},
    }


def _round_metrics(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if k == "per_class":
            out[k] = {cls: {kk: (_round(vv) if kk != "support" else int(vv)) for kk, vv in vals.items()}
                      for cls, vals in v.items()}
        elif k in ("n", "support_0", "support_1"):
            out[k] = int(v)
        else:
            out[k] = _round(v)
    return out


def load_or_train(train: pd.DataFrame):
    if MODEL_PATH.exists():
        import joblib

        pipe = joblib.load(MODEL_PATH)
        return pipe, str(MODEL_PATH.relative_to(ROOT))
    return model.train(train), "retrained (models/risk_model.joblib was not present)"


def analyze(
    train: pd.DataFrame | None = None,
    test: pd.DataFrame | None = None,
    pipe=None,
    threshold: float = DEFAULT_THRESHOLD,
    n_boot: int = BOOTSTRAP_RESAMPLES,
    n_perm: int = PERMUTATION_RESAMPLES,
    n_coef_boot: int = COEFFICIENT_BOOTSTRAPS,
    seed: int = SEED,
    plot: bool = True,
    out_dir: Path = REPORTS_DIR,
) -> dict:
    if train is None or test is None:
        train, test = data.split(data.load_clean())
    source = "caller-supplied pipeline"
    if pipe is None:
        pipe, source = load_or_train(train)
    y = test[TARGET].to_numpy().astype(int)
    proba = pipe.predict_proba(test[FEATURES])[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_counts(y, pred)
    metrics = metrics_from_cm(**cm)
    wilson = wilson_for_metrics(**cm)
    boot = stratified_bootstrap_cis(y, proba, threshold, n_boot=n_boot, seed=seed)
    delong = delong_auc_ci(y, proba)
    acc_nir = binomial_accuracy_vs_nir(**cm)
    mc = mcnemar_vs_majority(**cm)
    sweep = threshold_metrics_table(y, proba, THRESHOLD_GRID)

    oof = oof_predict_proba(train, N_CV_FOLDS, seed)
    y_train = train[TARGET].to_numpy().astype(int)
    youden = youden_threshold(y_train, oof)
    youden_t = youden["threshold"]
    youden_test_cm = confusion_counts(y, (proba >= youden_t).astype(int))
    youden_test_metrics = _round_metrics(metrics_from_cm(**youden_test_cm))
    cv_cm = confusion_counts(y_train, (oof >= threshold).astype(int))
    cv_metrics = _round_metrics(metrics_from_cm(**cv_cm))
    cv_auc = _safe_auc(y_train, oof)

    bands = age_band_tables(test, proba, threshold)
    brier = float(brier_score_loss(y, proba))
    hl = hosmer_lemeshow(y, proba)
    ece = expected_calibration_error(y, proba)
    calib = calibration_slope_intercept(y, proba)
    dca = decision_curve(y, proba, DECISION_THRESHOLDS)
    # Fine grid for the plot only.
    dca_fine = decision_curve(y, proba, [round(x, 3) for x in np.arange(0.01, 0.91, 0.01)])
    perm = permutation_auc_test(y, proba, n_perm=n_perm, seed=seed)
    ap = _safe_ap(y, proba)
    prevalence = float(y.mean())
    coefs = coefficient_table(train, n_boot=n_coef_boot, seed=seed)
    welch = stats_analysis.welch_tests(data.load_clean())

    plots = {}
    if plot:
        plots = write_plots(out_dir, y, proba, cm, dca_fine, prevalence)

    report = {
        "model_source": source,
        "split": {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "seed": int(seed),
            "threshold": float(threshold),
            "test_positives": int(y.sum()),
            "test_negatives": int((y == 0).sum()),
        },
        "test_confusion_matrix": cm,
        "test_confusion_matrix_normalized_by_true_class": {
            "tn": _round(cm["tn"] / max(cm["tn"] + cm["fp"], 1)),
            "fp": _round(cm["fp"] / max(cm["tn"] + cm["fp"], 1)),
            "fn": _round(cm["fn"] / max(cm["tp"] + cm["fn"], 1)),
            "tp": _round(cm["tp"] / max(cm["tp"] + cm["fn"], 1)),
        },
        "test_metrics": _round_metrics(metrics),
        "wilson_cis": wilson,
        "bootstrap_cis": boot,
        "auc_delong": delong,
        "average_precision": {
            "estimate": _round(ap),
            "prevalence_baseline": _round(prevalence),
            "bootstrap": boot["average_precision"],
        },
        "significance": {
            "accuracy_vs_no_information_rate": acc_nir,
            "mcnemar_vs_majority_baseline": mc,
            "permutation_auc_vs_chance": perm,
        },
        "threshold_analysis": {
            "note": "The Youden threshold is chosen on out-of-fold predictions from 5-fold stratified CV on the training split, then applied to the test split. The grid table below is reported on the test split and is descriptive, not used for selection.",
            "grid_on_test": sweep,
            "youden_from_train_oof": youden,
            "youden_applied_to_test": {"threshold": youden_t, "confusion_matrix": youden_test_cm,
                                       "metrics": youden_test_metrics},
        },
        "cv_pooled_train": {
            "note": "Pooled out-of-fold predictions, 5-fold stratified CV on the 614-row training split, threshold 0.5. Compared with the single 154-row test matrix to show split-to-split variation.",
            "n_folds": N_CV_FOLDS,
            "n": int(len(train)),
            "confusion_matrix": cv_cm,
            "metrics": cv_metrics,
            "roc_auc": _round(cv_auc),
        },
        "age_bands": bands,
        "calibration": {
            "brier_score": _round(brier, 4),
            **calib,
            "hosmer_lemeshow": hl,
            "expected_calibration_error": ece,
        },
        "decision_curve": {
            "note": "Net benefit on the test split. treat-none is 0; treat-all is prevalence - (1-prevalence)*pt/(1-pt). Positive net benefit above both references would support using the model at that threshold. This is a descriptive plot on n=154, not a decision-policy recommendation.",
            "table": dca,
        },
        "coefficients": coefs,
        "welch_tests": {
            "note": "Welch two-sample t-tests per feature on the published 768-row cleaned table (not the holdout alone), with Cohen's d and Bonferroni / Benjamini-Hochberg adjustments over the eight features.",
            "rows": welch.to_dict(orient="records"),
        },
        "plots": plots,
    }
    return report


def write_plots(out_dir: Path, y, proba, cm: dict, dca_fine: list[dict], prevalence: float) -> dict:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        from sklearn.calibration import calibration_curve
    except ImportError:
        return {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _save(fig, name):
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    # 1) Confusion matrix, raw and row-normalized.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    raw = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], dtype=float)
    row_n = raw / raw.sum(axis=1, keepdims=True)
    for ax, mat, title, fmt in (
        (axes[0], raw, "Test confusion matrix (counts)", "{:.0f}"),
        (axes[1], row_n, "Normalized by true class", "{:.2f}"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0 if title.endswith("counts)") else 0, vmax=None if title.endswith("counts)") else 1)
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"])
        ax.set_title(title)
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, fmt.format(val), ha="center", va="center",
                    color="white" if val > (mat.max() / 2 if mat.max() else 1) else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "confusion_matrix.png")

    # 2) ROC
    fpr, tpr, _ = roc_curve(y, proba)
    auc = float(roc_auc_score(y, proba))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(fpr, tpr, label=f"Logistic regression (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", c="0.5", label="Chance (AUC 0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Test-set ROC curve")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    _save(fig, "roc_curve.png")

    # 3) Precision-recall
    prec, rec, _ = precision_recall_curve(y, proba)
    ap = float(average_precision_score(y, proba))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(rec, prec, label=f"Logistic regression (AP {ap:.3f})")
    ax.axhline(prevalence, ls="--", c="0.5", label=f"Prevalence baseline ({prevalence:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test-set precision-recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left")
    _save(fig, "pr_curve.png")

    # 4) Reliability curve
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=N_CALIBRATION_BINS, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], ls="--", c="0.5", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label="Test predictions")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed diabetic fraction")
    ax.set_title("Test-set reliability curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    _save(fig, "calibration_curve.png")

    # 5) Decision curve
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ts = [r["threshold"] for r in dca_fine]
    ax.plot(ts, [r["net_benefit"] for r in dca_fine], label="Model")
    ax.plot(ts, [r["treat_all"] for r in dca_fine], ls="--", label="Treat all")
    ax.plot(ts, [r["treat_none"] for r in dca_fine], ls=":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis (test split)")
    ax.set_xlim(0.05, 0.75)
    ymin = min(-0.05, min(r["net_benefit"] for r in dca_fine) - 0.02)
    ax.set_ylim(ymin, max(0.4, max(r["net_benefit"] for r in dca_fine) + 0.02))
    ax.legend(loc="upper right")
    _save(fig, "decision_curve.png")
    return paths


def write_report(report: dict, out_dir: Path = REPORTS_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "confusion_matrix_analysis.json"

    def _default(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return _py(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(report, indent=2, default=_default))
    return path


def run(out_dir: Path = REPORTS_DIR, **kwargs) -> dict:
    cfg = _load_settings()
    kwargs.setdefault("threshold", cfg["threshold"])
    kwargs.setdefault("n_boot", cfg["bootstrap_resamples"])
    kwargs.setdefault("n_perm", cfg["permutation_resamples"])
    kwargs.setdefault("n_coef_boot", cfg["coefficient_bootstraps"])
    kwargs.setdefault("seed", cfg["seed"])
    report = analyze(out_dir=out_dir, **kwargs)
    path = write_report(report, out_dir)
    cm = report["test_confusion_matrix"]
    m = report["test_metrics"]
    print(f"Wrote {path}")
    print(f"Test CM at 0.5: TN {cm['tn']} FP {cm['fp']} FN {cm['fn']} TP {cm['tp']}; "
          f"acc={m['accuracy']} rec={m['recall']} auc={report['auc_delong']['estimate']}")
    return report


if __name__ == "__main__":
    run()
