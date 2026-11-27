"""Explicit KS, PSI and chi-square drift statistics for the Pima feature split.

Run:  python -m ihds.drift_stats

Writes `reports/drift_stats.json` and `reports/drift_stats_psi.png`.

Compares a reference frame (the training split) to a current frame in two
scenarios:

- holdout: the held-out test split (same stratified 614/154 split as `ihds.data.split`)
- shifted: that test split with `monitoring.GLUCOSE_SHIFT` (+25 mg/dL) added to Glucose

A third GAN-synthetic scenario is not included: the generator is not cheap or
deterministic enough for a locked DVC stage.

This module does not replace Evidently. `ihds.monitoring.DataDriftPreset` (used
by `ihds/mlops/drift.py`) already runs a two-sample KS test on numeric columns
at p < 0.05 by default. This module reports that KS test explicitly, and adds
PSI and a binned chi-square test of homogeneity, plus a missing-rate comparison
(Insulin is missing in about half the rows).

PSI is computed on non-NaN values with 10 quantile bins taken from the
reference, first/last edges replaced by -inf/+inf so out-of-range current
values fall into the end bins, and proportions clipped at 1e-4 (no
renormalisation). Bin cuts can collapse on heavily tied columns (for example
Pregnancies), so the number of bins can be less than 10. The 0.10 / 0.25 PSI
bands are industry rules of thumb, not a property of this dataset; they are
configurable in `params.yaml`.

Verdict rule (documented and tested): a feature is an **alert** if PSI is
major (> 0.25) or if the Benjamini-Hochberg-adjusted KS or chi-square p-value
is below alpha. A **moderate** PSI (0.10-0.25) without a significant test is a
**watch**, not an alert. That is why holdout Insulin (PSI about 0.24, KS and
chi-square p around 0.10-0.12) does not raise an overall alert. An overall
alert fires only when at least one feature is an alert.

This module is not imported by the Streamlit app or the agent graph.
Matplotlib is used only when writing the plot and is optional for the JSON
report. The analysis is a DVC stage and a `python -m` command, not a Prefect
task.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import data, monitoring, stats_analysis
from .config import FEATURES, ROOT, TARGET

REPORTS_DIR = ROOT / "reports"
MODEL_PATH = ROOT / "models" / "risk_model.joblib"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"

DEFAULT_N_BINS = 10
DEFAULT_EPSILON = 1e-4
DEFAULT_PSI_STABLE_MAX = 0.10
DEFAULT_PSI_MAJOR_MIN = 0.25
DEFAULT_ALPHA = 0.05
DEFAULT_THRESHOLD = 0.5

VERDICT_RULE = (
    "A feature is an alert if PSI is major (greater than psi_major_min) or if the "
    "Benjamini-Hochberg-adjusted KS or chi-square p-value is below alpha. A moderate "
    "PSI (psi_stable_max to psi_major_min inclusive) without a significant test is a "
    "watch, not an alert. The overall scenario alert is true only when at least one "
    "feature is an alert."
)


def _load_settings() -> dict:
    """Defaults, overlaid with params.yaml when that file is present."""
    cfg = {
        "n_bins": DEFAULT_N_BINS,
        "epsilon": DEFAULT_EPSILON,
        "psi_stable_max": DEFAULT_PSI_STABLE_MAX,
        "psi_major_min": DEFAULT_PSI_MAJOR_MIN,
        "alpha": DEFAULT_ALPHA,
        "threshold": DEFAULT_THRESHOLD,
    }
    path = ROOT / "params.yaml"
    if not path.exists():
        return cfg
    try:
        import yaml

        params = yaml.safe_load(path.read_text()) or {}
        section = params.get("drift_stats") or {}
        train = params.get("train") or {}
        if section.get("n_bins") is not None:
            cfg["n_bins"] = int(section["n_bins"])
        if section.get("epsilon") is not None:
            cfg["epsilon"] = float(section["epsilon"])
        if section.get("psi_stable_max") is not None:
            cfg["psi_stable_max"] = float(section["psi_stable_max"])
        if section.get("psi_major_min") is not None:
            cfg["psi_major_min"] = float(section["psi_major_min"])
        if section.get("alpha") is not None:
            cfg["alpha"] = float(section["alpha"])
        if train.get("threshold") is not None:
            cfg["threshold"] = float(train["threshold"])
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


def _finite(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def missing_rate(values) -> float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return None
    return float(np.mean(~np.isfinite(arr)))


def reference_bin_edges(ref, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """Quantile edges from the reference, with ±inf end bins.

    `np.unique` is applied because integer or heavily tied columns can collapse
    some quantile cuts. A constant column becomes two bins split at that value.
    """
    ref = _finite(ref)
    if len(ref) == 0:
        raise ValueError("reference has no finite values")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges = np.asarray(edges, dtype=float)
    if len(edges) == 1:
        # Three bins: below, equal to, and above the constant. A two-bin split
        # at c would put every x >= c in the last bin, so a shifted sample
        # of larger values would look identical to the reference.
        c = edges[0]
        return np.array([-np.inf, c, np.nextafter(c, np.inf), np.inf])
    if len(edges) == 2:
        return np.array([-np.inf, edges[0], edges[1], np.inf])
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_counts(values, edges: np.ndarray) -> np.ndarray:
    return np.histogram(_finite(values), bins=edges)[0].astype(int)


def population_stability_index(
    ref,
    cur,
    n_bins: int = DEFAULT_N_BINS,
    epsilon: float = DEFAULT_EPSILON,
    edges: np.ndarray | None = None,
) -> dict:
    """PSI on finite values, with proportions clipped at `epsilon` (no renormalisation)."""
    if edges is None:
        edges = reference_bin_edges(ref, n_bins=n_bins)
    ref_c = bin_counts(ref, edges)
    cur_c = bin_counts(cur, edges)
    ref_n = int(ref_c.sum())
    cur_n = int(cur_c.sum())
    if ref_n == 0 or cur_n == 0:
        return {
            "psi": None,
            "n_bins_used": int(len(edges) - 1),
            "n_ref": ref_n,
            "n_cur": cur_n,
            "ref_counts": ref_c.tolist(),
            "cur_counts": cur_c.tolist(),
            "edges_note": "empty sample after dropping non-finite values",
        }
    pref = np.clip(ref_c / ref_n, epsilon, None)
    pcur = np.clip(cur_c / cur_n, epsilon, None)
    psi = float(np.sum((pcur - pref) * np.log(pcur / pref)))
    # Numerical noise on identical samples can be a tiny negative; clip at 0.
    if psi < 0 and abs(psi) < 1e-12:
        psi = 0.0
    return {
        "psi": psi,
        "n_bins_used": int(len(edges) - 1),
        "n_ref": ref_n,
        "n_cur": cur_n,
        "ref_counts": ref_c.tolist(),
        "cur_counts": cur_c.tolist(),
    }


def ks_two_sample(ref, cur) -> dict:
    """Two-sample Kolmogorov-Smirnov test on finite values."""
    a, b = _finite(ref), _finite(cur)
    if len(a) == 0 or len(b) == 0:
        return {"statistic": None, "pvalue": None, "n_ref": int(len(a)), "n_cur": int(len(b))}
    res = stats.ks_2samp(a, b, alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "n_ref": int(len(a)),
        "n_cur": int(len(b)),
    }


def chi_square_binned(ref, cur, edges: np.ndarray) -> dict:
    """Chi-square test of homogeneity on the same PSI bins; empty bins (both counts 0) are dropped."""
    ref_c = bin_counts(ref, edges)
    cur_c = bin_counts(cur, edges)
    mask = (ref_c + cur_c) > 0
    table = np.vstack([ref_c[mask], cur_c[mask]])
    if table.shape[1] < 2 or table[0].sum() == 0 or table[1].sum() == 0:
        return {
            "statistic": None,
            "pvalue": None,
            "dof": None,
            "n_bins_used": int(table.shape[1]),
        }
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "statistic": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
        "n_bins_used": int(table.shape[1]),
    }


def psi_band(psi, psi_stable_max: float = DEFAULT_PSI_STABLE_MAX, psi_major_min: float = DEFAULT_PSI_MAJOR_MIN) -> str:
    if psi is None:
        return "unknown"
    if psi < psi_stable_max:
        return "stable"
    if psi <= psi_major_min:
        return "moderate"
    return "major"


def feature_verdict(
    band: str,
    ks_p_bh,
    chi2_p_bh,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    """Alert on major PSI or a BH-significant test; moderate PSI alone is a watch."""
    significant = (
        (ks_p_bh is not None and ks_p_bh < alpha)
        or (chi2_p_bh is not None and chi2_p_bh < alpha)
    )
    if band == "major" or significant:
        return "alert"
    if band == "moderate":
        return "watch"
    return "stable"


def label_chi_square(ref: pd.Series, cur: pd.Series) -> dict:
    """Chi-square test of homogeneity on the binary Outcome counts."""
    labels = [0, 1]
    ref_counts = [(ref == lab).sum() for lab in labels]
    cur_counts = [(cur == lab).sum() for lab in labels]
    table = np.array([ref_counts, cur_counts], dtype=int)
    chi2, p, dof, _ = stats.chi2_contingency(table)
    return {
        "ref_counts": {"0": int(ref_counts[0]), "1": int(ref_counts[1])},
        "cur_counts": {"0": int(cur_counts[0]), "1": int(cur_counts[1])},
        "chi2_statistic": _round(chi2, 4),
        "chi2_pvalue": _py(p),
        "chi2_dof": int(dof),
        "note": "Compares the Outcome class counts in the reference and current frames. The stratified split keeps the same rate, so the holdout p-value is 1.0. The shifted scenario does not change labels.",
    }


def predicted_class_tests(ref_pred: np.ndarray, cur_pred: np.ndarray) -> dict:
    """Chi-square and two-proportion z-test on predicted-class (0/1) rates."""
    ref_pred = np.asarray(ref_pred, dtype=int)
    cur_pred = np.asarray(cur_pred, dtype=int)
    ref_counts = [int((ref_pred == 0).sum()), int((ref_pred == 1).sum())]
    cur_counts = [int((cur_pred == 0).sum()), int((cur_pred == 1).sum())]
    n_ref, n_cur = len(ref_pred), len(cur_pred)
    rate_ref = float(ref_pred.mean()) if n_ref else None
    rate_cur = float(cur_pred.mean()) if n_cur else None
    table = np.array([ref_counts, cur_counts], dtype=int)
    chi2, p_chi, dof, _ = stats.chi2_contingency(table)
    # Pooled two-proportion z-test.
    if n_ref == 0 or n_cur == 0:
        z, p_z = None, None
    else:
        p_pool = (ref_pred.sum() + cur_pred.sum()) / (n_ref + n_cur)
        denom = math.sqrt(p_pool * (1 - p_pool) * (1 / n_ref + 1 / n_cur)) if 0 < p_pool < 1 else 0.0
        if denom == 0:
            z, p_z = 0.0, 1.0
        else:
            z = (rate_ref - rate_cur) / denom
            p_z = float(2 * stats.norm.sf(abs(z)))
    return {
        "ref_counts": {"0": ref_counts[0], "1": ref_counts[1]},
        "cur_counts": {"0": cur_counts[0], "1": cur_counts[1]},
        "ref_positive_rate": _round(rate_ref, 4),
        "cur_positive_rate": _round(rate_cur, 4),
        "chi2_statistic": _round(chi2, 4),
        "chi2_pvalue": _py(p_chi),
        "chi2_dof": int(dof),
        "z_statistic": _round(z, 4),
        "z_pvalue": _py(p_z),
        "note": "Compares the model's predicted-class rate (threshold from params.yaml / train.threshold, default 0.5) on the reference vs current frames. Not used in the per-feature verdict or the overall alert.",
    }


def analyze_scenario(
    name: str,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    n_bins: int = DEFAULT_N_BINS,
    epsilon: float = DEFAULT_EPSILON,
    psi_stable_max: float = DEFAULT_PSI_STABLE_MAX,
    psi_major_min: float = DEFAULT_PSI_MAJOR_MIN,
    alpha: float = DEFAULT_ALPHA,
    pipe=None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Run KS, PSI and chi-square on each feature, then apply BH and the verdict rule."""
    features = list(features or FEATURES)
    rows = []
    edges_by_feature = {}
    for feat in features:
        ref = reference[feat]
        cur = current[feat]
        edges = reference_bin_edges(ref, n_bins=n_bins)
        edges_by_feature[feat] = edges
        ks = ks_two_sample(ref, cur)
        psi = population_stability_index(ref, cur, n_bins=n_bins, epsilon=epsilon, edges=edges)
        chi = chi_square_binned(ref, cur, edges)
        miss_ref = missing_rate(ref)
        miss_cur = missing_rate(cur)
        miss_diff = None if miss_ref is None or miss_cur is None else miss_cur - miss_ref
        rows.append(
            {
                "feature": feat,
                "n_ref": ks["n_ref"],
                "n_cur": ks["n_cur"],
                "missing_rate_ref": _round(miss_ref, 4),
                "missing_rate_cur": _round(miss_cur, 4),
                "missing_rate_diff": _round(miss_diff, 4),
                "ks_statistic": _round(ks["statistic"], 4),
                "ks_pvalue": _py(ks["pvalue"]),
                "psi": _round(psi["psi"], 4),
                "psi_n_bins": psi["n_bins_used"],
                "chi2_statistic": _round(chi["statistic"], 4),
                "chi2_pvalue": _py(chi["pvalue"]),
                "chi2_dof": chi["dof"],
            }
        )

    ks_p = [r["ks_pvalue"] for r in rows]
    chi_p = [r["chi2_pvalue"] for r in rows]
    # BH is applied separately within each test family, over features with a p-value.
    def _bh(pvals):
        idx = [i for i, p in enumerate(pvals) if p is not None]
        adj = [None] * len(pvals)
        if idx:
            corrected = stats_analysis.adjust_pvalues([pvals[i] for i in idx], "bh")
            for i, val in zip(idx, corrected):
                adj[i] = _py(val)
        return adj

    ks_bh = _bh(ks_p)
    chi_bh = _bh(chi_p)
    for i, row in enumerate(rows):
        row["ks_pvalue_bh"] = ks_bh[i]
        row["chi2_pvalue_bh"] = chi_bh[i]
        row["psi_band"] = psi_band(row["psi"], psi_stable_max, psi_major_min)
        row["verdict"] = feature_verdict(row["psi_band"], row["ks_pvalue_bh"], row["chi2_pvalue_bh"], alpha)

    ks_drifted = [r["feature"] for r in rows if r["ks_pvalue_bh"] is not None and r["ks_pvalue_bh"] < alpha]
    chi_drifted = [r["feature"] for r in rows if r["chi2_pvalue_bh"] is not None and r["chi2_pvalue_bh"] < alpha]
    psi_moderate = [r["feature"] for r in rows if r["psi_band"] == "moderate"]
    psi_major = [r["feature"] for r in rows if r["psi_band"] == "major"]
    watch = [r["feature"] for r in rows if r["verdict"] == "watch"]
    alert = [r["feature"] for r in rows if r["verdict"] == "alert"]

    out = {
        "name": name,
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "features": rows,
        "label": label_chi_square(reference[TARGET], current[TARGET]),
        "summary": {
            "ks_drifted": ks_drifted,
            "chi2_drifted": chi_drifted,
            "psi_moderate": psi_moderate,
            "psi_major": psi_major,
            "watch": watch,
            "alert": alert,
            "overall_alert": bool(alert),
        },
        "verdict_rule": VERDICT_RULE,
    }
    if pipe is not None:
        ref_pred = (pipe.predict_proba(reference[features])[:, 1] >= threshold).astype(int)
        cur_pred = (pipe.predict_proba(current[features])[:, 1] >= threshold).astype(int)
        out["predicted_class"] = predicted_class_tests(ref_pred, cur_pred)
    return out


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prefer the DVC-processed parquet files; fall back to `data.split`."""
    if TRAIN_PARQUET.exists() and TEST_PARQUET.exists():
        return pd.read_parquet(TRAIN_PARQUET), pd.read_parquet(TEST_PARQUET)
    return data.split(data.load_clean())


def load_pipe():
    if MODEL_PATH.exists():
        import joblib

        return joblib.load(MODEL_PATH)
    return None


def write_psi_plot(scenarios: dict, out_path: Path, psi_stable_max: float, psi_major_min: float) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    names = list(scenarios)
    n = len(names)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    colors = {"stable": "#4C78A8", "moderate": "#F2A202", "major": "#C44E52", "unknown": "#9E9E9E"}
    y_pos = None
    for ax, name in zip(axes, names):
        rows = scenarios[name]["features"]
        labels = [r["feature"] for r in rows][::-1]
        values = [0.0 if r["psi"] is None else r["psi"] for r in rows][::-1]
        bands = [r["psi_band"] for r in rows][::-1]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, values, color=[colors.get(b, "#9E9E9E") for b in bands], height=0.7)
        ax.axvline(psi_stable_max, color="#555555", ls="--", lw=1, label=f"stable < {psi_stable_max:g}")
        ax.axvline(psi_major_min, color="#C44E52", ls=":", lw=1, label=f"major > {psi_major_min:g}")
        ax.set_title(name)
        ax.set_xlabel("PSI")
        ax.set_xlim(0, max(1.0, max(values + [psi_major_min]) * 1.15))
        ax.set_yticks(y_pos, labels)
    axes[0].set_ylabel("Feature")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Population Stability Index (train vs current)", fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def analyze(
    reference: pd.DataFrame | None = None,
    current: pd.DataFrame | None = None,
    pipe=None,
    out_dir: Path = REPORTS_DIR,
    **kwargs,
) -> dict:
    cfg = _load_settings()
    for key in ("n_bins", "epsilon", "psi_stable_max", "psi_major_min", "alpha", "threshold"):
        kwargs.setdefault(key, cfg[key])
    if reference is None or current is None:
        reference, current = load_splits()
    if pipe is None:
        pipe = load_pipe()
    shifted = monitoring.simulate_glucose_calibration_shift(current, delta=monitoring.GLUCOSE_SHIFT)
    holdout = analyze_scenario("holdout", reference, current, pipe=pipe, **kwargs)
    shift = analyze_scenario("shifted", reference, shifted, pipe=pipe, **kwargs)
    report = {
        "n_bins": kwargs["n_bins"],
        "epsilon": kwargs["epsilon"],
        "psi_bands": {"stable_max": kwargs["psi_stable_max"], "major_min": kwargs["psi_major_min"]},
        "alpha": kwargs["alpha"],
        "threshold": kwargs["threshold"],
        "glucose_shift_mg_dl": monitoring.GLUCOSE_SHIFT,
        "n_reference": int(len(reference)),
        "n_holdout": int(len(current)),
        "verdict_rule": VERDICT_RULE,
        "note": (
            "Reference is the training split. Holdout is the test split. Shifted is the test split "
            f"with Glucose increased by {monitoring.GLUCOSE_SHIFT:g} mg/dL (monitoring.GLUCOSE_SHIFT). "
            "KS and chi-square p-values are Benjamini-Hochberg-adjusted across the eight features, "
            "separately for each test. PSI bands are industry rules of thumb. Insulin has a high "
            "missing rate (~50%), so its PSI uses about 324 / 70 finite values on this split and is "
            "noisier than the complete features. No GAN scenario is included."
        ),
        "scenarios": {"holdout": holdout, "shifted": shift},
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = write_psi_plot(
        report["scenarios"],
        out_dir / "drift_stats_psi.png",
        kwargs["psi_stable_max"],
        kwargs["psi_major_min"],
    )
    if plot_path is not None:
        rel = str(plot_path.relative_to(ROOT) if plot_path.is_relative_to(ROOT) else plot_path)
        report["plots"] = {"drift_stats_psi.png": rel}
    else:
        report["plots"] = {}
    return report


def write_report(report: dict, out_dir: Path = REPORTS_DIR) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "drift_stats.json"

    def _default(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return _py(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(report, indent=2, default=_default))
    return path


def run(out_dir: Path = REPORTS_DIR, **kwargs) -> dict:
    report = analyze(out_dir=out_dir, **kwargs)
    path = write_report(report, out_dir)
    print(f"Wrote {path}")
    for name, sc in report["scenarios"].items():
        summary = sc["summary"]
        print(
            f"{name}: overall_alert={summary['overall_alert']} "
            f"alert={summary['alert']} watch={summary['watch']} "
            f"label_chi2_p={sc['label']['chi2_pvalue']}"
        )
        if "predicted_class" in sc:
            pc = sc["predicted_class"]
            print(
                f"  predicted positive rate {pc['ref_positive_rate']} -> {pc['cur_positive_rate']} "
                f"(z p={pc['z_pvalue']})"
            )
    return report


if __name__ == "__main__":
    run()
