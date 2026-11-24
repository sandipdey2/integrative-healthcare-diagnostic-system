"""Evidently AI monitoring.

Builds three reports with the `evidently` library (0.7.x API):

1. holdout   - reference = training split (with model predictions),
               current   = held-out test split. Data drift + classification quality.
2. shifted   - reference = training split, current = test split with a SIMULATED
               glucose-meter calibration error (+25 mg/dL on Glucose) applied to the
               features while the true labels stay unchanged. Shows that Evidently
               flags the drift and that classification quality degrades.
3. synthetic - reference = training split (imputed), current = GAN synthetic rows.
               Data drift only (synthetic rows have no labels): a fidelity check of the
               generator, in the spirit of evaluating generated samples in generative-mnist.

Each report is saved as HTML and JSON in monitoring/reports/, and a compact
summary.json with the headline numbers is written alongside.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from .config import FEATURES, REPORTS_DIR, TARGET

GLUCOSE_SHIFT = 25.0


def with_predictions(pipe, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out["prediction_proba"] = pipe.predict_proba(out[FEATURES])[:, 1]
    out["prediction"] = (out["prediction_proba"] >= threshold).astype(int)
    return out


def simulate_glucose_calibration_shift(df: pd.DataFrame, delta: float = GLUCOSE_SHIFT) -> pd.DataFrame:
    out = df.copy()
    out["Glucose"] = out["Glucose"] + delta
    return out


def _definition(classification: bool) -> DataDefinition:
    if classification:
        return DataDefinition(
            numerical_columns=FEATURES,
            classification=[BinaryClassification(target=TARGET, prediction_labels="prediction")],
        )
    return DataDefinition(numerical_columns=FEATURES)


def _summarize(snapshot) -> dict:
    """Pull headline numbers out of an Evidently snapshot dict."""
    d = snapshot.dict()
    out = {"drifted_columns": {}, "drift_p_values": {}}
    for m in d["metrics"]:
        name, val = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            out["drifted_columns_count"] = int(val["count"])
            out["drifted_columns_share"] = round(float(val["share"]), 3)
        elif name.startswith("ValueDrift("):
            col = m["config"]["column"]
            p = float(val)
            out["drift_p_values"][col] = round(p, 5)
            out["drifted_columns"][col] = p < m["config"].get("threshold", 0.05)
        else:
            for key in ("Accuracy", "Precision", "Recall", "F1Score", "RocAuc"):
                if name.startswith(key + "(") and isinstance(val, (int, float)):
                    out[key.lower()] = round(float(val), 4)
    out["drifted_feature_names"] = [c for c, v in out["drifted_columns"].items() if v and c in FEATURES]
    return out


def run_report(reference: pd.DataFrame, current: pd.DataFrame, name: str, classification: bool,
               out_dir: Path = REPORTS_DIR) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = FEATURES + ([TARGET, "prediction"] if classification else [])
    definition = _definition(classification)
    metrics = [DataDriftPreset(columns=FEATURES)] + ([ClassificationPreset()] if classification else [])
    snapshot = Report(metrics).run(
        Dataset.from_pandas(current[cols].reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference[cols].reset_index(drop=True), data_definition=definition),
    )
    html, js = out_dir / f"{name}_report.html", out_dir / f"{name}_report.json"
    snapshot.save_html(str(html))
    snapshot.save_json(str(js))
    summary = _summarize(snapshot)
    summary.update({"report": name, "html": html.name, "json": js.name,
                    "reference_rows": int(len(reference)), "current_rows": int(len(current))})
    return summary


def run_all(pipe, train: pd.DataFrame, test: pd.DataFrame, synthetic: pd.DataFrame | None = None,
            out_dir: Path = REPORTS_DIR) -> dict:
    ref = with_predictions(pipe, train)
    results = {
        "holdout": run_report(ref, with_predictions(pipe, test), "holdout", True, out_dir),
        "shifted": run_report(ref, with_predictions(pipe, simulate_glucose_calibration_shift(test)), "shifted", True, out_dir),
    }
    if synthetic is not None:
        imputed_ref = pd.DataFrame(pipe.named_steps["impute"].transform(train[FEATURES]), columns=FEATURES)
        results["synthetic"] = run_report(imputed_ref, synthetic, "synthetic", False, out_dir)
    results["notes"] = {
        "shifted": f"Simulated +{GLUCOSE_SHIFT:g} mg/dL glucose calibration error on the test split; labels unchanged.",
        "synthetic": "GAN rows vs imputed training rows; drift here means the generator did not match that feature's distribution.",
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "summary.json").write_text(json.dumps(results, indent=2))
    return results


def load_summary(out_dir: Path = REPORTS_DIR) -> dict | None:
    p = Path(out_dir) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ agent-output evaluation
AGENT_NUMERIC = ["probability", "grounding_score", "revisions", "tool_calls", "draft_words"]


def agent_outputs_frame(results, labels=None, case_ids=None) -> pd.DataFrame:
    """One row per agent run (AdvisoryResult) with the fields Evidently evaluates."""
    rows = []
    for i, r in enumerate(results):
        rows.append({
            "case_id": case_ids[i] if case_ids is not None else i,
            "status": r.status,
            "probability": r.probability if r.probability is not None else float("nan"),
            "prediction": int(r.probability >= 0.5) if r.probability is not None else -1,
            "grounding_score": r.review.get("grounding_score", float("nan")),
            "approved": int(bool(r.review.get("approved", False))),
            "revisions": int(r.revisions),
            "escalated": int(r.escalated),
            "tool_calls": len(r.tool_calls),
            "draft_words": len(r.draft.split()),
            "draft": r.draft,
        })
    df = pd.DataFrame(rows)
    if labels is not None:
        df[TARGET] = list(labels)
    return df


def _text_descriptors():
    from evidently.descriptors import Contains, DoesNotContain, RegExp, SentenceCount, TextLength

    return [
        TextLength("draft", alias="draft_length"),
        SentenceCount("draft", alias="draft_sentences"),
        RegExp("draft", reg_exp=r"\[K\d+\]", alias="has_citation"),
        Contains("draft", items=["not a diagnosis"], case_sensitive=False, alias="has_disclaimer"),
        DoesNotContain("draft", items=["you have diabetes", "diagnosed with", "prescribe", "mg of"],
                       case_sensitive=False, alias="no_forbidden_phrases"),
    ]


def run_agent_eval(holdout: pd.DataFrame, shifted: pd.DataFrame, guard_cases: pd.DataFrame,
                   out_dir: Path = REPORTS_DIR) -> dict:
    """Evidently evaluation of the LangGraph agents' outputs.

    agent_text_eval   - TextEvals descriptors on every released draft (holdout runs) + agent classification
                        quality (Outcome vs prediction) .
    agent_output_drift- drift of agent outputs (probability, grounding, revisions, tool calls, length):
                        holdout runs (reference) vs runs on the simulated glucose-shifted inputs (current).
    guard_eval        - safety guard as a classifier: expected_blocked vs predicted_blocked on labelled cases.
    """
    import warnings

    from evidently.presets import TextEvals

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # constant columns (e.g. revisions=0) in Evidently stats
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    ok = holdout[holdout["status"] == "ok"].reset_index(drop=True)
    dd = DataDefinition(text_columns=["draft"], numerical_columns=AGENT_NUMERIC,
                        classification=[BinaryClassification(target=TARGET, prediction_labels="prediction")])
    ds = Dataset.from_pandas(ok[["draft", TARGET, "prediction"] + AGENT_NUMERIC], data_definition=dd,
                             descriptors=_text_descriptors())
    snap = Report([TextEvals(), ClassificationPreset()]).run(ds, None)
    snap.save_html(str(out_dir / "agent_text_eval_report.html"))
    snap.save_json(str(out_dir / "agent_text_eval_report.json"))
    desc = ds.as_dataframe()
    cls = _summarize(snap)
    summary["agent_text_eval"] = {
        "runs": int(len(ok)),
        "has_citation_rate": round(float(desc["has_citation"].mean()), 3),
        "has_disclaimer_rate": round(float(desc["has_disclaimer"].mean()), 3),
        "no_forbidden_phrases_rate": round(float(desc["no_forbidden_phrases"].mean()), 3),
        "mean_draft_length_chars": round(float(desc["draft_length"].mean()), 1),
        "reviewer_approved_rate": round(float(ok["approved"].mean()), 3),
        "mean_grounding": round(float(ok["grounding_score"].mean()), 3),
        "accuracy": cls.get("accuracy"), "precision": cls.get("precision"), "recall": cls.get("recall"),
        "html": "agent_text_eval_report.html",
    }

    sh = shifted[shifted["status"] == "ok"].reset_index(drop=True)
    num_dd = DataDefinition(numerical_columns=AGENT_NUMERIC)
    snap = Report([DataDriftPreset(columns=AGENT_NUMERIC)]).run(
        Dataset.from_pandas(sh[AGENT_NUMERIC], data_definition=num_dd),
        Dataset.from_pandas(ok[AGENT_NUMERIC], data_definition=num_dd))
    snap.save_html(str(out_dir / "agent_output_drift_report.html"))
    snap.save_json(str(out_dir / "agent_output_drift_report.json"))
    s = _summarize(snap)
    summary["agent_output_drift"] = {
        "drifted_outputs": [c for c, v in s["drifted_columns"].items() if v],
        "drift_p_values": s["drift_p_values"],
        "mean_probability_holdout": round(float(ok["probability"].mean()), 3),
        "mean_probability_shifted": round(float(sh["probability"].mean()), 3),
        "html": "agent_output_drift_report.html",
    }

    gdd = DataDefinition(categorical_columns=["category"],
                         classification=[BinaryClassification(target="expected_blocked", prediction_labels="predicted_blocked")])
    snap = Report([ClassificationPreset()]).run(
        Dataset.from_pandas(guard_cases[["category", "expected_blocked", "predicted_blocked"]], data_definition=gdd), None)
    snap.save_html(str(out_dir / "guard_eval_report.html"))
    snap.save_json(str(out_dir / "guard_eval_report.json"))
    g = _summarize(snap)
    missed = guard_cases[(guard_cases.expected_blocked == 1) & (guard_cases.predicted_blocked == 0)]
    summary["guard_eval"] = {
        "cases": int(len(guard_cases)), "accuracy": g.get("accuracy"), "precision": g.get("precision"),
        "recall": g.get("recall"), "missed": missed["description"].tolist(), "html": "guard_eval_report.html",
    }
    (out_dir / "agent_eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_agent_summary(out_dir: Path = REPORTS_DIR) -> dict | None:
    p = Path(out_dir) / "agent_eval_summary.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ live LLM vs template drafts
LLM_EVAL_NUMERIC = ["grounding_score", "approved", "revisions", "escalated", "n_citations", "latency_s"]


def run_llm_vs_template_eval(llm_df: pd.DataFrame, template_df: pd.DataFrame, out_dir: Path = REPORTS_DIR) -> dict:
    """Evidently TextEvals on live gpt-4o-mini drafts (current) vs template drafts (reference) for the same cases.

    Both frames need: draft, grounding_score, approved, revisions, escalated, n_citations, latency_s.
    """
    import warnings

    from evidently.presets import TextEvals

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dd = DataDefinition(text_columns=["draft"], numerical_columns=LLM_EVAL_NUMERIC)
    cols = ["draft"] + LLM_EVAL_NUMERIC
    cur = Dataset.from_pandas(llm_df[cols].reset_index(drop=True), data_definition=dd, descriptors=_text_descriptors())
    ref = Dataset.from_pandas(template_df[cols].reset_index(drop=True), data_definition=dd, descriptors=_text_descriptors())
    snap = Report([TextEvals(), DataDriftPreset(columns=["grounding_score", "n_citations"])]).run(cur, ref)
    snap.save_html(str(out_dir / "llm_vs_template_report.html"))
    snap.save_json(str(out_dir / "llm_vs_template_report.json"))

    def stats(ds, df):
        d = ds.as_dataframe()
        return {
            "runs": int(len(df)),
            "reviewer_approved_first_try": round(float(((df["approved"] == 1) & (df["revisions"] == 0)).mean()), 3),
            "reviewer_approved_final": round(float(df["approved"].mean()), 3),
            "escalated": int(df["escalated"].sum()),
            "mean_revisions": round(float(df["revisions"].mean()), 3),
            "has_citation_rate": round(float(d["has_citation"].mean()), 3),
            "has_disclaimer_rate": round(float(d["has_disclaimer"].mean()), 3),
            "no_forbidden_phrases_rate": round(float(d["no_forbidden_phrases"].mean()), 3),
            "mean_citations": round(float(df["n_citations"].mean()), 2),
            "mean_grounding": round(float(df["grounding_score"].mean()), 3),
            "min_grounding": round(float(df["grounding_score"].min()), 3),
            "mean_length_chars": round(float(d["draft_length"].mean()), 1),
            "mean_sentences": round(float(d["draft_sentences"].mean()), 2),
            "mean_latency_s": round(float(df["latency_s"].mean()), 2),
        }

    summary = {"llm": stats(cur, llm_df), "template": stats(ref, template_df),
               "drift_p_values": _summarize(snap)["drift_p_values"], "html": "llm_vs_template_report.html"}
    (out_dir / "llm_eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
