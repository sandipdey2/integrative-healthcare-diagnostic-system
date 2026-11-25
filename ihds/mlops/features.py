"""Feast feature store: apply definitions, materialize to the SQLite online store, fetch online features.

    python -m ihds.mlops.features                     # apply + materialize + fetch patient 0
    python -m ihds.mlops.features --patient-id 5 --patient-id 42 --no-apply

Equivalent CLI (from feature_repo/):  feast apply && feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
Local only: registry and online store are SQLite files in feature_repo/data/ (gitignored). The risk model is
scored on the fetched features to show the online path end to end; the Streamlit / API app does not read Feast.
"""
from __future__ import annotations

import argparse
import importlib.util
from datetime import datetime, timezone

import joblib
import pandas as pd

from ..config import FEATURES
from .common import FEAST_PARQUET, FEATURE_REPO, MLOPS_REPORTS, MODEL_PATH, write_json

FEATURE_REFS = [f"patient_clinical_features:{f}" for f in FEATURES]


def _definitions():
    spec = importlib.util.spec_from_file_location("ihds_feast_definitions", FEATURE_REPO / "features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def store():
    from feast import FeatureStore

    return FeatureStore(repo_path=str(FEATURE_REPO))


def apply_and_materialize(fs=None) -> dict:
    if not FEAST_PARQUET.exists():
        from .prepare import run as prepare

        prepare()
    fs = fs or store()
    d = _definitions()
    fs.apply([d.patient, d.clinical_source, d.clinical_features, d.risk_model_features])
    fs.materialize_incremental(end_date=datetime.now(timezone.utc))
    return {"feature_views": [fv.name for fv in fs.list_feature_views()],
            "feature_services": [s.name for s in fs.list_feature_services()]}


def online_features(patient_ids: list[int], fs=None) -> pd.DataFrame:
    fs = fs or store()
    resp = fs.get_online_features(features=FEATURE_REFS, entity_rows=[{"patient_id": int(i)} for i in patient_ids])
    return resp.to_df()


def run(patient_ids: list[int] | None = None, apply: bool = True) -> dict:
    patient_ids = patient_ids or [0]
    fs = store()
    info = apply_and_materialize(fs) if apply else {}
    df = online_features(patient_ids, fs)
    source = pd.read_parquet(FEAST_PARQUET).set_index("patient_id")
    matches = all(
        (pd.isna(source.loc[pid, f]) and pd.isna(row[f])) or abs(float(source.loc[pid, f]) - float(row[f])) < 1e-9
        for pid, (_, row) in zip(patient_ids, df.iterrows()) for f in FEATURES)
    out = {**info, "patient_ids": patient_ids,
           "features": [{k: (None if pd.isna(v) else (int(v) if k == "patient_id" else float(v)))
                         for k, v in r.items()} for r in df.to_dict(orient="records")],
           "matches_source_parquet": bool(matches)}
    if MODEL_PATH.exists():
        pipe = joblib.load(MODEL_PATH)
        out["risk_probability"] = [round(float(p), 4) for p in pipe.predict_proba(df[FEATURES])[:, 1]]
    write_json(MLOPS_REPORTS / "feast_online_sample.json", out)
    print(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--patient-id", type=int, action="append", help="row number in data/pima_diabetes.csv")
    ap.add_argument("--no-apply", action="store_true", help="skip apply/materialize (store already built)")
    a = ap.parse_args()
    run(a.patient_id, apply=not a.no_apply)


if __name__ == "__main__":
    main()
