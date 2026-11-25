"""Evidently drift step for the DVC pipeline / Prefect flow.

    python -m ihds.mlops.drift

Reuses ihds.monitoring.run_all (the same Evidently 0.7 reports the app ships) on the processed split and the
DVC-trained model: holdout (train vs test) and shifted (test with the simulated +25 mg/dL glucose meter error, monitoring.GLUCOSE_SHIFT).
Writes to reports/drift/ so the committed monitoring/reports/ used by the Streamlit Monitoring tab is not touched.
The GAN synthetic report stays in `python -m ihds.pipeline` (it needs the GAN).
"""
from __future__ import annotations

import joblib

from .. import monitoring
from .common import MLOPS_REPORTS, MODEL_PATH, load_splits, write_json

DRIFT_DIR = MLOPS_REPORTS / "drift"


def run(train=None, test=None, pipe=None) -> dict:
    if train is None or test is None:
        train, test = load_splits()
    pipe = pipe if pipe is not None else joblib.load(MODEL_PATH)
    res = monitoring.run_all(pipe, train, test, synthetic=None, out_dir=DRIFT_DIR)
    out = {name: {k: res[name].get(k) for k in ("drifted_feature_names", "drifted_columns_share", "accuracy",
                                                 "precision", "recall")} for name in ("holdout", "shifted")}
    out["glucose_shift_mg_dl"] = monitoring.GLUCOSE_SHIFT
    write_json(MLOPS_REPORTS / "drift_summary.json", out)
    print(out)
    return out


if __name__ == "__main__":
    run()
