"""Train the risk model with MLflow experiment tracking (local store under ./mlruns, no tracking server).

    python -m ihds.mlops.train_tracked
    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db     # optional: browse runs at http://127.0.0.1:5000

MLflow 3.x refuses the plain ./mlruns file backend unless MLFLOW_ALLOW_FILE_STORE=true, so run metadata goes
to a SQLite file (mlruns/mlflow.db) and artifacts to mlruns/artifacts/. Both are local and gitignored.

Same pipeline as ihds.model (median impute -> StandardScaler -> LogisticRegression), with C / max_iter /
class_weight / threshold read from params.yaml. Logs params, test-split metrics, 5-fold CV ROC-AUC on the
training split, the coefficient table, and the fitted sklearn model (with signature) to ./mlruns (gitignored).
Also writes models/risk_model.joblib and reports/train_metrics.json for the DVC pipeline.
"""
from __future__ import annotations

import os

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .. import model
from ..config import FEATURES, SEED, TARGET
from .common import MLOPS_REPORTS, MLRUNS_DIR, MODEL_PATH, load_params, load_splits, write_json


def tracking_uri() -> str:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(MLRUNS_DIR / 'mlflow.db').resolve()}"


def ensure_experiment(mlflow, name: str) -> None:
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(MLRUNS_DIR / "artifacts").resolve().as_uri())
    mlflow.set_experiment(name)


def build(p: dict):
    pipe = model.build_pipeline()
    pipe.set_params(clf__C=p["C"], clf__max_iter=p["max_iter"], clf__class_weight=p["class_weight"])
    return pipe


def run(train: pd.DataFrame | None = None, test: pd.DataFrame | None = None) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    p = load_params("train")
    if train is None or test is None:
        train, test = load_splits()
    pipe = build(p).fit(train[FEATURES], train[TARGET])
    metrics = model.evaluate(pipe, test, threshold=p["threshold"])
    cv = cross_val_score(build(p), train[FEATURES], train[TARGET], scoring="roc_auc",
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    mlflow.set_tracking_uri(tracking_uri())
    ensure_experiment(mlflow, p["experiment"])
    with mlflow.start_run(run_name="logreg-balanced") as active:
        mlflow.log_params({"model": "LogisticRegression", "imputer": "median", "scaler": "standard",
                           "C": p["C"], "max_iter": p["max_iter"], "class_weight": p["class_weight"],
                           "threshold": p["threshold"], "seed": SEED, "n_train": len(train), "n_test": len(test),
                           "features": ",".join(FEATURES)})
        flat = {f"test_{k}": v for k, v in metrics.items() if isinstance(v, float) and k != "threshold"}
        flat.update({f"test_{k}": v for k, v in metrics["confusion_matrix"].items()})
        flat.update({"cv_roc_auc_mean": round(float(cv.mean()), 4), "cv_roc_auc_std": round(float(cv.std()), 4)})
        mlflow.log_metrics(flat)
        mlflow.log_dict(model.coefficients(pipe).round(4).to_dict(orient="records"), "coefficients.json")
        mlflow.set_tags({"project": "ihds", "data": "pima_diabetes.csv", "stage": "dvc:train"})
        x_test = test[FEATURES].astype("float64")  # float schema: any feature may be missing at inference time
        sig = infer_signature(x_test, pipe.predict_proba(x_test)[:, 1])
        info = mlflow.sklearn.log_model(pipe, name="risk_model", signature=sig,
                                        input_example=x_test.head(3), skops_trusted_types=["numpy.dtype"])
        run_id = active.info.run_id

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    out = {"mlflow_run_id": run_id, "experiment": p["experiment"], "model_uri": info.model_uri, **flat}
    write_json(MLOPS_REPORTS / "train_metrics.json", out)
    print(out)
    return {**out, "pipe": pipe}


if __name__ == "__main__":
    run()
