"""Great Expectations data validation for the clinical input data (GX 1.x API, ephemeral context).

    python -m ihds.mlops.validate            # exits 1 if any expectation fails

Validates data/pima_diabetes.csv after the same zero-to-missing cleaning the model uses: column schema and
order, row count, numeric dtypes, non-null shares, plausible ranges (config.VALID_RANGES) and a {0,1} label.
Non-null thresholds sit just below the observed shares (e.g. Insulin is ~51% present), so they catch a
pipeline that drops more values than the source data already lacks; they do not claim the data is complete.
Writes reports/validation.json and the suite definition to reports/gx_suite.json.
"""
from __future__ import annotations

import sys

import pandas as pd

from .. import data
from ..config import DATA_PATH, FEATURES, TARGET, VALID_RANGES
from .common import MLOPS_REPORTS, load_params, write_json

SUITE_NAME = "pima_clinical_input"
# Minimum share of non-null values per column (after zeros -> NaN). Observed shares in the shipped CSV:
# Glucose 0.993, BloodPressure 0.954, SkinThickness 0.704, Insulin 0.513, BMI 0.986, others 1.0.
NOT_NULL_MOSTLY = {"Glucose": 0.97, "BloodPressure": 0.93, "SkinThickness": 0.65, "Insulin": 0.45, "BMI": 0.97}


def build_suite(params: dict | None = None):
    import great_expectations as gx
    import great_expectations.expectations as gxe

    p = params or load_params("validate")
    suite = gx.ExpectationSuite(name=SUITE_NAME)
    suite.add_expectation(gxe.ExpectTableColumnsToMatchOrderedList(column_list=FEATURES + [TARGET]))
    suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=p["min_rows"], max_value=p["max_rows"]))
    for col in FEATURES + [TARGET]:
        suite.add_expectation(gxe.ExpectColumnValuesToBeInTypeList(column=col, type_list=["int64", "float64"]))
        suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col, mostly=NOT_NULL_MOSTLY.get(col, 1.0)))
    for col, (lo, hi) in VALID_RANGES.items():
        suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column=col, min_value=lo, max_value=hi,
                                                                mostly=p["range_mostly"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1]))
    suite.add_expectation(gxe.ExpectColumnMeanToBeBetween(column=TARGET, min_value=0.2, max_value=0.5))
    return suite


def validate_frame(df: pd.DataFrame, params: dict | None = None) -> dict:
    import great_expectations as gx

    context = gx.get_context(mode="ephemeral")
    suite = context.suites.add(build_suite(params))
    batch_def = (context.data_sources.add_pandas("pandas")
                 .add_dataframe_asset("clinical_input")
                 .add_batch_definition_whole_dataframe("whole"))
    vd = context.validation_definitions.add(gx.ValidationDefinition(name="clinical_input_check", data=batch_def,
                                                                     suite=suite))
    res = vd.run(batch_parameters={"dataframe": df})
    failed = []
    for r in res.results:
        if not r.success:
            cfg = r.expectation_config
            failed.append({"expectation": cfg.type, "kwargs": {k: v for k, v in cfg.kwargs.items() if k != "batch_id"},
                           "unexpected_percent": (r.result or {}).get("unexpected_percent")})
    stats = res.statistics
    return {"suite": SUITE_NAME, "success": bool(res.success),
            "evaluated": int(stats["evaluated_expectations"]), "passed": int(stats["successful_expectations"]),
            "failed": int(stats["unsuccessful_expectations"]), "failures": failed,
            "suite_json": suite.to_json_dict()}


def _drop_ids(obj):
    """GX gives every expectation a random UUID; drop them so the committed suite file is deterministic."""
    if isinstance(obj, dict):
        return {k: _drop_ids(v) for k, v in obj.items() if k != "id"}
    if isinstance(obj, list):
        return [_drop_ids(v) for v in obj]
    return obj


def run(path=DATA_PATH) -> dict:
    df = data.clean(pd.read_csv(path))
    out = validate_frame(df)
    write_json(MLOPS_REPORTS / "gx_suite.json", _drop_ids(out.pop("suite_json")))
    out["source"] = str(path.relative_to(path.parents[1])) if hasattr(path, "parents") else str(path)
    write_json(MLOPS_REPORTS / "validation.json", out)
    print({k: out[k] for k in ("success", "evaluated", "passed", "failed")})
    return out


if __name__ == "__main__":
    sys.exit(0 if run()["success"] else 1)
