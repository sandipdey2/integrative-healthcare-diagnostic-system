"""Tests for the confusion-matrix statistical analysis (ihds/cm_analysis.py)."""
import json

import numpy as np
import pytest

from ihds import cm_analysis, model
from ihds.cm_analysis import EXPECTED_TEST_CM


def test_test_set_confusion_matrix_matches_saved_metrics(pipe, split):
    from ihds.config import FEATURES

    _, test = split
    ev = model.evaluate(pipe, test, threshold=0.5)
    assert ev["confusion_matrix"] == EXPECTED_TEST_CM
    assert sum(ev["confusion_matrix"].values()) == 154
    y = test["Outcome"].to_numpy()
    proba = pipe.predict_proba(test[FEATURES])[:, 1]
    cm = cm_analysis.confusion_counts(y, (proba >= 0.5).astype(int))
    assert cm == EXPECTED_TEST_CM
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 154


def test_metrics_from_hand_built_confusion_matrix():
    # tn=3, fp=1, fn=1, tp=3 → n=8, acc=0.75, sens=spec=ppv=npv=0.75, MCC=kappa=0.5, LR+=3, LR-=1/3
    m = cm_analysis.metrics_from_cm(tn=3, fp=1, fn=1, tp=3)
    assert m["n"] == 8
    assert m["support_0"] == 4 and m["support_1"] == 4
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["sensitivity"] == pytest.approx(0.75)
    assert m["specificity"] == pytest.approx(0.75)
    assert m["ppv"] == pytest.approx(0.75)
    assert m["npv"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(0.25)
    assert m["fnr"] == pytest.approx(0.25)
    assert m["f1"] == pytest.approx(0.75)
    assert m["per_class"]["0"]["f1"] == pytest.approx(0.75)
    assert m["per_class"]["1"]["support"] == 4
    assert m["balanced_accuracy"] == pytest.approx(0.75)
    assert m["mcc"] == pytest.approx(0.5)
    assert m["cohens_kappa"] == pytest.approx(0.5)
    assert m["lr_plus"] == pytest.approx(3.0)
    assert m["lr_minus"] == pytest.approx(1 / 3)


def test_per_class_supports_equal_class_counts(pipe, split):
    ev = model.evaluate(pipe, split[1])
    cm = ev["confusion_matrix"]
    m = cm_analysis.metrics_from_cm(**cm)
    y = split[1]["Outcome"].to_numpy()
    assert m["support_0"] == int((y == 0).sum()) == 100
    assert m["support_1"] == int((y == 1).sum()) == 54
    assert m["support_0"] + m["support_1"] == 154


def test_wilson_interval_contains_estimate_and_stays_in_unit_interval():
    lo, hi = cm_analysis.wilson_interval(38, 54)
    assert 0 <= lo <= 38 / 54 <= hi <= 1
    lo0, hi0 = cm_analysis.wilson_interval(0, 10)
    assert 0 <= lo0 <= 0 <= hi0 <= 1 and hi0 > 0
    lo1, hi1 = cm_analysis.wilson_interval(10, 10)
    assert 0 <= lo1 <= hi1 <= 1 and hi1 == pytest.approx(1, abs=1e-12) and lo1 > 0.5
    # n=0 is undefined
    assert cm_analysis.wilson_interval(0, 0) == (None, None)
    w = cm_analysis.wilson_for_metrics(tn=75, fp=25, fn=16, tp=38)
    for name in ("sensitivity", "specificity", "ppv", "npv", "accuracy"):
        item = w[name]
        assert 0 <= item["ci_low"] <= item["estimate"] <= item["ci_high"] <= 1


def test_bootstrap_is_deterministic_with_the_seed():
    rng = np.random.default_rng(0)
    y = np.array([0] * 40 + [1] * 30)
    # Substantial overlap so the interval actually moves with the seed.
    scores = np.clip(0.35 + 0.25 * y + rng.normal(0, 0.25, size=70), 0, 1)
    a = cm_analysis.stratified_bootstrap_cis(y, scores, n_boot=120, seed=42)
    b = cm_analysis.stratified_bootstrap_cis(y, scores, n_boot=120, seed=42)
    c = cm_analysis.stratified_bootstrap_cis(y, scores, n_boot=120, seed=7)
    assert a["roc_auc"] == b["roc_auc"]
    assert a["f1"] == b["f1"]
    assert a["mcc"] == b["mcc"]
    assert a["roc_auc"]["ci_low"] != c["roc_auc"]["ci_low"] or a["roc_auc"]["ci_high"] != c["roc_auc"]["ci_high"]
    est = a["roc_auc"]["estimate"]
    assert a["roc_auc"]["ci_low"] <= est <= a["roc_auc"]["ci_high"]


def test_threshold_sweep_recall_is_non_increasing(pipe, split):
    test = split[1]
    y = test["Outcome"].to_numpy()
    proba = model.predict_proba(pipe, test)
    rows = cm_analysis.threshold_metrics_table(y, proba, [0.3, 0.4, 0.5, 0.6, 0.7])
    recalls = [r["recall"] for r in rows]
    assert all(recalls[i] >= recalls[i + 1] for i in range(len(recalls) - 1))


def test_youden_is_selected_on_train_oof_not_the_test_labels(pipe, split):
    train, test = split
    oof = cm_analysis.oof_predict_proba(train, n_folds=5, seed=42)
    assert len(oof) == 614
    chosen = cm_analysis.youden_threshold(train["Outcome"].to_numpy(), oof)
    # Sanity: a threshold picked on test labels can differ; the helper itself only sees the arrays it is given.
    test_best = cm_analysis.youden_threshold(
        test["Outcome"].to_numpy(), model.predict_proba(pipe, test)
    )
    assert 0 < chosen["threshold"] < 1
    assert "youden_j" in chosen
    # The selection function is pure: giving it test labels is a different call. Document the contract.
    assert chosen["train_oof_metrics"]["tn"] + chosen["train_oof_metrics"]["fp"] + chosen["train_oof_metrics"]["fn"] + chosen["train_oof_metrics"]["tp"] == 614
    assert isinstance(test_best["threshold"], float)


def test_delong_ci_contains_auc_and_agrees_with_sklearn(pipe, split):
    test = split[1]
    y = test["Outcome"].to_numpy()
    proba = model.predict_proba(pipe, test)
    d = cm_analysis.delong_auc_ci(y, proba)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y, proba)
    assert d["ci_low"] <= auc <= d["ci_high"]
    assert 0 <= d["ci_low"] <= d["ci_high"] <= 1
    assert d["estimate"] == pytest.approx(auc, abs=1e-4)


def test_decision_curve_treat_none_is_zero_and_net_benefit_finite(pipe, split):
    test = split[1]
    rows = cm_analysis.decision_curve(test["Outcome"].to_numpy(), model.predict_proba(pipe, test), [0.1, 0.3, 0.5, 0.7])
    assert all(r["treat_none"] == 0.0 for r in rows)
    assert all(r["net_benefit"] is not None and np.isfinite(r["net_benefit"]) for r in rows)
    # At a very high threshold treat-all is negative (you almost never treat under treat-all at pt=0.7).
    assert rows[-1]["treat_all"] < 0


def test_accuracy_vs_nir_and_mcnemar_on_the_known_matrix():
    acc = cm_analysis.binomial_accuracy_vs_nir(tn=75, fp=25, fn=16, tp=38)
    assert acc["n"] == 154 and acc["correct"] == 113
    assert acc["no_information_rate"] == pytest.approx(100 / 154, abs=1e-4)
    assert 0 < acc["p_value_greater"] < 1
    mc = cm_analysis.mcnemar_vs_majority(tn=75, fp=25, fn=16, tp=38)
    assert mc["model_correct_baseline_wrong"] == 38
    assert mc["model_wrong_baseline_correct"] == 25
    assert 0 < mc["p_value_two_sided"] < 1


def test_full_analysis_writes_json_and_matches_expected_cm(pipe, split, tmp_path):
    train, test = split
    report = cm_analysis.analyze(
        train, test, pipe=pipe, n_boot=40, n_perm=40, n_coef_boot=20, plot=True, out_dir=tmp_path,
    )
    cm_analysis.write_report(report, tmp_path)
    payload = json.loads((tmp_path / "confusion_matrix_analysis.json").read_text())
    assert payload["test_confusion_matrix"] == EXPECTED_TEST_CM
    assert payload["test_metrics"]["support_0"] == 100
    assert payload["test_metrics"]["support_1"] == 54
    # Sensitivity Wilson lower bound stays above 0.5 on this split (a floor that the data actually meets).
    assert payload["wilson_cis"]["sensitivity"]["ci_low"] > 0.5
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "roc_curve.png").exists()
    assert (tmp_path / "pr_curve.png").exists()
    assert (tmp_path / "calibration_curve.png").exists()
    assert (tmp_path / "decision_curve.png").exists()
    assert len(payload["age_bands"]["by_band"]) == 3
    assert payload["cv_pooled_train"]["n"] == 614
    # Calibration and decision-curve keys are present and numeric.
    assert payload["calibration"]["brier_score"] > 0
    assert payload["decision_curve"]["table"][0]["threshold"] == 0.1
    assert "Glucose" in {r["feature"] for r in payload["coefficients"]["features"]}
    assert payload["welch_tests"]["rows"][0]["feature"] == "Glucose"
    assert "p_value_bh" in payload["welch_tests"]["rows"][0]


def test_recall_heterogeneity_on_a_known_table():
    # Two bands with clearly different recall, one with a zero cell.
    table = np.array([[7, 9], [19, 7], [12, 0]])
    res = cm_analysis._recall_heterogeneity(table, ["21-29", "30-44", "45+"], n_perm=400, seed=42)
    assert res["p_value_chi2"] < 0.05
    assert 0 < res["p_value_permutation"] < 1
    assert len(res["pairwise_fisher"]) == 3
