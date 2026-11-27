"""Tests for the explicit KS / PSI / chi-square drift module (ihds/drift_stats.py)."""
import numpy as np
import pytest

from ihds import data, drift_stats, monitoring, stats_analysis
from ihds.config import FEATURES, TARGET


def _by_feature(scenario: dict, name: str) -> dict:
    return next(r for r in scenario["features"] if r["feature"] == name)


def test_psi_zero_on_identical_samples_and_larger_when_shifted():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    same = drift_stats.population_stability_index(x, x)
    assert same["psi"] is not None
    assert same["psi"] == pytest.approx(0.0, abs=1e-12)
    assert same["psi"] >= 0
    shifted = drift_stats.population_stability_index(x, x + 1.5)
    assert shifted["psi"] > same["psi"]
    assert shifted["psi"] > 0.25


def test_psi_is_non_negative_and_handles_nan_and_constant_columns():
    constant = np.array([2.0, 2.0, 2.0, np.nan, 2.0, 2.0])
    same = drift_stats.population_stability_index(constant, np.array([2.0, np.nan, 2.0, 2.0]))
    assert same["psi"] is not None and same["psi"] >= 0
    assert same["psi"] == pytest.approx(0.0, abs=1e-9)

    moved = drift_stats.population_stability_index(constant, np.array([2.0, 5.0, 8.0, np.nan]))
    assert moved["psi"] is not None and moved["psi"] >= 0
    assert moved["psi"] > 0

    mixed = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan])
    other = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    psi = drift_stats.population_stability_index(mixed, other)
    assert psi["psi"] is not None and psi["psi"] >= 0
    assert psi["psi"] == pytest.approx(0.0, abs=1e-9)


def test_ks_and_chi_square_pvalues_high_when_identical_and_tiny_when_shifted():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 400)
    ks_same = drift_stats.ks_two_sample(x, x)
    assert ks_same["statistic"] == pytest.approx(0.0, abs=1e-12)
    assert ks_same["pvalue"] == pytest.approx(1.0)

    edges = drift_stats.reference_bin_edges(x)
    chi_same = drift_stats.chi_square_binned(x, x, edges)
    assert chi_same["pvalue"] == pytest.approx(1.0)

    y = x + 3.0
    ks_shift = drift_stats.ks_two_sample(x, y)
    chi_shift = drift_stats.chi_square_binned(x, y, edges)
    assert ks_shift["pvalue"] < 1e-10
    assert chi_shift["pvalue"] < 1e-10
    assert ks_shift["statistic"] > 0.5


def test_bh_adjustment_is_monotone():
    raw = np.array([0.001, 0.04, 0.20, 0.80])
    bh = stats_analysis.adjust_pvalues(raw, "bh")
    assert (bh >= raw - 1e-15).all()
    order = np.argsort(raw)
    assert all(bh[order[i]] <= bh[order[i + 1]] + 1e-15 for i in range(len(raw) - 1))

    # On the real holdout, adjusted p-values stay above the raw ones and above 0.1.
    train, test = data.split(data.load_clean())
    scenario = drift_stats.analyze_scenario("holdout", train, test)
    for row in scenario["features"]:
        assert row["ks_pvalue_bh"] >= row["ks_pvalue"] - 1e-15
        assert row["chi2_pvalue_bh"] >= row["chi2_pvalue"] - 1e-15
        assert row["ks_pvalue"] > 0.1
    ks_bh = np.array([r["ks_pvalue_bh"] for r in scenario["features"]])
    ks_raw = np.array([r["ks_pvalue"] for r in scenario["features"]])
    order = np.argsort(ks_raw)
    assert all(ks_bh[order[i]] <= ks_bh[order[i + 1]] + 1e-15 for i in range(len(order) - 1))


def test_holdout_has_no_alert_and_insulin_is_watch(split):
    train, test = split
    scenario = drift_stats.analyze_scenario("holdout", train, test)
    assert scenario["summary"]["overall_alert"] is False
    assert scenario["summary"]["alert"] == []
    assert scenario["summary"]["watch"] == ["Insulin"]
    insulin = _by_feature(scenario, "Insulin")
    assert insulin["psi_band"] == "moderate"
    assert insulin["verdict"] == "watch"
    assert insulin["psi"] == pytest.approx(0.241, abs=0.005)
    assert insulin["ks_pvalue"] > 0.1
    assert insulin["chi2_pvalue"] > 0.1
    assert insulin["ks_pvalue_bh"] >= insulin["ks_pvalue"]
    for row in scenario["features"]:
        if row["feature"] != "Insulin":
            assert row["verdict"] == "stable"
            assert row["psi"] < 0.10
        assert row["ks_pvalue"] > 0.1
    assert scenario["label"]["chi2_pvalue"] == pytest.approx(1.0)
    assert scenario["label"]["ref_counts"] == {"0": 400, "1": 214}
    assert scenario["label"]["cur_counts"] == {"0": 100, "1": 54}


def test_shifted_scenario_alerts_on_glucose_only(split):
    train, test = split
    shifted = monitoring.simulate_glucose_calibration_shift(test)
    scenario = drift_stats.analyze_scenario("shifted", train, shifted)
    assert scenario["summary"]["overall_alert"] is True
    assert scenario["summary"]["alert"] == ["Glucose"]
    assert scenario["summary"]["ks_drifted"] == ["Glucose"]
    assert scenario["summary"]["chi2_drifted"] == ["Glucose"]
    assert scenario["summary"]["psi_major"] == ["Glucose"]
    glucose = _by_feature(scenario, "Glucose")
    assert glucose["verdict"] == "alert"
    assert glucose["psi_band"] == "major"
    assert glucose["ks_statistic"] == pytest.approx(0.366, abs=0.005)
    assert glucose["psi"] == pytest.approx(0.784, abs=0.005)
    assert glucose["ks_pvalue"] < 1e-10
    assert glucose["chi2_pvalue"] < 1e-10
    assert glucose["ks_pvalue_bh"] < 1e-10
    assert glucose["chi2_pvalue_bh"] < 1e-10
    # Insulin is unchanged by the glucose shift, so it stays a watch, not an alert.
    assert _by_feature(scenario, "Insulin")["verdict"] == "watch"
    for row in scenario["features"]:
        if row["feature"] != "Glucose":
            assert row["verdict"] != "alert"
    assert scenario["label"]["chi2_pvalue"] == pytest.approx(1.0)


def test_params_bands_are_respected(split):
    train, test = split
    insulin_psi = 0.2405
    assert drift_stats.psi_band(insulin_psi, 0.10, 0.25) == "moderate"
    assert drift_stats.psi_band(insulin_psi, 0.10, 0.20) == "major"
    assert drift_stats.psi_band(insulin_psi, 0.30, 0.50) == "stable"
    assert drift_stats.feature_verdict("moderate", ks_p_bh=0.20, chi2_p_bh=0.15, alpha=0.05) == "watch"
    assert drift_stats.feature_verdict("moderate", ks_p_bh=0.01, chi2_p_bh=0.20, alpha=0.05) == "alert"
    assert drift_stats.feature_verdict("major", ks_p_bh=0.9, chi2_p_bh=0.9, alpha=0.05) == "alert"
    assert drift_stats.feature_verdict("stable", ks_p_bh=0.9, chi2_p_bh=0.9, alpha=0.05) == "stable"

    default = drift_stats.analyze_scenario("holdout", train, test, psi_stable_max=0.10, psi_major_min=0.25)
    assert _by_feature(default, "Insulin")["verdict"] == "watch"
    assert default["summary"]["overall_alert"] is False

    tighter = drift_stats.analyze_scenario("holdout", train, test, psi_stable_max=0.10, psi_major_min=0.20)
    assert _by_feature(tighter, "Insulin")["psi_band"] == "major"
    assert _by_feature(tighter, "Insulin")["verdict"] == "alert"
    assert tighter["summary"]["overall_alert"] is True

    looser = drift_stats.analyze_scenario("holdout", train, test, psi_stable_max=0.30, psi_major_min=0.50)
    assert _by_feature(looser, "Insulin")["psi_band"] == "stable"
    assert _by_feature(looser, "Insulin")["verdict"] == "stable"
    assert looser["summary"]["overall_alert"] is False

    cfg = drift_stats._load_settings()
    assert cfg["psi_stable_max"] == 0.10
    assert cfg["psi_major_min"] == 0.25
    assert cfg["n_bins"] == 10
    assert cfg["epsilon"] == 1e-4
    assert cfg["alpha"] == 0.05


def test_label_chi_square_p_is_one_on_stratified_split(split):
    train, test = split
    result = drift_stats.label_chi_square(train[TARGET], test[TARGET])
    assert result["chi2_pvalue"] == pytest.approx(1.0)
    assert result["ref_counts"] == {"0": 400, "1": 214}
    assert result["cur_counts"] == {"0": 100, "1": 54}
    # Stratified split keeps the rates as close as the integer counts allow
    # (214/614 vs 54/154); the chi-square test of those counts has p = 1.0.


def test_report_and_plot_are_written(split, pipe, tmp_path):
    train, test = split
    report = drift_stats.analyze(reference=train, current=test, pipe=pipe, out_dir=tmp_path)
    path = drift_stats.write_report(report, tmp_path)
    assert path.exists()
    payload = path.read_text()
    assert "holdout" in payload and "shifted" in payload
    assert report["scenarios"]["holdout"]["summary"]["overall_alert"] is False
    assert report["scenarios"]["shifted"]["summary"]["alert"] == ["Glucose"]
    assert report["plots"].get("drift_stats_psi.png")
    assert (tmp_path / "drift_stats_psi.png").exists()
    assert set(r["feature"] for r in report["scenarios"]["holdout"]["features"]) == set(FEATURES)
    assert "predicted_class" in report["scenarios"]["holdout"]
    assert "predicted_class" in report["scenarios"]["shifted"]
