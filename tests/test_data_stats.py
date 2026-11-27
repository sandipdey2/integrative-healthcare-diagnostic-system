import numpy as np
import pytest

from ihds import data, stats_analysis
from ihds.config import FEATURES, TARGET


def test_raw_shape_and_target():
    df = data.load_raw()
    assert df.shape == (768, 9)
    assert set(df[TARGET].unique()) == {0, 1}
    assert df[TARGET].sum() == 268


def test_clean_replaces_impossible_zeros():
    raw = data.load_raw()
    clean = data.clean(raw)
    assert (clean["Glucose"] == 0).sum() == 0
    assert clean["Insulin"].isna().sum() == (raw["Insulin"] == 0).sum() == 374
    assert (clean["Pregnancies"] == 0).sum() == (raw["Pregnancies"] == 0).sum()  # zero is valid here


def test_split_is_stratified(split):
    train, test = split
    assert len(train) == 614 and len(test) == 154
    assert abs(train[TARGET].mean() - test[TARGET].mean()) < 0.01


def test_welch_glucose_most_significant():
    res = stats_analysis.welch_tests(data.load_clean())
    assert res.iloc[0]["feature"] == "Glucose"
    assert res.iloc[0]["p_value"] < 1e-10
    assert set(res["feature"]) == set(FEATURES)
    assert np.isfinite(res["t_stat"]).all()


def test_welch_pvalues_are_multiplicity_adjusted():
    from ihds import data, stats_analysis

    # Bonferroni and BH never shrink a p-value; Glucose remains significant after either correction.
    raw = [0.01, 0.04, 0.03, 0.20]
    bh = stats_analysis.adjust_pvalues(raw, "bh")
    bon = stats_analysis.adjust_pvalues(raw, "bonferroni")
    raw = np.asarray(raw, dtype=float)
    assert (bon >= raw - 1e-15).all() and (bh >= raw - 1e-15).all()
    assert bon[-1] == pytest.approx(min(1.0, 0.20 * 4))
    # BH is monotone in the p-value order after adjustment.
    order = sorted(range(4), key=lambda i: raw[i])
    assert all(bh[order[i]] <= bh[order[i + 1]] + 1e-15 for i in range(3))

    res = stats_analysis.welch_tests(data.load_clean())
    assert res.iloc[0]["feature"] == "Glucose"
    assert res.iloc[0]["p_value_bh"] <= res.iloc[0]["p_value_bonferroni"]
    assert res.iloc[0]["p_value_bh"] < 0.05
    assert res.iloc[0]["significant_bh"]
    assert {"p_value_bh", "p_value_bonferroni", "significant_bh", "significant_bonferroni"} <= set(res.columns)
