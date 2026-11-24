import numpy as np

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
