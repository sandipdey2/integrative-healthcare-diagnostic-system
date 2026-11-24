import math

from ihds import model


def test_test_metrics_reasonable(pipe, split):
    m = model.evaluate(pipe, split[1])
    assert m["roc_auc"] > 0.78
    assert m["recall"] > 0.6
    cm = m["confusion_matrix"]
    assert sum(cm.values()) == 154


def test_probability_monotonic_in_glucose(pipe, good_record):
    lo = model.predict_proba(pipe, {**good_record, "Glucose": 90})[0]
    hi = model.predict_proba(pipe, {**good_record, "Glucose": 190})[0]
    assert 0 <= lo < hi <= 1


def test_contributions_sum_to_logit(pipe, good_record):
    contribs = model.contributions(pipe, good_record)
    p = model.predict_proba(pipe, good_record)[0]
    intercept = pipe.named_steps["clf"].intercept_[0]
    logit = intercept + sum(c.contribution for c in contribs)
    assert math.isclose(1 / (1 + math.exp(-logit)), p, rel_tol=1e-6)


def test_missing_optional_value_is_imputed(pipe, good_record):
    p = model.predict_proba(pipe, {**good_record, "Insulin": float("nan")})[0]
    assert 0 < p < 1
