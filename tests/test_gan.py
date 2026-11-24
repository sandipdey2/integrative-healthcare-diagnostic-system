import numpy as np

from ihds import gan
from ihds.config import FEATURES, VALID_RANGES


def test_backprop_matches_finite_difference():
    rng = np.random.default_rng(0)
    net = gan.MLP([3, 5, 2], rng)
    x = rng.normal(size=(4, 3))
    w = rng.normal(size=(4, 2))  # loss = sum(out * w)
    net.forward(x)
    grads, _ = net.backward(w)
    eps = 1e-6
    W = net.W[0]
    i, j = 1, 2
    old = W[i, j]
    W[i, j] = old + eps
    lp = (net.forward(x) * w).sum()
    W[i, j] = old - eps
    lm = (net.forward(x) * w).sum()
    W[i, j] = old
    assert abs((lp - lm) / (2 * eps) - grads[0][i, j]) < 1e-5


def test_gan_trains_and_samples_valid_rows(imputed_train):
    g = gan.TabularGAN(steps=400).fit(imputed_train)
    s = g.sample(200)
    assert list(s.columns) == FEATURES and len(s) == 200
    assert np.isfinite(s.to_numpy(dtype=float)).all()
    for col, (lo, hi) in VALID_RANGES.items():
        assert s[col].between(lo, hi).all()
    assert len(g.history["d_loss"]) == 8
    # not collapsed to a single point
    assert (s.std() > 0).all()


def test_gan_glucose_mean_close_after_full_training(imputed_train):
    s = gan.TabularGAN(steps=3000).fit(imputed_train).sample(500)
    assert abs(s["Glucose"].mean() - imputed_train["Glucose"].mean()) < 10
