"""Tabular GAN for synthetic patient-feature generation (NumPy only).

Adapted from generative-mnist, which trained a PyTorch GAN on MNIST with an
MLP Generator/Discriminator, LeakyReLU(0.2), binary cross-entropy and Adam
(lr=2e-4, betas=(0.5, 0.999)). The same recipe is re-implemented here with
NumPy and hand-written backpropagation so the deployed container does not
need PyTorch. Differences from the MNIST GAN: the data are 8 standardized
tabular features (not 784 pixels), the generator output is linear (not
Tanh), and networks are smaller (2 hidden layers of 64 units).

Synthetic rows are used only for stress-testing (borderline cases, fidelity
monitoring). They are never used to train the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FEATURES, INTEGER_FEATURES, SEED, VALID_RANGES


def _leaky(x, a=0.2):
    return np.where(x > 0, x, a * x)


def _leaky_grad(x, a=0.2):
    return np.where(x > 0, 1.0, a)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softplus(x):
    return np.logaddexp(0.0, x)


class MLP:
    """Fully connected net: LeakyReLU hidden layers, linear output."""

    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        self.cache = []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            pre = h @ W + b
            self.cache.append((h, pre))
            h = pre if i == len(self.W) - 1 else _leaky(pre)
        return h

    def backward(self, grad_out):
        """Returns (param grads, grad wrt input)."""
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = grad_out
        for i in reversed(range(len(self.W))):
            h, pre = self.cache[i]
            if i != len(self.W) - 1:
                g = g * _leaky_grad(pre)
            gW[i] = h.T @ g
            gb[i] = g.sum(axis=0)
            g = g @ self.W[i].T
        return gW + gb, g

    def adam_step(self, grads, lr=2e-4, b1=0.5, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        for k, (p, gr) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr**2
            mh = self.m[k] / (1 - b1**self.t)
            vh = self.v[k] / (1 - b2**self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)


@dataclass
class TabularGAN:
    latent_dim: int = 16
    hidden: int = 64
    steps: int = 3000
    batch_size: int = 64
    lr: float = 2e-4
    seed: int = SEED
    history: dict = field(default_factory=lambda: {"d_loss": [], "g_loss": []})

    def fit(self, X: pd.DataFrame) -> "TabularGAN":
        """X: real feature rows with no missing values (impute before calling)."""
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        data = X[FEATURES].to_numpy(dtype=float)
        self.mu, self.sd = data.mean(0), data.std(0) + 1e-8
        Z = (data - self.mu) / self.sd
        d = Z.shape[1]
        self.G = MLP([self.latent_dim, self.hidden, self.hidden, d], rng)
        self.D = MLP([d, self.hidden, self.hidden, 1], rng)
        n = len(Z)
        for step in range(self.steps):
            real = Z[rng.integers(0, n, self.batch_size)]
            z = rng.normal(size=(self.batch_size, self.latent_dim))
            fake = self.G.forward(z)
            # --- Discriminator step: BCE(real=1) + BCE(fake=0) on logits
            s_real = self.D.forward(real)
            g_real, _ = self.D.backward((_sigmoid(s_real) - 1.0) / self.batch_size)
            s_fake = self.D.forward(fake)
            g_fake, _ = self.D.backward(_sigmoid(s_fake) / self.batch_size)
            self.D.adam_step([a + b for a, b in zip(g_real, g_fake)], lr=self.lr)
            d_loss = float(_softplus(-s_real).mean() + _softplus(s_fake).mean())
            # --- Generator step: non-saturating loss -log D(G(z))
            z = rng.normal(size=(self.batch_size, self.latent_dim))
            fake = self.G.forward(z)
            s = self.D.forward(fake)
            _, g_x = self.D.backward((_sigmoid(s) - 1.0) / self.batch_size)
            g_G, _ = self.G.backward(g_x)
            self.G.adam_step(g_G, lr=self.lr)
            g_loss = float(_softplus(-s).mean())
            if step % 50 == 0:
                self.history["d_loss"].append(d_loss)
                self.history["g_loss"].append(g_loss)
        return self

    def sample(self, n: int, seed: int | None = None, postprocess: bool = True) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + 1 if seed is None else seed)
        z = rng.normal(size=(n, self.latent_dim))
        out = self.G.forward(z) * self.sd + self.mu
        df = pd.DataFrame(out, columns=FEATURES)
        return postprocess_synthetic(df) if postprocess else df


def postprocess_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, (lo, hi) in VALID_RANGES.items():
        out[col] = out[col].clip(lo, hi)
    for col in INTEGER_FEATURES:
        out[col] = out[col].round().astype(int)
    out["DiabetesPedigreeFunction"] = out["DiabetesPedigreeFunction"].round(3)
    out["BMI"] = out["BMI"].round(1)
    return out
