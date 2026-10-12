"""Paired statistics shared by the analyses.

All functions are pure (no I/O), so they can be tested on synthetic data with a known
answer (tests/test_stats.py). The bootstrap seed defaults to 42 everywhere and is always
an explicit argument.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, wilcoxon


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, in the order of the input.

    NaN p-values are treated as p = 1 (an inconclusive test must never become
    "significant" through sorting). The threshold q is left to the caller.
    """
    p = np.asarray(pvals, dtype=float)
    p_filled = np.where(np.isfinite(p), p, 1.0)
    n = len(p_filled)
    if n == 0:
        return p_filled
    order = np.argsort(p_filled)
    ranked = p_filled[order]
    q = ranked * n / (np.arange(n) + 1)
    for k in range(n - 2, -1, -1):
        q[k] = min(q[k], q[k + 1])
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def paired_cohens_d(diff: np.ndarray) -> float:
    """Paired Cohen's d (d_z) = mean(diff) / std(diff, ddof=1); NaN if the variance is zero."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return float("nan")
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(diff) / sd)


def wilcoxon_p_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Paired Wilcoxon signed-rank p-value; NaN if undefined (all differences zero, n < 1)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 1 or np.all(a == b):
        return float("nan")
    try:
        _, p = wilcoxon(a, b)
        return float(p)
    except ValueError:
        return float("nan")


def bca_bounds(orig: float, boot_stats: np.ndarray, jack_stats: np.ndarray,
               alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) interval from precomputed bootstrap and
    jackknife statistics."""
    boot_stats = np.asarray(boot_stats, dtype=float)
    jack_stats = np.asarray(jack_stats, dtype=float)
    n_boot = len(boot_stats)
    if n_boot == 0 or len(jack_stats) < 2:
        return float("nan"), float("nan")

    prop_less = np.clip(np.mean(boot_stats < orig), 1.0 / (n_boot + 1), 1 - 1.0 / (n_boot + 1))
    z0 = norm.ppf(prop_less)

    jack_mean = jack_stats.mean()
    num = np.sum((jack_mean - jack_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a_hat = num / den if den > 0 else 0.0

    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)

    def _adj(z):
        denom = 1 - a_hat * (z0 + z)
        if denom == 0:
            return norm.cdf(z0 + (z0 + z))
        return norm.cdf(z0 + (z0 + z) / denom)

    lo_pct = np.clip(_adj(z_lo) * 100, 100.0 / (n_boot + 1), 100 - 100.0 / (n_boot + 1))
    hi_pct = np.clip(_adj(z_hi) * 100, 100.0 / (n_boot + 1), 100 - 100.0 / (n_boot + 1))
    if lo_pct > hi_pct:
        lo_pct, hi_pct = hi_pct, lo_pct
    ci_lo, ci_hi = np.percentile(boot_stats, [lo_pct, hi_pct])
    return float(ci_lo), float(ci_hi)


def bca_median_diff(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                     seed: int = 42) -> tuple[float, float, float]:
    """Median of paired differences with a BCa interval (cases resampled).

    The bootstrap is vectorised; the jackknife of the median has no closed form and
    loops over the n cases (negligible for a few hundred cases).
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    orig = float(np.median(diff))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = np.median(diff[idx], axis=1)
    jack = np.array([np.median(np.delete(diff, i)) for i in range(n)])
    ci_lo, ci_hi = bca_bounds(orig, boot, jack, alpha)
    return orig, ci_lo, ci_hi


def bca_ratio(num: np.ndarray, den: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
              seed: int = 42) -> tuple[float, float, float]:
    """Ratio median(|num|) / median(|den|) with a BCa interval.

    Numerator and denominator are resampled with the SAME case indices (a coupled
    pair, not two independent bootstraps).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den)
    num, den = num[mask], den[mask]
    n = len(num)
    if n < 3:
        return float("nan"), float("nan"), float("nan")

    def _stat(a, b):
        db = np.median(np.abs(b))
        if db == 0:
            return float("nan")
        return float(np.median(np.abs(a)) / db)

    orig = _stat(num, den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    num_b, den_b = num[idx], den[idx]
    den_med = np.median(np.abs(den_b), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot = np.median(np.abs(num_b), axis=1) / den_med
    boot = boot[np.isfinite(boot)]
    jack = np.array([_stat(np.delete(num, i), np.delete(den, i)) for i in range(n)])
    if len(boot) < 3 or not np.isfinite(orig):
        return orig, float("nan"), float("nan")
    ci_lo, ci_hi = bca_bounds(orig, boot, jack, alpha)
    return orig, ci_lo, ci_hi


def bca_mean_diff(diff: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                   seed: int = 42) -> tuple[float, float, float]:
    """Mean of paired differences with a BCa interval, fully vectorised (the jackknife
    of the mean has a closed form)."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    orig = float(diff.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    total = diff.sum()
    jack = (total - diff) / (n - 1)
    ci_lo, ci_hi = bca_bounds(orig, boot, jack, alpha)
    return orig, ci_lo, ci_hi
