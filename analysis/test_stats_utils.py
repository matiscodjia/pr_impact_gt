"""Tests sur données synthétiques dont le résultat est connu analytiquement."""

import numpy as np

from stats_utils import (bca_mean_diff, bca_median_diff, bca_ratio,
                          bh_fdr, paired_cohens_d, wilcoxon_p_safe)


def test_bh_fdr_known_example():
    # Exemple manuel (Benjamini-Hochberg, n=4) :
    # p = [0.01, 0.02, 0.03, 0.20] triés -> rang 1..4
    # q = p*n/rang puis minimum cumulatif depuis la fin :
    #   rang4: 0.20*4/4=0.20
    #   rang3: min(0.03*4/3=0.04, 0.20)=0.04
    #   rang2: min(0.02*4/2=0.04, 0.04)=0.04
    #   rang1: min(0.01*4/1=0.04, 0.04)=0.04
    p = np.array([0.01, 0.02, 0.03, 0.20])
    q = bh_fdr(p)
    expected = np.array([0.04, 0.04, 0.04, 0.20])
    np.testing.assert_allclose(q, expected, atol=1e-9)


def test_bh_fdr_nan_treated_as_one():
    p = np.array([0.001, np.nan])
    q = bh_fdr(p)
    assert q[1] == 1.0
    assert q[0] <= 1.0


def test_bh_fdr_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)
    q = bh_fdr(p)
    assert np.all(q >= 0) and np.all(q <= 1)
    # q_(i) doit être croissant une fois p trié (propriété BH)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_paired_cohens_d_known_value():
    # diff constant + bruit nul -> d = mean/std ; ici diff = [1,3] -> mean=2, std=sqrt(2)
    diff = np.array([1.0, 3.0])
    d = paired_cohens_d(diff)
    assert np.isclose(d, 2.0 / np.sqrt(2.0))


def test_paired_cohens_d_zero_variance_is_nan():
    diff = np.array([5.0, 5.0, 5.0])
    assert np.isnan(paired_cohens_d(diff))


def test_wilcoxon_p_safe_all_equal_is_nan():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert np.isnan(wilcoxon_p_safe(a, b))


def test_wilcoxon_p_safe_obvious_difference_is_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=40)
    b = a + 5.0  # décalage énorme, systématique -> p doit être minuscule
    p = wilcoxon_p_safe(a, b)
    assert p < 1e-6


def test_bca_median_diff_degenerate_array():
    # tous les cas identiques -> médiane exacte, IC dégénéré au même point
    diff = np.full(85, 3.0)
    med, lo, hi = bca_median_diff(diff, n_boot=500, seed=42)
    assert med == 3.0
    assert lo == 3.0 and hi == 3.0


def test_bca_median_diff_recovers_true_median_on_symmetric_data():
    # Distribution symétrique autour de mu=2 -> la médiane empirique et l'IC doivent
    # s'encadrer mutuellement (l'IC doit contenir le point estimé) ; pour une coverage
    # du paramètre théorique (flaky à ~5% sur un seul tirage), on utilise un n large
    # pour réduire l'écart-type de la médiane (~1.2533*sigma/sqrt(n)) sous 0.05.
    rng = np.random.default_rng(7)
    diff = rng.normal(loc=2.0, scale=1.0, size=5000)
    med, lo, hi = bca_median_diff(diff, n_boot=2000, seed=42)
    assert lo <= med <= hi
    assert np.isclose(med, np.median(diff))
    assert abs(med - 2.0) < 0.05  # SE théorique ~0.018 ici, marge large


def test_bca_mean_diff_matches_closed_form_jackknife():
    rng = np.random.default_rng(3)
    diff = rng.normal(0, 1, size=50)
    mean, lo, hi = bca_mean_diff(diff, n_boot=2000, seed=42)
    assert np.isclose(mean, diff.mean())
    assert lo < mean < hi


def test_bca_ratio_known_constant_ratio():
    # num toujours 3x den en valeur absolue -> ratio exact = 3 partout, y compris bootstrap.
    rng = np.random.default_rng(5)
    den = rng.normal(0, 1, size=90)
    den[den == 0] = 0.1
    num = 3.0 * den
    ratio, lo, hi = bca_ratio(num, den, n_boot=500, seed=42)
    assert np.isclose(ratio, 3.0)
    assert np.isclose(lo, 3.0, atol=1e-9)
    assert np.isclose(hi, 3.0, atol=1e-9)


def test_bca_ratio_five_to_one():
    rng = np.random.default_rng(9)
    den = rng.normal(0, 2, size=85)
    den[np.abs(den) < 0.05] += 1.0
    num = 5.0 * den + rng.normal(0, 0.01, size=85)  # quasi-exact 5x, bruit négligeable
    ratio, lo, hi = bca_ratio(num, den, n_boot=2000, seed=42)
    assert 4.5 < ratio < 5.5
    assert lo < ratio < hi
