import numpy as np
import pytest

from drift_utils import bin_proportions, compute_psi, make_bins


def test_make_bins_returns_interior_edges():
    values = np.arange(100, dtype=float)  # 0..99
    edges = make_bins(values, n_bins=10)
    assert len(edges) == 9               # 9 interior edges -> 10 bins
    assert edges[0] == pytest.approx(9.9)
    assert np.all(np.diff(edges) > 0)


def test_make_bins_collapses_duplicate_edges():
    # constant feature -> quantiles identical -> a single unique edge
    edges = make_bins(np.ones(50), n_bins=10)
    assert len(edges) == 1


def test_make_bins_ignores_nan():
    values = np.concatenate([np.arange(100, dtype=float), [np.nan] * 100])
    edges = make_bins(values, n_bins=10)
    assert not np.any(np.isnan(edges))
    assert len(edges) == 9


def test_bin_proportions_sum_to_one_and_bin0_is_missing():
    values = np.array([np.nan, np.nan, 1.0, 5.0, 9.0])
    edges = np.array([3.0, 7.0])         # 3 value bins + missing bin
    props = bin_proportions(values, edges)
    assert len(props) == 4               # bin0 missing + 3 value bins
    assert props.sum() == pytest.approx(1.0)
    assert props[0] == pytest.approx(2 / 5)   # 2 NaN of 5
    assert props[1] == pytest.approx(1 / 5)   # 1.0  <= 3
    assert props[2] == pytest.approx(1 / 5)   # 5.0 in (3, 7]
    assert props[3] == pytest.approx(1 / 5)   # 9.0  > 7


def test_psi_identical_distributions_is_zero():
    props = np.array([0.0, 0.25, 0.25, 0.5])
    assert compute_psi(props, props) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 10_000)
    cur = rng.normal(1.0, 1, 10_000)     # mean shift of 1 sd
    edges = make_bins(ref, n_bins=10)
    psi = compute_psi(bin_proportions(ref, edges), bin_proportions(cur, edges))
    assert psi > 0.2                     # clearly above the drift threshold


def test_psi_empty_bin_is_finite():
    ref = np.array([0.5, 0.5, 0.0])
    cur = np.array([0.0, 0.5, 0.5])
    psi = compute_psi(ref, cur, epsilon=1e-6)
    assert np.isfinite(psi)
    assert psi > 0


def test_psi_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        compute_psi(np.array([1.0]), np.array([0.2, 0.3, 0.5]))


def test_bin_proportions_rejects_empty_values():
    with pytest.raises(ValueError):
        bin_proportions(np.array([]), np.array([1.0, 2.0]))
