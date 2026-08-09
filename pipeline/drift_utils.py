"""
Drift-monitoring utilities: equal-frequency binning and PSI.
=============================================================
PSI follows thesis eq. 3.9: proportions are smoothed with a small epsilon
so empty bins never produce undefined logarithms.

Binning convention: bin 0 is a dedicated missing-value bin (ipi_mean is
47.5% NaN in the feature matrix, ipi_std 63.5% — missingness itself can
drift). Bins 1..len(edges)+1 partition the real line at the interior edges.
"""

import numpy as np


def make_bins(reference, n_bins: int = 10) -> np.ndarray:
    """Interior equal-frequency bin edges from reference values (NaN ignored).

    Returns up to n_bins - 1 strictly increasing edges; duplicate quantiles
    (low-cardinality features) are collapsed, so fewer edges may be returned.
    """
    reference = np.asarray(reference, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.nanquantile(reference, qs))


def bin_proportions(values, edges) -> np.ndarray:
    """Proportion of values in each bin. Index 0 = missing (NaN);
    index b (1-based) = the b-th interval defined by the interior edges,
    right-inclusive. Proportions sum to 1 over all values."""
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError("cannot compute proportions of an empty array")
    missing = np.isnan(values)
    # right=True: value <= edge falls in the lower bin, matching quantile edges
    idx = np.digitize(values[~missing], edges, right=True) + 1
    counts = np.bincount(idx, minlength=len(edges) + 2).astype(float)
    counts[0] = missing.sum()
    return counts / n


def compute_psi(ref_props, cur_props, epsilon: float = 1e-6) -> float:
    """Population Stability Index between two binned distributions."""
    ref = np.asarray(ref_props, dtype=float)
    cur = np.asarray(cur_props, dtype=float)
    if ref.shape != cur.shape:
        raise ValueError(
            f"ref_props and cur_props must have the same length, got "
            f"{ref.shape[0]} and {cur.shape[0]}"
        )
    ref = ref + epsilon
    cur = cur + epsilon
    return float(np.sum((cur - ref) * np.log(cur / ref)))
