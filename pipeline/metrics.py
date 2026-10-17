"""
Evaluation metrics for the factorial experiment (thesis eq. 3.12-3.15).
Recall@k reflects SME intervention capacity; Jaccard measures top-k list
stability between consecutive batches.
"""

import math

import numpy as np
from sklearn.metrics import f1_score


def recall_at_frac(y_true, y_prob, frac) -> float:
    """Share of true churners captured in the top ceil(frac*n) by risk.
    NaN when the batch has no churners (undefined)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_churn = int(y_true.sum())
    if n_churn == 0:
        return float("nan")
    k = math.ceil(frac * len(y_true))
    top_idx = np.argsort(-y_prob)[:k]
    return float(y_true[top_idx].sum() / n_churn)


def f1_at_threshold(y_true, y_prob, threshold) -> float:
    return float(f1_score(np.asarray(y_true),
                          (np.asarray(y_prob) >= threshold).astype(int)))


def top_frac_keys(keys, y_prob, frac) -> set:
    """Set of customer keys in the top ceil(frac*n) by predicted risk."""
    keys = np.asarray(keys)
    y_prob = np.asarray(y_prob)
    k = math.ceil(frac * len(keys))
    return set(keys[np.argsort(-y_prob)[:k]])


def jaccard(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|; 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
