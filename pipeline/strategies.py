"""
Update-strategy decision logic for the 2x4 factorial experiment.
=================================================================
S1 static, S2 periodic (all matured data), S3 rolling window,
S4 drift-triggered (PSI or AUC-decline; rolling window data).

All training-batch selection flows through experiment_config.usable_batches,
so the label-maturity/no-leakage rule cannot be bypassed by any strategy.
"""

import pandas as pd

from experiment_config import ROLLING_WINDOW_MONTHS, usable_batches


def training_batches(strategy, t, scheme, close_dates,
                     rolling_months: int = ROLLING_WINDOW_MONTHS) -> list:
    """Batches a retraining event at time t may use, per strategy.

    S1 uses this only for its single initial fit (all matured data).
    S2: all matured batches. S3/S4: matured batches whose prediction point
    lies in [t - rolling_months, t).
    """
    if strategy not in ("S1", "S2", "S3", "S4"):
        raise ValueError(f"Unknown strategy: {strategy!r}")
    matured = usable_batches(t, scheme, close_dates)
    if strategy in ("S1", "S2"):
        return matured
    window_start = pd.Timestamp(t) - pd.DateOffset(months=rolling_months)
    return [pp for pp in matured if pp >= window_start]


def s4_signals(psi_max, auc_history, psi_threshold, auc_threshold):
    """Drift-trigger signals at a decision point.

    psi_max: max PSI over monitored features, current batch vs reference.
    auc_history: own-label AUCs of batches matured since the last training
    event, in chronological order (last entry = most recent matured batch).
    The AUC signal compares the most recent matured AUC against the running
    mean of the earlier ones (thesis eq. 3.9-3.10 vicinity) and needs at
    least two entries to be defined.
    """
    psi_fired = bool(psi_max > psi_threshold)
    auc_fired = False
    if len(auc_history) >= 2:
        running_mean = sum(auc_history[:-1]) / (len(auc_history) - 1)
        auc_fired = bool(running_mean - auc_history[-1] > auc_threshold)
    return psi_fired, auc_fired


def should_retrain(strategy, psi_fired, auc_fired) -> bool:
    """Whether a strategy retrains at this decision point."""
    if strategy == "S1":
        return False
    if strategy in ("S2", "S3"):
        return True
    if strategy == "S4":
        return psi_fired or auc_fired
    raise ValueError(f"Unknown strategy: {strategy!r}")
