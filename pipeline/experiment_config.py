"""
Shared experiment configuration (Step 7 + Phase-2 factorial experiment)
========================================================================
Single source of truth for the train/evaluation split, feature list,
drift-monitoring parameters, and the label-maturity rule that prevents
label leakage at every (re)training event.
"""

import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "profiling_output")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")

DATASET_FINAL = os.path.join(OUTPUT_DIR, "dataset_final.parquet")
BATCH_METADATA = os.path.join(OUTPUT_DIR, "batch_metadata.csv")
L2_THRESHOLDS_PATH = os.path.join(OUTPUT_DIR, "l2_thresholds.csv")
CLASS_BALANCE = os.path.join(OUTPUT_DIR, "class_balance.csv")

# ---------------------------------------------------------------------------
# Temporal split
# 20 batches: 2024-08-01 .. 2026-03-01.
# Initial training period: first 8 batches (2024-08-01 .. 2025-03-01).
# Evaluation: last 12 batches (2025-04-01 .. 2026-03-01) — one year of
# monthly evaluation batches, per the methodology (chap03 §3.2.2).
# ---------------------------------------------------------------------------
FIRST_EVAL_PP = pd.Timestamp("2025-04-01")

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOW_MONTHS = 6      # S3: covers P95 per-customer median IPI (~180d, Step 1 profiling)
PSI_THRESHOLD = 0.2            # S4 feature-drift trigger (Siddiqi 2006; thesis eq. 3.9)
AUC_DECLINE_THRESHOLD = 0.02   # S4 performance-decline trigger; sensitivity-tested in Phase 2
N_PSI_BINS = 10                # equal-frequency bins for PSI
PSI_EPSILON = 1e-6             # smoothing constant for empty bins (thesis eq. 3.9)
N_MONITORED_FEATURES = 10      # top-K features by gain importance to monitor
RECALL_AT_K_PCT = 0.10         # Recall@10% — SME intervention capacity

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
KEY_COLS = ["server_id", "shop_id", "company_id", "prediction_point"]

FEATURE_COLS = [
    # customer-level (14)
    "recency", "frequency", "monetary", "avg_monetary", "ipi_mean", "ipi_std",
    "recent_activity", "freq_decline_ratio", "unique_products", "avg_basket_size",
    "total_qty", "pay_ratio", "debt_frequency", "customer_tenure",
    # store-level (6)
    "store_txn_volume", "store_active_customers", "store_median_ipi",
    "recency_vs_store", "frequency_vs_store", "monetary_vs_store",
    # time-level (5)
    "month", "quarter", "is_cny_season", "is_double11", "is_spring_festival",
]

# Calendar features change deterministically with the prediction point, so they
# would register permanent "drift" — excluded from PSI monitoring.
MONITOR_EXCLUDE = ["month", "quarter", "is_cny_season", "is_double11", "is_spring_festival"]

# ---------------------------------------------------------------------------
# Provisional XGBoost hyperparameters.
# Used by step7_2 to derive feature-importance rankings for drift monitoring.
# Phase 2 tunes hyperparameters once via time-series CV on the initial
# training period, then freezes them; rerun step7_2 afterwards to regenerate
# the monitored-feature list under the tuned model.
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


def label_close_dates(batch_meta: pd.DataFrame, l2_thresholds: pd.DataFrame) -> pd.DataFrame:
    """Per batch, the date at which each scheme's labels are fully observable.

    L1 closes at l1_label_end (prediction_point + 30 days).
    L2 closes at prediction_point + max(tau_c) over categories at that
    prediction point (tau_c is already capped at 90 days upstream).
    """
    out = pd.DataFrame({
        "prediction_point": pd.to_datetime(batch_meta["prediction_point"]),
        "l1_label_close": pd.to_datetime(batch_meta["l1_label_end"]),
    })
    tau = l2_thresholds.copy()
    tau["prediction_point"] = pd.to_datetime(tau["prediction_point"])
    max_tau = tau.groupby("prediction_point")["tau_c"].max()
    out["l2_label_close"] = out["prediction_point"] + pd.to_timedelta(
        out["prediction_point"].map(max_tau), unit="D"
    )
    missing = out.loc[out["l2_label_close"].isna(), "prediction_point"]
    if not missing.empty:
        raise ValueError(
            "No L2 thresholds found for prediction point(s): "
            + ", ".join(f"{p:%Y-%m-%d}" for p in missing)
        )
    return out


def usable_batches(t, scheme: str, close_dates: pd.DataFrame) -> list:
    """Prediction points whose labels are fully observable at time t.

    This is the no-leakage rule for every (re)training event: a model
    trained at time t may only use batches returned by this function.
    `t` and the close-date columns are tz-naive, date-level timestamps.
    """
    if scheme == "L1":
        col = "l1_label_close"
    elif scheme == "L2":
        col = "l2_label_close"
    else:
        raise ValueError(f"Unknown scheme: {scheme!r} (expected 'L1' or 'L2')")
    ok = close_dates[close_dates[col] <= pd.Timestamp(t)]
    return sorted(ok["prediction_point"].tolist())


# ---------------------------------------------------------------------------
# Phase 2: factorial experiment
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiment_output")

CLEANED_SALES = os.path.join(OUTPUT_DIR, "srv_sales_step2_3.parquet")
TRAIN_EVAL_SPLIT = os.path.join(OUTPUT_DIR, "train_eval_split.csv")
PSI_REFERENCE_PATH = os.path.join(OUTPUT_DIR, "psi_reference.json")
PSI_DISTRIBUTIONS_PATH = os.path.join(OUTPUT_DIR, "psi_batch_distributions.csv")

EVAL_LABELS_PATH = os.path.join(EXPERIMENT_DIR, "eval_labels.parquet")
TUNING_RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "tuning_results.csv")
TUNED_PARAMS_PATH = os.path.join(EXPERIMENT_DIR, "tuned_params.json")
FROZEN_FEATURES_PATH = os.path.join(EXPERIMENT_DIR, "frozen_feature_set.json")
PREDICTIONS_DIR = os.path.join(EXPERIMENT_DIR, "predictions")
RETRAIN_LOG_PATH = os.path.join(EXPERIMENT_DIR, "retrain_log.csv")

# Shared downstream evaluation outcome: churn = no purchase in [pp, pp + 60d).
# 60d = P75 of per-customer median IPI (Step 1 profiling); between L1's 30d
# and L2's 90d cap; closes inside the data boundary for all 20 batches.
EVAL_HORIZON_DAYS = 60

# RFM baseline (thesis eq. 3.11-3.13 + IPI statistics). ipi_cv derived.
RFM_FEATURE_COLS = ["recency", "frequency", "avg_monetary",
                    "ipi_mean", "ipi_std", "ipi_cv"]

STRATEGIES = ("S1", "S2", "S3", "S4")

# Maturity modes: "scheme" = primary analysis (Step-7 scheme-specific rule);
# "equalized" = sensitivity arm with a uniform lag for BOTH schemes, removing
# the differential data recency inherent in scheme-specific maturity.
MATURITY_MODES = ("scheme", "equalized")
EQUALIZED_LAG_DAYS = 90

F1_THRESHOLD = 0.5           # probability cutoff for F1 (documented decision)
N_TUNING_SAMPLES = 20        # random-search draws, seed 42
TUNING_ROW_CAP = 500_000     # per-fold training row cap during tuning
MIN_CV_TRAIN_BATCHES = 2     # smallest expanding-window training set

TUNING_PARAM_SPACE = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 300, 500],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def add_ipi_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ipi_cv = ipi_std / ipi_mean (NaN where undefined)."""
    out = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = out["ipi_std"].to_numpy(dtype=float) / out["ipi_mean"].to_numpy(dtype=float)
    cv[~np.isfinite(cv)] = np.nan
    out["ipi_cv"] = cv
    return out


def equalized_close_dates(batch_meta: pd.DataFrame) -> pd.DataFrame:
    """Close dates under the equalized-lag sensitivity arm.

    Labels of BOTH schemes are treated as observable exactly
    EQUALIZED_LAG_DAYS after the prediction point, so L1 and L2 train on
    identically aged data. Downstream code (usable_batches, strategies) is
    unchanged — the arm is selected purely by which close-dates frame is
    constructed.
    """
    pps = pd.to_datetime(batch_meta["prediction_point"])
    close = pps + pd.Timedelta(days=EQUALIZED_LAG_DAYS)
    return pd.DataFrame({
        "prediction_point": pps,
        "l1_label_close": close,
        "l2_label_close": close,
    })


def tscv_folds(scheme: str, close_dates: pd.DataFrame,
               min_train: int = MIN_CV_TRAIN_BATCHES) -> list:
    """Expanding-window time-series CV folds inside the initial training period.

    A fold (train_pps, val_pp) validates on batch val_pp using models trained
    only on batches whose labels closed by val_pp (usable_batches), so the
    fold structure itself enforces the no-leakage rule. Validation batches are
    restricted to batches matured at FIRST_EVAL_PP (their labels must be
    observable when tuning runs).
    """
    matured = usable_batches(FIRST_EVAL_PP, scheme, close_dates)
    folds = []
    for val_pp in matured:
        train_pps = usable_batches(val_pp, scheme, close_dates)
        if len(train_pps) >= min_train:
            folds.append((train_pps, val_pp))
    return folds
