"""
Shared experiment configuration (Step 7 + Phase-2 factorial experiment)
========================================================================
Single source of truth for the train/evaluation split, feature list,
drift-monitoring parameters, and the label-maturity rule that prevents
label leakage at every (re)training event.
"""

import os

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
    return out


def usable_batches(t, scheme: str, close_dates: pd.DataFrame) -> list:
    """Prediction points whose labels are fully observable at time t.

    This is the no-leakage rule for every (re)training event: a model
    trained at time t may only use batches returned by this function.
    """
    if scheme == "L1":
        col = "l1_label_close"
    elif scheme == "L2":
        col = "l2_label_close"
    else:
        raise ValueError(f"Unknown scheme: {scheme!r} (expected 'L1' or 'L2')")
    ok = close_dates[close_dates[col] <= pd.Timestamp(t)]
    return sorted(ok["prediction_point"].tolist())
