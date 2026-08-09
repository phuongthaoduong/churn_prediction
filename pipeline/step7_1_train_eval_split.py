"""
Step 7.1: Define Train/Evaluation Split
========================================
Assigns each of the 20 monthly batches a role:
  - initial_train: batches before FIRST_EVAL_PP (2025-04-01) — reserved for
    initial model training and Phase-2 hyperparameter tuning
  - evaluation:    2025-04-01 .. 2026-03-01 — 12 monthly prequential batches

Also records, per batch, the date at which each label scheme becomes fully
observable (label-maturity rule), and snapshots all experiment constants to
experiment_config.json for reproducibility.

Input:
  - profiling_output/batch_metadata.csv
  - profiling_output/l2_thresholds.csv
  - profiling_output/dataset_final.parquet   (row counts per role)

Output:
  - profiling_output/train_eval_split.csv
  - profiling_output/experiment_config.json
  - profiling_output/step7_1_split_summary.txt
"""

import json
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

from experiment_config import (
    AUC_DECLINE_THRESHOLD,
    BATCH_METADATA,
    CLASS_BALANCE,
    DATASET_FINAL,
    FEATURE_COLS,
    FIRST_EVAL_PP,
    L2_THRESHOLDS_PATH,
    MONITOR_EXCLUDE,
    N_MONITORED_FEATURES,
    N_PSI_BINS,
    OUTPUT_DIR,
    PSI_EPSILON,
    PSI_THRESHOLD,
    RECALL_AT_K_PCT,
    ROLLING_WINDOW_MONTHS,
    XGB_PARAMS,
    label_close_dates,
    usable_batches,
)


def main():
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)

    close = label_close_dates(batch_meta, l2_thresholds)
    close["role"] = close["prediction_point"].apply(
        lambda pp: "initial_train" if pp < FIRST_EVAL_PP else "evaluation"
    )
    # `role` alone is a leakage trap: it marks a batch as "initial_train" purely
    # by prediction_point < FIRST_EVAL_PP, regardless of whether that batch's
    # labels are actually observable by FIRST_EVAL_PP. For L2 in particular,
    # only 6 of the 8 initial_train batches are label-matured at the first
    # evaluation point (longer, category-dependent label windows). Phase-2
    # hyperparameter tuning must select training rows on these per-scheme
    # maturity flags, not on `role`.
    close["matured_for_L1_at_first_eval"] = close["l1_label_close"] <= FIRST_EVAL_PP
    close["matured_for_L2_at_first_eval"] = close["l2_label_close"] <= FIRST_EVAL_PP
    eligible = batch_meta.copy()
    eligible["prediction_point"] = pd.to_datetime(eligible["prediction_point"])
    close["eligible_customers"] = close["prediction_point"].map(
        eligible.set_index("prediction_point")["eligible_customers"]
    )

    # Row counts per role from the final dataset (reads one column only)
    pps = pq.read_table(DATASET_FINAL, columns=["prediction_point"]).to_pandas()
    counts = pps["prediction_point"].value_counts()
    close["n_rows"] = close["prediction_point"].map(counts).astype("Int64")

    split_path = os.path.join(OUTPUT_DIR, "train_eval_split.csv")
    close.to_csv(split_path, index=False)

    # Batches usable for the initial models (trained at FIRST_EVAL_PP)
    usable = {s: usable_batches(FIRST_EVAL_PP, s, close) for s in ("L1", "L2")}

    config = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "first_eval_pp": str(FIRST_EVAL_PP.date()),
        "n_batches_total": len(close),
        "n_batches_initial_train": int((close["role"] == "initial_train").sum()),
        "n_batches_evaluation": int((close["role"] == "evaluation").sum()),
        "initial_train_usable_L1": [str(p.date()) for p in usable["L1"]],
        "initial_train_usable_L2": [str(p.date()) for p in usable["L2"]],
        "rolling_window_months_S3": ROLLING_WINDOW_MONTHS,
        "psi_threshold_S4": PSI_THRESHOLD,
        "auc_decline_threshold_S4": AUC_DECLINE_THRESHOLD,
        "n_psi_bins": N_PSI_BINS,
        "psi_epsilon": PSI_EPSILON,
        "n_monitored_features": N_MONITORED_FEATURES,
        "monitor_exclude": MONITOR_EXCLUDE,
        "recall_at_k_pct": RECALL_AT_K_PCT,
        "feature_cols": FEATURE_COLS,
        "xgb_params_provisional": XGB_PARAMS,
        "class_balance_source": os.path.basename(CLASS_BALANCE),
    }
    config_path = os.path.join(OUTPUT_DIR, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    lines = []
    w = lines.append
    w("=" * 70)
    w("STEP 7.1: TRAIN/EVALUATION SPLIT")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)
    w("")
    w(f"Total batches:            {len(close)}")
    w(f"Initial training period:  {config['n_batches_initial_train']} batches "
      f"({close['prediction_point'].min():%Y-%m-%d} .. "
      f"{close.loc[close['role'] == 'initial_train', 'prediction_point'].max():%Y-%m-%d})")
    w(f"Evaluation period:        {config['n_batches_evaluation']} batches "
      f"({FIRST_EVAL_PP:%Y-%m-%d} .. "
      f"{close['prediction_point'].max():%Y-%m-%d})")
    w("")
    w("LABEL MATURITY AT FIRST EVALUATION POINT")
    w("-" * 70)
    w(f"Batches usable for initial L1 model: {len(usable['L1'])} "
      f"(labels close 30 days after prediction point)")
    w(f"Batches usable for initial L2 model: {len(usable['L2'])} "
      f"(labels close up to max category tau, capped 90 days)")
    w("L2 trains on fewer matured batches than L1 by design: under the")
    w("prequential protocol, labels are revealed only after the label window")
    w("closes, and L2 windows are longer.")
    w("")
    w("PER-BATCH DETAIL")
    w("-" * 70)
    w(f"{'prediction_point':<18}{'role':<15}{'L1 close':<13}{'L2 close':<13}{'rows':>10}")
    for _, r in close.iterrows():
        w(f"{r['prediction_point']:%Y-%m-%d}        {r['role']:<15}"
          f"{r['l1_label_close']:%Y-%m-%d}   {r['l2_label_close']:%Y-%m-%d}   "
          f"{r['n_rows']:>8,}")
    w("")
    w("OUTPUT FILES")
    w("-" * 70)
    w(f"  {os.path.basename(split_path)}")
    w(f"  {os.path.basename(config_path)}")
    w("=" * 70)

    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "step7_1_split_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
