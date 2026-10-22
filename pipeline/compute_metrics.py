"""
Step 8.4: Per-Batch Metrics
============================
For every (maturity mode, scheme, strategy, evaluation batch): AUC / F1 /
Recall@10% against the SHARED 60-day evaluation label (primary, cross-scheme
comparable) and against the config's own training label (secondary
diagnostic), plus the Jaccard similarity of consecutive top-10% risk lists.

The "scheme" maturity mode is the PRIMARY analysis; the "equalized" mode
(uniform 90-day lag, both schemes) is the pre-specified sensitivity arm.
The summary reports both, plus the equalized-minus-scheme AUC delta per
configuration and the mean effective data lag per configuration.

Summary aggregation per configuration: mean and CV of each metric across the
12 batches, joined with retraining cost (events, rows, cumulative fit time).
Statistical testing (Wilcoxon etc.) is Phase 2b.

Input:  experiment_output/predictions/<mode>/<strategy>/*.parquet,
        eval_labels.parquet, retrain_log.csv
Output: metrics_per_batch.csv, metrics_summary.csv, step8_4_metrics_summary.txt
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_config import (
    EVAL_LABELS_PATH,
    EXPERIMENT_DIR,
    F1_THRESHOLD,
    KEY_COLS,
    MATURITY_MODES,
    PREDICTIONS_DIR,
    RECALL_AT_K_PCT,
    RETRAIN_LOG_PATH,
    STRATEGIES,
)
from metrics import f1_at_threshold, jaccard, recall_at_frac, top_frac_keys

SCHEMES = ("L1", "L2")


def main():
    ev = pd.read_parquet(EVAL_LABELS_PATH)
    rows = []
    for mode in MATURITY_MODES:
        pred_dir = os.path.join(PREDICTIONS_DIR, mode)
        for scheme in SCHEMES:
            for strategy in STRATEGIES:
                strat_dir = os.path.join(pred_dir, strategy)
                prev_top = None
                files = sorted(f for f in os.listdir(strat_dir)
                               if f.startswith(f"{scheme}_"))
                for fname in files:
                    pp = fname.rsplit("_", 1)[1].replace(".parquet", "")
                    p = pd.read_parquet(os.path.join(strat_dir, fname))
                    p = p.merge(ev, on=KEY_COLS, how="left", validate="1:1")
                    if p["label_eval"].isna().any():
                        raise ValueError(f"{mode}/{fname}: rows missing eval label")
                    keys = (p["server_id"].astype(str) + "|"
                            + p["shop_id"].astype(str) + "|"
                            + p["company_id"].astype(str)).to_numpy()
                    top = top_frac_keys(keys, p["y_prob"].to_numpy(),
                                        RECALL_AT_K_PCT)
                    rows.append({
                        "maturity_mode": mode, "scheme": scheme,
                        "strategy": strategy, "prediction_point": pp,
                        "n_scored": len(p),
                        "auc_shared": roc_auc_score(p["label_eval"], p["y_prob"]),
                        "f1_shared": f1_at_threshold(p["label_eval"], p["y_prob"],
                                                     F1_THRESHOLD),
                        "recall10_shared": recall_at_frac(p["label_eval"],
                                                          p["y_prob"],
                                                          RECALL_AT_K_PCT),
                        "auc_own": roc_auc_score(p[f"label_{scheme}"], p["y_prob"]),
                        "jaccard_top10_vs_prev": (np.nan if prev_top is None
                                                  else jaccard(top, prev_top)),
                    })
                    prev_top = top

    per_batch = pd.DataFrame(rows)
    per_batch.to_csv(os.path.join(EXPERIMENT_DIR, "metrics_per_batch.csv"),
                     index=False)

    log = pd.read_csv(RETRAIN_LOG_PATH)
    grp = ["maturity_mode", "scheme", "strategy"]
    cost = (log[log["trigger"] != "initial"]
            .groupby(grp)
            .agg(n_retrains=("t", "count"),
                 total_train_rows=("n_train_rows", "sum"),
                 total_fit_seconds=("fit_seconds", "sum")))
    lag = log.groupby(grp).agg(
        mean_effective_lag_days=("effective_lag_days", "mean"))
    perf = per_batch.groupby(grp).agg(
        mean_auc_shared=("auc_shared", "mean"),
        cv_auc_shared=("auc_shared", lambda s: s.std() / s.mean()),
        mean_f1_shared=("f1_shared", "mean"),
        mean_recall10=("recall10_shared", "mean"),
        mean_auc_own=("auc_own", "mean"),
        mean_jaccard=("jaccard_top10_vs_prev", "mean"),
    )
    summary = perf.join(cost, how="left").join(lag, how="left").fillna(
        {"n_retrains": 0, "total_train_rows": 0, "total_fit_seconds": 0})
    summary.to_csv(os.path.join(EXPERIMENT_DIR, "metrics_summary.csv"))

    # Sensitivity delta: equalized minus scheme, per (scheme, strategy)
    wide = summary["mean_auc_shared"].unstack("maturity_mode")
    wide["delta_equalized_minus_scheme"] = wide["equalized"] - wide["scheme"]

    lines = ["=" * 70, "STEP 8.4: FACTORIAL METRICS SUMMARY",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70, "",
             "PRIMARY ANALYSIS (maturity mode: scheme)", "-" * 70,
             summary.loc["scheme"].round(4).to_string(), "",
             "SENSITIVITY ARM (maturity mode: equalized, uniform 90-day lag)",
             "-" * 70,
             summary.loc["equalized"].round(4).to_string(), "",
             "Mean shared-outcome AUC, equalized minus scheme:", "-" * 70,
             wide.round(4).to_string(), "",
             "Primary metric: auc_shared / recall10_shared (60-day shared outcome).",
             "auc_own is the secondary own-label diagnostic. The equalized arm",
             "removes the differential data recency of scheme-specific maturity;",
             "agreement with the primary analysis attributes scheme differences",
             "to the labels themselves rather than to training-data staleness.",
             "=" * 70]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_4_metrics_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
