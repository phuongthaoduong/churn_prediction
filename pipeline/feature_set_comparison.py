"""
Step 8.2: Feature Set Comparison (Preliminary Research Question)
=================================================================
Compares the LLM-guided 25-feature set against the RFM baseline
(recency, frequency, avg_monetary, ipi_mean, ipi_std, ipi_cv) using the
tuned hyperparameters on the same leakage-safe TSCV folds. The better
mean-AUC set per scheme is frozen for the factorial experiment.

Also produces a mutual-information diagnostic report (thesis eq. 3.10).
DEVIATION (documented): the thesis's MI + importance *filtering* gate is
simplified to a report — the 25-feature set was already constructed and
validated in Step 4; no features are dropped here.

Input:  dataset_final.parquet, batch_metadata.csv, l2_thresholds.csv,
        experiment_output/tuned_params.json
Output: feature_set_comparison.csv, frozen_feature_set.json, mi_report.csv,
        step8_2_feature_summary.txt
"""

import json
import os
from datetime import datetime

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from experiment_config import (
    BATCH_METADATA,
    DATASET_FINAL,
    EXPERIMENT_DIR,
    FEATURE_COLS,
    FROZEN_FEATURES_PATH,
    L2_THRESHOLDS_PATH,
    RFM_FEATURE_COLS,
    TUNED_PARAMS_PATH,
    add_ipi_cv,
    label_close_dates,
    tscv_folds,
)

FEATURE_SETS = {"llm_guided": FEATURE_COLS, "rfm_baseline": RFM_FEATURE_COLS}
MI_SAMPLE = 200_000  # rows for the MI diagnostic (seeded)


def main():
    df = add_ipi_cv(pd.read_parquet(DATASET_FINAL))
    close = label_close_dates(pd.read_csv(BATCH_METADATA),
                              pd.read_csv(L2_THRESHOLDS_PATH))
    tuned = json.load(open(TUNED_PARAMS_PATH))

    rows = []
    for scheme in ("L1", "L2"):
        params = tuned[scheme]["params"]
        for set_name, cols in FEATURE_SETS.items():
            for train_pps, val_pp in tscv_folds(scheme, close):
                train = df[df["prediction_point"].isin(train_pps)]
                val = df[df["prediction_point"] == val_pp]
                y = train[f"label_{scheme}"]
                spw = int((y == 0).sum()) / int((y == 1).sum())
                model = XGBClassifier(**params, scale_pos_weight=spw)
                model.fit(train[cols], y)
                auc = roc_auc_score(val[f"label_{scheme}"],
                                    model.predict_proba(val[cols])[:, 1])
                rows.append({"scheme": scheme, "feature_set": set_name,
                             "val_pp": f"{val_pp:%Y-%m-%d}", "auc": auc})
                print(f"  {scheme} {set_name:<13} val={val_pp:%Y-%m-%d} AUC={auc:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(EXPERIMENT_DIR, "feature_set_comparison.csv"), index=False)

    frozen, lines = {}, ["=" * 70, "STEP 8.2: FEATURE SET COMPARISON",
                         f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70]
    for scheme in ("L1", "L2"):
        agg = (res[res["scheme"] == scheme]
               .groupby("feature_set")["auc"].mean().sort_values(ascending=False))
        winner = agg.index[0]
        frozen[scheme] = {"winner": winner, "feature_cols": FEATURE_SETS[winner],
                          "mean_auc": {k: float(v) for k, v in agg.items()}}
        lines += ["", f"SCHEME {scheme}:"] + [
            f"  {name:<13} mean val AUC = {v:.4f}" for name, v in agg.items()
        ] + [f"  -> frozen: {winner} ({len(FEATURE_SETS[winner])} features)"]

    with open(FROZEN_FEATURES_PATH, "w") as f:
        json.dump(frozen, f, indent=2)

    # MI diagnostic on the pooled L1-matured training slice (seeded sample)
    mi_train_pps = tscv_folds("L1", close)[-1][0]  # largest expanding window
    sample = (df[df["prediction_point"].isin(mi_train_pps)]
              .sample(MI_SAMPLE, random_state=42))
    X = sample[FEATURE_COLS + ["ipi_cv"]].fillna(-999.0)  # sentinel, documented
    mi_rows = []
    for scheme in ("L1", "L2"):
        mi = mutual_info_classif(X, sample[f"label_{scheme}"], random_state=42)
        for feat, v in zip(X.columns, mi):
            mi_rows.append({"scheme": scheme, "feature": feat, "mutual_info": v})
    pd.DataFrame(mi_rows).to_csv(os.path.join(EXPERIMENT_DIR, "mi_report.csv"),
                                 index=False)
    lines += ["", f"MI diagnostic on {MI_SAMPLE:,} sampled rows -> mi_report.csv",
              "(NaN filled with -999 sentinel for MI only; XGBoost uses native NaN)",
              "=" * 70]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_2_feature_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
