"""
Step 7.2: Precompute Drift Monitoring Inputs
=============================================
For each label scheme (L1, L2):
  1. Train a provisional initial XGBoost model on all label-matured batches
     at the first evaluation point (leakage-safe via usable_batches).
  2. Rank features by gain importance; select top-K behavioral features to
     monitor (calendar features excluded — they change every batch by
     construction).
  3. Freeze equal-frequency bin edges on the reference batch (last matured
     training batch) and store reference bin proportions.
  4. Precompute bin proportions for every batch under the frozen edges, so
     the Phase-2 experiment can evaluate PSI between any reference/current
     pair as a lookup + sum.

NOTE: hyperparameters are provisional defaults (experiment_config.XGB_PARAMS).
After Phase-2 tuning, rerun this script to regenerate importances/baselines.

Notes for Phase-2 consumers:
  (a) psi_batch_distributions.csv proportions are computed under the
      ORIGINAL frozen edges from this script's reference batch. If S4
      resets its reference after a retraining event, re-derive edges from
      the new reference via drift_utils.make_bins rather than reusing
      these rows.
  (b) Bin edges live in psi_reference.json, keyed by (scheme, feature) —
      join on those keys to recover edges for the CSV.
  (c) Gain values in feature_importance.csv are raw total gain, not
      normalized.

Input:
  - profiling_output/dataset_final.parquet
  - profiling_output/batch_metadata.csv
  - profiling_output/l2_thresholds.csv

Output:
  - profiling_output/initial_model_L1.json / initial_model_L2.json
  - profiling_output/feature_importance.csv
  - profiling_output/psi_reference.json
  - profiling_output/psi_batch_distributions.csv
  - profiling_output/step7_2_drift_summary.txt
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from drift_utils import bin_proportions, compute_psi, make_bins
from experiment_config import (
    BATCH_METADATA,
    DATASET_FINAL,
    FEATURE_COLS,
    FIRST_EVAL_PP,
    L2_THRESHOLDS_PATH,
    MONITOR_EXCLUDE,
    N_MONITORED_FEATURES,
    N_PSI_BINS,
    OUTPUT_DIR,
    PSI_EPSILON,
    XGB_PARAMS,
    label_close_dates,
    usable_batches,
)

SCHEMES = ("L1", "L2")


def main():
    df = pd.read_parquet(DATASET_FINAL)
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)
    close = label_close_dates(batch_meta, l2_thresholds)
    all_pps = sorted(df["prediction_point"].unique())

    importance_rows = []
    dist_rows = []
    psi_reference = {}
    summary = []
    w = summary.append
    w("=" * 70)
    w("STEP 7.2: DRIFT MONITORING BASELINES")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)

    for scheme in SCHEMES:
        train_pps = usable_batches(FIRST_EVAL_PP, scheme, close)
        train = df[df["prediction_point"].isin(train_pps)]
        X = train[FEATURE_COLS]
        y = train[f"label_{scheme}"]
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"{scheme}: degenerate class balance in matured training data "
                f"(pos={n_pos}, neg={n_neg}) — cannot compute scale_pos_weight"
            )
        spw = n_neg / n_pos

        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X, y)
        train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

        model_path = os.path.join(OUTPUT_DIR, f"initial_model_{scheme}.json")
        model.save_model(model_path)

        gains = model.get_booster().get_score(importance_type="gain")
        imp = (
            pd.DataFrame(
                {"feature": FEATURE_COLS,
                 "gain": [gains.get(f, 0.0) for f in FEATURE_COLS]}
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        imp["rank"] = imp.index + 1
        imp["scheme"] = scheme
        importance_rows.append(imp)

        monitored = (
            imp[~imp["feature"].isin(MONITOR_EXCLUDE)]
            .head(N_MONITORED_FEATURES)["feature"]
            .tolist()
        )

        ref_pp = max(train_pps)
        ref = df[df["prediction_point"] == ref_pp]
        feat_info = {}
        for feat in monitored:
            edges = make_bins(ref[feat].to_numpy(dtype=float), N_PSI_BINS)
            props = bin_proportions(ref[feat].to_numpy(dtype=float), edges)
            feat_info[feat] = {
                "edges": edges.tolist(),
                "reference_proportions": props.tolist(),
            }
        psi_reference[scheme] = {
            "reference_pp": f"{pd.Timestamp(ref_pp):%Y-%m-%d}",
            "train_pps": [f"{pd.Timestamp(p):%Y-%m-%d}" for p in train_pps],
            "scale_pos_weight": spw,
            "train_auc_in_sample": float(train_auc),
            "monitored_features": monitored,
            "features": feat_info,
        }

        # Bin proportions for every batch under the frozen edges + PSI vs ref
        psi_vs_ref = {feat: {} for feat in monitored}
        for pp in all_pps:
            cur = df[df["prediction_point"] == pp]
            for feat in monitored:
                edges = np.array(feat_info[feat]["edges"])
                props = bin_proportions(cur[feat].to_numpy(dtype=float), edges)
                psi_vs_ref[feat][pp] = compute_psi(
                    feat_info[feat]["reference_proportions"], props, PSI_EPSILON
                )
                for b, p in enumerate(props):
                    dist_rows.append({
                        "scheme": scheme,
                        "prediction_point": f"{pd.Timestamp(pp):%Y-%m-%d}",
                        "feature": feat,
                        "bin": b,
                        "proportion": p,
                    })

        w("")
        w(f"SCHEME {scheme}")
        w("-" * 70)
        w(f"  Training batches (matured at {FIRST_EVAL_PP:%Y-%m-%d}): "
          f"{len(train_pps)} ({psi_reference[scheme]['train_pps'][0]} .. "
          f"{psi_reference[scheme]['train_pps'][-1]})")
        w(f"  Training rows:       {len(train):,}")
        w(f"  scale_pos_weight:    {spw:.4f}")
        w(f"  In-sample AUC:       {train_auc:.4f}  (provisional hyperparameters)")
        w(f"  Reference batch:     {psi_reference[scheme]['reference_pp']}")
        w(f"  Monitored features:  {', '.join(monitored)}")
        w("")
        w("  Top 10 features by gain:")
        for _, r in imp.head(10).iterrows():
            w(f"    {r['rank']:>2}. {r['feature']:<25} {r['gain']:>12.2f}")
        w("")
        w(f"  PSI vs reference per evaluation batch (threshold 0.2), "
          f"max over monitored features:")
        for pp in all_pps:
            if pd.Timestamp(pp) < FIRST_EVAL_PP:
                continue
            worst = max(monitored, key=lambda f: psi_vs_ref[f][pp])
            w(f"    {pd.Timestamp(pp):%Y-%m-%d}  max PSI = "
              f"{psi_vs_ref[worst][pp]:.4f}  ({worst})")

    pd.concat(importance_rows).to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
    )
    with open(os.path.join(OUTPUT_DIR, "psi_reference.json"), "w") as f:
        json.dump(psi_reference, f, indent=2)
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(OUTPUT_DIR, "psi_batch_distributions.csv"), index=False
    )

    w("")
    w("OUTPUT FILES")
    w("-" * 70)
    for name in ("initial_model_L1.json", "initial_model_L2.json",
                 "feature_importance.csv", "psi_reference.json",
                 "psi_batch_distributions.csv"):
        w(f"  {name}")
    w("=" * 70)

    report = "\n".join(summary)
    with open(os.path.join(OUTPUT_DIR, "step7_2_drift_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
