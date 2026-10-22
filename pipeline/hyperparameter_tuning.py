"""
Step 8.1: Hyperparameter Tuning (tune once, then freeze)
=========================================================
Random search (N_TUNING_SAMPLES combos, seed 42) per label scheme, scored by
mean validation AUC over the expanding-window TSCV folds from
experiment_config.tscv_folds (leakage-safe by construction). Training folds
are row-capped at TUNING_ROW_CAP (seeded sample) for tractability;
validation batches are scored in full.

Checkpointing: each (scheme, combo, fold) result is appended to
tuning_results.csv as soon as it finishes; on rerun, completed combos are
skipped, so the script can be interrupted freely.

Only scale_pos_weight is recomputed per fit (thesis eq. 3.8); everything
else in a combo is frozen across folds.

Input:  profiling_output/step6_dataset/dataset_final.parquet,
        profiling_output/step3_temporal/batch_metadata.csv,
        profiling_output/step5_labels/l2_thresholds.csv
Output: experiment_output/tuning_results.csv, tuned_params.json,
        step8_1_tuning_summary.txt
"""

import itertools
import json
import os
import random
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from experiment_config import (
    BATCH_METADATA,
    DATASET_FINAL,
    EXPERIMENT_DIR,
    FEATURE_COLS,
    L2_THRESHOLDS_PATH,
    N_TUNING_SAMPLES,
    TUNED_PARAMS_PATH,
    TUNING_PARAM_SPACE,
    TUNING_RESULTS_PATH,
    TUNING_ROW_CAP,
    label_close_dates,
    tscv_folds,
)

FIXED = {"objective": "binary:logistic", "tree_method": "hist",
         "eval_metric": "auc", "random_state": 42, "n_jobs": -1}


def sample_combos(n, seed=42):
    keys = sorted(TUNING_PARAM_SPACE)
    all_combos = list(itertools.product(*(TUNING_PARAM_SPACE[k] for k in keys)))
    rng = random.Random(seed)
    picked = rng.sample(all_combos, n)
    return [dict(zip(keys, combo)) for combo in picked]


def combo_id(params):
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def fit_and_score(df, train_pps, val_pp, scheme, params):
    train = df[df["prediction_point"].isin(train_pps)]
    if len(train) > TUNING_ROW_CAP:
        train = train.sample(TUNING_ROW_CAP, random_state=42)
    val = df[df["prediction_point"] == val_pp]
    y_tr = train[f"label_{scheme}"]
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"degenerate fold {val_pp} for {scheme}")
    model = XGBClassifier(**FIXED, **params, scale_pos_weight=n_neg / n_pos)
    t0 = time.time()
    model.fit(train[FEATURE_COLS], y_tr)
    fit_s = time.time() - t0
    auc = roc_auc_score(val[f"label_{scheme}"],
                        model.predict_proba(val[FEATURE_COLS])[:, 1])
    return auc, fit_s


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    df = pd.read_parquet(DATASET_FINAL)
    close = label_close_dates(pd.read_csv(BATCH_METADATA),
                              pd.read_csv(L2_THRESHOLDS_PATH))
    combos = sample_combos(N_TUNING_SAMPLES)

    done = set()
    if os.path.exists(TUNING_RESULTS_PATH):
        prev = pd.read_csv(TUNING_RESULTS_PATH)
        done = set(zip(prev["scheme"], prev["combo_id"], prev["val_pp"]))
        print(f"Resuming: {len(done)} fold-results already recorded")
    else:
        pd.DataFrame(columns=["scheme", "combo_id", "val_pp", "auc",
                              "fit_seconds"]).to_csv(TUNING_RESULTS_PATH, index=False)

    for scheme in ("L1", "L2"):
        folds = tscv_folds(scheme, close)
        print(f"{scheme}: {len(folds)} folds x {len(combos)} combos")
        for params in combos:
            cid = combo_id(params)
            for train_pps, val_pp in folds:
                key = (scheme, cid, f"{val_pp:%Y-%m-%d}")
                if key in done:
                    continue
                auc, fit_s = fit_and_score(df, train_pps, val_pp, scheme, params)
                pd.DataFrame([{"scheme": scheme, "combo_id": cid,
                               "val_pp": f"{val_pp:%Y-%m-%d}",
                               "auc": auc, "fit_seconds": fit_s}]).to_csv(
                    TUNING_RESULTS_PATH, mode="a", header=False, index=False)
                print(f"  {scheme} {cid} val={val_pp:%Y-%m-%d} "
                      f"AUC={auc:.4f} ({fit_s:.0f}s)")

    results = pd.read_csv(TUNING_RESULTS_PATH)
    tuned, lines = {}, ["=" * 70, "STEP 8.1: HYPERPARAMETER TUNING",
                       f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70]
    for scheme in ("L1", "L2"):
        agg = (results[results["scheme"] == scheme]
               .groupby("combo_id")["auc"].agg(["mean", "std", "count"])
               .sort_values("mean", ascending=False))
        n_folds = len(tscv_folds(scheme, close))
        complete = agg[agg["count"] == n_folds]
        if complete.empty:
            raise ValueError(f"{scheme}: no combo has all {n_folds} folds finished")
        best_id = complete.index[0]
        best = next(p for p in sample_combos(N_TUNING_SAMPLES)
                    if combo_id(p) == best_id)
        tuned[scheme] = {"params": {**FIXED, **best},
                         "cv_mean_auc": float(complete.loc[best_id, "mean"]),
                         "cv_std_auc": float(complete.loc[best_id, "std"]),
                         "n_folds": int(n_folds)}
        lines += ["", f"SCHEME {scheme} — best of {len(complete)} complete combos:",
                  f"  {best_id}",
                  f"  mean val AUC = {tuned[scheme]['cv_mean_auc']:.4f} "
                  f"(std {tuned[scheme]['cv_std_auc']:.4f}, {n_folds} folds)"]

    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(tuned, f, indent=2)
    lines += ["", f"Frozen to {os.path.basename(TUNED_PARAMS_PATH)}", "=" * 70]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_1_tuning_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
