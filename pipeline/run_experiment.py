"""
Step 8.3: Prequential Factorial Experiment Run
===============================================
For each of the 8 configurations (L1/L2 x S1-S4), walks the 12 evaluation
batches in order. At each batch t:
  1. The strategy may retrain (S2/S3 monthly; S4 on PSI/AUC signal; S1 never)
     using ONLY label-matured batches (strategies.training_batches).
  2. The current model scores batch t; predictions are persisted.

PSI for S4 is a pure lookup in the frozen-edge psi_batch_distributions.csv
(reference batch = last matured batch at the previous training event).
The S4 AUC signal uses own-label AUCs of matured batches only.

The full run executes TWICE, once per maturity mode:
  - "scheme"    (PRIMARY): Step-7 scheme-specific label-maturity rule
  - "equalized" (SENSITIVITY): uniform 90-day lag for both schemes,
    removing the differential data recency of scheme-specific maturity
Predictions are written under predictions/<mode>/<strategy>/; every training
event logs its maturity_mode and effective_lag_days (days from the newest
usable batch to the training time).

Deterministic: random_state=42 everywhere; rerunning reproduces the run.
NOT resumable mid-config (model state is sequential): rerunning redoes all
configs and overwrites predictions/ and retrain_log.csv.

Input:  dataset_final.parquet, train_eval_split metadata, tuned_params.json,
        frozen_feature_set.json, psi_reference.json,
        psi_batch_distributions.csv
Output: experiment_output/predictions/<mode>/<strategy>/*.parquet,
        retrain_log.csv, step8_3_run_summary.txt
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from experiment_config import (
    AUC_DECLINE_THRESHOLD,
    BATCH_METADATA,
    DATASET_FINAL,
    EQUALIZED_LAG_DAYS,
    EXPERIMENT_DIR,
    FIRST_EVAL_PP,
    FROZEN_FEATURES_PATH,
    L2_THRESHOLDS_PATH,
    MATURITY_MODES,
    PREDICTIONS_DIR,
    PSI_DISTRIBUTIONS_PATH,
    PSI_REFERENCE_PATH,
    PSI_THRESHOLD,
    RETRAIN_LOG_PATH,
    STRATEGIES,
    TUNED_PARAMS_PATH,
    add_ipi_cv,
    equalized_close_dates,
    label_close_dates,
)
from drift_utils import compute_psi
from strategies import s4_signals, should_retrain, training_batches

SCHEMES = ("L1", "L2")


def fit_model(df, pps, scheme, cols, params):
    train = df[df["prediction_point"].isin(pps)]
    y = train[f"label_{scheme}"]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"degenerate training set for {scheme} at {pps}")
    model = XGBClassifier(**params, scale_pos_weight=n_neg / n_pos)
    t0 = time.time()
    model.fit(train[cols], y)
    return model, len(train), time.time() - t0


def psi_lookup(dist, scheme, monitored, ref_pp, cur_pp):
    """Max PSI over monitored features between two batches (frozen edges)."""
    sub = dist[dist["scheme"] == scheme]
    worst = 0.0
    for feat in monitored:
        ref = sub[(sub["prediction_point"] == ref_pp) & (sub["feature"] == feat)]
        cur = sub[(sub["prediction_point"] == cur_pp) & (sub["feature"] == feat)]
        worst = max(worst, compute_psi(ref.sort_values("bin")["proportion"].to_numpy(),
                                       cur.sort_values("bin")["proportion"].to_numpy()))
    return worst


def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    df = add_ipi_cv(pd.read_parquet(DATASET_FINAL))
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)
    tuned = json.load(open(TUNED_PARAMS_PATH))
    frozen = json.load(open(FROZEN_FEATURES_PATH))
    psi_ref = json.load(open(PSI_REFERENCE_PATH))
    dist = pd.read_csv(PSI_DISTRIBUTIONS_PATH)

    all_pps = sorted(df["prediction_point"].unique())
    eval_pps = [pp for pp in all_pps if pd.Timestamp(pp) >= FIRST_EVAL_PP]
    assert len(eval_pps) == 12

    retrain_rows, summary = [], []
    w = summary.append
    w("=" * 70)
    w("STEP 8.3: PREQUENTIAL FACTORIAL RUN")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)

    for mode in MATURITY_MODES:
        close = (label_close_dates(batch_meta, l2_thresholds)
                 if mode == "scheme" else equalized_close_dates(batch_meta))
        close_idx = close.set_index("prediction_point")
        pred_dir = os.path.join(PREDICTIONS_DIR, mode)
        os.makedirs(pred_dir, exist_ok=True)
        w("")
        w(f"MATURITY MODE: {mode}"
          + ("" if mode == "scheme"
             else f" (uniform {EQUALIZED_LAG_DAYS}-day lag, both schemes)"))

        for scheme in SCHEMES:
            params = tuned[scheme]["params"]
            cols = frozen[scheme]["feature_cols"]
            monitored = psi_ref[scheme]["monitored_features"]
            label_close_col = "l1_label_close" if scheme == "L1" else "l2_label_close"

            for strategy in STRATEGIES:
                strat_dir = os.path.join(pred_dir, strategy)
                os.makedirs(strat_dir, exist_ok=True)

                # Initial fit at FIRST_EVAL_PP
                init_pps = training_batches(
                    strategy if strategy in ("S3", "S4") else "S2",
                    FIRST_EVAL_PP, scheme, close)
                model, n_rows, fit_s = fit_model(df, init_pps, scheme, cols, params)
                retrain_rows.append({
                    "maturity_mode": mode, "scheme": scheme, "strategy": strategy,
                    "t": f"{FIRST_EVAL_PP:%Y-%m-%d}", "trigger": "initial",
                    "n_train_rows": n_rows, "n_train_batches": len(init_pps),
                    "effective_lag_days": int((FIRST_EVAL_PP - max(init_pps)).days),
                    "fit_seconds": round(fit_s, 1)})
                ref_pp = f"{max(init_pps):%Y-%m-%d}"   # S4 PSI reference
                last_train_t = FIRST_EVAL_PP

                for t in eval_pps:
                    t = pd.Timestamp(t)
                    if t > FIRST_EVAL_PP:
                        # 1) retraining decision at t (before scoring batch t)
                        psi_fired = auc_fired = False
                        if strategy == "S4":
                            psi_max = psi_lookup(dist, scheme, monitored,
                                                 ref_pp, f"{t:%Y-%m-%d}")
                            # own-label AUCs of batches scored since last
                            # training whose labels have closed by t
                            auc_hist = []
                            for prev in eval_pps:
                                prev = pd.Timestamp(prev)
                                if not (last_train_t <= prev < t):
                                    continue
                                if close_idx.loc[prev, label_close_col] > t:
                                    continue
                                saved = pd.read_parquet(os.path.join(
                                    strat_dir,
                                    f"{scheme}_{prev:%Y-%m-%d}.parquet"))
                                auc_hist.append(roc_auc_score(
                                    saved[f"label_{scheme}"], saved["y_prob"]))
                            psi_fired, auc_fired = s4_signals(
                                psi_max, auc_hist, PSI_THRESHOLD,
                                AUC_DECLINE_THRESHOLD)
                        if should_retrain(strategy, psi_fired, auc_fired):
                            pps = training_batches(strategy, t, scheme, close)
                            model, n_rows, fit_s = fit_model(df, pps, scheme,
                                                             cols, params)
                            trigger = ("monthly" if strategy in ("S2", "S3") else
                                       "+".join(s for s, fired in
                                                (("psi", psi_fired),
                                                 ("auc", auc_fired)) if fired))
                            retrain_rows.append({
                                "maturity_mode": mode, "scheme": scheme,
                                "strategy": strategy, "t": f"{t:%Y-%m-%d}",
                                "trigger": trigger, "n_train_rows": n_rows,
                                "n_train_batches": len(pps),
                                "effective_lag_days": int((t - max(pps)).days),
                                "fit_seconds": round(fit_s, 1)})
                            ref_pp = f"{max(pps):%Y-%m-%d}"
                            last_train_t = t

                    # 2) score batch t with the current model
                    batch = df[df["prediction_point"] == t]
                    out = batch[["server_id", "shop_id", "company_id",
                                 "prediction_point", "label_L1", "label_L2"]].copy()
                    out["y_prob"] = model.predict_proba(batch[cols])[:, 1]
                    out.to_parquet(os.path.join(
                        strat_dir, f"{scheme}_{t:%Y-%m-%d}.parquet"),
                        index=False)

                n_retrains = sum(
                    1 for r in retrain_rows
                    if r["maturity_mode"] == mode and r["scheme"] == scheme
                    and r["strategy"] == strategy and r["trigger"] != "initial")
                w(f"  {mode} {scheme} {strategy}: initial fit + {n_retrains} retrains")
                print(summary[-1])

    pd.DataFrame(retrain_rows).to_csv(RETRAIN_LOG_PATH, index=False)
    w("")
    w(f"Predictions: "
      f"{len(MATURITY_MODES) * len(SCHEMES) * len(STRATEGIES) * len(eval_pps)} "
      f"files under {os.path.basename(PREDICTIONS_DIR)}/<mode>/<strategy>/")
    w(f"Retrain log: {os.path.basename(RETRAIN_LOG_PATH)} "
      f"({len(retrain_rows)} training events incl. initial fits)")
    w("=" * 70)
    report = "\n".join(summary)
    with open(os.path.join(EXPERIMENT_DIR, "step8_3_run_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
