"""Step 9: SHAP Interpretation
===============================
Resolves the open question flagged in Step 7 (weekly_report_step7.md):
whether ipi_std's position as the top gain-importance feature (both
schemes) despite 63.5% NaN reflects genuine predictive signal or a
structural artifact of XGBoost's NaN-routing. Gain importance (used in
step7_2 for drift-feature selection) measures how often/effectively a
feature is used for splits; SHAP measures actual marginal contribution to
each prediction, which can diverge from gain when NaN-routing lets a
feature "win" splits without carrying much per-row information.

Fits one canonical model per scheme (same tuned hyperparameters and frozen
feature set as the main factorial; trained on ALL matured batches as of
the final evaluation point 2026-03-01 — i.e. the same training data S2's
last retrain used, the best-performing strategy in the finished study).
This is "the frozen baseline model" referred to elsewhere; it is a fresh
fit for SHAP purposes, not a reuse of any strategy's in-memory model from
Step 8.3 (none were persisted to disk).

SHAP values are computed via TreeExplainer on a fixed random sample of the
final evaluation batch (not the full ~316k rows, for tractability — see
SAMPLE_N below), which also serves as the pool of examples for the
downstream LLM explanation-faithfulness trial.

Input:  dataset_final.parquet, tuned_params.json, frozen_feature_set.json,
        train_eval_split metadata, l2_thresholds.csv
Output: experiment_output/shap_values_<scheme>.parquet (row-level SHAP
        matrix + feature values, for the LLM trial to consume),
        experiment_output/shap_summary.csv (mean |SHAP| per feature per
        scheme, with gain-importance and NaN-rate for comparison),
        experiment_output/step9_shap_summary.txt
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import (
    BATCH_METADATA, DATASET_FINAL, EXPERIMENT_DIR, FROZEN_FEATURES_PATH,
    L2_THRESHOLDS_PATH, TUNED_PARAMS_PATH, add_ipi_cv, label_close_dates,
    usable_batches,
)
from run_experiment import fit_model

SEED = 42
SAMPLE_N = 15_000
FINAL_PP = pd.Timestamp("2026-03-01")
GAIN_IMPORTANCE_PATH = os.path.join(
    os.path.dirname(EXPERIMENT_DIR), "profiling_output",
    "step7_split_drift", "feature_importance.csv")


def main():
    df = add_ipi_cv(pd.read_parquet(DATASET_FINAL))
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)
    close = label_close_dates(batch_meta, l2_thresholds)
    tuned = json.load(open(TUNED_PARAMS_PATH))
    frozen = json.load(open(FROZEN_FEATURES_PATH))
    gain_imp = (pd.read_csv(GAIN_IMPORTANCE_PATH)
                if os.path.exists(GAIN_IMPORTANCE_PATH) else None)

    summary_lines = ["=" * 70, "STEP 9: SHAP INTERPRETATION",
                     f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70]
    w = summary_lines.append
    summary_rows = []

    for scheme in ("L1", "L2"):
        cols = frozen[scheme]["feature_cols"]
        params = tuned[scheme]["params"]
        train_pps = usable_batches(FINAL_PP, scheme, close)
        model, n_rows, fit_s = fit_model(df, train_pps, scheme, cols, params)
        w(f"\n{scheme}: fit on {n_rows} rows from {len(train_pps)} matured "
          f"batches (same training set S2's final retrain used), "
          f"{fit_s:.1f}s")

        final_batch = df[df["prediction_point"] == FINAL_PP]
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(final_batch), size=min(SAMPLE_N, len(final_batch)),
                         replace=False)
        sample = final_batch.iloc[idx].reset_index(drop=True)

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(sample[cols])
        w(f"  SHAP computed on {len(sample)} rows (final evaluation batch "
          f"{FINAL_PP:%Y-%m-%d}, seed {SEED})")

        mean_abs_shap = np.abs(sv).mean(axis=0)
        nan_rate = sample[cols].isna().mean()
        shap_df = pd.DataFrame({
            "feature": cols, "mean_abs_shap": mean_abs_shap,
            "nan_rate": nan_rate.values,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        shap_df["shap_rank"] = shap_df.index + 1
        if gain_imp is not None:
            g = gain_imp[gain_imp.scheme == scheme][["feature", "gain", "rank"]]
            g = g.rename(columns={"rank": "gain_rank"})
            shap_df = shap_df.merge(g, on="feature", how="left")
        shap_df["scheme"] = scheme
        summary_rows.append(shap_df)

        w(f"  Top 5 by mean |SHAP|: "
          + ", ".join(f"{r.feature} ({r.mean_abs_shap:.4f})"
                      for r in shap_df.head(5).itertuples()))

        ipi_row = shap_df[shap_df.feature == "ipi_std"].iloc[0]
        w(f"  ipi_std: SHAP rank {int(ipi_row.shap_rank)}/{len(cols)} "
          f"(mean |SHAP| = {ipi_row.mean_abs_shap:.4f}), "
          f"gain rank {int(ipi_row.gain_rank) if pd.notna(ipi_row.get('gain_rank')) else 'n/a'}, "
          f"NaN rate {ipi_row.nan_rate:.1%}")

        # Save row-level SHAP matrix + raw feature values for the LLM trial
        keys = sample[["server_id", "shop_id", "company_id", "prediction_point",
                       f"label_{scheme}"]].copy()
        keys["y_prob"] = model.predict_proba(sample[cols])[:, 1]
        sv_df = pd.DataFrame(sv, columns=[f"shap_{c}" for c in cols])
        val_df = sample[cols].add_suffix("_value")
        out = pd.concat([keys, sv_df, val_df], axis=1)
        out["base_value"] = explainer.expected_value
        out.to_parquet(os.path.join(EXPERIMENT_DIR, f"shap_values_{scheme}.parquet"),
                       index=False)

    full_summary = pd.concat(summary_rows, ignore_index=True)
    full_summary.to_csv(os.path.join(EXPERIMENT_DIR, "shap_summary.csv"), index=False)

    w("\n" + "-" * 70)
    w("RESOLUTION: ipi_std gain-vs-SHAP comparison (the open question from "
      "Step 7)")
    nan_split_rows = []
    for scheme in ("L1", "L2"):
        row = full_summary[(full_summary.scheme == scheme) &
                           (full_summary.feature == "ipi_std")].iloc[0]
        sv = pd.read_parquet(os.path.join(EXPERIMENT_DIR, f"shap_values_{scheme}.parquet"))
        is_nan = sv["ipi_std_value"].isna()
        shap_abs = sv["shap_ipi_std"].abs()
        mean_nan, mean_present = shap_abs[is_nan].mean(), shap_abs[~is_nan].mean()
        share_mass_nan = shap_abs[is_nan].sum() / shap_abs.sum()
        verdict = ("genuine signal from the ACTUAL inter-purchase-interval "
                   "value, not primarily from missingness"
                   if mean_present > mean_nan and share_mass_nan < is_nan.mean()
                   else "signal concentrated in missingness itself, not the "
                   "numeric value — closer to a NaN-routing artifact")
        w(f"  {scheme}: gain_rank={int(row.gain_rank) if pd.notna(row.gain_rank) else 'n/a'}, "
          f"shap_rank={int(row.shap_rank)}, nan_rate={row.nan_rate:.1%}")
        w(f"        mean|SHAP| when present={mean_present:.4f} vs when NaN={mean_nan:.4f} "
          f"(NaN rows are {is_nan.mean():.1%} of data but only "
          f"{share_mass_nan:.1%} of total |SHAP| mass)")
        w(f"        -> {verdict}")
        nan_split_rows.append({"scheme": scheme, "mean_abs_shap_when_present": mean_present,
                               "mean_abs_shap_when_nan": mean_nan,
                               "nan_row_share": is_nan.mean(),
                               "nan_row_shap_mass_share": share_mass_nan})
    pd.DataFrame(nan_split_rows).to_csv(
        os.path.join(EXPERIMENT_DIR, "shap_ipi_std_nan_split.csv"), index=False)
    w("=" * 70)

    report = "\n".join(summary_lines)
    with open(os.path.join(EXPERIMENT_DIR, "step9_shap_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
