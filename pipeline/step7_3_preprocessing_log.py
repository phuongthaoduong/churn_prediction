"""
Step 7.3: Consolidated Preprocessing Decision Log
==================================================
Assembles a single markdown document recording every preprocessing decision:
exclusion criteria and affected counts, window-length choices with profiling
justification, L2 tier proportions and threshold stability, class balance per
batch, the train/evaluation split, and drift-monitoring configuration.

This is the reproducibility record required by Data_Preprocessing_Plan.md
Step 7.3 and the thesis methodology chapter.

Input: all profiling_output artifacts from steps 2-7.2
Output: docs/preprocessing_log.md
"""

import json
import os
import re
from datetime import datetime

import pandas as pd

from experiment_config import (
    CLASS_BALANCE,
    DATASET_FINAL,
    DOCS_DIR,
    FIRST_EVAL_PP,
    L2_THRESHOLDS_PATH,
    OUTPUT_DIR,
)

def count_rows(path):
    """Row count of a possibly large CSV without loading it into memory."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


def parse_summary_counts(path, pattern):
    """Extract one or more integers from a profiling summary text file.

    Raises ValueError naming the file if the pattern does not match —
    exclusion counts must never be silently guessed or defaulted.
    Returns a tuple of ints, one per capture group.
    """
    with open(path) as f:
        text = f.read()
    m = re.search(pattern, text)
    if not m:
        raise ValueError(
            f"Could not parse expected count from {path!r} using pattern "
            f"{pattern!r}"
        )
    return tuple(int(g.replace(",", "")) for g in m.groups())


def md_table(df, float_fmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def main():
    lines = []
    w = lines.append

    w("# Preprocessing Decision Log")
    w("")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S} by "
      "`pipeline/step7_3_preprocessing_log.py`. Regenerate after any "
      "pipeline rerun — do not edit by hand.")
    w("")

    # --- 1. Data scope ----------------------------------------------------
    w("## 1. Data scope")
    w("")
    w("- Raw inputs: `srv_sales.csv` (2.1 GB), `srv_saleitems.csv` (5.1 GB), "
      "`industry_servereid.csv` (13 industry categories, 100% merchant coverage).")
    w("- Span: 2024-05-01 to 2026-05-31 (~25 months).")
    w("- Customer identity: composite key `(server_id, shop_id, company_id)` — "
      "ids are per-merchant auto-increments and are NOT globally unique.")
    w("")

    # --- 2. Exclusions ----------------------------------------------------
    w("## 2. Exclusion criteria and affected record counts (Step 2)")
    w("")
    n_21 = count_rows(os.path.join(OUTPUT_DIR, "exclusion_log_step2_1.csv"))
    n_23 = count_rows(os.path.join(OUTPUT_DIR, "exclusion_log_step2_3.csv"))
    n_accounts_22, n_txn_22 = parse_summary_counts(
        os.path.join(OUTPUT_DIR, "test_account_detection_summary.txt"),
        r"REMOVE:\s*([\d,]+) customers,\s*([\d,]+) transactions",
    )
    (n_24,) = parse_summary_counts(
        os.path.join(OUTPUT_DIR, "step2_4_orphan_summary.txt"),
        r"Orphan line items removed:\s*([\d,]+)",
    )
    exclusion_rows = [
        {"Cleaning step": "2.1 Extreme financial outliers",
         "Records excluded": f"{n_21:,}",
         "Log file": "`exclusion_log_step2_1.csv`",
         "Log contents": "complete"},
        {"Cleaning step": "2.2 System/test accounts",
         "Records excluded": f"{n_accounts_22:,} accounts ({n_txn_22:,} transactions)",
         "Log file": "`exclusion_log_step2_2.csv`",
         "Log contents": "one row per removed account"},
        {"Cleaning step": "2.3 Returns and corrections",
         "Records excluded": f"{n_23:,}",
         "Log file": "`exclusion_log_step2_3.csv`",
         "Log contents": "complete"},
        {"Cleaning step": "2.4 Orphan line items",
         "Records excluded": f"{n_24:,}",
         "Log file": "`exclusion_log_step2_4.csv`",
         "Log contents": "10,000-row sample"},
    ]
    w(md_table(pd.DataFrame(exclusion_rows)))
    w("")
    w("Per-step rationale is recorded in the corresponding "
      "`profiling_output/step2_*_summary.txt` / analysis reports. Note: "
      "the 2.2 and 2.4 log files above do not contain one row per excluded "
      "record — 2.2 logs one row per removed *account* (each covering many "
      "transactions) and 2.4 is a fixed-size sample of a larger removal; "
      "true removal counts are parsed from the corresponding summary text "
      "files, not from `count_rows` on those two logs.")
    w("")

    # --- 3. Window choices ------------------------------------------------
    w("## 3. Window-length choices and justification (Steps 3 & 5)")
    w("")
    w("| Parameter | Value | Justification |")
    w("|---|---|---|")
    w("| Observation window | 90 days | Methodology §4.2; covers P75 of "
      "per-customer median IPI (60d) with margin |")
    w("| L1 label window | 30 days | Fixed inactivity threshold per methodology |")
    w("| L2 label window | P95 of pooled category IPIs, capped at 90 days | "
      "Thesis eq. 3.6; cap preserves 12 evaluation batches |")
    w("| Batch cadence | Monthly (1st of month) | Matches SME review cycle |")
    w("| S3 rolling window | 6 months | Covers P95 per-customer median IPI "
      "(~180d, Step 1 profiling); sensitivity-tested in Phase 2 |")
    w("")

    # --- 4. Train/eval split ---------------------------------------------
    w("## 4. Train/evaluation split (Step 7.1)")
    w("")
    split = pd.read_csv(
        os.path.join(OUTPUT_DIR, "train_eval_split.csv"),
        parse_dates=["prediction_point", "l1_label_close", "l2_label_close"],
    )
    for col in ("prediction_point", "l1_label_close", "l2_label_close"):
        split[col] = split[col].dt.strftime("%Y-%m-%d")
    w(md_table(split))
    w("")
    cfg = json.load(open(os.path.join(OUTPUT_DIR, "experiment_config.json")))
    w(f"- First evaluation point: **{cfg['first_eval_pp']}** — "
      f"{cfg['n_batches_initial_train']} initial-training batches, "
      f"{cfg['n_batches_evaluation']} evaluation batches (one year).")
    w(f"- Label-maturity rule: batches usable for the initial L1 model: "
      f"{len(cfg['initial_train_usable_L1'])}; for L2: "
      f"{len(cfg['initial_train_usable_L2'])}. L2 labels close later "
      "(longer windows), so fewer batches are matured — enforced by "
      "`experiment_config.usable_batches` at every retraining event.")
    w("")

    # --- 5. L2 tiers and threshold stability ------------------------------
    w("## 5. L2 fallback tiers and threshold stability (Step 5)")
    w("")
    tiers = pd.read_parquet(DATASET_FINAL, columns=["prediction_point", "L2_tier"])
    tier_tbl = (
        tiers.groupby([tiers["prediction_point"].dt.strftime("%Y-%m-%d"), "L2_tier"])
        .size().unstack(fill_value=0)
    )
    tier_pct = (tier_tbl.div(tier_tbl.sum(axis=1), axis=0) * 100).round(2)
    tier_pct.index.name = "prediction_point"
    w("Tier share per batch (% of customers):")
    w("")
    w(md_table(tier_pct.reset_index()))
    w("")
    tau = pd.read_csv(L2_THRESHOLDS_PATH)
    stab = tau.groupby("category")["tau_c"].agg(["min", "max", "mean", "std"])
    stab["cv_pct"] = (stab["std"] / stab["mean"] * 100).round(1)
    stab = stab.round(1).reset_index()
    w("Category threshold (tau_c, days) stability across batches:")
    w("")
    w(md_table(stab))
    w("")

    # --- 6. Class balance -------------------------------------------------
    w("## 6. Class balance per batch per scheme (Step 6)")
    w("")
    bal = pd.read_csv(CLASS_BALANCE)
    w(md_table(bal))
    w("")
    w("`scale_pos_weight` is recomputed at each retraining from the pooled "
      "training data of that event (thesis eq. 3.8); the per-batch values "
      "above are diagnostics.")
    w("")

    # --- 7. Drift monitoring ----------------------------------------------
    w("## 7. Drift-monitoring configuration (Step 7.2)")
    w("")
    ref = json.load(open(os.path.join(OUTPUT_DIR, "psi_reference.json")))
    w(f"- PSI: {cfg['n_psi_bins']} equal-frequency bins + dedicated "
      f"missing-value bin, epsilon = {cfg['psi_epsilon']}, trigger threshold "
      f"= {cfg['psi_threshold_S4']}.")
    w(f"- AUC-decline trigger threshold = {cfg['auc_decline_threshold_S4']}.")
    w("- Calendar features excluded from monitoring (deterministic drift): "
      f"{', '.join(cfg['monitor_exclude'])}.")
    for s in ("L1", "L2"):
        w(f"- **{s}**: reference batch {ref[s]['reference_pp']}, in-sample AUC "
          f"{ref[s]['train_auc_in_sample']:.4f} (provisional hyperparameters), "
          f"monitored features: {', '.join(ref[s]['monitored_features'])}.")
    w("")

    # --- 8. Deviations ----------------------------------------------------
    w("## 8. Deviations from the original plan")
    w("")
    w("- **L2 tier design**: Data_Preprocessing_Plan.md §5.2 described a "
      "3-tier scheme (customer/store/category). The implemented scheme is the "
      "thesis's 2-tier category + global-fallback design (eq. 3.6–3.7), which "
      "pools IPIs at category level for statistical stability.")
    w("- **Zero bill_amt records retained** in Step 2.3 (only negative-amount "
      "returns removed); see `step2_3_returns_analysis.txt`.")
    w("- **Hyperparameters provisional**: initial models in Step 7.2 use "
      "documented defaults; Phase 2 tunes once via time-series CV on the "
      "initial training period, then rerun `step7_2_drift_baselines.py`.")
    w("")

    out_path = os.path.join(DOCS_DIR, "preprocessing_log.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
