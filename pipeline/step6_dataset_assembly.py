"""
Step 6: Dataset Assembly and Validation
=========================================
Merges features (Step 4) and labels (Step 5) into the final
experiment-ready dataset. Runs validation checks for leakage,
completeness, and consistency.

Input:
  - feature_matrix.parquet   (Step 4: 6.3M rows × 29 cols)
  - label_table.parquet      (Step 5: 6.3M rows × labels)

Output:
  - dataset_final.parquet    (merged features + labels)
  - class_balance.csv        (per-batch churn rates for scale_pos_weight)
  - step6_validation_report.txt
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
FEATURE_PATH = os.path.join(OUTPUT_DIR, "feature_matrix.parquet")
LABEL_PATH = os.path.join(OUTPUT_DIR, "label_table.parquet")

DATASET_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_final.parquet")
CLASS_BALANCE_PATH = os.path.join(OUTPUT_DIR, "class_balance.csv")
REPORT_PATH = os.path.join(OUTPUT_DIR, "step6_validation_report.txt")

JOIN_KEYS = ["server_id", "shop_id", "company_id", "prediction_point"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# Phase 1: Merge
# ===================================================================
def merge_features_labels():
    log("Phase 1: Loading and merging features + labels...")

    features = pd.read_parquet(FEATURE_PATH)
    labels = pd.read_parquet(LABEL_PATH)

    log(f"  Features: {len(features):,} rows × {len(features.columns)} cols")
    log(f"  Labels:   {len(labels):,} rows × {len(labels.columns)} cols")

    # Inner join on composite key
    dataset = features.merge(labels, on=JOIN_KEYS, how="inner")

    n_features_only = len(features) - len(dataset)
    n_labels_only = len(labels) - len(dataset)
    log(f"  Merged:   {len(dataset):,} rows")
    if n_features_only > 0:
        log(f"  WARNING: {n_features_only:,} feature rows had no matching label")
    if n_labels_only > 0:
        log(f"  WARNING: {n_labels_only:,} label rows had no matching features")

    del features, labels
    return dataset


# ===================================================================
# Phase 2: Compute class balance
# ===================================================================
def compute_class_balance(dataset):
    log("Phase 2: Computing class balance per batch...")

    rows = []
    for pp in sorted(dataset["prediction_point"].unique()):
        batch = dataset[dataset["prediction_point"] == pp]
        n = len(batch)
        for label_col, scheme in [("label_L1", "L1"), ("label_L2", "L2")]:
            n_churn = int(batch[label_col].sum())
            n_active = n - n_churn
            churn_rate = n_churn / n if n > 0 else 0
            spw = n_active / n_churn if n_churn > 0 else 1.0
            rows.append({
                "prediction_point": pp,
                "scheme": scheme,
                "n_total": n,
                "n_churn": n_churn,
                "n_active": n_active,
                "churn_rate": round(churn_rate, 4),
                "scale_pos_weight": round(spw, 4),
            })

    balance_df = pd.DataFrame(rows)
    balance_df.to_csv(CLASS_BALANCE_PATH, index=False)
    log(f"  Saved to {CLASS_BALANCE_PATH}")
    return balance_df


# ===================================================================
# Phase 3: Validation checks
# ===================================================================
def run_validations(dataset):
    """Run all validation checks, return list of (check_name, passed, detail)."""
    log("Phase 3: Running validation checks...")
    results = []

    feature_cols = [
        "recency", "frequency", "monetary", "avg_monetary",
        "ipi_mean", "ipi_std", "recent_activity", "freq_decline_ratio",
        "unique_products", "avg_basket_size", "total_qty",
        "pay_ratio", "debt_frequency", "customer_tenure",
        "store_txn_volume", "store_active_customers", "store_median_ipi",
        "recency_vs_store", "frequency_vs_store", "monetary_vs_store",
        "month", "quarter", "is_cny_season", "is_double11", "is_spring_festival",
    ]

    # --- Check 1: Row count consistency ---
    n = len(dataset)
    passed = n > 0
    results.append(("Row count > 0", passed, f"{n:,} rows"))

    # --- Check 2: No duplicate keys ---
    n_dupes = dataset.duplicated(subset=JOIN_KEYS).sum()
    passed = n_dupes == 0
    results.append(("No duplicate keys", passed,
                     f"{n_dupes:,} duplicates" if n_dupes > 0 else "OK"))

    # --- Check 3: Recency >= 0 (no future leakage) ---
    bad_recency = (dataset["recency"] < 0).sum()
    passed = bad_recency == 0
    results.append(("Recency >= 0 (no leakage)", passed,
                     f"{bad_recency:,} violations" if bad_recency > 0 else "OK"))

    # --- Check 4: Frequency >= 1 (eligible customers must have purchases) ---
    bad_freq = (dataset["frequency"] < 1).sum()
    passed = bad_freq == 0
    results.append(("Frequency >= 1", passed,
                     f"{bad_freq:,} violations" if bad_freq > 0 else "OK"))

    # --- Check 5: Labels are binary (0 or 1) ---
    l1_valid = dataset["label_L1"].isin([0, 1]).all()
    l2_valid = dataset["label_L2"].isin([0, 1]).all()
    passed = l1_valid and l2_valid
    results.append(("Labels are binary {0,1}", passed,
                     "OK" if passed else "Invalid label values found"))

    # --- Check 6: No missing labels ---
    l1_na = dataset["label_L1"].isna().sum()
    l2_na = dataset["label_L2"].isna().sum()
    passed = l1_na == 0 and l2_na == 0
    results.append(("No missing labels", passed,
                     f"L1: {l1_na}, L2: {l2_na}" if not passed else "OK"))

    # --- Check 7: Feature completeness ---
    # Expected NaN: ipi_mean (single-purchase customers), ipi_std (<=2 dates),
    # store_median_ipi (stores with no IPI data), customer_tenure (missing company_create)
    expected_nan = {"ipi_mean", "ipi_std", "store_median_ipi", "customer_tenure"}
    unexpected_nan_cols = []
    for col in feature_cols:
        na_count = dataset[col].isna().sum()
        if na_count > 0 and col not in expected_nan:
            unexpected_nan_cols.append((col, na_count))
    passed = len(unexpected_nan_cols) == 0
    detail = "OK" if passed else "; ".join(
        f"{c}: {cnt:,}" for c, cnt in unexpected_nan_cols
    )
    results.append(("No unexpected NaN features", passed, detail))

    # --- Check 8: L1 and L2 labels differ meaningfully ---
    disagree = (dataset["label_L1"] != dataset["label_L2"]).mean() * 100
    passed = disagree >= 5.0  # at least 5% disagreement
    results.append(("L1 vs L2 disagree >= 5%", passed,
                     f"{disagree:.1f}% disagree"))

    # --- Check 9: Recency within obs window bounds ---
    bad_recency_high = (dataset["recency"] > 90).sum()
    passed = bad_recency_high == 0
    results.append(("Recency <= 90 (obs window)", passed,
                     f"{bad_recency_high:,} violations" if bad_recency_high > 0 else "OK"))

    # --- Check 10: Every batch has data ---
    n_batches = dataset["prediction_point"].nunique()
    passed = n_batches == 20
    results.append(("All 20 batches present", passed,
                     f"{n_batches} batches found"))

    # --- Check 11: Leakage spot-check ---
    # Recency should be >= 1 (last purchase at least 1 day before prediction point)
    # Actually recency can be 0 if purchase on obs_end-1 day... no.
    # recency = pp - last_purchase. If last_purchase is on obs_end - 1 day (= pp - 1),
    # recency = 1. If last_purchase == obs_start, recency = 90.
    # recency = 0 would mean last_purchase == pp, but obs window is [start, pp),
    # so pp is excluded. Minimum recency should be 1.
    zero_recency = (dataset["recency"] == 0).sum()
    passed = zero_recency == 0
    results.append(("Recency > 0 (strict leakage)", passed,
                     f"{zero_recency:,} have recency=0" if zero_recency > 0 else "OK"))

    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        log(f"  [{status}] {name}: {detail}")

    return results


# ===================================================================
# Phase 4: Per-batch feature distribution summary (drift detection)
# ===================================================================
def compute_batch_distributions(dataset):
    """Compute per-batch mean/std for key features to detect drift."""
    key_features = [
        "recency", "frequency", "monetary", "avg_monetary",
        "ipi_mean", "pay_ratio", "debt_frequency",
        "store_txn_volume", "store_active_customers",
    ]
    rows = []
    for pp in sorted(dataset["prediction_point"].unique()):
        batch = dataset[dataset["prediction_point"] == pp]
        row = {"prediction_point": pp, "n": len(batch)}
        for feat in key_features:
            vals = batch[feat].dropna()
            row[f"{feat}_mean"] = vals.mean()
            row[f"{feat}_std"] = vals.std()
        rows.append(row)
    return pd.DataFrame(rows)


# ===================================================================
# Summary report
# ===================================================================
def write_report(dataset, balance_df, validations, dist_df):
    with open(REPORT_PATH, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 6: DATASET ASSEMBLY AND VALIDATION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # --- Dataset overview ---
        f.write("DATASET OVERVIEW\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total rows:           {len(dataset):,}\n")
        f.write(f"  Total columns:        {len(dataset.columns)}\n")
        f.write(f"  Prediction points:    {dataset['prediction_point'].nunique()}\n")
        n_cust = dataset.groupby(["server_id", "shop_id", "company_id"]).ngroups
        f.write(f"  Unique customers:     {n_cust:,}\n")
        n_stores = dataset.groupby(["server_id", "shop_id"]).ngroups
        f.write(f"  Unique stores:        {n_stores:,}\n")
        n_cats = dataset["category"].nunique()
        f.write(f"  Industry categories:  {n_cats}\n")
        fsize = os.path.getsize(DATASET_OUTPUT_PATH)
        f.write(f"  File size:            {fsize / 1e6:.1f} MB\n\n")

        # --- Column listing ---
        f.write("COLUMNS\n")
        f.write("-" * 50 + "\n")
        for i, col in enumerate(dataset.columns, 1):
            dtype = dataset[col].dtype
            na_pct = dataset[col].isna().mean() * 100
            na_str = f"  ({na_pct:.1f}% NaN)" if na_pct > 0.01 else ""
            f.write(f"  {i:>2d}. {col:<25s}  {str(dtype):<15s}{na_str}\n")
        f.write("\n")

        # --- Validation results ---
        f.write("VALIDATION CHECKS\n")
        f.write("-" * 70 + "\n")
        all_passed = True
        for name, passed, detail in validations:
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_passed = False
            f.write(f"  [{status}] {name}: {detail}\n")
        f.write(f"\n  Overall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}\n\n")

        # --- Class balance ---
        f.write("CLASS BALANCE (for scale_pos_weight)\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Prediction':>12s}  {'Scheme':>6s}  {'Total':>10s}"
                f"  {'Churn':>8s}  {'Active':>8s}"
                f"  {'Rate':>7s}  {'SPW':>7s}\n")
        f.write(f"  {'─' * 12}  {'─' * 6}  {'─' * 10}"
                f"  {'─' * 8}  {'─' * 8}"
                f"  {'─' * 7}  {'─' * 7}\n")

        for _, row in balance_df.iterrows():
            f.write(f"  {pd.Timestamp(row['prediction_point']).date()!s:>12s}"
                    f"  {row['scheme']:>6s}"
                    f"  {int(row['n_total']):>10,}"
                    f"  {int(row['n_churn']):>8,}"
                    f"  {int(row['n_active']):>8,}"
                    f"  {row['churn_rate']:>6.1%}"
                    f"  {row['scale_pos_weight']:>7.2f}\n")

        # --- Feature distribution stability ---
        f.write(f"\n\nFEATURE STABILITY ACROSS BATCHES (mean values)\n")
        f.write("-" * 70 + "\n")
        key_features = ["recency", "frequency", "monetary", "pay_ratio",
                        "debt_frequency"]
        f.write(f"  {'Prediction':>12s}")
        for feat in key_features:
            f.write(f"  {feat:>12s}")
        f.write("\n")
        f.write(f"  {'─' * 12}")
        for _ in key_features:
            f.write(f"  {'─' * 12}")
        f.write("\n")

        for _, row in dist_df.iterrows():
            f.write(f"  {pd.Timestamp(row['prediction_point']).date()!s:>12s}")
            for feat in key_features:
                f.write(f"  {row[f'{feat}_mean']:>12.1f}")
            f.write("\n")

        # Compute CV across batches for each feature
        f.write(f"\n  {'CV%':>12s}")
        for feat in key_features:
            means = dist_df[f"{feat}_mean"]
            cv = means.std() / means.mean() * 100 if means.mean() != 0 else 0
            f.write(f"  {cv:>11.1f}%")
        f.write("\n")

        # Output files
        f.write(f"\n\nOUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        f.write(f"  dataset_final.parquet     — {len(dataset):,} rows,"
                f" {fsize / 1e6:.1f} MB\n")
        f.write(f"  class_balance.csv         — per-batch churn rates + SPW\n")
        f.write(f"  step6_validation_report.txt — this file\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Report saved to {REPORT_PATH}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 6: Dataset Assembly and Validation — START")

    # Phase 1: Merge
    dataset = merge_features_labels()

    # Reorder columns: keys, features, labels/metadata
    feature_cols = [
        "recency", "frequency", "monetary", "avg_monetary",
        "ipi_mean", "ipi_std", "recent_activity", "freq_decline_ratio",
        "unique_products", "avg_basket_size", "total_qty",
        "pay_ratio", "debt_frequency", "customer_tenure",
        "store_txn_volume", "store_active_customers", "store_median_ipi",
        "recency_vs_store", "frequency_vs_store", "monetary_vs_store",
        "month", "quarter", "is_cny_season", "is_double11", "is_spring_festival",
    ]
    label_cols = ["label_L1", "label_L2", "L2_tier", "L2_threshold_days", "category"]
    col_order = JOIN_KEYS + feature_cols + label_cols
    dataset = dataset[col_order]

    # Phase 2: Class balance
    balance_df = compute_class_balance(dataset)

    # Phase 3: Validation
    validations = run_validations(dataset)

    # Phase 4: Feature distributions
    dist_df = compute_batch_distributions(dataset)

    # Save final dataset
    log("Saving final dataset...")
    dataset.to_parquet(DATASET_OUTPUT_PATH, index=False)
    log(f"  {len(dataset):,} rows × {len(dataset.columns)} cols →"
        f" {os.path.getsize(DATASET_OUTPUT_PATH) / 1e6:.1f} MB")

    # Report
    write_report(dataset, balance_df, validations, dist_df)

    log("Step 6: Dataset Assembly and Validation — COMPLETE")


if __name__ == "__main__":
    main()
