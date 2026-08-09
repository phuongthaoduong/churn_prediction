"""
Step 2.1: Remove Extreme Financial Outliers
============================================
Identifies and removes transactions with implausible financial amounts.

Problem: Profiling found max bill_amt and item amt of ~10 trillion
(9,999,989,000,001) — certainly data entry errors.

Action:
  - Pass 1: Analyze bill_amt distribution above P99 to find natural break
  - Pass 2: Remove (not cap) srv_sales records where |bill_amt| > threshold
  - Pass 3: Remove srv_saleitems records where |amt| > threshold
  - Log all removed records with original values

This runs FIRST in the cleaning sequence, before Step 2.2.

Output:
  - srv_sales_step2_1.parquet:     sales with outliers removed
  - srv_saleitems_step2_1.parquet: line items with outliers removed
  - exclusion_log_step2_1.csv:     every removed record with source table + reason
  - step2_1_outlier_summary.txt:   distribution analysis and threshold justification
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sale_data")
SALES_PATH = os.path.join(DATA_DIR, "srv_sales.csv")
SALEITEMS_PATH = os.path.join(DATA_DIR, "srv_saleitems.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
CHUNK_SIZE = 500_000

# Upper bound: any record with |bill_amt| or |amt| above this is removed.
# 10M is proposed in the cleaning plan — no legitimate SME retail transaction
# reaches this scale. Wholesale orders can be large but not tens of millions.
OUTLIER_THRESHOLD = 10_000_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column dtypes for consistent reading
SALES_DTYPES = {
    "server_id": str,
    "shop_id": "int32",
    "id": "int64",
    "code": str,
    "company_id": "int64",
    "company_type": str,
    "acct_id": "int64",
    "acct_name": str,
}

SALEITEMS_DTYPES = {
    "server_id": str,
    "shop_id": "int32",
    "id": "int64",
    "sale_id": "int64",
    "product_id": "int64",
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# PASS 1: Analyze bill_amt distribution to justify threshold
# ===================================================================
def analyze_distribution():
    """Collect bill_amt values above P99 and compute distribution stats."""
    log("PASS 1: Analyzing bill_amt distribution...")

    all_bill_amts = []

    for chunk_idx, chunk in enumerate(pd.read_csv(
        SALES_PATH,
        chunksize=CHUNK_SIZE,
        usecols=["bill_amt"],
        low_memory=False,
    )):
        all_bill_amts.append(chunk["bill_amt"].values)
        if chunk_idx % 10 == 0:
            log(f"  Chunk {chunk_idx}...")

    bill_amts = np.concatenate(all_bill_amts)
    total_rows = len(bill_amts)
    log(f"  Total rows: {total_rows:,}")

    abs_amts = np.abs(bill_amts)

    percentiles = {
        "P50": 50, "P90": 90, "P95": 95, "P99": 99,
        "P99.9": 99.9, "P99.99": 99.99, "P99.999": 99.999,
    }
    pct_values = {}
    for label, p in percentiles.items():
        pct_values[label] = np.percentile(abs_amts, p)

    # Records that would be removed at various thresholds
    thresholds_to_test = [1_000_000, 5_000_000, 10_000_000, 50_000_000, 100_000_000]
    threshold_impact = {}
    for t in thresholds_to_test:
        count = int(np.sum(abs_amts > t))
        threshold_impact[t] = {
            "count": count,
            "pct": count / total_rows * 100,
        }

    # Distribution above P99 for natural break analysis
    p99_val = pct_values["P99"]
    above_p99 = abs_amts[abs_amts > p99_val]
    above_p99_stats = {
        "count": len(above_p99),
        "min": float(np.min(above_p99)),
        "median": float(np.median(above_p99)),
        "mean": float(np.mean(above_p99)),
        "P75": float(np.percentile(above_p99, 75)),
        "P90": float(np.percentile(above_p99, 90)),
        "P95": float(np.percentile(above_p99, 95)),
        "P99": float(np.percentile(above_p99, 99)),
        "max": float(np.max(above_p99)),
    }

    stats = {
        "total_rows": total_rows,
        "percentiles": pct_values,
        "threshold_impact": threshold_impact,
        "above_p99_stats": above_p99_stats,
    }

    log(f"  |bill_amt| percentiles:")
    for label, val in pct_values.items():
        log(f"    {label}: {val:,.2f}")
    log(f"  Records above proposed threshold ({OUTLIER_THRESHOLD:,}):"
        f" {threshold_impact[OUTLIER_THRESHOLD]['count']:,}"
        f" ({threshold_impact[OUTLIER_THRESHOLD]['pct']:.4f}%)")

    return stats


# ===================================================================
# PASS 2: Clean srv_sales — remove outliers, write Parquet
# ===================================================================
def clean_sales():
    """Remove rows with |bill_amt| > OUTLIER_THRESHOLD from srv_sales."""
    log(f"PASS 2: Cleaning srv_sales (threshold = {OUTLIER_THRESHOLD:,})...")

    out_path = os.path.join(OUTPUT_DIR, "srv_sales_step2_1.parquet")
    excl_path = os.path.join(OUTPUT_DIR, "exclusion_log_step2_1.csv")

    writer = None
    excl_header_written = False
    total_rows = 0
    kept_rows = 0
    removed_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        SALES_PATH,
        chunksize=CHUNK_SIZE,
        dtype=SALES_DTYPES,
        parse_dates=["opt_on", "company_create"],
        low_memory=False,
    )):
        total_rows += len(chunk)
        if chunk_idx % 5 == 0:
            log(f"  Chunk {chunk_idx}, rows so far: {total_rows:,}")

        mask_outlier = chunk["bill_amt"].abs() > OUTLIER_THRESHOLD
        n_outlier = int(mask_outlier.sum())

        # Write outlier records to exclusion log
        if n_outlier > 0:
            excl = chunk[mask_outlier].copy()
            excl["source_table"] = "srv_sales"
            excl["exclusion_step"] = "2.1"
            excl["exclusion_reason"] = "extreme_financial_outlier"
            excl["threshold_used"] = OUTLIER_THRESHOLD
            excl.to_csv(
                excl_path,
                mode="a" if excl_header_written else "w",
                header=not excl_header_written,
                index=False,
            )
            excl_header_written = True

        # Write kept records to Parquet
        chunk_kept = chunk[~mask_outlier]
        kept_rows += len(chunk_kept)
        removed_rows += n_outlier

        table = pa.Table.from_pandas(chunk_kept, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"  srv_sales: {total_rows:,} total, {kept_rows:,} kept, {removed_rows:,} removed")
    log(f"  Saved to {out_path}")

    return {
        "total": total_rows,
        "kept": kept_rows,
        "removed": removed_rows,
    }


# ===================================================================
# PASS 3: Clean srv_saleitems — remove outliers, write Parquet
# ===================================================================
def clean_saleitems():
    """Remove rows with |amt| > OUTLIER_THRESHOLD from srv_saleitems."""
    log(f"PASS 3: Cleaning srv_saleitems (threshold = {OUTLIER_THRESHOLD:,})...")

    out_path = os.path.join(OUTPUT_DIR, "srv_saleitems_step2_1.parquet")
    excl_path = os.path.join(OUTPUT_DIR, "exclusion_log_step2_1.csv")

    writer = None
    # Exclusion log already has header from pass 2 (if any sales outliers existed)
    excl_header_written = os.path.exists(excl_path)
    total_rows = 0
    kept_rows = 0
    removed_rows = 0

    for chunk_idx, chunk in enumerate(pd.read_csv(
        SALEITEMS_PATH,
        chunksize=CHUNK_SIZE,
        dtype=SALEITEMS_DTYPES,
        parse_dates=["opt_on"],
        low_memory=False,
    )):
        total_rows += len(chunk)
        if chunk_idx % 10 == 0:
            log(f"  Chunk {chunk_idx}, rows so far: {total_rows:,}")

        mask_outlier = chunk["amt"].abs() > OUTLIER_THRESHOLD
        n_outlier = int(mask_outlier.sum())

        if n_outlier > 0:
            excl = chunk[mask_outlier].copy()
            excl["source_table"] = "srv_saleitems"
            excl["exclusion_step"] = "2.1"
            excl["exclusion_reason"] = "extreme_financial_outlier"
            excl["threshold_used"] = OUTLIER_THRESHOLD
            excl.to_csv(
                excl_path,
                mode="a",
                header=not excl_header_written,
                index=False,
            )
            excl_header_written = True

        chunk_kept = chunk[~mask_outlier]
        kept_rows += len(chunk_kept)
        removed_rows += n_outlier

        table = pa.Table.from_pandas(chunk_kept, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"  srv_saleitems: {total_rows:,} total, {kept_rows:,} kept, {removed_rows:,} removed")
    log(f"  Saved to {out_path}")

    return {
        "total": total_rows,
        "kept": kept_rows,
        "removed": removed_rows,
    }


# ===================================================================
# Summary report
# ===================================================================
def write_summary(dist_stats, sales_stats, items_stats):
    """Write human-readable summary with distribution analysis."""
    txt_path = os.path.join(OUTPUT_DIR, "step2_1_outlier_summary.txt")

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 2.1: EXTREME FINANCIAL OUTLIER REMOVAL\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Threshold: |amount| > {OUTLIER_THRESHOLD:,}\n")
        f.write("=" * 70 + "\n\n")

        # Distribution analysis
        f.write("BILL_AMT DISTRIBUTION ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total srv_sales rows: {dist_stats['total_rows']:,}\n\n")

        f.write("  |bill_amt| percentiles:\n")
        for label, val in dist_stats["percentiles"].items():
            f.write(f"    {label:>10s}: {val:>20,.2f}\n")

        f.write("\n  Distribution above P99:\n")
        for label, val in dist_stats["above_p99_stats"].items():
            if isinstance(val, int):
                f.write(f"    {label:>8s}: {val:>20,}\n")
            else:
                f.write(f"    {label:>8s}: {val:>20,.2f}\n")

        f.write("\n  Removal impact at various thresholds:\n")
        f.write(f"    {'Threshold':>15s}  {'Records':>10s}  {'% of total':>10s}\n")
        f.write(f"    {'─' * 15}  {'─' * 10}  {'─' * 10}\n")
        for t, info in dist_stats["threshold_impact"].items():
            marker = " <-- USED" if t == OUTLIER_THRESHOLD else ""
            f.write(f"    {t:>15,}  {info['count']:>10,}  {info['pct']:>9.4f}%{marker}\n")

        # Removal results
        f.write(f"\n\n{'=' * 70}\n")
        f.write("REMOVAL RESULTS\n")
        f.write("-" * 50 + "\n\n")

        f.write("  srv_sales:\n")
        f.write(f"    Total rows:   {sales_stats['total']:>15,}\n")
        f.write(f"    Kept:         {sales_stats['kept']:>15,}\n")
        f.write(f"    Removed:      {sales_stats['removed']:>15,}"
                f"  ({sales_stats['removed'] / sales_stats['total'] * 100:.4f}%)\n")

        f.write("\n  srv_saleitems:\n")
        f.write(f"    Total rows:   {items_stats['total']:>15,}\n")
        f.write(f"    Kept:         {items_stats['kept']:>15,}\n")
        f.write(f"    Removed:      {items_stats['removed']:>15,}"
                f"  ({items_stats['removed'] / items_stats['total'] * 100:.4f}%)\n")

        total_removed = sales_stats["removed"] + items_stats["removed"]
        total_all = sales_stats["total"] + items_stats["total"]
        f.write(f"\n  Combined:\n")
        f.write(f"    Total removed: {total_removed:>14,} / {total_all:,}"
                f"  ({total_removed / total_all * 100:.4f}%)\n")

        # Output files
        f.write(f"\n\n{'=' * 70}\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        f.write(f"  srv_sales_step2_1.parquet      — cleaned sales\n")
        f.write(f"  srv_saleitems_step2_1.parquet   — cleaned line items\n")
        f.write(f"  exclusion_log_step2_1.csv       — all removed records\n")
        f.write(f"  step2_1_outlier_summary.txt     — this file\n")

        f.write(f"\n\nNOTE: Step 2.2 (test account detection) should read from\n")
        f.write(f"srv_sales_step2_1.parquet, not the raw srv_sales.csv.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {txt_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 2.1: Remove Extreme Financial Outliers — START")
    log(f"Threshold: |amount| > {OUTLIER_THRESHOLD:,}")

    dist_stats = analyze_distribution()
    sales_stats = clean_sales()
    items_stats = clean_saleitems()
    write_summary(dist_stats, sales_stats, items_stats)

    log(f"Removed {sales_stats['removed']:,} sales"
        f" + {items_stats['removed']:,} line items")
    log("Step 2.1: Remove Extreme Financial Outliers — COMPLETE")


if __name__ == "__main__":
    main()
