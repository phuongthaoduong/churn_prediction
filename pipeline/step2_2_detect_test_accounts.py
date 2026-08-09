"""
Step 2.2: Detect and Remove System/Test Accounts (Multi-Signal Scoring)
========================================================================
Reads from Step 2.1 output (Parquet) — extreme financial outliers already removed.

Identifies likely system or POS test accounts using 3 signals:
  S1: Extreme transaction count (above P99.9 or natural gap)
  S2: High proportion of trivial-amount transactions (bill_amt 0-1)
      Only fires if customer has >= 5 transactions (avoids false positives)
  S3: Default/placeholder company_create date (1980-01-01)

Classification:
  REMOVE: 2+ signals fired, OR txn_count >= EXTREME_TXN_CEILING (hard override)
  REVIEW: 1 signal fired
  KEEP:   0 signals fired

Output:
  - flagged_test_accounts.csv: all flagged candidates with signal values
  - test_account_detection_summary.txt: human-readable summary
  - exclusion_log_step2_2.csv: removed customer keys + txn counts + reasons
  - srv_sales_step2_2.parquet: sales data with test accounts removed

Reads Parquet in row-group batches to keep memory usage manageable.
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import os
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
SALES_PATH = os.path.join(OUTPUT_DIR, "srv_sales_step2_1.parquet")
BATCH_SIZE = 500_000

# Signal thresholds
DEFAULT_DATE = "1980-01-01"
TRIVIAL_AMT_UPPER = 1.0  # bill_amt in [0, 1] is considered trivial

# S1: txn_count is above P99.9
# S2: trivial_amt_ratio >= 0.5 AND txn_count >= 5 (avoid false positives on low-txn customers)
# S3: has_default_date == 1 (binary)
S2_THRESHOLD = 0.5
S2_MIN_TXN = 5  # minimum transactions for S2 to fire

# Hard override: auto-REMOVE if txn_count >= this ceiling regardless of other signals
EXTREME_TXN_CEILING = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# PASS 1: Compute S1, S2, S3 from srv_sales parquet
# ===================================================================
def compute_sales_signals():
    """Single pass over srv_sales parquet to compute per-customer signals."""
    log("PASS 1: Computing S1/S2/S3 from srv_sales_step2_1.parquet...")

    # Accumulators per customer (server_id, shop_id, company_id)
    cust_txn_count = defaultdict(int)        # S1
    cust_trivial_count = defaultdict(int)     # S2: count of bill_amt in [0, 1]
    cust_default_date = {}                    # S3: whether company_create is default

    pf = pq.ParquetFile(SALES_PATH)
    for batch_idx, batch in enumerate(pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["server_id", "shop_id", "company_id", "bill_amt", "company_create"],
    )):
        chunk = batch.to_pandas()
        if batch_idx % 5 == 0:
            log(f"  Batch {batch_idx}, customers so far: {len(cust_txn_count):,}")

        keys = list(zip(chunk["server_id"], chunk["shop_id"], chunk["company_id"]))
        bill_vals = chunk["bill_amt"].values
        create_vals = chunk["company_create"].astype(str).values

        is_trivial = (bill_vals >= 0) & (bill_vals <= TRIVIAL_AMT_UPPER)

        for i, key in enumerate(keys):
            cust_txn_count[key] += 1
            if is_trivial[i]:
                cust_trivial_count[key] += 1
            # S3: store default date flag (only need to check once per customer)
            if key not in cust_default_date:
                cust_default_date[key] = create_vals[i].startswith(DEFAULT_DATE)
            elif not cust_default_date[key] and create_vals[i].startswith(DEFAULT_DATE):
                cust_default_date[key] = True

    log(f"  Total unique customers: {len(cust_txn_count):,}")

    # Build DataFrame
    all_keys = list(cust_txn_count.keys())
    df = pd.DataFrame({
        "server_id": [k[0] for k in all_keys],
        "shop_id": [k[1] for k in all_keys],
        "company_id": [k[2] for k in all_keys],
        "txn_count": [cust_txn_count[k] for k in all_keys],
        "trivial_amt_count": [cust_trivial_count.get(k, 0) for k in all_keys],
        "has_default_date": [1 if cust_default_date.get(k, False) else 0 for k in all_keys],
    })

    df["trivial_amt_ratio"] = df["trivial_amt_count"] / df["txn_count"]

    return df


# ===================================================================
# Determine S1 threshold from transaction count distribution
# ===================================================================
def compute_txn_threshold(df):
    """Find P99.9 threshold for extreme transaction counts."""
    p999 = np.percentile(df["txn_count"], 99.9)
    log(f"  Transaction count P99.9 = {p999:,.0f}")

    # Also show the distribution above P99 for context
    p99 = np.percentile(df["txn_count"], 99)
    above_p99 = df[df["txn_count"] > p99]["txn_count"].describe()
    log(f"  Transaction count P99 = {p99:,.0f}")
    log(f"  Distribution above P99:\n{above_p99}")

    return p999


# ===================================================================
# Score and classify candidates
# ===================================================================
def score_candidates(df, txn_threshold):
    """Compute composite score and classify each customer."""

    # S1: binary — above txn threshold
    df["s1_extreme_txn"] = (df["txn_count"] >= txn_threshold).astype(int)

    # S2: binary — majority trivial amounts, only if txn_count >= S2_MIN_TXN
    df["s2_trivial_amt"] = (
        (df["trivial_amt_ratio"] >= S2_THRESHOLD) & (df["txn_count"] >= S2_MIN_TXN)
    ).astype(int)

    # S3: already binary (has_default_date)
    df["s3_default_date"] = df["has_default_date"]

    # Composite: count of fired signals
    df["signals_fired"] = (
        df["s1_extreme_txn"]
        + df["s2_trivial_amt"]
        + df["s3_default_date"]
    )

    # Hard override: extreme txn count => auto-REMOVE
    df["extreme_override"] = (df["txn_count"] >= EXTREME_TXN_CEILING).astype(int)

    # Classification
    def classify(row):
        if row["extreme_override"] == 1:
            return "REMOVE"
        if row["signals_fired"] >= 2:
            return "REMOVE"
        if row["signals_fired"] == 1:
            return "REVIEW"
        return "KEEP"

    df["recommendation"] = df.apply(classify, axis=1)

    return df


# ===================================================================
# Generate summary report
# ===================================================================
def generate_summary(df_flagged, df_all_stats, txn_threshold):
    """Write human-readable summary."""
    txt_path = os.path.join(OUTPUT_DIR, "test_account_detection_summary.txt")

    total_customers = len(df_all_stats)
    total_txns = df_all_stats["txn_count"].sum()

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 2.2: TEST ACCOUNT DETECTION SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("SIGNAL DEFINITIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  S1: Transaction count >= {txn_threshold:,.0f} (P99.9)\n")
        f.write(f"  S2: Trivial amount ratio >= {S2_THRESHOLD:.0%}"
                f" (bill_amt in [0, {TRIVIAL_AMT_UPPER}])"
                f" AND txn_count >= {S2_MIN_TXN}\n")
        f.write(f"  S3: Default company_create date ({DEFAULT_DATE})\n")
        f.write("\n")

        f.write("CLASSIFICATION RULES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  REMOVE: 2+ signals fired, OR txn_count >= {EXTREME_TXN_CEILING:,} (hard override)\n")
        f.write("  REVIEW: 1 signal fired\n")
        f.write("  KEEP:   0 signals fired (not flagged)\n\n")

        f.write("RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total customers analyzed: {total_customers:,}\n")
        f.write(f"  Total transactions:       {total_txns:,}\n\n")

        for rec in ["REMOVE", "REVIEW"]:
            subset = df_flagged[df_flagged["recommendation"] == rec]
            if len(subset) == 0:
                f.write(f"  {rec}: 0 customers\n\n")
                continue
            txn_sum = subset["txn_count"].sum()
            f.write(f"  {rec}: {len(subset):,} customers, "
                    f"{txn_sum:,} transactions "
                    f"({txn_sum / total_txns * 100:.2f}% of total)\n")
            f.write(f"    Signals breakdown:\n")
            for sig in ["s1_extreme_txn", "s2_trivial_amt", "s3_default_date"]:
                count = subset[sig].sum()
                f.write(f"      {sig}: {count:,} customers\n")
            f.write("\n")

        # Top flagged accounts detail
        flagged = df_flagged[df_flagged["recommendation"].isin(["REMOVE", "REVIEW"])]
        flagged_sorted = flagged.sort_values("signals_fired", ascending=False)

        f.write("\nTOP FLAGGED ACCOUNTS (sorted by signals fired, then txn count)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'server_id':<15} {'shop':>4} {'cid':>8} {'txns':>10} "
                f"{'trivial%':>8} {'default':>7} "
                f"{'signals':>7} {'rec':>7}\n")
        f.write("-" * 70 + "\n")

        for _, row in flagged_sorted.head(50).iterrows():
            f.write(f"{row['server_id']:<15} {row['shop_id']:>4} "
                    f"{row['company_id']:>8} {row['txn_count']:>10,} "
                    f"{row['trivial_amt_ratio']:>7.1%} "
                    f"{row['has_default_date']:>7} "
                    f"{row['signals_fired']:>7} {row['recommendation']:>7}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {txt_path}")


# ===================================================================
# Remove test accounts from srv_sales parquet
# ===================================================================
def remove_test_accounts(remove_keys):
    """Read srv_sales parquet, exclude REMOVE customers, write cleaned parquet."""
    if not remove_keys:
        log("No accounts to remove. Copying input to output.")
        # Copy as-is
        import shutil
        src = SALES_PATH
        dst = os.path.join(OUTPUT_DIR, "srv_sales_step2_2.parquet")
        shutil.copy2(src, dst)
        pf = pq.ParquetFile(src)
        total = pf.metadata.num_rows
        return total, 0

    remove_set = set(remove_keys)
    cleaned_path = os.path.join(OUTPUT_DIR, "srv_sales_step2_2.parquet")

    log(f"PASS 2: Removing {len(remove_set):,} test accounts from srv_sales_step2_1.parquet...")

    pf = pq.ParquetFile(SALES_PATH)
    writer = None
    total_rows = 0
    kept_rows = 0
    removed_rows = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        chunk = batch.to_pandas()
        total_rows += len(chunk)
        if batch_idx % 5 == 0:
            log(f"  Batch {batch_idx}, rows processed: {total_rows:,}")

        # Build customer key for each row
        keys = list(zip(chunk["server_id"], chunk["shop_id"], chunk["company_id"]))
        mask_keep = [k not in remove_set for k in keys]

        chunk_kept = chunk[mask_keep]
        kept_rows += len(chunk_kept)
        removed_rows += len(chunk) - len(chunk_kept)

        # Write cleaned data to parquet
        table = pa.Table.from_pandas(chunk_kept, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(cleaned_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"  Total: {total_rows:,} rows, kept {kept_rows:,}, removed {removed_rows:,}")
    log(f"  Cleaned sales saved to {cleaned_path}")

    return kept_rows, removed_rows


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 2.2: Test Account Detection — START")

    # Pass 1: S1, S2, S3 from srv_sales
    df = compute_sales_signals()

    # Determine S1 threshold
    txn_threshold = compute_txn_threshold(df)

    # Score and classify all customers
    df_scored = score_candidates(df, txn_threshold)

    # Save flagged accounts CSV (REMOVE + REVIEW only)
    df_flagged = df_scored[df_scored["recommendation"].isin(["REMOVE", "REVIEW"])].copy()
    csv_path = os.path.join(OUTPUT_DIR, "flagged_test_accounts.csv")
    df_flagged.to_csv(csv_path, index=False)
    log(f"Flagged accounts saved to {csv_path} ({len(df_flagged):,} rows)")

    # Generate summary
    generate_summary(df_flagged, df_scored, txn_threshold)

    # Print quick stats
    remove_df = df_scored[df_scored["recommendation"] == "REMOVE"]
    review_df = df_scored[df_scored["recommendation"] == "REVIEW"]
    log(f"Results: {len(remove_df):,} REMOVE, {len(review_df):,} REVIEW")

    # Save exclusion log for REMOVE accounts
    exclusion_path = os.path.join(OUTPUT_DIR, "exclusion_log_step2_2.csv")
    if len(remove_df) > 0:
        exclusion_log = remove_df[["server_id", "shop_id", "company_id", "txn_count",
                                    "trivial_amt_ratio", "has_default_date",
                                    "signals_fired", "extreme_override"]].copy()
        exclusion_log["reason"] = exclusion_log.apply(
            lambda r: "extreme_txn_override" if r["extreme_override"] == 1
            else "multi_signal", axis=1
        )
        exclusion_log.to_csv(exclusion_path, index=False)
        log(f"Exclusion log saved to {exclusion_path}")

    # Pass 2: Remove test accounts and write cleaned sales data
    remove_keys = list(zip(
        remove_df["server_id"], remove_df["shop_id"], remove_df["company_id"]
    ))
    kept, removed = remove_test_accounts(remove_keys)
    log(f"Cleaned dataset: {kept:,} rows kept, {removed:,} rows removed")

    log("Step 2.2: Test Account Detection — COMPLETE")


if __name__ == "__main__":
    main()
