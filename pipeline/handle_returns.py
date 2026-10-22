"""
Step 2.3: Handle Returns and Corrections (Negative/Zero Amounts)
=================================================================
Reads from Step 2.2 output (srv_sales_step2_2.parquet) and
Step 2.1 output (srv_saleitems_step2_1.parquet).

Three phases:
  Phase A — Analyze: count and classify negative/zero bill_amt and amt records
  Phase B — Extract: compute per-customer return statistics BEFORE removal
  Phase C — Remove: delete negative bill_amt sales + cascade to saleitems

Zero bill_amt records are investigated and reported but NOT removed by default.
Review the summary report and set REMOVE_ZERO_BILL_AMT = True if warranted.

Output:
  - step2_3_returns_analysis.txt:   detailed breakdown of negative/zero records
  - return_statistics.csv:          per-customer return stats (computed before removal)
  - srv_sales_step2_3.parquet:      sales with returns removed
  - srv_saleitems_step2_3.parquet:  line items with return items removed
  - exclusion_log_step2_3.csv:      every removed record
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output", "step2_cleaning")
SALES_PATH = os.path.join(OUTPUT_DIR, "srv_sales_step2_2.parquet")
SALEITEMS_PATH = os.path.join(OUTPUT_DIR, "srv_saleitems_step2_1.parquet")
BATCH_SIZE = 500_000

# Set to True after reviewing the analysis to also remove bill_amt == 0 records
REMOVE_ZERO_BILL_AMT = False

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# PHASE A: Analyze negative/zero distributions
# ===================================================================
def analyze_sales():
    """Count and classify negative/zero bill_amt in srv_sales."""
    log("PHASE A: Analyzing bill_amt distribution in srv_sales...")

    pf = pq.ParquetFile(SALES_PATH)
    total_rows = 0
    neg_count = 0           # bill_amt < 0
    zero_count = 0          # bill_amt == 0
    pos_count = 0           # bill_amt > 0

    neg_amt_sum = 0.0       # total of negative bill_amt (absolute)
    zero_pay_with_zero_bill = 0   # bill_amt == 0 AND pay_amt == 0
    zero_bill_with_owe = 0        # bill_amt == 0 AND owe_amt > 0
    zero_bill_with_pos_pay = 0    # bill_amt == 0 AND pay_amt > 0

    # Collect samples for reporting
    neg_samples = []
    zero_samples = []
    MAX_SAMPLES = 20

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["server_id", "shop_id", "id", "company_id",
                 "bill_amt", "pay_amt", "owe_amt", "opt_on"],
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        mask_neg = chunk["bill_amt"] < 0
        mask_zero = chunk["bill_amt"] == 0
        mask_pos = chunk["bill_amt"] > 0

        neg_count += int(mask_neg.sum())
        zero_count += int(mask_zero.sum())
        pos_count += int(mask_pos.sum())

        neg_amt_sum += chunk.loc[mask_neg, "bill_amt"].abs().sum()

        # Zero bill_amt sub-classification
        zero_sub = chunk[mask_zero]
        zero_pay_with_zero_bill += int((zero_sub["pay_amt"] == 0).sum())
        zero_bill_with_owe += int((zero_sub["owe_amt"] > 0).sum())
        zero_bill_with_pos_pay += int((zero_sub["pay_amt"] > 0).sum())

        # Collect samples
        if len(neg_samples) < MAX_SAMPLES:
            neg_samples.extend(
                chunk[mask_neg].head(MAX_SAMPLES - len(neg_samples)).to_dict("records")
            )
        if len(zero_samples) < MAX_SAMPLES:
            zero_samples.extend(
                chunk[mask_zero].head(MAX_SAMPLES - len(zero_samples)).to_dict("records")
            )

    sales_stats = {
        "total_rows": total_rows,
        "neg_count": neg_count,
        "zero_count": zero_count,
        "pos_count": pos_count,
        "neg_pct": neg_count / total_rows * 100 if total_rows else 0,
        "zero_pct": zero_count / total_rows * 100 if total_rows else 0,
        "neg_amt_sum": neg_amt_sum,
        "zero_pay_with_zero_bill": zero_pay_with_zero_bill,
        "zero_bill_with_owe": zero_bill_with_owe,
        "zero_bill_with_pos_pay": zero_bill_with_pos_pay,
        "neg_samples": neg_samples,
        "zero_samples": zero_samples,
    }

    log(f"  Total rows:      {total_rows:,}")
    log(f"  bill_amt < 0:    {neg_count:,} ({sales_stats['neg_pct']:.2f}%)")
    log(f"  bill_amt == 0:   {zero_count:,} ({sales_stats['zero_pct']:.2f}%)")
    log(f"  bill_amt > 0:    {pos_count:,}")

    return sales_stats


def analyze_saleitems():
    """Count and classify negative/zero amt and qty in srv_saleitems."""
    log("PHASE A: Analyzing amt/qty distribution in srv_saleitems...")

    pf = pq.ParquetFile(SALEITEMS_PATH)
    total_rows = 0
    neg_amt_count = 0       # amt < 0
    zero_amt_count = 0      # amt == 0
    neg_qty_count = 0       # qty < 0
    zero_qty_count = 0      # qty == 0

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["amt", "qty"],
    ):
        chunk = batch.to_pandas()
        total_rows += len(chunk)

        neg_amt_count += int((chunk["amt"] < 0).sum())
        zero_amt_count += int((chunk["amt"] == 0).sum())
        neg_qty_count += int((chunk["qty"] < 0).sum())
        zero_qty_count += int((chunk["qty"] == 0).sum())

    items_stats = {
        "total_rows": total_rows,
        "neg_amt_count": neg_amt_count,
        "zero_amt_count": zero_amt_count,
        "neg_qty_count": neg_qty_count,
        "zero_qty_count": zero_qty_count,
        "neg_amt_pct": neg_amt_count / total_rows * 100 if total_rows else 0,
        "zero_amt_pct": zero_amt_count / total_rows * 100 if total_rows else 0,
        "neg_qty_pct": neg_qty_count / total_rows * 100 if total_rows else 0,
        "zero_qty_pct": zero_qty_count / total_rows * 100 if total_rows else 0,
    }

    log(f"  Total rows:   {total_rows:,}")
    log(f"  amt < 0:      {neg_amt_count:,} ({items_stats['neg_amt_pct']:.2f}%)")
    log(f"  amt == 0:     {zero_amt_count:,} ({items_stats['zero_amt_pct']:.2f}%)")
    log(f"  qty < 0:      {neg_qty_count:,} ({items_stats['neg_qty_pct']:.2f}%)")
    log(f"  qty == 0:     {zero_qty_count:,} ({items_stats['zero_qty_pct']:.2f}%)")

    return items_stats


# ===================================================================
# PHASE B: Extract per-customer return statistics BEFORE removal
# ===================================================================
def extract_return_statistics():
    """Compute return_count, return_amt_total, return_ratio per customer."""
    log("PHASE B: Computing per-customer return statistics...")

    pf = pq.ParquetFile(SALES_PATH)

    # Accumulators: customer_key -> counts
    cust_total_txn = defaultdict(int)
    cust_return_count = defaultdict(int)
    cust_return_amt = defaultdict(float)

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["server_id", "shop_id", "company_id", "bill_amt"],
    ):
        chunk = batch.to_pandas()

        # Count all transactions per customer
        for (srv, shop, cid), grp in chunk.groupby(
            ["server_id", "shop_id", "company_id"]
        ):
            key = (srv, shop, cid)
            cust_total_txn[key] += len(grp)

            neg_mask = grp["bill_amt"] < 0
            n_neg = int(neg_mask.sum())
            if n_neg > 0:
                cust_return_count[key] += n_neg
                cust_return_amt[key] += grp.loc[neg_mask, "bill_amt"].abs().sum()

    # Build and save DataFrame
    all_keys = list(cust_total_txn.keys())
    df = pd.DataFrame({
        "server_id": [k[0] for k in all_keys],
        "shop_id": [k[1] for k in all_keys],
        "company_id": [k[2] for k in all_keys],
        "total_txn_count": [cust_total_txn[k] for k in all_keys],
        "return_count": [cust_return_count.get(k, 0) for k in all_keys],
        "return_amt_total": [cust_return_amt.get(k, 0.0) for k in all_keys],
    })
    df["return_ratio"] = df["return_count"] / df["total_txn_count"]

    out_path = os.path.join(OUTPUT_DIR, "return_statistics.csv")
    df.to_csv(out_path, index=False)

    customers_with_returns = int((df["return_count"] > 0).sum())
    log(f"  Total customers: {len(df):,}")
    log(f"  Customers with returns: {customers_with_returns:,}"
        f" ({customers_with_returns / len(df) * 100:.2f}%)")
    log(f"  Saved to {out_path}")

    return df


# ===================================================================
# PHASE C: Remove negative bill_amt from srv_sales
# ===================================================================
def clean_sales():
    """Remove records with bill_amt < 0 (and optionally == 0) from srv_sales."""
    log("PHASE C: Cleaning srv_sales (removing returns)...")

    out_path = os.path.join(OUTPUT_DIR, "srv_sales_step2_3.parquet")
    excl_path = os.path.join(OUTPUT_DIR, "exclusion_log_step2_3.csv")

    pf = pq.ParquetFile(SALES_PATH)
    writer = None
    excl_header_written = False
    total_rows = 0
    kept_rows = 0
    removed_neg = 0
    removed_zero = 0

    # Collect sale composite keys of removed records for cascading to saleitems
    removed_sale_keys = set()

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        chunk = batch.to_pandas()
        total_rows += len(chunk)
        if batch_idx % 5 == 0:
            log(f"  Batch {batch_idx}, rows so far: {total_rows:,}")

        # Determine which rows to remove
        mask_neg = chunk["bill_amt"] < 0
        if REMOVE_ZERO_BILL_AMT:
            mask_remove = mask_neg | (chunk["bill_amt"] == 0)
        else:
            mask_remove = mask_neg

        n_neg = int(mask_neg.sum())
        n_zero = int((~mask_neg & mask_remove).sum()) if REMOVE_ZERO_BILL_AMT else 0
        removed_neg += n_neg
        removed_zero += n_zero

        # Log removed records
        if mask_remove.any():
            excl = chunk[mask_remove].copy()
            excl["source_table"] = "srv_sales"
            excl["exclusion_step"] = "2.3"
            excl["exclusion_reason"] = np.where(
                excl["bill_amt"] < 0,
                "negative_bill_amt_return",
                "zero_bill_amt",
            )
            excl.to_csv(
                excl_path,
                mode="a" if excl_header_written else "w",
                header=not excl_header_written,
                index=False,
            )
            excl_header_written = True

            # Track removed sale keys for cascade
            for srv, shop, sid in zip(
                chunk.loc[mask_remove, "server_id"],
                chunk.loc[mask_remove, "shop_id"],
                chunk.loc[mask_remove, "id"],
            ):
                removed_sale_keys.add((srv, shop, sid))

        # Write kept rows
        chunk_kept = chunk[~mask_remove]
        kept_rows += len(chunk_kept)

        table = pa.Table.from_pandas(chunk_kept, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"  srv_sales: {total_rows:,} total, {kept_rows:,} kept")
    log(f"    Removed negative bill_amt: {removed_neg:,}")
    log(f"    Removed zero bill_amt:     {removed_zero:,}")
    log(f"  Sale keys to cascade:        {len(removed_sale_keys):,}")
    log(f"  Saved to {out_path}")

    sales_result = {
        "total": total_rows,
        "kept": kept_rows,
        "removed_neg": removed_neg,
        "removed_zero": removed_zero,
    }

    return sales_result, removed_sale_keys


# ===================================================================
# PHASE C (continued): Cascade removal to srv_saleitems
# ===================================================================
def clean_saleitems(removed_sale_keys):
    """Remove line items whose parent sale was removed (cascade)."""
    log("PHASE C: Cascading removal to srv_saleitems...")

    out_path = os.path.join(OUTPUT_DIR, "srv_saleitems_step2_3.parquet")
    excl_path = os.path.join(OUTPUT_DIR, "exclusion_log_step2_3.csv")
    excl_header_written = os.path.exists(excl_path)

    pf = pq.ParquetFile(SALEITEMS_PATH)
    writer = None
    total_rows = 0
    kept_rows = 0
    removed_cascade = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        chunk = batch.to_pandas()
        total_rows += len(chunk)
        if batch_idx % 10 == 0:
            log(f"  Batch {batch_idx}, rows so far: {total_rows:,}")

        # Check if line item's parent sale was removed
        item_keys = list(zip(chunk["server_id"], chunk["shop_id"], chunk["sale_id"]))
        mask_cascade = np.array([k in removed_sale_keys for k in item_keys])

        n_cascade = int(mask_cascade.sum())
        removed_cascade += n_cascade

        # Log removed records
        if n_cascade > 0:
            excl = chunk[mask_cascade].copy()
            excl["source_table"] = "srv_saleitems"
            excl["exclusion_step"] = "2.3"
            excl["exclusion_reason"] = "cascade_parent_sale_removed"
            excl.to_csv(
                excl_path,
                mode="a",
                header=not excl_header_written,
                index=False,
            )
            excl_header_written = True

        # Write kept rows
        chunk_kept = chunk[~mask_cascade]
        kept_rows += len(chunk_kept)

        table = pa.Table.from_pandas(chunk_kept, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"  srv_saleitems: {total_rows:,} total, {kept_rows:,} kept,"
        f" {removed_cascade:,} cascade-removed")
    log(f"  Saved to {out_path}")

    return {
        "total": total_rows,
        "kept": kept_rows,
        "removed_cascade": removed_cascade,
    }


# ===================================================================
# Summary report
# ===================================================================
def write_summary(sales_stats, items_stats, return_stats_df,
                  sales_result, items_result):
    """Write human-readable analysis and removal summary."""
    txt_path = os.path.join(OUTPUT_DIR, "step2_3_returns_analysis.txt")

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 2.3: RETURNS AND CORRECTIONS ANALYSIS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"REMOVE_ZERO_BILL_AMT = {REMOVE_ZERO_BILL_AMT}\n")
        f.write("=" * 70 + "\n\n")

        # --- srv_sales analysis ---
        f.write("SRV_SALES: BILL_AMT CLASSIFICATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total rows:      {sales_stats['total_rows']:>12,}\n\n")
        f.write(f"  {'Category':<25s} {'Count':>12s} {'Percent':>10s}\n")
        f.write(f"  {'─' * 25} {'─' * 12} {'─' * 10}\n")
        f.write(f"  {'bill_amt < 0 (returns)':<25s}"
                f" {sales_stats['neg_count']:>12,}"
                f" {sales_stats['neg_pct']:>9.2f}%\n")
        f.write(f"  {'bill_amt == 0':<25s}"
                f" {sales_stats['zero_count']:>12,}"
                f" {sales_stats['zero_pct']:>9.2f}%\n")
        f.write(f"  {'bill_amt > 0 (normal)':<25s}"
                f" {sales_stats['pos_count']:>12,}"
                f" {(sales_stats['pos_count'] / sales_stats['total_rows'] * 100):>9.2f}%\n")
        f.write(f"\n  Total absolute value of negative bill_amt:"
                f" {sales_stats['neg_amt_sum']:>20,.2f}\n")

        # Zero bill_amt sub-analysis
        zc = sales_stats['zero_count']
        if zc > 0:
            f.write(f"\n  ZERO BILL_AMT SUB-ANALYSIS ({zc:,} records)\n")
            f.write(f"  {'─' * 50}\n")
            f.write(f"  {'pay_amt == 0 (no payment)':<35s}"
                    f" {sales_stats['zero_pay_with_zero_bill']:>10,}"
                    f" ({sales_stats['zero_pay_with_zero_bill'] / zc * 100:.1f}%)\n")
            f.write(f"  {'pay_amt > 0 (payment recorded)':<35s}"
                    f" {sales_stats['zero_bill_with_pos_pay']:>10,}"
                    f" ({sales_stats['zero_bill_with_pos_pay'] / zc * 100:.1f}%)\n")
            f.write(f"  {'owe_amt > 0 (debt recorded)':<35s}"
                    f" {sales_stats['zero_bill_with_owe']:>10,}"
                    f" ({sales_stats['zero_bill_with_owe'] / zc * 100:.1f}%)\n")

        # Sample negative records
        f.write(f"\n  SAMPLE NEGATIVE BILL_AMT RECORDS (up to 20)\n")
        f.write(f"  {'─' * 50}\n")
        for s in sales_stats["neg_samples"][:10]:
            f.write(f"    sale ({s.get('server_id','')}, {s.get('shop_id','')}, {s.get('id','')}):"
                    f" bill={s.get('bill_amt',0):,.2f}"
                    f" pay={s.get('pay_amt',0):,.2f}"
                    f" owe={s.get('owe_amt',0):,.2f}"
                    f" date={s.get('opt_on','')}\n")

        # Sample zero records
        f.write(f"\n  SAMPLE ZERO BILL_AMT RECORDS (up to 20)\n")
        f.write(f"  {'─' * 50}\n")
        for s in sales_stats["zero_samples"][:10]:
            f.write(f"    sale ({s.get('server_id','')}, {s.get('shop_id','')}, {s.get('id','')}):"
                    f" bill={s.get('bill_amt',0):,.2f}"
                    f" pay={s.get('pay_amt',0):,.2f}"
                    f" owe={s.get('owe_amt',0):,.2f}"
                    f" date={s.get('opt_on','')}\n")

        # --- srv_saleitems analysis ---
        f.write(f"\n\nSRV_SALEITEMS: AMT/QTY CLASSIFICATION\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total rows:   {items_stats['total_rows']:>12,}\n\n")
        f.write(f"  {'Category':<20s} {'Count':>12s} {'Percent':>10s}\n")
        f.write(f"  {'─' * 20} {'─' * 12} {'─' * 10}\n")
        f.write(f"  {'amt < 0':<20s}"
                f" {items_stats['neg_amt_count']:>12,}"
                f" {items_stats['neg_amt_pct']:>9.2f}%\n")
        f.write(f"  {'amt == 0':<20s}"
                f" {items_stats['zero_amt_count']:>12,}"
                f" {items_stats['zero_amt_pct']:>9.2f}%\n")
        f.write(f"  {'qty < 0':<20s}"
                f" {items_stats['neg_qty_count']:>12,}"
                f" {items_stats['neg_qty_pct']:>9.2f}%\n")
        f.write(f"  {'qty == 0':<20s}"
                f" {items_stats['zero_qty_count']:>12,}"
                f" {items_stats['zero_qty_pct']:>9.2f}%\n")

        # --- Return statistics ---
        f.write(f"\n\nPER-CUSTOMER RETURN STATISTICS\n")
        f.write("-" * 50 + "\n")
        has_returns = return_stats_df[return_stats_df["return_count"] > 0]
        f.write(f"  Total customers:          {len(return_stats_df):>10,}\n")
        f.write(f"  Customers with returns:   {len(has_returns):>10,}"
                f" ({len(has_returns) / len(return_stats_df) * 100:.2f}%)\n")
        if len(has_returns) > 0:
            f.write(f"\n  Among customers WITH returns:\n")
            f.write(f"    return_count  — median: {has_returns['return_count'].median():.0f},"
                    f" mean: {has_returns['return_count'].mean():.1f},"
                    f" max: {has_returns['return_count'].max():,}\n")
            f.write(f"    return_ratio  — median: {has_returns['return_ratio'].median():.2%},"
                    f" mean: {has_returns['return_ratio'].mean():.2%},"
                    f" max: {has_returns['return_ratio'].max():.2%}\n")
            f.write(f"    return_amt    — median: {has_returns['return_amt_total'].median():,.2f},"
                    f" mean: {has_returns['return_amt_total'].mean():,.2f}\n")

        # --- Removal results ---
        f.write(f"\n\n{'=' * 70}\n")
        f.write("REMOVAL RESULTS\n")
        f.write("-" * 50 + "\n\n")

        f.write("  srv_sales:\n")
        f.write(f"    Total rows:              {sales_result['total']:>12,}\n")
        f.write(f"    Removed (bill_amt < 0):  {sales_result['removed_neg']:>12,}\n")
        f.write(f"    Removed (bill_amt == 0): {sales_result['removed_zero']:>12,}\n")
        f.write(f"    Kept:                    {sales_result['kept']:>12,}\n")

        f.write(f"\n  srv_saleitems:\n")
        f.write(f"    Total rows:              {items_result['total']:>12,}\n")
        f.write(f"    Cascade-removed:         {items_result['removed_cascade']:>12,}\n")
        f.write(f"    Kept:                    {items_result['kept']:>12,}\n")

        # --- Output files ---
        f.write(f"\n\n{'=' * 70}\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        f.write("  srv_sales_step2_3.parquet      — cleaned sales\n")
        f.write("  srv_saleitems_step2_3.parquet   — cleaned line items\n")
        f.write("  return_statistics.csv           — per-customer return stats\n")
        f.write("  exclusion_log_step2_3.csv       — all removed records\n")
        f.write("  step2_3_returns_analysis.txt    — this file\n")

        if not REMOVE_ZERO_BILL_AMT and sales_stats["zero_count"] > 0:
            f.write(f"\n\nNOTE: {sales_stats['zero_count']:,} zero bill_amt records were KEPT.\n")
            f.write("Review the sub-analysis above. To remove them, set\n")
            f.write("REMOVE_ZERO_BILL_AMT = True and re-run.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {txt_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 2.3: Handle Returns and Corrections — START")

    # Phase A: Analyze
    sales_stats = analyze_sales()
    items_stats = analyze_saleitems()

    # Phase B: Extract return statistics before removal
    return_stats_df = extract_return_statistics()

    # Phase C: Remove and cascade
    sales_result, removed_sale_keys = clean_sales()
    items_result = clean_saleitems(removed_sale_keys)

    # Summary report
    write_summary(sales_stats, items_stats, return_stats_df,
                  sales_result, items_result)

    total_removed = (sales_result["removed_neg"] + sales_result["removed_zero"]
                     + items_result["removed_cascade"])
    log(f"Total removed: {total_removed:,} records"
        f" ({sales_result['removed_neg']:,} neg sales"
        f" + {sales_result['removed_zero']:,} zero sales"
        f" + {items_result['removed_cascade']:,} cascade items)")
    log("Step 2.3: Handle Returns and Corrections — COMPLETE")


if __name__ == "__main__":
    main()
