"""
Step 2.4: Remove Orphan Line Items
====================================
Reads from Step 2.3 outputs:
  - srv_sales_step2_3.parquet     (source of valid sale keys)
  - srv_saleitems_step2_3.parquet (to filter orphans)

An orphan is a line item whose (server_id, shop_id, sale_id) has no matching
(server_id, shop_id, id) in srv_sales. These records cannot be attributed to
a customer or date, so they are unusable for feature construction.

Note: Step 2.2 removed test accounts from srv_sales but did NOT cascade to
srv_saleitems. Those line items are now orphans and will be caught here.

Output:
  - srv_saleitems_step2_4.parquet: line items with orphans removed
  - exclusion_log_step2_4.csv:     removed orphan records (sample)
  - step2_4_orphan_summary.txt:    counts and diagnostics
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
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
SALES_PATH = os.path.join(OUTPUT_DIR, "srv_sales_step2_3.parquet")
SALEITEMS_PATH = os.path.join(OUTPUT_DIR, "srv_saleitems_step2_3.parquet")
BATCH_SIZE = 500_000

# Max orphan records to write to the exclusion log (full file would be large)
EXCL_LOG_SAMPLE_SIZE = 10_000

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# PASS 1: Collect valid sale keys from srv_sales
# ===================================================================
def load_valid_sale_keys():
    """Read all (server_id, shop_id, id) from cleaned srv_sales."""
    log("PASS 1: Loading valid sale keys from srv_sales_step2_3.parquet...")

    valid_keys = set()
    pf = pq.ParquetFile(SALES_PATH)

    for batch_idx, batch in enumerate(pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["server_id", "shop_id", "id"],
    )):
        chunk = batch.to_pandas()
        keys = set(zip(chunk["server_id"], chunk["shop_id"], chunk["id"]))
        valid_keys.update(keys)
        if batch_idx % 5 == 0:
            log(f"  Batch {batch_idx}, valid keys so far: {len(valid_keys):,}")

    log(f"  Total valid sale keys: {len(valid_keys):,}")
    return valid_keys


# ===================================================================
# PASS 2: Filter orphans from srv_saleitems
# ===================================================================
def remove_orphans(valid_keys):
    """Remove line items whose parent sale doesn't exist in cleaned sales."""
    log("PASS 2: Removing orphan line items from srv_saleitems...")

    out_path = os.path.join(OUTPUT_DIR, "srv_saleitems_step2_4.parquet")
    excl_path = os.path.join(OUTPUT_DIR, "exclusion_log_step2_4.csv")

    pf = pq.ParquetFile(SALEITEMS_PATH)
    writer = None
    excl_header_written = False
    excl_logged = 0

    total_rows = 0
    kept_rows = 0
    orphan_rows = 0

    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE)):
        chunk = batch.to_pandas()
        total_rows += len(chunk)
        if batch_idx % 10 == 0:
            log(f"  Batch {batch_idx}, rows so far: {total_rows:,}")

        # Check parent sale existence
        item_keys = list(zip(chunk["server_id"], chunk["shop_id"], chunk["sale_id"]))
        mask_orphan = np.array([k not in valid_keys for k in item_keys])

        n_orphan = int(mask_orphan.sum())
        orphan_rows += n_orphan

        # Log a sample of orphan records
        if n_orphan > 0 and excl_logged < EXCL_LOG_SAMPLE_SIZE:
            remaining = EXCL_LOG_SAMPLE_SIZE - excl_logged
            excl = chunk[mask_orphan].head(remaining).copy()
            excl["source_table"] = "srv_saleitems"
            excl["exclusion_step"] = "2.4"
            excl["exclusion_reason"] = "orphan_no_parent_sale"
            excl.to_csv(
                excl_path,
                mode="a" if excl_header_written else "w",
                header=not excl_header_written,
                index=False,
            )
            excl_header_written = True
            excl_logged += len(excl)

        # Write kept rows
        chunk_kept = chunk[~mask_orphan]
        kept_rows += len(chunk_kept)

        table = pa.Table.from_pandas(chunk_kept, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

    if writer is not None:
        writer.close()

    log(f"  srv_saleitems: {total_rows:,} total, {kept_rows:,} kept,"
        f" {orphan_rows:,} orphans removed")
    log(f"  Saved to {out_path}")
    if excl_logged > 0:
        log(f"  Exclusion log: {excl_logged:,} sample records saved to {excl_path}")

    return {
        "total": total_rows,
        "kept": kept_rows,
        "orphan_rows": orphan_rows,
        "excl_logged": excl_logged,
    }


# ===================================================================
# Summary report
# ===================================================================
def write_summary(n_valid_keys, result):
    """Write human-readable summary."""
    txt_path = os.path.join(OUTPUT_DIR, "step2_4_orphan_summary.txt")

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 2.4: ORPHAN LINE ITEM REMOVAL\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("CONTEXT\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Valid sale keys in srv_sales:  {n_valid_keys:>12,}\n")
        f.write(f"  Total line items checked:      {result['total']:>12,}\n\n")

        f.write("RESULTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Orphan line items removed:     {result['orphan_rows']:>12,}"
                f"  ({result['orphan_rows'] / result['total'] * 100:.2f}%)\n")
        f.write(f"  Line items kept:               {result['kept']:>12,}"
                f"  ({result['kept'] / result['total'] * 100:.2f}%)\n\n")

        f.write("ORPHAN SOURCES\n")
        f.write("-" * 50 + "\n")
        f.write("  Orphans come from three sources:\n")
        f.write("    1. Originally orphaned in raw data (pre-existing referential gaps)\n")
        f.write("    2. Step 2.1: parent sale removed as financial outlier\n")
        f.write("    3. Step 2.2: parent sale removed as test account\n")
        f.write("       (Step 2.2 did not cascade to saleitems — caught here)\n")
        f.write("  Note: Step 2.3 return cascades were already handled in step 2.3.\n\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        f.write("  srv_saleitems_step2_4.parquet   — cleaned line items\n")
        f.write("  exclusion_log_step2_4.csv       — sample of removed orphans"
                f" ({result['excl_logged']:,} records)\n")
        f.write("  step2_4_orphan_summary.txt      — this file\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {txt_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 2.4: Remove Orphan Line Items — START")

    valid_keys = load_valid_sale_keys()
    result = remove_orphans(valid_keys)
    write_summary(len(valid_keys), result)

    log(f"Removed {result['orphan_rows']:,} orphan line items"
        f" ({result['orphan_rows'] / result['total'] * 100:.2f}%)")
    log("Step 2.4: Remove Orphan Line Items — COMPLETE")


if __name__ == "__main__":
    main()
