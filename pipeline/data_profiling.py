"""
Step 1: Data Profiling
======================
Covers all subsections from Data_Preprocessing_Plan.md:
  1.1 Assess Data Span and Scale
  1.2 Identify Verticals
  1.3 Compute Median Repurchase Cycles
  1.4 Assess Sufficient-History Proportion
  1.5 Data Quality Checks

Uses chunked reading to handle 7.2 GB of data without loading everything into memory.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sale_data")
SALES_PATH = os.path.join(DATA_DIR, "srv_sales.csv")
SALEITEMS_PATH = os.path.join(DATA_DIR, "srv_saleitems.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output", "step1_profiling")
CHUNK_SIZE = 500_000  # rows per chunk

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# PASS 1 — srv_sales.csv: Aggregate statistics in a single pass
# ===================================================================
def profile_sales() -> dict:
    """Single chunked pass over srv_sales.csv collecting all profiling stats."""
    log("Starting PASS 1: srv_sales.csv profiling...")

    # --- Accumulators ---
    total_rows = 0
    date_min, date_max = None, None
    unique_servers = set()
    unique_stores = set()  # (server_id, shop_id) — unique store composite key
    unique_customers = set()  # (server_id, shop_id, company_id) — true unique customer

    # 1.2 Verticals
    company_type_counts = defaultdict(int)

    # 1.5 Duplicate IDs — check both id-only and composite key
    id_only_set = set()
    composite_key_set = set()  # (server_id, shop_id, id)
    duplicate_id_only = 0
    duplicate_composite = 0

    # 1.5 Negative/zero amounts
    neg_zero_bill = 0
    neg_zero_pay = 0
    max_bill_amt = float("-inf")

    # Per-customer aggregates (for 1.1, 1.3, 1.4)
    # Store sorted purchase dates per (company_id, server_id, shop_id)
    customer_dates = defaultdict(list)  # key: (company_id, server_id, shop_id) -> [opt_on, ...]

    # Transactions per customer (for 1.1 distribution)
    customer_txn_count = defaultdict(int)

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            SALES_PATH,
            chunksize=CHUNK_SIZE,
            dtype={
                "server_id": str,
                "shop_id": "int32",
                "id": "int64",
                "code": str,
                "company_id": "int64",
                "company_type": str,
                "acct_id": "int64",
                "acct_name": str,
            },
            parse_dates=["opt_on"],
            low_memory=False,
        )
    ):
        total_rows += len(chunk)
        if chunk_idx % 5 == 0:
            log(f"  sales chunk {chunk_idx}, rows so far: {total_rows:,}")

        # 1.1 Span & Scale
        chunk_min = chunk["opt_on"].min()
        chunk_max = chunk["opt_on"].max()
        if date_min is None or chunk_min < date_min:
            date_min = chunk_min
        if date_max is None or chunk_max > date_max:
            date_max = chunk_max

        unique_servers.update(chunk["server_id"].dropna().unique())
        # Store = (server_id, shop_id) composite key; shop_id alone is NOT unique across merchants
        # Customer = (server_id, shop_id, company_id); company_id is per-merchant auto-increment
        store_pairs = set(zip(chunk["server_id"], chunk["shop_id"]))
        unique_stores.update(store_pairs)
        cust_triples = set(zip(chunk["server_id"], chunk["shop_id"], chunk["company_id"]))
        unique_customers.update(cust_triples)

        # 1.2 Verticals
        type_vc = chunk["company_type"].fillna("").value_counts()
        for val, cnt in type_vc.items():
            company_type_counts[val] += cnt

        # 1.5 Duplicate check: id-only vs composite key
        for srv, shop, rid in zip(chunk["server_id"], chunk["shop_id"], chunk["id"]):
            if rid in id_only_set:
                duplicate_id_only += 1
            else:
                id_only_set.add(rid)
            comp = (srv, shop, rid)
            if comp in composite_key_set:
                duplicate_composite += 1
            else:
                composite_key_set.add(comp)

        # 1.5 Negative/zero amounts
        neg_zero_bill += int((chunk["bill_amt"] <= 0).sum())
        neg_zero_pay += int((chunk["pay_amt"] <= 0).sum())
        chunk_max_bill = chunk["bill_amt"].max()
        if chunk_max_bill > max_bill_amt:
            max_bill_amt = chunk_max_bill

        # Per-customer purchase dates (for repurchase cycle)
        for (cid, srv, sid), grp in chunk.groupby(["company_id", "server_id", "shop_id"]):
            customer_dates[(cid, srv, sid)].extend(grp["opt_on"].tolist())

        # Transactions per true customer (server_id, shop_id, company_id)
        for (srv, shop, cid), grp in chunk.groupby(["server_id", "shop_id", "company_id"]):
            customer_txn_count[(srv, shop, cid)] += len(grp)

    log(f"  sales total rows: {total_rows:,}")

    # --- 1.1 Transactions per customer distribution ---
    txn_counts = np.array(list(customer_txn_count.values()))
    txn_dist = {
        "min": int(txn_counts.min()),
        "median": float(np.median(txn_counts)),
        "mean": float(txn_counts.mean()),
        "p90": float(np.percentile(txn_counts, 90)),
        "max": int(txn_counts.max()),
    }

    # --- 1.3 Compute Median Repurchase Cycles ---
    log("  Computing inter-purchase intervals...")
    customer_median_ipis = []  # per (customer, server_id, shop_id) median IPI
    per_store_ipis = defaultdict(list)  # (server_id, shop_id) -> list of all IPIs
    all_ipis = []
    customers_with_ge3 = 0

    for (cid, srv, sid), dates in customer_dates.items():
        dates_sorted = sorted(set(dates))  # deduplicate same-day purchases
        if len(dates_sorted) < 3:
            continue
        customers_with_ge3 += 1
        intervals = [
            (dates_sorted[i + 1] - dates_sorted[i]).days
            for i in range(len(dates_sorted) - 1)
        ]
        intervals = [iv for iv in intervals if iv > 0]  # exclude same-day
        if not intervals:
            continue
        med = float(np.median(intervals))
        customer_median_ipis.append(med)
        per_store_ipis[(srv, sid)].extend(intervals)
        all_ipis.extend(intervals)

    if all_ipis:
        all_ipis_arr = np.array(all_ipis)
        ipi_global = {
            "p5": float(np.percentile(all_ipis_arr, 5)),
            "p25": float(np.percentile(all_ipis_arr, 25)),
            "p50": float(np.percentile(all_ipis_arr, 50)),
            "p75": float(np.percentile(all_ipis_arr, 75)),
            "p95": float(np.percentile(all_ipis_arr, 95)),
            "mean": float(all_ipis_arr.mean()),
        }
    else:
        ipi_global = {}

    if customer_median_ipis:
        cust_med_arr = np.array(customer_median_ipis)
        ipi_customer_median_dist = {
            "p5": float(np.percentile(cust_med_arr, 5)),
            "p25": float(np.percentile(cust_med_arr, 25)),
            "p50": float(np.percentile(cust_med_arr, 50)),
            "p75": float(np.percentile(cust_med_arr, 75)),
            "p95": float(np.percentile(cust_med_arr, 95)),
            "mean": float(cust_med_arr.mean()),
        }
    else:
        ipi_customer_median_dist = {}

    # Per-store median IPI (keyed by (server_id, shop_id) composite)
    store_median_ipis = {}
    for (srv, sid), ipis in per_store_ipis.items():
        store_median_ipis[f"{srv}_{sid}"] = float(np.median(ipis))

    # --- 1.4 Sufficient-history proportion ---
    total_customers = len(customer_dates)
    sufficient_history = {}
    for n_thresh in [3, 4, 5]:
        count_meeting = sum(
            1
            for dates in customer_dates.values()
            if len(set(dates)) >= (n_thresh + 1)  # need N+1 unique dates for N intervals
        )
        sufficient_history[f"N={n_thresh}"] = {
            "count": count_meeting,
            "total_customers": total_customers,
            "proportion": round(count_meeting / total_customers, 4)
            if total_customers > 0
            else 0,
        }

    results = {
        "1.1_data_span_and_scale": {
            "total_sales_rows": total_rows,
            "date_range": {
                "min": str(date_min.date()) if date_min else None,
                "max": str(date_max.date()) if date_max else None,
            },
            "unique_merchants_server_id": len(unique_servers),
            "unique_stores_server_id_shop_id": len(unique_stores),
            "unique_customers_srv_shop_cid": len(unique_customers),
            "note_customer_key": "True customer = (server_id, shop_id, company_id). company_id is per-merchant auto-increment.",
            "transactions_per_customer": txn_dist,
        },
        "1.2_verticals": {
            "company_type_distribution": dict(company_type_counts),
            "note": "company_type encodes retail(零售)/wholesale(批发). No food&beverage vs supermarket vertical field found in dataset.",
        },
        "1.3_repurchase_cycles": {
            "customers_with_ge3_purchases": customers_with_ge3,
            "global_ipi_distribution": ipi_global,
            "per_customer_median_ipi_distribution": ipi_customer_median_dist,
            "per_store_median_ipi": store_median_ipis,
        },
        "1.4_sufficient_history": sufficient_history,
        "1.5_quality_sales": {
            "primary_key": "(server_id, shop_id, id)",
            "unique_id_only": len(id_only_set),
            "duplicate_id_only": duplicate_id_only,
            "unique_composite_key": len(composite_key_set),
            "duplicate_composite_key": duplicate_composite,
            "negative_or_zero_bill_amt": neg_zero_bill,
            "negative_or_zero_pay_amt": neg_zero_pay,
            "max_bill_amt": float(max_bill_amt),
        },
    }

    return results, composite_key_set


# ===================================================================
# PASS 2 — srv_saleitems.csv: Quality checks
# ===================================================================
def profile_saleitems(sale_composite_set: set) -> dict:
    """Single chunked pass over srv_saleitems.csv for quality checks.
    sale_composite_set contains (server_id, shop_id, id) tuples from srv_sales."""
    log("Starting PASS 2: srv_saleitems.csv profiling...")

    total_rows = 0
    unique_products = set()
    orphan_count = 0
    neg_zero_amt = 0
    max_item_amt = float("-inf")

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            SALEITEMS_PATH,
            chunksize=CHUNK_SIZE,
            dtype={
                "server_id": str,
                "shop_id": "int32",
                "id": "int64",
                "sale_id": "int64",
                "product_id": "int64",
            },
            parse_dates=["opt_on"],
            low_memory=False,
        )
    ):
        total_rows += len(chunk)
        if chunk_idx % 10 == 0:
            log(f"  saleitems chunk {chunk_idx}, rows so far: {total_rows:,}")

        unique_products.update(chunk["product_id"].unique())

        # Orphan line items (using composite key)
        for srv, shop, saleid in zip(chunk["server_id"], chunk["shop_id"], chunk["sale_id"]):
            if (srv, shop, saleid) not in sale_composite_set:
                orphan_count += 1

        # Negative/zero amounts
        neg_zero_amt += int((chunk["amt"] <= 0).sum())
        chunk_max = chunk["amt"].max()
        if chunk_max > max_item_amt:
            max_item_amt = chunk_max

    log(f"  saleitems total rows: {total_rows:,}")

    return {
        "1.5_quality_saleitems": {
            "total_saleitems_rows": total_rows,
            "unique_products": len(unique_products),
            "orphan_line_items": orphan_count,
            "negative_or_zero_amt": neg_zero_amt,
            "max_item_amt": float(max_item_amt),
        }
    }


# ===================================================================
# PASS 3 — Date consistency check (sample-based to save memory)
# ===================================================================
def check_date_consistency() -> dict:
    """Check opt_on consistency using composite key (server_id, shop_id, id/sale_id)."""
    log("Starting PASS 3: Date consistency check (composite key)...")

    # Load mapping: (server_id, shop_id, id) -> opt_on from sales
    sale_date_map = {}
    for chunk in pd.read_csv(
        SALES_PATH,
        chunksize=CHUNK_SIZE,
        usecols=["server_id", "shop_id", "id", "opt_on"],
        dtype={"server_id": str, "shop_id": "int32", "id": "int64"},
        parse_dates=["opt_on"],
    ):
        for srv, shop, rid, dt in zip(
            chunk["server_id"], chunk["shop_id"], chunk["id"], chunk["opt_on"]
        ):
            sale_date_map[(srv, shop, rid)] = dt

    log(f"  Loaded {len(sale_date_map):,} composite sale dates into memory")

    # Check saleitems against this map
    total_checked = 0
    matches = 0
    mismatches = 0
    not_found = 0
    mismatch_examples = []

    for chunk in pd.read_csv(
        SALEITEMS_PATH,
        chunksize=CHUNK_SIZE,
        usecols=["server_id", "shop_id", "sale_id", "opt_on"],
        dtype={"server_id": str, "shop_id": "int32", "sale_id": "int64"},
        parse_dates=["opt_on"],
    ):
        for srv, shop, saleid, item_date in zip(
            chunk["server_id"], chunk["shop_id"], chunk["sale_id"], chunk["opt_on"]
        ):
            total_checked += 1
            sale_date = sale_date_map.get((srv, shop, saleid))
            if sale_date is None:
                not_found += 1
            elif sale_date == item_date:
                matches += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append(
                        {
                            "key": f"({srv}, {shop}, {saleid})",
                            "sales_opt_on": str(sale_date.date()),
                            "saleitems_opt_on": str(item_date.date()),
                        }
                    )

    log(f"  Checked {total_checked:,} rows: {matches:,} match, {mismatches:,} mismatch, {not_found:,} not found")

    return {
        "1.5_date_consistency": {
            "total_checked": total_checked,
            "matches": matches,
            "match_rate": round(matches / total_checked, 6) if total_checked else 0,
            "mismatches": mismatches,
            "mismatch_rate": round(mismatches / total_checked, 6) if total_checked else 0,
            "orphans_not_found": not_found,
            "orphan_rate": round(not_found / total_checked, 6) if total_checked else 0,
            "mismatch_examples": mismatch_examples,
        }
    }


# ===================================================================
# Generate report
# ===================================================================
def generate_report(results: dict):
    """Write profiling report as JSON and human-readable text."""
    # JSON output
    json_path = os.path.join(OUTPUT_DIR, "profiling_report.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"JSON report saved to {json_path}")

    # Human-readable text report
    txt_path = os.path.join(OUTPUT_DIR, "profiling_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 1: DATA PROFILING REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # 1.1
        s = results["1.1_data_span_and_scale"]
        f.write("1.1 DATA SPAN AND SCALE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total sales rows:        {s['total_sales_rows']:>15,}\n")
        f.write(f"  Date range:              {s['date_range']['min']} to {s['date_range']['max']}\n")
        f.write(f"  Merchants (server_id):   {s['unique_merchants_server_id']:>15,}\n")
        f.write(f"  Stores (server_id+shop_id): {s['unique_stores_server_id_shop_id']:>12,}\n")
        f.write(f"  Customers (srv+shop+cid):{s['unique_customers_srv_shop_cid']:>15,}\n")
        f.write(f"  NOTE: company_id is per-merchant auto-increment, NOT globally unique.\n")
        f.write(f"        True customer = (server_id, shop_id, company_id).\n")
        f.write(f"\n  Transactions per customer:\n")
        for k, v in s["transactions_per_customer"].items():
            f.write(f"    {k:>8s}: {v:>15,.1f}\n")

        # 1.2
        f.write(f"\n\n1.2 VERTICALS\n")
        f.write("-" * 40 + "\n")
        v = results["1.2_verticals"]
        for ctype, cnt in sorted(v["company_type_distribution"].items(), key=lambda x: -x[1]):
            label = ctype if ctype else "(empty)"
            f.write(f"  {label:>12s}: {cnt:>12,}\n")
        f.write(f"\n  Note: {v['note']}\n")

        # 1.3
        f.write(f"\n\n1.3 MEDIAN REPURCHASE CYCLES\n")
        f.write("-" * 40 + "\n")
        r = results["1.3_repurchase_cycles"]
        f.write(f"  Customers with >= 3 purchases: {r['customers_with_ge3_purchases']:,}\n")
        f.write(f"\n  Global IPI distribution (days):\n")
        for k, v in r["global_ipi_distribution"].items():
            f.write(f"    {k:>6s}: {v:>10.1f}\n")
        f.write(f"\n  Per-customer median IPI distribution (days):\n")
        for k, v in r["per_customer_median_ipi_distribution"].items():
            f.write(f"    {k:>6s}: {v:>10.1f}\n")
        store_ipis = r["per_store_median_ipi"]
        sorted_store_ipis = sorted(store_ipis.items(), key=lambda x: x[1])
        f.write(f"\n  Per-store median IPI (days) — store = (server_id, shop_id):\n")
        f.write(f"    Total stores with IPI data: {len(store_ipis)}\n")
        ipi_vals = [v for _, v in sorted_store_ipis]
        if ipi_vals:
            f.write(f"    Store IPI range: {min(ipi_vals):.1f} to {max(ipi_vals):.1f} days\n")
            f.write(f"    Store IPI median: {float(np.median(ipi_vals)):.1f} days\n")
            f.write(f"\n    Bottom 10 (fastest repurchase):\n")
            for store_key, med in sorted_store_ipis[:10]:
                f.write(f"      store {store_key}: {med:>8.1f}\n")
            f.write(f"\n    Top 10 (slowest repurchase):\n")
            for store_key, med in sorted_store_ipis[-10:]:
                f.write(f"      store {store_key}: {med:>8.1f}\n")

        # 1.4
        f.write(f"\n\n1.4 SUFFICIENT-HISTORY PROPORTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  True customer = (server_id, shop_id, company_id)\n\n")
        for thresh, info in results["1.4_sufficient_history"].items():
            f.write(
                f"  {thresh}: {info['count']:>10,} / {info['total_customers']:>10,}"
                f"  ({info['proportion']:.2%})\n"
            )

        # 1.5
        f.write(f"\n\n1.5 DATA QUALITY CHECKS\n")
        f.write("-" * 40 + "\n")
        q_sales = results["1.5_quality_sales"]

        # Key discovery: composite key comparison
        f.write(f"\n  PRIMARY KEY ANALYSIS (srv_sales):\n")
        f.write(f"    Correct primary key: {q_sales['primary_key']}\n\n")
        f.write(f"    {'Metric':<30s} {'id only':>15s} {'composite key':>15s}\n")
        f.write(f"    {'─' * 30} {'─' * 15} {'─' * 15}\n")
        f.write(f"    {'Unique keys':<30s} {q_sales['unique_id_only']:>15,} {q_sales['unique_composite_key']:>15,}\n")
        f.write(f"    {'Duplicate rows':<30s} {q_sales['duplicate_id_only']:>15,} {q_sales['duplicate_composite_key']:>15,}\n")

        q_dates = results["1.5_date_consistency"]
        f.write(f"    {'Date matches':<30s} {'N/A':>15s} {q_dates['matches']:>15,}\n")
        f.write(f"    {'Date mismatches':<30s} {'N/A':>15s} {q_dates['mismatches']:>15,}\n")
        f.write(f"    {'Orphan items (not found)':<30s} {'N/A':>15s} {q_dates['orphans_not_found']:>15,}\n")

        f.write(f"\n  FINANCIAL QUALITY (srv_sales):\n")
        f.write(f"    Neg/zero bill_amt:          {q_sales['negative_or_zero_bill_amt']:>12,}\n")
        f.write(f"    Neg/zero pay_amt:           {q_sales['negative_or_zero_pay_amt']:>12,}\n")
        f.write(f"    Max bill_amt:               {q_sales['max_bill_amt']:>20,.2f}\n")

        q_items = results["1.5_quality_saleitems"]
        f.write(f"\n  QUALITY (srv_saleitems):\n")
        f.write(f"    Total rows:                 {q_items['total_saleitems_rows']:>12,}\n")
        f.write(f"    Unique products:            {q_items['unique_products']:>12,}\n")
        f.write(f"    Orphan line items:          {q_items['orphan_line_items']:>12,}\n")
        f.write(f"    Neg/zero amt:               {q_items['negative_or_zero_amt']:>12,}\n")
        f.write(f"    Max item amt:               {q_items['max_item_amt']:>20,.2f}\n")

        f.write(f"\n  DATE CONSISTENCY (composite key):\n")
        f.write(f"    Total checked:              {q_dates['total_checked']:>12,}\n")
        f.write(f"    Matches:                    {q_dates['matches']:>12,}  ({q_dates['match_rate']:>8.4%})\n")
        f.write(f"    Mismatches:                 {q_dates['mismatches']:>12,}  ({q_dates['mismatch_rate']:>8.4%})\n")
        f.write(f"    Orphans (not found):        {q_dates['orphans_not_found']:>12,}  ({q_dates['orphan_rate']:>8.4%})\n")
        if q_dates["mismatch_examples"]:
            f.write(f"    Mismatch examples:\n")
            for ex in q_dates["mismatch_examples"]:
                f.write(
                    f"      key={ex['key']}: "
                    f"sales={ex['sales_opt_on']}, items={ex['saleitems_opt_on']}\n"
                )

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF PROFILING REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Text report saved to {txt_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 1: Data Profiling — START")

    # Pass 1: srv_sales profiling
    sales_results, sale_composite_set = profile_sales()

    # Pass 2: srv_saleitems profiling (uses composite key for orphan check)
    items_results = profile_saleitems(sale_composite_set)

    # Free the large set before pass 3 loads sale dates
    del sale_composite_set

    # Pass 3: Date consistency
    date_results = check_date_consistency()

    # Merge all results
    all_results = {**sales_results, **items_results, **date_results}

    # Generate report
    generate_report(all_results)

    log("Step 1: Data Profiling — COMPLETE")


if __name__ == "__main__":
    main()
