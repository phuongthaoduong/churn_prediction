"""
Step 4: Feature Engineering
============================
Constructs features for each (server_id, shop_id, company_id, prediction_point)
tuple using ONLY data within the observation window [obs_start, obs_end).

Features (25 total):
  Customer-level (14): recency, frequency, monetary, avg_monetary, ipi_mean,
    ipi_std, recent_activity, freq_decline_ratio, unique_products,
    avg_basket_size, total_qty, pay_ratio, debt_frequency, customer_tenure
  Store-level (6): store_txn_volume, store_active_customers, store_median_ipi,
    recency_vs_store, frequency_vs_store, monetary_vs_store
  Time-level (5): month, quarter, is_cny_season, is_double11, is_spring_festival

Input:
  - srv_sales_step2_3.parquet       (cleaned sales, 14.8M rows)
  - srv_saleitems_step2_4.parquet   (cleaned line items, 46M rows)
  - batch_assignments.parquet       (6.3M customer-batch assignments)

Output:
  - feature_matrix.parquet          (one row per customer x batch, all features)
  - step4_feature_summary.txt       (human-readable report)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROFILING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
STEP2_DIR = os.path.join(PROFILING_DIR, "step2_cleaning")
STEP3_DIR = os.path.join(PROFILING_DIR, "step3_temporal")
OUTPUT_DIR = os.path.join(PROFILING_DIR, "step4_features")

SALES_PATH = os.path.join(STEP2_DIR, "srv_sales_step2_3.parquet")
SALEITEMS_PATH = os.path.join(STEP2_DIR, "srv_saleitems_step2_4.parquet")
BATCH_ASSIGN_PATH = os.path.join(STEP3_DIR, "batch_assignments.parquet")

FEATURE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "feature_matrix.parquet")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "step4_feature_summary.txt")

OBS_WINDOW_DAYS = 90
SI_CHUNK_SIZE = 2_000_000

CUST_COLS = ["server_id", "shop_id", "company_id"]
STORE_COLS = ["server_id", "shop_id"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# Phase 1: Load sales data
# ===================================================================
def load_sales():
    """Load cleaned sales with only the columns needed for feature engineering."""
    log("Phase 1: Loading cleaned sales data...")
    sales = pd.read_parquet(SALES_PATH, columns=[
        "server_id", "shop_id", "id", "opt_on", "company_id",
        "company_create", "bill_amt", "pay_amt", "owe_amt",
    ])
    # Ensure consistent dtypes for join keys
    sales["shop_id"] = sales["shop_id"].astype("int64")
    sales["opt_on"] = pd.to_datetime(sales["opt_on"])
    sales["company_create"] = pd.to_datetime(sales["company_create"])
    log(f"  Loaded {len(sales):,} sales rows")
    return sales


# ===================================================================
# Phase 2: Pre-aggregate saleitems (chunked)
# ===================================================================
def preaggregate_saleitems(sale_lookup):
    """Read saleitems in chunks, join with sales to get customer info.

    Returns:
        sale_stats:    per-sale (n_items, total_qty) — merged back to sales later
        cust_products: deduped (customer, opt_on, product_id) for unique_products
    """
    log("Phase 2: Pre-aggregating saleitems (chunked read)...")

    pf = pq.ParquetFile(SALEITEMS_PATH)
    sale_stats_parts = []
    cust_products_parts = []
    total_rows = 0

    for batch_idx, batch in enumerate(pf.iter_batches(
        batch_size=SI_CHUNK_SIZE,
        columns=["server_id", "shop_id", "sale_id", "product_id", "qty"],
    )):
        chunk = batch.to_pandas()
        chunk["shop_id"] = chunk["shop_id"].astype("int64")
        total_rows += len(chunk)
        if batch_idx % 5 == 0:
            log(f"  Chunk {batch_idx}, rows processed: {total_rows:,}")

        # Inner join with sales to get company_id + opt_on (filters to valid sales)
        chunk = chunk.merge(sale_lookup, on=["server_id", "shop_id", "sale_id"],
                            how="inner")

        # Per-sale aggregation: item count and total quantity
        ss = chunk.groupby(["server_id", "shop_id", "sale_id"]).agg(
            n_items=("product_id", "count"),
            total_qty=("qty", "sum"),
        ).reset_index()
        sale_stats_parts.append(ss)

        # Customer-product pairs for unique_products (dedup within chunk)
        cp = chunk[["server_id", "shop_id", "company_id", "opt_on", "product_id"]] \
            .drop_duplicates()
        cust_products_parts.append(cp)

    log(f"  Total saleitems processed: {total_rows:,}")

    # Re-aggregate sale_stats (a sale's items may span chunks)
    log("  Combining sale-level stats...")
    sale_stats = pd.concat(sale_stats_parts, ignore_index=True)
    sale_stats = sale_stats.groupby(["server_id", "shop_id", "sale_id"]).agg(
        n_items=("n_items", "sum"),
        total_qty=("total_qty", "sum"),
    ).reset_index()
    log(f"  Sale stats: {len(sale_stats):,} unique sales")
    del sale_stats_parts

    # Dedup customer-product pairs across chunks
    log("  Combining customer-product pairs...")
    cust_products = pd.concat(cust_products_parts, ignore_index=True).drop_duplicates()
    log(f"  Customer-product pairs: {len(cust_products):,}")
    del cust_products_parts

    return sale_stats, cust_products


# ===================================================================
# Phase 3: Compute features for one batch
# ===================================================================
def compute_batch_features(sales, sale_stats, cust_products, eligible, pp):
    """Compute all 25 features for one prediction point.

    Args:
        sales:         full cleaned sales DataFrame
        sale_stats:    per-sale item stats (n_items, total_qty)
        cust_products: deduped (customer, opt_on, product_id) pairs
        eligible:      batch_assignments rows for this prediction point
        pp:            prediction point timestamp
    """
    obs_start = pp - pd.Timedelta(days=OBS_WINDOW_DAYS)
    obs_end = pp

    # ---- Filter sales to observation window ----
    date_mask = (sales["opt_on"] >= obs_start) & (sales["opt_on"] < obs_end)
    s = sales[date_mask].copy()

    # Keep only eligible customers
    elig_keys = eligible[CUST_COLS].drop_duplicates()
    s = s.merge(elig_keys, on=CUST_COLS, how="inner")

    # ================================================================
    # Customer-level features
    # ================================================================

    # --- RFM + payment features (single groupby) ---
    agg = s.groupby(CUST_COLS).agg(
        last_purchase=("opt_on", "max"),
        frequency=("id", "count"),
        monetary=("bill_amt", "sum"),
        avg_monetary=("bill_amt", "mean"),
        _sum_pay=("pay_amt", "sum"),
        debt_frequency=("owe_amt", lambda x: (x > 0).mean()),
        _company_create=("company_create", "first"),
    ).reset_index()

    agg["recency"] = (pp - agg["last_purchase"]).dt.days
    agg["pay_ratio"] = np.where(
        agg["monetary"] != 0, agg["_sum_pay"] / agg["monetary"], 0.0
    )
    agg["customer_tenure"] = (pp - agg["_company_create"]).dt.days
    agg.drop(columns=["last_purchase", "_sum_pay", "_company_create"], inplace=True)

    # --- IPI features (inter-purchase interval) ---
    unique_dates = (
        s[CUST_COLS + ["opt_on"]]
        .drop_duplicates()
        .sort_values(CUST_COLS + ["opt_on"])
    )
    unique_dates["_prev"] = unique_dates.groupby(CUST_COLS)["opt_on"].shift(1)
    unique_dates["ipi"] = (unique_dates["opt_on"] - unique_dates["_prev"]).dt.days
    unique_dates = unique_dates.dropna(subset=["ipi"])

    ipi_stats = unique_dates.groupby(CUST_COLS)["ipi"].agg(
        ipi_mean="mean", ipi_std="std"
    ).reset_index()

    agg = agg.merge(ipi_stats, on=CUST_COLS, how="left")

    # --- Recent activity (last 30 days of obs window) ---
    recent_start = obs_end - pd.Timedelta(days=30)
    recent = (
        s[s["opt_on"] >= recent_start]
        .groupby(CUST_COLS)["id"]
        .count()
        .reset_index()
        .rename(columns={"id": "recent_activity"})
    )
    agg = agg.merge(recent, on=CUST_COLS, how="left")
    agg["recent_activity"] = agg["recent_activity"].fillna(0).astype("int64")

    # --- Frequency decline ratio (2nd half / 1st half of obs window) ---
    obs_mid = obs_start + pd.Timedelta(days=OBS_WINDOW_DAYS // 2)
    half1 = (
        s[s["opt_on"] < obs_mid]
        .groupby(CUST_COLS)["id"].count()
        .reset_index().rename(columns={"id": "_h1"})
    )
    half2 = (
        s[s["opt_on"] >= obs_mid]
        .groupby(CUST_COLS)["id"].count()
        .reset_index().rename(columns={"id": "_h2"})
    )
    agg = agg.merge(half1, on=CUST_COLS, how="left")
    agg = agg.merge(half2, on=CUST_COLS, how="left")
    agg["_h1"] = agg["_h1"].fillna(0)
    agg["_h2"] = agg["_h2"].fillna(0)
    agg["freq_decline_ratio"] = np.where(
        agg["_h1"] > 0,
        agg["_h2"] / agg["_h1"],
        np.where(agg["_h2"] > 0, 2.0, 1.0),
    )
    agg.drop(columns=["_h1", "_h2"], inplace=True)

    # --- Saleitems-derived features ---
    # Merge per-sale item stats into obs-window sales
    s_items = s[CUST_COLS + ["id"]].merge(
        sale_stats.rename(columns={"sale_id": "id"}),
        on=["server_id", "shop_id", "id"],
        how="left",
    )
    s_items["n_items"] = s_items["n_items"].fillna(0)
    s_items["total_qty"] = s_items["total_qty"].fillna(0)

    basket = s_items.groupby(CUST_COLS).agg(
        avg_basket_size=("n_items", "mean"),
        total_qty=("total_qty", "sum"),
    ).reset_index()
    agg = agg.merge(basket, on=CUST_COLS, how="left")

    # Unique products per customer in obs window
    cp_mask = (cust_products["opt_on"] >= obs_start) & (cust_products["opt_on"] < obs_end)
    cp_obs = cust_products[cp_mask].merge(elig_keys, on=CUST_COLS, how="inner")
    unique_prods = (
        cp_obs.groupby(CUST_COLS)["product_id"]
        .nunique()
        .reset_index()
        .rename(columns={"product_id": "unique_products"})
    )
    agg = agg.merge(unique_prods, on=CUST_COLS, how="left")
    agg["unique_products"] = agg["unique_products"].fillna(0).astype("int64")

    # ================================================================
    # Store-level features
    # ================================================================
    # Store volume and active customers (from ALL sales in obs window, not just eligible)
    s_all = sales[date_mask]
    store_volume = s_all.groupby(STORE_COLS).agg(
        store_txn_volume=("id", "count"),
        store_active_customers=("company_id", "nunique"),
    ).reset_index()

    # Store medians from eligible customers' features
    store_medians = agg.groupby(STORE_COLS).agg(
        _med_recency=("recency", "median"),
        _med_frequency=("frequency", "median"),
        _med_monetary=("monetary", "median"),
    ).reset_index()

    # Store median IPI (median of per-customer ipi_mean)
    ipi_for_store = ipi_stats[ipi_stats["ipi_mean"].notna()]
    if len(ipi_for_store) > 0:
        store_ipi = (
            ipi_for_store.groupby(STORE_COLS)["ipi_mean"]
            .median()
            .reset_index()
            .rename(columns={"ipi_mean": "store_median_ipi"})
        )
    else:
        store_ipi = pd.DataFrame(columns=STORE_COLS + ["store_median_ipi"])

    agg = agg.merge(store_volume, on=STORE_COLS, how="left")
    agg = agg.merge(store_medians, on=STORE_COLS, how="left")
    agg = agg.merge(store_ipi, on=STORE_COLS, how="left")

    # Relative features (customer vs. store)
    agg["recency_vs_store"] = np.where(
        agg["_med_recency"] > 0, agg["recency"] / agg["_med_recency"], 1.0
    )
    agg["frequency_vs_store"] = np.where(
        agg["_med_frequency"] > 0, agg["frequency"] / agg["_med_frequency"], 1.0
    )
    agg["monetary_vs_store"] = np.where(
        agg["_med_monetary"] > 0, agg["monetary"] / agg["_med_monetary"], 1.0
    )
    agg.drop(columns=["_med_recency", "_med_frequency", "_med_monetary"], inplace=True)

    # Log1p transform to reduce extreme monetary skew (max ~2.7B)
    agg["monetary"] = np.log1p(agg["monetary"])
    agg["avg_monetary"] = np.log1p(agg["avg_monetary"])

    # ================================================================
    # Time-level features
    # ================================================================
    agg["month"] = pp.month
    agg["quarter"] = pp.quarter
    agg["is_cny_season"] = 1 if pp.month in (1, 2) else 0

    # is_double11: obs window overlaps any November
    agg["is_double11"] = 1 if any(
        obs_start <= pd.Timestamp(f"{y}-11-30") and obs_end >= pd.Timestamp(f"{y}-11-01")
        for y in range(obs_start.year, obs_end.year + 1)
    ) else 0

    # is_spring_festival: obs window overlaps late Jan–mid Feb
    agg["is_spring_festival"] = 1 if any(
        obs_start <= pd.Timestamp(f"{y}-02-15") and obs_end >= pd.Timestamp(f"{y}-01-20")
        for y in range(obs_start.year, obs_end.year + 1)
    ) else 0

    agg["prediction_point"] = pp

    return agg


# ===================================================================
# Summary report
# ===================================================================
def write_summary(fm, prediction_points):
    """Write human-readable summary of the feature matrix."""
    key_cols = ["server_id", "shop_id", "company_id", "prediction_point"]
    feature_cols = [c for c in fm.columns if c not in key_cols]

    with open(SUMMARY_PATH, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 4: FEATURE ENGINEERING\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Total rows:        {len(fm):,}\n")
        f.write(f"  Total columns:     {len(fm.columns)}"
                f" ({len(key_cols)} keys + {len(feature_cols)} features)\n")
        f.write(f"  Prediction points: {len(prediction_points)}\n")
        n_cust = fm.groupby(CUST_COLS).ngroups
        f.write(f"  Unique customers:  {n_cust:,}\n\n")

        # Per-batch row counts
        f.write("PER-BATCH COUNTS\n")
        f.write("-" * 50 + "\n")
        batch_counts = fm.groupby("prediction_point").size()
        for pp, cnt in batch_counts.items():
            f.write(f"  {pd.Timestamp(pp).date()}: {cnt:>10,}\n")
        f.write("\n")

        # Feature statistics
        f.write("FEATURE STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Feature':<25s}  {'Mean':>12s}  {'Std':>12s}"
                f"  {'Min':>12s}  {'Max':>12s}  {'Miss%':>7s}\n")
        f.write(f"  {'─' * 25}  {'─' * 12}  {'─' * 12}"
                f"  {'─' * 12}  {'─' * 12}  {'─' * 7}\n")

        for col in feature_cols:
            if fm[col].dtype in ("float64", "float32", "int64", "int32"):
                miss_pct = fm[col].isna().mean() * 100
                vals = fm[col].dropna()
                if len(vals) > 0:
                    f.write(f"  {col:<25s}"
                            f"  {vals.mean():>12.2f}"
                            f"  {vals.std():>12.2f}"
                            f"  {vals.min():>12.2f}"
                            f"  {vals.max():>12.2f}"
                            f"  {miss_pct:>6.2f}%\n")
                else:
                    f.write(f"  {col:<25s}  {'N/A':>12s}  {'N/A':>12s}"
                            f"  {'N/A':>12s}  {'N/A':>12s}  {miss_pct:>6.2f}%\n")

        # File size
        f.write(f"\n\nOUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        fsize = os.path.getsize(FEATURE_OUTPUT_PATH)
        f.write(f"  feature_matrix.parquet  — {len(fm):,} rows,"
                f" {fsize / 1e6:.1f} MB\n")
        f.write(f"  step4_feature_summary.txt  — this file\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {SUMMARY_PATH}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 4: Feature Engineering — START")

    # Phase 1: Load sales
    sales = load_sales()

    # Build sale lookup for saleitems join: (server_id, shop_id, sale_id) → (company_id, opt_on)
    sale_lookup = sales[["server_id", "shop_id", "id", "company_id", "opt_on"]].copy()
    sale_lookup.rename(columns={"id": "sale_id"}, inplace=True)

    # Phase 2: Pre-aggregate saleitems
    sale_stats, cust_products = preaggregate_saleitems(sale_lookup)
    del sale_lookup

    # Phase 3: Load batch assignments
    log("Phase 3: Loading batch assignments...")
    assignments = pd.read_parquet(BATCH_ASSIGN_PATH)
    prediction_points = sorted(assignments["prediction_point"].unique())
    log(f"  {len(prediction_points)} prediction points,"
        f" {len(assignments):,} total assignments")

    # Phase 4: Compute features per batch
    all_features = []
    for i, pp in enumerate(prediction_points):
        log(f"Batch {i + 1}/{len(prediction_points)}: {pp.date()}")
        eligible = assignments[assignments["prediction_point"] == pp]
        features = compute_batch_features(sales, sale_stats, cust_products,
                                          eligible, pp)
        all_features.append(features)
        log(f"  → {len(features):,} rows")

    # Phase 5: Combine and save
    log("Combining all batches...")
    feature_matrix = pd.concat(all_features, ignore_index=True)
    del all_features

    # Reorder columns
    key_cols = ["server_id", "shop_id", "company_id", "prediction_point"]
    customer_features = [
        "recency", "frequency", "monetary", "avg_monetary",
        "ipi_mean", "ipi_std", "recent_activity", "freq_decline_ratio",
        "unique_products", "avg_basket_size", "total_qty",
        "pay_ratio", "debt_frequency", "customer_tenure",
    ]
    store_features = [
        "store_txn_volume", "store_active_customers", "store_median_ipi",
        "recency_vs_store", "frequency_vs_store", "monetary_vs_store",
    ]
    time_features = [
        "month", "quarter", "is_cny_season", "is_double11", "is_spring_festival",
    ]
    col_order = key_cols + customer_features + store_features + time_features
    feature_matrix = feature_matrix[col_order]

    log(f"Saving feature matrix: {len(feature_matrix):,} rows,"
        f" {len(feature_matrix.columns)} columns...")
    feature_matrix.to_parquet(FEATURE_OUTPUT_PATH, index=False)
    log(f"  Saved to {FEATURE_OUTPUT_PATH}")

    # Summary report
    write_summary(feature_matrix, prediction_points)

    log("Step 4: Feature Engineering — COMPLETE")


if __name__ == "__main__":
    main()
