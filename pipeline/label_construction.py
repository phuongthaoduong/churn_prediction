"""
Step 5: Label Construction
============================
Creates churn labels under two schemes for each
(server_id, shop_id, company_id, prediction_point) tuple.

L1 (Fixed threshold):
  Churned if no purchase in [pp, pp + 30 days).

L2 (Category-level dynamic threshold):
  For each industry category c, pool ALL inter-purchase intervals
  observed up to the prediction point. Set tau_c = P95 of pooled IPIs.
  Global fallback for categories with < MIN_CATEGORY_IPIS.
  Churned if no purchase in [pp, pp + tau_c).
  tau_c is capped at MAX_LABEL_DAYS (90).

Input:
  - srv_sales_step2_3.parquet          (cleaned sales)
  - batch_assignments.parquet          (customer-batch assignments)
  - sale_data/industry_servereid.csv   (server_id → industry category)

Output:
  - label_table.parquet                (labels per customer × batch)
  - l2_thresholds.csv                  (per-category thresholds per batch)
  - step5_label_summary.txt            (human-readable report)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROFILING_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
STEP2_DIR = os.path.join(PROFILING_DIR, "step2_cleaning")
STEP3_DIR = os.path.join(PROFILING_DIR, "step3_temporal")
OUTPUT_DIR = os.path.join(PROFILING_DIR, "step5_labels")

SALES_PATH = os.path.join(STEP2_DIR, "srv_sales_step2_3.parquet")
BATCH_ASSIGN_PATH = os.path.join(STEP3_DIR, "batch_assignments.parquet")
INDUSTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sale_data", "industry_servereid.csv")

LABEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "label_table.parquet")
L2_THRESH_PATH = os.path.join(OUTPUT_DIR, "l2_thresholds.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "step5_label_summary.txt")

L1_LABEL_DAYS = 30      # fixed threshold
MAX_LABEL_DAYS = 90      # cap for L2 thresholds
L2_PERCENTILE = 95       # P95 of pooled IPIs
MIN_CATEGORY_IPIS = 100  # fallback if category has fewer pooled IPIs

CUST_COLS = ["server_id", "shop_id", "company_id"]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# Phase 1: Load data
# ===================================================================
def load_sales_dates():
    """Load (customer_key, opt_on) from cleaned sales."""
    log("Loading sales dates...")
    sales = pd.read_parquet(SALES_PATH, columns=[
        "server_id", "shop_id", "id", "company_id", "opt_on",
    ])
    sales["shop_id"] = sales["shop_id"].astype("int64")
    sales["opt_on"] = pd.to_datetime(sales["opt_on"])
    log(f"  {len(sales):,} sales rows loaded")
    return sales


def load_industry_mapping():
    """Load server_id → industry category mapping."""
    log("Loading industry mapping...")
    ind = pd.read_csv(INDUSTRY_PATH, usecols=["industry", "serverid"])
    # serverid is int64 in CSV, server_id is string in sales
    ind["server_id"] = ind["serverid"].astype(str)
    mapping = dict(zip(ind["server_id"], ind["industry"]))
    log(f"  {len(mapping)} server_id → category mappings"
        f" ({ind['industry'].nunique()} categories)")
    return mapping


# ===================================================================
# Phase 2: Pre-compute all inter-purchase intervals
# ===================================================================
def compute_all_ipis(sales):
    """Compute all inter-purchase intervals with their end dates.

    For each customer, sorts unique purchase dates and computes consecutive
    diffs. Returns DataFrame with (server_id, shop_id, company_id,
    ipi_days, end_date) where end_date is the later purchase date.
    """
    log("Computing all inter-purchase intervals...")

    # Get unique (customer, date) pairs
    cust_dates = (
        sales[CUST_COLS + ["opt_on"]]
        .drop_duplicates()
        .sort_values(CUST_COLS + ["opt_on"])
    )
    log(f"  {len(cust_dates):,} unique (customer, date) pairs")

    # Compute consecutive diffs within each customer
    cust_dates["prev_date"] = cust_dates.groupby(CUST_COLS)["opt_on"].shift(1)
    ipis = cust_dates.dropna(subset=["prev_date"]).copy()
    ipis["ipi_days"] = (ipis["opt_on"] - ipis["prev_date"]).dt.days
    ipis.rename(columns={"opt_on": "end_date"}, inplace=True)
    ipis = ipis[CUST_COLS + ["ipi_days", "end_date"]]

    log(f"  {len(ipis):,} inter-purchase intervals computed")
    log(f"  IPI stats: mean={ipis['ipi_days'].mean():.1f},"
        f" median={ipis['ipi_days'].median():.0f},"
        f" P95={ipis['ipi_days'].quantile(0.95):.0f}")

    return ipis


# ===================================================================
# Phase 3: Compute L2 thresholds per category per prediction point
# ===================================================================
def compute_l2_thresholds(ipis, industry_map, prediction_points):
    """Compute per-category P95 thresholds for each prediction point.

    Uses only IPIs with end_date < prediction_point (no future leakage).

    Returns:
        thresholds_df: DataFrame (prediction_point, category, tau_c,
                        n_ipis, tier)
        tau_per_pp: dict[pp] → dict[category] → tau_c (capped)
    """
    log("Computing L2 thresholds per category per prediction point...")

    # Map each IPI to its industry category
    ipis = ipis.copy()
    ipis["category"] = ipis["server_id"].map(industry_map)
    # Drop IPIs for merchants not in industry mapping (2 merchants)
    n_unmapped = ipis["category"].isna().sum()
    if n_unmapped > 0:
        log(f"  Warning: {n_unmapped:,} IPIs from unmapped merchants (dropped)")
        ipis = ipis.dropna(subset=["category"])

    threshold_rows = []
    tau_per_pp = {}

    for pp in prediction_points:
        # Filter to IPIs observed before this prediction point
        valid = ipis[ipis["end_date"] < pp]

        # Per-category P95
        cat_stats = valid.groupby("category")["ipi_days"].agg(
            tau_raw=lambda x: np.percentile(x, L2_PERCENTILE),
            n_ipis="count",
        ).reset_index()

        # Global fallback
        tau_global = np.percentile(valid["ipi_days"], L2_PERCENTILE) if len(valid) > 0 else L1_LABEL_DAYS
        tau_global_capped = min(tau_global, MAX_LABEL_DAYS)

        # Assign tier and cap
        pp_thresholds = {}
        for _, row in cat_stats.iterrows():
            cat = row["category"]
            if row["n_ipis"] >= MIN_CATEGORY_IPIS:
                tau = min(row["tau_raw"], MAX_LABEL_DAYS)
                tier = "category"
            else:
                tau = tau_global_capped
                tier = "global"
            pp_thresholds[cat] = tau
            threshold_rows.append({
                "prediction_point": pp,
                "category": cat,
                "tau_c": tau,
                "tau_raw": row["tau_raw"],
                "n_ipis": int(row["n_ipis"]),
                "tier": tier,
            })

        # Store global fallback for unmapped categories
        pp_thresholds["__global__"] = tau_global_capped
        tau_per_pp[pp] = pp_thresholds

        n_global = sum(1 for r in threshold_rows
                       if r["prediction_point"] == pp and r["tier"] == "global")
        log(f"  {pp.date()}: global_tau={tau_global_capped:.0f}d,"
            f" {len(cat_stats)} categories"
            f" ({n_global} using global fallback)")

    thresholds_df = pd.DataFrame(threshold_rows)
    return thresholds_df, tau_per_pp


# ===================================================================
# Phase 4: Construct labels
# ===================================================================
def construct_labels(sales, assignments, tau_per_pp, industry_map,
                     prediction_points):
    """Construct L1 and L2 labels for all batch assignments."""
    log("Constructing labels...")

    all_labels = []

    for i, pp in enumerate(prediction_points):
        log(f"  Batch {i + 1}/{len(prediction_points)}: {pp.date()}")

        eligible = assignments[assignments["prediction_point"] == pp][CUST_COLS].copy()

        # Get all sales in the maximum label window [pp, pp + MAX_LABEL_DAYS)
        label_end_max = pp + pd.Timedelta(days=MAX_LABEL_DAYS)
        label_sales = sales[
            (sales["opt_on"] >= pp) & (sales["opt_on"] < label_end_max)
        ]

        # First repurchase date per customer after prediction point
        first_repurchase = (
            label_sales
            .groupby(CUST_COLS)["opt_on"]
            .min()
            .reset_index()
            .rename(columns={"opt_on": "first_repurchase"})
        )

        # Merge with eligible customers
        labels = eligible.merge(first_repurchase, on=CUST_COLS, how="left")

        # --- L1: Fixed threshold (30 days) ---
        l1_end = pp + pd.Timedelta(days=L1_LABEL_DAYS)
        labels["label_L1"] = (
            labels["first_repurchase"].isna() | (labels["first_repurchase"] >= l1_end)
        ).astype("int8")

        # --- L2: Category-level dynamic threshold ---
        pp_tau = tau_per_pp[pp]
        tau_global = pp_tau["__global__"]

        # Map each customer to their category's threshold
        labels["category"] = labels["server_id"].map(industry_map)
        labels["L2_threshold_days"] = labels["category"].map(
            lambda c: pp_tau.get(c, tau_global) if pd.notna(c) else tau_global
        )

        # Determine tier
        labels["L2_tier"] = labels["category"].map(
            lambda c: "category" if pd.notna(c) and c in pp_tau else "global"
        )

        # L2 label
        l2_ends = pp + pd.to_timedelta(labels["L2_threshold_days"], unit="D")
        labels["label_L2"] = (
            labels["first_repurchase"].isna() | (labels["first_repurchase"] >= l2_ends)
        ).astype("int8")

        labels["prediction_point"] = pp
        labels.drop(columns=["first_repurchase"], inplace=True)

        all_labels.append(labels)

        # Log batch stats
        n = len(labels)
        l1_rate = labels["label_L1"].mean() * 100
        l2_rate = labels["label_L2"].mean() * 100
        log(f"    L1 churn: {l1_rate:.1f}%  |  L2 churn: {l2_rate:.1f}%"
            f"  |  {n:,} customers")

    label_df = pd.concat(all_labels, ignore_index=True)
    return label_df


# ===================================================================
# Summary report
# ===================================================================
def write_summary(label_df, thresholds_df, prediction_points):
    """Write human-readable summary."""
    with open(SUMMARY_PATH, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 5: LABEL CONSTRUCTION\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write(f"  L1 fixed threshold:   {L1_LABEL_DAYS} days\n")
        f.write(f"  L2 percentile:        P{L2_PERCENTILE}\n")
        f.write(f"  L2 max cap:           {MAX_LABEL_DAYS} days\n")
        f.write(f"  Min category IPIs:    {MIN_CATEGORY_IPIS}\n")
        f.write(f"  Total assignments:    {len(label_df):,}\n\n")

        # --- Per-batch churn rates ---
        f.write("PER-BATCH CHURN RATES\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Prediction':>12s}  {'Eligible':>10s}"
                f"  {'L1 Churn%':>10s}  {'L2 Churn%':>10s}"
                f"  {'Agreement%':>11s}\n")
        f.write(f"  {'─' * 12}  {'─' * 10}"
                f"  {'─' * 10}  {'─' * 10}"
                f"  {'─' * 11}\n")

        for pp in prediction_points:
            batch = label_df[label_df["prediction_point"] == pp]
            n = len(batch)
            l1_rate = batch["label_L1"].mean() * 100
            l2_rate = batch["label_L2"].mean() * 100
            agree = (batch["label_L1"] == batch["label_L2"]).mean() * 100
            f.write(f"  {pp.date()!s:>12s}  {n:>10,}"
                    f"  {l1_rate:>9.1f}%  {l2_rate:>9.1f}%"
                    f"  {agree:>10.1f}%\n")

        # Overall
        l1_overall = label_df["label_L1"].mean() * 100
        l2_overall = label_df["label_L2"].mean() * 100
        agree_overall = (label_df["label_L1"] == label_df["label_L2"]).mean() * 100
        f.write(f"\n  {'Overall':>12s}  {len(label_df):>10,}"
                f"  {l1_overall:>9.1f}%  {l2_overall:>9.1f}%"
                f"  {agree_overall:>10.1f}%\n")

        # --- Label cross-tabulation ---
        f.write(f"\n\nLABEL CROSS-TABULATION\n")
        f.write("-" * 50 + "\n")
        ct = pd.crosstab(label_df["label_L1"], label_df["label_L2"],
                         margins=True)
        ct_pct = ct / len(label_df) * 100
        f.write(f"  {'':>16s}  {'L2=0':>10s}  {'L2=1':>10s}  {'Total':>10s}\n")
        for l1_val in [0, 1]:
            row_label = f"L1={l1_val}"
            f.write(f"  {row_label:>16s}"
                    f"  {ct_pct.loc[l1_val, 0]:>9.1f}%"
                    f"  {ct_pct.loc[l1_val, 1]:>9.1f}%"
                    f"  {ct_pct.loc[l1_val, 'All']:>9.1f}%\n")
        f.write(f"  {'Total':>16s}"
                f"  {ct_pct.loc['All', 0]:>9.1f}%"
                f"  {ct_pct.loc['All', 1]:>9.1f}%"
                f"  {100.0:>9.1f}%\n")

        # --- L2 tier distribution ---
        f.write(f"\n\nL2 TIER DISTRIBUTION\n")
        f.write("-" * 50 + "\n")
        tier_counts = label_df["L2_tier"].value_counts()
        for tier, cnt in tier_counts.items():
            f.write(f"  {tier:>12s}: {cnt:>10,} ({cnt / len(label_df) * 100:.1f}%)\n")

        # --- L2 thresholds per category (latest batch) ---
        f.write(f"\n\nL2 CATEGORY THRESHOLDS (latest batch: {prediction_points[-1].date()})\n")
        f.write("-" * 70 + "\n")
        latest = thresholds_df[thresholds_df["prediction_point"] == prediction_points[-1]]
        latest = latest.sort_values("tau_c", ascending=False)
        f.write(f"  {'Category':<12s}  {'tau_c':>8s}  {'tau_raw':>10s}"
                f"  {'n_IPIs':>10s}  {'Tier':>10s}\n")
        f.write(f"  {'─' * 12}  {'─' * 8}  {'─' * 10}"
                f"  {'─' * 10}  {'─' * 10}\n")
        for _, row in latest.iterrows():
            f.write(f"  {row['category']:<12s}"
                    f"  {row['tau_c']:>7.0f}d"
                    f"  {row['tau_raw']:>9.1f}d"
                    f"  {int(row['n_ipis']):>10,}"
                    f"  {row['tier']:>10s}\n")

        # --- L2 threshold stability across batches ---
        f.write(f"\n\nL2 THRESHOLD STABILITY ACROSS BATCHES\n")
        f.write("-" * 70 + "\n")
        pivot = thresholds_df.pivot_table(
            index="category", columns="prediction_point",
            values="tau_c", aggfunc="first"
        )
        f.write(f"  {'Category':<12s}  {'Min tau':>8s}  {'Max tau':>8s}"
                f"  {'Range':>8s}  {'CV':>8s}\n")
        f.write(f"  {'─' * 12}  {'─' * 8}  {'─' * 8}"
                f"  {'─' * 8}  {'─' * 8}\n")
        for cat in sorted(pivot.index):
            vals = pivot.loc[cat].dropna()
            f.write(f"  {cat:<12s}"
                    f"  {vals.min():>7.0f}d"
                    f"  {vals.max():>7.0f}d"
                    f"  {vals.max() - vals.min():>7.0f}d"
                    f"  {vals.std() / vals.mean() * 100 if vals.mean() > 0 else 0:>7.1f}%\n")

        # Output files
        f.write(f"\n\nOUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        fsize = os.path.getsize(LABEL_OUTPUT_PATH)
        f.write(f"  label_table.parquet   — {len(label_df):,} rows,"
                f" {fsize / 1e6:.1f} MB\n")
        f.write(f"  l2_thresholds.csv     — per-category thresholds per batch\n")
        f.write(f"  step5_label_summary.txt — this file\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {SUMMARY_PATH}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 5: Label Construction — START")

    # Phase 1: Load data
    sales = load_sales_dates()
    industry_map = load_industry_mapping()

    # Phase 2: Pre-compute all IPIs
    ipis = compute_all_ipis(sales)

    # Phase 3: Load batch assignments and compute L2 thresholds
    log("Loading batch assignments...")
    assignments = pd.read_parquet(BATCH_ASSIGN_PATH)
    prediction_points = sorted(assignments["prediction_point"].unique())
    log(f"  {len(prediction_points)} prediction points")

    thresholds_df, tau_per_pp = compute_l2_thresholds(
        ipis, industry_map, prediction_points
    )
    del ipis  # free memory

    # Phase 4: Construct labels
    label_df = construct_labels(
        sales, assignments, tau_per_pp, industry_map, prediction_points
    )
    del sales  # free memory

    # Reorder columns
    col_order = CUST_COLS + [
        "prediction_point", "label_L1", "label_L2",
        "L2_tier", "L2_threshold_days", "category",
    ]
    label_df = label_df[col_order]

    # Phase 5: Save
    log("Saving outputs...")
    label_df.to_parquet(LABEL_OUTPUT_PATH, index=False)
    log(f"  label_table.parquet: {len(label_df):,} rows")

    thresholds_df.to_csv(L2_THRESH_PATH, index=False)
    log(f"  l2_thresholds.csv: {len(thresholds_df)} rows")

    # Summary report
    write_summary(label_df, thresholds_df, prediction_points)

    log("Step 5: Label Construction — COMPLETE")


if __name__ == "__main__":
    main()
