"""
Step 3: Temporal Structuring
=============================
Partitions cleaned transaction data into monthly batches with strict
observation/label window separation.

For each prediction point (1st of each month):
  - Observation window: [t - 90 days, t)     → features computed here
  - Label window:       [t, t + label_days)  → churn label computed here

A customer is eligible for batch t if they have >= 1 purchase in the
observation window. Batches whose label window exceeds the data boundary
(2026-05-31) are excluded.

Input:
  - srv_sales_step2_3.parquet (cleaned sales)

Output:
  - batch_assignments.parquet:         (customer_key, prediction_point, n_purchases_in_obs)
  - batch_metadata.csv:                per-batch summary (window dates, eligible counts)
  - step3_temporal_summary.txt:        human-readable report
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output")
SALES_PATH = os.path.join(OUTPUT_DIR, "srv_sales_step2_3.parquet")
BATCH_SIZE = 500_000

OBS_WINDOW_DAYS = 90       # observation window length
L1_LABEL_DAYS = 30         # fixed threshold label window
MAX_LABEL_DAYS = 90        # maximum label window (caps L2); determines last usable batch

os.makedirs(OUTPUT_DIR, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ===================================================================
# PASS 1: Load customer purchase dates
# ===================================================================
def load_purchase_dates():
    """Read all (customer_key, opt_on) from cleaned sales."""
    log("PASS 1: Loading customer purchase dates...")

    # customer_key -> set of unique purchase dates
    customer_dates = defaultdict(set)
    date_min = None
    date_max = None

    pf = pq.ParquetFile(SALES_PATH)
    total_rows = 0

    for batch_idx, batch in enumerate(pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["server_id", "shop_id", "company_id", "opt_on"],
    )):
        chunk = batch.to_pandas()
        total_rows += len(chunk)
        if batch_idx % 5 == 0:
            log(f"  Batch {batch_idx}, rows: {total_rows:,}")

        # Track data boundaries
        cmin = chunk["opt_on"].min()
        cmax = chunk["opt_on"].max()
        if date_min is None or cmin < date_min:
            date_min = cmin
        if date_max is None or cmax > date_max:
            date_max = cmax

        # Accumulate per-customer dates
        for srv, shop, cid, dt in zip(
            chunk["server_id"], chunk["shop_id"],
            chunk["company_id"], chunk["opt_on"],
        ):
            customer_dates[(srv, shop, cid)].add(dt)

    log(f"  Total rows: {total_rows:,}")
    log(f"  Unique customers: {len(customer_dates):,}")
    log(f"  Data range: {date_min.date()} to {date_max.date()}")

    return customer_dates, date_min, date_max


# ===================================================================
# Generate prediction points
# ===================================================================
def generate_prediction_points(date_min, date_max):
    """Generate monthly prediction points (1st of each month).

    First prediction point: earliest month where a full 90-day observation
    window fits within the data.
    Last prediction point: latest month where the maximum label window
    fits before the data boundary.
    """
    # First: need 90 days before prediction point
    first_possible = date_min + pd.Timedelta(days=OBS_WINDOW_DAYS)
    # Round up to 1st of next month
    if first_possible.day > 1:
        first_pred = (first_possible + relativedelta(months=1)).replace(day=1)
    else:
        first_pred = first_possible.replace(day=1)

    # Last: need MAX_LABEL_DAYS after prediction point
    last_possible = date_max - pd.Timedelta(days=MAX_LABEL_DAYS)
    # Round down to 1st of that month
    last_pred = last_possible.replace(day=1)

    # Generate monthly sequence
    points = []
    current = first_pred
    while current <= last_pred:
        points.append(pd.Timestamp(current))
        current += relativedelta(months=1)

    log(f"  Prediction points: {len(points)}"
        f" ({points[0].date()} to {points[-1].date()})")

    return points


# ===================================================================
# Assign customers to batches
# ===================================================================
def assign_batches(customer_dates, prediction_points):
    """For each prediction point, find eligible customers."""
    log("Assigning customers to batches...")

    assignments = []

    for pp in prediction_points:
        obs_start = pp - pd.Timedelta(days=OBS_WINDOW_DAYS)
        obs_end = pp  # exclusive

        eligible_count = 0
        for cust_key, dates in customer_dates.items():
            # Count purchases in observation window [obs_start, obs_end)
            n_in_obs = sum(1 for d in dates if obs_start <= d < obs_end)
            if n_in_obs > 0:
                eligible_count += 1
                assignments.append({
                    "server_id": cust_key[0],
                    "shop_id": cust_key[1],
                    "company_id": cust_key[2],
                    "prediction_point": pp,
                    "obs_start": obs_start,
                    "obs_end": obs_end,
                    "n_purchases_in_obs": n_in_obs,
                })

        log(f"  {pp.date()}: {eligible_count:,} eligible customers")

    df = pd.DataFrame(assignments)
    log(f"  Total batch assignments: {len(df):,}")

    return df


# ===================================================================
# Batch metadata
# ===================================================================
def compute_batch_metadata(df_assignments, prediction_points):
    """Compute per-batch summary statistics."""
    rows = []
    for pp in prediction_points:
        obs_start = pp - pd.Timedelta(days=OBS_WINDOW_DAYS)
        batch = df_assignments[df_assignments["prediction_point"] == pp]

        rows.append({
            "prediction_point": pp,
            "obs_start": obs_start,
            "obs_end": pp,
            "l1_label_end": pp + pd.Timedelta(days=L1_LABEL_DAYS),
            "max_label_end": pp + pd.Timedelta(days=MAX_LABEL_DAYS),
            "eligible_customers": len(batch),
            "median_purchases": batch["n_purchases_in_obs"].median(),
            "mean_purchases": batch["n_purchases_in_obs"].mean(),
            "pct_single_purchase": (batch["n_purchases_in_obs"] == 1).mean() * 100,
        })

    return pd.DataFrame(rows)


# ===================================================================
# Summary report
# ===================================================================
def write_summary(df_assignments, df_metadata, prediction_points,
                  date_min, date_max, n_customers):
    """Write human-readable summary."""
    txt_path = os.path.join(OUTPUT_DIR, "step3_temporal_summary.txt")

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("STEP 3: TEMPORAL STRUCTURING\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        f.write("PARAMETERS\n")
        f.write("-" * 50 + "\n")
        f.write(f"  Data span:            {date_min.date()} to {date_max.date()}\n")
        f.write(f"  Observation window:   {OBS_WINDOW_DAYS} days\n")
        f.write(f"  L1 label window:      {L1_LABEL_DAYS} days (fixed)\n")
        f.write(f"  Max label window:     {MAX_LABEL_DAYS} days (L2 cap)\n")
        f.write(f"  Prediction cadence:   Monthly (1st of each month)\n")
        f.write(f"  Total customers:      {n_customers:,}\n\n")

        f.write("PREDICTION POINTS\n")
        f.write("-" * 50 + "\n")
        f.write(f"  First: {prediction_points[0].date()}\n")
        f.write(f"  Last:  {prediction_points[-1].date()}\n")
        f.write(f"  Count: {len(prediction_points)}\n\n")

        f.write("BATCH SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Prediction':>12s}  {'Obs Start':>12s}  {'Eligible':>10s}"
                f"  {'Med Purch':>10s}  {'% Single':>9s}\n")
        f.write(f"  {'─' * 12}  {'─' * 12}  {'─' * 10}"
                f"  {'─' * 10}  {'─' * 9}\n")

        for _, row in df_metadata.iterrows():
            f.write(f"  {str(row['prediction_point'].date()):>12s}"
                    f"  {str(row['obs_start'].date()):>12s}"
                    f"  {row['eligible_customers']:>10,}"
                    f"  {row['median_purchases']:>10.1f}"
                    f"  {row['pct_single_purchase']:>8.1f}%\n")

        total_assignments = len(df_assignments)
        unique_customers = df_assignments.groupby(
            ["server_id", "shop_id", "company_id"]
        ).ngroups

        f.write(f"\n  Total batch assignments: {total_assignments:,}\n")
        f.write(f"  Unique customers across all batches: {unique_customers:,}\n")
        f.write(f"  Customers never eligible (no obs window purchase): "
                f"{n_customers - unique_customers:,}\n")

        # Batch appearance distribution
        appearances = df_assignments.groupby(
            ["server_id", "shop_id", "company_id"]
        ).size()
        f.write(f"\n  Batch appearances per customer:\n")
        for p in [25, 50, 75, 90]:
            f.write(f"    P{p}: {np.percentile(appearances, p):.0f}\n")
        f.write(f"    Mean: {appearances.mean():.1f}\n")
        f.write(f"    Max:  {appearances.max()}\n")

        f.write(f"\n\nDATA LEAKAGE SAFEGUARDS\n")
        f.write("-" * 50 + "\n")
        f.write("  - obs_end == prediction_point (exclusive upper bound)\n")
        f.write("  - Features use ONLY data in [obs_start, obs_end)\n")
        f.write("  - Labels use ONLY data in [prediction_point, label_end)\n")
        f.write("  - Batches with label window exceeding data boundary excluded\n")
        f.write(f"  - Last usable prediction point: {prediction_points[-1].date()}"
                f" (label ends {prediction_points[-1].date() + pd.Timedelta(days=MAX_LABEL_DAYS)})\n")

        f.write(f"\n\nOUTPUT FILES\n")
        f.write("-" * 50 + "\n")
        f.write("  batch_assignments.parquet  — (customer, prediction_point, n_purchases)\n")
        f.write("  batch_metadata.csv         — per-batch window dates and stats\n")
        f.write("  step3_temporal_summary.txt  — this file\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    log(f"Summary saved to {txt_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    log("Step 3: Temporal Structuring — START")

    # Load purchase dates
    customer_dates, date_min, date_max = load_purchase_dates()

    # Generate prediction points
    prediction_points = generate_prediction_points(date_min, date_max)

    # Assign customers to batches
    df_assignments = assign_batches(customer_dates, prediction_points)

    # Compute batch metadata
    df_metadata = compute_batch_metadata(df_assignments, prediction_points)

    # Save outputs
    assign_path = os.path.join(OUTPUT_DIR, "batch_assignments.parquet")
    df_assignments.to_parquet(assign_path, index=False)
    log(f"Batch assignments saved to {assign_path} ({len(df_assignments):,} rows)")

    meta_path = os.path.join(OUTPUT_DIR, "batch_metadata.csv")
    df_metadata.to_csv(meta_path, index=False)
    log(f"Batch metadata saved to {meta_path}")

    # Summary report
    write_summary(df_assignments, df_metadata, prediction_points,
                  date_min, date_max, len(customer_dates))

    log("Step 3: Temporal Structuring — COMPLETE")


if __name__ == "__main__":
    main()
