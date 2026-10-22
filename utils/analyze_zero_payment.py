"""
Zero-Payment Analysis: User-Level Aggregation and Segmentation
===============================================================
Analyzes the distribution of zero pay_amt transactions at the user level
to estimate the likelihood of two causes:
  1. Data entry omission (user didn't record payment)
  2. Credit/accounts receivable (downstream customer took goods on credit)

Uses chunked reading to handle the full 2.1 GB srv_sales.csv.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sale_data")
SALES_PATH = os.path.join(DATA_DIR, "srv_sales.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling_output", "step1_profiling")
CHUNK_SIZE = 500_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f"]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    log("Starting zero-payment analysis...")

    # -----------------------------------------------------------------------
    # Pass 1: Aggregate per user (chunked)
    # -----------------------------------------------------------------------
    log("Pass 1: Aggregating per user from srv_sales.csv...")

    # Accumulators: user_key -> [total_txn, zero_pay_txn, total_bill_amt]
    user_total = {}
    user_zero = {}
    user_bill = {}

    for chunk_idx, chunk in enumerate(pd.read_csv(
        SALES_PATH,
        chunksize=CHUNK_SIZE,
        dtype={"server_id": str, "shop_id": "int32", "company_id": "int64"},
        usecols=["server_id", "shop_id", "company_id", "pay_amt", "bill_amt"],
        low_memory=False,
    )):
        if chunk_idx % 5 == 0:
            log(f"  Chunk {chunk_idx}, users so far: {len(user_total):,}")

        # Create user key
        keys = list(zip(chunk["server_id"], chunk["shop_id"], chunk["company_id"]))
        is_zero = (chunk["pay_amt"] == 0).values
        bill_vals = chunk["bill_amt"].values

        for i, key in enumerate(keys):
            user_total[key] = user_total.get(key, 0) + 1
            if is_zero[i]:
                user_zero[key] = user_zero.get(key, 0) + 1
            bill = bill_vals[i]
            if not np.isnan(bill):
                user_bill[key] = user_bill.get(key, 0.0) + bill

    log(f"  Total unique users: {len(user_total):,}")

    # -----------------------------------------------------------------------
    # Build user-level DataFrame
    # -----------------------------------------------------------------------
    log("Building user-level DataFrame...")

    all_keys = list(user_total.keys())
    df = pd.DataFrame({
        "server_id": [k[0] for k in all_keys],
        "shop_id": [k[1] for k in all_keys],
        "company_id": [k[2] for k in all_keys],
        "total_txn": [user_total[k] for k in all_keys],
        "zero_pay_txn": [user_zero.get(k, 0) for k in all_keys],
        "total_bill_amt": [user_bill.get(k, 0.0) for k in all_keys],
    })

    df["zero_pay_ratio"] = df["zero_pay_txn"] / df["total_txn"]
    df["nonzero_txn"] = df["total_txn"] - df["zero_pay_txn"]

    # -----------------------------------------------------------------------
    # Segmentation
    # -----------------------------------------------------------------------
    log("Segmenting users...")

    def classify(ratio):
        if ratio == 1.0:
            return "Strictly Zero (100% zero-pay)"
        elif ratio == 0.0:
            return "Strictly Non-Zero (0% zero-pay)"
        else:
            return "Mixed (partial zero-pay)"

    df["segment"] = df["zero_pay_ratio"].apply(classify)

    total_users = len(df)

    # Segment stats
    seg_order = [
        "Strictly Zero (100% zero-pay)",
        "Mixed (partial zero-pay)",
        "Strictly Non-Zero (0% zero-pay)",
    ]
    seg_stats = []
    for seg in seg_order:
        subset = df[df["segment"] == seg]
        seg_stats.append({
            "segment": seg,
            "user_count": len(subset),
            "user_pct": len(subset) / total_users * 100,
            "total_txn": subset["total_txn"].sum(),
            "txn_pct": subset["total_txn"].sum() / df["total_txn"].sum() * 100,
            "median_txn": subset["total_txn"].median(),
            "mean_txn": subset["total_txn"].mean(),
            "total_bill": subset["total_bill_amt"].sum(),
            "median_bill_per_user": subset["total_bill_amt"].median(),
        })

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ZERO-PAYMENT ANALYSIS: USER-LEVEL SEGMENTATION")
    print("=" * 90)

    print(f"\nTotal users analyzed: {total_users:,}")
    print(f"Total transactions:  {df['total_txn'].sum():,}")
    print(f"Total zero-pay txn:  {df['zero_pay_txn'].sum():,} "
          f"({df['zero_pay_txn'].sum()/df['total_txn'].sum()*100:.1f}%)")

    print(f"\n{'Segment':<40} {'Users':>10} {'%':>8} {'Txns':>12} "
          f"{'Txn%':>8} {'Med Txn':>9} {'Mean Txn':>10}")
    print("-" * 90)
    for s in seg_stats:
        print(f"{s['segment']:<40} {s['user_count']:>10,} {s['user_pct']:>7.1f}% "
              f"{s['total_txn']:>12,} {s['txn_pct']:>7.1f}% "
              f"{s['median_txn']:>9.0f} {s['mean_txn']:>10.1f}")

    # -----------------------------------------------------------------------
    # Deeper analysis of Mixed users
    # -----------------------------------------------------------------------
    mixed = df[df["segment"] == "Mixed (partial zero-pay)"]
    if len(mixed) > 0:
        print(f"\n{'=' * 90}")
        print("MIXED USERS: ZERO-PAY RATIO DISTRIBUTION")
        print(f"{'=' * 90}")
        print(f"\nTotal mixed users: {len(mixed):,}")
        ratio_percentiles = [10, 25, 50, 75, 90]
        for p in ratio_percentiles:
            val = np.percentile(mixed["zero_pay_ratio"], p)
            print(f"  P{p:<3}: {val:.2%}")
        print(f"  Mean: {mixed['zero_pay_ratio'].mean():.2%}")

        # Sub-segment mixed users
        mixed_high = mixed[mixed["zero_pay_ratio"] >= 0.8]
        mixed_mid = mixed[(mixed["zero_pay_ratio"] >= 0.2) & (mixed["zero_pay_ratio"] < 0.8)]
        mixed_low = mixed[mixed["zero_pay_ratio"] < 0.2]
        print(f"\n  Mixed sub-segments:")
        print(f"    High ratio (≥80% zero): {len(mixed_high):>10,} ({len(mixed_high)/total_users*100:.1f}%)")
        print(f"    Mid ratio  (20-80%):    {len(mixed_mid):>10,} ({len(mixed_mid)/total_users*100:.1f}%)")
        print(f"    Low ratio  (<20% zero): {len(mixed_low):>10,} ({len(mixed_low)/total_users*100:.1f}%)")

    # -----------------------------------------------------------------------
    # Analysis of Strictly Zero users: single vs multi transaction
    # -----------------------------------------------------------------------
    strictly_zero = df[df["segment"] == "Strictly Zero (100% zero-pay)"]
    if len(strictly_zero) > 0:
        print(f"\n{'=' * 90}")
        print("STRICTLY ZERO USERS: TRANSACTION COUNT DISTRIBUTION")
        print(f"{'=' * 90}")
        sz_single = strictly_zero[strictly_zero["total_txn"] == 1]
        sz_multi = strictly_zero[strictly_zero["total_txn"] > 1]
        print(f"\n  Single-transaction (1 txn):  {len(sz_single):>10,} "
              f"({len(sz_single)/len(strictly_zero)*100:.1f}% of strictly-zero)")
        print(f"  Multi-transaction  (2+ txn): {len(sz_multi):>10,} "
              f"({len(sz_multi)/len(strictly_zero)*100:.1f}% of strictly-zero)")
        if len(sz_multi) > 0:
            print(f"\n  Multi-transaction strictly-zero user stats:")
            print(f"    Median txn count: {sz_multi['total_txn'].median():.0f}")
            print(f"    Mean txn count:   {sz_multi['total_txn'].mean():.1f}")
            print(f"    Max txn count:    {sz_multi['total_txn'].max():,}")
            print(f"    Median bill_amt:  {sz_multi['total_bill_amt'].median():,.0f}")

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    sz_pct = seg_stats[0]["user_pct"]
    mixed_pct = seg_stats[1]["user_pct"]
    nz_pct = seg_stats[2]["user_pct"]

    print(f"\n{'=' * 90}")
    print("INTERPRETATION")
    print(f"{'=' * 90}")
    print(f"""
  Strictly Zero users ({sz_pct:.1f}%):
    These users NEVER have a recorded payment across ALL their transactions.
    - If single-transaction: could be either data omission or credit sale (ambiguous).
    - If multi-transaction with consistent zero-pay: strongly suggests systematic
      credit/accounts receivable arrangements, not random data entry omission.
      A user who forgets to enter payment once is plausible; a user who forgets
      every single time across many transactions is not.

  Mixed users ({mixed_pct:.1f}%):
    These users have SOME zero and SOME non-zero payments.
    - High ratio (≥80% zero): likely credit accounts with occasional cash payments.
    - Low ratio (<20% zero): likely data entry omissions on a few transactions.
    - This segment is strong evidence that BOTH causes coexist in the data.

  Strictly Non-Zero users ({nz_pct:.1f}%):
    These users always have recorded payments. No action needed for this group.
""")

    # -----------------------------------------------------------------------
    # Charts
    # -----------------------------------------------------------------------
    log("Generating charts...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Segment pie chart
    seg_counts = [s["user_count"] for s in seg_stats]
    seg_labels = [
        f"Strictly Zero\n({seg_stats[0]['user_pct']:.1f}%)",
        f"Mixed\n({seg_stats[1]['user_pct']:.1f}%)",
        f"Strictly Non-Zero\n({seg_stats[2]['user_pct']:.1f}%)",
    ]
    axes[0, 0].pie(seg_counts, labels=seg_labels, colors=[COLORS[2], COLORS[1], COLORS[0]],
                   autopct=lambda p: f"{p*total_users/100:,.0f}", startangle=90,
                   textprops={"fontsize": 9})
    axes[0, 0].set_title("User Segmentation by Zero-Pay Pattern", fontsize=11, fontweight="bold")

    # 2. Transaction share by segment
    seg_txns = [s["total_txn"] for s in seg_stats]
    seg_txn_labels = [
        f"Strictly Zero\n({seg_stats[0]['txn_pct']:.1f}%)",
        f"Mixed\n({seg_stats[1]['txn_pct']:.1f}%)",
        f"Strictly Non-Zero\n({seg_stats[2]['txn_pct']:.1f}%)",
    ]
    axes[0, 1].pie(seg_txns, labels=seg_txn_labels, colors=[COLORS[2], COLORS[1], COLORS[0]],
                   autopct=lambda p: f"{p*df['total_txn'].sum()/100:,.0f}", startangle=90,
                   textprops={"fontsize": 9})
    axes[0, 1].set_title("Transaction Share by Segment", fontsize=11, fontweight="bold")

    # 3. Zero-pay ratio histogram (mixed users only)
    if len(mixed) > 0:
        axes[1, 0].hist(mixed["zero_pay_ratio"], bins=50, color=COLORS[1],
                        alpha=0.8, edgecolor="white")
        axes[1, 0].set_title("Zero-Pay Ratio Distribution (Mixed Users)", fontsize=11, fontweight="bold")
        axes[1, 0].set_xlabel("Zero-Pay Ratio")
        axes[1, 0].set_ylabel("Number of Users")
        axes[1, 0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        axes[1, 0].axvline(0.2, color="green", linestyle="--", alpha=0.7, label="20% threshold")
        axes[1, 0].axvline(0.8, color="red", linestyle="--", alpha=0.7, label="80% threshold")
        axes[1, 0].legend(fontsize=8)

    # 4. Strictly-zero: single vs multi transaction
    if len(strictly_zero) > 0:
        sz_cats = ["Single txn\n(ambiguous)", "Multi txn\n(likely credit)"]
        sz_vals = [len(sz_single), len(sz_multi)]
        bars = axes[1, 1].bar(sz_cats, sz_vals, color=[COLORS[3], COLORS[2]], width=0.5)
        axes[1, 1].set_title("Strictly-Zero Users: Single vs Multi-Txn", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Number of Users")
        axes[1, 1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        for bar, val in zip(bars, sz_vals):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sz_vals) * 0.01,
                            f"{val:,}\n({val/len(strictly_zero)*100:.1f}%)",
                            ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, "zero_payment_analysis.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"Chart saved to {chart_path}")

    # -----------------------------------------------------------------------
    # Save detailed CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(OUTPUT_DIR, "zero_payment_user_segments.csv")
    df.to_csv(csv_path, index=False)
    log(f"User-level data saved to {csv_path}")

    log("Zero-payment analysis complete.")


if __name__ == "__main__":
    main()
