"""
Step 8.0: Shared Evaluation Label Construction
===============================================
Builds label_eval: churn = no purchase in [pp, pp + EVAL_HORIZON_DAYS) for
every (customer, prediction_point) row in dataset_final. This is the common
downstream business outcome the thesis uses as the PRIMARY basis for
comparing labeling schemes (own-label metrics are secondary diagnostics).

Validation invariant: a customer silent for 60 days was silent for 30, so
label_eval == 1 implies label_L1 == 1 (label_eval <= label_L1 elementwise).

Input:
  - profiling_output/step6_dataset/dataset_final.parquet   (keys + label_L1 for validation)
  - profiling_output/step2_cleaning/srv_sales_step2_3.parquet (cleaned sales)

Output:
  - experiment_output/eval_labels.parquet  (keys + label_eval)
  - experiment_output/step8_0_eval_label_summary.txt
"""

import os
from datetime import datetime

import pandas as pd

from experiment_config import (
    CLEANED_SALES,
    DATASET_FINAL,
    EVAL_HORIZON_DAYS,
    EVAL_LABELS_PATH,
    EXPERIMENT_DIR,
    KEY_COLS,
)


def customer_key(df):
    return (df["server_id"].astype(str) + "|"
            + df["shop_id"].astype(str) + "|"
            + df["company_id"].astype(str))


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    base = pd.read_parquet(DATASET_FINAL, columns=KEY_COLS + ["label_L1"])
    base["ckey"] = customer_key(base)

    sales = pd.read_parquet(
        CLEANED_SALES, columns=["server_id", "shop_id", "company_id", "opt_on"]
    )
    sales["opt_on"] = pd.to_datetime(sales["opt_on"])
    sales["ckey"] = customer_key(sales)

    parts = []
    for pp in sorted(base["prediction_point"].unique()):
        pp = pd.Timestamp(pp)
        window_end = pp + pd.Timedelta(days=EVAL_HORIZON_DAYS)
        active = set(
            sales.loc[(sales["opt_on"] >= pp) & (sales["opt_on"] < window_end), "ckey"]
        )
        batch = base[base["prediction_point"] == pp].copy()
        batch["label_eval"] = (~batch["ckey"].isin(active)).astype("int8")
        parts.append(batch)
        print(f"  {pp:%Y-%m-%d}: {batch['label_eval'].mean():.4f} eval churn rate")

    out = pd.concat(parts, ignore_index=True)

    # Validation: 60d churn implies 30d churn
    violations = int((out["label_eval"] > out["label_L1"]).sum())
    if violations:
        raise ValueError(
            f"{violations} rows have label_eval=1 but label_L1=0 — "
            "eval window logic is wrong"
        )

    out[KEY_COLS + ["label_eval"]].to_parquet(EVAL_LABELS_PATH, index=False)

    rates = out.groupby("prediction_point")["label_eval"].mean()
    lines = [
        "=" * 70,
        "STEP 8.0: SHARED EVALUATION LABEL (60-DAY HORIZON)",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 70,
        "",
        f"Rows labeled:        {len(out):,}",
        f"Horizon:             {EVAL_HORIZON_DAYS} days",
        f"Overall churn rate:  {out['label_eval'].mean():.4f}",
        f"L1-implication check: PASS (0 violations)",
        "",
        "Per-batch eval churn rate:",
    ] + [f"  {pd.Timestamp(pp):%Y-%m-%d}  {r:.4f}" for pp, r in rates.items()] + [
        "=" * 70,
    ]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_0_eval_label_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
