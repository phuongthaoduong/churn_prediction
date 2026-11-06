"""Step 1: cheap pilot for Option (a) (LLM as behavioral-sequence encoder).

Instead of building full embedding-extraction + PCA/autoencoder-control
infrastructure upfront, this tests the core hypothesis directly: does an
LLM reasoning over a customer's RAW transaction sequence (via chat
completions, no embeddings needed) produce real predictive signal, and
does that signal depend on actual sequence ORDER (not just aggregate
stats)? If not, there's no point building the full encoder pipeline.

Samples 150 customers with "sufficient history" (frequency_value >= 5 in
the 90-day observation window, from pipeline/shap_analysis.py's existing
sample) stratified across risk levels, pulls each one's FULL real
transaction history up to the prediction point from the cleaned sales
data, and builds both a real-chronological-order and a shuffled-order
serialization. The LLM never sees merchant/customer identity — only
(days_before_prediction_point, bill_amt, pay_amt, owe_amt) tuples.

Output: data/pilot_examples.json
"""
import json

import numpy as np
import pandas as pd

SCHEME = "L2"
N_SAMPLE = 250
MIN_FREQ = 5
MAX_TXNS = 100
SEED = 42

EXP = "/Users/phuongthaoduong/Project/Data_Processing/experiment_output"
CLEANED_SALES = "/Users/phuongthaoduong/Project/Data_Processing/profiling_output/step2_cleaning/srv_sales_step2_3.parquet"


def main():
    shap_df = pd.read_parquet(f"{EXP}/shap_values_{SCHEME}.parquet")
    eval_labels = pd.read_parquet(f"{EXP}/eval_labels.parquet")
    key_cols = ["server_id", "shop_id", "company_id", "prediction_point"]
    shap_df = shap_df.merge(eval_labels[key_cols + ["label_eval"]],
                            on=key_cols, how="left", validate="1:1")

    pool = shap_df[shap_df["frequency_value"] >= MIN_FREQ].copy()
    n = len(pool)
    # Simple uniform random sample (NOT stratified by xgb_y_prob — that would
    # make the "existing XGBoost AUC on the same customers" comparison unfair,
    # since the model's own score would have been used to pick who's in the
    # sample). Positives are naturally ~8% of this pool, so a 150-row random
    # sample yields a similar few-positives rate without needing to force it.
    chosen = pool.sample(n=min(N_SAMPLE, n), random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(chosen)} customers with frequency>=5 "
          f"(simple random sample from a pool of {n})")

    sales = pd.read_parquet(CLEANED_SALES,
                            columns=["server_id", "shop_id", "company_id",
                                     "opt_on", "bill_amt", "pay_amt", "owe_amt"])

    examples = []
    rng2 = np.random.default_rng(SEED)
    for i, row in chosen.iterrows():
        pp = pd.Timestamp(row["prediction_point"])
        cust = sales[(sales.server_id == row["server_id"]) &
                     (sales.shop_id == row["shop_id"]) &
                     (sales.company_id == row["company_id"]) &
                     (sales.opt_on <= pp)].sort_values("opt_on")
        if len(cust) < MIN_FREQ:
            continue
        txns = [{"days_before_pp": int((pp - r.opt_on).days),
                 "bill_amt": round(float(r.bill_amt), 2),
                 "pay_amt": round(float(r.pay_amt), 2),
                 "owe_amt": round(float(r.owe_amt), 2)}
                for r in cust.itertuples()]
        n_total = len(txns)
        txns = txns[-MAX_TXNS:]  # most recent MAX_TXNS, still chronological
        shuffled = txns.copy()
        rng2.shuffle(shuffled)
        examples.append({
            "example_id": f"{SCHEME}_{i}",
            "n_txns": len(txns),
            "n_txns_total_lifetime": n_total,
            "label_eval": (None if pd.isna(row["label_eval"]) else int(row["label_eval"])),
            "xgb_y_prob": round(float(row["y_prob"]), 4),
            "chronological": txns,
            "shuffled": shuffled,
        })

    with open("data/pilot_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    n_valid = sum(1 for e in examples if e["label_eval"] is not None)
    print(f"{len(examples)} examples written to data/pilot_examples.json "
          f"({n_valid} with a valid shared eval label)")


if __name__ == "__main__":
    main()
