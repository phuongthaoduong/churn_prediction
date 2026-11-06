"""Step 1: cheap pilot for Option (a)'s cold-start analog-reasoner half.

For a TRUE cold-start customer, all we know is their first transaction.
We want to predict whether they'll make a second purchase within 60 days
("returned_60d") — the same direction as the project's churn framing
(churn = no purchase in the window). Ground truth is directly computable
from the raw data (did a 2nd transaction actually happen within 60 days)
— no need to simulate a time-truncation; the cleaned sales data already
gives us real future behavior to check against for every customer, cold-
start or not.

Design (matches the original falsification spec): for each TEST customer
(cold start, judged ONLY on their first-transaction signature), build an
ANALOG POOL of 20 OTHER customers from the SAME STORE, each with their
own first-transaction signature AND their real, true returned_60d
outcome. Three predictors are compared downstream:
  (a) LLM reasoning over the test customer + the 20 analogs
  (b) cohort-mean: the plain proportion of the 20 analogs who returned
  (c) population base rate (no store info at all)
Failure condition (pre-registered): if (a) doesn't beat (b), the LLM is
just reproducing a GROUP BY, not adding real reasoning value.

Output: data/examples.json
"""
import json

import numpy as np
import pandas as pd

N_TEST = 150
N_ANALOGS = 20
MIN_STORE_CUSTOMERS = 30
SEED = 42
OBS_CUTOFF = pd.Timestamp("2026-03-31")  # need 60d after first txn observable

SALES_PATH = "/Users/phuongthaoduong/Project/Data_Processing/profiling_output/step2_cleaning/srv_sales_step2_3.parquet"


def main():
    sales = pd.read_parquet(SALES_PATH,
                            columns=["server_id", "shop_id", "company_id",
                                     "opt_on", "bill_amt", "pay_amt", "owe_amt"])
    sales = sales.sort_values(["server_id", "shop_id", "company_id", "opt_on"])
    g = sales.groupby(["server_id", "shop_id", "company_id"])
    first = g.nth(0).set_index(["server_id", "shop_id", "company_id"])
    second = g.nth(1).set_index(["server_id", "shop_id", "company_id"])[["opt_on"]]
    merged = first.join(second, rsuffix="_2", how="left")
    merged["gap_days"] = (merged["opt_on_2"] - merged["opt_on"]).dt.days
    merged["returned_60d"] = (merged["gap_days"] <= 60).fillna(False)
    merged = merged.reset_index()
    pop_base_rate = merged.loc[merged.opt_on <= OBS_CUTOFF, "returned_60d"].mean()

    obs = merged[merged.opt_on <= OBS_CUTOFF].copy()
    store_counts = obs.groupby(["server_id", "shop_id"]).size()
    eligible_stores = store_counts[store_counts >= MIN_STORE_CUSTOMERS].index
    obs = obs[obs.set_index(["server_id", "shop_id"]).index.isin(eligible_stores)]

    rng = np.random.default_rng(SEED)
    test_customers = obs.sample(n=min(N_TEST, len(obs)), random_state=SEED)

    def sig(row):
        return {"bill_amt": round(float(row.bill_amt), 2),
               "pay_amt": round(float(row.pay_amt), 2),
               "owe_amt": round(float(row.owe_amt), 2),
               "month": int(row.opt_on.month),
               "day_of_week": int(row.opt_on.dayofweek)}

    examples = []
    for _, test_row in test_customers.iterrows():
        pool = obs[(obs.server_id == test_row.server_id) &
                   (obs.shop_id == test_row.shop_id) &
                   (obs.company_id != test_row.company_id)]
        if len(pool) < N_ANALOGS:
            continue
        analog_rows = pool.sample(n=N_ANALOGS, random_state=SEED)
        analogs = [{"signature": sig(r), "returned_60d": bool(r.returned_60d)}
                  for r in analog_rows.itertuples()]
        cohort_mean = analog_rows["returned_60d"].mean()
        examples.append({
            "example_id": f"{test_row.server_id}_{test_row.shop_id}_{test_row.company_id}",
            "test_signature": sig(test_row),
            "true_returned_60d": bool(test_row.returned_60d),
            "analogs": analogs,
            "cohort_mean_prediction": round(float(cohort_mean), 4),
            "population_base_rate_prediction": round(float(pop_base_rate), 4),
        })

    with open("data/examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    n_pos = sum(e["true_returned_60d"] for e in examples)
    print(f"{len(examples)} examples written to data/examples.json "
          f"({n_pos} returned within 60d, population base rate {pop_base_rate:.1%})")


if __name__ == "__main__":
    main()
