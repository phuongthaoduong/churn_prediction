"""Step 1: build the small data files the trial needs.

Reads sale_data/srv_sales.csv once and writes:
  data/merchants.csv          one row per merchant (server_id): industry, mode, name
  data/purchase_days.pkl      one row per (customer, purchase day); customer = (server_id, shop_id, company_id)
  data/merchant_ipi_train.csv per-merchant IPI stats using data up to TRAIN_END only
  data/pilot_sample.csv       PILOT_SIZE merchants, stratified by industry, for the first LLM run

Same cleaning as the pipeline where it matters: negative bills dropped, the 26 test
accounts from step 2.2 dropped. The 356 amount outliers of step 2.1 are not removed
(they do not change purchase dates).
"""
import numpy as np
import pandas as pd
import config as C

C.DATA.mkdir(parents=True, exist_ok=True)

ind = pd.read_csv(C.SALE_DATA / "industry_servereid.csv", dtype=str)
merchants = (ind.drop_duplicates("serverid")
                .rename(columns={"serverid": "server_id"})[["server_id", "industry", "mode", "name"]])
merchants["mode"] = merchants["mode"].fillna("")
merchants.to_csv(C.DATA / "merchants.csv", index=False)
print("merchants:", len(merchants))

test = pd.read_csv(C.TEST_ACCOUNTS, dtype={"server_id": str})
test_keys = set(zip(test.server_id, test.shop_id, test.company_id))

parts = []
cols = ["server_id", "shop_id", "company_id", "opt_on", "bill_amt"]
dtypes = {"server_id": str, "shop_id": "int32", "company_id": "int64", "opt_on": str, "bill_amt": "float64"}
for chunk in pd.read_csv(C.SALE_DATA / "srv_sales.csv", usecols=cols, dtype=dtypes, chunksize=3_000_000):
    chunk = chunk[chunk.bill_amt >= 0].drop(columns="bill_amt").drop_duplicates()
    parts.append(chunk)
days = pd.concat(parts).drop_duplicates()
days["opt_on"] = pd.to_datetime(days.opt_on, errors="coerce")
days = days.dropna(subset=["opt_on"])
days = days[(days.opt_on >= C.DATA_START) & (days.opt_on <= C.DATA_END)]
key = list(zip(days.server_id, days.shop_id, days.company_id))
days = days[[k not in test_keys for k in key]]
days = days[days.server_id.isin(merchants.server_id)]
days = days.sort_values(["server_id", "shop_id", "company_id", "opt_on"]).reset_index(drop=True)
days.to_pickle(C.DATA / "purchase_days.pkl")
print("purchase days:", len(days), " customers:", days.groupby(["server_id", "shop_id", "company_id"]).ngroups)

# IPI stats in the training period only (no look-ahead into the evaluation period)
tr = days[days.opt_on <= C.TRAIN_END]
gap = tr.groupby(["server_id", "shop_id", "company_id"], sort=False).opt_on.diff().dt.days
tr = tr.assign(ipi=gap).dropna(subset=["ipi"])
stats = tr.groupby("server_id").ipi.agg(
    n_ipi="size", ipi_median="median", ipi_p95=lambda x: np.percentile(x, C.PERCENTILE)).reset_index()
stats = merchants.merge(stats, on="server_id", how="left")
stats["n_ipi"] = stats.n_ipi.fillna(0).astype(int)
stats.to_csv(C.DATA / "merchant_ipi_train.csv", index=False)

# pilot sample: stratified by industry, only merchants with enough data to check the LLM against
ok = stats[stats.n_ipi >= 100]
per = int(np.ceil(C.PILOT_SIZE / ok.industry.nunique()))
pilot = (ok.sample(frac=1, random_state=C.SEED)
           .groupby("industry").head(per)
           .sample(frac=1, random_state=C.SEED).head(C.PILOT_SIZE))
pilot[["server_id", "industry", "mode", "name"]].to_csv(C.DATA / "pilot_sample.csv", index=False)
print("pilot sample:", len(pilot))
