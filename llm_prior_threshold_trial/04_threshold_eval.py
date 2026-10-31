"""Step 4: does an LLM prior give better churn thresholds than data alone?

For every monthly prediction point (PP) and every merchant, compute the churn
threshold tau (days) four ways, using only purchase gaps (IPIs) that ended before PP:

  L2       pooled P95 of the merchant's industry (what the pipeline does now, uncapped)
  L2-M     merchant P95 shrunk toward an industry prior          (data only, strong baseline)
  L2-K     merchant P95 shrunk toward an industry + LLM prior     (the idea being tested)
  L2-Kshuf same as L2-K but LLM values shuffled within industry  (control: is it the content?)

Shrinkage is done on log days:  log tau = (n*log P95_m + kappa*log prior) / (n + kappa),
with n = number of IPIs the merchant has before PP. kappa is picked on the training PPs.

How the thresholds are scored (no new labels needed)
  M1  log error vs the merchant's FUTURE P95: |log tau - log P95 of gaps that start after PP|
      (merchants with >= MIN_FUTURE_GAPS such gaps; PP up to 2025-09 so gaps can close).
  M2  false churn: share of customers labelled churn at PP (no purchase within tau, tau capped
      at 90 like the pipeline) who buy again within FOLLOWUP_DAYS after PP + tau.

Usage
  python3 04_threshold_eval.py --model deepseek-chat
  python3 04_threshold_eval.py --model mock-oracle      # after `02_llm_knowledge.py mock`
Writes results/threshold_eval_<model>.csv (per PP x method x size bucket) and a summary txt.
"""
import argparse
import numpy as np
import pandas as pd
import config as C

MIN_FUTURE_GAPS = 30
LAST_M1_PP = "2025-09-01"
CAP = 90
LLM_COL = "long_gap_days"

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--masked", action="store_true", help="use the name-masked LLM answers")
a = ap.parse_args()
C.RESULTS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(C.SEED)

# ---------- data ----------
mer = pd.read_csv(C.DATA / "merchants.csv", dtype=str)
know = pd.read_csv(C.DATA / "llm_knowledge.csv", dtype={"server_id": str})
know = know[(know.model == a.model) & (know.masked.astype(bool) == a.masked)].dropna(subset=[LLM_COL])
if know.empty:
    raise SystemExit(f"no LLM knowledge for model={a.model} masked={a.masked}; run 02_llm_knowledge.py first")
mer = mer.merge(know[["server_id", LLM_COL]], on="server_id", how="left")
mer["m"] = np.arange(len(mer))
m_of = dict(zip(mer.server_id, mer.m))
ind_codes, ind_names = pd.factorize(mer.industry)
has_k = mer[LLM_COL].notna().values
log_llm = np.log(mer[LLM_COL].values)
# shuffled control: permute LLM values among merchants of the same industry
shuf = log_llm.copy()
for g in np.unique(ind_codes):
    idx = np.where((ind_codes == g) & has_k)[0]
    shuf[idx] = log_llm[rng.permutation(idx)]
print(f"merchants with LLM knowledge: {has_k.sum()} of {len(mer)}")

days = pd.read_pickle(C.DATA / "purchase_days.pkl")
t0 = pd.Timestamp("2024-01-01")
days["d"] = (days.opt_on - t0).dt.days.astype(np.int32)
days["m"] = days.server_id.map(m_of).astype(np.int32)
days["cid"] = days.groupby(["server_id", "shop_id", "company_id"], sort=False).ngroup().astype(np.int32)
days = days[["cid", "m", "d"]].sort_values(["cid", "d"]).reset_index(drop=True)
same = days.cid.values[1:] == days.cid.values[:-1]
gaps = pd.DataFrame({"m": days.m.values[1:][same], "start": days.d.values[:-1][same],
                     "end": days.d.values[1:][same]})
gaps["gap"] = gaps.end - gaps.start
day = lambda s: (pd.Timestamp(s) - t0).days
END = day(C.DATA_END)

pps = pd.date_range(C.FIRST_PP, C.LAST_PP, freq="MS")

def p95_by(keys, vals, size):
    out = np.full(size, np.nan); n = np.zeros(size, int)
    if len(vals):
        s = pd.Series(vals).groupby(keys)
        q = s.quantile(C.PERCENTILE / 100); c = s.size()
        out[q.index.values] = q.values; n[c.index.values] = c.values
    return out, n

_hist = {}
def hist_stats(pp_day):
    if pp_day not in _hist:
        h = gaps[gaps.end < pp_day]
        p95_m, n_m = p95_by(h.m.values, h.gap.values, len(mer))
        p95_i, _ = p95_by(ind_codes[h.m.values], h.gap.values, len(ind_names))
        _hist[pp_day] = (p95_m, n_m, p95_i)
    return _hist[pp_day]

def thresholds(pp_day, kappa):
    p95_m, n_m, p95_i = hist_stats(pp_day)
    l2 = p95_i[ind_codes]
    # industry prior = mean log P95 of well-observed merchants in the industry (fallback: pooled)
    fit = (n_m >= C.MIN_IPI_FOR_REGRESSION) & np.isfinite(p95_m)
    lp = np.log(np.where(fit, p95_m, np.nan))
    ind_mean = pd.Series(lp).groupby(ind_codes).mean().reindex(range(len(ind_names))).values
    prior_m = np.where(np.isfinite(ind_mean[ind_codes]), ind_mean[ind_codes], np.log(l2))

    def k_prior(llm):
        # LLM adds only within-industry information: z = llm - industry mean of llm
        zm = pd.Series(np.where(has_k, llm, np.nan)).groupby(ind_codes).transform("mean").values
        z = llm - zm
        ok = fit & has_k & np.isfinite(z)
        beta = np.polyfit(z[ok], lp[ok] - prior_m[ok], 1)[0] if ok.sum() >= 10 else 0.0
        return prior_m + beta * np.where(np.isfinite(z), z, 0.0), beta

    prior_k, beta = k_prior(log_llm)
    prior_s, _ = k_prior(shuf)
    def shrink(prior, kap):
        lm = np.log(np.where(np.isfinite(p95_m) & (p95_m > 0), p95_m, 1.0))
        w = n_m / (n_m + kap)
        return np.exp(w * lm + (1 - w) * prior)
    if not isinstance(kappa, dict):
        kappa = {k: kappa for k in ("L2-M", "L2-K", "L2-Kshuf")}
    taus = {"L2": l2, "L2-M": shrink(prior_m, kappa["L2-M"]), "L2-K": shrink(prior_k, kappa["L2-K"]),
            "L2-Kshuf": shrink(prior_s, kappa["L2-Kshuf"])}
    return taus, n_m, beta

_fut = {}
def future_p95(pp_day):
    if pp_day not in _fut:
        f = gaps[(gaps.start >= pp_day)]
        _fut[pp_day] = p95_by(f.m.values, f.gap.values, len(mer))
    return _fut[pp_day]

def bucket(n):
    return pd.cut(n, C.SIZE_BUCKETS, right=False, labels=["<100", "100-500", "500-2000", ">=2000"])

def m1(pp_day, kappa):
    taus, n_m, beta = thresholds(pp_day, kappa)
    fp, fn = future_p95(pp_day)
    ok = has_k & (fn >= MIN_FUTURE_GAPS) & np.isfinite(fp) & (fp > 0)
    rows = []
    for k, t in taus.items():
        err = np.abs(np.log(t[ok]) - np.log(fp[ok]))
        rows.append(pd.DataFrame({"method": k, "err": err, "bucket": bucket(n_m[ok])}))
    return pd.concat(rows), beta

# ---------- choose kappa on training PPs ----------
train_pps = [p for p in pps if p <= pd.Timestamp(C.TRAIN_END)]
best = {}
for kappa in C.KAPPA_GRID:
    e = pd.concat([m1(day(p), kappa)[0] for p in train_pps])
    best[kappa] = {m: e[e.method == m].err.mean() for m in ("L2-M", "L2-K", "L2-Kshuf")}
    print(f"kappa={kappa}: train M1 " + "  ".join(f"{m} {v:.4f}" for m, v in best[kappa].items()))
# each method gets its own best kappa, chosen on training PPs only
kappa = {m: min(best, key=lambda k: best[k][m]) for m in ("L2-M", "L2-K", "L2-Kshuf")}
print(f"chosen kappa per method: {kappa}")

# ---------- M1 and M2 over all PPs ----------
rows, m2rows = [], []
for p in pps:
    pd_ = day(p)
    taus, n_m, beta = thresholds(pd_, kappa)
    split = "train" if p <= pd.Timestamp(C.TRAIN_END) else "eval"
    if p <= pd.Timestamp(LAST_M1_PP):
        e, _ = m1(pd_, kappa)
        g = e.groupby(["method", "bucket"], observed=True).err.agg(["mean", "size"]).reset_index()
        g2 = e.groupby("method").err.agg(["mean", "size"]).reset_index(); g2["bucket"] = "all"
        for x in (g, g2):
            x["pp"] = p.date(); x["split"] = split; x["metric"] = "M1_log_err"; x["beta"] = beta
            rows.append(x)
    # M2: customers active in the 90 days before PP, at merchants with LLM knowledge
    before = days[(days.d < pd_) & (days.d >= pd_ - C.OBS_WINDOW_DAYS)].groupby("cid").m.first()
    after = days[days.d >= pd_].groupby("cid").d.min()
    cust = pd.DataFrame({"m": before}).join(after.rename("next"))
    cust = cust[has_k[cust.m.values]]
    for k, t in taus.items():
        tau = np.minimum(t[cust.m.values], CAP)
        churn = cust.next.isna().values | (cust.next.values >= pd_ + tau)
        # same customers for every method: decidable even at the 90-day cap
        decided = np.full(len(cust), pd_ + CAP + C.FOLLOWUP_DAYS <= END)
        back = churn & cust.next.notna().values & (cust.next.values <= pd_ + tau + C.FOLLOWUP_DAYS)
        sel = decided
        if sel.sum() == 0:
            continue
        m2rows.append({"pp": p.date(), "split": split, "method": k, "metric": "M2",
                       "churn_rate": churn[sel].mean(), "false_churn_rate": back[sel].sum() / max(churn[sel].sum(), 1),
                       "n_customers": int(sel.sum()), "coverage": sel.mean()})

r1 = pd.concat(rows); r2 = pd.DataFrame(m2rows)
tag = a.model.replace("/", "_") + ("__masked" if a.masked else "")
r1.to_csv(C.RESULTS / f"threshold_eval_M1_{tag}.csv", index=False)
r2.to_csv(C.RESULTS / f"threshold_eval_M2_{tag}.csv", index=False)

lines = [f"model={a.model} masked={a.masked} kappa={kappa} merchants_with_knowledge={has_k.sum()}"]
if a.model.startswith("mock"):
    lines.insert(0, "!!! MOCK knowledge: tests the code only.")
for split in ["train", "eval"]:
    s = r1[(r1.split == split)]
    if s.empty: continue
    t = s.pivot_table(index="bucket", columns="method", values="mean", aggfunc="mean")
    lines += [f"\nM1 mean |log tau - log future P95|, {split} PPs (lower is better):", t.round(4).to_string()]
for split in ["train", "eval"]:
    s = r2[r2.split == split]
    if s.empty: continue
    t = s.groupby("method")[["churn_rate", "false_churn_rate", "coverage"]].mean()
    lines += [f"\nM2 (tau capped at {CAP}), {split} PPs:", t.round(4).to_string()]
txt = "\n".join(lines); print(txt)
(C.RESULTS / f"threshold_eval_summary_{tag}.txt").write_text(txt, encoding="utf-8")
