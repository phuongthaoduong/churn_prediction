"""Step 3: how reliable is the LLM knowledge?

Reads data/llm_knowledge_reps.csv, data/llm_knowledge.csv, data/merchant_ipi_train.csv
(and data/human_labels.csv if you made one: columns server_id, business_type).
Writes results/reliability.txt.

Checks
  A. Self-consistency over repeated calls: Fleiss kappa (business type), ICC(1) on log days.
  B. Agreement between models (if more than one model was called).
  C. Agreement with real data (training period only): Spearman rho between the LLM's
     cycle and the merchant's real median / P95 IPI, overall and WITHIN industry.
     The within-industry rho is the one that matters: it says whether the LLM knows
     more than the 13-industry label already tells us.
  D. Name masked vs not masked (if the --mask-name run exists).
  E. Human labels (optional).

Decision rule suggested for the pilot: continue if within-industry rho > 0.3 with a
bootstrap 95% CI above 0.
"""
import numpy as np
import pandas as pd
import config as C

C.RESULTS.mkdir(parents=True, exist_ok=True)
out = []
def p(*x):
    s = " ".join(str(v) for v in x); print(s); out.append(s)

def spearman(a, b):
    a, b = pd.Series(a).rank(), pd.Series(b).rank()
    return a.corr(b)

def boot_ci(a, b, n=1000, seed=C.SEED):
    rng = np.random.default_rng(seed); a = np.asarray(a); b = np.asarray(b)
    v = [spearman(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return np.nanpercentile(v, [2.5, 97.5])

def fleiss_kappa(labels):  # labels: DataFrame merchants x reps
    cats = pd.unique(labels.values.ravel())
    counts = np.stack([(labels == c).sum(1).values for c in cats], 1).astype(float)
    n = counts.sum(1); keep = n > 1; counts, n = counts[keep], n[keep]
    P_i = ((counts ** 2).sum(1) - n) / (n * (n - 1))
    p_j = counts.sum(0) / counts.sum()
    Pe = (p_j ** 2).sum()
    return (P_i.mean() - Pe) / (1 - Pe) if Pe < 1 else np.nan

def icc1(x):  # x: DataFrame merchants x reps (log values)
    x = x.dropna(); k = x.shape[1]; n = len(x)
    if n < 3 or k < 2: return np.nan
    gm = x.values.mean(); rm = x.mean(1).values
    msb = k * ((rm - gm) ** 2).sum() / (n - 1)
    msw = ((x.values - rm[:, None]) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw)

reps = pd.read_csv(C.DATA / "llm_knowledge_reps.csv", dtype={"server_id": str})
agg = pd.read_csv(C.DATA / "llm_knowledge.csv", dtype={"server_id": str})
st = pd.read_csv(C.DATA / "merchant_ipi_train.csv", dtype={"server_id": str})
if reps.model.astype(str).str.startswith("mock").any():
    p("!!! MOCK knowledge detected: these numbers only test the code, they say nothing about LLMs.")

for (model, masked), r in reps.groupby(["model", "masked"]):
    p(f"\n=== {model}{' (name masked)' if masked else ''}: {r.server_id.nunique()} merchants, {len(r)} answers")
    p(f"info_sufficient = true: {r.groupby('server_id').info_sufficient.mean().gt(.5).mean():.1%} of merchants")
    p(f"business_type '无法判断': {(r.business_type == '无法判断').mean():.1%} of answers")
    bt = r.pivot_table(index="server_id", columns="rep", values="business_type", aggfunc="first")
    p(f"A. Fleiss kappa business_type over reps: {fleiss_kappa(bt):.3f}")
    for col in ["typical_days", "long_gap_days"]:
        pv = r.pivot_table(index="server_id", columns="rep", values=col)
        x = np.log(pv.where(pv > 0))
        p(f"A. ICC(1) log {col} over reps: {icc1(x):.3f}")

    a = agg[(agg.model == model) & (agg.masked == masked)].merge(st, on="server_id")
    a = a[a.n_ipi >= 100].dropna(subset=["typical_days", "long_gap_days"])
    if len(a) < 10:
        p("C. too few merchants with >=100 IPIs to compare with data"); continue
    for llm_col, data_col in [("typical_days", "ipi_median"), ("long_gap_days", "ipi_p95")]:
        rho = spearman(a[llm_col], a[data_col]); lo, hi = boot_ci(a[llm_col], a[data_col])
        # within industry: remove industry mean of log values on both sides
        ly = np.log(a[llm_col]) - np.log(a[llm_col]).groupby(a.industry).transform("mean")
        ld = np.log(a[data_col]) - np.log(a[data_col]).groupby(a.industry).transform("mean")
        rw = spearman(ly, ld); wlo, whi = boot_ci(ly, ld)
        p(f"C. {llm_col} vs real {data_col} (n={len(a)}): rho={rho:.3f} [{lo:.2f},{hi:.2f}]"
          f" | within industry rho={rw:.3f} [{wlo:.2f},{whi:.2f}]")
    ratio = np.log(a.long_gap_days / a.ipi_p95)
    p(f"C. scale: median LLM long_gap / real P95 = {np.exp(ratio.median()):.2f} (calibrated away in step 4)")
    sure = a[a.info_sufficient > .5]
    if len(sure) >= 10:
        p(f"C. only info_sufficient merchants (n={len(sure)}): rho long_gap vs P95 = {spearman(sure.long_gap_days, sure.ipi_p95):.3f}")

models = agg[~agg.masked.astype(bool)].model.unique()
if len(models) > 1:
    p("\n=== B. between models")
    w = agg[~agg.masked.astype(bool)].pivot_table(index="server_id", columns="model", values="long_gap_days")
    p(w.rank().corr().round(3).to_string())
    bt = agg[~agg.masked.astype(bool)].pivot_table(index="server_id", columns="model", values="business_type", aggfunc="first")
    p(f"Fleiss kappa business_type across models: {fleiss_kappa(bt.dropna()):.3f}")

if agg.masked.astype(bool).any():
    p("\n=== D. name masked vs not masked (same model)")
    for model in agg[agg.masked.astype(bool)].model.unique():
        x = agg[agg.model == model].pivot_table(index="server_id", columns="masked", values="long_gap_days").dropna()
        x = x.join(st.set_index("server_id")[["ipi_p95", "n_ipi"]]); x = x[x.n_ipi >= 100]
        if len(x) >= 10:
            p(f"{model}: rho vs real P95 with name={spearman(x[False], x.ipi_p95):.3f}, "
              f"without name={spearman(x[True], x.ipi_p95):.3f} (n={len(x)})")

hl = C.DATA / "human_labels.csv"
if hl.exists():
    h = pd.read_csv(hl, dtype={"server_id": str})
    p("\n=== E. LLM business_type vs your labels")
    for model in models:
        x = h.merge(agg[(agg.model == model) & ~agg.masked.astype(bool)], on="server_id", suffixes=("_human", "_llm"))
        p(f"{model}: accuracy {(x.business_type_human == x.business_type_llm).mean():.1%} (n={len(x)})")

(C.RESULTS / "reliability.txt").write_text("\n".join(out), encoding="utf-8")
print("\n-> results/reliability.txt")
