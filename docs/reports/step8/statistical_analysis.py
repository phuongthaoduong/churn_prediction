"""Step 8.5: executes the statistical analysis plan pre-registered in Step 7
(Section 7 of weekly_report_step7.md) against the realized factorial results
in experiment_output/metrics_per_batch.csv and experiment_output/retrain_log.csv.

No R / ARTool / pingouin is available in this environment, so the Aligned
Rank Transform (Wobbrock, Findlay, Gergle, & Higgins, 2011) is reimplemented
directly in Python via its closed-form effect decomposition for a balanced
additive design (scheme x strategy x batch, no scheme:batch or strategy:batch
interaction terms, consistent with the pre-registered reference model).
Mauchly's sphericity test is likewise reimplemented from its standard
orthonormal-contrast formulation. Every number below is computed from the
data on disk; nothing is hand-entered.

Outputs:
  - statistical_results.json  (all computed numbers, for quoting in the report)
  - statistical_analysis_log.txt (human-readable run log, same content as stdout)
  - trajectory_plot.png (month-by-month AUC_shared, all 8 conditions, primary arm)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).parent
DATA = HERE.parent.parent.parent / "experiment_output"
RNG = np.random.default_rng(42)

OUT = {}
LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


# ─────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────
mb = pd.read_csv(DATA / "metrics_per_batch.csv", parse_dates=["prediction_point"])
retrain = pd.read_csv(DATA / "retrain_log.csv", parse_dates=["t"])
summary = pd.read_csv(DATA / "metrics_summary.csv")

mb = mb.sort_values(["maturity_mode", "scheme", "strategy", "prediction_point"]).reset_index(drop=True)

FINAL_BATCH = pd.Timestamp("2026-03-01")
STRATEGIES = ["S1", "S2", "S3", "S4"]
SCHEMES = ["L1", "L2"]


# ─────────────────────────────────────────────────────────────────────────
# Helpers: balanced additive decomposition (for ART) and ANOVA on ranks
# ─────────────────────────────────────────────────────────────────────────
def decompose(df, value_col, a_col="scheme", b_col="strategy", block_col="prediction_point"):
    """Balanced additive decomposition Y = mu + a_i + b_j + ab_ij + t_t + e.
    df must be one row per (a_col, b_col, block_col) cell (fully crossed, n=1/cell)."""
    d = df[[a_col, b_col, block_col, value_col]].copy()
    mu = d[value_col].mean()
    a_eff = d.groupby(a_col)[value_col].mean() - mu
    b_eff = d.groupby(b_col)[value_col].mean() - mu
    t_eff = d.groupby(block_col)[value_col].mean() - mu
    d["a_i"] = d[a_col].map(a_eff)
    d["b_j"] = d[b_col].map(b_eff)
    d["t_t"] = d[block_col].map(t_eff)
    ab_mean = d.groupby([a_col, b_col])[value_col].mean()
    d["ab_ij"] = d.set_index([a_col, b_col]).index.map(ab_mean) - mu - d["a_i"].values - d["b_j"].values
    d["fitted_noint"] = mu + d["a_i"] + d["b_j"] + d["t_t"]
    d["fitted_full"] = d["fitted_noint"] + d["ab_ij"]
    d["resid"] = d[value_col] - d["fitted_full"]
    return mu, a_eff, b_eff, t_eff, d


def art_test(df, value_col, effect, a_col="scheme", b_col="strategy", block_col="prediction_point"):
    """Aligned Rank Transform F-test for `effect` in {'A','B','AB'}.
    Returns dict with F, df1, df2, p, and partial eta-squared on the aligned ranks."""
    mu, a_eff, b_eff, t_eff, d = decompose(df, value_col, a_col, b_col, block_col)

    if effect == "A":
        aligned = d["a_i"] + d["resid"]
    elif effect == "B":
        aligned = d["b_j"] + d["resid"]
    elif effect == "AB":
        aligned = d["ab_ij"] + d["resid"]
    else:
        raise ValueError(effect)

    d = d.copy()
    d["aligned"] = aligned
    d["rank"] = rankdata(d["aligned"])

    model = ols(f"rank ~ C({a_col}) * C({b_col}) + C({block_col})", data=d).fit()
    table = anova_lm(model, typ=2)

    term_map = {
        "A": f"C({a_col})",
        "B": f"C({b_col})",
        "AB": f"C({a_col}):C({b_col})",
    }
    term = term_map[effect]
    row = table.loc[term]
    resid_row = table.loc["Residual"]
    f_val = row["F"]
    p_val = row["PR(>F)"]
    df1 = row["df"]
    df2 = resid_row["df"]
    ss_effect = row["sum_sq"]
    ss_resid = resid_row["sum_sq"]
    partial_eta2 = ss_effect / (ss_effect + ss_resid)
    return {
        "effect": effect, "F": float(f_val), "df1": float(df1), "df2": float(df2),
        "p": float(p_val), "partial_eta2": float(partial_eta2),
    }


def mauchly_test(wide):
    """wide: n_subjects x k_conditions matrix (numpy array). Standard Mauchly's W
    via an orthonormal contrast transform of the k conditions."""
    n, k = wide.shape
    p = k - 1
    # Helmert-style orthonormal contrast basis for k levels -> (k, p)
    C = np.zeros((k, p))
    for j in range(p):
        C[: j + 1, j] = 1.0
        C[j + 1, j] = -(j + 1)
        C[:, j] /= np.sqrt((j + 1) * (j + 2))
    Z = wide @ C  # n x p
    S = np.cov(Z, rowvar=False, ddof=1)  # p x p
    det_S = np.linalg.det(S)
    tr_S = np.trace(S)
    W = det_S / (tr_S / p) ** p
    d_corr = 1 - (2 * p ** 2 + p + 2) / (6 * p * (n - 1))
    chi2 = -(n - 1) * d_corr * np.log(W)
    df = p * (p + 1) / 2 - 1
    pval = 1 - stats.chi2.cdf(chi2, df)
    return {"W": float(W), "chi2": float(chi2), "df": float(df), "p": float(pval), "n": int(n), "k": int(k)}


def rank_biserial_wilcoxon(x, y):
    diff = np.asarray(x) - np.asarray(y)
    nz = diff[diff != 0]
    if len(nz) == 0:
        return {"n_nonzero": 0, "r_rb": 0.0, "stat": None, "p": 1.0}
    ranks = rankdata(np.abs(nz))
    w_plus = ranks[nz > 0].sum()
    w_minus = ranks[nz < 0].sum()
    r_rb = (w_plus - w_minus) / (w_plus + w_minus)
    try:
        res = stats.wilcoxon(x, y, zero_method="wilcox", mode="exact")
    except ValueError:
        res = stats.wilcoxon(x, y, zero_method="wilcox", mode="approx")
    return {"n_nonzero": int(len(nz)), "r_rb": float(r_rb), "stat": float(res.statistic), "p": float(res.pvalue)}


def moving_block_bootstrap_ci(diffs, block_len=3, n_boot=10000, seed=42):
    diffs = np.asarray(diffs)
    n = len(diffs)
    n_blocks_needed = int(np.ceil(n / block_len))
    starts = np.arange(0, n - block_len + 1)
    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks_needed, replace=True)
        sample = np.concatenate([diffs[s:s + block_len] for s in chosen])[:n]
        boot_medians[b] = np.median(sample)
    lo, hi = np.percentile(boot_medians, [2.5, 97.5])
    return float(lo), float(hi), float(np.median(diffs))


def holm_table(pvals, labels):
    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    return [
        {"comparison": lbl, "p_raw": float(p), "p_holm": float(pa), "reject_h0": bool(rj)}
        for lbl, p, pa, rj in zip(labels, pvals, p_adj, reject)
    ]


# ─────────────────────────────────────────────────────────────────────────
# Step 1: response tables (primary arm = scheme-specific maturity)
# ─────────────────────────────────────────────────────────────────────────
log("=" * 72)
log("STEP 8.5: STATISTICAL ANALYSIS OF THE FACTORIAL EVALUATION")
log("=" * 72)

primary = mb[mb.maturity_mode == "scheme"].copy()
equalized = mb[mb.maturity_mode == "equalized"].copy()
assert len(primary) == 96 and len(equalized) == 96, (len(primary), len(equalized))

OUT["n_primary_rows"] = len(primary)
OUT["n_equalized_rows"] = len(equalized)
log(f"\nPrimary arm (scheme maturity): {len(primary)} batch-level rows "
    f"(8 conditions x 12 batches). Equalized arm: {len(equalized)} rows.")

# ─────────────────────────────────────────────────────────────────────────
# Step 2: assumption assessment (primary arm, auc_shared)
# ─────────────────────────────────────────────────────────────────────────
log("\n" + "-" * 72)
log("STEP 2: ASSUMPTION ASSESSMENT (primary arm, AUC_shared)")
log("-" * 72)

mu, a_eff, b_eff, t_eff, dec = decompose(primary, "auc_shared")
sh_stat, sh_p = stats.shapiro(dec["resid"])
OUT["shapiro"] = {"W": float(sh_stat), "p": float(sh_p), "n": int(len(dec))}
log(f"Shapiro-Wilk on residuals of the reference additive model: "
    f"W = {sh_stat:.4f}, p = {sh_p:.4g} (n = {len(dec)}).")

wide = primary.assign(cond=primary.scheme + "_" + primary.strategy).pivot(
    index="prediction_point", columns="cond", values="auc_shared"
).sort_index()
mauchly = mauchly_test(wide.values)
OUT["mauchly"] = mauchly
log(f"Mauchly's sphericity test on the 8-condition x 12-batch matrix: "
    f"W = {mauchly['W']:.4g}, chi2({mauchly['df']:.0f}) = {mauchly['chi2']:.3f}, "
    f"p = {mauchly['p']:.4g}. With p = k-1 = 7 contrast dimensions estimated from "
    f"only n-1 = 11 covariance degrees of freedom, this test is acknowledged to be "
    f"underpowered/unstable at this sample size — reported as a diagnostic, not as "
    f"a gate on which branch of the protocol is taken (that branch is decided in "
    f"advance per the Step 7 pre-registration, which expects violation at n = 12).")

# Reference parametric model (F-tests, for completeness/diagnostic only)
full_model = ols("auc_shared ~ C(scheme) * C(strategy) + C(prediction_point)", data=primary).fit()
param_table = anova_lm(full_model, typ=2)
OUT["parametric_reference_anova"] = {
    term: {"df": float(param_table.loc[term, "df"]),
           "F": float(param_table.loc[term, "F"]),
           "p": float(param_table.loc[term, "PR(>F)"])}
    for term in ["C(scheme)", "C(strategy)", "C(scheme):C(strategy)"]
}
log("\nParametric reference model (two-way RM-ANOVA, batch as blocking factor), "
    "reported for completeness though its assumptions are not tenable:")
for term, vals in OUT["parametric_reference_anova"].items():
    log(f"  {term}: F({vals['df']:.0f}, {param_table.loc['Residual','df']:.0f}) = "
        f"{vals['F']:.3f}, p = {vals['p']:.4g}")

# ─────────────────────────────────────────────────────────────────────────
# Step 3: confirmatory ART-ANOVA (primary arm, auc_shared)
# ─────────────────────────────────────────────────────────────────────────
log("\n" + "-" * 72)
log("STEP 3: ART-ANOVA OMNIBUS TESTS (primary arm, AUC_shared)")
log("-" * 72)

art_A = art_test(primary, "auc_shared", "A")
art_B = art_test(primary, "auc_shared", "B")
art_AB = art_test(primary, "auc_shared", "AB")
OUT["art_primary_auc"] = {"A_scheme": art_A, "B_strategy": art_B, "AB_interaction": art_AB}

for name, res in [("Main effect A (scheme)", art_A), ("Main effect B (strategy)", art_B),
                  ("A x B interaction (confirmatory)", art_AB)]:
    log(f"{name}: F({res['df1']:.0f}, {res['df2']:.0f}) = {res['F']:.3f}, "
        f"p = {res['p']:.4g}, partial eta^2 = {res['partial_eta2']:.4f}")

INTERACTION_SIGNIFICANT = art_AB["p"] < 0.05
OUT["interaction_significant"] = bool(INTERACTION_SIGNIFICANT)
log(f"\nInteraction {'IS' if INTERACTION_SIGNIFICANT else 'is NOT'} significant at "
    f"alpha = 0.05 -> branch {'4a' if INTERACTION_SIGNIFICANT else '4b'} of the protocol.")

# ─────────────────────────────────────────────────────────────────────────
# Step 4: post-hoc decomposition
# ─────────────────────────────────────────────────────────────────────────
log("\n" + "-" * 72)
log("STEP 4: POST-HOC DECOMPOSITION")
log("-" * 72)

posthoc = {}

if INTERACTION_SIGNIFICANT:
    # 4a(i): Friedman within each scheme across S1-S4, then Wilcoxon+Holm
    for scheme in SCHEMES:
        sub = primary[primary.scheme == scheme]
        arrs = [sub[sub.strategy == s].sort_values("prediction_point")["auc_shared"].values for s in STRATEGIES]
        fr_stat, fr_p = stats.friedmanchisquare(*arrs)
        pairs, pvals, effects = [], [], []
        for i in range(4):
            for j in range(i + 1, 4):
                lbl = f"{STRATEGIES[i]} vs {STRATEGIES[j]}"
                rb = rank_biserial_wilcoxon(arrs[i], arrs[j])
                lo, hi, med = moving_block_bootstrap_ci(arrs[i] - arrs[j])
                pairs.append(lbl); pvals.append(rb["p"])
                effects.append({"comparison": lbl, "r_rb": rb["r_rb"], "median_diff": med,
                                 "ci95_lo": lo, "ci95_hi": hi, "p_raw": rb["p"]})
        holm = holm_table(pvals, pairs)
        posthoc[f"within_{scheme}"] = {
            "friedman_chi2": float(fr_stat), "friedman_p": float(fr_p),
            "pairwise": [dict(e, p_holm=h["p_holm"], reject_h0=h["reject_h0"])
                         for e, h in zip(effects, holm)],
        }
        log(f"\nWithin {scheme}: Friedman chi2(3) = {fr_stat:.3f}, p = {fr_p:.4g}")
        for row in posthoc[f"within_{scheme}"]["pairwise"]:
            log(f"  {row['comparison']}: median(diff) = {row['median_diff']:+.4f} "
                f"[{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}], r_rb = {row['r_rb']:+.3f}, "
                f"p_raw = {row['p_raw']:.4g}, p_holm = {row['p_holm']:.4g}, "
                f"reject = {row['reject_h0']}")

    # 4a(ii): L1 vs L2 within each strategy, Holm over 4 tests
    pairs, pvals, effects = [], [], []
    for strat in STRATEGIES:
        l1 = primary[(primary.scheme == "L1") & (primary.strategy == strat)].sort_values("prediction_point")["auc_shared"].values
        l2 = primary[(primary.scheme == "L2") & (primary.strategy == strat)].sort_values("prediction_point")["auc_shared"].values
        rb = rank_biserial_wilcoxon(l1, l2)
        lo, hi, med = moving_block_bootstrap_ci(l1 - l2)
        lbl = f"L1 vs L2 ({strat})"
        pairs.append(lbl); pvals.append(rb["p"])
        effects.append({"comparison": lbl, "r_rb": rb["r_rb"], "median_diff": med,
                         "ci95_lo": lo, "ci95_hi": hi, "p_raw": rb["p"]})
    holm = holm_table(pvals, pairs)
    posthoc["scheme_within_strategy"] = [
        dict(e, p_holm=h["p_holm"], reject_h0=h["reject_h0"]) for e, h in zip(effects, holm)
    ]
    log("\nScheme effect within each strategy (L1 vs L2), Holm over 4 tests:")
    for row in posthoc["scheme_within_strategy"]:
        log(f"  {row['comparison']}: median(diff) = {row['median_diff']:+.4f} "
            f"[{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}], r_rb = {row['r_rb']:+.3f}, "
            f"p_raw = {row['p_raw']:.4g}, p_holm = {row['p_holm']:.4g}, "
            f"reject = {row['reject_h0']}")
else:
    # 4b: main-effect contrasts
    if art_B["p"] < 0.05:
        pairs, pvals, effects = [], [], []
        pooled = primary.copy()
        for i in range(4):
            for j in range(i + 1, 4):
                si, sj = STRATEGIES[i], STRATEGIES[j]
                xi = pooled[pooled.strategy == si].sort_values(["scheme", "prediction_point"])["auc_shared"].values
                xj = pooled[pooled.strategy == sj].sort_values(["scheme", "prediction_point"])["auc_shared"].values
                rb = rank_biserial_wilcoxon(xi, xj)
                lo, hi, med = moving_block_bootstrap_ci(xi - xj)
                lbl = f"{si} vs {sj} (pooled across schemes)"
                pairs.append(lbl); pvals.append(rb["p"])
                effects.append({"comparison": lbl, "r_rb": rb["r_rb"], "median_diff": med,
                                 "ci95_lo": lo, "ci95_hi": hi, "p_raw": rb["p"]})
        holm = holm_table(pvals, pairs)
        posthoc["strategy_main_effect_pooled"] = [
            dict(e, p_holm=h["p_holm"], reject_h0=h["reject_h0"]) for e, h in zip(effects, holm)
        ]
        log("\nFactor B (strategy) main-effect contrasts, pooled across schemes, Holm over 6 tests:")
        for row in posthoc["strategy_main_effect_pooled"]:
            log(f"  {row['comparison']}: median(diff) = {row['median_diff']:+.4f} "
                f"[{row['ci95_lo']:+.4f}, {row['ci95_hi']:+.4f}], r_rb = {row['r_rb']:+.3f}, "
                f"p_raw = {row['p_raw']:.4g}, p_holm = {row['p_holm']:.4g}, "
                f"reject = {row['reject_h0']}")
    if art_A["p"] < 0.05:
        l1 = primary[primary.scheme == "L1"].sort_values(["strategy", "prediction_point"])["auc_shared"].values
        l2 = primary[primary.scheme == "L2"].sort_values(["strategy", "prediction_point"])["auc_shared"].values
        rb = rank_biserial_wilcoxon(l1, l2)
        lo, hi, med = moving_block_bootstrap_ci(l1 - l2)
        posthoc["scheme_main_effect"] = {"median_diff": med, "ci95_lo": lo, "ci95_hi": hi,
                                          "r_rb": rb["r_rb"], "p": rb["p"]}
        log(f"\nFactor A (scheme) main effect (no decomposition needed, 2 levels): "
            f"median(L1-L2) = {med:+.4f} [{lo:+.4f}, {hi:+.4f}], r_rb = {rb['r_rb']:+.3f}, "
            f"p = {rb['p']:.4g}")
    if art_A["p"] >= 0.05 and art_B["p"] >= 0.05:
        log("\nNeither main effect reaches significance; no further decomposition performed.")

OUT["posthoc"] = posthoc

# ─────────────────────────────────────────────────────────────────────────
# Primary vs. sensitivity (equalized) arm reconciliation
# ─────────────────────────────────────────────────────────────────────────
log("\n" + "-" * 72)
log("PRIMARY VS. SENSITIVITY (EQUALIZED) ARM")
log("-" * 72)

art_A_eq = art_test(equalized, "auc_shared", "A")
art_B_eq = art_test(equalized, "auc_shared", "B")
art_AB_eq = art_test(equalized, "auc_shared", "AB")
OUT["art_equalized_auc"] = {"A_scheme": art_A_eq, "B_strategy": art_B_eq, "AB_interaction": art_AB_eq}

log(f"Equalized arm ART-ANOVA: A p = {art_A_eq['p']:.4g}, B p = {art_B_eq['p']:.4g}, "
    f"A x B p = {art_AB_eq['p']:.4g} (interaction "
    f"{'significant' if art_AB_eq['p'] < 0.05 else 'not significant'}, matching primary = "
    f"{(art_AB_eq['p'] < 0.05) == INTERACTION_SIGNIFICANT}).")

summary_wide = summary.pivot_table(index=["scheme", "strategy"], columns="maturity_mode",
                                    values="mean_auc_shared")
summary_wide.columns.name = None
summary_wide = summary_wide.rename(columns={"scheme": "mean_auc_scheme_arm", "equalized": "mean_auc_equalized_arm"})
summary_wide["delta_equalized_minus_scheme"] = summary_wide["mean_auc_equalized_arm"] - summary_wide["mean_auc_scheme_arm"]
OUT["arm_delta_table"] = summary_wide.reset_index().to_dict(orient="records")
max_abs_delta = summary_wide["delta_equalized_minus_scheme"].abs().max()
OUT["max_abs_arm_delta"] = float(max_abs_delta)
log(f"Maximum |equalized - scheme| mean-AUC delta across the 8 conditions: {max_abs_delta:.4f}.")

# ─────────────────────────────────────────────────────────────────────────
# Robustness checks
# ─────────────────────────────────────────────────────────────────────────
log("\n" + "-" * 72)
log("ROBUSTNESS CHECKS")
log("-" * 72)

# (a) Recall@10% as response
art_AB_recall = art_test(primary, "recall10_shared", "AB")
art_A_recall = art_test(primary, "recall10_shared", "A")
art_B_recall = art_test(primary, "recall10_shared", "B")
OUT["robustness_recall10"] = {"A": art_A_recall, "B": art_B_recall, "AB": art_AB_recall}
log(f"(a) Recall@10% as response: A p = {art_A_recall['p']:.4g}, B p = {art_B_recall['p']:.4g}, "
    f"A x B p = {art_AB_recall['p']:.4g} (interaction "
    f"{'significant' if art_AB_recall['p'] < 0.05 else 'not significant'}).")

# (b) exclude final batch
trimmed = primary[primary.prediction_point != FINAL_BATCH]
art_AB_trim = art_test(trimmed, "auc_shared", "AB")
art_A_trim = art_test(trimmed, "auc_shared", "A")
art_B_trim = art_test(trimmed, "auc_shared", "B")
OUT["robustness_exclude_final_batch"] = {"A": art_A_trim, "B": art_B_trim, "AB": art_AB_trim,
                                          "n_rows": int(len(trimmed))}
log(f"(b) Excluding the final batch (2026-03-01, n = {len(trimmed)} rows): "
    f"A p = {art_A_trim['p']:.4g}, B p = {art_B_trim['p']:.4g}, A x B p = {art_AB_trim['p']:.4g} "
    f"(interaction {'significant' if art_AB_trim['p'] < 0.05 else 'not significant'}, matching "
    f"full-data conclusion = {(art_AB_trim['p'] < 0.05) == INTERACTION_SIGNIFICANT}).")

# (c) trajectory plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = {"L1": "#1f77b4", "L2": "#d62728"}
styles = {"S1": "-", "S2": "--", "S3": "-.", "S4": ":"}
for scheme in SCHEMES:
    for strat in STRATEGIES:
        sub = primary[(primary.scheme == scheme) & (primary.strategy == strat)].sort_values("prediction_point")
        ax.plot(sub["prediction_point"], sub["auc_shared"], color=colors[scheme],
                linestyle=styles[strat], marker="o", markersize=3,
                label=f"{scheme}-{strat}", linewidth=1.4)
ax.set_xlabel("Evaluation batch (prediction point)")
ax.set_ylabel("AUC_shared")
ax.set_title("Batch-level AUC_shared by condition, primary (scheme-maturity) arm")
ax.legend(ncol=2, fontsize=8, loc="lower left")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
plot_path = HERE / "trajectory_plot.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
log(f"(c) Trajectory plot saved to {plot_path.name}.")

# ─────────────────────────────────────────────────────────────────────────
# Cost-performance frontier (descriptive, primary arm)
# ─────────────────────────────────────────────────────────────────────────
log("\n" + "-" * 72)
log("COST-PERFORMANCE FRONTIER (descriptive, primary arm)")
log("-" * 72)
cost_perf = summary[summary.maturity_mode == "scheme"][
    ["scheme", "strategy", "mean_auc_shared", "n_retrains", "total_fit_seconds", "total_train_rows"]
].sort_values(["scheme", "strategy"])
OUT["cost_performance_primary"] = cost_perf.to_dict(orient="records")
for _, row in cost_perf.iterrows():
    log(f"  {row.scheme} {row.strategy}: mean AUC = {row.mean_auc_shared:.4f}, "
        f"retrains = {int(row.n_retrains)}, fit time = {row.total_fit_seconds:.1f}s, "
        f"rows processed = {int(row.total_train_rows):,}")

# ─────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────
with open(HERE / "statistical_results.json", "w") as f:
    json.dump(OUT, f, indent=2, default=str)
with open(HERE / "statistical_analysis_log.txt", "w") as f:
    f.write("\n".join(LOG_LINES) + "\n")

log(f"\nSaved statistical_results.json and statistical_analysis_log.txt to {HERE}")
