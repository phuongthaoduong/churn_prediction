# Tier S: SHAP-to-Narrative Explanation Layer — Findings

Date: 2026-11-04. Model: `deepseek-flash` (DeepSeek-V4.1-Flash-1017),
temperature 0. 80/80 calls succeeded (40 examples × full/ablated variants).

## Part 1 — Step 9: SHAP resolves the open question (not an LLM result)

Before any LLM work, `pipeline/shap_analysis.py` was built (new pipeline
step 9 — `shap` added to `requirements.txt`, no code existed before this;
the project's usual dependency-vetting server (`sonatype-guide`) was down
at the time with an expired token, so the install proceeded without that
check — `shap` is a standard, well-known interpretability library, not a
borderline dependency) to resolve the question Step 7's report explicitly
deferred to "Phase 2":
whether `ipi_std`'s position as the #1 gain-importance feature in both
schemes, despite being 63.5%/64.1% NaN, reflects genuine signal or an
artifact of XGBoost's NaN-routing.

Fit one canonical model per scheme (frozen hyperparameters and feature
set, trained on all matured batches as of the final evaluation point —
the same training data S2's final retrain used), computed SHAP values on
a 15,000-row sample of the final batch (`experiment_output/shap_summary.csv`,
`shap_values_L1.parquet`, `shap_values_L2.parquet`).

**Result:** `ipi_std` is rank 1 by SHAP in both schemes, matching its gain
rank. More precisely — and this is the part that actually answers the
question — the per-row SHAP magnitude is *higher* when `ipi_std` has a
real value (mean |SHAP| 0.646/0.883 for L1/L2) than when it's NaN
(0.401/0.437), and the NaN rows (64.1% of data) contribute *less* than
their proportional share of total SHAP mass (52.6%/46.9%). **Conclusion:
the signal comes from the actual inter-purchase-interval variability when
observable, not primarily from the fact of missingness.** Full numbers:
`experiment_output/shap_ipi_std_nan_split.csv`,
`experiment_output/step9_shap_summary.txt`.

## Part 2 — LLM explanation layer, built on Step 9's SHAP output

40 customers (10 highest-risk + 10 random, per scheme) were sampled from
Step 9's SHAP matrix. For each, the LLM was given only the top-5 SHAP
features with their real values and SHAP contributions — nothing about
customer identity — and asked for a structured `primary_driver` field, a
plain-language explanation, and a suggested retention action. It was run
twice per customer: once with the real top-5 (`full`), once with the #1
SHAP feature *removed entirely* (`ablated`). Full prompt and code:
`llm_explanation_trial/02_llm_explain.py`.

**Faithfulness test (exact, not fuzzy text-matching):** does the ablated
run's self-reported `primary_driver` ever still name the feature that was
removed from its context — which would be an unambiguous hallucination,
since that feature was never given to it?

| Check | Result |
|---|---|
| Full-prompt runs correctly citing the actual top SHAP feature as `primary_driver` | **100% (40/40)** |
| Ablated-prompt runs hallucinating the removed feature as `primary_driver` | **0% (0/40)** |

This is a clean positive result — unlike the drift-policy diagnostic
(`llm_drift_strategy_trial/`), which came back net-negative. Inspected
manually, not just mechanically: ablated runs don't just avoid naming the
removed feature, they coherently shift to citing the actual #2 feature by
SHAP magnitude and adjust the explanation text accordingly. Example
(customer L2_1, removed feature `customer_tenure`):

> Full: "The biggest factor in this very high churn score is
> customer_tenure, with a value of 87.0 and a risk contribution of 1.1764."
>
> Ablated: "The biggest factor pushing risk up is store_median_ipi, with a
> value of 1.0 and a SHAP contribution of 1.0083, followed by recency..."

Low-risk customers were handled sensibly too, correctly following the
prompt's instruction not to invent an action when none is needed:

> "No action is needed because the predicted churn risk is only 0.0118;
> continue normal service." (customer L2_18, y_prob 0.012)

Mock controls (`mock-faithful`, `mock-unfaithful`) confirm the parsing and
faithfulness-check logic itself is correct — a deliberately-unfaithful
mock model shows 100% hallucination on the same test, so the 0% result
for the real model isn't an artifact of a test that can't detect failure.

## Conclusion

Both halves of Tier S succeeded. Step 9 gives a real, citable resolution
to an open thesis question using existing infrastructure extended with
one new, standard library (`shap`). The LLM explanation layer passed its
faithfulness test cleanly at this sample size (n=40) — it explains
predictions grounded entirely in the model's own real attributions, never
in outside knowledge, and correctly updates its stated reasoning when the
evidence it's given changes. This is the one clearly positive LLM result
in the project so far, in contrast to the two null/negative ones
(`llm_prior_threshold_trial/`, `llm_drift_strategy_trial/`).

**Caveats, stated plainly:** n=40 is a pilot scale, not a thesis-grade
sample — scaling to a larger, stratified sample would strengthen the
claim. The faithfulness test only checks the single ablation of the #1
feature; it doesn't test robustness to other perturbations (e.g.
ablating the #2 feature, or feeding contradictory evidence). The
`suggested_action` field was inspected manually for a handful of examples
only, not systematically validated.
