# Option (a) Cheap Pilot: LLM Cold-Start Analog-Reasoner — Findings

Date: 2026-11-06. Model: `deepseek-flash` (DeepSeek-V4.1-Flash-1017),
temperature 0, `reasoning_effort: "none"` (see
`docs/reports/llm_sequence_pilot/findings.md` for why). 150/150 calls
succeeded.

## What this tested

The second and last untested half of Option (a): for a genuinely
cold-start customer (known only by their first-ever transaction), can an
LLM reasoning over real analog customers from the same store predict
whether they'll return within 60 days better than a plain cohort
average would? This is the project's pre-registered falsification
condition for this idea, stated when it was first proposed: *"Failure =
no improvement over (b) [cohort-mean imputation] — i.e., the LLM's
reasoning reduces to the same cohort average a GROUP BY would give."*

Built entirely from the raw cleaned sales data (`llm_coldstart_pilot/`),
not the windowed 90-day feature framing — ground truth is directly
observable: for every customer, did a second transaction happen within
60 days of their first? (Base rate: 38.3% across ~967k customers whose
first purchase was early enough to observe a full 60-day window.) 150
test customers were sampled from stores with ≥30 eligible customers
(2,183 of 2,581 stores qualify). For each, 20 *other* real customers from
the *same store* were pulled as analogs — each shown with their own
first-transaction signature (bill_amt, pay_amt, owe_amt, month,
day-of-week) **and their true actual outcome**. The LLM saw only this:
no customer or merchant identity anywhere.

Three predictors compared on the same 150 test customers:
- **(a) LLM-with-analogs** — reasons over the 20 analogs to estimate
  return probability.
- **(b) cohort-mean** — the plain proportion of the 20 analogs who
  returned. No LLM, no reasoning, just an average.
- **(c) population base rate** — 38.3% for everyone, no store
  information at all (reported for context; a constant prediction has no
  defined AUC/ranking ability).

## Result: clean falsification

| Predictor | AUC |
|---|---|
| LLM-with-analogs | **0.6770** |
| Cohort-mean (no LLM) | **0.6881** |
| Population base rate | n/a (constant) |

The LLM does not beat the cohort mean — it's numerically slightly
*behind* it. A 10,000-draw bootstrap on the paired difference confirms
this isn't just a favorable noise floor for the null: median difference
(LLM − cohort) = **−0.0111**, 95% CI **[−0.0545, 0.0305]** — comfortably
contains zero, and only 30.4% of bootstrap draws even favor the LLM.

**This is exactly the pre-registered failure condition.** Both store
context (which analogs to draw) and the simple average of their outcomes
already capture essentially all the usable signal; the LLM's additional
reasoning over the same 20 data points adds nothing beyond that.

## Inspected the reasoning — it's coherent, just not better

The LLM is not hallucinating or ignoring the analogs — it genuinely
identifies the most similar ones and reasons from them sensibly:

> "All 20 analogs, including the closest matches on bill amount (e.g.,
> #12 at 2703, #17 at 2600, #3 at 2350) and day_of_week=3 (e.g., #19),
> returned False, so no analog supports a second purchase within 60
> days." (correct call, true label False)

> "The new customer's unpaid balance (owe_amt=1550, pay_amt=0) and
> day_of_week=4 most resemble analogs 2, 3, 5, 10, and 17 ... which split
> 3 returned vs 2 not returned ... yielding roughly even odds." (true
> label False; LLM said 0.5 — genuinely uncertain, not wrong so much as
> unresolved, which is itself a fair read of a genuinely mixed cohort)

This is the expected result when a cohort of 20 is already a reasonably
tight, representative local sample: a careful qualitative read of 20 data
points and a numeric average of the same 20 data points tend to converge,
because there isn't much more information to extract from so few,
already-relevant analogs. The LLM's selectivity (picking out the "closest"
analogs by amount/day-of-week rather than using all 20 equally) didn't
translate into an accuracy edge over the unweighted average in this test.

Mock controls (`mock-beats-cohort`, `mock-same-as-cohort`) confirm the
test construction itself is sound: a deliberately-better mock model shows
AUC 1.00 vs. the real cohort-mean's 0.69, and a deliberately-cohort-mimicking
mock model lands at 0.68 — close to the real cohort-mean baseline, as
designed — so the real result landing essentially on top of the cohort
baseline is a genuine finding, not a test that can't discriminate.

## Conclusion

**Both halves of Option (a) are now closed by their cheap pilots.** The
sequence-encoder half failed to beat XGBoost even directly
(`llm_sequence_pilot/`); the cold-start half fails to beat a plain
GROUP BY. Running tally across the project: **4 null/negative LLM
results** (merchant-name-knowledge prior, drift-policy override,
sequence-encoder, cold-start analog-reasoner) **against 1 positive**
(SHAP explanation layer, `llm_explanation_trial/`). The cheap-pilot
discipline caught all four before any investment in their full
infrastructure builds.

**Recommendation: close Option (a) entirely.** No full build was done
for either half — this pilot, like the others, used only raw data and
chat completions, no embedding/encoder infrastructure, no PCA/autoencoder
controls, no matched-cohort validation harness. Tier S (the SHAP
explanation layer) stands as the thesis's one validated LLM contribution.
