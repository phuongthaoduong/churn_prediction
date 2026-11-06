# Option (a) Cheap Pilot: LLM-as-Sequence-Reasoner — Findings

Date: 2026-11-06. Model: `deepseek-flash` (DeepSeek-V4.1-Flash-1017),
temperature 0, `reasoning_effort: "none"`. 500/500 calls succeeded (250
customers × chronological/shuffled variants).

## What this tested

Before building the full Option (a) infrastructure (tokenization design,
embedding extraction, PCA/autoencoder control, matched-cohort cold-start
reasoner — see `docs/LLM_Augmentation_Execution_Plan.md`), this ran a
cheap proxy test of the core hypothesis behind its most novel half (the
sequence encoder): can an LLM, reasoning directly over a customer's raw
transaction sequence via chat completions (no embedding infrastructure
needed), produce real predictive signal — and does that signal depend on
genuine sequence *order*, not just aggregate statistics?

250 customers with "sufficient history" (≥5 transactions in the 90-day
observation window) were sampled uniformly at random (not stratified by
the existing model's own score, to keep the XGBoost comparison fair) from
`pipeline/shap_analysis.py`'s existing sample. Each one's full real
transaction history (capped at the most recent 100 transactions) was
pulled from the cleaned sales data and given to the LLM as a plain
sequence of `(days_before_pp, bill_amt, pay_amt, owe_amt)` tuples — no
customer identity, no merchant name, no feature names, nothing beyond the
raw numbers. Run twice per customer: real chronological order, and the
same transactions shuffled. Full code: `llm_sequence_pilot/`.

## A process note worth keeping on record

The first run of this pilot failed almost entirely: 26/60 calls errored
with `IncompleteRead(0 bytes read)` after ~60+ seconds each, and the
whole batch was on pace to take several hours. Diagnosed by hand: a
trivial "say hello" call revealed `deepseek-flash` returns a
`reasoning_content` field by default — it's in "thinking mode," and for
a complex 100-transaction reasoning prompt that mode apparently runs long
enough to hit a server/proxy-side timeout before any response bytes are
sent. Empirically tested four candidate fixes directly against the live
API (the documentation site wouldn't render via fetch or curl); adding
`"reasoning_effort": "none"` to the request body disabled thinking mode
and cut the same prompt's response time from 63s-and-failing to 2.1s and
succeeding. All runs reported below used this fix. Worth carrying into
any future calls to this model on non-trivial prompts.

## Result: clear negative, on the core hypothesis

| Metric | AUC |
|---|---|
| LLM, chronological order | **0.5506** |
| LLM, shuffled order | **0.5145** |
| Existing XGBoost, same 250 customers | **0.7011** |

The LLM is barely better than random guessing (0.55 vs. 0.50) and far
below the existing model's accuracy *on the exact same customers* — this
is an apples-to-apples comparison, not a different population. There is
a small, real order effect (chronological beats shuffled by 0.036 AUC),
so the LLM is using *some* genuine sequence structure, not purely
reading off aggregate stats — but the effect is small and the overall
accuracy is weak regardless of order.

**Caveat on the XGBoost baseline:** 0.70 here is lower than the main
study's ~0.85 average because this is a single small (n=250), restricted
(frequency≥5 only) sample, not the full 12-batch evaluation — it is not
evidence the main model is worse than reported elsewhere, just the right
number for this specific fair comparison.

## Why it fails: inspected reasoning, not just the number

The LLM's reasoning is coherent and genuinely grounded in the numbers
given — it is not hallucinating — but it's miscalibrated. Example
(customer L2_0, true label 0 = did not churn, XGBoost correctly scored
0.084, LLM scored 0.65):

> "The customer had regular transactions through day 85 but then a
> cluster of unpaid bills (days 53, 51, 45, 42, 35 totaling 13,890 owed)
> followed by only one payment at day 25, and no activity in the 25 days
> since, suggesting elevated churn risk despite the long prior history."

This is a sensible-sounding read of the numbers that turned out to be
wrong — the LLM over-weights unpaid balances and recent silence as churn
signals in a way that doesn't match this dataset's actual patterns well
in aggregate, even though XGBoost (trained on the real label distribution)
does better with the same kind of information, aggregated differently.
One case went the other way (L2_47, true label 1 = did churn, XGBoost
badly underscored it at 0.27, LLM correctly flagged it at 0.85, citing
"all 12 transactions show pay_amt=0.0 with total owed of 28,912.0") — so
the LLM isn't uniformly worse on every case, it's just not reliably
better in aggregate.

## Conclusion

The core hypothesis behind Option (a)'s sequence-encoder half does not
hold up even in this most generous, direct test (full chat-based
reasoning over the raw sequence, no embedding bottleneck at all). This is
now the third null/negative LLM result in the project
(`llm_prior_threshold_trial/`, `llm_drift_strategy_trial/`, this one),
against one positive (`llm_explanation_trial/`). The cheap-pilot
discipline did its job: it caught this before any investment in
tokenization design, embedding extraction, or the PCA/autoencoder control
pipeline that the full build would have required.

**Recommendation: do not build the full sequence-encoder half of Option
(a).** The cold-start analog-reasoner half is a separate, untested
mechanism (different approach — reasoning from other customers' analogs,
not from the thin customer's own sequence) and isn't automatically killed
by this result; whether to still pilot it, or close Option (a) entirely
and consolidate on Tier S as the thesis's LLM contribution, is the next
decision.
