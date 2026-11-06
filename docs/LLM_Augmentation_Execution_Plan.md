# LLM Augmentation Execution Plan

Status: CONCLUDED. Both Option (a) and Option (b) were tested via cheap
pilots and closed (null/negative results); Tier S (SHAP explanation layer)
is the thesis's one validated, surviving LLM contribution. Written
2026-11-02, revised 2026-11-02 (v2 — restarted on merit, not repo-fit; see
§0), revised 2026-11-04 after Option (b)'s diagnostic, revised again
2026-11-04 after building and testing Tier S, revised again 2026-11-06
after closing both halves of Option (a).

## 0. How this plan was actually built (methodology note, kept for honesty)

This went through two passes, and the difference matters for how much to
trust each idea below:

- **Pass 1** picked candidates partly because they fit cleanly into the
  existing pipeline (reused `strategies.py`, PSI infra, etc.). That's a real
  bias: "easy to bolt on" is not the same as "best idea."
- **Pass 2** (this version) restarted from zero: two fresh agents with *no
  knowledge of this repo's code* were asked, independently, what the single
  best LLM application for a non-contractual-churn thesis would be, judged
  purely on business value or technical novelty — explicitly told to ignore
  whether it's convenient to retrofit into any existing codebase.
- I then checked each Pass-2 idea against real, already-gathered evidence
  from this repo (the finished baseline results, the error-analysis
  artifacts, the SHAP commitment, the single-transaction-customer
  profiling) — as a *feasibility/grounding check*, not as the ranking
  criterion. Where repo evidence strengthens or weakens an idea, that's
  stated explicitly and separately from the independent ranking.

So: rank order below comes from Pass 2. Feasibility notes come from
checking Pass 2's ideas against this repo afterward.

## 1. What already exists (unchanged from Pass 1 — this part is just fact)

**The non-LLM baseline is finished.** Steps 1-8.4 of `pipeline/` ran the
full 2×4 factorial (L1/L2 labeling × S1-S4 update strategy) prequentially
over 12 evaluation batches, with a confirmatory ART-ANOVA. Results are in
`experiment_output/` and `docs/reports/step8/weekly_report_step8.md`.
Nothing to rebuild for the "baseline" requirement.

Key numbers: AUC_shared ranges 0.8510-0.8527 across all 8 conditions (15bp
spread — effects in this dataset are small everywhere). S2 (periodic
retrain) is the best strategy; S4 (drift-triggered) is the worst and fires
only once per scheme across the whole evaluation year. L2 beats L1 by a
small but significant margin. The scheme×strategy interaction is not
significant. The report's own stated open question: "whether a more
sensitive threshold would change S4's standing... outside current scope."

**Already tried and killed:** `llm_prior_threshold_trial/` asked an LLM
what it "knows" about a named merchant to prime the L2 threshold. Null
result (correlation ≈0, indistinguishable from a shuffled control). Closed.
Nothing below repeats this mechanism.

## 2. Ranked candidates (Pass 2 — merit first, repo-fit checked after)

### Tier S — do this regardless of which primary bet you pick

**SHAP-to-narrative explanation/action layer — BUILT AND DONE, 2026-11-04.**
Feed the frozen baseline model's real SHAP contributions for a prediction
into an LLM, which writes a plain-language "why this customer is flagged"
explanation grounded entirely in those numbers (never in outside knowledge
about the customer). Test faithfulness by ablating the top SHAP feature
and checking the explanation changes accordingly.

**Result: clean positive, the best LLM result in the project so far.**
Built `pipeline/shap_analysis.py` (new Step 9, `shap` added to
`requirements.txt`) and `llm_explanation_trial/`. Step 9 resolved the
open question itself: `ipi_std` is rank-1 by SHAP in both schemes, and the
signal is concentrated in rows where it has a real value (not in its
missingness) — genuine signal, not a NaN-routing artifact. The LLM layer
passed its exact faithfulness test at n=40: 100% of full-prompt runs
correctly cited the real top SHAP feature, 0% of ablated runs
hallucinated the removed feature, with mock faithful/unfaithful controls
confirming the test itself can detect failure. Full findings and example
transcripts: `docs/reports/llm_explanation_trial/findings.md`. Caveats:
n=40 is pilot-scale, not thesis-grade; only one ablation condition tested.

- *Why it's safe to commit to:* independently picked as the top idea by
  **both** business-merit passes (repo-aware and repo-blind) and converges
  with the original technical fork's "explanation-faithfulness" idea. It
  structurally cannot repeat the falsified "ask the LLM what it knows"
  mistake, because every claim traces to a real, computed number.
- *Repo grounding (checked, not assumed):* this isn't a new scope item —
  `docs/reports/step7/weekly_report_step7.md` explicitly commits to "SHAP
  analysis in Phase 2" to resolve a specific open question (whether
  `ipi_std`'s status as top-gain feature despite 63.5% NaN is real signal
  or a NaN-routing artifact). You already owe this to the thesis
  independent of any LLM work.
- *Honest caveat:* `grep -rn "import shap"` across the whole repo returns
  zero matches. No SHAP code exists yet — this is being built from
  scratch, not a quick extension of existing tooling.
- *Failure mode:* none that threatens the thesis. Whether the answer is
  "faithful" or "not faithful," it's a valid, reportable finding.

### Tier A — primary novel bet

**Option (a), ambitious — LLM as behavioral-sequence encoder, paired with
LLM as cold-start analog-reasoner.**

**Cheap pilot run 2026-11-06 on the encoder half — CLOSED.** Before
building the full tokenization/embedding/control infrastructure, tested
the core hypothesis directly: can an LLM reasoning over a customer's raw
transaction sequence (chat completions, no embedding infra) beat or match
XGBoost, and does accuracy depend on real sequence order? Result on 250
customers, same-subject comparison: LLM chronological AUC 0.5506, LLM
shuffled AUC 0.5145, existing XGBoost on the same customers 0.7011. The
LLM is barely above random and far below the existing model; there is a
small genuine order effect (chronological beats shuffled) but it's not
enough to matter. Full findings and reasoning-text examples:
`docs/reports/llm_sequence_pilot/findings.md`. **The sequence-encoder
half of Option (a) is closed — do not build the full embedding/PCA
control pipeline.** The cold-start analog-reasoner half is untested and
not automatically killed by this (different mechanism), but see the
updated recommendation below.

- *Mechanism (encoder):* serialize each customer's raw event sequence
  (inter-transaction gaps, amounts, product/category tokens, debt ratio —
  all numeric, no semantics needed) and use an LLM as a sequence
  representation learner, producing embeddings tested against hand-crafted
  RFM features. Controls: (i) shuffle token order — if performance is
  unchanged, the LLM isn't using sequence structure; (ii) a non-LLM encoder
  (PCA/autoencoder) matched for embedding dimensionality — if the LLM
  doesn't beat this, its pretrained priors add nothing a cheaper method
  can't match.
- *Mechanism (cold-start pairing):* for single-transaction customers, feed
  the LLM the real first-transaction signatures of many customers *with
  full histories*, have it find behavioral analogs in-context (grounded in
  real data, never in business-name knowledge), and use the analog
  cohort's actual subsequent behavior as a pseudo-RFM prior.

  **Cheap pilot run 2026-11-06 — CLOSED.** 150 true cold-start customers
  (known only by their first transaction), each given 20 real same-store
  analogs (first-transaction signature + true 60-day-return outcome).
  LLM-with-analogs AUC 0.6770 vs. plain cohort-mean (no LLM) AUC 0.6881 —
  the LLM does not beat a GROUP BY, which was the pre-registered failure
  condition for this exact idea. 10,000-draw bootstrap on the paired
  difference: median −0.0111, 95% CI [−0.0545, 0.0305] — contains zero.
  Full findings and reasoning-text examples:
  `docs/reports/llm_coldstart_pilot/findings.md`. **Both halves of Option
  (a) are now closed by their cheap pilots — no full build was ever
  done for either.**
- *Why this is the most novel option on the table:* rated #1 by the
  repo-blind technical pass specifically because it gives the LLM a
  structurally different role — representation learning, not an advisor
  bolted onto XGBoost. This is the only option that plausibly produces a
  genuinely new modeling capability.
- *Honest caveat the repo-blind agent didn't have:* median transactions
  per customer in this dataset is **2**, and only ~28-35% of customers
  clear even a lenient 4-5-transaction "sufficient history" bar (measured
  already in `profiling_output/step1_profiling/profiling_report.json`,
  key `1.4_sufficient_history`). The sequence-encoder half of this option
  has a real, narrow natural scope — it won't help most of your customer
  base on its own. This is exactly why it's paired with the cold-start
  reasoner: together they cover both ends of the history-length
  distribution, but that also means you're committing to building and
  evaluating *two* non-trivial pieces, not one.
- *Risk, stated plainly:* highest effort and timeline risk of everything
  in this plan. Needs a tokenization design, an embedding-extraction setup
  (likely a different cost/infra model than the chat-completion API priced
  earlier — encoder-style usage is usually far cheaper per customer than
  chat completions, but hasn't been estimated), and two separate
  falsification pipelines.

### 2a. Option (b) — tested and CLOSED 2026-11-04

**Option (b) was: LLM as drift/update-policy reasoner**, replacing or
extending S4's brittle `psi_fired OR auc_fired` rule
(`pipeline/strategies.py:51-59`) with an LLM reading a compact text summary
of drift evidence and deciding `hold`/`retrain`/`reset-reference`. It
converged independently from two directions (repo-aware pass: S4 is
demonstrably broken; repo-blind pass: "LLM reasons over drift diagnostics"
without knowing that) and was cheap enough ($0.02-$0.12 estimated) to just
test directly rather than keep debating on paper.

**What was actually run, before committing to the full live
re-simulation:** a cheap diagnostic (`llm_drift_strategy_trial/`) fed
`deepseek-flash` the *exact* drift evidence S4's rule saw at all 44 of its
real decision points (both schemes, both maturity modes, 5 reps each, 220
real calls, 0 failures). Full results and every rationale quote:
`docs/reports/llm_drift_strategy_trial/rationale_findings.md`.

**Result: closed.**
- Of 44 decision points, 40 are trivial (PSI far under threshold) — the
  LLM just restated the rule, adding nothing.
- The only non-trivial moment in the whole 12-month evaluation is the
  final batch, where S4's rule fired "retrain" because of a PSI spike that
  is a known data-boundary artifact. At all 4 of those points (both
  schemes × both modes), across all 5 reps each, the LLM consistently
  overrode the rule and said "hold," explicitly and correctly identifying
  that the spike was concentrated in `recency` and citing the artifact.
  This was a real, deliberate, coherent override — not noise.
- **Checked against real data that already exists** (S1, the never-retrain
  strategy, is a genuine counterfactual for "what if we'd held"): retraining
  (what S4's rule did, what the LLM said not to do) was actually *better*
  in 3 of 4 scheme/mode conditions (+0.003 to +0.009 AUC) and worse in only
  1 (-0.007 AUC). The LLM's override was **wrong on balance**.
- Confidence scores were uninformative throughout (210/220 responses at
  the maximum, 5; never lower than 4, including at the near-miss case and
  at the point where it was overriding the rule).
- Structural problem underneath all of this: S4's rule produces exactly
  one non-trivial event in the entire 12-month evaluation. There is almost
  no variation in this dataset for any decision policy — rule-based or
  LLM — to meaningfully prove itself on. Building the full live
  re-simulation (actually refitting XGBoost at whatever points the LLM
  chooses, which is what a real S5 strategy would require) would hit this
  same wall.

**Decision: do not build the full S5 re-simulation.** The cheap diagnostic
did its job — it caught a likely dead end before the expensive engineering
effort (tokenization of per-batch state, sequential refit loop, two-arm
statistical comparison) was spent on it. Tier A now has one live option.

### Tier C — dropped from this plan

**LLM reranker for borderline-probability customers** (was "Fallback 1" in
Pass 1). Repo evidence kills this on its own terms, not just on cost:
`experiment_output/error_analysis/` shows the dominant false-positive cause
(39-64% of FPs) is "hidden history" — customers whose relevant purchases
fall outside the 90-day observation window. An LLM given the same visible
features has the identical blind spot XGBoost already has; it cannot
reason its way to information it was never given. Only 7-9% of errors are
even plausibly fixable by better reasoning over existing features, and the
report's own ceiling analysis caps the realistic upside at +1.2 to +3.2pp
AUC *even with full information*. Separately, the real borderline band
(predicted probability in [0.4, 0.6]) contains ~53,000 customers per batch
— not the "low hundreds to a few thousand" Pass 1 assumed, a ~100x cost
miscalculation. Removed.

### Permanently closed — do not revisit

**LLM parametric-knowledge elicitation from merchant name/identity**
(including any cold-start variant that reduces to the same move — one of
the Pass-2 business-merit ideas correctly flagged its own weaker idea as
this same failure mode independently). Already falsified by a real pilot
with a pre-registered stop rule. Nothing resurrects this.

## 3. Decision flow

```
Tier S (SHAP explanation layer): DONE, 2026-11-04. Clean positive result.
       ↓
Option (b) (drift-policy reasoner): tested via cheap diagnostic BEFORE
the full build, 2026-11-04. CLOSED — see §2a. No full re-simulation built.
       ↓
Option (a), encoder half: cheap pilot run 2026-11-06. CLOSED — LLM
barely above random (AUC 0.55) vs existing XGBoost (0.70) on the same
250 customers. No full embedding/control pipeline built.
       ↓
Option (a), cold-start half: cheap pilot run 2026-11-06. CLOSED — LLM
(0.677 AUC) did not beat plain cohort-mean (0.688 AUC), the exact
pre-registered failure condition. No full build done.
       ↓
Option (a) is now fully closed, both halves. Tier S (SHAP explanation
layer) is the thesis's one validated LLM contribution.
```

**Final tally: 4 null/negative LLM results** (merchant-name-knowledge
prior, drift-policy override, sequence-encoder, cold-start analog-
reasoner) **against 1 positive** (SHAP explanation layer). Every one of
the four negatives was caught by a cheap pilot before any investment in
full infrastructure — none of them cost more than a focused afternoon
and a few hundred API calls at a few cents to a few tens of cents each.

**Honest pattern worth weighing:** two straightforward LLM ideas have now
been tested on this data and both came back null/net-negative (the
merchant-name-knowledge prior, and the drift-policy override). Option (a)
is mechanistically different from both (representation learning, not
knowledge elicitation or rule-arbitration), so this isn't a reason to
abandon it — but it's a reason to pilot cheaply before committing the full
build, not to assume Option (a) is owed a win.

## 4. What's not yet done, regardless of path

- Tier S: done (see §Tier S above and `docs/reports/llm_explanation_trial/findings.md`).
  Remaining if pursued further: scale past the n=40 pilot, test more
  ablation conditions.
- Option (a), both halves: closed by their cheap pilots (see
  `docs/reports/llm_sequence_pilot/findings.md` and
  `docs/reports/llm_coldstart_pilot/findings.md`). No tokenization,
  embedding, or encoder infrastructure was ever built for either half —
  both pilots used only raw data and chat completions.
- Option (b): closed. The diagnostic (`llm_drift_strategy_trial/`) only
  reused `experiment_config.py`/`drift_utils.py` read-only against
  already-finished artifacts — nothing in `pipeline/` was modified, and no
  full S5 strategy was ever added to `strategies.py`/`run_experiment.py`/
  `experiment_config.STRATEGIES`. The locked 2×4 factorial and its
  confirmatory ART-ANOVA (`docs/reports/step8/statistical_analysis.py`)
  are untouched.
- Tier S and Option (b)'s diagnostic are both real completed results now
  (see §Tier S and §2a). Only Option (a) remains unstarted.
