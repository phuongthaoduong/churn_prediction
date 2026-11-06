# LLM Drift-Policy Diagnostic: Rationale Findings

Date: 2026-11-03. Model: `deepseek-flash` (DeepSeek-V4.1-Flash-1017),
temperature 0, 5 reps per decision point. 220/220 calls succeeded.

## What this tested

Before building a full live re-simulation of an LLM-driven update strategy
(Option B / "S5" in `docs/LLM_Augmentation_Execution_Plan.md`), this ran a
cheap diagnostic: feed the LLM the *exact* drift evidence S4's rule
(`psi_fired OR auc_fired`, PSI threshold 0.2) actually saw at each of its 11
real decision points across both schemes and both maturity modes (44
decision points total), and see whether its reasoning diverges from the
rule in a meaningful way. The prompt explicitly told the model that the
final evaluation batch (2026-03-01) is known to carry a measurement
artifact — a sharp late-batch PSI spike in `recency` caused by the data
boundary, not necessarily genuine drift — and asked it to take that into
account. Full prompt: `llm_drift_strategy_trial/02_llm_decision.py`.

## Headline result

Of 44 decision points, 40 are trivial: PSI is well under 0.2 and the LLM
said "hold" every time, with no meaningful variation in reasoning style
(see "Ordinary decisions" below). The only non-trivial moment in the
entire 12-month evaluation is the final batch, where S4's rule fired
"retrain" in all 4 scheme/mode combinations because of the PSI spike.

**At all 4 of those points, across all 5 reps each (20/20), the LLM said
"hold," explicitly citing the artifact.** This is a deliberate, consistent,
explained override of the rule — not noise. Representative quotes (full
text in `llm_drift_strategy_trial/data/llm_raw/deepseek-flash.jsonl`):

> "The max PSI of 0.5348 is concentrated in recency on the known final-batch
> artifact, while all other monitored features are <=0.0363 and recent
> own-label AUC remains stable at 0.8506, so no retraining or reference
> reset is warranted." — scheme/L1, rep 0, confidence 4

> "Although recency PSI is 0.3798 above the 0.2 trigger, it is a known
> late-batch measurement artifact in the final 2026-03-01 batch, other
> feature PSIs are low (≤0.0746), and mature AUCs remain stable around
> 0.854–0.857, so no retrain or reference reset is warranted." —
> equalized/L2, rep 1, confidence 5

Every one of the 20 final-batch responses follows this same structure:
name the PSI value, name that it's concentrated in `recency` specifically
(correctly identifying the single feature driving the spike — it wasn't
told which feature would be high, only given the top-5 list), invoke the
artifact framing, and cite AUC stability as corroborating evidence that
nothing is actually wrong. This is coherent, grounded reasoning over the
numbers it was given — it is not hallucinating external facts about the
merchants or customers involved.

## Ordinary decisions (for contrast)

Away from the final batch, the LLM's reasoning reduces to restating the
rule it was told about — including at a near-miss case where PSI was
close to (but under) the 0.2 trigger:

> "Max PSI is 0.1738, below the 0.2 trigger, and the most recent matured
> batch AUC is 0.847 with only 1 month since training, so there is no need
> to retrain or reset the reference." — scheme/L1, 2025-05-01, psi_max
> 0.1738, confidence 5

So across the 40 trivial points, the LLM isn't adding anything S4's rule
doesn't already do — it's functioning as a plain-language restatement of
the threshold comparison. The artifact override at the final batch is the
only point where it did something the rule couldn't.

## Was the override actually right?

Checked against real data that already exists: `S1` (the never-retrain
strategy) is a genuine counterfactual for "what if the model had held
instead of retraining" at that exact batch, since S1 and S4 share the
identical model up through the second-to-last batch and diverge only at
the final one. Comparing `auc_shared` on 2026-03-01
(`experiment_output/metrics_per_batch.csv`):

| Mode | Scheme | S1 (held) | S4 (retrained) | Retraining helped? |
|---|---|---|---|---|
| equalized | L1 | 0.8294 | 0.8385 | Yes, +0.0091 |
| equalized | L2 | 0.8318 | 0.8351 | Yes, +0.0033 |
| scheme | L1 | 0.8429 | 0.8359 | **No, −0.0070** |
| scheme | L2 | 0.8318 | 0.8351 | Yes, +0.0033 |

Retraining in response to the PSI spike was net *better* than holding in 3
of 4 conditions. The LLM's uniform "it's just an artifact, hold" call was
therefore **wrong on balance** — real signal was mixed in with the
artifact, and the rule's reflexive retrain captured more of that real
signal than the LLM's more cautious read did, most of the time. (One
honest caveat: these are single-batch deltas, noisier than the whole-year
averages the main factorial study reports — not directly comparable in
scale to the <0.1pp contrasts found there.)

## Confidence scores were uninformative

210/220 responses were confidence 5 (the maximum), the remaining 10 were
confidence 4. The model never expressed meaningfully lower confidence
anywhere, including at the near-miss case (PSI 0.1738) or at the final
batch where it was overriding the existing rule. Confidence as elicited
here carries no discriminating signal — it isn't tracking the model's own
uncertainty in any usable way.

## Conclusion

The LLM did exactly one interesting thing in this whole diagnostic: it
recognized the specific artifact-causing feature unprompted and overrode
the rule consistently and coherently because of it. That override turned
out to be the wrong call most of the time, per data that already exists.
Combined with the structural problem that S4's rule only produces one
non-trivial event across the entire 12-month evaluation (so there is
almost no variation anywhere in this dataset for a smarter decision
policy to prove itself on), this diagnostic does not support investing in
the full live re-simulation (building S5 as an actual sequential retrain
policy with its own compounding model trajectory). Recommendation: close
this direction and move to Option (a) from the execution plan.
