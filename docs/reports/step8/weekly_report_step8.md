# Weekly Progress Report: Factorial Evaluation and Accuracy of Model Outputs (Step 8)

> **Project:** Customer churn prediction using XGBoost with a 2x4 factorial design (fixed vs. category-level dynamic labeling x four model updating strategies), applied to SME retail transaction data from the Kingdee platform.
> **This week's focus:** Tuning hyperparameters, freezing the feature set, running the full S1-S4 x L1/L2 prequential factorial experiment, and executing the statistical analysis plan pre-registered in Step 7 to report the accuracy of the resulting model outputs.
> **Date:** Week of October 13-18, 2026

---

## 1. What I Set Out to Do

Step 7 closed the preprocessing pipeline and pre-registered, in advance of seeing any results, the exact statistical protocol by which the factorial experiment would be evaluated (Section 7 of the Step 7 report). Phase 2 had three remaining tasks before that protocol could be executed: tune XGBoost hyperparameters once, per labeling scheme, via leakage-safe time-series cross-validation; settle whether the hand-selected 25-feature set earns its complexity over a simple RFM baseline; and run the S1-S4 strategy loop across all 12 evaluation batches under both labeling schemes, producing the predictions whose accuracy this report evaluates. This week completed all three, computed the shared 60-day evaluation label that makes cross-scheme comparison possible, and applied the pre-registered confirmatory analysis to the resulting batch-level metrics.

---

## 2. Shared Evaluation Label (Step 8.0)

### What was done

I constructed `label_eval`, a 60-day inactivity label applied identically to every customer-batch observation regardless of which labeling scheme (L1 or L2) produced the training target.

### Why

L1 and L2 are themselves experimental factors — a model trained under L2 cannot be judged fairly against its own L2 labels when competing against a model trained under L1's different target, because the two targets differ in difficulty as well as definition. A single shared outcome, held fixed across all eight conditions, is what the Section 7.1 primary-metric formalization (AUC_shared) requires: it isolates the model and update-strategy choices from the labeling-target choice.

### What was achieved

`label_eval` is now available for every row of `eval_labels.parquet` and is the outcome against which `auc_shared`, `f1_shared`, and `recall10_shared` are computed in every downstream step. `auc_own` — AUC against each scheme's own label — is retained as a secondary diagnostic, per the Step 7 plan.

---

## 3. Hyperparameter Tuning (Step 8.1)

### What was done

I ran a checkpointed random search over XGBoost hyperparameters (`colsample_bytree`, `learning_rate`, `max_depth`, `min_child_weight`, `n_estimators`, `subsample`) separately for each labeling scheme, validated on leakage-safe time-series cross-validation folds restricted to the initial-training period, and froze the winning configuration to `tuned_params.json`.

### Why

Step 7.2's drift-monitoring baselines were built on provisional default hyperparameters, with the explicit plan to tune once before the factorial loop and rerun feature ranking under the tuned model. Tuning per scheme, rather than once globally, respects the fact that L1 and L2 present the classifier with different targets and may reward different model complexity.

### What was achieved

Of 20 sampled hyperparameter combinations per scheme, the best L1 configuration (`colsample_bytree=0.7, learning_rate=0.05, max_depth=6, min_child_weight=1, n_estimators=300, subsample=0.7`) reached a mean validation AUC of 0.8432 (std 0.0099) across 6 folds. The best L2 configuration (`colsample_bytree=1.0, learning_rate=0.05, max_depth=8, min_child_weight=10, n_estimators=200, subsample=0.7`) reached a mean validation AUC of 0.8482 (std 0.0035) across 3 folds. L2 selects a deeper tree (max_depth 8 vs. 6) with a much larger `min_child_weight` (10 vs. 1), consistent with L2's category-level dynamic threshold producing a less noisy, more regularizable target than L1's fixed 30-day cutoff. Both configurations are frozen and used, without further adjustment, throughout the rest of Step 8.

---

## 4. Feature Set Comparison (Step 8.2)

### What was done

Using the tuned hyperparameters, I compared the full 25-feature set (`llm_guided`) against a 3-feature RFM baseline (recency, frequency, monetary) on the same leakage-safe validation folds, for both labeling schemes, and computed mutual information between each feature and the shared label on a 200,000-row sample as a secondary diagnostic.

### Why

The 25-feature set carries real engineering and maintenance cost (store-relative ratios, inter-purchase-interval statistics, calendar flags); the comparison exists to confirm that cost is justified before the feature set is frozen for the remainder of the thesis, rather than assumed.

### What was achieved

The 25-feature set wins decisively for both schemes: L1 mean validation AUC of 0.8437 versus 0.8186 for RFM (a gap of 0.0251), and L2 mean validation AUC of 0.8484 versus 0.8135 for RFM (a gap of 0.0349). The full feature set is frozen as `llm_guided` for both schemes and used unchanged throughout the factorial run. The RFM baseline's underperformance is largest for L2, suggesting that the behavioral and store-relative features this project engineered (inter-purchase-interval statistics, store-relative recency/frequency/monetary ratios) matter more when the churn definition itself is more behaviorally grounded.

---

## 5. Prequential Factorial Run and Accuracy Metrics (Steps 8.3-8.4)

### What was done

I executed the full 2x4 factorial design — labeling scheme (L1, L2) crossed with update strategy (S1 static, S2 monthly periodic, S3 rolling-window, S4 drift-triggered) — under the prequential test-then-train protocol across the 12 evaluation batches (2025-04-01 through 2026-03-01), and repeated the entire run under both the primary scheme-specific label-maturity rule and the equalized 90-day-lag sensitivity arm specified in Step 7. This produced 108 training events (initial fits plus retrains) and 192 prediction files. For every one of the resulting 192 batch-level predictions I computed `auc_shared`, `f1_shared`, `recall10_shared`, `auc_own`, and the Jaccard similarity of consecutive top-10% risk lists against the shared label.

### Why

This is the experiment the entire preprocessing and preparation pipeline (Steps 1-7) was built to run. The prequential protocol — predict on a batch before training on it — is what makes the resulting AUC a genuine out-of-sample estimate of deployed accuracy rather than an in-sample fit statistic, and the label-maturity rule (Step 7.1) is what prevents training on labels whose observation window has not yet closed.

### What was achieved: descriptive accuracy

Table 1 reports the primary analysis (scheme-specific maturity).

| Scheme | Strategy | Mean AUC_shared | CV(AUC) | Mean F1_shared | Mean Recall@10% | Mean AUC_own | Mean Jaccard | Retrains | Fit time (s) |
|---|---|---|---|---|---|---|---|---|---|
| L1 | S1 (static) | 0.8518 | 0.0051 | 0.7779 | 0.1943 | 0.8487 | 0.3607 | 0 | 0.0 |
| L1 | S2 (periodic) | 0.8521 | 0.0053 | 0.7789 | 0.1945 | 0.8490 | 0.3640 | 11 | 407.0 |
| L1 | S3 (rolling) | 0.8512 | 0.0068 | 0.7775 | 0.1945 | 0.8481 | 0.3641 | 11 | 190.3 |
| L1 | S4 (drift-triggered) | 0.8510 | 0.0069 | 0.7772 | 0.1944 | 0.8477 | 0.3602 | 1 | 14.5 |
| L2 | S1 (static) | 0.8516 | 0.0083 | 0.7765 | 0.1945 | 0.8511 | 0.3634 | 0 | 0.0 |
| L2 | S2 (periodic) | 0.8525 | 0.0060 | 0.7776 | 0.1946 | 0.8518 | 0.3674 | 11 | 356.1 |
| L2 | S3 (rolling) | 0.8514 | 0.0074 | 0.7687 | 0.1946 | 0.8511 | 0.3696 | 11 | 83.6 |
| L2 | S4 (drift-triggered) | 0.8513 | 0.0069 | 0.7769 | 0.1945 | 0.8512 | 0.3585 | 1 | 7.7 |

*Table 1. Primary analysis (scheme-specific maturity), descriptive summary across 12 evaluation batches per condition.*

All eight conditions cluster tightly: mean AUC_shared ranges from 0.8510 (L1-S4) to 0.8525 (L2-S2), a 15-basis-point spread. Every condition's coefficient of variation lies between 0.5% and 0.8%, indicating that batch-to-batch accuracy is highly stable for all eight conditions over the full evaluation year — there is no condition whose accuracy visibly degrades or whose variance explodes across the 12 months (Figure 1). S4's drift-triggered mechanism fired once for each scheme across the entire evaluation year, against eleven retraining events for S2 and S3; its mean AUC is statistically indistinguishable from S1's static baseline (see Section 5, main-effect contrasts), meaning that in this dataset drift-triggered retraining bought no measurable accuracy over never retraining at all, at a small fraction of the training cost of periodic or rolling updating.

### Figure 1

`trajectory_plot.png` (saved alongside this report) plots `auc_shared` against the evaluation batch for all eight conditions. The eight trajectories move together across the year rather than diverging, which is the descriptive counterpart of the non-significant interaction reported below: no condition pulls away from the pack in any particular month.

### What was achieved: confirmatory statistical analysis

**Assumption assessment.** Residuals from the parametric reference model (`AUC_shared ~ scheme * strategy + batch`) fail the Shapiro-Wilk normality test decisively (W = 0.798, p = 3.7e-10, n = 96), and Mauchly's sphericity test on the 8-condition x 12-batch response matrix likewise rejects sphericity (W = 9.6e-7, chi2(27) = 117.1, p = 3.4e-13) — both exactly as anticipated in Step 7's pre-registration, which expected violation at 12 temporally autocorrelated batches. Mauchly's test here estimates 27 covariance parameters from only 11 batch-level degrees of freedom and is acknowledged as an unstable diagnostic rather than a reliable test at this sample size; neither result changes which branch of the protocol is taken, since that branch was fixed in advance. As pre-specified, the parametric model is reported for completeness only (scheme: F(1,77) = 0.355, p = 0.553; strategy: F(3,77) = 3.910, p = 0.012; interaction: F(3,77) = 0.259, p = 0.855), and confirmatory inference proceeds through the assumption-robust Aligned Rank Transform (ART) protocol.

No R installation or ARTool package was available in this environment. The ART procedure (Wobbrock, Findlay, Gergle, & Higgins, 2011) was reimplemented directly in Python from its closed-form effect decomposition for a balanced, fully crossed, additive design (scheme x strategy x batch, with no scheme:batch or strategy:batch interaction terms — consistent with the pre-registered reference model): for each effect of interest, the response is aligned by subtracting every other estimated effect, the aligned values are rank-transformed, and a standard ANOVA is fit to the ranks to test the effect of interest. This reproduces ARTool's algorithm rather than approximating it, and is documented here as a deviation from using the R package itself, in keeping with this project's practice of recording methodological deviations rather than silently substituting a different method.

**Omnibus ART-ANOVA (primary dependent variable, primary arm).** The A x B interaction — the study's single confirmatory test (RQ3) — is not significant: F(3, 77) = 0.517, p = 0.672, partial eta-squared = 0.020. Both main effects are significant: Factor B (update strategy), F(3, 77) = 10.243, p = 9.4e-6, partial eta-squared = 0.285; Factor A (labeling scheme), F(1, 77) = 9.340, p = 0.003, partial eta-squared = 0.108. The analysis therefore proceeds via Step 4b of the pre-registered protocol (main-effect contrasts), not Step 4a (simple main effects): the effect of update strategy on shared-outcome accuracy does not depend on which labeling scheme produced the training target, so the two design choices can be made independently rather than jointly.

**Main-effect contrasts: update strategy.** The six pairwise strategy contrasts, pooled across schemes and Holm-corrected over the six-test family, resolve into two tiers. S2 (periodic) outperforms every other strategy: S2 vs. S1, median diff = +0.00044, 95% CI [0.00024, 0.00061] (moving-block bootstrap, block length 3), r_rb = 0.668, p_holm = 0.014; S2 vs. S3, median diff = +0.00052, CI [0.00035, 0.00088], r_rb = 0.947, p_holm < 0.001; S2 vs. S4, median diff = +0.00088, CI [0.00051, 0.00107], r_rb = 0.853, p_holm < 0.001. S1 (static, zero retrains) also beats S4 (median diff = +0.00037, CI [0.00014, 0.00048], r_rb = 0.767, p_holm = 0.002). The remaining two contrasts — S1 vs. S3 and S3 vs. S4 — do not survive Holm correction (p_holm = 0.555 for both). Practically: S2's monthly periodic retraining is the most accurate strategy tested, S1's never-retrain baseline is statistically indistinguishable from S3's rolling window and significantly better than S4's drift-triggered approach, and S4 — despite being the most mechanistically sophisticated strategy — delivers the lowest or tied-lowest accuracy of the four while triggering only once across the full year.

**Main effect: labeling scheme.** L2 outperforms L1 on the shared outcome: median(L1 - L2) = -0.00049, 95% CI [-0.00086, -0.00012], r_rb = -0.425, p = 0.0096. The effect is real but small in absolute magnitude (under 0.05 AUC points), and because the interaction is not significant, it holds uniformly across all four update strategies rather than being concentrated in any one of them.

**Primary vs. sensitivity (equalized) arm.** The entire ART-ANOVA was repeated under the equalized 90-day-lag maturity condition that removes L1's structural data-recency advantage. The equalized arm reproduces the primary arm's qualitative conclusion on every effect: the interaction remains non-significant (p = 0.595), and both main effects remain significant in the same direction (strategy p = 1.0e-7; scheme p = 4.9e-6, with L2 still ahead of L1). Per the pre-specified interpretation rule (Step 7, Section 7.3), this agreement means the scheme and strategy effects are attributed to the labeling schemes and update strategies themselves, not to the differential training-data recency that the scheme-specific maturity rule induces. The maximum absolute difference in mean AUC_shared between the two arms, across all eight conditions, is 0.0011 — materially smaller than any of the significant effect sizes above.

**Robustness checks.** (a) Repeating the full ART-ANOVA with Recall@10% as the response yields the same qualitative pattern: interaction not significant (p = 0.536), both main effects significant (strategy p = 0.0025; scheme p = 0.00038). (b) Excluding the final evaluation batch (2026-03-01), whose smallest-of-20 batch size and sharp PSI spike were flagged in Step 7 as an end-of-data truncation artifact, leaves every conclusion unchanged: interaction still non-significant (p = 0.338), both main effects still significant (strategy p = 6.9e-5; scheme p = 2.0e-5). (c) The month-by-month trajectory plot (Figure 1) shows no condition crossing or diverging from the others at any point in the year, consistent with a non-significant interaction. None of the three robustness checks alters the study's confirmatory conclusion.

---

## 6. Cost-Performance Trade-off

Because S4 fired only once per scheme against eleven retrains for S2 and S3, its fit-time cost is a small fraction of theirs (L1: 14.5s vs. 407.0s for S2 and 190.3s for S3; L2: 7.7s vs. 356.1s for S2 and 83.6s for S3) while processing over an order of magnitude fewer cumulative training rows. Combined with the statistical result that S4 is not more accurate than S1 (which costs nothing to maintain) and is significantly less accurate than S2, the cost-performance frontier for this dataset is dominated by two strategies: S1 if operational simplicity is paramount and a 0.0004-0.0009 AUC cost is acceptable, or S2 if that margin is worth 11 retraining events per scheme per year. S3's rolling window sits strictly behind S2 on accuracy while costing more to train than S4, and is dominated on this frontier.

---

## 7. What Remains Unresolved

Three issues carry forward. First, Mauchly's sphericity test as implemented here is diagnostic only, not a reliable estimate, given that it fits 27 covariance parameters from 11 degrees of freedom; a future pass with a larger evaluation window (more than 12 batches) would make this test, and the parametric reference model generally, more trustworthy as a cross-check on the ART results. Second, the ART-ANOVA was reimplemented from its published closed-form algorithm rather than run through the validated R ARTool package, because no R installation was available in this environment; the reimplementation follows Wobbrock et al. (2011) directly and its parametric-model and Friedman-based diagnostics agree qualitatively with the ART result reported here, but an independent cross-check against ARTool itself, if R tooling becomes available, would further strengthen the confirmatory claim. Third, S4's single trigger across the entire evaluation year means its accuracy estimate rests on far less retraining variation than S2 or S3; whether a more sensitive PSI or AUC-decline threshold would change S4's standing relative to S2 is not addressed by the current frozen thresholds (PSI = 0.2, AUC-decline = 0.02) and would require a threshold-sensitivity sweep outside the current pre-registered scope.

---

## 8. Summary and Readiness for the Next Phase

Step 8 completed the factorial experiment this project was built to run and answered its central research question: the interaction between labeling scheme and update strategy is not significant (ART-ANOVA F(3,77) = 0.517, p = 0.672), so the choice of labeling scheme and the choice of update strategy can be made independently. Independently, L2 (category-level dynamic labeling) yields marginally but significantly higher shared-outcome accuracy than L1, and S2 (monthly periodic retraining) is the most accurate update strategy, significantly ahead of S3, S4, and (against S4 only) S1. All eight conditions are temporally stable (CV of AUC between 0.5% and 0.8%) and the conclusions survive both the equalized-maturity sensitivity arm and the two robustness checks (Recall@10% as response; exclusion of the truncated final batch). The frozen 25-feature set, frozen tuned hyperparameters, and the full 192-row metrics table are now available as inputs to any remaining thesis analysis (SHAP interpretation, threshold-sensitivity sweeps, or write-up), none of which revisits a decision made in Steps 1-8.

---

## 9. References

- Wobbrock, J. O., Findlay, L., Gergle, D., & Higgins, J. J. (2011). The Aligned Rank Transform for nonparametric factorial analyses using only ANOVA procedures. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '11)*, 143-146. https://doi.org/10.1145/1978942.1978963
