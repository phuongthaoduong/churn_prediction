# Weekly Progress Report: Experiment Preparation (Step 7)

> **Project:** Customer churn prediction using XGBoost with a 2x4 factorial design (fixed vs. category-level dynamic labeling x four model updating strategies), applied to SME retail transaction data from the Kingdee platform.
> **This week's focus:** Preparing the experimental infrastructure — temporal split, drift-monitoring baselines, and a consolidated preprocessing record — so that Phase 2 can begin without revisiting any preprocessing decisions.
> **Date:** Week of July 3–9, 2026

---

## 1. What I Set Out to Do

Steps 3–6 produced a validated dataset of 6,340,667 customer-batch observations with 25 features and dual churn labels (L1 fixed, L2 category-level dynamic) across 20 monthly batches. The dataset itself is ready, but the 2x4 factorial experiment cannot start until three preparatory questions are resolved. First, the prequential protocol requires a temporal split that assigns batches to training or evaluation roles and enforces a label-maturity rule — preventing the model from training on labels whose observation window has not yet closed. Second, the S4 strategy (drift-triggered retraining) needs precomputed PSI references so it can detect distributional shifts at each evaluation point, which in turn requires knowing which features matter most. Third, the thesis methodology chapter requires a single reproducibility record tracing every preprocessing decision back to its data-driven justification. This week I addressed all three.

---

## 2. Train/Evaluation Split (Step 7.1)

### What was done

I partitioned the 20 monthly batches into an initial-training period and a prequential evaluation period, then computed the date at which each batch's labels become fully observable under each labeling scheme.

### Why

The factorial experiment uses a prequential (test-then-train) protocol: at each monthly evaluation point, the model predicts on the current batch before incorporating it into training. This requires a clean temporal boundary between initial training and evaluation. More subtly, L1 and L2 have different label windows (30 days vs. up to 90 days), so a batch that is "closed" for L1 may still have unobserved L2 labels — training on it would constitute label leakage.

### What was achieved

The split places the first evaluation point at 2025-04-01, yielding 8 initial-training batches (2024-08-01 to 2025-03-01) and 12 evaluation batches (2025-04-01 to 2026-03-01), covering one full year of monthly evaluation.

The label-maturity rule reveals an important asymmetry between the two schemes. For L1, all 8 initial-training batches are usable at the first evaluation point because every L1 label window (fixed at 30 days) closes by 2025-03-31, before the evaluation begins. For L2, only 6 of the 8 are usable: the two latest training batches (2025-02-01 and 2025-03-01) have L2 label windows that close at 2025-04-19 and 2025-05-21, respectively — both after the first evaluation point. L2 therefore trains its initial model on fewer batches than L1, which is by design: under the prequential protocol, labels are revealed only after the full label window closes, and L2's category-level windows can extend up to 90 days. Phase-2 code must select training batches via the `usable_batches()` function or the per-scheme maturity flags in `train_eval_split.csv`, never via the `role` column alone.

This step produced two output files. `train_eval_split.csv` records the prediction point, role, L1 and L2 label-close dates, maturity flags, and row counts for all 20 batches. `experiment_config.json` snapshots all experiment constants — split boundaries, PSI and AUC thresholds, the feature list, and provisional XGBoost hyperparameters — in a single reproducibility artifact.

---

## 3. Drift-Monitoring Baselines (Step 7.2)

### What was done

I trained a provisional XGBoost model for each labeling scheme on its label-matured training batches, ranked features by gain importance to identify which ones to monitor for drift, froze equal-frequency PSI bin edges on each scheme's reference batch, and precomputed bin proportions for all 20 batches under those frozen edges.

### Why

The S4 strategy needs to detect when the feature distribution at a new evaluation batch has shifted relative to a reference. Population Stability Index (PSI) quantifies this shift, but it requires three inputs that must be fixed before the experiment begins: a gain-based feature ranking to decide which features matter enough to monitor, fixed bin edges so PSI values remain comparable across the entire study, and precomputed per-batch distributions so that any reference-vs-current PSI can be computed as a lookup rather than re-scanning the raw data.

The models trained here are provisional — they use documented default hyperparameters. Phase 2 will tune hyperparameters via time-series CV, then rerun this step to regenerate the monitored-feature list and baselines under the tuned model.

### What was achieved

**Provisional model performance.** The L1 model trained on 2,549,860 rows (all 8 initial-training batches) with a scale_pos_weight of 0.6549 and achieved an in-sample AUC of 0.8544. The L2 model trained on 1,931,339 rows (6 matured batches) with a scale_pos_weight of 0.8556 and achieved an in-sample AUC of 0.8580. Both figures exceed 0.85, consistent with the literature for RFM-based B2B churn prediction (Gattermann-Itschert & Thonemann, 2022). These are in-sample results on provisional hyperparameters; out-of-sample performance under tuned parameters will be reported in Phase 2.

**Feature importance.** The top 7 features are identical across both schemes, with only minor rank swaps: ipi_std ranks first in both (L1 gain: 2,490; L2 gain: 2,363), followed by frequency (L1: 1,815; L2: 1,236), recent_activity, ipi_mean, customer_tenure, recency, and frequency_vs_store. This consistency suggests that the behavioral signals driving churn prediction are stable regardless of labeling scheme. The divergence in ranks 8–10 — L1 selects is_cny_season and debt_frequency, while L2 selects quarter and is_double11 — reflects the different seasonal sensitivities of the two schemes. Notably, ipi_std dominates despite having 63.5% NaN values; XGBoost's native missing-value handling effectively learns a "has repeat purchase" indicator at each split. Calendar features rank 8th–10th but are excluded from drift monitoring because they change deterministically with the prediction point.

**Drift-monitoring configuration.** For each scheme, the top 10 behavioral features (excluding the 5 calendar features) are monitored using 10 equal-frequency bins plus a dedicated NaN bin. The PSI trigger threshold is set at 0.2 — the standard industry convention for a severe distributional shift — and the AUC-decline trigger threshold at 0.02; both thresholds are sensitivity-tested in Phase 2. This dual-signal design, combining a feature-distribution signal with a delayed performance-decline signal, is consistent with the performance-aware drift-detection taxonomy surveyed by Bayram, Ahmed, and Kassler (2022). Bin edges are frozen from each scheme's reference batch — 2025-03-01 for L1 (its last matured training batch) and 2025-01-01 for L2 — and never re-derived, even when S4 resets its reference batch after a retraining event.

**Baseline PSI results.** Three patterns emerged from computing PSI across all 12 evaluation batches. First, L1 fires a drift trigger at the very first evaluation batch (2025-04-01, PSI = 0.24 on freq_decline_ratio). This is expected adjacent-batch volatility: PSI drops below the 0.2 threshold by the next month and stays well below it for 10 consecutive months. Second, L2 shows no drift triggers across the first 11 evaluation batches — the maximum PSI never exceeds 0.07, reflecting more stable feature distributions under L2's reference window. Third, both schemes spike sharply at the final batch (2026-03-01): L1 reaches 0.53 and L2 reaches 0.38, both on recency. This batch has the smallest size across all 20 batches (290,948 rows) because some customers' last purchases fall outside the final observation window. This is an end-of-data truncation artifact, not genuine distributional drift, and Phase-2 evaluation should flag it accordingly rather than treating the spike as a retraining signal.

This step produced four output files: the provisional models (`initial_model_L1.json`, `initial_model_L2.json`), per-scheme feature rankings with raw total gain (`feature_importance.csv`), frozen bin edges with reference proportions and monitored feature lists (`psi_reference.json`), and precomputed bin proportions for all 20 batches (`psi_batch_distributions.csv`).

---

## 4. Consolidated Preprocessing Decision Log (Step 7.3)

### What was done

I built a script (`step7_3_preprocessing_log.py`) that reads all profiling artifacts from Steps 2–7.2 and assembles a single auto-generated markdown document (`docs/preprocessing_log.md`).

### Why

Preprocessing decisions are currently scattered across individual step summaries: exclusion counts live in Step 2 text files, window-length justifications in Step 3, L2 threshold stability in Step 5, class balance in Step 6, and drift configuration in Step 7.2. The thesis methodology chapter needs a single document that traces each decision back to its justification, and that document must stay in sync with the pipeline — if any step is rerun, the record must update automatically. A generated document, not a hand-written one, guarantees this.

### What was achieved

The decision log covers eight sections: data scope (raw inputs, date span, composite customer key), exclusion criteria (per-step record counts with log file references from Steps 2.1–2.4, 3, and 5), window-length choices (90-day observation, 30-day L1, P95-capped L2, monthly cadence — each linked to profiling evidence), the full train/evaluation split with maturity flags, L2 tier proportions (100% category tier across all batches — the global fallback never fired) and threshold stability (CV 18–27% across categories), per-batch class balance and scale_pos_weight for both schemes, drift-monitoring configuration, and three documented deviations from the original Data_Preprocessing_Plan.md.

The three deviations are worth noting. First, L2 uses a 2-tier design (category + global fallback) rather than the planned 3-tier scheme (customer/store/category), because pooling IPIs at category level provides greater statistical stability. Second, zero bill_amt records were retained in Step 2.3 — only negative-amount returns were removed. Third, the initial models use provisional default hyperparameters; Phase 2 tunes once via time-series CV, then reruns Step 7.2 to regenerate baselines under the tuned model.

---

## 5. What Remains Unresolved

Four issues carry forward into Phase 2. The most consequential is that the current feature importance rankings and drift-monitoring baselines rest on provisional hyperparameters — once Phase 2 tunes these via time-series CV, the rankings may shift, changing which features are monitored. The plan is to tune first, then rerun Step 7.2 before beginning the S1–S4 strategy loop.

The PSI spike at the final batch (2026-03-01, L1: 0.53, L2: 0.38) is an end-of-data truncation artifact and should be flagged in Phase-2 evaluation rather than treated as genuine drift that warrants retraining. Conversely, the L1 drift trigger at the first evaluation batch (PSI = 0.24) is expected adjacent-batch volatility and validates that the monitoring mechanism is functioning correctly — S4 should proceed with retraining if triggered.

Finally, ipi_std's position as the top gain feature in both schemes despite 63.5% NaN values raises the question of whether its importance reflects genuine predictive signal or a structural artifact of XGBoost's NaN-routing. SHAP analysis in Phase 2 will clarify this.

---

## 6. Comparative Analysis with a Recent Published Study: El Attar & El-Hajj (2026)

This section situates the present project within the recent literature by examining a peer-reviewed study that likewise employs XGBoost for customer churn prediction: El Attar and El-Hajj (2026), "Explainable AI-driven customer churn prediction: a multi-model ensemble approach with SHAP-based feature analysis," *Frontiers in Artificial Intelligence*, 9, 1748799. The study was selected because it represents current methodological practice, its pipeline is documented in full from data acquisition through interpretation, and XGBoost emerges as its best-performing classifier, which permits a direct methodological comparison.

### Research pipeline of the reference study

**Data.** The reference study employs the IBM Telco Customer Churn dataset, a public benchmark comprising 7,043 telecommunications customers described by 21 attributes across three groups: demographics, account information (contract type, payment method, charges), and service subscriptions. The churn label is provided directly in the dataset, as the operator records contract cancellations; the observed churn rate is 26.5%.

**Data preparation.** Eleven records (0.16%) contained blank `TotalCharges` values stored as whitespace strings; these were resolved by median imputation following type conversion. All numeric features were standardized with StandardScaler, and contract type was encoded numerically by its duration in months (month-to-month = 1, one-year = 12, two-year = 24).

**Feature engineering.** Two derived variables were introduced: `AvgMonthlyCharge`, defined as TotalCharges / (tenure + 1), which exhibited a stronger correlation with churn (r ≈ 0.23) than the raw monthly charge (r ≈ 0.19), and `HasMultipleServices`, a count of active services across nine categories.

**Class imbalance.** SMOTE oversampling (k = 5) was applied exclusively to the training partition, rebalancing it to a 50/50 class distribution; the test partition remained unmodified.

**Partitioning and model development.** Model development employed a single stratified 80/20 train/test split (5,634 and 1,409 customers, respectively; seed 42) with stratified five-fold cross-validation for model selection. Seven classifiers were evaluated — XGBoost, LightGBM, Gradient Boosting, Random Forest, AdaBoost, logistic regression, and a multilayer perceptron — together with a soft-voting ensemble of the three strongest. Hyperparameters were optimized with Optuna under the same cross-validation scheme.

**Evaluation.** XGBoost achieved the strongest overall performance, with accuracy, precision, recall, and F1 all equal to 0.84 and the highest AUC-ROC of 0.932; the soft-voting ensemble matched the F1 score but not the AUC (0.918). Two refinements followed: probability calibration by isotonic regression (XGBoost Brier score 0.106) and cost-sensitive decision-threshold optimization, in which false negatives were weighted five times more heavily than false positives, yielding an operating threshold of 0.528.

**Interpretation.** SHAP analysis identified month-to-month contract status as the dominant churn driver (mean |SHAP| = 0.284; 60% observed churn against 10% for two-year contracts), followed by tenure (customers with fewer than 12 months of tenure exhibited 3.7 times the churn probability of those exceeding 24 months) and electronic-check payment. The SHAP profiles were subsequently used to segment the customer base into high-risk (18% of customers; churn probability 0.78), medium-risk (42%; 0.35), and low-risk (40%; 0.09) groups.

### Methodological comparison

| Aspect | El Attar & El-Hajj (2026) | The present study |
|---|---|---|
| Domain | Contractual telecommunications (B2C) | Non-contractual SME retail (B2B) |
| Data scale | 7,043 customers, 21 attributes, single snapshot | 15.3M raw transactions → 6.34M customer-batch observations, 25 features, 25 months |
| Churn label | Recorded in the data (observed cancellations) | Constructed from transaction behavior; the labeling scheme (L1 vs. L2) is itself an experimental factor |
| Temporal design | Single stratified 80/20 split; no temporal dimension | 20 monthly batches; prequential test-then-train protocol with a label-maturity rule |
| Class imbalance | SMOTE to 50/50 (churn constituted the 26.5% minority) | scale_pos_weight reweighting only (churn constitutes the ~60% majority; no synthetic instances) |
| Missing values | 0.16% missing, median-imputed | Up to 63.5% NaN in inter-purchase-interval features, retained via XGBoost's native missing-value routing; a dedicated NaN bin is used in PSI monitoring |
| Feature scaling | StandardScaler applied to all features | None required (tree ensembles are scale-invariant; no linear or neural models in the design) |
| Model comparison | Seven learners plus an ensemble; XGBoost selected empirically | XGBoost held fixed as a control variable; experimental variation lies in labeling schemes and update strategies |
| Model maintenance | Single static model; no updating protocol | Central research object: four update strategies (S1–S4) with PSI and AUC-decline drift triggers |
| Interpretability | SHAP analysis central to the contribution | Gain-based importance at present; SHAP analysis scheduled for Phase 2 |
| Headline result | XGBoost AUC of 0.932 on a 20% held-out set | Provisional in-sample AUC of 0.854/0.858; out-of-sample prequential results to be reported in Phase 2 |

### Discussion

Three observations follow from the comparison. First, although both studies culminate in an XGBoost classifier, they address structurally different problems. In the contractual telecommunications setting, the churn event is directly observed and recorded by the operator; the labeling problem that occupies a central position in the present work — the choice between fixed and category-level dynamic inactivity thresholds, together with label-window maturity — does not arise. Second, the evaluation protocol of the reference study is atemporal: a single stratified split evaluates one model at one point in time and therefore yields no evidence concerning model aging, distributional drift, or retraining economics. The reported AUC of 0.932 characterizes a single snapshot; whether that performance persists six or twelve months after deployment is precisely the question the present S1–S4 factorial design is constructed to answer. It should further be noted that a stratified rather than temporal split is defensible for cross-sectional data but would constitute leakage in the rolling transactional setting studied here. Third, the leakage discipline of the reference study reflects, at smaller scale, the same principle enforced throughout this project: restricting SMOTE to the training partition parallels the label-maturity rule, in that both prevent information from the evaluation side from contaminating model training.

Two elements of the reference methodology merit consideration for Phase 2. The probability-calibration and cost-sensitive thresholding procedure (false-negative cost set at five times the false-positive cost; threshold 0.528) constitutes a business-realistic refinement that transfers naturally to the SME retention setting, in which the cost of an undetected churner exceeds that of an unnecessary retention contact; such a procedure would complement the Recall@10% metric already specified. The SHAP-based risk segmentation likewise suggests a merchant-facing presentation format for Phase-2 model output. Conversely, the reported AUC values are not directly comparable across the two studies, owing to differences in domain, task difficulty, evaluation protocol, and in-sample versus held-out measurement; the reference study's AUC of 0.932 should therefore not be treated as a performance benchmark for the present work.

---

## 7. Statistical Analysis Plan for the Phase-2 Factorial Evaluation

This section specifies, in advance of the experiment, the statistical methodology by which the 2 × 4 factorial results will be analyzed. Pre-specification guards against post-hoc analytic flexibility and renders the evaluation publication-ready. Factor A denotes the labeling scheme (L1 fixed; L2 category-level dynamic) and Factor B the update strategy (S1 static; S2 periodic; S3 rolling; S4 drift-triggered), yielding eight conditions. All eight conditions are evaluated on the same twelve monthly evaluation batches (2025-04-01 through 2026-03-01); the batch therefore constitutes a blocking (repeated-measures) unit, and every between-condition comparison is paired within batch. The pipeline is deterministic (fixed seeds), so no replicate runs exist; stochastic variation enters through batch composition and temporal dynamics.

### 7.1 Dependent variable formalization

**Primary performance metric.** The primary dependent variable is the batch-level area under the ROC curve computed against the shared 60-day evaluation label: Y_ijt = AUC_shared(i, j, t), where i ∈ {L1, L2} indexes the labeling scheme, j ∈ {S1, …, S4} the update strategy, and t = 1, …, 12 the evaluation batch. AUC is selected as primary because it is threshold-free and, measured against the common downstream outcome, is comparable across labeling schemes — models trained on different labels cannot be compared on their own training targets without conflating target difficulty with model quality. Secondary performance metrics, reported but treated as supporting evidence, are Recall@10% (reflecting SME intervention capacity), F1 at the 0.5 threshold, and own-label AUC as a diagnostic.

**Stability metric.** Temporal stability is quantified per condition by the coefficient of variation of AUC across the twelve batches, CV_ij = σ(Y_ij·)/μ(Y_ij·), and by the mean Jaccard similarity of consecutive top-10% risk lists.

**Operational efficiency metric.** Efficiency is measured by the number of retraining events R_ij, the cumulative training time, and the total rows processed across retraining events, all taken from the retraining log. Because the pipeline is deterministic, R_ij is a single realized value per condition rather than a sample from a distribution; efficiency is therefore analyzed descriptively — as a cost–performance frontier plotting mean AUC against retraining count — and is not subjected to inferential testing.

**Temporal aggregation.** The analysis retains the per-batch structure (repeated measures) rather than collapsing each condition to a single averaged value. Averaging would discard the within-batch pairing that gives the design its statistical power at n = 12 blocks, would conflate level and stability of performance, and would reduce the analysis to eight data points. Averages and CVs are reported descriptively; all inference operates on the 96 batch-level observations with batch as the blocking factor.

### 7.2 Hypotheses and model specification

Let μ + α_i + β_j + (αβ)_ij denote the fixed-effects decomposition of expected performance. Three hypothesis families are tested on the primary dependent variable:

- **Main effect of Factor A (labeling).** H0: α_L1 = α_L2 = 0 — the distribution of batch-level performance does not differ between labeling schemes. H1: the schemes differ.
- **Main effect of Factor B (updating).** H0: β_S1 = β_S2 = β_S3 = β_S4 = 0 — update strategies do not differ in batch-level performance. H1: at least one strategy differs.
- **A × B interaction.** H0: (αβ)_ij = 0 for all (i, j) — the effect of the update strategy is invariant to the labeling scheme. H1: at least one interaction contrast is non-zero. The interaction is the study's central research question (RQ3): rejection of this H0 would indicate that labeling and updating decisions must be designed jointly.

**Reference model.** The parametric reference specification is a two-way fully within-block (repeated-measures) ANOVA with batch as a random blocking factor:

Y_ijt = μ + α_i + β_j + (αβ)_ij + π_t + ε_ijt,  π_t ~ N(0, σ²_π), ε_ijt ~ N(0, σ²),

equivalently expressed as the mixed model AUC ~ scheme × strategy + (1 | batch). Its assumptions — normality of residuals (Shapiro–Wilk), sphericity (Mauchly's test, with Greenhouse–Geisser correction where violated), and independence of blocks — are checked but are expected to be untenable: twelve temporally ordered monthly batches are autocorrelated, and normality tests are underpowered at n = 12. The parametric model therefore serves as the reference specification, while primary inference is conducted through the assumption-robust protocol of Section 7.3. This is consistent with the methodology chapter, which specifies non-parametric paired comparisons (Wilcoxon signed-rank) on the grounds that the number of batches is limited and normality cannot be assumed, and with current guidance on statistical testing for machine-learning model comparison (Rainio, Teuho, & Klén, 2024).

### 7.3 Interaction testing protocol and post-hoc procedures

The confirmatory analysis proceeds in the following pre-specified order, with the family-wise significance level set at α = 0.05.

**Step 1 — Construct the response table.** Assemble the 96 batch-level values of the primary dependent variable (8 conditions × 12 batches) from `metrics_per_batch.csv`.

**Step 2 — Assumption assessment.** Fit the parametric reference model; examine residual normality (Shapiro–Wilk), sphericity (Mauchly), and residual-versus-fitted diagnostics. Record all results regardless of outcome.

**Step 3 — Omnibus tests.** If the parametric assumptions are tenable, test the interaction and main effects with the repeated-measures ANOVA, applying the Greenhouse–Geisser correction to degrees of freedom where sphericity is violated, and report F statistics, corrected degrees of freedom, p-values, and partial η² for each effect. In the expected case that assumptions are violated, the Aligned Rank Transform (ART) ANOVA, as implemented in the ARTool package (Kay, Elkin, Higgins, & Wobbrock, 2025), is employed instead: the data are aligned and ranked separately for each of the three effects (A, B, A × B), and the repeated-measures ANOVA is fitted to the aligned ranks, yielding valid non-parametric tests of both main effects and — critically — the interaction, which classical rank tests (Friedman) cannot test directly. The interaction test is designated the single confirmatory test of the study; main-effect tests are interpreted in light of it.

**Step 4a — Interaction significant: simple main effects.** If the A × B interaction is significant, main effects are not interpreted in isolation; the analysis decomposes into simple main effects:

1. *Effect of update strategy within each labeling scheme.* A Friedman test across S1–S4 within L1 (12 blocks) and, separately, within L2. Where the Friedman test is significant, the six pairwise strategy contrasts within that scheme are tested by exact Wilcoxon signed-rank tests (exact distributions are available at n = 12), with Holm's sequentially rejective correction applied within each six-test family.
2. *Effect of labeling scheme within each update strategy.* Four paired Wilcoxon signed-rank tests (L1 versus L2 under S1, S2, S3, and S4), Holm-corrected as a four-test family.

**Step 4b — Interaction not significant: main-effect contrasts.** Main effects are interpreted from the omnibus model; a significant Factor B main effect is followed by the six pairwise strategy contrasts pooled across schemes (Wilcoxon signed-rank on 24 paired scheme-batch values, Holm-corrected); a significant Factor A main effect requires no decomposition (two levels).

**Choice of correction.** Tukey's HSD is not employed: it presupposes independent groups sharing a pooled error term, a condition violated by the paired, repeated-measures structure of this design. The Wilcoxon-plus-Holm procedure is the appropriate paired analogue; Holm's method is preferred to the plain Bonferroni correction because it controls the family-wise error rate at the same level while being uniformly more powerful.

**Effect sizes and interval estimates.** Every pairwise comparison is reported with the matched-pairs rank-biserial correlation r_rb = (W⁺ − W⁻)/(W⁺ + W⁻) and the median paired difference with a 95% confidence interval. Because the twelve batches are temporally autocorrelated, confidence intervals are obtained by a moving-block bootstrap (block length 3) rather than an i.i.d. bootstrap, and all p-values are interpreted as conditional on the realized evaluation year rather than as strictly exchangeable-sample inference.

**Primary and sensitivity analysis arms.** The confirmatory protocol above is applied to results generated under the scheme-specific label-maturity rule described in Section 2, which constitutes the primary analysis. Because that rule closes L1 label windows earlier than L2 windows, L1-trained models operate on systematically fresher training data at every retraining event; a performance difference between schemes could therefore reflect either the labels themselves or the differential data recency they induce. To separate these explanations, the entire factorial experiment is additionally executed under an equalized maturity condition in which both schemes treat labels as observable exactly 90 days after the prediction point, so that L1 and L2 train on data of identical age. Both arms share the frozen feature set, the tuned hyperparameters, and the frozen PSI baselines; only the maturity frame differs. Each training event additionally records its effective data lag, defined as the number of days between the most recent usable batch and the training time, which renders the asymmetry quantifiable rather than merely descriptive.

The two arms are interpreted according to a pre-specified rule. Where the equalized arm reproduces the direction and significance of the primary arm's interaction and main effects, the conclusions are attributed to the labeling schemes themselves. Where the arms diverge, the difference in effective data lag is identified as a contributing mechanism, and the delayed observability of category-level dynamic labels is reported as an intrinsic operational cost of that labeling scheme rather than as an analytic artifact. The primary arm retains confirmatory status in either case; the equalized arm is diagnostic and is reported alongside it.

**Further robustness analyses.** Three additional pre-specified checks accompany the confirmatory analysis: (a) repetition of the full protocol with Recall@10% as the response, reported as supporting evidence without confirmatory status; (b) exclusion of the final batch (2026-03-01), whose end-of-data truncation artifact was documented in Section 5, to verify that conclusions do not hinge on it; and (c) month-by-month trajectory plots of all eight conditions, since a significant interaction at the aggregate level should be visible as non-parallel temporal profiles. Secondary metrics beyond the primary dependent variable are treated as exploratory, and no family-wise correction is applied across dependent variables; the single confirmatory claim of the study rests on the interaction test for AUC_shared under the primary maturity arm.

---

## 8. Summary and Readiness for Phase 2

Step 7 closes the preprocessing pipeline. Starting from 15.3 million raw sales records in Step 1, the pipeline has now produced a validated, leakage-free dataset with dual churn labels, a temporal split with label-maturity enforcement, drift-monitoring baselines with frozen bin edges, and a consolidated preprocessing decision log. All artifacts required by the 2x4 factorial experiment — the split definition, experiment constants, provisional models, feature importance rankings, PSI baselines, and the reproducibility record — are in place.

Phase 2 proceeds with three tasks: tuning hyperparameters via time-series CV on the initial training period, rerunning Step 7.2 under the tuned parameters to finalize the drift baselines, and executing the S1–S4 strategy loop across 12 monthly evaluation batches.

---

## 9. References

- Bayram, F., Ahmed, B. S., & Kassler, A. (2022). From concept drift to model degradation: An overview on performance-aware drift detectors. *Knowledge-Based Systems*, 245, 108632. https://doi.org/10.1016/j.knosys.2022.108632

- El Attar, A., & El-Hajj, M. (2026). Explainable AI-driven customer churn prediction: a multi-model ensemble approach with SHAP-based feature analysis. *Frontiers in Artificial Intelligence*, 9, 1748799. https://doi.org/10.3389/frai.2026.1748799

- Gattermann-Itschert, T., & Thonemann, U. W. (2022). Proactive customer retention management in a non-contractual B2B setting based on churn prediction with random forests. *Industrial Marketing Management*, 107, 134–147.

- Kay, M., Elkin, L. A., Higgins, J. J., & Wobbrock, J. O. (2025). *ARTool: Aligned Rank Transform for Nonparametric Factorial ANOVAs* (R package version 0.11.2). https://github.com/mjskay/ARTool

- Rainio, O., Teuho, J., & Klén, R. (2024). Evaluation metrics and statistical tests for machine learning. *Scientific Reports*, 14, 6086. https://doi.org/10.1038/s41598-024-56706-x
