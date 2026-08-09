# Preprocessing Decision Log

Generated: 2026-08-10 00:17:38 by `pipeline/step7_3_preprocessing_log.py`. Regenerate after any pipeline rerun — do not edit by hand.

## 1. Data scope

- Raw inputs: `srv_sales.csv` (2.1 GB), `srv_saleitems.csv` (5.1 GB), `industry_servereid.csv` (13 industry categories, 100% merchant coverage).
- Span: 2024-05-01 to 2026-05-31 (~25 months).
- Customer identity: composite key `(server_id, shop_id, company_id)` — ids are per-merchant auto-increments and are NOT globally unique.

## 2. Exclusion criteria and affected record counts (Step 2)

| Cleaning step | Records excluded | Log file | Log contents |
|---|---|---|---|
| 2.1 Extreme financial outliers | 356 | `exclusion_log_step2_1.csv` | complete |
| 2.2 System/test accounts | 26 accounts (160,947 transactions) | `exclusion_log_step2_2.csv` | one row per removed account |
| 2.3 Returns and corrections | 1,308,639 | `exclusion_log_step2_3.csv` | complete |
| 2.4 Orphan line items | 611,091 | `exclusion_log_step2_4.csv` | 10,000-row sample |
| 3. Never-eligible customers (no purchase in any observation window) | 80,210 customers | `step3_temporal_summary.txt` | summary count only |
| 5. Unmapped merchants (no industry mapping) | not persisted | n/a | see `step5_label_construction.py` |

Per-step rationale is recorded in the corresponding `profiling_output/step2_*_summary.txt` / analysis reports. Note: the 2.2 and 2.4 log files above do not contain one row per excluded record — 2.2 logs one row per removed *account* (each covering many transactions) and 2.4 is a fixed-size sample of a larger removal; true removal counts are parsed from the corresponding summary text files, not from `count_rows` on those two logs. The Step 3 row is mandated by the methodology: customers with no purchase in the observation window are excluded from scoring (excluded from batch eligibility, not from the raw sales data). The Step 5 row records IPIs dropped for merchants absent from `industry_servereid.csv`; `step5_label_construction.py` logs this count to stdout at run time (`compute_l2_thresholds`) but does not persist it to any `profiling_output` artifact, so no number is reported here — see that script if an exact count is needed.

## 3. Window-length choices and justification (Steps 3 & 5)

| Parameter | Value | Justification |
|---|---|---|
| Observation window | 90 days | Methodology §4.2; covers P75 of per-customer median IPI (60d) with margin |
| L1 label window | 30 days | Fixed inactivity threshold per methodology |
| L2 label window | P95 of pooled category IPIs, capped at 90 days | Thesis eq. 3.6; cap preserves 12 evaluation batches |
| Batch cadence | Monthly (1st of month) | Matches SME review cycle |
| S3 rolling window | 6 months | Covers P95 per-customer median IPI (~180d, Step 1 profiling); sensitivity-tested in Phase 2 |

## 4. Train/evaluation split (Step 7.1)

| prediction_point | l1_label_close | l2_label_close | role | matured_for_L1_at_first_eval | matured_for_L2_at_first_eval | eligible_customers | n_rows |
|---|---|---|---|---|---|---|---|
| 2024-08-01 | 2024-08-31 | 2024-09-10 | initial_train | True | True | 313427 | 313427 |
| 2024-09-01 | 2024-10-01 | 2024-10-20 | initial_train | True | True | 314611 | 314611 |
| 2024-10-01 | 2024-10-31 | 2024-11-26 | initial_train | True | True | 323227 | 323227 |
| 2024-11-01 | 2024-12-01 | 2025-01-03 | initial_train | True | True | 325398 | 325398 |
| 2024-12-01 | 2024-12-31 | 2025-02-07 | initial_train | True | True | 327637 | 327637 |
| 2025-01-01 | 2025-01-31 | 2025-03-15 | initial_train | True | True | 327039 | 327039 |
| 2025-02-01 | 2025-03-03 | 2025-04-19 | initial_train | True | False | 314175 | 314175 |
| 2025-03-01 | 2025-03-31 | 2025-05-21 | initial_train | True | False | 304346 | 304346 |
| 2025-04-01 | 2025-05-01 | 2025-06-25 | evaluation | False | False | 306840 | 306840 |
| 2025-05-01 | 2025-05-31 | 2025-07-29 | evaluation | False | False | 318082 | 318082 |
| 2025-06-01 | 2025-07-01 | 2025-08-30 | evaluation | False | False | 327914 | 327914 |
| 2025-07-01 | 2025-07-31 | 2025-09-29 | evaluation | False | False | 324100 | 324100 |
| 2025-08-01 | 2025-08-31 | 2025-10-30 | evaluation | False | False | 319980 | 319980 |
| 2025-09-01 | 2025-10-01 | 2025-11-30 | evaluation | False | False | 317431 | 317431 |
| 2025-10-01 | 2025-10-31 | 2025-12-30 | evaluation | False | False | 323572 | 323572 |
| 2025-11-01 | 2025-12-01 | 2026-01-30 | evaluation | False | False | 321738 | 321738 |
| 2025-12-01 | 2025-12-31 | 2026-03-01 | evaluation | False | False | 319253 | 319253 |
| 2026-01-01 | 2026-01-31 | 2026-04-01 | evaluation | False | False | 311871 | 311871 |
| 2026-02-01 | 2026-03-03 | 2026-05-02 | evaluation | False | False | 309078 | 309078 |
| 2026-03-01 | 2026-03-31 | 2026-05-30 | evaluation | False | False | 290948 | 290948 |

- First evaluation point: **2025-04-01** — 8 initial-training batches, 12 evaluation batches (one year).
- Label-maturity rule: batches usable for the initial L1 model: 8; for L2: 6. L2 labels close later (longer windows), so fewer batches are matured — enforced by `experiment_config.usable_batches` at every retraining event.

## 5. L2 fallback tiers and threshold stability (Step 5)

Tier share per batch (% of customers):

| prediction_point | category |
|---|---|
| 2024-08-01 | 100.00 |
| 2024-09-01 | 100.00 |
| 2024-10-01 | 100.00 |
| 2024-11-01 | 100.00 |
| 2024-12-01 | 100.00 |
| 2025-01-01 | 100.00 |
| 2025-02-01 | 100.00 |
| 2025-03-01 | 100.00 |
| 2025-04-01 | 100.00 |
| 2025-05-01 | 100.00 |
| 2025-06-01 | 100.00 |
| 2025-07-01 | 100.00 |
| 2025-08-01 | 100.00 |
| 2025-09-01 | 100.00 |
| 2025-10-01 | 100.00 |
| 2025-11-01 | 100.00 |
| 2025-12-01 | 100.00 |
| 2026-01-01 | 100.00 |
| 2026-02-01 | 100.00 |
| 2026-03-01 | 100.00 |

The global-fallback tier fired for **0.00%** of customer-batch rows in all 20 batches — every category cleared the minimum pooled-IPI requirement at every prediction point.

Category threshold (tau_c, days) stability across batches:

| category | min | max | mean | std | cv_pct |
|---|---|---|---|---|---|
| 包装印刷 | 27.0 | 57.0 | 47.2 | 9.0 | 19.0 |
| 医药器械 | 37.0 | 90.0 | 77.4 | 17.5 | 22.7 |
| 商超 | 30.0 | 69.0 | 54.6 | 11.5 | 21.1 |
| 安防电子 | 34.0 | 90.0 | 72.5 | 18.1 | 25.0 |
| 家居装饰 | 34.0 | 87.0 | 66.6 | 16.5 | 24.7 |
| 工业材料 | 33.0 | 87.0 | 67.2 | 16.6 | 24.6 |
| 数码电器 | 33.0 | 90.0 | 71.8 | 19.4 | 27.0 |
| 文体用品 | 34.0 | 83.0 | 65.6 | 14.4 | 22.0 |
| 服装鞋帽 | 24.0 | 62.0 | 48.6 | 11.3 | 23.3 |
| 母婴用品 | 34.0 | 86.0 | 67.2 | 15.2 | 22.6 |
| 精细化工 | 35.0 | 88.0 | 68.6 | 16.4 | 24.0 |
| 美容配饰 | 40.0 | 90.0 | 79.0 | 15.8 | 19.9 |
| 食品饮料 | 29.0 | 57.0 | 47.4 | 8.7 | 18.4 |

## 6. Class balance per batch per scheme (Step 6)

| prediction_point | scheme | n_total | n_churn | n_active | churn_rate | scale_pos_weight |
|---|---|---|---|---|---|---|
| 2024-08-01 | L1 | 313427 | 186679 | 126748 | 0.5956 | 0.6790 |
| 2024-08-01 | L2 | 313427 | 181743 | 131684 | 0.5799 | 0.7246 |
| 2024-09-01 | L1 | 314611 | 183078 | 131533 | 0.5819 | 0.7185 |
| 2024-09-01 | L2 | 314611 | 171187 | 143424 | 0.5441 | 0.8378 |
| 2024-10-01 | L1 | 323227 | 194156 | 129071 | 0.6007 | 0.6648 |
| 2024-10-01 | L2 | 323227 | 172765 | 150462 | 0.5345 | 0.8709 |
| 2024-11-01 | L1 | 325398 | 194241 | 131157 | 0.5969 | 0.6752 |
| 2024-11-01 | L2 | 325398 | 168456 | 156942 | 0.5177 | 0.9316 |
| 2024-12-01 | L1 | 327637 | 195267 | 132370 | 0.5960 | 0.6779 |
| 2024-12-01 | L2 | 327637 | 167996 | 159641 | 0.5128 | 0.9503 |
| 2025-01-01 | L1 | 327039 | 209774 | 117265 | 0.6414 | 0.5590 |
| 2025-01-01 | L2 | 327039 | 178671 | 148368 | 0.5463 | 0.8304 |
| 2025-02-01 | L1 | 314175 | 199775 | 114400 | 0.6359 | 0.5726 |
| 2025-02-01 | L2 | 314175 | 159689 | 154486 | 0.5083 | 0.9674 |
| 2025-03-01 | L1 | 304346 | 177860 | 126486 | 0.5844 | 0.7112 |
| 2025-03-01 | L2 | 304346 | 142815 | 161531 | 0.4693 | 1.1311 |
| 2025-04-01 | L1 | 306840 | 176733 | 130107 | 0.5760 | 0.7362 |
| 2025-04-01 | L2 | 306840 | 139875 | 166965 | 0.4559 | 1.1937 |
| 2025-05-01 | L1 | 318082 | 187942 | 130140 | 0.5909 | 0.6924 |
| 2025-05-01 | L2 | 318082 | 146296 | 171786 | 0.4599 | 1.1742 |
| 2025-06-01 | L1 | 327914 | 197228 | 130686 | 0.6015 | 0.6626 |
| 2025-06-01 | L2 | 327914 | 154425 | 173489 | 0.4709 | 1.1235 |
| 2025-07-01 | L1 | 324100 | 196766 | 127334 | 0.6071 | 0.6471 |
| 2025-07-01 | L2 | 324100 | 151032 | 173068 | 0.4660 | 1.1459 |
| 2025-08-01 | L1 | 319980 | 192136 | 127844 | 0.6005 | 0.6654 |
| 2025-08-01 | L2 | 319980 | 145523 | 174457 | 0.4548 | 1.1988 |
| 2025-09-01 | L1 | 317431 | 183302 | 134129 | 0.5775 | 0.7317 |
| 2025-09-01 | L2 | 317431 | 142122 | 175309 | 0.4477 | 1.2335 |
| 2025-10-01 | L1 | 323572 | 196528 | 127044 | 0.6074 | 0.6464 |
| 2025-10-01 | L2 | 323572 | 149160 | 174412 | 0.4610 | 1.1693 |
| 2025-11-01 | L1 | 321738 | 196565 | 125173 | 0.6109 | 0.6368 |
| 2025-11-01 | L2 | 321738 | 147119 | 174619 | 0.4573 | 1.1869 |
| 2025-12-01 | L1 | 319253 | 191166 | 128087 | 0.5988 | 0.6700 |
| 2025-12-01 | L2 | 319253 | 148345 | 170908 | 0.4647 | 1.1521 |
| 2026-01-01 | L1 | 311871 | 185422 | 126449 | 0.5945 | 0.6820 |
| 2026-01-01 | L2 | 311871 | 144953 | 166918 | 0.4648 | 1.1515 |
| 2026-02-01 | L1 | 309078 | 215505 | 93573 | 0.6973 | 0.4342 |
| 2026-02-01 | L2 | 309078 | 144213 | 164865 | 0.4666 | 1.1432 |
| 2026-03-01 | L1 | 290948 | 170315 | 120633 | 0.5854 | 0.7083 |
| 2026-03-01 | L2 | 290948 | 126929 | 164019 | 0.4363 | 1.2922 |

`scale_pos_weight` is recomputed at each retraining from the pooled training data of that event (thesis eq. 3.8); the per-batch values above are diagnostics.

## 7. Drift-monitoring configuration (Step 7.2)

- PSI: 10 equal-frequency bins + dedicated missing-value bin, epsilon = 1e-06, trigger threshold = 0.2.
- AUC-decline trigger threshold = 0.02.
- Calendar features excluded from monitoring (deterministic drift): month, quarter, is_cny_season, is_double11, is_spring_festival.
- **L1**: reference batch 2025-03-01, in-sample AUC 0.8544 (provisional hyperparameters), monitored features: ipi_std, frequency, recent_activity, ipi_mean, customer_tenure, recency, frequency_vs_store, freq_decline_ratio, debt_frequency, total_qty.
- **L2**: reference batch 2025-01-01, in-sample AUC 0.8580 (provisional hyperparameters), monitored features: ipi_std, frequency, recent_activity, customer_tenure, ipi_mean, recency, frequency_vs_store, freq_decline_ratio, store_median_ipi, unique_products.
- PSI bin edges are frozen from the Step-7.2 reference batches for the entire study; reference resets in S4 reuse the precomputed per-batch distributions under these edges.

## 8. Deviations from the original plan

- **L2 tier design**: Data_Preprocessing_Plan.md §5.2 described a 3-tier scheme (customer/store/category). The implemented scheme is the thesis's 2-tier category + global-fallback design (eq. 3.6–3.7), which pools IPIs at category level for statistical stability.
- **Zero bill_amt records retained** in Step 2.3 (only negative-amount returns removed); see `step2_3_returns_analysis.txt`.
- **Hyperparameters provisional**: initial models in Step 7.2 use documented defaults; Phase 2 tunes once via time-series CV on the initial training period, then rerun `step7_2_drift_baselines.py`.

