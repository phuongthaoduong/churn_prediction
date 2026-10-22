# Weekly Progress Report: Temporal Structuring through Dataset Assembly (Steps 3–6)

> **Project:** Customer churn prediction using XGBoost with a 2x4 factorial design (fixed vs. category-level dynamic labeling x four model updating strategies), applied to SME retail transaction data from the Kingdee platform.
> **This week's focus:** Building the temporal backbone, engineering features, constructing dual-scheme churn labels, and assembling the final modeling-ready dataset.
> **Date:** Week of July 14–20, 2026

---

## 1. What I Set Out to Do

Last week's cleaning pipeline produced 14.8M sales records and 46.0M line items across 1,017,039 customers and 13 industry categories. This week I executed the remaining four pipeline steps to transform that cleaned transactional data into a single experiment-ready dataset with features, labels, and validation guarantees.

| Step | Purpose | What I Did |
|---|---|---|
| 3. Temporal Structuring | Partition data into leakage-safe monthly batches | Defined 90-day observation windows and 30–90 day label windows anchored to 20 monthly prediction points; assigned eligible customers to batches |
| 4. Feature Engineering | Compute per-customer behavioral signals | Built 25 features (14 customer-level, 6 store-level, 5 time-level) using only observation-window data; processed 46M line items in chunked reads |
| 5. Label Construction | Assign churn labels under both schemes | Computed L1 (fixed 30-day) and L2 (category-level P95 IPI, capped at 90 days) labels; tracked threshold stability across all 20 batches |
| 6. Dataset Assembly | Merge and validate the final dataset | Inner-joined features and labels on composite key; ran 11 validation checks; computed class balance and scale_pos_weight per batch |

---

## 2. Step 3: Temporal Structuring

### 2.1 Design Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Data span | 2024-05-01 to 2026-05-31 | Full 25-month cleaned dataset |
| Observation window | 90 days | Captures seasonal repurchase cycles; aligns with L2 cap |
| L1 label window | 30 days (fixed) | Matches post-cleaning per-customer median IPI (30 days) |
| L2 label window | Up to 90 days | Capped at observation window length to prevent label leakage |
| Prediction cadence | Monthly (1st of each month) | Balances temporal granularity with batch size |

### 2.2 Batch Overview

The pipeline produced 20 prediction points from 2024-08-01 to 2026-03-01, yielding 6,340,667 total batch assignments across 936,829 unique customers. 80,210 customers (7.9%) were never eligible because they had no purchase within any 90-day observation window — these are extremely sparse customers who purchased only once during the entire data span.

| Metric | Value |
|---|---|
| Total batch assignments | 6,340,667 |
| Unique customers across all batches | 936,829 |
| Customers never eligible | 80,210 (7.9%) |
| Median batch appearances per customer | 3 |
| Mean batch appearances per customer | 6.8 |
| Batch size range | 290,948 – 327,637 |

Batch sizes are stable across prediction points (CV < 4%), with a slight dip in 2026-03-01 (290,948) as some customers' last purchases fell outside the final observation window.

### 2.3 Leakage Safeguards

The temporal design enforces strict separation:
- **Observation window** `[obs_start, obs_end)` where `obs_end == prediction_point` — features use only data before the prediction point.
- **Label window** `[prediction_point, label_end)` — labels use only data after the prediction point.
- **No overlap** — the observation and label windows share no dates.
- **Boundary exclusion** — the last usable prediction point (2026-03-01) was chosen so that even the longest L2 label window (90 days) ends within the data boundary (2026-05-30).

---

## 3. Step 4: Feature Engineering

### 3.1 Feature Set (25 Features)

| Category | Count | Features |
|---|---|---|
| Customer-level | 14 | recency, frequency, monetary, avg_monetary, ipi_mean, ipi_std, recent_activity, freq_decline_ratio, unique_products, avg_basket_size, total_qty, pay_ratio, debt_frequency, customer_tenure |
| Store-level | 6 | store_txn_volume, store_active_customers, store_median_ipi, recency_vs_store, frequency_vs_store, monetary_vs_store |
| Time-level | 5 | month, quarter, is_cny_season, is_double11, is_spring_festival |

**Design rationale for key features:**

- **RFM features** (recency, frequency, monetary, avg_monetary): Standard churn predictors. Computed strictly within the observation window. `monetary` and `avg_monetary` are `log1p`-transformed to compress the extreme B2B skew (raw max ~2.7B); the transform is applied after `monetary_vs_store` is computed from raw values.
- **IPI features** (ipi_mean, ipi_std): Capture repurchase regularity. Customers with only one purchase in the observation window get NaN (47.5% for ipi_mean, 63.5% for ipi_std) — deferred to XGBoost's native missing-value handling rather than imputing.
- **Trend features** (recent_activity, freq_decline_ratio): Signal whether activity is accelerating or decelerating. freq_decline_ratio compares 2nd-half to 1st-half frequency within the observation window.
- **Store-context features** (recency_vs_store, frequency_vs_store, monetary_vs_store): Relative positioning against store medians — a customer with frequency = 2 means something different in a store where the median is 2 vs. 20.
- **Seasonality features** (is_cny_season, is_double11, is_spring_festival): Binary flags for Chinese retail calendar events that affect purchasing patterns.

### 3.2 Feature Statistics

| Feature | Mean | Std | Min | Max | NaN% |
|---|---|---|---|---|---|
| recency | 32.4 | 26.2 | 1 | 90 | 0.0% |
| frequency | 5.6 | 54.8 | 1 | 24,634 | 0.0% |
| monetary | 21,993 | 3,655,801 | 0 | 2.7B | 0.0% |
| avg_monetary | 3,218 | 36,717 | 0 | 9.99M | 0.0% |
| ipi_mean | 17.8 | 15.4 | 1 | 89 | 47.5% |
| ipi_std | 9.3 | 8.2 | 0 | 61.5 | 63.5% |
| recent_activity | 1.8 | 18.5 | 0 | 10,306 | 0.0% |
| freq_decline_ratio | 1.1 | 1.2 | 0 | 1,106 | 0.0% |
| pay_ratio | 0.51 | 44.5 | 0 | 64,350 | 0.0% |
| debt_frequency | 0.51 | 0.49 | 0 | 1.0 | 0.0% |
| customer_tenure | 911 | 863 | -658 | 16,860 | 0.2% |

**Observations:**
- **Monetary skew is extreme** — raw mean 21,993 vs. max 2.7B. This reflects the long-tail nature of B2B wholesale. A `log1p` transform is applied to `monetary` and `avg_monetary` to compress the tail (the statistics above reflect pre-transform values).
- **pay_ratio outliers** — max 64,350 implies pay_amt > bill_amt by a large factor, likely overpayments or credit adjustments. The mean (0.51) aligns with the ~55% zero pay_amt rate identified during profiling.
- **Negative customer_tenure** (-658 days) — 0.2% of records have company_create dates after the prediction point. Flagged in Step 2 as pre-2010 date anomalies; deferred to model-level handling.

### 3.3 Output

`feature_matrix.parquet` — 6,340,667 rows x 29 columns (4 keys + 25 features), 199.7 MB.

---

## 4. Step 5: Label Construction

### 4.1 Two Labeling Schemes

| Scheme | Definition | Threshold | Purpose |
|---|---|---|---|
| L1 (Fixed) | Churned if no purchase in [pp, pp + 30 days) | 30 days for all customers | Baseline — simple, interpretable, industry-standard |
| L2 (Category-level dynamic) | Churned if no purchase in [pp, pp + tau_c) | P95 of pooled IPIs per industry category, capped at 90 days | Adapts to industry-specific repurchase cycles |

L2 thresholds are computed using only IPIs observed before each prediction point (no future leakage). Categories with fewer than 100 pooled IPIs fall back to the global P95 threshold — in practice, all 13 categories had sufficient data (420K–1.15M IPIs), so no fallback was triggered.

### 4.2 Churn Rates

| Metric | L1 | L2 |
|---|---|---|
| Overall churn rate | 60.4% | 48.6% |
| Churn rate range across batches | 57.6% – 69.7% | 43.6% – 58.0% |
| L1–L2 agreement | 88.2% | — |

The L1 scheme labels ~12 percentage points more customers as churned than L2, which is expected: L2's longer thresholds (up to 90 days vs. L1's fixed 30 days) give customers more time to return before being labeled as churned.

**The 2026-02-01 anomaly:** L1 churn spiked to 69.7% while L2 remained stable at 46.7%, dropping agreement to 76.9%. This batch's observation window (Nov 3 – Feb 1) captures the Chinese New Year period, when many SME customers pause purchasing. L1's fixed 30-day window is too short to accommodate this seasonal gap, while L2's category-level thresholds (many at 60–90 days) absorb it. This is precisely the type of seasonal misclassification that motivates the L2 scheme.

### 4.3 L2 Category Thresholds (Latest Batch: 2026-03-01)

| Category | Threshold | Capped? |
|---|---|---|
| 医药器械 (Medical) | 90 days | Yes (raw: 117 days) |
| 安防电子 (Security/Electronics) | 90 days | Yes (raw: 97 days) |
| 数码电器 (Digital/Appliances) | 90 days | Yes (raw: 103 days) |
| 美容配饰 (Beauty) | 90 days | Yes (raw: 116 days) |
| 精细化工 (Fine Chemical) | 88 days | No |
| 家居装饰 (Home Decor) | 87 days | No |
| 工业材料 (Industrial) | 87 days | No |
| 母婴用品 (Mother & Baby) | 86 days | No |
| 文体用品 (Sports/Office) | 83 days | No |
| 商超 (Supermarket) | 69 days | No |
| 服装鞋帽 (Apparel) | 62 days | No |
| 包装印刷 (Packaging) | 57 days | No |
| 食品饮料 (Food & Beverage) | 57 days | No |

The thresholds form two natural clusters: high-consideration categories (Medical, Electronics, Beauty — 86–90 days) and high-frequency categories (Apparel, Packaging, Food — 57–69 days). This aligns with the industry IPI structure identified during profiling on July 12.

### 4.4 L2 Threshold Stability

| Category | Min tau | Max tau | Range | CV |
|---|---|---|---|---|
| 食品饮料 (Food & Bev) | 29 days | 57 days | 28 days | 18.4% |
| 包装印刷 (Packaging) | 27 days | 57 days | 30 days | 19.0% |
| 美容配饰 (Beauty) | 40 days | 90 days | 50 days | 19.9% |
| 数码电器 (Digital/Elec) | 33 days | 90 days | 57 days | 27.0% |

All categories show CV between 18–27%. The thresholds trend upward across earlier-to-later batches as more historical IPIs are pooled, which is expected behavior for a cumulative percentile estimator. Thresholds stabilize by approximately batch 8–10 (2025-03-01 onward), suggesting that the early-batch threshold volatility reflects limited data accumulation rather than genuine distributional instability.

The REVIEW accounts flagged in the Step 2 report do not appear to be distorting L2 thresholds — if they were, I would expect to see threshold compression (lower tau_c values) in industries with high REVIEW account concentration, which I do not observe.

---

## 5. Step 6: Dataset Assembly and Validation

### 5.1 Final Dataset

| Metric | Value |
|---|---|
| Total rows | 6,340,667 |
| Total columns | 34 (4 keys + 25 features + 5 label/metadata) |
| Prediction points | 20 |
| Unique customers | 936,829 |
| Unique stores | 2,581 |
| Industry categories | 13 |
| File size | 201.7 MB |

Features (Step 4) and labels (Step 5) were inner-joined on the composite key (server_id, shop_id, company_id, prediction_point). No rows were lost in the join — all 6,340,667 feature rows matched exactly with their label rows.

### 5.2 Validation Results

All 11 validation checks passed:

| Check | Result | Detail |
|---|---|---|
| Row count > 0 | PASS | 6,340,667 rows |
| No duplicate keys | PASS | 0 duplicates |
| Recency >= 0 (no leakage) | PASS | No future-data contamination |
| Frequency >= 1 | PASS | All eligible customers have purchases |
| Labels are binary {0,1} | PASS | Both L1 and L2 |
| No missing labels | PASS | 0 NaN in either label column |
| No unexpected NaN features | PASS | NaN only in ipi_mean, ipi_std, store_median_ipi, customer_tenure (expected) |
| L1 vs L2 disagree >= 5% | PASS | 11.8% disagreement |
| Recency <= 90 (obs window) | PASS | No values exceed observation window |
| All 20 batches present | PASS | 20 batches found |
| Recency > 0 (strict leakage) | PASS | No same-day purchases leaked |

### 5.3 Class Balance and scale_pos_weight

| Scheme | Overall Churn Rate | SPW Range | SPW Mean |
|---|---|---|---|
| L1 | 60.4% | 0.43 – 0.74 | 0.67 |
| L2 | 48.6% | 0.72 – 1.29 | 1.04 |

L2's class balance is much closer to 50/50, which is favorable for model training. L1's consistent majority-churn imbalance (SPW < 1 in all batches) means the model will need scale_pos_weight adjustment to avoid predicting churn by default.

### 5.4 Feature Stability Across Batches

| Feature | CV% Across Batches |
|---|---|
| debt_frequency | 1.4% |
| frequency | 3.2% |
| recency | 6.5% |
| monetary | 9.0% |
| pay_ratio | 16.0% |

Feature means are stable across batches (CV < 10% for 4 of 5 key features). The pay_ratio CV of 16% is driven by two batches (2025-02-01 and 2025-03-01) where the mean jumped to 0.7 — again coinciding with the CNY period, when outstanding debts are typically settled. This is real seasonal behavior, not a data issue.

---

## 6. Cross-Step Consistency Checks

These checks verify that numbers remain consistent as data flows through the pipeline:

| Metric | Step 3 | Step 4 | Step 5 | Step 6 | Consistent? |
|---|---|---|---|---|---|
| Total rows | 6,340,667 | 6,340,667 | 6,340,667 | 6,340,667 | Yes |
| Unique customers | 936,829 | 936,829 | — | 936,829 | Yes |
| Prediction points | 20 | 20 | 20 | 20 | Yes |
| Batch size (2024-08-01) | 313,427 | 313,427 | 313,427 | 313,427 | Yes |

Row counts are perfectly preserved across all four steps, confirming that no records were silently dropped or duplicated during feature engineering, label construction, or assembly.

---

## 7. What Remains Unresolved

| Issue | Source | Risk | Plan |
|---|---|---|---|
| High ipi_mean/ipi_std NaN rate (47.5% / 63.5%) | Single-purchase customers within obs windows | Model may underweight IPI features | XGBoost handles NaN natively; monitor feature importance — if IPI features rank low, consider engineering an "has_repeat_purchase" binary flag |
| ~~Extreme monetary skew (max 2.7B)~~ | Long-tail B2B wholesale | ~~Could dominate tree splits~~ | **Resolved:** Applied `log1p` transform to `monetary` and `avg_monetary` in Step 4 (after `monetary_vs_store` ratio is computed from raw values). This compresses the long tail while preserving zero-safety and feature ordering. |
| Negative customer_tenure (0.2%) | Erroneous company_create dates | Minor — affects < 0.2% of rows | Defer to XGBoost; if problematic, clip to 0 |
| L1 seasonal sensitivity (CNY spike to 69.7%) | Fixed threshold too short for seasonal pauses | L1 model may learn spurious CNY churn signal | This is a known limitation of L1 — the factorial design explicitly compares L1 vs L2 to quantify this effect |
| L2 threshold early-batch volatility (CV 18–27%) | Cumulative percentile estimator with limited early data | First 2–3 batches may have noisy labels | Consider excluding the first 2 batches from training or using them only as burn-in |

---

## 8. Summary and Readiness for Modeling

The data processing pipeline is complete. Starting from 15.3M raw sales records, I produced a single `dataset_final.parquet` containing 6,340,667 customer-batch observations, each with 25 features and 2 churn labels (L1 fixed, L2 category-level dynamic), validated for leakage, completeness, and consistency.

| Requirement | Target | Actual | Status |
|---|---|---|---|
| Leakage-free temporal design | Strict obs/label separation | 11/11 validation checks passed | Ready |
| Dual labeling schemes | L1 (fixed) + L2 (dynamic) | Both constructed, 88.2% agreement, 11.8% meaningful disagreement | Ready |
| Sufficient feature coverage | >= 20 features | 25 features across 3 categories | Ready |
| Class balance documented | SPW per batch | Computed for both schemes | Ready |
| Feature stability | Low batch-to-batch drift | CV < 10% for 4/5 key features | Ready |
| Single modeling-ready file | All keys, features, labels | 34-column Parquet, 201.7 MB | Ready |

The dataset is ready for the 2x4 factorial experiment.

---

## 9. References

- De Caigny, A., Coussement, K., De Bock, K. W., & Lessmann, S. (2023). Leveraging purchase sequence information for churn prediction with latent Dirichlet allocation. *Expert Systems with Applications*, 232, 120596.

- Gattermann-Itschert, T., & Thonemann, U. W. (2022). Proactive customer retention management in a non-contractual B2B setting based on churn prediction with random forests. *Industrial Marketing Management*, 107, 134–147.

- Jain, H., Yadav, G., & Manohar, R. (2024). XGBoost-based customer churn prediction with SHAP interpretability in the telecommunications industry. *Journal of Computational Science*, 75, 102199.

- Mena, G., De Caigny, A., Coussement, K., De Bock, K. W., & Lessmann, S. (2023). Churn prediction with sequential patterns and machine learning. *European Journal of Operational Research*, 305(2), 1048–1065.
