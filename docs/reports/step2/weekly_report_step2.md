# Weekly Progress Report: Data Cleaning (Step 2)

> **Project:** Customer churn prediction using XGBoost with a 2x4 factorial design (fixed vs. category-level dynamic labeling x four model updating strategies), applied to SME retail transaction data from the Kingdee platform.
> **This week's focus:** Executing the four-step data cleaning pipeline to resolve all critical and high-severity issues identified during profiling (July 12 report).
> **Date:** Week of July 14–20, 2026

---

## 1. What I Set Out to Do

Last week's profiling flagged four issues that needed resolution before I could proceed to temporal structuring and label construction. This week I built and ran the cleaning pipeline to address them. The full raw data profile — entity counts, composite key analysis, IPI distributions, industry classification — was covered in the July 12 report and is not repeated here.

The pipeline processed `srv_sales.csv` (15.3M rows) and `srv_saleitems.csv` (47.7M rows) in four sequential steps, each targeting a specific issue. The order matters: I removed outliers first so they would not contaminate the test-account detection logic downstream.

| Step | Target Issue | What I Did |
|---|---|---|
| 2.1 | Extreme outliers (max ~10 trillion) | Removed records with \|bill_amt\| or \|amt\| > 10M |
| 2.2 | System/test accounts (max 103K txns) | Multi-signal scoring; removed accounts with 2+ flags |
| 2.3 | Returns (negative bill_amt, 4.6%) | Removed all bill_amt < 0; preserved return stats separately |
| 2.4 | Orphan line items (0.52%) | Removed line items with no matching parent sale |

I also evaluated two issues and intentionally left them unchanged: the 55.7% zero pay_amt rate (confirmed as legitimate B2B trade credit) and the 2.14% missing company_type (deferred to XGBoost's native missing-value handling).

---

## 2. Cleaning Results

### 2.1 How Much Was Removed

| Step | srv_sales Removed | srv_saleitems Removed |
|---|---|---|
| 2.1 Outliers | 303 (0.002%) | 53 (0.0001%) |
| 2.2 Test accounts | 160,947 (1.05%) | — (cascaded in 2.4) |
| 2.3 Returns | 299,922 (1.98%) | 1,008,717 (2.12%) |
| 2.4 Orphans | — | 611,091 (1.31%) |
| **Total** | **461,172 (3.01%)** | **1,619,861 (3.40%)** |

After cleaning: 14,845,948 sales records and 46,044,976 line items remain — a combined removal rate of 3.30%. Only 26 customers were removed outright; the customer base dropped from 1,019,023 to 1,017,039, with most of the difference coming from customers who had only return or outlier transactions.

This is a conservative cleaning rate. Gattermann-Itschert and Thonemann (2022) reported 5–8% removal in a comparable B2B churn dataset. Converting from CSV (7.2 GB) to Parquet (1.3 GB) also achieved 5.5x compression.

![](media/image1.png) *Figure 1. Record counts after each cleaning step for both tables.*

### 2.2 Key Decisions and Rationale

**Outlier threshold (10M).** I chose this by examining the bill_amt upper tail: P99.99 = 3.4M, while the maximum was ~10 trillion. The 10M cutoff sits well above any plausible wholesale order but far below the data-entry errors. I removed records entirely rather than capping them, since capping would create artificial values that distort monetary features.

**Test account detection.** Rather than using a single transaction-count cutoff, I designed a multi-signal scoring system with three indicators: extreme transaction volume (>= P99.9), majority trivial amounts (>= 50% of transactions at bill_amt 0 or 1), and default creation date (1980-01-01). Accounts triggering 2+ signals were removed; those with only 1 signal were flagged as REVIEW and kept. All 26 removed accounts triggered both the volume and trivial-amount signals. The 7,109 REVIEW accounts (3.23M transactions) remain in the data — I discuss the risk this poses in Section 5.

**Return handling.** I removed all negative bill_amt transactions because returns are not purchase events — including them would inflate frequency counts and create artificial inter-purchase intervals. Before deletion, I extracted per-customer return statistics (counts, ratios) into a separate file (`return_statistics.csv`) so the LLM can still consider return behavior as a candidate feature. The 349,125 zero-amount transactions were kept: 99.96% have zero across all monetary fields, suggesting exchanges or system records with no distortive effect.

---

## 3. Before vs. After: Did Cleaning Distort the Data?

This is the critical check. Cleaning should remove noise, not reshape the underlying distributions. All "Raw" values come from the July 12 profiling; "Cleaned" values are recomputed from the post-cleaning Parquet files. Record-level counts are already in Section 2 and not repeated.

### 3.1 Transaction Frequency per Customer

| Statistic | Raw | Cleaned | Change |
|---|---|---|---|
| Median | 2 | 2 | — |
| Mean | 15.0 | 14.6 | -0.4 |
| P90 | 25 | 24 | -1 |
| Max | 103,246 | 95,970 | -7,276 |
| Sparse customers (1–3 txn) | 63.9% | 64.3% | +0.4pp |

The distribution is virtually unchanged. The mean dropped slightly because the removed test accounts were high-frequency, but the median, quartiles, and sparsity rate held steady. The 64% sparse-customer challenge identified on July 12 persists — cleaning did not create it and did not resolve it.

### 3.2 Financial Anomalies

| Issue | Raw | Cleaned | Outcome |
|---|---|---|---|
| Max bill_amt | ~10 trillion | 10,000,000 | Resolved |
| Negative bill_amt | 4.60% | 0.00% | Resolved |
| Zero pay_amt | 55.73% | 55.06% | Stable (trade credit) |
| Negative item_amt | 6.97% | 0.25% | 97% reduction |
| Orphan line items | 0.52% | 0.00% | Resolved |

The zero pay_amt rate barely moved (~55%), confirming the July 12 conclusion that this reflects B2B credit terms rather than an artifact of the records I removed.

### 3.3 Repurchase Cycle (IPI) — The Most Important Shift

| Metric | Raw | Cleaned | Change |
|---|---|---|---|
| Global IPI median | 5 days | 5 days | — |
| Per-customer median IPI (P50) | 25 days | 30 days | +5 days |
| Per-customer median IPI (P75) | 60 days | 82 days | +22 days |
| Store-level median IPI | 7 days | 24 days | +17 days |

The IPI shift is the most substantive change from cleaning, and it is a desirable one. The global median held at 5 days because it is dominated by high-frequency buyers unaffected by cleaning. But the per-customer and store-level medians shifted upward because test accounts (with near-daily artificial transactions) and return transactions (creating spurious short intervals) no longer compress the distribution.

The per-customer median moving from 25 to 30 days actually strengthens the case for the 30-day L1 fixed threshold — it now aligns almost exactly with the empirical repurchase center.

![](media/image2.png) *Figure 2. Per-customer median IPI distributions before and after cleaning.*

### 3.4 Payment Segments and L2 Coverage

| Metric | Raw | Cleaned | Change |
|---|---|---|---|
| Strictly Zero payment users | 42.3% | 42.3% | — |
| Mixed payment users | 12.5% | 12.3% | -0.2pp |
| L2-eligible customers (N=3) | 35.26% | 35.0% | -0.26pp |

Both payment segmentation and L2 individual-level coverage are essentially unchanged. The July 12 conclusion — that only ~35% of customers qualify for individual-level dynamic thresholds, necessitating the category-level L2 redesign — holds.

### 3.5 Industry-Level IPI After Cleaning

On July 12 I computed industry IPI from store-level medians (N = 171–203 stores per category). This week I recomputed from pooled individual intervals (N = 481K–1.32M per category), which is what the L2 threshold computation actually uses.

| Industry | Store Median (Jul 12) | Pooled Median (Jul 20) | Pooled P95 |
|---|---|---|---|
| 服装鞋帽 (Apparel) | 3 days | 2 days | 62 days |
| 包装印刷 (Packaging) | 5 days | 4 days | 57 days |
| 商超 (Supermarket) | 6 days | 5 days | 69 days |
| 食品饮料 (Food & Bev) | 6 days | 5 days | 57 days |
| 工业材料 (Industrial) | 6 days | 5 days | 87 days |
| 家居装饰 (Home Decor) | 7 days | 5 days | 87 days |
| 文体用品 (Sports/Office) | 7 days | 6 days | 83 days |
| 母婴用品 (Mother & Baby) | 7 days | 6 days | 86 days |
| 精细化工 (Fine Chemical) | 7 days | 6 days | 88 days |
| 安防电子 (Security/Elec) | 8 days | 6 days | 90 days (cap) |
| 数码电器 (Digital/Elec) | 8 days | 5 days | 90 days (cap) |
| 美容配饰 (Beauty) | 9.8 days | 7 days | 90 days (cap) |
| 医药器械 (Medical) | 10 days | 7 days | 90 days (cap) |

The Kruskal-Wallis test on pooled intervals confirms that inter-industry differences remain highly significant (H = 178,325, p < 0.001, eta-squared = 0.017). The effect size is smaller than the store-level estimate (0.047) from July 12, which is expected — pooled individual intervals have much higher within-group variance. On the other hand, 75 of 78 pairwise comparisons are now significant (vs. 22/78 at store level), meaning the larger sample reveals finer-grained industry structure.

**Bottom line on Section 3:** Cleaning preserved the dataset's fundamental structure. The one meaningful shift — upward IPI movement — is desirable and strengthens (rather than undermines) the design decisions made on July 12.

---

## 4. Why Each Cleaning Step Matters Downstream

Every removal has a specific reason tied to either label construction, feature quality, or model validity. I did not clean anything "just to be safe."

| What I Removed | Why It Matters for the Model |
|---|---|
| 303 extreme-amount records | Without removal, a single ~10T transaction would dominate any monetary feature mean or standard deviation, making avg_monetary and monetary_vs_store meaningless |
| 26 test accounts (160K txns) | These accounts would dominate store-level IPI pools and inflate store_txn_volume, biasing both L2 thresholds and store-context features |
| 300K return transactions | Returns are not purchase events — they would create artificial inter-purchase intervals and inflate frequency counts, contaminating the IPI distributions that L2 thresholds depend on |
| 611K orphan line items | Orphans have no parent sale, so they cannot be attributed to any customer or date — keeping them would introduce noise in product diversity and basket size features |

I also saved per-customer return statistics before deletion, so the LLM can still consider return behavior as a candidate feature without the return records polluting the core transaction data.

---

## 5. What Remains Unresolved

| Issue | Status | My Plan |
|---|---|---|
| 7,109 REVIEW-flagged accounts (21.7% of cleaned txn volume) | Kept — only 1 detection signal fired | Monitor L2 threshold stability across batches; if thresholds are unstable in specific industries, re-examine REVIEW accounts there |
| Pre-2010 company_create dates | Kept as-is | Defer to Step 4 — the LLM can propose tenure transformations |
| Residual negative item_amt (0.25%) | Kept | Likely partial-return line items within kept transactions; minimal downstream impact since line items feed only product diversity and basket size features |

The REVIEW accounts are the biggest concern. They represent a non-trivial share of the transaction volume, and if some are genuine test accounts, they could subtly compress IPI distributions and bias L2 thresholds downward. I plan to track L2 threshold stability across monthly batches and report model performance separately for stores with and without REVIEW accounts. A definitive answer would require Kingdee to provide a ground-truth list of test/demo accounts, which I have not yet requested.

---

## 6. Readiness Check for Step 3

| Requirement | Needed | Actual | Pass? |
|---|---|---|---|
| Transaction volume | >= 1M | 14,845,948 | Yes |
| Customer volume | >= 100K | 1,017,039 | Yes |
| Temporal depth | >= 12 months | 25 months | Yes |
| Industry coverage | 100% | 13 categories, 100% | Yes |
| Stores per industry | >= 100 | 175–213 | Yes |
| No extreme outliers | Max < 100M | Max = 10M | Yes |
| No return contamination | 0 negative bill_amt | 0 | Yes |
| Referential integrity | 0 orphans | 0 | Yes |
| Removal rate | < 10% | 3.30% | Yes |
| Customer loss | < 1% | 0.003% | Yes |

All checks pass. The cleaned dataset is ready for temporal structuring (Step 3).

---

## 7. References

- De Caigny, A., Coussement, K., De Bock, K. W., & Lessmann, S. (2023). Leveraging purchase sequence information for churn prediction with latent Dirichlet allocation. *Expert Systems with Applications*, 232, 120596.

- Gattermann-Itschert, T., & Thonemann, U. W. (2022). Proactive customer retention management in a non-contractual B2B setting based on churn prediction with random forests. *Industrial Marketing Management*, 107, 134–147.

- Jain, H., Yadav, G., & Manohar, R. (2024). XGBoost-based customer churn prediction with SHAP interpretability in the telecommunications industry. *Journal of Computational Science*, 75, 102199.

- Mena, G., De Caigny, A., Coussement, K., De Bock, K. W., & Lessmann, S. (2023). Churn prediction with sequential patterns and machine learning. *European Journal of Operational Research*, 305(2), 1048–1065.

---

## Next Steps

With the cleaned dataset now ready, the remaining pipeline proceeds as follows:

| Step | Target | What I Will Do |
|---|---|---|
| 3. Temporal Structuring | Partition data into monthly prediction batches with strict observation/label window separation | Define 90-day observation windows and label windows for each prediction point (1st of each month); assign eligible customers to batches; enforce data leakage safeguards |
| 4. Feature Engineering | Compute per-customer behavioral features within each observation window | Build 25 features across RFM, IPI, payment behavior, store-context, and seasonality dimensions; handle NaN for single-transaction customers |
| 5. Label Construction | Assign churn labels under both L1 (fixed) and L2 (category-level dynamic) schemes | Compute L1 using the 30-day threshold; compute L2 using industry-level P95 IPI as category thresholds (capped at 90 days); monitor threshold stability across batches — this is also where I will check whether the 7,109 REVIEW accounts are distorting L2 thresholds |
| 6. Dataset Assembly | Merge features and labels into a single modeling-ready dataset | Join feature matrix with both label schemes; validate no leakage, no duplicates, correct binary labels; compute class balance and scale_pos_weight per batch |
