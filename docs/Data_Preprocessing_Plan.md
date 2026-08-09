# Data Preprocessing Plan

## Overview

This plan describes how to transform the raw `srv_sales.csv` and `srv_saleitems.csv` into analysis-ready datasets for the 2×4 factorial XGBoost churn prediction experiment. The plan follows the research methodology's five-phase framework, covering Phase 1 (Data Preparation) in detail.

---

## Step 1: Data Profiling

**Goal:** Understand data characteristics and validate assumptions before any transformation.

### 1.1 Assess Data Span and Scale
- Confirm transaction date range (currently 2024-05-01 to 2026-05-31, ~25 months).
- Count unique merchants (`server_id`), stores (`server_id + shop_id` composite key — `shop_id` alone is NOT unique), and customers (`server_id + shop_id + company_id` composite key — `company_id` is per-merchant auto-increment, NOT globally unique).
- Compute transactions per true customer distribution (min, median, mean, 90th percentile, max).

**Profiling results (corrected):**
- 2,533 merchants, 2,583 stores, 1,019,023 true customers
- Median txns/customer: 2, Mean: 15.0, Max: 103,246
- 44.2% of customers have only 1 transaction; 63.9% have ≤3

### 1.2 Identify Verticals
- Industry classification obtained (`industry_servereid.csv`): 13 categories, 175–213 stores each, 100% coverage of all merchants.
- Categories: 服装鞋帽, 医药器械, 包装印刷, 安防电子, 数码电器, 商超, 精细化工, 工业材料, 美容配饰, 食品饮料, 家居装饰, 文体用品, 母婴用品.
- `company_type` (零售/批发) is a customer-level attribute, not an industry classification.
- Note: industry labels may have some noise (a few name-industry mismatches observed); distribution is suspiciously uniform (CV=0.044), suggesting a stratified sample from Kingdee's platform.

### 1.3 Compute Median Repurchase Cycles
- For each true customer (`server_id + shop_id + company_id`) with ≥ 3 unique purchase dates, compute all inter-purchase intervals (days between consecutive `opt_on` dates in `srv_sales`).
- Report per-customer median interval, then aggregate: global median, per-store median, and distribution (5th, 25th, 50th, 75th, 95th percentiles).
- This informs: (a) observation window length, (b) label window length, (c) rolling-window length for S3, (d) the "sufficient history" threshold for dynamic labeling.

**Profiling results (corrected):**
- 427,974 customers (42.0%) have ≥3 unique purchase dates
- Global IPI: P50=5 days, P75=15 days, P95=80 days, mean=18.1 days
- Per-customer median IPI: P50=25 days, P75=60 days, P95=179.5 days, mean=47.6 days
- Per-store median IPI: range 1–230 days, median 7 days (across 2,579 stores)

### 1.4 Assess Sufficient-History Proportion
- Define "sufficient history" as having ≥ N completed inter-purchase intervals (e.g., N = 3–5, to be determined from profiling).
- Report the proportion of customers meeting this threshold — this determines the viability of dynamic labeling (L2) and the expected fallback tier distribution.

**Profiling results (corrected):**
- N=3: 359,292 / 1,019,023 (35.26%)
- N=4: 313,724 / 1,019,023 (30.79%)
- N=5: 280,213 / 1,019,023 (27.50%)
- 58% of customers have <3 unique purchase dates — most will fall back to store/industry-level thresholds

### 1.5 Data Quality Checks
- **Duplicate transactions:** Check for duplicate `(server_id, shop_id, id)` composite keys in `srv_sales`. Note: `id` alone is NOT unique — it is per-merchant auto-increment.
- **Orphan line items:** Check for `(server_id, shop_id, sale_id)` values in `srv_saleitems` that have no matching `(server_id, shop_id, id)` in `srv_sales`.
- **Date consistency:** Verify that `opt_on` in `srv_saleitems` matches the corresponding sale's `opt_on` in `srv_sales` (joined on `server_id + shop_id + sale_id = id`).
- **Negative/zero amounts:** Flag records where `bill_amt`, `pay_amt`, or `amt` ≤ 0 (possible returns or corrections).
- **Extreme outliers:** Identify the max item amount (~10 trillion in current data) and investigate — likely a data entry error.

**Output:** A profiling report with summary statistics, distributions, and quality issue counts.

---

## Step 2: Data Cleaning

**Goal:** Remove noise and anomalous records while preserving legitimate business variation.

### 2.1 Remove System/Test Accounts
- Filter out customers (identified by `server_id + shop_id + company_id` composite key) with extreme transaction counts. The current max is 103,246 transactions in 25 months (~137 txns/day) — investigate whether this is a legitimate high-frequency buyer or automated/system account.
- Cross-reference with `company_create` dates: accounts created before a reasonable cutoff (e.g., before 2010) with minimal recent activity may be legacy artifacts.
- Document all exclusion criteria and the number of records/customers removed.

### 2.2 Handle Returns and Corrections
- Identify transactions with negative `bill_amt` or negative `qty` in line items — these likely represent returns.
- **Decision:** Either remove return transactions entirely or net them against their original sale. Document the chosen approach and its rationale.

### 2.3 Remove Financial Outliers
- Investigate and remove records with implausible amounts (e.g., single item amounts exceeding a business-reasonable threshold).
- Apply a cap or remove based on domain knowledge rather than purely statistical rules, since legitimate wholesale orders can be large.

### 2.4 Handle Missing Values
- **`company_type`:** 2.1% of records have empty values. Options:
  - If vertical filtering is applied (Step 2.5), these records would be excluded anyway.
  - Otherwise, assign to an "unknown" category — XGBoost can handle this natively.
- **`acct_name`:** Masked as "??" — not used as a feature, so no action needed.
- **`acct_id` = 0:** Indicates no payment account linked. Treat as a categorical level (unpaid/cash).

### 2.5 Apply Vertical Filtering (if applicable)
- Industry classification is available (`industry_servereid.csv`): 13 categories, 175–213 stores each, 100% coverage of all 2,533 merchants (2,583 stores).
- If the research scope requires specific verticals (e.g., food & beverage, supermarket/convenience), filter by matching `server_id` to the industry file.
- Otherwise, proceed with the full dataset using all 13 categories for L2 category-level thresholds.

**Output:** A cleaned transaction dataset with an exclusion log documenting every filter and the number of records affected.

---

## Step 3: Temporal Structuring

**Goal:** Partition data into monthly batches with strict observation/label window separation.

### 3.1 Define Window Parameters
Based on profiling results (Step 1.3), set:
- **Observation window length:** 90 days (as specified in methodology Section 4.2).
- **Label window length:** Determined by the fixed threshold (L1) or the maximum plausible dynamic threshold. For L1, a 30-day inactivity window is specified. For L2, the label window must be at least as long as the 90th-percentile inter-purchase interval at the global level.
- **Monthly batch cadence:** Prediction points at the 1st of each month.

### 3.2 Define the Timeline
```
Data span: 2024-05-01 to 2026-05-31

Initial training period:  2024-05-01 to [first prediction point]
  - Must accommodate: observation window + label window
  - Example: If obs = 90 days, label = 90 days → first prediction point = 2024-11-01
  - Training obs window: 2024-08-01 to 2024-10-31
  - Training label window: 2024-11-01 to 2025-01-31

Evaluation batches: Monthly from [first evaluation point] onward
  - Each batch t has:
    - Prediction point: 1st of month t
    - Observation window: [t - 90 days, t)
    - Label window: [t, t + label_length)
```

### 3.3 Assign Customers to Batches
- A customer is identified by the composite key `(server_id, shop_id, company_id)`. Each merchant's customers are independent — there is no shared identity across merchants.
- For each prediction point, identify customers who have **at least one purchase** in the observation window — these are "eligible" customers.
- Customers with **no purchase** in the observation window are excluded (trivially inactive).
- Customers whose label window extends beyond the data boundary (2026-05-31) are excluded to avoid label noise.
- Note: 44.2% of customers have only 1 transaction across all 25 months; many will only appear in a single batch.

### 3.4 Prevent Data Leakage
- **Hard rule:** Features are computed ONLY from the observation window. Labels are computed ONLY from the label window. No information from the label window may leak into features.
- Implement this as a code-level constraint: feature construction and label construction use separate date-filtered subsets.

**Output:** A batch assignment table mapping `(server_id, shop_id, company_id, prediction_point)` → eligible/excluded, with the observation and label window boundaries for each batch.

---

## Step 4: Feature Engineering

**Goal:** Construct the feature set specified in the methodology (Section 4.2) for each `(server_id, shop_id, company_id, prediction_point)` tuple.

### 4.1 Customer-Level Features (from observation window)

| Feature | Computation |
|---------|-------------|
| `recency` | Days from last purchase to prediction point |
| `frequency` | Total number of purchases in observation window |
| `monetary` | Total `bill_amt` across purchases in observation window |
| `avg_monetary` | Mean `bill_amt` per purchase |
| `ipi_mean` | Mean inter-purchase interval (days between consecutive purchases) |
| `ipi_std` | Standard deviation of inter-purchase intervals |
| `recent_activity` | Purchase count in the most recent 30 days of the observation window |
| `freq_decline_ratio` | (Purchases in 2nd half of obs window) / (Purchases in 1st half of obs window). Handle division by zero: if 1st half = 0 and 2nd half > 0, set to a large value (e.g., 2.0); if both = 0, set to 1.0 |
| `unique_products` | Count of distinct `product_id` purchased (from `srv_saleitems`) |
| `avg_basket_size` | Mean number of line items per transaction |
| `total_qty` | Total quantity purchased |
| `pay_ratio` | `sum(pay_amt)` / `sum(bill_amt)` — payment behavior indicator |
| `debt_frequency` | Proportion of transactions with `owe_amt > 0` |
| `customer_tenure` | Days from `company_create` to prediction point |

### 4.2 Store-Level Features (computed per store `(server_id, shop_id)`, per batch)

| Feature | Computation |
|---------|-------------|
| `store_txn_volume` | Total transactions for the store in the observation window |
| `store_active_customers` | Count of distinct customers `(server_id, shop_id, company_id)` with purchases in the observation window |
| `store_median_ipi` | Median inter-purchase interval across all customers in the store |
| `recency_vs_store` | Customer's `recency` / store's median recency (ratio > 1 = more silent than typical) |
| `frequency_vs_store` | Customer's `frequency` / store's median frequency |
| `monetary_vs_store` | Customer's `monetary` / store's median monetary |

### 4.3 Time-Level Features

| Feature | Computation |
|---------|-------------|
| `month` | Calendar month of the prediction point (1–12) |
| `quarter` | Calendar quarter (1–4) |
| `is_cny_season` | 1 if prediction point falls in Jan–Feb (Chinese New Year period) |
| `is_double11` | 1 if observation window overlaps November (Double 11 / Singles' Day) |
| `is_spring_festival` | 1 if observation window overlaps late Jan–mid Feb |

### 4.4 Aggregation Strategy
- Join `srv_sales` and `srv_saleitems` on the composite key `(server_id, shop_id, id = sale_id)`. Note: `sale_id` alone is NOT unique — it is per-merchant auto-increment, so the join MUST include `server_id` and `shop_id`.
- All aggregations are at the `(server_id, shop_id, company_id, prediction_point)` level — this is one row per true customer per batch.
- Each merchant's customers are independent (no shared identity across merchants), so each `(server_id, shop_id, company_id)` is treated as a separate entity.
- Include `server_id` and `shop_id` as context features (or encode via store-level features in 4.2) for the single global model.

**Output:** A feature matrix with one row per `(server_id, shop_id, company_id, prediction_point)` and all features as columns.

---

## Step 5: Label Construction

**Goal:** Create churn labels under both labeling schemes.

### 5.1 Fixed-Threshold Label (L1)
- **Definition:** A customer `(server_id, shop_id, company_id)` is churned (label = 1) if they make **no purchase** in the label window [prediction_point, prediction_point + 30 days).
- Straightforward binary check against `srv_sales.opt_on`, filtered by the customer's composite key.

### 5.2 Dynamic-Threshold Label (L2)
- **Tier 1 — Customer-level:** For customers `(server_id, shop_id, company_id)` with ≥ N completed inter-purchase intervals (N=3 from profiling — 35.26% of customers qualify), compute the per-customer median IPI using ALL data up to the prediction point (not just the observation window). The customer is churned if no purchase occurs within P75 of their category's per-customer median IPI distribution after the prediction point.
- **Tier 2 — Store-level fallback:** For customers with insufficient history, use the store's `(server_id, shop_id)` pooled per-customer median IPI at P75 across all qualifying customers in that store.
- **Tier 3 — Category-level fallback:** For stores with insufficient qualifying customers, use the industry category's pooled per-customer median IPI at P75 (industry from `industry_servereid.csv`, 13 categories available).
- **Record the tier used** for each customer-batch pair. Report tier proportions per batch.
- Expected tier distribution: ~35% Tier 1, ~58% Tier 2, ~7% Tier 3 (based on sufficient-history profiling).

### 5.3 Label Window Boundary Check
- For each customer-batch pair, verify that the label window (prediction_point + threshold) does not exceed the data boundary (2026-05-31).
- If using per-customer median P75 (max ~60 days), cap at 90 days to ensure sufficient evaluation batches.
- Exclude customers whose label window is truncated.

**Output:** A label table with columns: `server_id`, `shop_id`, `company_id`, `prediction_point`, `label_L1`, `label_L2`, `L2_tier`, `L2_threshold_days`.

---

## Step 6: Dataset Assembly and Validation

**Goal:** Merge features and labels into final experiment-ready datasets.

### 6.1 Merge Features and Labels
- Inner join the feature matrix (Step 4) with the label table (Step 5) on `(server_id, shop_id, company_id, prediction_point)`.
- Only retain rows where both features and labels are valid (non-truncated).

### 6.2 Compute Class Balance per Batch
- For each (prediction_point, labeling_scheme), report:
  - Total eligible customers
  - Churn count and churn rate
  - This informs `scale_pos_weight` for XGBoost at each retraining event.

### 6.3 Final Validation Checks
- **No leakage check:** Confirm that no feature uses data from the label window. Spot-check by verifying that `recency` ≥ 0 and features for batch t don't change when label window data is removed.
- **Completeness:** Ensure every eligible customer has a full feature vector and label.
- **Feature distributions:** Generate summary statistics per batch to detect data drift or anomalies early.
- **Consistency:** Verify that L1 and L2 labels differ for a meaningful proportion of customers — if they are nearly identical, the factorial design loses power.

### 6.4 Output Format
Save the final dataset in a format suitable for efficient batch-wise access:
- **Option A:** One CSV/Parquet file per batch, with features + both labels.
- **Option B:** A single file with a `batch_id` column for filtering.
- Parquet is recommended for large data to preserve types and reduce file size.

**Output:** Final dataset files + a validation report confirming data integrity.

---

## Step 7: Prepare for Experimental Execution

### 7.1 Define Train/Evaluation Split
- Reserve the initial training period (batches before the first evaluation point) for model training and hyperparameter tuning.
- All subsequent batches are evaluation batches used in the prequential protocol.

### 7.2 Precompute Drift Monitoring Inputs
- For drift-triggered strategy (S4), precompute PSI reference distributions for key features at each batch boundary.
- Store feature importance rankings from the initial model to determine which features to monitor.

### 7.3 Document All Decisions
Create a preprocessing log recording:
- Every exclusion criterion and the number of records/customers affected
- Window length choices and their profiling justification
- Fallback tier proportions for L2
- Class balance per batch per labeling scheme
- Any deviations from the original methodology

---

## Summary Pipeline

```
Raw Data (srv_sales.csv + srv_saleitems.csv + industry_servereid.csv)
    │
    ▼
[Step 1] Data Profiling
    │   → profiling report, repurchase cycle stats
    │   → Key: customer = (server_id, shop_id, company_id)
    ▼
[Step 2] Data Cleaning
    │   → cleaned transactions, exclusion log
    ▼
[Step 3] Temporal Structuring
    │   → batch assignments, window boundaries
    ▼
[Step 4] Feature Engineering ──────────────────┐
    │   → feature matrix (per customer × batch)  │
    ▼                                            │
[Step 5] Label Construction                      │
    │   → label table (L1 + L2 per customer)     │
    │   → L2 uses industry categories (13)        │
    ▼                                            │
[Step 6] Dataset Assembly & Validation ◄─────────┘
    │   → final experiment-ready dataset
    ▼
[Step 7] Prepare for Execution
    │   → train/eval split, PSI baselines, documentation
    ▼
Ready for 2×4 Factorial Experiment
```

---

## Tools and Implementation Notes

- **Recommended stack:** Python with `pandas` (or `polars`/`dask` for large-file performance), `pyarrow` for Parquet I/O.
- **Memory management:** The raw files total 7.2 GB. Use chunked reading (`pd.read_csv(chunksize=...)`) or Dask/Polars for out-of-core processing. Do NOT load both files fully into memory simultaneously.
- **Reproducibility:** Set random seeds, version-control all preprocessing scripts, and log all parameter choices.
