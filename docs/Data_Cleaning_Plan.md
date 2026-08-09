# Data Cleaning Plan (Step 2)

## Context

This plan follows Step 1 (Data Profiling) and prepares the raw transaction data for Step 3 (Temporal Structuring) and beyond. All cleaning decisions are made with two downstream requirements in mind:

1. **LLM-guided feature construction (Phase 1 preliminary):** An LLM will scan the cleaned multi-table schema to propose candidate features. The cleaning step must preserve all columns and legitimate variation that the LLM might find predictive — we should remove noise, not signal. Over-aggressive cleaning (e.g., dropping columns, collapsing categories, imputing values prematurely) would reduce the LLM's ability to discover useful features.

2. **Category-level dynamic labeling (L2):** L2 uses the pooled inter-purchase interval (IPI) distribution within each **industry category** to set churn thresholds (e.g., P95), with a global fallback for categories with insufficient data. This is a two-tier design (category → global), not customer-level or store-level. Cleaning must therefore ensure that (a) industry labels are reliable, (b) IPI distributions are not contaminated by anomalous records, and (c) sufficient data remains per category after cleaning.

### Updated L2 Definition (from chap03.tex)

- **Category-level threshold:** For industry category $c$, pool all inter-purchase intervals $\{d_{j,c}\}$ observed for customers in that category up to the prediction point. Set $\tau_c = P_q(\{d_{j,c}\})$ where $q$ is a high percentile (e.g., 95).
- **Global fallback:** For categories with insufficient pooled purchase data, use $\tau_{\text{global}} = P_q(\{d_{j,\cdot}\})$ computed across all categories.
- **No customer-level or store-level tiers.** The rationale: purchase rhythm heterogeneity lies primarily between industry categories, and category-level pooling provides statistically stable estimates that individual-level thresholds cannot achieve at micro-retail data scales.

---

## Cleaning Steps

### 2.1 Remove Extreme Financial Outliers

**Problem:** Profiling found a maximum `bill_amt` and item `amt` of ~10 trillion (9,999,989,000,001). This is certainly a data entry error — no legitimate SME retail transaction reaches this scale.

**Action:**
- Set a business-reasonable upper bound for single-transaction `bill_amt`. Approach: examine the distribution of `bill_amt` above P99 to find a natural break point. A preliminary cap of 10,000,000 (10M) is proposed — wholesale orders can be large but not billions.
- Apply the same logic to `amt` in `srv_saleitems` (per-line-item amount).
- **Remove** (not cap) records exceeding the threshold, since capping would create artificial values that distort monetary features and IPI distributions.
- Log all removed records with their original values.

**Downstream impact:**
- Prevents extreme values from corrupting monetary features (avg_monetary, pay_ratio, etc.)
- Prevents IPI contamination: if these outlier transactions are the only purchase for some customers, removing them changes those customers' purchase history, which affects category-level IPI pools for L2.
- LLM feature construction benefits from clean monetary distributions.

**Expected impact:** < 0.01% of records.

### 2.2 Investigate and Remove System/Test Accounts

**Problem:** One customer has 103,246 transactions in 25 months (~137/day). This is almost certainly a system or POS test account, not a real customer.

**Action:**
1. Identify the customer(s) with extreme transaction counts. Examine the top-N customers by transaction count (e.g., top 50) and look for patterns:
   - Transactions spread evenly across all days (automated)?
   - Very small or zero amounts (test transactions)?
   - `company_create` dates that look like defaults (e.g., 1980-01-01)?
   - Same `product_id` repeated (test item)?
2. Set a transaction count threshold above which accounts are flagged for review. Approach: look for a natural gap in the transaction count distribution above P99.9.
3. **Remove all transactions** for confirmed system accounts (not just the excess transactions — the entire customer is invalid).
4. Log removed customer composite keys `(server_id, shop_id, company_id)` and their transaction counts.

**Downstream impact:**
- System accounts would dominate store-level and category-level IPI distributions, distorting L2 thresholds.
- Would bias store-level context features (e.g., `store_txn_volume`, `store_active_customers`).
- LLM-proposed features based on relative metrics (customer vs. store norm) would be skewed.

**Expected impact:** < 0.1% of records, but potentially large effect on IPI distributions.

### 2.3 Handle Returns and Corrections (Negative Amounts)

**Problem:** 704,733 records (4.6%) in `srv_sales` have `bill_amt <= 0`. 3,318,849 line items (7.0%) in `srv_saleitems` have `amt <= 0`. These likely represent returns, refunds, or accounting corrections.

**Action:**
1. **Classify negative-amount transactions:**
   - `bill_amt < 0`: Almost certainly a return/refund. These are reversals of prior purchases.
   - `bill_amt = 0`: Could be free samples, exchanges, or system records. Investigate the distribution of `pay_amt` and `owe_amt` for these records.
   - Negative `qty` in `srv_saleitems`: Return of specific items.

2. **Decision — Remove return transactions entirely:**
   - Returns represent the *reversal* of a purchase, not a purchase event. Including them would:
     - Inflate `frequency` (a return is not a new buying occasion)
     - Distort `monetary` features (negative amounts reduce totals)
     - Create artificial IPI values (a return on day 5 after a purchase on day 1 is not a 4-day repurchase cycle)
   - For L2: returns must not count as "purchases" when computing inter-purchase intervals per category.
   - **Remove** all records where `bill_amt < 0` from `srv_sales`, and their corresponding line items from `srv_saleitems`.

3. **Zero-amount transactions (`bill_amt = 0`):**
   - Investigate further. If these are exchanges or corrections with no monetary value, remove them.
   - If they represent legitimate zero-cost transactions (e.g., warranty claims, free samples), keep them but flag them — the LLM may find these patterns useful for feature construction (e.g., "proportion of zero-amount transactions" could be a churn signal).
   - **Decision deferred to investigation.** Run a sample analysis to determine the dominant pattern.

4. Log all removed records with reason codes.

**Downstream impact:**
- Cleaner IPI distributions for L2 category-level thresholds.
- More accurate RFM and monetary features.
- LLM can still discover return-related features if we preserve a `has_returns` flag at the customer level before removing the return records.

**Expected impact:** ~4.6% of `srv_sales` records (negative `bill_amt`); zero-amount TBD.

**Implementation note — Preserve return signal before removal:**
Before deleting return records, compute and store per-customer return statistics:
- `return_count`: number of return transactions per `(server_id, shop_id, company_id)`
- `return_amt_total`: total absolute value of negative `bill_amt`
- `return_ratio`: `return_count / total_transaction_count`

Save this as a separate lookup table. The LLM can then consider return behavior as a candidate feature without the return records contaminating the core transaction data.

### 2.4 Handle Orphan Line Items

**Problem:** 245,732 line items (0.52%) in `srv_saleitems` have no matching record in `srv_sales` when joined on `(server_id, shop_id, sale_id = id)`.

**Action:**
- **Remove** orphan line items from `srv_saleitems`. They cannot be attributed to a customer or date, so they are unusable for feature construction.
- Log the count and a sample for documentation.

**Expected impact:** 0.52% of `srv_saleitems`.

### 2.5 Handle Missing Values

**Problem:** Several fields have missing or placeholder values.

| Field | Issue | Count/Rate | Action |
|-------|-------|------------|--------|
| `company_type` | Empty string | 327,741 (2.1%) | Keep as-is. XGBoost handles missing values natively. The LLM may propose `company_type` as a feature — let it decide. Do NOT impute. |
| `acct_name` | Masked as "??" | Unknown | No action. Not useful as a feature (privacy-masked). |
| `acct_id` | Value = 0 | TBD | Treat as a categorical level meaning "no linked payment account" or "cash payment." Preserve for LLM to evaluate. |
| `company_create` | Dates back to 1980 | TBD | See 2.7. |

**Principle:** Do not impute or drop missing values at this stage. The LLM-guided feature construction step will evaluate which fields are useful. Premature imputation or removal reduces the LLM's discovery space. XGBoost natively handles missing values by learning optimal split directions.

### 2.6 Clean Customer Registration Dates

**Problem:** `company_create` dates range from 1980-01-01 to 2026-05-31. Dates before ~2010 are likely migration artifacts (the Kingdee cloud platform didn't exist in 1980).

**Action:**
- **Do NOT remove** customers with old `company_create` dates — they may be legitimate long-standing customers migrated to the platform.
- **Cap** `company_create` at a floor date for the purpose of computing `customer_tenure`. Proposed floor: the earliest plausible Kingdee cloud adoption date (investigate, likely ~2015).
- Alternatively, let the LLM decide whether to use `company_create` as a feature. If it does, the LLM can propose a transformation (e.g., log-tenure, tenure bins) that handles extreme values.
- **Flag** records with `company_create` before 2015 so downstream analysis can stratify if needed.

**Expected impact:** Affects tenure feature only; no records removed.

### 2.7 Validate Zero-Payment Pattern

**Problem:** 55.7% of transactions have `pay_amt <= 0`. This is unusually high but may reflect legitimate B2B trade credit (common in Chinese wholesale/retail).

**Action:**
1. **Do NOT remove** zero-payment records — they are likely legitimate business transactions with deferred payment terms.
2. **Cross-reference** with `owe_amt`: if `pay_amt = 0` and `owe_amt = bill_amt`, this confirms trade credit.
3. **Preserve** the payment fields (`pay_amt`, `owe_amt`, `acct_id`) for LLM feature construction. Payment behavior patterns (e.g., `pay_ratio`, `debt_frequency`, proportion of credit vs. cash) may be strong churn signals — customers who always pay on credit vs. cash may have different churn profiles.
4. Document the pattern and note that monetary features should consider both `bill_amt` (what was ordered) and `pay_amt` (what was paid).

**Downstream impact:**
- LLM should have access to both `bill_amt` and `pay_amt` to construct features like `pay_ratio = sum(pay_amt) / sum(bill_amt)`.
- For IPI computation (L2), use `opt_on` from transactions regardless of payment status — a purchase with deferred payment is still a purchase.

---

## Cleaning Sequence and Dependencies

```
Step 2.1: Remove extreme financial outliers
    |
Step 2.2: Remove system/test accounts
    |
Step 2.3: Handle returns (remove negative bill_amt; investigate zero bill_amt)
    |       --> Side output: return_statistics lookup table
    |
Step 2.4: Remove orphan line items
    |
Step 2.5: Handle missing values (no action — preserve for LLM)
    |
Step 2.6: Flag/cap old company_create dates
    |
Step 2.7: Validate zero-payment pattern (no removal — document only)
    |
    v
Output: Cleaned datasets + exclusion log
```

Steps 2.1–2.4 are **data removal** steps (order matters — remove outliers first so they don't affect system account investigation). Steps 2.5–2.7 are **validation/documentation** steps (no records removed).

Industry classification (`industry_servereid.csv`) is used as-is with all 13 original categories. 175–213 stores per category provides ample statistical mass for L2 threshold computation — no collapsing needed.

---

## Output Deliverables

| Deliverable | Format | Description |
|-------------|--------|-------------|
| Cleaned `srv_sales` | Parquet | Sales records after all removals |
| Cleaned `srv_saleitems` | Parquet | Line items after all removals, orphans excluded |
| `return_statistics.csv` | CSV | Per-customer return counts and amounts (computed before removal) |
| `exclusion_log.csv` | CSV | Every removed record with reason code, step number |
| `cleaning_summary.md` | Markdown | Record counts before/after each step, decisions made |

---

## Decisions Requiring Your Input

1. **Financial outlier cap:** Is 10M (`bill_amt`) a reasonable upper bound, or should we use a data-driven approach (e.g., remove records > P99.99)?

2. **System account threshold:** What transaction count threshold should trigger investigation? Proposed: top 50 customers by transaction count, then set threshold based on visual inspection of the gap.

3. **Zero-amount transactions (`bill_amt = 0`):** Remove, keep, or investigate first? Current proposal: investigate a sample before deciding.

4. **`company_create` floor date:** Should we cap at 2015, or use a different cutoff? Or leave it entirely to the LLM to handle?

5. **Return signal preservation:** Should we compute the return statistics lookup table before removing returns, so the LLM can consider return behavior as a feature?

---

## Relationship to Downstream Steps

### Step 3 (Temporal Structuring)
- Needs cleaned transaction dates (`opt_on`) free of return records to correctly assign customers to observation/label windows.
- Needs industry labels to be finalized (for L2 category-level IPI computation).

### Step 4 (Feature Engineering — LLM-Guided)
- The LLM receives the **cleaned data schema** (all columns preserved, anomalies removed).
- The LLM proposes candidate features from cross-table joins and aggregations.
- Cleaning preserves maximum column diversity for the LLM to explore.
- The return statistics lookup table provides return-related features without contaminating core data.

### Step 5 (Label Construction)
- **L1 (fixed threshold):** Computed on cleaned `srv_sales` — straightforward.
- **L2 (category-level dynamic):** Computed on cleaned `srv_sales` joined with `industry_servereid.csv` (13 categories). The category-level IPI pool uses all inter-purchase intervals from customers in each industry category. Cleaning ensures these pools are not contaminated by system accounts, returns, or outlier amounts.

---

## Implementation Notes

- **Memory management:** Raw files total 7.2 GB. Use chunked processing (`pd.read_csv(chunksize=...)`) or Polars for out-of-core operations. Never load both files simultaneously.
- **Parquet output:** Save cleaned data as Parquet to preserve dtypes and reduce file size (~3-5x compression vs. CSV).
- **Reproducibility:** Log all threshold values, random seeds, and parameter choices. Version-control the cleaning script.
- **Composite keys everywhere:** All joins use `(server_id, shop_id, id/sale_id)`. All customer-level operations use `(server_id, shop_id, company_id)`. Never use `id`, `sale_id`, or `company_id` alone.
