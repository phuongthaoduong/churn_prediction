# Data Report: Customer Churn Prediction Using XGBoost

## 1. Data Overview

This report describes the sales dataset used for building an XGBoost-based customer churn prediction model. The dataset consists of two CSV files located in `sale_data/`:

| File | Records | File Size | Description |
|------|---------|-----------|-------------|
| `srv_sales.csv` | 15,307,120 | 2.1 GB | Sales transaction records |
| `srv_saleitems.csv` | 47,664,837 | 5.1 GB | Line-item details for each sale |

**Total dataset size:** 7.2 GB across ~63 million records.

---

## 2. Data Schema

### 2.1 `srv_sales.csv` — Sales Transactions

| Column | Description |
|--------|-------------|
| `server_id` | Server/terminal identifier |
| `shop_id` | Shop/store identifier |
| `id` | Unique sale transaction ID |
| `opt_on` | Transaction date (YYYY-MM-DD) |
| `code` | Transaction code |
| `company_id` | Customer/company identifier |
| `company_type` | Customer type (retail/wholesale) |
| `company_create` | Customer registration date |
| `bill_amt` | Billed amount |
| `disc_amt` | Discount amount |
| `acct_id` | Payment account ID |
| `acct_name` | Payment account name |
| `pay_amt` | Amount paid |
| `owe_amt` | Outstanding/owed amount |

### 2.2 `srv_saleitems.csv` — Sale Line Items

| Column | Description |
|--------|-------------|
| `server_id` | Server/terminal identifier |
| `shop_id` | Shop/store identifier |
| `id` | Unique line-item ID |
| `sale_id` | Foreign key to `srv_sales.id` |
| `opt_on` | Transaction date (YYYY-MM-DD) |
| `product_id` | Product identifier |
| `oprc` | Original price |
| `disc` | Discount multiplier (1.0 = no discount) |
| `prc` | Final price after discount |
| `qty` | Quantity purchased |
| `amt` | Line-item total amount (prc x qty) |

---

## 3. Key Statistics

### 3.1 Scope and Coverage

| Metric | Value |
|--------|-------|
| Transaction period | 2024-05-01 to 2026-05-31 (25 months) |
| Unique merchants (server_id) | 2,533 |
| Unique stores (server_id + shop_id) | 2,583 |
| Unique shop_id values | 27 (reused across merchants — not store count) |
| Unique customers (server_id + shop_id + company_id) | 1,019,023 |
| Unique company_id values | 63,372 (per-merchant auto-increment — NOT customer count) |
| Unique products | 135,448 |
| Customer registration date range | 1980-01-01 to 2026-05-31 |

### 3.2 Customer Types

| Type | Records | Proportion |
|------|---------|------------|
| Retail (零售) | 14,952,450 | 97.7% |
| Wholesale (批发) | 26,929 | 0.2% |
| Unknown/Empty | 327,741 | 2.1% |

The dataset is overwhelmingly composed of **retail customers**.

### 3.3 Financial Summary

| Metric | Value |
|--------|-------|
| Total billed amount | ~15.2 trillion |
| Total paid amount | ~5.16 trillion |
| Total outstanding (owed) amount | ~10.0 trillion |
| Average bill per transaction | ~993,069 |
| Transactions with outstanding debt | 7,883,108 (51.5%) |

### 3.4 Transaction Frequency per Customer

| Metric | Value |
|--------|-------|
| Minimum transactions per customer | 1 |
| Maximum transactions per customer | 103,246 |
| Average transactions per customer | ~15.0 (median: 2) |

### 3.5 Product-Level Statistics

| Metric | Value |
|--------|-------|
| Average quantity per line item | ~273 |
| Average amount per line item | ~318,919 |
| Items with discount applied (disc < 1.0) | 641,787 (1.35%) |

---

## 4. Data Characteristics and Observations

### 4.1 Strengths

- **Large-scale dataset**: Over 15 million sales records and 47 million line items provide a robust foundation for training machine learning models.
- **Rich temporal coverage**: 25 months of transaction data enables the identification of temporal purchasing patterns and seasonality effects.
- **Granular product-level detail**: The `srv_saleitems` table provides item-level granularity, allowing feature engineering at the product, category, and basket level.
- **Customer lifecycle visibility**: The `company_create` field allows computation of customer tenure, a key predictor for churn models.

### 4.2 Potential Data Quality Issues

1. **Missing/Unknown company types**: ~2.1% of records have an empty `company_type`, which may need imputation or exclusion.
2. **Extreme transaction frequency outliers**: One customer has over 1.3 million transactions, suggesting potential data anomalies (e.g., automated/system accounts) that should be investigated and possibly filtered.
3. **High outstanding debt ratio**: Over 51% of transactions have outstanding balances. This pattern could be a strong signal for churn but also warrants validation — it may indicate business-specific payment terms rather than default behavior.
4. **Masked account names**: Some `acct_name` values appear as "??" (likely masked for privacy), limiting analysis of payment method effects.
5. **Historical registration dates**: Some `company_create` dates go back to 1980, which may represent legacy data migration artifacts.
6. **Low discount utilization**: Only 1.35% of line items have discounts applied, suggesting discounting is rare or the field may be underutilized.

---

## 5. Relevance to Churn Prediction

The following features can be engineered from this dataset for the XGBoost churn model:

| Feature Category | Examples |
|-----------------|----------|
| **Recency** | Days since last purchase, time gap trends |
| **Frequency** | Total transactions, purchase frequency, inter-purchase interval |
| **Monetary** | Total spend, average order value, spend trend |
| **Customer tenure** | Days since `company_create` |
| **Payment behavior** | Pay-to-bill ratio, outstanding debt ratio, debt frequency |
| **Product diversity** | Number of unique products purchased, basket size |
| **Discount sensitivity** | Frequency of discounted purchases |
| **Channel usage** | Number of shops visited, primary shop |
| **Temporal patterns** | Monthly/weekly purchase patterns, seasonality |

### 5.1 Churn Definition Considerations

Since the dataset does not include an explicit churn label, churn must be defined based on purchasing inactivity. Common approaches include:

- **Fixed window**: A customer is churned if they have not made a purchase in the last *N* days (e.g., 90 or 180 days).
- **Relative threshold**: Based on each customer's historical inter-purchase interval — if the gap exceeds a multiple of their average frequency, they are considered churned.

Given that the data ends at 2026-05-31, an observation window and prediction window must be carefully defined to avoid data leakage.

---

## 6. Recommendations for Preprocessing

1. **Filter system/anomalous accounts** — Remove customers with extreme transaction counts (e.g., > 100,000) that may be system-generated.
2. **Handle missing values** — Impute or categorize empty `company_type` values.
3. **Validate financial outliers** — Investigate extremely high bill amounts and the max item amount (~10 trillion), which may be data entry errors.
4. **Standardize dates** — Ensure `company_create` values are consistent; consider capping dates before a reasonable threshold (e.g., 2010).
5. **Define train/test split temporally** — Use time-based splitting to prevent data leakage (e.g., train on data before a cutoff date, test on data after).

---

## 7. Summary

This dataset provides a comprehensive view of customer purchasing behavior across 2,583 stores (2,533 merchants), covering 1,019,023 unique customers and 135,448 products over a 25-month period. Note: `company_id` is per-merchant auto-increment and NOT globally unique; the true customer key is `(server_id, shop_id, company_id)`. The data is well-suited for building an XGBoost churn prediction model, with rich transactional and financial features available for engineering. Key preprocessing steps should address outliers, missing values, and churn label definition before model training.
