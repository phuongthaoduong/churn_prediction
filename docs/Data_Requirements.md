# Data Requirements for Thesis Execution

> **Purpose:** Checklist of all data needed to execute the thesis methodology (Chapter 3).
> Each item is traced to the specific thesis component it supports and classified by necessity.
> **Date:** 2026-07-08

---

## Classification Definitions

| Level | Meaning | Consequence If Missing |
|---|---|---|
| **COMPULSORY** | The thesis methodology explicitly depends on this. Without it, one or more research questions cannot be answered. | Must redesign the thesis or abandon the affected research question. |
| **STRONGLY RECOMMENDED** | A major thesis claim relies on this. The experiment can technically run without it, but the contribution is hollow or the results are not credible. | The thesis is defensible only with significant scope reduction and added limitations. |
| **NICE TO HAVE** | Enriches the analysis or strengthens generalizability, but the core experiment does not depend on it. | Minor limitation footnote. No structural impact on the thesis. |

---

## 1. COMPULSORY

These items are **non-negotiable**. Without any one of them, the thesis cannot execute as designed.

### 1.1 Sales Transaction Records

**Status: HAVE** (`srv_sales.csv`)

| Field | Used By | Thesis Reference |
|---|---|---|
| Transaction ID | Deduplication, join key to line items | Sec 3.2 temporal batching |
| Server ID + Shop ID | Composite primary key (NOT `id` alone) | Profiling finding #1 |
| Customer ID (`company_id`) | Customer-level feature aggregation, label assignment | All RQs |
| Transaction date (`opt_on`) | Recency, frequency, IPI, batch assignment, label construction | Sec 3.3 (Eq 1–3), Sec 3.2 |
| Billed amount (`bill_amt`) | Monetary value features | Sec 3.4 (Eq 5–6) |
| Paid amount (`pay_amt`) | Pay ratio feature | Sec 3.4 feature list |
| Owed amount (`owe_amt`) | Debt frequency feature | Sec 3.4 feature list |
| Customer creation date (`company_create`) | Customer tenure feature | Sec 3.4 feature list |

**Known issues to resolve before use:**
- Composite key `(server_id, shop_id, id)` required for all joins
- 4.6% negative `bill_amt` (returns) — need handling strategy
- 55.7% zero `pay_amt` — must verify with Kingdee whether this is credit terms or data issue
- Extreme outlier amounts (~10 trillion) — must remove

---

### ~~1.2 Sale Line Items~~ → Moved to Section 3 (Nice to Have)

**Status: HAVE** (`srv_saleitems.csv`) — but **not essential for the core experiment**.

The 2x4 factorial design (labeling schemes x updating strategies) depends entirely on transaction-level data from `srv_sales`. The line items table supports only 3 secondary features (`unique_products`, `avg_basket_size`, `total_qty`), all of which correlate with frequency. Without the product master table, `product_id` is an opaque integer — the LLM cannot reason about it semantically, and counting distinct IDs adds minimal signal beyond what frequency already captures.

**Reclassification:** See Section 3.6. Becomes STRONGLY RECOMMENDED only if the product master table (Section 2.2) is obtained, because `product_id` then becomes the join key to category and brand metadata.

---

### 1.3 Merchant Industry Classification

**Status: HAVE** (`industry_servereid.csv` — obtained from Kingdee)

All 2,533 merchants (2,583 stores) matched at 100% coverage across 13 industry categories (175–213 stores each).

| Field | Used By | Thesis Reference |
|---|---|---|
| **Industry (行业)** | Category-level dynamic labeling (L2): threshold derived from pooled IPI per industry category | Sec 3.3 (Eq 2–3) |
| | Stratification variable for all cross-category analysis | Sec 3.1, RQ1, RQ3 |
| | Defines which customers share a churn threshold | Sec 3.3 label construction |
| **Sub-industry (子行业)** | Finer grouping if primary categories are too broad | Sec 3.1 category list |

**Why this is compulsory:**
- The 2x4 factorial design has TWO independent variables: labeling scheme (IV1) and update strategy (IV2)
- IV1 compares fixed-threshold (L1) vs category-level dynamic threshold (L2)
- L2 requires knowing each merchant's industry to pool customers by category
- Without industry labels, L2 cannot be constructed, IV1 has only one level, and the factorial design collapses to a 1x4 comparison
- RQ1 ("does category-level dynamic labeling improve churn identification?") becomes unanswerable
- RQ3 ("does the optimal update strategy depend on labeling scheme?") — the interaction test — becomes impossible
- RQ3 is described as "the most distinctive part of the study" (Chapter 3, Sec 3.1)

**Source:** `merchant` table in Kingdee's data model (documented in `data_fields_and_connections(3).md`, Section 6.1.1)

**Status update:** Industry classification has been obtained (`industry_servereid.csv`). All 2,533 merchants (2,583 stores) are matched with 100% coverage across 13 industry categories, each with 175–213 stores — more than sufficient for category-level statistics.

**Industry categories (13):** 服装鞋帽, 医药器械, 包装印刷, 安防电子, 数码电器, 商超, 精细化工, 工业材料, 美容配饰, 食品饮料, 家居装饰, 文体用品, 母婴用品.

---

## 2. STRONGLY RECOMMENDED

Without these, the thesis can technically run but a **major claimed contribution** becomes unsupportable or trivially answered.

### 2.1 Customer Profile Table

**Status: MISSING**

| Field | Used By | Thesis Reference |
|---|---|---|
| Customer ID (客户ID) | Join key | All |
| Customer creation time (客户创建时间) | Customer tenure feature (more reliable than `company_create` in srv_sales) | Sec 3.4 feature list |
| Customer category (分类) | Customer segmentation — potential feature and stratification variable | Sec 3.4 LLM feature candidates |
| Price level (价格等级) | Customer value tier — LLM feature candidate | Sec 3.4 |
| Exclusive discount (专属折扣) | Discount sensitivity context — LLM feature candidate | Sec 3.4 |
| Cumulative debt (累计欠款) | Payment behavior — LLM feature candidate | Sec 3.4 |
| Customer activity level (客户活跃度) | Platform-computed activity — useful benchmark | Sec 3.4 |
| Customer value level (客户价值度) | Platform-computed value — useful benchmark | Sec 3.4 |

**Why this is strongly recommended:**
- The thesis claims (Sec 3.4) that the LLM is provided with "customer profiles" as part of the multi-table schema
- Without customer profile data, the LLM has no customer-level attributes beyond what's derivable from transactions
- The Preliminary RQ ("does LLM-guided feature selection from the multi-table schema outperform RFM?") is designed to test whether cross-table features add value
- If the only tables available are transactions and line items, both the LLM and the RFM baseline draw from the same source — the comparison is nearly meaningless
- Customer tenure (`company_create` in srv_sales) partially covers one field, but may have quality issues (dates back to 1980)

**Impact if missing:**
- Phase 1 (LLM feature selection) produces features almost identical to manual RFM
- The Preliminary RQ is trivially answered: "no, because there was no multi-table schema to reason over"
- Academic contribution #4 (LLM-guided feature selection for non-contractual churn) is unsupported

---

### ~~2.2 Product Master Table~~ → Moved to Section 3 (Nice to Have)

**Status: MISSING** — demoted from STRONGLY RECOMMENDED.

The product master table requires a two-join dependency chain (product → saleitems → sales → customer) to produce customer-level features, all for the preliminary LLM phase that doesn't affect the main 2×4 factorial. The customer profile table (Section 2.1) provides direct entity-level attributes about the prediction target, which is far more valuable for LLM semantic reasoning. See Section 3.1 for details.

---

## 3. NICE TO HAVE

These enrich the analysis but are not required by the thesis methodology.

### 3.1 Product Master Table

**Status: MISSING**

| Field | Potential Use |
|---|---|
| Product ID (商品ID) | Join to srv_saleitems.product_id |
| Product category (商品类别) | Category-level purchase diversity per customer |
| Brand (品牌) | Brand loyalty / switching rate |
| Supplier (供应商) | Supplier concentration per customer |

**Why demoted from Strongly Recommended:**
- Only useful through a two-join chain: product → saleitems (5.1 GB) → sales → customer-level aggregation
- All derived features (category diversity, brand concentration) are indirect — they describe what the customer buys, not the customer themselves
- The customer profile table (Sec 2.1) gives the LLM direct entity-level attributes (price level, activity, value, debt) that are semantically richer for churn reasoning
- "Product category diversity" is the only Chapter 3 example feature that requires this table — the other two examples (payment behavior, monetary trends) come from srv_sales + customer profile
- The Preliminary RQ is answerable without it: a negative result ("LLM adds nothing with limited schema") is still valid
- The main 2×4 factorial is entirely unaffected by whether this table exists

**Value:** Becomes more useful if combined with srv_saleitems and product price table — the three together enable margin analysis, basket composition tracking, and category-level behavioral features. But this is enrichment, not a thesis requirement.

---

### 3.2 Product Price Table

**Status: MISSING**

| Field | Potential Use |
|---|---|
| Purchase/cost price (进货价) | Margin contribution per customer (selling price vs cost) |
| Retail price (零售价) | Discount depth = (retail - actual) / retail |
| Wholesale price (批发价) | Whether customer buys at wholesale vs retail tier |

**Value:** Enables margin-based features that could differentiate high-value from low-value churners. Strengthens the LLM feature selection phase by giving it price structure to reason about. Not referenced in the thesis methodology as a required input.

---

### 3.3 Product Inventory Table

**Status: MISSING**

| Field | Potential Use |
|---|---|
| Store (门店) | Store-product availability context |
| Shelf life (保质期) | Expected repurchase cycle for perishable products |
| Safety stock / min stock | Stockout risk — supply-side churn driver |

**Value:** Could reveal supply-side explanations for churn (customer leaves because product is frequently out of stock). Shelf life is a strong prior for expected repurchase intervals. Not referenced as a required input in Chapter 3.

---

### 3.4 Supplier Table

**Status: MISSING**

| Field | Potential Use |
|---|---|
| Category (分类) | Supplier-product category mapping |
| Supplier payable balance (供应商欠款) | Store financial health as context |

**Value:** Minimal direct relevance to customer churn prediction. Could serve as a store-level health indicator. Low priority.

---

### 3.5 Purchase Order Table

**Status: MISSING**

| Field | Potential Use |
|---|---|
| Business date (业务日期) | Store restocking frequency and patterns |
| Supplier (供应商) | Supply chain diversity |
| Payable/Paid amounts | Store-level cash flow health |

**Value:** Store restocking patterns could be an interesting store-level context feature (does the store restock regularly?). Not referenced in Chapter 3.

---

### 3.6 Sale Line Items (`srv_saleitems`)

**Status: HAVE** (5.1 GB, 47.6M rows)

| Field | Potential Use |
|---|---|
| Product ID (`product_id`) | Count unique products per customer (product diversity) |
| Quantity (`qty`) | Total quantity purchased, average basket size |
| Amount (`amt`) | Item-level monetary aggregation |

**Why demoted from Compulsory:**
- Only enables 3 features: `unique_products`, `avg_basket_size`, `total_qty`
- All 3 correlate with `frequency` (already from srv_sales) — marginal information gain is small
- Without the product master table, `product_id` is an opaque integer with no semantic meaning
- RFM baseline does not use line items
- Label construction does not use line items
- Updating strategies do not use line items
- The entire 2x4 factorial runs on srv_sales alone
- Processing cost is high: 5.1 GB, composite key joins, 245K orphans, 7% negative amounts

**Upgrade condition:** If you obtain the product master table (Section 2.2), this becomes **STRONGLY RECOMMENDED** because `product_id` becomes the join key to product category and brand — enabling genuinely meaningful cross-table features for LLM feature selection.

---

### 3.8 Additional Merchant Metadata (Beyond Industry)

**Status: MISSING**

| Field | Potential Use |
|---|---|
| Store registration time (店铺注册时间) | Store maturity as control variable |
| Number of employees (员工数) | Store size proxy |
| Province / City (省份/城市) | Regional seasonality controls |
| Number of stores (店铺数) | Multi-store merchant identification |

**Value:** Useful as control variables or for subgroup analysis. Province/city could help explain seasonal patterns (CNY effects vary by region). Not required by the thesis design.

---

## 4. Summary: Data Requirements by Methodology Component

### 4.1 Churn Label Construction (IV1 — Fixed vs Dynamic Labeling)

This is the first independent variable in the 2x4 factorial. Without it, the design collapses to 1x4.

| Data | Needed For | Status | Priority |
|---|---|---|---|
| srv_sales: `company_id`, `opt_on` | Both L1 and L2: determine whether customer purchased within threshold window | HAVE | **COMPULSORY** |
| Merchant industry (行业) | L2 only: pool customers by industry category to compute category-level IPI percentile | **HAVE** | **COMPULSORY** |

**Industry data obtained.** L2 (category-level dynamic labeling) is now fully constructible with 13 industry categories.

---

### 4.2 Feature Engineering (Observation Window Features for XGBoost)

These feed the single XGBoost learner shared across all 8 configurations.

| Data | Features It Enables | Status | Priority |
|---|---|---|---|
| srv_sales: `opt_on`, `bill_amt`, `pay_amt`, `owe_amt`, `company_create` | Recency, Frequency, Monetary, IPI (mean/std/CV), pay ratio, debt frequency, customer tenure, freq decline ratio, recent activity | HAVE | **COMPULSORY** |
| srv_sales: `server_id`, `shop_id` | Store-level context features (recency_vs_store, frequency_vs_store, monetary_vs_store) — store identified by `(server_id, shop_id)` composite key | HAVE | **COMPULSORY** |
| Customer profile: `分类`, `价格等级`, `客户活跃度`, `客户价值度`, `累计欠款` | Customer-level attributes for LLM feature selection — direct entity descriptors the LLM can reason about semantically | **MISSING** | **STRONGLY REC.** |
| Product master: `商品类别`, `品牌` + srv_saleitems: `product_id`, `qty` | Product diversity, basket size, brand concentration — indirect features requiring 2-join chain | **MISSING** / HAVE | Nice to have |

**Without customer profile:** RFM features still work. But LLM feature selection has no entity-level attributes beyond transaction aggregations, making the Preliminary RQ comparison trivially negative.

---

### 4.3 Model Updating Strategies (IV2 — Static / Periodic / Rolling / Drift-Triggered)

This is the second independent variable in the 2x4 factorial.

| Data | Needed For | Status | Priority |
|---|---|---|---|
| srv_sales: `opt_on`, all feature fields | S1–S4: temporal batching, retraining on accumulated/windowed data | HAVE | **COMPULSORY** |
| srv_sales: feature distributions per batch | S4: PSI computation for drift detection (Eq 7) | HAVE | **COMPULSORY** |
| srv_sales: labels per batch | S4: AUC decline signal (Eq 8) — requires labels from label window | HAVE | **COMPULSORY** |

**All update strategies run on srv_sales alone.** No additional tables needed.

---

### 4.4 Evaluation Protocol (Prequential / Test-Then-Train)

| Data | Needed For | Status | Priority |
|---|---|---|---|
| srv_sales: `opt_on`, `company_id` | Monthly batch assignment, eligible customer identification | HAVE | **COMPULSORY** |
| Merchant industry (行业) | Category-level stratified analysis, threshold drift monitoring across batches | **HAVE** | **COMPULSORY** |

---

### 4.5 LLM-Guided Feature Selection (Preliminary Phase)

| Data | Needed For | Status | Priority |
|---|---|---|---|
| srv_sales schema | Baseline: LLM sees transaction fields | HAVE | **COMPULSORY** |
| Customer profile schema | LLM reasons about customer attributes (price level, value, activity) — the strongest source of cross-table signal | **MISSING** | **STRONGLY REC.** |
| Product master + saleitems schema | LLM reasons about product categories and basket composition — indirect, requires 2-join chain | **MISSING** / HAVE | Nice to have |

---

### 4.6 Consolidated Priority Table

| Priority | Data | Status | Blocks Which Methodology Component |
|---|---|---|---|
| **COMPULSORY** | srv_sales (transaction records) | HAVE | All components — labels, features, updating, evaluation |
| **COMPULSORY** | Merchant industry classification (行业) | **HAVE** (`industry_servereid.csv` — 13 categories, 100% match) | Label construction (L2), evaluation stratification, threshold drift monitoring |
| **STRONGLY REC.** | Customer profile table | **MISSING** | LLM feature selection (Preliminary RQ) |
| Nice to have | Product master + saleitems + price | MISSING / HAVE | LLM feature enrichment (indirect features) |
| Nice to have | Inventory, supplier, purchase orders | MISSING | Not referenced in methodology |
| Nice to have | Additional merchant metadata | MISSING | Control variables only |

---

## 5. Decision Points

### If you get COMPULSORY + STRONGLY RECOMMENDED (merchant + customer):
- 2x4 factorial runs with credible category-level labeling
- LLM feature selection has customer-level attributes (price level, activity, value, debt) to reason about — meaningfully different from RFM
- Three core academic contributions are fully supportable
- LLM contribution is credible: direct entity attributes give the LLM semantic leverage that transaction aggregations alone cannot

### If you get COMPULSORY only (merchant industry labels):
- 2x4 factorial runs — RQ1, RQ2, RQ3 are answerable
- LLM feature selection phase must be **redesigned or downscoped**:
  - Option A: Frame as "LLM-guided feature engineering from transaction data" (not "from multi-table schema")
  - Option B: Report negative result — LLM adds no value when schema is limited to transactions
  - Option C: Drop Phase 1, use RFM features directly, and remove the Preliminary RQ
- Academic contribution #4 is weakened or removed
- Thesis is still defensible but with one fewer contribution

### If you get NOTHING additional:
- Thesis **cannot execute as written**
- Must redesign:
  - Replace category-level L2 with store-level thresholds (2,583 stores)
  - Drop LLM feature selection or reframe entirely
  - Rewrite Chapter 1 significance, Chapter 2 gap, Chapter 3 methodology, Chapter 4 innovations
  - Effectively a different thesis

---

## 6. Recommended Request to Kingdee

Industry classification has been obtained (`industry_servereid.csv`). Remaining request (covers STRONGLY RECOMMENDED):

```
Tables still needed:
1. customer table — must include: customer_id (客户ID), creation time (客户创建时间),
   category (分类), price level (价格等级)

Scope: Same merchants/stores already covered in srv_sales.csv (2,533 merchants, 2,583 stores)
Format: CSV or any structured format
```

If Kingdee can provide more, also request (nice to have): `product` (category + brand), `product_price`, `product_inventory`.

---

*End of requirements document.*
