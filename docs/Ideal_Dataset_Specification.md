# Ideal Dataset Specification for Thesis Execution

> **Purpose:** Define the ideal dataset across all dimensions, grounded in statistical requirements of the 2x4 factorial design. Each dimension specifies ideal, minimum viable, and current status.
> **Date:** 2026-07-08

---

## 0. Critical Finding: Pooled IPI vs Per-Customer Median IPI

Before specifying the ideal dataset, a methodological issue in the thesis must be resolved, because it fundamentally determines the temporal depth requirement.

Chapter 3 (Eq 2) defines the L2 threshold as:

> "a high percentile of the **pooled** inter-purchase interval distribution across all customers in that category"

**"Pooled" means all individual intervals from all customers are thrown into one distribution.** A customer who buys daily contributes ~730 intervals of 1 day each over 25 months. A customer who buys monthly contributes ~24 intervals of ~30 days. The pooled distribution is therefore dominated by high-frequency buyers.

Empirical comparison from the current dataset:

| Percentile | Pooled IPI | Per-Customer Median IPI | Gap |
|---|---|---|---|
| P50 | 5 days | 25 days | 20 days |
| P75 | 15 days | 60 days | 45 days |
| P90 | — | — | — |
| P95 | 80 days | 179.5 days | 99.5 days |

> Updated with correct customer key `(server_id, shop_id, company_id)`. Per-store median IPI ranges from **1 day** to **230 days** across 2,579 stores (median: 7 days).

### Why this matters

- **If L2 uses pooled P95 as written:** Most category thresholds will be 20–100 days. The L1 fixed threshold is 30 days. For many categories, **L2 and L1 produce nearly identical labels** — the factorial design loses statistical power because IV1 has no real variation.
- **If L2 uses per-customer median P75/P95:** Thresholds range 60–179.5 days. Strong differentiation from L1, but higher percentiles require longer data to accommodate label windows.
- **The thesis must choose one approach and justify it.** The current wording says "pooled," but pooled percentiles are dominated by frequent buyers, which undermines the stated goal of capturing "natural purchase rhythms" for the full customer base.

### Recommendation

Use **per-customer median IPI** at a moderate percentile (P75 or P90) rather than pooled IPI at P95. This:
- Gives equal weight to each customer regardless of purchase frequency
- Produces thresholds that meaningfully differ from L1
- Aligns with the thesis rationale (capturing purchase rhythm heterogeneity)
- Can be justified as: "each customer contributes one summary statistic, preventing high-frequency buyers from dominating the threshold"

**This choice determines the label window length, which cascades into every other dimension below.** The rest of this document presents three scenarios based on the L2 threshold approach chosen.

---

## 1. Dimension: Temporal Depth

### What determines the minimum

```
Minimum data span = Initial training + Evaluation batches + Label tail

Initial training = Observation window + Max label window across all categories
Label tail       = Max label window (so the last batch's labels can be confirmed)
Evaluation       = Number of batches needed for statistical tests
```

### Statistical requirement for evaluation batches

The thesis uses Wilcoxon signed-rank tests on paired monthly batch results:
- N < 8: insufficient — cannot detect anything but extreme effects
- N = 8–11: weak — detects only large effects (effect size r > 0.8)
- N = 12–15: acceptable — detects medium-to-large effects
- N = 16–20: good — detects medium effects with reasonable power
- N > 20: excellent

For a 2x4 factorial with 6 pairwise comparisons within IV2 and 1 comparison for IV1, corrected for multiple testing, **16+ batches is ideal, 12 is the minimum viable.**

### Scenarios

| Scenario | L2 Threshold Method | Typical Max Threshold | Training | Tail | Eval (ideal 16) | Total Needed | Current (25mo) | Verdict |
|---|---|---|---|---|---|---|---|---|
| A | Pooled P95 (as written) | ~80 days | 6 mo | 3 mo | 16 mo | **25 months** | 25 mo | Barely fits |
| B | Per-customer median P75 | ~60 days | 6 mo | 2 mo | 16 mo | **24 months** | 25 mo | ✅ Fits |
| C | Per-customer median P95 | ~179.5 days | 6 mo | 6 mo | 16 mo | **28 months** | 25 mo | 3 months short |

### Ideal

**30–36 months** of continuous transaction data. This accommodates per-customer median P75/P90 thresholds with 16+ evaluation batches and no need to cap the threshold artificially.

### Minimum viable

**25 months works** if you either:
- Use pooled IPI (as written) — but then L2 barely differs from L1
- Cap the L2 threshold at ~90 days if using higher percentiles
- Use per-customer median P75 (~60 days) — fits within 25 months comfortably

### Current status

25 months available. Feasible under Scenario A or with capping, but not ideal.

---

## 2. Dimension: Number of Industry Categories

### What determines the minimum

The thesis claims "between-category differences represent the main source of purchase rhythm heterogeneity." To test this:
- Need enough categories to show variation exists
- Need the categories to actually differ in purchase rhythms
- 2 categories → can only show "different vs same," no pattern
- 3–4 categories → can show a gradient (fast → slow)
- 5+ categories → can show robust between-category variation

### Additional constraint: factorial design power

The L2 threshold is computed per category. If categories are too few, the L2 scheme produces too few distinct thresholds, reducing its differentiation from L1 (which uses one global threshold).

| Categories | Distinct L2 Thresholds | L1 vs L2 Differentiation |
|---|---|---|
| 2 | 2 | Weak — only two threshold values |
| 3 | 3 | Moderate |
| 4–5 | 4–5 | Good |
| 6+ | 6+ | Strong, but need more shops to fill them |

### Ideal

**4–5 distinct industry categories** with meaningfully different purchase rhythms (e.g., convenience/FMCG, F&B, apparel, durables/hardware).

### Minimum viable

**3 categories** — enough to show a gradient but limits the robustness of between-category claims.

### Current status

**Unknown.** No industry labels available. Must request from Kingdee before assessing.

---

## 3. Dimension: Number of Shops per Category

### What determines the minimum

Category-level thresholds are computed by pooling all customers within a category. The shops within a category must be numerous enough that:
- The pooled IPI distribution reflects category behavior, not one shop's idiosyncrasies
- Within-category variation can be measured (to verify the thesis claim that between > within)
- One outlier shop doesn't dominate the category threshold

### Statistical consideration

To test between-category > within-category IPI variance, you need:
- Multiple shops per category to estimate within-category variance
- With only 2 shops: within-category variance has 1 degree of freedom — unreliable
- With 5 shops: within-category variance has 4 degrees of freedom — minimally estimable
- With 10+ shops: robust estimates

### The binding constraint with 26 shops

| Categories | Shops per Category (even split) | Within-Category DF | Assessment |
|---|---|---|---|
| 3 | ~8–9 | 7–8 | Reasonable |
| 4 | ~6–7 | 5–6 | Minimally viable |
| 5 | ~5 | 4 | Tight |
| 6+ | ~4 | 3 | Insufficient |

Real distributions will be uneven. If one category has 15 shops and another has 3, the small category's threshold is unreliable.

### Ideal

**10+ shops per category**, across 4 categories = **40+ shops total**.

### Minimum viable

**5 shops per category**, across 3–4 categories = **15–20 shops total**.

### Current status

26 usable shops. Supports 3–4 categories at minimum viable levels if the distribution is not too skewed. Must verify after obtaining industry labels.

---

## 4. Dimension: Number of Customers per Category

### What determines the minimum

The category-level threshold is a percentile of the IPI distribution. Percentile stability depends on sample size:
- P90 requires ~50+ observations to be stable (rule of thumb: 10 / (1-p) for percentile p)
- P95 requires ~200+ observations
- P99 requires ~1000+ observations

For pooled IPI, "observations" = number of inter-purchase intervals (abundant — millions).
For per-customer median IPI, "observations" = number of customers with sufficient history.

### Current status

Per-shop customer counts range from 1 (shop 21) to 45,188 (shop 1). Even small shops have hundreds of customers. After grouping into 3–4 categories, each category will have thousands of customers — more than enough for stable percentile estimates.

**This dimension is not a binding constraint.** The data is adequate.

---

## 5. Dimension: Transaction Density per Customer

### What determines the minimum

Features require different amounts of history:

| Feature | Minimum Transactions Needed |
|---|---|
| Recency | 1 |
| Frequency | 1 |
| Monetary (average) | 1 |
| IPI mean, std, CV | 3 (need 2+ intervals) |
| Frequency decline ratio | 2 (need purchases in both halves of observation window) |
| Basket diversity trend | 3+ |

### The sparse customer problem

From profiling:
- 44.2% of customers have exactly 1 transaction in 25 months (based on true customer key: server_id + shop_id + company_id)
- 63.9% have 1–3 transactions
- Per batch (90-day observation window): ~35% of eligible customers have only 1 purchase in the window

For these sparse customers:
- IPI features are undefined or meaningless
- Frequency decline ratio is undefined
- The model receives mostly null/imputed feature values

### Ideal

**70%+ of eligible customers per batch should have >= 3 transactions in the observation window**, enabling computation of all core features.

This could be achieved by:
- Longer observation window (e.g., 120 days instead of 90) — but this consumes more of the limited data span
- Requiring minimum transaction count for eligibility (e.g., >= 2 purchases) — but this excludes early-disengagement signals
- Accepting sparse customers and engineering features that degrade gracefully (e.g., using transaction count as a feature so the model learns different patterns for sparse vs dense customers)

### Minimum viable

**50%+ of eligible customers with >= 2 transactions in the observation window.** The remaining can be handled by XGBoost's native missing value mechanism, with model performance reported separately for sparse vs dense customers.

### Current status

~65% of eligible customers have >= 2 distinct purchase dates per batch. **Marginally adequate.** The 35% sparse customers will dilute overall AUC. Recommend stratified performance reporting.

---

## 6. Dimension: Data Tables

### What determines the minimum

Traced to methodology components:

| Table | Needed For | Priority |
|---|---|---|
| Sales transactions | Features (RFM, IPI), labels (L1, L2), batching, drift detection | COMPULSORY |
| Merchant industry | L2 threshold grouping, category stratification | COMPULSORY |
| Sale line items | Product diversity, basket features | Available |
| Customer profile | LLM feature selection (entity-level attributes) | NOT AVAILABLE — confirmed Kingdee cannot provide (most merchants don't populate classification fields) |
| Product master | LLM feature selection (semantic product attributes) | Nice to have |
| Product price, inventory | Margin, supply-side features | Nice to have |
| Supplier, purchase orders | Store-level context | Nice to have |

### Ideal

**Sales + merchant + line items + customer + product.** This gives the LLM a genuine multi-table schema with entity attributes (customer), behavioral data (transactions), and semantic context (product categories).

### Minimum viable

**Sales + merchant industry labels + line items.** The 2x4 factorial runs. LLM feature selection works with transaction aggregations, product diversity features, and industry context — no entity-level customer attributes, but still a multi-table schema.

### Current status

**Have: sales + line items + merchant industry labels.** This meets the minimum viable requirement. Customer profile table is confirmed unavailable. LLM feature selection will operate on transaction-level features, product-level features (from line items), and industry context.

---

## 7. Dimension: L1 Fixed Threshold

### What determines the value

L1 must be a defensible "conventional" threshold that represents standard practice. The thesis cites common values: 30, 60, 90, 180 days.

### Consideration from the data

- Global customer median IPI (P50) = 25 days → a 30-day threshold means "missed approximately one expected purchase cycle"
- This is reasonable for a default retail setting

### Ideal

**30 days**, matching the customer median IPI and the most common convention for high-frequency retail.

### Alternative

Test 60 or 90 days in sensitivity analysis. The thesis already plans this.

### Current status

30 days is specified in the methodology and supported by data.

---

## 8. Dimension: L2 Threshold Design

### The three design choices that must be made together

| Choice | Options | Impact |
|---|---|---|
| Aggregation unit | Pooled IPI vs per-customer median IPI | Determines threshold range and label window length |
| Percentile | P75, P90, P95 | Higher = more conservative (fewer false churn labels) but longer label windows |
| Cap | None, 90d, 120d, 150d | Caps prevent slow categories from consuming evaluation batches |

### Decision matrix

| Combination | Typical Max Threshold | L1 vs L2 Gap | Eval Batches (25mo) | Feasible? |
|---|---|---|---|---|
| Pooled P90 | ~18 days | **Negative** — L2 < L1 | 21 | L2 is more aggressive than L1, inverts the design |
| Pooled P95 | ~80 days | Moderate (50 days) | 16 | Feasible, moderate differentiation from L1 |
| Pooled P99 | ~173 days | Large (143 days) | 11 | OK power, but P99 is extreme — hard to justify |
| CustMedian P75 | ~60 days | Moderate (30 days) | 16+ | Good balance |
| CustMedian P75 capped at 90d | 90 days | Moderate (60 days) | 16 | Good for 25mo data |
| CustMedian P95 | ~179.5 days | Large | 10 | Tight |
| CustMedian P95 capped at 120d | 120 days | Large (90 days) | 14 | Acceptable |

### Ideal

**Per-customer median IPI at P75, uncapped**, with 30+ months of data. This:
- Gives equal weight to each customer
- Produces thresholds (50–105 days) meaningfully different from L1 (30 days)
- Is defensible as "the customer has been inactive longer than 75% of their category peers' typical purchase rhythms"
- P75 is less extreme than P90/P95, requiring less data depth

### Minimum viable for 25 months

**Per-customer median IPI at P75, capped at 90 days.** This gives 16 evaluation batches and a 60-day gap between L1 (30d) and the maximum L2 threshold.

### Current status

Thesis says "pooled P95" — but as shown above, this produces thresholds too close to L1. **The thesis should revise to per-customer median P75 or justify why pooled is appropriate despite the frequency bias.**

---

## 9. Summary: The Ideal Dataset

| Dimension | Ideal | Minimum Viable | Current | Gap |
|---|---|---|---|---|
| **Temporal depth** | 30–36 months | 25 months (with L2 cap) | 25 months | Adequate if capped |
| **Industry categories** | 4–5 distinct categories | 3 categories | 13 categories (175–213 stores each) | ✅ Exceeds ideal |
| **Shops per category** | 10+ per category | 5+ per category | 175–213 per category | ✅ Exceeds ideal |
| **Total shops** | 40–50 | 15–20 | 2,583 | ✅ Exceeds ideal |
| **Customers per category** | 1,000+ | 200+ | Likely adequate once grouped | OK |
| **Transaction density** | 70%+ with >= 3 txns per window | 50%+ with >= 2 txns per window | 65% with >= 2 | Marginally adequate |
| **Data tables** | Sales + merchant + customer + product + line items | Sales + merchant + line items | Sales + line items + merchant industry | Customer profile NOT AVAILABLE |
| **L1 threshold** | 30 days | 30 days | 30 days (planned) | OK |
| **L2 threshold** | CustMedian P75, uncapped | CustMedian P75, capped at 90d | Pooled P95 (as written — problematic) | Methodology revision needed |
| **Evaluation batches** | 16–20 | 12 | 15–16 (with L2 cap at 90d) | Adequate if capped |

---

## 10. Action Items in Priority Order

### Must Do (thesis cannot proceed without)

1. ~~**Request merchant industry labels from Kingdee.**~~ **DONE.** Industry classification obtained (`industry_servereid.csv`). All 2,533 merchants (2,583 stores) matched at 100% coverage across 13 categories (175–213 stores each). Remaining action: verify whether between-category IPI variance exceeds within-category variance.

2. **Resolve the L2 threshold methodology.** The current "pooled P95" specification produces a global threshold of ~80 days vs L1 = 30 days. Consider:
   - Switch to per-customer median IPI at P75 and justify in the thesis, or
   - Use pooled P99 (but justify why P99 is appropriate), or
   - Acknowledge that pooled P95 produces small L1-vs-L2 differences and test whether even small label differences affect updating strategy rankings (a valid but riskier research position)

### Should Do (major contribution weakened without)

3. ~~**Request customer profile table.**~~ **NOT AVAILABLE.** Confirmed that Kingdee cannot provide this — most merchants don't populate customer classification fields (`客户分类`, `价格等级`, etc.). LLM feature selection will work with transaction + line item + industry data only.

4. **Compute within-category vs between-category IPI variance.** If between <= within, the entire L2 rationale is unsupported. Must know this before committing to the factorial design.

5. **Design a threshold capping strategy.** If using per-customer median P75 with 25 months of data, cap at 90 days to preserve 16 evaluation batches. Document the cap and affected proportion per category.

### Nice to Have

6. **Request more historical data** (back to Jan 2023 for 36+ months). Eliminates the need for capping and gives 20+ evaluation batches.

7. **Request product master table.** Enriches LLM feature selection but not essential.

---

*End of specification.*
