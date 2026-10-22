"""Generate weekly_report_step3_6.docx from structured data."""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "weekly_report_step3_6.docx")


def set_cell_shading(cell, color_hex):
    """Set background shading for a table cell."""
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), color_hex)
    shading.set(qn("w:val"), "clear")
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    """Set borders for a table cell."""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for edge, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        if val:
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val.get("val", "single"))
            el.set(qn("w:sz"), val.get("sz", "4"))
            el.set(qn("w:color"), val.get("color", "000000"))
            el.set(qn("w:space"), "0")
            tcBorders.append(el)
    tcPr.append(tcBorders)


def add_styled_table(doc, headers, rows, col_widths=None, header_color="2F5496"):
    """Add a formatted table with header styling."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header row
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cell_shading(cell, header_color)

    # Data rows
    for i, row_data in enumerate(rows):
        for j, val in enumerate(row_data):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            # Alternate row shading
            if i % 2 == 1:
                set_cell_shading(cell, "F2F2F2")

    # Apply column widths if provided
    if col_widths:
        for i, width in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(width)

    # Spacing after table
    doc.add_paragraph("")
    return table


def build_report():
    doc = Document()

    # --- Page margins ---
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(2.54)
        section.right_margin = Cm(2.54)

    # --- Default font ---
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)

    # =====================================================================
    # TITLE
    # =====================================================================
    title = doc.add_heading("Weekly Progress Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    subtitle = doc.add_heading(
        "Temporal Structuring through Dataset Assembly (Steps 3\u20136)", level=1
    )

    # Project info block
    info = doc.add_paragraph()
    info.paragraph_format.space_after = Pt(2)
    r = info.add_run("Project: ")
    r.bold = True
    r.font.size = Pt(10)
    info.add_run(
        "Customer churn prediction using XGBoost with a 2\u00d74 factorial design "
        "(fixed vs. category-level dynamic labeling \u00d7 four model updating strategies), "
        "applied to SME retail transaction data from the Kingdee platform."
    ).font.size = Pt(10)

    info2 = doc.add_paragraph()
    info2.paragraph_format.space_after = Pt(2)
    r = info2.add_run("This week\u2019s focus: ")
    r.bold = True
    r.font.size = Pt(10)
    info2.add_run(
        "Building the temporal backbone, engineering features, constructing "
        "dual-scheme churn labels, and assembling the final modeling-ready dataset."
    ).font.size = Pt(10)

    info3 = doc.add_paragraph()
    info3.paragraph_format.space_after = Pt(6)
    r = info3.add_run("Date: ")
    r.bold = True
    r.font.size = Pt(10)
    info3.add_run("Week of July 14\u201320, 2026").font.size = Pt(10)

    # =====================================================================
    # 1. WHAT I SET OUT TO DO
    # =====================================================================
    doc.add_heading("1. What I Set Out to Do", level=2)

    doc.add_paragraph(
        "Last week\u2019s cleaning pipeline produced 14.8M sales records and 46.0M line items "
        "across 1,017,039 customers and 13 industry categories. This week I executed the "
        "remaining four pipeline steps to transform that cleaned transactional data into a "
        "single experiment-ready dataset with features, labels, and validation guarantees."
    )

    add_styled_table(doc,
        ["Step", "Purpose", "What I Did"],
        [
            ["3. Temporal Structuring",
             "Partition data into leakage-safe monthly batches",
             "Defined 90-day observation windows and 30\u201390 day label windows anchored to "
             "20 monthly prediction points; assigned eligible customers to batches"],
            ["4. Feature Engineering",
             "Compute per-customer behavioral signals",
             "Built 25 features (14 customer-level, 6 store-level, 5 time-level) using only "
             "observation-window data; processed 46M line items in chunked reads"],
            ["5. Label Construction",
             "Assign churn labels under both schemes",
             "Computed L1 (fixed 30-day) and L2 (category-level P95 IPI, capped at 90 days) "
             "labels; tracked threshold stability across all 20 batches"],
            ["6. Dataset Assembly",
             "Merge and validate the final dataset",
             "Inner-joined features and labels on composite key; ran 11 validation checks; "
             "computed class balance and scale_pos_weight per batch"],
        ],
        col_widths=[4.5, 4.5, 8.5],
    )

    # =====================================================================
    # 2. STEP 3: TEMPORAL STRUCTURING
    # =====================================================================
    doc.add_heading("2. Step 3: Temporal Structuring", level=2)
    doc.add_heading("2.1 Design Parameters", level=3)

    add_styled_table(doc,
        ["Parameter", "Value", "Rationale"],
        [
            ["Data span", "2024-05-01 to 2026-05-31", "Full 25-month cleaned dataset"],
            ["Observation window", "90 days", "Captures seasonal repurchase cycles; aligns with L2 cap"],
            ["L1 label window", "30 days (fixed)", "Matches post-cleaning per-customer median IPI (30 days)"],
            ["L2 label window", "Up to 90 days", "Capped at observation window length to prevent label leakage"],
            ["Prediction cadence", "Monthly (1st of each month)", "Balances temporal granularity with batch size"],
        ],
        col_widths=[4, 4.5, 9],
    )

    doc.add_heading("2.2 Batch Overview", level=3)
    doc.add_paragraph(
        "The pipeline produced 20 prediction points from 2024-08-01 to 2026-03-01, "
        "yielding 6,340,667 total batch assignments across 936,829 unique customers. "
        "80,210 customers (7.9%) were never eligible because they had no purchase within "
        "any 90-day observation window \u2014 these are extremely sparse customers who "
        "purchased only once during the entire data span."
    )

    add_styled_table(doc,
        ["Metric", "Value"],
        [
            ["Total batch assignments", "6,340,667"],
            ["Unique customers across all batches", "936,829"],
            ["Customers never eligible", "80,210 (7.9%)"],
            ["Median batch appearances per customer", "3"],
            ["Mean batch appearances per customer", "6.8"],
            ["Batch size range", "290,948 \u2013 327,637"],
        ],
        col_widths=[8, 8],
    )

    doc.add_paragraph(
        "Batch sizes are stable across prediction points (CV < 4%), with a slight dip "
        "in 2026-03-01 (290,948) as some customers\u2019 last purchases fell outside the "
        "final observation window."
    )

    doc.add_heading("2.3 Leakage Safeguards", level=3)
    doc.add_paragraph("The temporal design enforces strict separation:")

    leakage_items = [
        ("Observation window", "[obs_start, obs_end) where obs_end == prediction_point \u2014 "
         "features use only data before the prediction point."),
        ("Label window", "[prediction_point, label_end) \u2014 labels use only data after "
         "the prediction point."),
        ("No overlap", "The observation and label windows share no dates."),
        ("Boundary exclusion", "The last usable prediction point (2026-03-01) was chosen so "
         "that even the longest L2 label window (90 days) ends within the data boundary (2026-05-30)."),
    ]
    for term, desc in leakage_items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(term + ": ")
        r.bold = True
        p.add_run(desc)

    # =====================================================================
    # 3. STEP 4: FEATURE ENGINEERING
    # =====================================================================
    doc.add_heading("3. Step 4: Feature Engineering", level=2)
    doc.add_heading("3.1 Feature Set (25 Features)", level=3)

    add_styled_table(doc,
        ["Category", "Count", "Features"],
        [
            ["Customer-level", "14",
             "recency, frequency, monetary, avg_monetary, ipi_mean, ipi_std, "
             "recent_activity, freq_decline_ratio, unique_products, avg_basket_size, "
             "total_qty, pay_ratio, debt_frequency, customer_tenure"],
            ["Store-level", "6",
             "store_txn_volume, store_active_customers, store_median_ipi, "
             "recency_vs_store, frequency_vs_store, monetary_vs_store"],
            ["Time-level", "5",
             "month, quarter, is_cny_season, is_double11, is_spring_festival"],
        ],
        col_widths=[3.5, 2, 12],
    )

    doc.add_paragraph("Design rationale for key features:").bold = True

    rationale_items = [
        ("RFM features (recency, frequency, monetary, avg_monetary)",
         "Standard churn predictors. Computed strictly within the observation window. "
         "monetary and avg_monetary are log1p-transformed to compress the extreme B2B skew "
         "(raw max ~2.7B); the transform is applied after monetary_vs_store is computed from raw values."),
        ("IPI features (ipi_mean, ipi_std)",
         "Capture repurchase regularity. Customers with only one purchase in the observation "
         "window get NaN (47.5% for ipi_mean, 63.5% for ipi_std) \u2014 deferred to XGBoost\u2019s "
         "native missing-value handling rather than imputing."),
        ("Trend features (recent_activity, freq_decline_ratio)",
         "Signal whether activity is accelerating or decelerating. freq_decline_ratio compares "
         "2nd-half to 1st-half frequency within the observation window."),
        ("Store-context features (recency_vs_store, frequency_vs_store, monetary_vs_store)",
         "Relative positioning against store medians \u2014 a customer with frequency = 2 means "
         "something different in a store where the median is 2 vs. 20."),
        ("Seasonality features (is_cny_season, is_double11, is_spring_festival)",
         "Binary flags for Chinese retail calendar events that affect purchasing patterns."),
    ]
    for term, desc in rationale_items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(term + ": ")
        r.bold = True
        p.add_run(desc)

    doc.add_heading("3.2 Feature Statistics", level=3)

    add_styled_table(doc,
        ["Feature", "Mean", "Std", "Min", "Max", "NaN%"],
        [
            ["recency", "32.4", "26.2", "1", "90", "0.0%"],
            ["frequency", "5.6", "54.8", "1", "24,634", "0.0%"],
            ["monetary", "21,993", "3,655,801", "0", "2.7B", "0.0%"],
            ["avg_monetary", "3,218", "36,717", "0", "9.99M", "0.0%"],
            ["ipi_mean", "17.8", "15.4", "1", "89", "47.5%"],
            ["ipi_std", "9.3", "8.2", "0", "61.5", "63.5%"],
            ["recent_activity", "1.8", "18.5", "0", "10,306", "0.0%"],
            ["freq_decline_ratio", "1.1", "1.2", "0", "1,106", "0.0%"],
            ["pay_ratio", "0.51", "44.5", "0", "64,350", "0.0%"],
            ["debt_frequency", "0.51", "0.49", "0", "1.0", "0.0%"],
            ["customer_tenure", "911", "863", "\u2013658", "16,860", "0.2%"],
        ],
        col_widths=[4, 2.5, 3, 2, 2.5, 2],
    )

    doc.add_paragraph("Observations:").bold = True

    obs_items = [
        ("Monetary skew is extreme",
         "Raw mean 21,993 vs. max 2.7B. This reflects the long-tail nature of B2B wholesale. "
         "A log1p transform is applied to monetary and avg_monetary to compress the tail "
         "(the statistics above reflect pre-transform values)."),
        ("pay_ratio outliers",
         "Max 64,350 implies pay_amt > bill_amt by a large factor, likely overpayments or "
         "credit adjustments. The mean (0.51) aligns with the ~55% zero pay_amt rate identified during profiling."),
        ("Negative customer_tenure (\u2013658 days)",
         "0.2% of records have company_create dates after the prediction point. Flagged in "
         "Step 2 as pre-2010 date anomalies; deferred to model-level handling."),
    ]
    for term, desc in obs_items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(term + ": ")
        r.bold = True
        p.add_run(desc)

    p = doc.add_paragraph()
    r = p.add_run("Output: ")
    r.bold = True
    p.add_run("feature_matrix.parquet \u2014 6,340,667 rows \u00d7 29 columns "
              "(4 keys + 25 features), 199.7 MB.")

    # =====================================================================
    # 4. STEP 5: LABEL CONSTRUCTION
    # =====================================================================
    doc.add_heading("4. Step 5: Label Construction", level=2)
    doc.add_heading("4.1 Two Labeling Schemes", level=3)

    add_styled_table(doc,
        ["Scheme", "Definition", "Threshold", "Purpose"],
        [
            ["L1 (Fixed)",
             "Churned if no purchase in [pp, pp + 30 days)",
             "30 days for all customers",
             "Baseline \u2014 simple, interpretable, industry-standard"],
            ["L2 (Category-level dynamic)",
             "Churned if no purchase in [pp, pp + \u03c4_c)",
             "P95 of pooled IPIs per industry category, capped at 90 days",
             "Adapts to industry-specific repurchase cycles"],
        ],
        col_widths=[3.5, 4.5, 4.5, 5],
    )

    doc.add_paragraph(
        "L2 thresholds are computed using only IPIs observed before each prediction point "
        "(no future leakage). Categories with fewer than 100 pooled IPIs fall back to the "
        "global P95 threshold \u2014 in practice, all 13 categories had sufficient data "
        "(420K\u20131.15M IPIs), so no fallback was triggered."
    )

    doc.add_heading("4.2 Churn Rates", level=3)

    add_styled_table(doc,
        ["Metric", "L1", "L2"],
        [
            ["Overall churn rate", "60.4%", "48.6%"],
            ["Churn rate range across batches", "57.6% \u2013 69.7%", "43.6% \u2013 58.0%"],
            ["L1\u2013L2 agreement", "88.2%", "\u2014"],
        ],
        col_widths=[7, 4.5, 4.5],
    )

    doc.add_paragraph(
        "The L1 scheme labels ~12 percentage points more customers as churned than L2, "
        "which is expected: L2\u2019s longer thresholds (up to 90 days vs. L1\u2019s fixed 30 days) "
        "give customers more time to return before being labeled as churned."
    )

    p = doc.add_paragraph()
    r = p.add_run("The 2026-02-01 anomaly: ")
    r.bold = True
    p.add_run(
        "L1 churn spiked to 69.7% while L2 remained stable at 46.7%, dropping agreement "
        "to 76.9%. This batch\u2019s observation window (Nov 3 \u2013 Feb 1) captures the Chinese "
        "New Year period, when many SME customers pause purchasing. L1\u2019s fixed 30-day "
        "window is too short to accommodate this seasonal gap, while L2\u2019s category-level "
        "thresholds (many at 60\u201390 days) absorb it. This is precisely the type of seasonal "
        "misclassification that motivates the L2 scheme."
    )

    doc.add_heading("4.3 L2 Category Thresholds (Latest Batch: 2026-03-01)", level=3)

    add_styled_table(doc,
        ["Category", "Threshold", "Capped?"],
        [
            ["\u533b\u836f\u5668\u68b0 (Medical)", "90 days", "Yes (raw: 117 days)"],
            ["\u5b89\u9632\u7535\u5b50 (Security/Electronics)", "90 days", "Yes (raw: 97 days)"],
            ["\u6570\u7801\u7535\u5668 (Digital/Appliances)", "90 days", "Yes (raw: 103 days)"],
            ["\u7f8e\u5bb9\u914d\u9970 (Beauty)", "90 days", "Yes (raw: 116 days)"],
            ["\u7cbe\u7ec6\u5316\u5de5 (Fine Chemical)", "88 days", "No"],
            ["\u5bb6\u5c45\u88c5\u9970 (Home Decor)", "87 days", "No"],
            ["\u5de5\u4e1a\u6750\u6599 (Industrial)", "87 days", "No"],
            ["\u6bcd\u5a74\u7528\u54c1 (Mother & Baby)", "86 days", "No"],
            ["\u6587\u4f53\u7528\u54c1 (Sports/Office)", "83 days", "No"],
            ["\u5546\u8d85 (Supermarket)", "69 days", "No"],
            ["\u670d\u88c5\u978b\u5e3d (Apparel)", "62 days", "No"],
            ["\u5305\u88c5\u5370\u5237 (Packaging)", "57 days", "No"],
            ["\u98df\u54c1\u996e\u6599 (Food & Beverage)", "57 days", "No"],
        ],
        col_widths=[6, 3, 6],
    )

    doc.add_paragraph(
        "The thresholds form two natural clusters: high-consideration categories "
        "(Medical, Electronics, Beauty \u2014 86\u201390 days) and high-frequency categories "
        "(Apparel, Packaging, Food \u2014 57\u201369 days). This aligns with the industry IPI "
        "structure identified during profiling on July 12."
    )

    doc.add_heading("4.4 L2 Threshold Stability", level=3)

    add_styled_table(doc,
        ["Category", "Min tau", "Max tau", "Range", "CV"],
        [
            ["\u98df\u54c1\u996e\u6599 (Food & Bev)", "29 days", "57 days", "28 days", "18.4%"],
            ["\u5305\u88c5\u5370\u5237 (Packaging)", "27 days", "57 days", "30 days", "19.0%"],
            ["\u7f8e\u5bb9\u914d\u9970 (Beauty)", "40 days", "90 days", "50 days", "19.9%"],
            ["\u5546\u8d85 (Supermarket)", "30 days", "69 days", "39 days", "21.1%"],
            ["\u6bcd\u5a74\u7528\u54c1 (Mother & Baby)", "34 days", "86 days", "52 days", "22.6%"],
            ["\u6587\u4f53\u7528\u54c1 (Sports/Office)", "34 days", "83 days", "49 days", "22.0%"],
            ["\u670d\u88c5\u978b\u5e3d (Apparel)", "24 days", "62 days", "38 days", "23.3%"],
            ["\u7cbe\u7ec6\u5316\u5de5 (Fine Chemical)", "35 days", "88 days", "53 days", "24.0%"],
            ["\u5bb6\u5c45\u88c5\u9970 (Home Decor)", "34 days", "87 days", "53 days", "24.7%"],
            ["\u5de5\u4e1a\u6750\u6599 (Industrial)", "33 days", "87 days", "54 days", "24.6%"],
            ["\u5b89\u9632\u7535\u5b50 (Security/Elec)", "34 days", "90 days", "56 days", "25.0%"],
            ["\u6570\u7801\u7535\u5668 (Digital/Elec)", "33 days", "90 days", "57 days", "27.0%"],
            ["\u533b\u836f\u5668\u68b0 (Medical)", "37 days", "90 days", "53 days", "22.7%"],
        ],
        col_widths=[5, 2.5, 2.5, 2.5, 2.5],
    )

    doc.add_paragraph(
        "All categories show CV between 18\u201327%. The thresholds trend upward across "
        "earlier-to-later batches as more historical IPIs are pooled, which is expected "
        "behavior for a cumulative percentile estimator. Thresholds stabilize by approximately "
        "batch 8\u201310 (2025-03-01 onward), suggesting that the early-batch threshold volatility "
        "reflects limited data accumulation rather than genuine distributional instability."
    )

    doc.add_paragraph(
        "The REVIEW accounts flagged in the Step 2 report do not appear to be distorting "
        "L2 thresholds \u2014 if they were, I would expect to see threshold compression "
        "(lower \u03c4_c values) in industries with high REVIEW account concentration, "
        "which I do not observe."
    )

    # =====================================================================
    # 5. STEP 6: DATASET ASSEMBLY AND VALIDATION
    # =====================================================================
    doc.add_heading("5. Step 6: Dataset Assembly and Validation", level=2)
    doc.add_heading("5.1 Final Dataset", level=3)

    add_styled_table(doc,
        ["Metric", "Value"],
        [
            ["Total rows", "6,340,667"],
            ["Total columns", "34 (4 keys + 25 features + 5 label/metadata)"],
            ["Prediction points", "20"],
            ["Unique customers", "936,829"],
            ["Unique stores", "2,581"],
            ["Industry categories", "13"],
            ["File size", "201.7 MB"],
        ],
        col_widths=[8, 8],
    )

    doc.add_paragraph(
        "Features (Step 4) and labels (Step 5) were inner-joined on the composite key "
        "(server_id, shop_id, company_id, prediction_point). No rows were lost in the "
        "join \u2014 all 6,340,667 feature rows matched exactly with their label rows."
    )

    doc.add_heading("5.2 Validation Results", level=3)
    doc.add_paragraph("All 11 validation checks passed:")

    add_styled_table(doc,
        ["Check", "Result", "Detail"],
        [
            ["Row count > 0", "PASS", "6,340,667 rows"],
            ["No duplicate keys", "PASS", "0 duplicates"],
            ["Recency >= 0 (no leakage)", "PASS", "No future-data contamination"],
            ["Frequency >= 1", "PASS", "All eligible customers have purchases"],
            ["Labels are binary {0,1}", "PASS", "Both L1 and L2"],
            ["No missing labels", "PASS", "0 NaN in either label column"],
            ["No unexpected NaN features", "PASS", "NaN only in ipi_mean, ipi_std, store_median_ipi, customer_tenure (expected)"],
            ["L1 vs L2 disagree >= 5%", "PASS", "11.8% disagreement"],
            ["Recency <= 90 (obs window)", "PASS", "No values exceed observation window"],
            ["All 20 batches present", "PASS", "20 batches found"],
            ["Recency > 0 (strict leakage)", "PASS", "No same-day purchases leaked"],
        ],
        col_widths=[5, 2, 9],
    )

    doc.add_heading("5.3 Class Balance and scale_pos_weight", level=3)

    add_styled_table(doc,
        ["Scheme", "Overall Churn Rate", "SPW Range", "SPW Mean"],
        [
            ["L1", "60.4%", "0.43 \u2013 0.74", "0.67"],
            ["L2", "48.6%", "0.72 \u2013 1.29", "1.04"],
        ],
        col_widths=[3, 4, 4, 4],
    )

    doc.add_paragraph(
        "L2\u2019s class balance is much closer to 50/50, which is favorable for model training. "
        "L1\u2019s consistent majority-churn imbalance (SPW < 1 in all batches) means the model "
        "will need scale_pos_weight adjustment to avoid predicting churn by default."
    )

    doc.add_heading("5.4 Feature Stability Across Batches", level=3)

    add_styled_table(doc,
        ["Feature", "CV% Across Batches"],
        [
            ["debt_frequency", "1.4%"],
            ["frequency", "3.2%"],
            ["recency", "6.5%"],
            ["monetary", "9.0%"],
            ["pay_ratio", "16.0%"],
        ],
        col_widths=[6, 6],
    )

    doc.add_paragraph(
        "Feature means are stable across batches (CV < 10% for 4 of 5 key features). "
        "The pay_ratio CV of 16% is driven by two batches (2025-02-01 and 2025-03-01) "
        "where the mean jumped to 0.7 \u2014 again coinciding with the CNY period, when "
        "outstanding debts are typically settled. This is real seasonal behavior, not a data issue."
    )

    # =====================================================================
    # 6. CROSS-STEP CONSISTENCY CHECKS
    # =====================================================================
    doc.add_heading("6. Cross-Step Consistency Checks", level=2)
    doc.add_paragraph(
        "These checks verify that numbers remain consistent as data flows through the pipeline:"
    )

    add_styled_table(doc,
        ["Metric", "Step 3", "Step 4", "Step 5", "Step 6", "Consistent?"],
        [
            ["Total rows", "6,340,667", "6,340,667", "6,340,667", "6,340,667", "Yes"],
            ["Unique customers", "936,829", "936,829", "\u2014", "936,829", "Yes"],
            ["Prediction points", "20", "20", "20", "20", "Yes"],
            ["Batch size (2024-08-01)", "313,427", "313,427", "313,427", "313,427", "Yes"],
        ],
        col_widths=[4, 2.5, 2.5, 2.5, 2.5, 2.5],
    )

    doc.add_paragraph(
        "Row counts are perfectly preserved across all four steps, confirming that no "
        "records were silently dropped or duplicated during feature engineering, label "
        "construction, or assembly."
    )

    # =====================================================================
    # 7. WHAT REMAINS UNRESOLVED
    # =====================================================================
    doc.add_heading("7. What Remains Unresolved", level=2)

    add_styled_table(doc,
        ["Issue", "Source", "Risk", "Plan"],
        [
            ["High ipi_mean/ipi_std NaN rate (47.5% / 63.5%)",
             "Single-purchase customers within obs windows",
             "Model may underweight IPI features",
             "XGBoost handles NaN natively; monitor feature importance \u2014 if IPI features "
             "rank low, consider engineering an \u201chas_repeat_purchase\u201d binary flag"],
            ["Extreme monetary skew (max 2.7B) — RESOLVED",
             "Long-tail B2B wholesale",
             "Could dominate tree splits",
             "Resolved: Applied log1p transform to monetary and avg_monetary in Step 4 "
             "(after monetary_vs_store ratio is computed from raw values)."],
            ["Negative customer_tenure (0.2%)",
             "Erroneous company_create dates",
             "Minor \u2014 affects < 0.2% of rows",
             "Defer to XGBoost; if problematic, clip to 0"],
            ["L1 seasonal sensitivity (CNY spike to 69.7%)",
             "Fixed threshold too short for seasonal pauses",
             "L1 model may learn spurious CNY churn signal",
             "This is a known limitation of L1 \u2014 the factorial design explicitly compares "
             "L1 vs L2 to quantify this effect"],
            ["L2 threshold early-batch volatility (CV 18\u201327%)",
             "Cumulative percentile estimator with limited early data",
             "First 2\u20133 batches may have noisy labels",
             "Consider excluding the first 2 batches from training or using them only as burn-in"],
        ],
        col_widths=[4, 3.5, 3.5, 6],
    )

    # =====================================================================
    # 8. SUMMARY AND READINESS
    # =====================================================================
    doc.add_heading("8. Summary and Readiness for Modeling", level=2)

    doc.add_paragraph(
        "The data processing pipeline is complete. Starting from 15.3M raw sales records, "
        "I produced a single dataset_final.parquet containing 6,340,667 customer-batch "
        "observations, each with 25 features and 2 churn labels (L1 fixed, L2 category-level "
        "dynamic), validated for leakage, completeness, and consistency."
    )

    add_styled_table(doc,
        ["Requirement", "Target", "Actual", "Status"],
        [
            ["Leakage-free temporal design", "Strict obs/label separation",
             "11/11 validation checks passed", "Ready"],
            ["Dual labeling schemes", "L1 (fixed) + L2 (dynamic)",
             "Both constructed, 88.2% agreement, 11.8% meaningful disagreement", "Ready"],
            ["Sufficient feature coverage", "\u2265 20 features",
             "25 features across 3 categories", "Ready"],
            ["Class balance documented", "SPW per batch",
             "Computed for both schemes", "Ready"],
            ["Feature stability", "Low batch-to-batch drift",
             "CV < 10% for 4/5 key features", "Ready"],
            ["Single modeling-ready file", "All keys, features, labels",
             "34-column Parquet, 201.7 MB", "Ready"],
        ],
        col_widths=[4, 4, 5.5, 2.5],
    )

    p = doc.add_paragraph()
    r = p.add_run("The dataset is ready for the 2\u00d74 factorial experiment.")
    r.bold = True

    # =====================================================================
    # 9. REFERENCES
    # =====================================================================
    doc.add_heading("9. References", level=2)

    refs = [
        "De Caigny, A., Coussement, K., De Bock, K. W., & Lessmann, S. (2023). "
        "Leveraging purchase sequence information for churn prediction with latent "
        "Dirichlet allocation. Expert Systems with Applications, 232, 120596.",

        "Gattermann-Itschert, T., & Thonemann, U. W. (2022). Proactive customer "
        "retention management in a non-contractual B2B setting based on churn prediction "
        "with random forests. Industrial Marketing Management, 107, 134\u2013147.",

        "Jain, H., Yadav, G., & Manohar, R. (2024). XGBoost-based customer churn "
        "prediction with SHAP interpretability in the telecommunications industry. "
        "Journal of Computational Science, 75, 102199.",

        "Mena, G., De Caigny, A., Coussement, K., De Bock, K. W., & Lessmann, S. (2023). "
        "Churn prediction with sequential patterns and machine learning. European Journal "
        "of Operational Research, 305(2), 1048\u20131065.",
    ]
    for ref in refs:
        p = doc.add_paragraph(ref, style="List Bullet")
        p.paragraph_format.space_after = Pt(4)

    # Save
    doc.save(OUTPUT_PATH)
    print(f"Report saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
