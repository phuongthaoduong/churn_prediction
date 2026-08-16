# Phase 2a: Factorial Experiment Foundation and Run — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the thesis's 2×4 factorial experiment end-to-end: construct the shared evaluation label, tune and freeze XGBoost hyperparameters, settle the LLM-guided-vs-RFM feature question, execute the prequential loop for all 8 configurations (2 label schemes × 4 update strategies) over the 12 evaluation batches under both the primary scheme-specific maturity rule and a pre-specified equalized-lag sensitivity arm, and produce per-batch metrics ready for statistical analysis.

**Architecture:** New scripts continue the `pipeline/` convention with `step8_*` names; all Phase-2 outputs go to a new gitignored `experiment_output/` directory (already in .gitignore). Two new TDD'd modules join the existing shared ones: `strategies.py` (retraining decisions + training-window selection for S1–S4) and `metrics.py` (Recall@k, F1, Jaccard). Everything imports the leakage rule from `pipeline/experiment_config.py` (`usable_batches`) — no script may select training batches any other way.

**Tech Stack:** Python 3.13 (system), pandas 2.2.3, pyarrow 23.0.1, numpy 2.3.3, scikit-learn 1.8.0, xgboost 3.4.0, pytest 8.3.4 — all already installed; no new dependencies.

**Out of scope (Phase 2b, separate plan once this produces metrics):** Wilcoxon signed-rank tests, rank-biserial effect sizes, bootstrap CIs, the cost–performance frontier, remaining sensitivity analyses (alternative fixed threshold, L2 percentile, S4 trigger thresholds, eval horizon), and thesis figures. The equalized-lag maturity sensitivity arm is IN scope here (it requires harness reruns); its statistical comparison against the primary arm belongs to Phase 2b.

---

## Context an engineer needs (read this first)

**The project:** Thesis experiment on Kingdee SME retail churn. Step 7 is DONE (see `docs/preprocessing_log.md` and the Execution record in `docs/superpowers/plans/2026-08-09-step7-experiment-preparation.md`). You have:

- `profiling_output/dataset_final.parquet` — 6,340,667 rows × 34 cols; keys `server_id (str), shop_id (int), company_id (int), prediction_point (datetime)`; 25 features; labels `label_L1`, `label_L2`; plus `L2_tier`, `L2_threshold_days`, `category`. 20 monthly batches 2024-08-01 … 2026-03-01.
- `profiling_output/train_eval_split.csv` — per batch: `prediction_point, l1_label_close, l2_label_close, role, eligible_customers, n_rows, matured_for_L1_at_first_eval, matured_for_L2_at_first_eval`. 8 `initial_train` + 12 `evaluation` batches.
- `profiling_output/psi_reference.json` + `psi_batch_distributions.csv` — frozen-edge PSI baselines: PSI between ANY two batches = `compute_psi(props_a, props_b)` over their precomputed rows (per scheme × feature). Monitored features per scheme are in the JSON.
- `profiling_output/srv_sales_step2_3.parquet` — cleaned sales (~14.8M rows) with `server_id, shop_id, company_id, opt_on, …`.
- `pipeline/experiment_config.py` — constants + `label_close_dates(batch_meta, l2_thresholds)` + `usable_batches(t, scheme, close_dates)` (the no-leakage rule). `FIRST_EVAL_PP = 2025-04-01`, `ROLLING_WINDOW_MONTHS = 6`, `PSI_THRESHOLD = 0.2`, `AUC_DECLINE_THRESHOLD = 0.02`, `RECALL_AT_K_PCT = 0.10`, `FEATURE_COLS` (25), `XGB_PARAMS` (provisional).
- `pipeline/drift_utils.py` — `make_bins`, `bin_proportions`, `compute_psi` (guards included).
- `tests/` — 17 passing tests; `tests/conftest.py` puts `pipeline/` on sys.path. Run pytest from the project root.

**The experiment (thesis chap03.tex):** prequential (test-then-train) evaluation over the 12 evaluation batches. Strategies: **S1** static (train once, never update), **S2** periodic (retrain monthly on ALL matured data), **S3** rolling (retrain monthly on the last 6 months of matured data), **S4** drift-triggered (retrain only when PSI > 0.2 on monitored features OR the matured-label AUC declines > 0.02 vs the running mean since last training; uses the rolling window). Uniform full retraining; only `scale_pos_weight` is recomputed per fit. Metrics: AUC, F1, Recall@10% per batch; top-10% list Jaccard between consecutive batches; retraining cost.

**CRITICAL — label maturity:** at any (re)training time `t`, ONLY batches returned by `usable_batches(t, scheme, close_dates)` may contribute training rows. The `role` column alone is a leakage trap (documented in `train_eval_split.csv`): at `FIRST_EVAL_PP`, L1 has 8 matured batches but L2 only 6.

**Design decisions locked in by this plan:**

1. **Shared evaluation outcome (`label_eval`)**: churn = no purchase in `[pp, pp + 60 days)`. The thesis requires the primary scheme comparison to use a *common* downstream horizon rather than either training label. 60 days = the P75 of per-customer median inter-purchase intervals (Step 1 profiling), sits between L1's 30d and L2's cap of 90d, and closes within the data boundary for every batch (last pp 2026-03-01 + 60d = 2026-04-30 < 2026-05-31). Own-label metrics are also computed as the secondary diagnostic. Horizon is a config constant for Phase-2b sensitivity.
2. **Hyperparameter tuning**: random search (20 sampled combos, seed 42) per scheme, evaluated by mean AUC over expanding-window time-series CV folds inside the initial training period (folds defined by `tscv_folds` — leakage-safe via `usable_batches`). Training folds are row-capped at 500,000 (seeded sample) for tractability; validation batches are scored in full. Winner frozen to `experiment_output/tuned_params.json`. Tuned per scheme (the thesis freezes hyperparameters "across all batches and strategies"; per-scheme tuning gives each labeling scheme its best-faith configuration and is recorded as a documented decision).
3. **RFM baseline feature set** (thesis eq. 3.11–3.13 + IPI statistics): `recency, frequency, avg_monetary, ipi_mean, ipi_std, ipi_cv` where `ipi_cv = ipi_std / ipi_mean` (NaN-safe division; NaN where undefined — XGBoost handles natively). The preliminary-RQ comparison trains both sets with tuned params on the same CV folds; the better mean-AUC set per scheme is frozen for the factorial run. A mutual-information report is produced as a diagnostic; the thesis's full MI+importance *filtering* gate is simplified to a report because the 25-feature set already exists (record as a deviation).
4. **Prequential ordering**: at each evaluation batch `t`: (1) strategy may retrain using only `usable_batches(t, …)` data, (2) the current model scores batch `t`, (3) predictions are persisted. Retraining at `t` cannot see batch `t` labels (they close after `t`), so retrain-before-score is leakage-free and matches monthly deployment.
5. **S4 signals at time `t`**: PSI = max over the scheme's monitored features of `compute_psi(reference_props, batch_t_props)` using the FROZEN-edge precomputed distributions (pure lookup; reference batch = last matured batch at the previous training event). AUC signal uses own-label AUCs of batches whose labels have closed by `t` only. Either signal fires → retrain.
6. **F1 threshold = 0.5** (documented; predicted probabilities are class-weighted via scale_pos_weight). **k = ceil(10% of scored customers)** for Recall@k and Jaccard.
7. **Initial models at `FIRST_EVAL_PP`**: S1 and S2 train on ALL matured batches; S3 and S4 train on the matured batches within the 6-month rolling window. All use tuned params + frozen feature set. (The Step-7.2 `initial_model_*.json` files were provisional for drift baselines and are NOT reused.)
8. **Runtime expectations**: Task 3 (tuning) is the long pole — several hours; it checkpoints per-combo so it can be interrupted and rerun. The factorial run executes twice (one per maturity mode, decision 9), ~30 fits per scheme per mode worst case (~2–4 h total). Scripts print progress.
9. **Maturity-mode sensitivity arm**: the primary analysis uses the scheme-specific label-maturity rule (Step-7 `usable_batches` over `label_close_dates`). A pre-specified sensitivity arm reruns the entire factorial under an **equalized 90-day maturity lag** applied to BOTH schemes (`equalized_close_dates`: every batch's labels treated as observable at `pp + 90d`). Rationale: scheme-specific maturity gives L1 systematically fresher training data than L2 (labels close earlier); the equalized arm removes that differential recency, so agreement between arms attributes scheme differences to the labels themselves rather than to training-data staleness. Every training event additionally logs `effective_lag_days` = days from the newest usable batch to the training time, making the asymmetry quantifiable. Both arms share the frozen feature set, tuned hyperparameters, and frozen PSI baselines; only the close-dates frame differs.

**File structure created by this plan:**

```
pipeline/
  step8_0_eval_labels.py        # shared 60-day evaluation label
  step8_1_hyperparameter_tuning.py
  step8_2_feature_set_comparison.py
  step8_3_run_experiment.py     # prequential factorial loop
  step8_4_compute_metrics.py
  strategies.py                 # S1-S4 decision logic (TDD)
  metrics.py                    # recall@k, f1, jaccard (TDD)
  experiment_config.py          # MODIFIED: Phase-2 constants + tscv_folds + rfm helper
tests/
  test_tscv_folds.py
  test_strategies.py
  test_metrics.py
experiment_output/              # gitignored
  eval_labels.parquet, tuning_results.csv, tuned_params.json,
  feature_set_comparison.csv, frozen_feature_set.json, mi_report.csv,
  predictions/<mode>/{scheme}_{strategy}_{pp}.parquet, retrain_log.csv,
  metrics_per_batch.csv, metrics_summary.csv, step8_*_summary.txt
```

---

### Task 1: Phase-2 config — constants, `tscv_folds`, RFM helper (TDD)

**Files:**
- Modify: `pipeline/experiment_config.py` (append; do not change existing constants/functions)
- Test: `tests/test_tscv_folds.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_tscv_folds.py`:
```python
import numpy as np
import pandas as pd
import pytest

from experiment_config import (
    EVAL_HORIZON_DAYS,
    RFM_FEATURE_COLS,
    add_ipi_cv,
    equalized_close_dates,
    label_close_dates,
    tscv_folds,
    usable_batches,
)


@pytest.fixture
def close_dates():
    batch_meta = pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-09-01", "2024-10-01", "2024-11-01"],
        "l1_label_end": ["2024-08-31", "2024-10-01", "2024-10-31", "2024-12-01"],
    })
    l2 = pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-09-01", "2024-10-01", "2024-11-01"],
        "category": ["A", "A", "A", "A"],
        "tau_c": [40.0, 44.0, 90.0, 90.0],
    })
    return label_close_dates(batch_meta, l2)


def test_eval_horizon_is_60_days():
    assert EVAL_HORIZON_DAYS == 60


def test_rfm_feature_cols():
    assert RFM_FEATURE_COLS == [
        "recency", "frequency", "avg_monetary", "ipi_mean", "ipi_std", "ipi_cv"
    ]


def test_add_ipi_cv_nan_safe():
    df = pd.DataFrame({"ipi_mean": [10.0, 0.0, np.nan], "ipi_std": [5.0, 3.0, 1.0]})
    out = add_ipi_cv(df)
    assert out["ipi_cv"].iloc[0] == pytest.approx(0.5)
    assert np.isnan(out["ipi_cv"].iloc[1])   # division by zero -> NaN
    assert np.isnan(out["ipi_cv"].iloc[2])   # NaN mean -> NaN
    assert "ipi_cv" not in df.columns        # input not mutated


def test_tscv_folds_l1_expanding_and_leakage_safe(close_dates):
    folds = tscv_folds("L1", close_dates, min_train=2)
    # L1 closes: Aug 08-31, Sep 10-01, Oct 10-31, Nov 12-01.
    # val=Aug: no batch closed by 08-01 -> skipped
    # val=Sep: only Aug closed by 09-01 -> 1 < min_train, skipped
    # val=2024-10-01: usable at 10-01 = {Aug (08-31), Sep (10-01, inclusive)} -> fold
    # val=2024-11-01: usable at 11-01 = {Aug, Sep, Oct (10-31)} -> fold
    assert len(folds) == 2
    train0, val0 = folds[0]
    assert val0 == pd.Timestamp("2024-10-01")
    assert train0 == [pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01")]
    train1, val1 = folds[1]
    assert val1 == pd.Timestamp("2024-11-01")
    assert len(train1) == 3
    # leakage check: every train batch's L1 close <= val pp
    idx = close_dates.set_index("prediction_point")
    for train, val in folds:
        assert all(idx.loc[b, "l1_label_close"] <= val for b in train)


def test_tscv_folds_l2_is_stricter(close_dates):
    # L2 closes: Aug 09-10, Sep 10-15, Oct 12-30, Nov 2025-01-30
    # val=Oct (10-01): usable = {Aug} -> skipped at min_train=2
    # val=Nov (11-01): usable = {Aug (09-10), Sep (10-15)} -> fold
    folds = tscv_folds("L2", close_dates, min_train=2)
    assert len(folds) == 1
    train, val = folds[0]
    assert val == pd.Timestamp("2024-11-01")
    assert train == [pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01")]


def test_tscv_folds_validation_batch_never_in_train(close_dates):
    for scheme in ("L1", "L2"):
        for train, val in tscv_folds(scheme, close_dates, min_train=1):
            assert val not in train


def test_equalized_close_dates_uniform_lag():
    batch_meta = pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-09-01"],
        "l1_label_end": ["2024-08-31", "2024-10-01"],
    })
    eq = equalized_close_dates(batch_meta)
    assert list(eq.columns) == ["prediction_point", "l1_label_close", "l2_label_close"]
    assert (eq["l1_label_close"] == eq["l2_label_close"]).all()
    assert eq["l1_label_close"].iloc[0] == pd.Timestamp("2024-10-30")  # 08-01 + 90d


def test_equalized_mode_removes_scheme_asymmetry():
    batch_meta = pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-09-01", "2024-10-01"],
        "l1_label_end": ["2024-08-31", "2024-10-01", "2024-10-31"],
    })
    eq = equalized_close_dates(batch_meta)
    for t in ("2024-11-01", "2024-12-01", "2025-02-01"):
        assert usable_batches(t, "L1", eq) == usable_batches(t, "L2", eq)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/phuongthaoduong/Project/Data_Processing && python3 -m pytest tests/test_tscv_folds.py -v`
Expected: FAIL — `ImportError: cannot import name 'EVAL_HORIZON_DAYS'`

- [ ] **Step 3: Append the implementation to `pipeline/experiment_config.py`**

Append at the end of the file (keep every existing definition untouched):

```python
# ---------------------------------------------------------------------------
# Phase 2: factorial experiment
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = os.path.join(PROJECT_DIR, "experiment_output")

CLEANED_SALES = os.path.join(OUTPUT_DIR, "srv_sales_step2_3.parquet")
TRAIN_EVAL_SPLIT = os.path.join(OUTPUT_DIR, "train_eval_split.csv")
PSI_REFERENCE_PATH = os.path.join(OUTPUT_DIR, "psi_reference.json")
PSI_DISTRIBUTIONS_PATH = os.path.join(OUTPUT_DIR, "psi_batch_distributions.csv")

EVAL_LABELS_PATH = os.path.join(EXPERIMENT_DIR, "eval_labels.parquet")
TUNING_RESULTS_PATH = os.path.join(EXPERIMENT_DIR, "tuning_results.csv")
TUNED_PARAMS_PATH = os.path.join(EXPERIMENT_DIR, "tuned_params.json")
FROZEN_FEATURES_PATH = os.path.join(EXPERIMENT_DIR, "frozen_feature_set.json")
PREDICTIONS_DIR = os.path.join(EXPERIMENT_DIR, "predictions")
RETRAIN_LOG_PATH = os.path.join(EXPERIMENT_DIR, "retrain_log.csv")

# Shared downstream evaluation outcome: churn = no purchase in [pp, pp + 60d).
# 60d = P75 of per-customer median IPI (Step 1 profiling); between L1's 30d
# and L2's 90d cap; closes inside the data boundary for all 20 batches.
EVAL_HORIZON_DAYS = 60

# RFM baseline (thesis eq. 3.11-3.13 + IPI statistics). ipi_cv derived.
RFM_FEATURE_COLS = ["recency", "frequency", "avg_monetary",
                    "ipi_mean", "ipi_std", "ipi_cv"]

STRATEGIES = ("S1", "S2", "S3", "S4")

# Maturity modes: "scheme" = primary analysis (Step-7 scheme-specific rule);
# "equalized" = sensitivity arm with a uniform lag for BOTH schemes, removing
# the differential data recency inherent in scheme-specific maturity.
MATURITY_MODES = ("scheme", "equalized")
EQUALIZED_LAG_DAYS = 90

F1_THRESHOLD = 0.5           # probability cutoff for F1 (documented decision)
N_TUNING_SAMPLES = 20        # random-search draws, seed 42
TUNING_ROW_CAP = 500_000     # per-fold training row cap during tuning
MIN_CV_TRAIN_BATCHES = 2     # smallest expanding-window training set

TUNING_PARAM_SPACE = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [200, 300, 500],
    "min_child_weight": [1, 5, 10],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def add_ipi_cv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ipi_cv = ipi_std / ipi_mean (NaN where undefined)."""
    out = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = out["ipi_std"].to_numpy(dtype=float) / out["ipi_mean"].to_numpy(dtype=float)
    cv[~np.isfinite(cv)] = np.nan
    out["ipi_cv"] = cv
    return out


def equalized_close_dates(batch_meta: pd.DataFrame) -> pd.DataFrame:
    """Close dates under the equalized-lag sensitivity arm.

    Labels of BOTH schemes are treated as observable exactly
    EQUALIZED_LAG_DAYS after the prediction point, so L1 and L2 train on
    identically aged data. Downstream code (usable_batches, strategies) is
    unchanged — the arm is selected purely by which close-dates frame is
    constructed.
    """
    pps = pd.to_datetime(batch_meta["prediction_point"])
    close = pps + pd.Timedelta(days=EQUALIZED_LAG_DAYS)
    return pd.DataFrame({
        "prediction_point": pps,
        "l1_label_close": close,
        "l2_label_close": close,
    })


def tscv_folds(scheme: str, close_dates: pd.DataFrame,
               min_train: int = MIN_CV_TRAIN_BATCHES) -> list:
    """Expanding-window time-series CV folds inside the initial training period.

    A fold (train_pps, val_pp) validates on batch val_pp using models trained
    only on batches whose labels closed by val_pp (usable_batches), so the
    fold structure itself enforces the no-leakage rule. Validation batches are
    restricted to batches matured at FIRST_EVAL_PP (their labels must be
    observable when tuning runs).
    """
    matured = usable_batches(FIRST_EVAL_PP, scheme, close_dates)
    folds = []
    for val_pp in matured:
        train_pps = usable_batches(val_pp, scheme, close_dates)
        if len(train_pps) >= min_train:
            folds.append((train_pps, val_pp))
    return folds
```

Also add `import numpy as np` next to the existing `import pandas as pd` at the top of `pipeline/experiment_config.py` (the module previously only needed pandas).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_tscv_folds.py -v`
Expected: 8 passed. Then the full suite: `python3 -m pytest tests/ -v` → 25 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/experiment_config.py tests/test_tscv_folds.py
git commit -m "feat: add phase-2 config, leakage-safe TSCV folds, and RFM helper"
```

---

### Task 2: `step8_0_eval_labels.py` — shared 60-day evaluation label

**Files:**
- Create: `pipeline/step8_0_eval_labels.py`
- Output: `experiment_output/eval_labels.parquet`, `experiment_output/step8_0_eval_label_summary.txt`

- [ ] **Step 1: Write the script**

```python
"""
Step 8.0: Shared Evaluation Label Construction
===============================================
Builds label_eval: churn = no purchase in [pp, pp + EVAL_HORIZON_DAYS) for
every (customer, prediction_point) row in dataset_final. This is the common
downstream business outcome the thesis uses as the PRIMARY basis for
comparing labeling schemes (own-label metrics are secondary diagnostics).

Validation invariant: a customer silent for 60 days was silent for 30, so
label_eval == 1 implies label_L1 == 1 (label_eval <= label_L1 elementwise).

Input:
  - profiling_output/dataset_final.parquet   (keys + label_L1 for validation)
  - profiling_output/srv_sales_step2_3.parquet (cleaned sales)

Output:
  - experiment_output/eval_labels.parquet  (keys + label_eval)
  - experiment_output/step8_0_eval_label_summary.txt
"""

import os
from datetime import datetime

import pandas as pd

from experiment_config import (
    CLEANED_SALES,
    DATASET_FINAL,
    EVAL_HORIZON_DAYS,
    EVAL_LABELS_PATH,
    EXPERIMENT_DIR,
    KEY_COLS,
)


def customer_key(df):
    return (df["server_id"].astype(str) + "|"
            + df["shop_id"].astype(str) + "|"
            + df["company_id"].astype(str))


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)

    base = pd.read_parquet(DATASET_FINAL, columns=KEY_COLS + ["label_L1"])
    base["ckey"] = customer_key(base)

    sales = pd.read_parquet(
        CLEANED_SALES, columns=["server_id", "shop_id", "company_id", "opt_on"]
    )
    sales["opt_on"] = pd.to_datetime(sales["opt_on"])
    sales["ckey"] = customer_key(sales)

    parts = []
    for pp in sorted(base["prediction_point"].unique()):
        pp = pd.Timestamp(pp)
        window_end = pp + pd.Timedelta(days=EVAL_HORIZON_DAYS)
        active = set(
            sales.loc[(sales["opt_on"] >= pp) & (sales["opt_on"] < window_end), "ckey"]
        )
        batch = base[base["prediction_point"] == pp].copy()
        batch["label_eval"] = (~batch["ckey"].isin(active)).astype("int8")
        parts.append(batch)
        print(f"  {pp:%Y-%m-%d}: {batch['label_eval'].mean():.4f} eval churn rate")

    out = pd.concat(parts, ignore_index=True)

    # Validation: 60d churn implies 30d churn
    violations = int((out["label_eval"] > out["label_L1"]).sum())
    if violations:
        raise ValueError(
            f"{violations} rows have label_eval=1 but label_L1=0 — "
            "eval window logic is wrong"
        )

    out[KEY_COLS + ["label_eval"]].to_parquet(EVAL_LABELS_PATH, index=False)

    rates = out.groupby("prediction_point")["label_eval"].mean()
    lines = [
        "=" * 70,
        "STEP 8.0: SHARED EVALUATION LABEL (60-DAY HORIZON)",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "=" * 70,
        "",
        f"Rows labeled:        {len(out):,}",
        f"Horizon:             {EVAL_HORIZON_DAYS} days",
        f"Overall churn rate:  {out['label_eval'].mean():.4f}",
        f"L1-implication check: PASS (0 violations)",
        "",
        "Per-batch eval churn rate:",
    ] + [f"  {pd.Timestamp(pp):%Y-%m-%d}  {r:.4f}" for pp, r in rates.items()] + [
        "=" * 70,
    ]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_0_eval_label_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step8_0_eval_labels.py`
Expected: 20 per-batch lines; overall eval churn rate BELOW the L1 rate (~0.60, since a 60-day window gives more chance to return; expect roughly 0.40–0.55); implication check passes. Takes a few minutes (14.8M-row sales scan × 20 windows).

- [ ] **Step 3: Sanity-check**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import pandas as pd
ev = pd.read_parquet('../experiment_output/eval_labels.parquet')
assert len(ev) == 6_340_667
assert set(ev['label_eval'].unique()) <= {0, 1}
assert ev.duplicated(['server_id','shop_id','company_id','prediction_point']).sum() == 0
print('step8_0 outputs OK')
"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step8_0_eval_labels.py
git commit -m "feat: add shared 60-day evaluation label construction"
```

---

### Task 3: `step8_1_hyperparameter_tuning.py` — tune once, freeze

**Files:**
- Create: `pipeline/step8_1_hyperparameter_tuning.py`
- Output: `experiment_output/tuning_results.csv` (checkpointed), `experiment_output/tuned_params.json`, `experiment_output/step8_1_tuning_summary.txt`

- [ ] **Step 1: Write the script**

```python
"""
Step 8.1: Hyperparameter Tuning (tune once, then freeze)
=========================================================
Random search (N_TUNING_SAMPLES combos, seed 42) per label scheme, scored by
mean validation AUC over the expanding-window TSCV folds from
experiment_config.tscv_folds (leakage-safe by construction). Training folds
are row-capped at TUNING_ROW_CAP (seeded sample) for tractability;
validation batches are scored in full.

Checkpointing: each (scheme, combo, fold) result is appended to
tuning_results.csv as soon as it finishes; on rerun, completed combos are
skipped, so the script can be interrupted freely.

Only scale_pos_weight is recomputed per fit (thesis eq. 3.8); everything
else in a combo is frozen across folds.

Input:  profiling_output/dataset_final.parquet, batch_metadata.csv,
        l2_thresholds.csv
Output: experiment_output/tuning_results.csv, tuned_params.json,
        step8_1_tuning_summary.txt
"""

import itertools
import json
import os
import random
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from experiment_config import (
    BATCH_METADATA,
    DATASET_FINAL,
    EXPERIMENT_DIR,
    FEATURE_COLS,
    L2_THRESHOLDS_PATH,
    N_TUNING_SAMPLES,
    TUNED_PARAMS_PATH,
    TUNING_PARAM_SPACE,
    TUNING_RESULTS_PATH,
    TUNING_ROW_CAP,
    label_close_dates,
    tscv_folds,
)

FIXED = {"objective": "binary:logistic", "tree_method": "hist",
         "eval_metric": "auc", "random_state": 42, "n_jobs": -1}


def sample_combos(n, seed=42):
    keys = sorted(TUNING_PARAM_SPACE)
    all_combos = list(itertools.product(*(TUNING_PARAM_SPACE[k] for k in keys)))
    rng = random.Random(seed)
    picked = rng.sample(all_combos, n)
    return [dict(zip(keys, combo)) for combo in picked]


def combo_id(params):
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def fit_and_score(df, train_pps, val_pp, scheme, params):
    train = df[df["prediction_point"].isin(train_pps)]
    if len(train) > TUNING_ROW_CAP:
        train = train.sample(TUNING_ROW_CAP, random_state=42)
    val = df[df["prediction_point"] == val_pp]
    y_tr = train[f"label_{scheme}"]
    n_pos, n_neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"degenerate fold {val_pp} for {scheme}")
    model = XGBClassifier(**FIXED, **params, scale_pos_weight=n_neg / n_pos)
    t0 = time.time()
    model.fit(train[FEATURE_COLS], y_tr)
    fit_s = time.time() - t0
    auc = roc_auc_score(val[f"label_{scheme}"],
                        model.predict_proba(val[FEATURE_COLS])[:, 1])
    return auc, fit_s


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    df = pd.read_parquet(DATASET_FINAL)
    close = label_close_dates(pd.read_csv(BATCH_METADATA),
                              pd.read_csv(L2_THRESHOLDS_PATH))
    combos = sample_combos(N_TUNING_SAMPLES)

    done = set()
    if os.path.exists(TUNING_RESULTS_PATH):
        prev = pd.read_csv(TUNING_RESULTS_PATH)
        done = set(zip(prev["scheme"], prev["combo_id"], prev["val_pp"]))
        print(f"Resuming: {len(done)} fold-results already recorded")
    else:
        pd.DataFrame(columns=["scheme", "combo_id", "val_pp", "auc",
                              "fit_seconds"]).to_csv(TUNING_RESULTS_PATH, index=False)

    for scheme in ("L1", "L2"):
        folds = tscv_folds(scheme, close)
        print(f"{scheme}: {len(folds)} folds x {len(combos)} combos")
        for params in combos:
            cid = combo_id(params)
            for train_pps, val_pp in folds:
                key = (scheme, cid, f"{val_pp:%Y-%m-%d}")
                if key in done:
                    continue
                auc, fit_s = fit_and_score(df, train_pps, val_pp, scheme, params)
                pd.DataFrame([{"scheme": scheme, "combo_id": cid,
                               "val_pp": f"{val_pp:%Y-%m-%d}",
                               "auc": auc, "fit_seconds": fit_s}]).to_csv(
                    TUNING_RESULTS_PATH, mode="a", header=False, index=False)
                print(f"  {scheme} {cid} val={val_pp:%Y-%m-%d} "
                      f"AUC={auc:.4f} ({fit_s:.0f}s)")

    results = pd.read_csv(TUNING_RESULTS_PATH)
    tuned, lines = {}, ["=" * 70, "STEP 8.1: HYPERPARAMETER TUNING",
                       f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70]
    for scheme in ("L1", "L2"):
        agg = (results[results["scheme"] == scheme]
               .groupby("combo_id")["auc"].agg(["mean", "std", "count"])
               .sort_values("mean", ascending=False))
        n_folds = len(tscv_folds(scheme, close))
        complete = agg[agg["count"] == n_folds]
        if complete.empty:
            raise ValueError(f"{scheme}: no combo has all {n_folds} folds finished")
        best_id = complete.index[0]
        best = next(p for p in sample_combos(N_TUNING_SAMPLES)
                    if combo_id(p) == best_id)
        tuned[scheme] = {"params": {**FIXED, **best},
                         "cv_mean_auc": float(complete.loc[best_id, "mean"]),
                         "cv_std_auc": float(complete.loc[best_id, "std"]),
                         "n_folds": int(n_folds)}
        lines += ["", f"SCHEME {scheme} — best of {len(complete)} complete combos:",
                  f"  {best_id}",
                  f"  mean val AUC = {tuned[scheme]['cv_mean_auc']:.4f} "
                  f"(std {tuned[scheme]['cv_std_auc']:.4f}, {n_folds} folds)"]

    with open(TUNED_PARAMS_PATH, "w") as f:
        json.dump(tuned, f, indent=2)
    lines += ["", f"Frozen to {os.path.basename(TUNED_PARAMS_PATH)}", "=" * 70]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_1_tuning_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the fold structure before the long run**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import pandas as pd
from experiment_config import BATCH_METADATA, L2_THRESHOLDS_PATH, label_close_dates, tscv_folds
close = label_close_dates(pd.read_csv(BATCH_METADATA), pd.read_csv(L2_THRESHOLDS_PATH))
for s in ('L1','L2'):
    folds = tscv_folds(s, close)
    print(s, len(folds), 'folds; val batches:', [f'{v:%Y-%m}' for _, v in folds])
"
```
Expected: L1 ~6 folds (val 2024-10 … 2025-03), L2 fewer (~4, val ending 2025-01). Record the actual counts.

- [ ] **Step 3: Run the tuning (long — hours; checkpointed)**

Run: `cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step8_1_hyperparameter_tuning.py 2>&1 | tee ../experiment_output/step8_1_run.log`

This is the long pole (~20 combos × ~10 folds ≈ 200 capped fits). It may be run in the background or overnight; if interrupted, rerun — it resumes. Expected: best mean val AUC in the 0.80–0.90 range for both schemes (val AUC will be LOWER than Step 7.2's in-sample 0.85 — that's normal). If a scheme's best combo ties the provisional defaults, that's fine — freeze it anyway.

- [ ] **Step 4: Sanity-check**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import json
t = json.load(open('../experiment_output/tuned_params.json'))
for s in ('L1','L2'):
    p = t[s]['params']
    assert p['tree_method'] == 'hist' and p['random_state'] == 42
    assert 0.5 < t[s]['cv_mean_auc'] < 1.0
    print(s, t[s]['cv_mean_auc'], {k: p[k] for k in ('max_depth','learning_rate','n_estimators')})
print('step8_1 outputs OK')
"
```

- [ ] **Step 5: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step8_1_hyperparameter_tuning.py
git commit -m "feat: add leakage-safe hyperparameter tuning with checkpointed random search"
```

---

### Task 4: `step8_2_feature_set_comparison.py` — preliminary RQ, freeze the feature set

**Files:**
- Create: `pipeline/step8_2_feature_set_comparison.py`
- Output: `experiment_output/feature_set_comparison.csv`, `frozen_feature_set.json`, `mi_report.csv`, `step8_2_feature_summary.txt`

- [ ] **Step 1: Write the script**

```python
"""
Step 8.2: Feature Set Comparison (Preliminary Research Question)
=================================================================
Compares the LLM-guided 25-feature set against the RFM baseline
(recency, frequency, avg_monetary, ipi_mean, ipi_std, ipi_cv) using the
tuned hyperparameters on the same leakage-safe TSCV folds. The better
mean-AUC set per scheme is frozen for the factorial experiment.

Also produces a mutual-information diagnostic report (thesis eq. 3.10).
DEVIATION (documented): the thesis's MI + importance *filtering* gate is
simplified to a report — the 25-feature set was already constructed and
validated in Step 4; no features are dropped here.

Input:  dataset_final.parquet, batch_metadata.csv, l2_thresholds.csv,
        experiment_output/tuned_params.json
Output: feature_set_comparison.csv, frozen_feature_set.json, mi_report.csv,
        step8_2_feature_summary.txt
"""

import json
import os
from datetime import datetime

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from experiment_config import (
    BATCH_METADATA,
    DATASET_FINAL,
    EXPERIMENT_DIR,
    FEATURE_COLS,
    FROZEN_FEATURES_PATH,
    L2_THRESHOLDS_PATH,
    RFM_FEATURE_COLS,
    TUNED_PARAMS_PATH,
    add_ipi_cv,
    label_close_dates,
    tscv_folds,
)

FEATURE_SETS = {"llm_guided": FEATURE_COLS, "rfm_baseline": RFM_FEATURE_COLS}
MI_SAMPLE = 200_000  # rows for the MI diagnostic (seeded)


def main():
    df = add_ipi_cv(pd.read_parquet(DATASET_FINAL))
    close = label_close_dates(pd.read_csv(BATCH_METADATA),
                              pd.read_csv(L2_THRESHOLDS_PATH))
    tuned = json.load(open(TUNED_PARAMS_PATH))

    rows = []
    for scheme in ("L1", "L2"):
        params = tuned[scheme]["params"]
        for set_name, cols in FEATURE_SETS.items():
            for train_pps, val_pp in tscv_folds(scheme, close):
                train = df[df["prediction_point"].isin(train_pps)]
                val = df[df["prediction_point"] == val_pp]
                y = train[f"label_{scheme}"]
                spw = int((y == 0).sum()) / int((y == 1).sum())
                model = XGBClassifier(**params, scale_pos_weight=spw)
                model.fit(train[cols], y)
                auc = roc_auc_score(val[f"label_{scheme}"],
                                    model.predict_proba(val[cols])[:, 1])
                rows.append({"scheme": scheme, "feature_set": set_name,
                             "val_pp": f"{val_pp:%Y-%m-%d}", "auc": auc})
                print(f"  {scheme} {set_name:<13} val={val_pp:%Y-%m-%d} AUC={auc:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(EXPERIMENT_DIR, "feature_set_comparison.csv"), index=False)

    frozen, lines = {}, ["=" * 70, "STEP 8.2: FEATURE SET COMPARISON",
                         f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70]
    for scheme in ("L1", "L2"):
        agg = (res[res["scheme"] == scheme]
               .groupby("feature_set")["auc"].mean().sort_values(ascending=False))
        winner = agg.index[0]
        frozen[scheme] = {"winner": winner, "feature_cols": FEATURE_SETS[winner],
                          "mean_auc": {k: float(v) for k, v in agg.items()}}
        lines += ["", f"SCHEME {scheme}:"] + [
            f"  {name:<13} mean val AUC = {v:.4f}" for name, v in agg.items()
        ] + [f"  -> frozen: {winner} ({len(FEATURE_SETS[winner])} features)"]

    with open(FROZEN_FEATURES_PATH, "w") as f:
        json.dump(frozen, f, indent=2)

    # MI diagnostic on the pooled L1-matured training slice (seeded sample)
    mi_train_pps = tscv_folds("L1", close)[-1][0]  # largest expanding window
    sample = (df[df["prediction_point"].isin(mi_train_pps)]
              .sample(MI_SAMPLE, random_state=42))
    X = sample[FEATURE_COLS + ["ipi_cv"]].fillna(-999.0)  # sentinel, documented
    mi_rows = []
    for scheme in ("L1", "L2"):
        mi = mutual_info_classif(X, sample[f"label_{scheme}"], random_state=42)
        for feat, v in zip(X.columns, mi):
            mi_rows.append({"scheme": scheme, "feature": feat, "mutual_info": v})
    pd.DataFrame(mi_rows).to_csv(os.path.join(EXPERIMENT_DIR, "mi_report.csv"),
                                 index=False)
    lines += ["", f"MI diagnostic on {MI_SAMPLE:,} sampled rows -> mi_report.csv",
              "(NaN filled with -999 sentinel for MI only; XGBoost uses native NaN)",
              "=" * 70]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_2_feature_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** (~1 hour: 2 sets × 2 schemes × ~10 folds, full rows)

Run: `cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step8_2_feature_set_comparison.py 2>&1 | tee ../experiment_output/step8_2_run.log`
Expected: llm_guided ≥ rfm_baseline mean AUC for both schemes (the thesis hypothesis; if RFM wins for a scheme, that IS a valid preliminary-RQ answer — freeze RFM for that scheme and flag it prominently in your report).

- [ ] **Step 3: Sanity-check**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import json
fz = json.load(open('../experiment_output/frozen_feature_set.json'))
for s in ('L1','L2'):
    assert fz[s]['winner'] in ('llm_guided','rfm_baseline')
    assert len(fz[s]['feature_cols']) in (6, 25)
    print(s, fz[s]['winner'], fz[s]['mean_auc'])
print('step8_2 outputs OK')
"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step8_2_feature_set_comparison.py
git commit -m "feat: add feature-set comparison answering the preliminary RQ"
```

---

### Task 5: `strategies.py` — S1–S4 decision logic (TDD)

**Files:**
- Create: `pipeline/strategies.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_strategies.py`:
```python
import pandas as pd
import pytest

from experiment_config import label_close_dates
from strategies import s4_signals, should_retrain, training_batches


@pytest.fixture
def close_dates():
    batch_meta = pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-09-01", "2024-10-01",
                             "2024-11-01", "2024-12-01", "2025-01-01"],
        "l1_label_end": ["2024-08-31", "2024-10-01", "2024-10-31",
                         "2024-12-01", "2024-12-31", "2025-01-31"],
    })
    l2 = pd.DataFrame({
        "prediction_point": batch_meta["prediction_point"],
        "category": ["A"] * 6,
        "tau_c": [40.0] * 6,
    })
    return label_close_dates(batch_meta, l2)


def test_s2_uses_all_matured(close_dates):
    pps = training_batches("S2", pd.Timestamp("2025-02-01"), "L1", close_dates)
    assert len(pps) == 6  # all six batches' L1 labels closed by 2025-02-01


def test_s3_rolling_window_limits_history(close_dates):
    # 6-month window ending 2025-02-01 starts 2024-08-01 (inclusive)
    pps = training_batches("S3", pd.Timestamp("2025-02-01"), "L1", close_dates,
                           rolling_months=3)
    # window [2024-11-01, 2025-02-01): matured batches Nov, Dec (Jan closes 01-31 <= 02-01 -> included)
    assert pps == [pd.Timestamp("2024-11-01"), pd.Timestamp("2024-12-01"),
                   pd.Timestamp("2025-01-01")]


def test_s3_never_exceeds_maturity(close_dates):
    # at 2024-12-15, Dec batch (closes 12-31) must NOT appear
    pps = training_batches("S3", pd.Timestamp("2024-12-15"), "L1", close_dates,
                           rolling_months=6)
    assert pd.Timestamp("2024-12-01") not in pps


def test_unknown_strategy_raises(close_dates):
    with pytest.raises(ValueError):
        training_batches("S9", pd.Timestamp("2025-01-01"), "L1", close_dates)


def test_should_retrain_matrix():
    assert should_retrain("S1", psi_fired=True, auc_fired=True) is False
    assert should_retrain("S2", psi_fired=False, auc_fired=False) is True
    assert should_retrain("S3", psi_fired=False, auc_fired=False) is True
    assert should_retrain("S4", psi_fired=False, auc_fired=False) is False
    assert should_retrain("S4", psi_fired=True, auc_fired=False) is True
    assert should_retrain("S4", psi_fired=False, auc_fired=True) is True


def test_s4_signals():
    psi_fired, auc_fired = s4_signals(
        psi_max=0.25, auc_history=[0.85, 0.84, 0.86],
        psi_threshold=0.2, auc_threshold=0.02)
    assert psi_fired is True
    assert auc_fired is False  # mean(0.85,0.84)=0.845; 0.845-0.86 < 0.02

    psi_fired, auc_fired = s4_signals(
        psi_max=0.05, auc_history=[0.86, 0.85, 0.80],
        psi_threshold=0.2, auc_threshold=0.02)
    assert psi_fired is False
    assert auc_fired is True   # mean(0.86,0.85)=0.855; 0.855-0.80 = 0.055 > 0.02

    # fewer than 2 matured AUCs -> AUC signal cannot fire
    _, auc_fired = s4_signals(psi_max=0.0, auc_history=[0.5],
                              psi_threshold=0.2, auc_threshold=0.02)
    assert auc_fired is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_strategies.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'strategies'`

- [ ] **Step 3: Write the implementation**

`pipeline/strategies.py`:
```python
"""
Update-strategy decision logic for the 2x4 factorial experiment.
=================================================================
S1 static, S2 periodic (all matured data), S3 rolling window,
S4 drift-triggered (PSI or AUC-decline; rolling window data).

All training-batch selection flows through experiment_config.usable_batches,
so the label-maturity/no-leakage rule cannot be bypassed by any strategy.
"""

import pandas as pd

from experiment_config import ROLLING_WINDOW_MONTHS, usable_batches


def training_batches(strategy, t, scheme, close_dates,
                     rolling_months: int = ROLLING_WINDOW_MONTHS) -> list:
    """Batches a retraining event at time t may use, per strategy.

    S1 uses this only for its single initial fit (all matured data).
    S2: all matured batches. S3/S4: matured batches whose prediction point
    lies in [t - rolling_months, t).
    """
    if strategy not in ("S1", "S2", "S3", "S4"):
        raise ValueError(f"Unknown strategy: {strategy!r}")
    matured = usable_batches(t, scheme, close_dates)
    if strategy in ("S1", "S2"):
        return matured
    window_start = pd.Timestamp(t) - pd.DateOffset(months=rolling_months)
    return [pp for pp in matured if pp >= window_start]


def s4_signals(psi_max, auc_history, psi_threshold, auc_threshold):
    """Drift-trigger signals at a decision point.

    psi_max: max PSI over monitored features, current batch vs reference.
    auc_history: own-label AUCs of batches matured since the last training
    event, in chronological order (last entry = most recent matured batch).
    The AUC signal compares the most recent matured AUC against the running
    mean of the earlier ones (thesis eq. 3.9-3.10 vicinity) and needs at
    least two entries to be defined.
    """
    psi_fired = bool(psi_max > psi_threshold)
    auc_fired = False
    if len(auc_history) >= 2:
        running_mean = sum(auc_history[:-1]) / (len(auc_history) - 1)
        auc_fired = bool(running_mean - auc_history[-1] > auc_threshold)
    return psi_fired, auc_fired


def should_retrain(strategy, psi_fired, auc_fired) -> bool:
    """Whether a strategy retrains at this decision point."""
    if strategy == "S1":
        return False
    if strategy in ("S2", "S3"):
        return True
    if strategy == "S4":
        return psi_fired or auc_fired
    raise ValueError(f"Unknown strategy: {strategy!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_strategies.py -v` → 7 passed. Full suite → 32 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/strategies.py tests/test_strategies.py
git commit -m "feat: add S1-S4 strategy decision logic routed through the maturity rule"
```

---

### Task 6: `metrics.py` — Recall@k, F1, Jaccard (TDD)

**Files:**
- Create: `pipeline/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest

from metrics import f1_at_threshold, jaccard, recall_at_frac, top_frac_keys


def test_recall_at_frac_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # top 20% = 2 customers, both churners
    assert recall_at_frac(y, prob, 0.2) == pytest.approx(1.0)
    # top 10% = 1 customer, 1 of 2 churners
    assert recall_at_frac(y, prob, 0.1) == pytest.approx(0.5)


def test_recall_at_frac_no_churners_is_nan():
    assert np.isnan(recall_at_frac(np.zeros(5), np.linspace(0, 1, 5), 0.2))


def test_f1_at_threshold():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.4, 0.6, 0.1])
    # threshold 0.5: pred = [1,0,1,0] -> TP=1 FP=1 FN=1 -> P=0.5 R=0.5 F1=0.5
    assert f1_at_threshold(y, prob, 0.5) == pytest.approx(0.5)


def test_top_frac_keys_and_jaccard():
    keys = np.array(["a", "b", "c", "d", "e"])
    prob = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    top = top_frac_keys(keys, prob, 0.4)  # ceil(0.4*5)=2 -> {a, b}
    assert top == {"a", "b"}
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == pytest.approx(0.0)  # defined as 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'metrics'`

- [ ] **Step 3: Write the implementation**

`pipeline/metrics.py`:
```python
"""
Evaluation metrics for the factorial experiment (thesis eq. 3.12-3.15).
Recall@k reflects SME intervention capacity; Jaccard measures top-k list
stability between consecutive batches.
"""

import math

import numpy as np
from sklearn.metrics import f1_score


def recall_at_frac(y_true, y_prob, frac) -> float:
    """Share of true churners captured in the top ceil(frac*n) by risk.
    NaN when the batch has no churners (undefined)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_churn = int(y_true.sum())
    if n_churn == 0:
        return float("nan")
    k = math.ceil(frac * len(y_true))
    top_idx = np.argsort(-y_prob)[:k]
    return float(y_true[top_idx].sum() / n_churn)


def f1_at_threshold(y_true, y_prob, threshold) -> float:
    return float(f1_score(np.asarray(y_true),
                          (np.asarray(y_prob) >= threshold).astype(int)))


def top_frac_keys(keys, y_prob, frac) -> set:
    """Set of customer keys in the top ceil(frac*n) by predicted risk."""
    keys = np.asarray(keys)
    y_prob = np.asarray(y_prob)
    k = math.ceil(frac * len(keys))
    return set(keys[np.argsort(-y_prob)[:k]])


def jaccard(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|; 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_metrics.py -v` → 4 passed. Full suite → 36 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/metrics.py tests/test_metrics.py
git commit -m "feat: add recall@k, thresholded F1, and jaccard metrics"
```

---

### Task 7: `step8_3_run_experiment.py` — the prequential factorial loop

**Files:**
- Create: `pipeline/step8_3_run_experiment.py`
- Output: `experiment_output/predictions/<mode>/{scheme}_{strategy}_{pp}.parquet` (192 files: 96 per maturity mode), `retrain_log.csv`, `step8_3_run_summary.txt`

- [ ] **Step 1: Write the script**

```python
"""
Step 8.3: Prequential Factorial Experiment Run
===============================================
For each of the 8 configurations (L1/L2 x S1-S4), walks the 12 evaluation
batches in order. At each batch t:
  1. The strategy may retrain (S2/S3 monthly; S4 on PSI/AUC signal; S1 never)
     using ONLY label-matured batches (strategies.training_batches).
  2. The current model scores batch t; predictions are persisted.

PSI for S4 is a pure lookup in the frozen-edge psi_batch_distributions.csv
(reference batch = last matured batch at the previous training event).
The S4 AUC signal uses own-label AUCs of matured batches only.

The full run executes TWICE, once per maturity mode:
  - "scheme"    (PRIMARY): Step-7 scheme-specific label-maturity rule
  - "equalized" (SENSITIVITY): uniform 90-day lag for both schemes,
    removing the differential data recency of scheme-specific maturity
Predictions are written under predictions/<mode>/; every training event
logs its maturity_mode and effective_lag_days (days from the newest
usable batch to the training time).

Deterministic: random_state=42 everywhere; rerunning reproduces the run.
NOT resumable mid-config (model state is sequential): rerunning redoes all
configs and overwrites predictions/ and retrain_log.csv.

Input:  dataset_final.parquet, train_eval_split metadata, tuned_params.json,
        frozen_feature_set.json, psi_reference.json,
        psi_batch_distributions.csv
Output: experiment_output/predictions/*.parquet, retrain_log.csv,
        step8_3_run_summary.txt
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from experiment_config import (
    AUC_DECLINE_THRESHOLD,
    BATCH_METADATA,
    DATASET_FINAL,
    EQUALIZED_LAG_DAYS,
    EXPERIMENT_DIR,
    FIRST_EVAL_PP,
    FROZEN_FEATURES_PATH,
    L2_THRESHOLDS_PATH,
    MATURITY_MODES,
    PREDICTIONS_DIR,
    PSI_DISTRIBUTIONS_PATH,
    PSI_REFERENCE_PATH,
    PSI_THRESHOLD,
    RETRAIN_LOG_PATH,
    STRATEGIES,
    TUNED_PARAMS_PATH,
    add_ipi_cv,
    equalized_close_dates,
    label_close_dates,
)
from drift_utils import compute_psi
from strategies import s4_signals, should_retrain, training_batches

SCHEMES = ("L1", "L2")


def fit_model(df, pps, scheme, cols, params):
    train = df[df["prediction_point"].isin(pps)]
    y = train[f"label_{scheme}"]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"degenerate training set for {scheme} at {pps}")
    model = XGBClassifier(**params, scale_pos_weight=n_neg / n_pos)
    t0 = time.time()
    model.fit(train[cols], y)
    return model, len(train), time.time() - t0


def psi_lookup(dist, scheme, monitored, ref_pp, cur_pp):
    """Max PSI over monitored features between two batches (frozen edges)."""
    sub = dist[dist["scheme"] == scheme]
    worst = 0.0
    for feat in monitored:
        ref = sub[(sub["prediction_point"] == ref_pp) & (sub["feature"] == feat)]
        cur = sub[(sub["prediction_point"] == cur_pp) & (sub["feature"] == feat)]
        worst = max(worst, compute_psi(ref.sort_values("bin")["proportion"].to_numpy(),
                                       cur.sort_values("bin")["proportion"].to_numpy()))
    return worst


def main():
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    df = add_ipi_cv(pd.read_parquet(DATASET_FINAL))
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)
    tuned = json.load(open(TUNED_PARAMS_PATH))
    frozen = json.load(open(FROZEN_FEATURES_PATH))
    psi_ref = json.load(open(PSI_REFERENCE_PATH))
    dist = pd.read_csv(PSI_DISTRIBUTIONS_PATH)

    all_pps = sorted(df["prediction_point"].unique())
    eval_pps = [pp for pp in all_pps if pd.Timestamp(pp) >= FIRST_EVAL_PP]
    assert len(eval_pps) == 12

    retrain_rows, summary = [], []
    w = summary.append
    w("=" * 70)
    w("STEP 8.3: PREQUENTIAL FACTORIAL RUN")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)

    for mode in MATURITY_MODES:
        close = (label_close_dates(batch_meta, l2_thresholds)
                 if mode == "scheme" else equalized_close_dates(batch_meta))
        close_idx = close.set_index("prediction_point")
        pred_dir = os.path.join(PREDICTIONS_DIR, mode)
        os.makedirs(pred_dir, exist_ok=True)
        w("")
        w(f"MATURITY MODE: {mode}"
          + ("" if mode == "scheme"
             else f" (uniform {EQUALIZED_LAG_DAYS}-day lag, both schemes)"))

        for scheme in SCHEMES:
            params = tuned[scheme]["params"]
            cols = frozen[scheme]["feature_cols"]
            monitored = psi_ref[scheme]["monitored_features"]
            label_close_col = "l1_label_close" if scheme == "L1" else "l2_label_close"

            for strategy in STRATEGIES:
                # Initial fit at FIRST_EVAL_PP
                init_pps = training_batches(
                    strategy if strategy in ("S3", "S4") else "S2",
                    FIRST_EVAL_PP, scheme, close)
                model, n_rows, fit_s = fit_model(df, init_pps, scheme, cols, params)
                retrain_rows.append({
                    "maturity_mode": mode, "scheme": scheme, "strategy": strategy,
                    "t": f"{FIRST_EVAL_PP:%Y-%m-%d}", "trigger": "initial",
                    "n_train_rows": n_rows, "n_train_batches": len(init_pps),
                    "effective_lag_days": int((FIRST_EVAL_PP - max(init_pps)).days),
                    "fit_seconds": round(fit_s, 1)})
                ref_pp = f"{max(init_pps):%Y-%m-%d}"   # S4 PSI reference
                last_train_t = FIRST_EVAL_PP

                for t in eval_pps:
                    t = pd.Timestamp(t)
                    if t > FIRST_EVAL_PP:
                        # 1) retraining decision at t (before scoring batch t)
                        psi_fired = auc_fired = False
                        if strategy == "S4":
                            psi_max = psi_lookup(dist, scheme, monitored,
                                                 ref_pp, f"{t:%Y-%m-%d}")
                            # own-label AUCs of batches scored since last
                            # training whose labels have closed by t
                            auc_hist = []
                            for prev in eval_pps:
                                prev = pd.Timestamp(prev)
                                if not (last_train_t <= prev < t):
                                    continue
                                if close_idx.loc[prev, label_close_col] > t:
                                    continue
                                saved = pd.read_parquet(os.path.join(
                                    pred_dir,
                                    f"{scheme}_{strategy}_{prev:%Y-%m-%d}.parquet"))
                                auc_hist.append(roc_auc_score(
                                    saved[f"label_{scheme}"], saved["y_prob"]))
                            psi_fired, auc_fired = s4_signals(
                                psi_max, auc_hist, PSI_THRESHOLD,
                                AUC_DECLINE_THRESHOLD)
                        if should_retrain(strategy, psi_fired, auc_fired):
                            pps = training_batches(strategy, t, scheme, close)
                            model, n_rows, fit_s = fit_model(df, pps, scheme,
                                                             cols, params)
                            trigger = ("monthly" if strategy in ("S2", "S3") else
                                       "+".join(s for s, fired in
                                                (("psi", psi_fired),
                                                 ("auc", auc_fired)) if fired))
                            retrain_rows.append({
                                "maturity_mode": mode, "scheme": scheme,
                                "strategy": strategy, "t": f"{t:%Y-%m-%d}",
                                "trigger": trigger, "n_train_rows": n_rows,
                                "n_train_batches": len(pps),
                                "effective_lag_days": int((t - max(pps)).days),
                                "fit_seconds": round(fit_s, 1)})
                            ref_pp = f"{max(pps):%Y-%m-%d}"
                            last_train_t = t

                    # 2) score batch t with the current model
                    batch = df[df["prediction_point"] == t]
                    out = batch[["server_id", "shop_id", "company_id",
                                 "prediction_point", "label_L1", "label_L2"]].copy()
                    out["y_prob"] = model.predict_proba(batch[cols])[:, 1]
                    out.to_parquet(os.path.join(
                        pred_dir, f"{scheme}_{strategy}_{t:%Y-%m-%d}.parquet"),
                        index=False)

                n_retrains = sum(
                    1 for r in retrain_rows
                    if r["maturity_mode"] == mode and r["scheme"] == scheme
                    and r["strategy"] == strategy and r["trigger"] != "initial")
                w(f"  {mode} {scheme} {strategy}: initial fit + {n_retrains} retrains")
                print(summary[-1])

    pd.DataFrame(retrain_rows).to_csv(RETRAIN_LOG_PATH, index=False)
    w("")
    w(f"Predictions: "
      f"{len(MATURITY_MODES) * len(SCHEMES) * len(STRATEGIES) * len(eval_pps)} "
      f"files under {os.path.basename(PREDICTIONS_DIR)}/<mode>/")
    w(f"Retrain log: {os.path.basename(RETRAIN_LOG_PATH)} "
      f"({len(retrain_rows)} training events incl. initial fits)")
    w("=" * 70)
    report = "\n".join(summary)
    with open(os.path.join(EXPERIMENT_DIR, "step8_3_run_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** (~2–4 hours — the full loop runs once per maturity mode; prints per-config progress)

Run: `cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step8_3_run_experiment.py 2>&1 | tee ../experiment_output/step8_3_run.log`

Expected retrain counts per mode (excluding initial fits): S1 = 0; S2 = 11; S3 = 11 (one per post-first batch each); S4 = somewhere in 1–8 (the Step-7.2 baselines showed PSI crossing 0.2 at the first eval batch for L1 and at the temporal edge — if S4 retrains EVERY batch or NEVER, inspect the PSI lookups and AUC histories before accepting the result; report it either way, do not tweak thresholds to force a "nice" count). In the equalized mode, both schemes' initial fits use the same 6 batches (pp ≤ 2025-01-01) and every effective lag is ≥ 90 days; S4's AUC signal matures later, so equalized-mode S4 may retrain less often.

- [ ] **Step 3: Sanity-check**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import os, pandas as pd
total = 0
for mode in ('scheme', 'equalized'):
    pred_dir = f'../experiment_output/predictions/{mode}'
    files = sorted(os.listdir(pred_dir))
    assert len(files) == 96, (mode, len(files))  # 2 schemes x 4 strategies x 12 batches
    total += len(files)
    p = pd.read_parquet(os.path.join(pred_dir, files[0]))
    assert set(['server_id','shop_id','company_id','prediction_point','label_L1','label_L2','y_prob']) <= set(p.columns)
    assert p['y_prob'].between(0, 1).all()
assert total == 192
log = pd.read_csv('../experiment_output/retrain_log.csv')
assert set(log['maturity_mode']) == {'scheme', 'equalized'}
s1 = log[(log['strategy']=='S1') & (log['trigger']!='initial')]
assert len(s1) == 0, 'S1 must never retrain'
s2 = log[(log['strategy']=='S2') & (log['trigger']!='initial')]
assert len(s2) == 44, 'S2 retrains 11x per scheme per mode'
eq = log[log['maturity_mode']=='equalized']
assert (eq['effective_lag_days'] >= 90).all(), 'equalized lag must be >= 90d'
sch_l1_init = log[(log['maturity_mode']=='scheme') & (log['scheme']=='L1') & (log['trigger']=='initial')]
assert (sch_l1_init['effective_lag_days'] == 31).all()  # 2025-04-01 minus 2025-03-01
print('step8_3 outputs OK;', len(log), 'training events')
"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step8_3_run_experiment.py
git commit -m "feat: add prequential factorial experiment runner for all 8 configurations"
```

---

### Task 8: `step8_4_compute_metrics.py` — per-batch metrics

**Files:**
- Create: `pipeline/step8_4_compute_metrics.py`
- Output: `experiment_output/metrics_per_batch.csv`, `metrics_summary.csv`, `step8_4_metrics_summary.txt`

- [ ] **Step 1: Write the script**

```python
"""
Step 8.4: Per-Batch Metrics
============================
For every (maturity mode, scheme, strategy, evaluation batch): AUC / F1 /
Recall@10% against the SHARED 60-day evaluation label (primary, cross-scheme
comparable) and against the config's own training label (secondary
diagnostic), plus the Jaccard similarity of consecutive top-10% risk lists.

The "scheme" maturity mode is the PRIMARY analysis; the "equalized" mode
(uniform 90-day lag, both schemes) is the pre-specified sensitivity arm.
The summary reports both, plus the equalized-minus-scheme AUC delta per
configuration and the mean effective data lag per configuration.

Summary aggregation per configuration: mean and CV of each metric across the
12 batches, joined with retraining cost (events, rows, cumulative fit time).
Statistical testing (Wilcoxon etc.) is Phase 2b.

Input:  experiment_output/predictions/<mode>/*.parquet, eval_labels.parquet,
        retrain_log.csv
Output: metrics_per_batch.csv, metrics_summary.csv, step8_4_metrics_summary.txt
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiment_config import (
    EVAL_LABELS_PATH,
    EXPERIMENT_DIR,
    F1_THRESHOLD,
    KEY_COLS,
    MATURITY_MODES,
    PREDICTIONS_DIR,
    RECALL_AT_K_PCT,
    RETRAIN_LOG_PATH,
    STRATEGIES,
)
from metrics import f1_at_threshold, jaccard, recall_at_frac, top_frac_keys

SCHEMES = ("L1", "L2")


def main():
    ev = pd.read_parquet(EVAL_LABELS_PATH)
    rows = []
    for mode in MATURITY_MODES:
        pred_dir = os.path.join(PREDICTIONS_DIR, mode)
        for scheme in SCHEMES:
            for strategy in STRATEGIES:
                prev_top = None
                files = sorted(f for f in os.listdir(pred_dir)
                               if f.startswith(f"{scheme}_{strategy}_"))
                for fname in files:
                    pp = fname.rsplit("_", 1)[1].replace(".parquet", "")
                    p = pd.read_parquet(os.path.join(pred_dir, fname))
                    p = p.merge(ev, on=KEY_COLS, how="left", validate="1:1")
                    if p["label_eval"].isna().any():
                        raise ValueError(f"{mode}/{fname}: rows missing eval label")
                    keys = (p["server_id"].astype(str) + "|"
                            + p["shop_id"].astype(str) + "|"
                            + p["company_id"].astype(str)).to_numpy()
                    top = top_frac_keys(keys, p["y_prob"].to_numpy(),
                                        RECALL_AT_K_PCT)
                    rows.append({
                        "maturity_mode": mode, "scheme": scheme,
                        "strategy": strategy, "prediction_point": pp,
                        "n_scored": len(p),
                        "auc_shared": roc_auc_score(p["label_eval"], p["y_prob"]),
                        "f1_shared": f1_at_threshold(p["label_eval"], p["y_prob"],
                                                     F1_THRESHOLD),
                        "recall10_shared": recall_at_frac(p["label_eval"],
                                                          p["y_prob"],
                                                          RECALL_AT_K_PCT),
                        "auc_own": roc_auc_score(p[f"label_{scheme}"], p["y_prob"]),
                        "jaccard_top10_vs_prev": (np.nan if prev_top is None
                                                  else jaccard(top, prev_top)),
                    })
                    prev_top = top

    per_batch = pd.DataFrame(rows)
    per_batch.to_csv(os.path.join(EXPERIMENT_DIR, "metrics_per_batch.csv"),
                     index=False)

    log = pd.read_csv(RETRAIN_LOG_PATH)
    grp = ["maturity_mode", "scheme", "strategy"]
    cost = (log[log["trigger"] != "initial"]
            .groupby(grp)
            .agg(n_retrains=("t", "count"),
                 total_train_rows=("n_train_rows", "sum"),
                 total_fit_seconds=("fit_seconds", "sum")))
    lag = log.groupby(grp).agg(
        mean_effective_lag_days=("effective_lag_days", "mean"))
    perf = per_batch.groupby(grp).agg(
        mean_auc_shared=("auc_shared", "mean"),
        cv_auc_shared=("auc_shared", lambda s: s.std() / s.mean()),
        mean_f1_shared=("f1_shared", "mean"),
        mean_recall10=("recall10_shared", "mean"),
        mean_auc_own=("auc_own", "mean"),
        mean_jaccard=("jaccard_top10_vs_prev", "mean"),
    )
    summary = perf.join(cost, how="left").join(lag, how="left").fillna(
        {"n_retrains": 0, "total_train_rows": 0, "total_fit_seconds": 0})
    summary.to_csv(os.path.join(EXPERIMENT_DIR, "metrics_summary.csv"))

    # Sensitivity delta: equalized minus scheme, per (scheme, strategy)
    wide = summary["mean_auc_shared"].unstack("maturity_mode")
    wide["delta_equalized_minus_scheme"] = wide["equalized"] - wide["scheme"]

    lines = ["=" * 70, "STEP 8.4: FACTORIAL METRICS SUMMARY",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", "=" * 70, "",
             "PRIMARY ANALYSIS (maturity mode: scheme)", "-" * 70,
             summary.loc["scheme"].round(4).to_string(), "",
             "SENSITIVITY ARM (maturity mode: equalized, uniform 90-day lag)",
             "-" * 70,
             summary.loc["equalized"].round(4).to_string(), "",
             "Mean shared-outcome AUC, equalized minus scheme:", "-" * 70,
             wide.round(4).to_string(), "",
             "Primary metric: auc_shared / recall10_shared (60-day shared outcome).",
             "auc_own is the secondary own-label diagnostic. The equalized arm",
             "removes the differential data recency of scheme-specific maturity;",
             "agreement with the primary analysis attributes scheme differences",
             "to the labels themselves rather than to training-data staleness.",
             "=" * 70]
    report = "\n".join(lines)
    with open(os.path.join(EXPERIMENT_DIR, "step8_4_metrics_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** (minutes)

Run: `cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step8_4_compute_metrics.py`
Expected: two 8-row summary blocks (primary "scheme" mode and equalized sensitivity arm) plus the equalized-minus-scheme AUC delta table. Plausibility: shared-outcome AUCs ~0.75–0.90; S2/S3 ≥ S1 on mean AUC (model aging); S1 shows the highest Jaccard (same model → stable lists); S4 retrain cost between S1 and S2/S3; equalized-mode L1 typically at or below scheme-mode L1 (staler data), equalized-mode mean_effective_lag_days ≥ 90.

- [ ] **Step 3: Sanity-check**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import pandas as pd
pb = pd.read_csv('../experiment_output/metrics_per_batch.csv')
assert len(pb) == 192  # 2 modes x 2 schemes x 4 strategies x 12 batches
assert pb.groupby(['maturity_mode','scheme','strategy']).size().eq(12).all()
assert pb['auc_shared'].between(0.5, 1.0).all()
sm = pd.read_csv('../experiment_output/metrics_summary.csv')
assert len(sm) == 16
assert sm[sm['maturity_mode']=='equalized']['mean_effective_lag_days'].min() >= 90
print('step8_4 outputs OK')
"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step8_4_compute_metrics.py
git commit -m "feat: add per-batch and summary metrics for the factorial experiment"
```

---

### Task 9: Final verification

- [ ] **Step 1: Full test suite**

Run: `cd /Users/phuongthaoduong/Project/Data_Processing && python3 -m pytest tests/ -v`
Expected: 36 passed.

- [ ] **Step 2: Leakage audit of the run**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 -c "
import pandas as pd
from experiment_config import (BATCH_METADATA, L2_THRESHOLDS_PATH,
                               equalized_close_dates, label_close_dates)
log = pd.read_csv('../experiment_output/retrain_log.csv', parse_dates=['t'])
bm = pd.read_csv(BATCH_METADATA)
closes = {'scheme': label_close_dates(bm, pd.read_csv(L2_THRESHOLDS_PATH)),
          'equalized': equalized_close_dates(bm)}
# every training event at time t used only batches matured at t under its
# own maturity mode: n_train_batches must equal the recomputed count
from strategies import training_batches
for _, r in log.iterrows():
    strat = 'S2' if r['trigger'] == 'initial' and r['strategy'] in ('S1','S2') else \
            ('S3' if r['trigger'] == 'initial' and r['strategy'] in ('S3','S4') else r['strategy'])
    close = closes[r['maturity_mode']]
    pps = training_batches(strat, r['t'], r['scheme'], close)
    assert r['n_train_batches'] == len(pps), (r.to_dict(), len(pps))
    assert r['effective_lag_days'] == (r['t'] - max(pps)).days, r.to_dict()
print('retrain log consistent with the maturity rule of both modes')
"
```

- [ ] **Step 3: Deliverables inventory**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing && ls -lh experiment_output/eval_labels.parquet \
  experiment_output/tuned_params.json experiment_output/frozen_feature_set.json \
  experiment_output/retrain_log.csv experiment_output/metrics_per_batch.csv \
  experiment_output/metrics_summary.csv && ls experiment_output/predictions/*/*.parquet | wc -l
```
Expected: all files present, predictions count = 192 (96 per maturity mode).

- [ ] **Step 4: Update the plan + decision log and commit**

Mark all checkboxes in this plan `- [x]`, append an "## Execution record" section (deviations, actual fold counts, tuned params chosen, feature-set winners, S4 retrain counts, headline metrics). Also append a short "## 9. Phase-2a execution" section to `docs/preprocessing_log.md` generation is NOT needed — instead record Phase-2a results in `experiment_output/` summaries only (the decision log stays a preprocessing record).

```bash
git add docs/superpowers/plans/2026-08-10-phase2-factorial-experiment.md
git commit -m "docs: mark phase-2a plan executed with results record"
```

---

## What Phase 2a hands to Phase 2b (the analysis plan)

| Deliverable | Feeds |
|---|---|
| `metrics_per_batch.csv` (192 rows: 96 primary + 96 sensitivity) | Wilcoxon signed-rank paired tests, rank-biserial effect sizes, bootstrap CIs, CV-of-AUC stability analysis, scheme×strategy interaction (RQ3) — run on the primary `maturity_mode == "scheme"` rows, then repeated on the equalized rows to confirm conclusions are not driven by differential data recency |
| `metrics_summary.csv` + `retrain_log.csv` | Cost–performance frontier (RQ2) |
| `tuned_params.json`, `frozen_feature_set.json`, `feature_set_comparison.csv`, `mi_report.csv` | Preliminary-RQ write-up; thesis chapter 4 methods |
| `eval_labels.parquet` | Sensitivity reruns at 30/90-day horizons |
| Deterministic scripts (seed 42 throughout) | Sensitivity analyses: alternative L1 threshold, L2 percentile, S4 trigger thresholds |
