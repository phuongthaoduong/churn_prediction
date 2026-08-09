# Step 7: Prepare for Experimental Execution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce everything the 2×4 factorial experiment needs before training begins: a leakage-safe train/evaluation split, initial XGBoost models per label scheme with feature-importance rankings, frozen PSI drift-monitoring baselines, and a consolidated preprocessing decision log (Data_Preprocessing_Plan.md §7.1–7.3).

**Architecture:** Three new pipeline scripts (`step7_1`, `step7_2`, `step7_3`) following the existing `pipeline/` conventions (config constants at top, outputs to `profiling_output/`), plus two shared modules that the Phase-2 experiment will reuse: `experiment_config.py` (split constants + label-maturity logic) and `drift_utils.py` (binning + PSI). Pure logic is TDD'd with pytest; scripts are verified by running them against the real data.

**Tech Stack:** Python 3.13 (system, `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`), pandas 2.2.3, pyarrow 23.0.1, scikit-learn 1.8.0 (already installed); xgboost + pytest (installed in Task 1).

---

## Context an engineer needs (read this first)

**The project:** Thesis experiment comparing 2 churn-label schemes × 4 model-update strategies on Kingdee SME retail data. Steps 1–6 are DONE: `profiling_output/dataset_final.parquet` holds 6,340,667 rows × 34 cols — one row per customer per monthly batch, 20 prediction points from 2024-08-01 to 2026-03-01, 25 features, two labels (`label_L1`, `label_L2`), plus `L2_tier`, `L2_threshold_days`, `category`.

**Key input files (all exist):**

| File | Contents |
|---|---|
| `profiling_output/dataset_final.parquet` | features + labels, keys `server_id, shop_id, company_id, prediction_point` |
| `profiling_output/batch_metadata.csv` | per batch: `prediction_point, obs_start, obs_end, l1_label_end, max_label_end, eligible_customers, …` |
| `profiling_output/l2_thresholds.csv` | per batch × industry category: `prediction_point, category, tau_c, tau_raw, n_ipis, tier` |
| `profiling_output/class_balance.csv` | per batch × scheme: `n_total, n_churn, churn_rate, scale_pos_weight` |
| `profiling_output/exclusion_log_step2_*.csv`, `exclusion_log_test_accounts.csv` | cleaning audit trails |

**The 25 feature columns** (everything in `dataset_final` except the 4 key columns, 2 labels, `L2_tier`, `L2_threshold_days`, `category`):
`recency, frequency, monetary, avg_monetary, ipi_mean, ipi_std, recent_activity, freq_decline_ratio, unique_products, avg_basket_size, total_qty, pay_ratio, debt_frequency, customer_tenure, store_txn_volume, store_active_customers, store_median_ipi, recency_vs_store, frequency_vs_store, monetary_vs_store, month, quarter, is_cny_season, is_double11, is_spring_festival`

**Design decisions locked in by this plan** (rationale documented in the outputs):

1. **Split:** first 8 batches (2024-08-01 … 2025-03-01) = initial training period; last 12 batches (2025-04-01 … 2026-03-01) = evaluation. The thesis specifies "approximately one year of evaluation batches"; 12 monthly batches is exactly that.
2. **Label maturity rule (no leakage):** at any (re)training time `t`, a batch is usable only if its labels are fully observable: `l1_label_end ≤ t` for L1; `prediction_point + max(tau_c at that prediction point) ≤ t` for L2 (max over categories, from `l2_thresholds.csv`, capped at 90 days upstream). This means the L2 initial model trains on fewer batches than L1 — that is correct behavior under the prequential protocol (labels are revealed only after the label window closes), not a bug.
3. **Initial models are provisional:** trained with documented default hyperparameters solely to derive feature-importance rankings for drift monitoring. Formal hyperparameter tuning (time-series CV) happens in Phase 2; rerunning `step7_2` with tuned params regenerates the monitored-feature list.
4. **Drift monitoring:** top 10 features by XGBoost gain importance per scheme, **excluding the 5 deterministic calendar features** (`month, quarter, is_cny_season, is_double11, is_spring_festival` change every batch by construction and would always "drift"). 10 equal-frequency bins with a dedicated missing-value bin (bin 0) — necessary because `ipi_mean` is 47.5% NaN and `ipi_std` 63.5% NaN. PSI threshold 0.2, ε = 1e-6 (thesis eq. 3.9). Reference batch = last matured training batch per scheme.
5. **S3 rolling window = 6 months**, covering the P95 of per-customer median inter-purchase intervals (~180 days from Step 1 profiling); recorded in config, flagged for sensitivity analysis.

**Environment notes:**
- The project is NOT yet a git repository — Task 1 initializes it (the preprocessing plan requires version control for reproducibility). `.gitignore` must exclude `sale_data/` (7.2 GB) and `profiling_output/` (GBs of parquet).
- There is a stray empty file `training` (0 bytes) at the repo root — ignore it in git; do not delete it.
- If `import xgboost` fails on macOS with a `libomp.dylib` error, run `brew install libomp` and retry.
- Run all pipeline scripts from the `pipeline/` directory (existing scripts resolve paths via `os.path.dirname(os.path.dirname(__file__))`, which works from anywhere, but tests assume `pytest` is run from the project root).

---

### Task 1: Version control + dependencies

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`

- [x] **Step 1: Initialize git and write .gitignore**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing && git init
```

Create `.gitignore`:

```gitignore
# Raw data (7.2 GB) and generated outputs — never commit
sale_data/
profiling_output/
experiment_output/

# OS / editor
.DS_Store
__pycache__/
*.pyc
.pytest_cache/

# Stray placeholder file at repo root
/training
```

- [x] **Step 2: Write requirements.txt**

```text
pandas>=2.2
pyarrow>=15
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
pytest>=8.0
```

- [x] **Step 3: Install xgboost and pytest**

Run:
```bash
python3 -m pip install xgboost pytest 2>&1 | tail -3
```

Verify:
```bash
python3 -c "import xgboost, pytest; print('xgboost', xgboost.__version__, '| pytest', pytest.__version__)"
```
Expected: version numbers print without ImportError. If `libomp.dylib` load error: `brew install libomp`, retry.

- [x] **Step 4: Initial commit of the existing project**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add .gitignore requirements.txt pipeline/ docs/ utils/ thesis/
git commit -m "chore: initialize repo with existing steps 1-6 pipeline and docs"
```

Expected: commit succeeds; `git status` shows no staged changes; `sale_data/` and `profiling_output/` untracked-but-ignored.

---

### Task 2: `experiment_config.py` — split constants + label-maturity logic (TDD)

This module is the single source of truth for the experiment. Phase-2 strategy code (S1–S4) will import `usable_batches` at every retraining event.

**Files:**
- Create: `pipeline/experiment_config.py`
- Create: `tests/conftest.py`
- Test: `tests/test_experiment_config.py`

- [x] **Step 1: Write conftest.py so tests can import pipeline modules**

`tests/conftest.py`:
```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
```

- [x] **Step 2: Write the failing tests**

`tests/test_experiment_config.py`:
```python
import pandas as pd
import pytest

from experiment_config import (
    FEATURE_COLS,
    FIRST_EVAL_PP,
    MONITOR_EXCLUDE,
    label_close_dates,
    usable_batches,
)


@pytest.fixture
def batch_meta():
    return pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-09-01", "2024-10-01"],
        "l1_label_end": ["2024-08-31", "2024-10-01", "2024-10-31"],
    })


@pytest.fixture
def l2_thresholds():
    # max tau per pp: Aug=40, Sep=44, Oct=90
    return pd.DataFrame({
        "prediction_point": ["2024-08-01", "2024-08-01", "2024-09-01", "2024-10-01"],
        "category": ["A", "B", "A", "A"],
        "tau_c": [27.0, 40.0, 44.0, 90.0],
    })


def test_feature_cols_has_25_features():
    assert len(FEATURE_COLS) == 25
    assert "recency" in FEATURE_COLS
    assert "label_L1" not in FEATURE_COLS
    assert "category" not in FEATURE_COLS


def test_monitor_exclude_is_calendar_features():
    assert set(MONITOR_EXCLUDE) == {
        "month", "quarter", "is_cny_season", "is_double11", "is_spring_festival"
    }


def test_first_eval_pp():
    assert FIRST_EVAL_PP == pd.Timestamp("2025-04-01")


def test_label_close_dates(batch_meta, l2_thresholds):
    close = label_close_dates(batch_meta, l2_thresholds)
    assert list(close.columns) == ["prediction_point", "l1_label_close", "l2_label_close"]
    aug = close[close["prediction_point"] == pd.Timestamp("2024-08-01")].iloc[0]
    assert aug["l1_label_close"] == pd.Timestamp("2024-08-31")
    # L2 close = pp + max tau over categories = 2024-08-01 + 40 days
    assert aug["l2_label_close"] == pd.Timestamp("2024-09-10")


def test_usable_batches_l1(batch_meta, l2_thresholds):
    close = label_close_dates(batch_meta, l2_thresholds)
    # at 2024-10-01: Aug closed (08-31), Sep closes exactly 10-01 (inclusive), Oct not
    pps = usable_batches("2024-10-01", "L1", close)
    assert pps == [pd.Timestamp("2024-08-01"), pd.Timestamp("2024-09-01")]


def test_usable_batches_l2_is_stricter(batch_meta, l2_thresholds):
    close = label_close_dates(batch_meta, l2_thresholds)
    # at 2024-10-01: Aug L2 closes 09-10 (usable); Sep closes 10-15 (not yet)
    pps = usable_batches("2024-10-01", "L2", close)
    assert pps == [pd.Timestamp("2024-08-01")]


def test_usable_batches_rejects_unknown_scheme(batch_meta, l2_thresholds):
    close = label_close_dates(batch_meta, l2_thresholds)
    with pytest.raises(ValueError):
        usable_batches("2024-10-01", "L3", close)
```

- [x] **Step 3: Run tests to verify they fail**

Run: `cd /Users/phuongthaoduong/Project/Data_Processing && python3 -m pytest tests/test_experiment_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiment_config'`

- [x] **Step 4: Write the implementation**

`pipeline/experiment_config.py`:
```python
"""
Shared experiment configuration (Step 7 + Phase-2 factorial experiment)
========================================================================
Single source of truth for the train/evaluation split, feature list,
drift-monitoring parameters, and the label-maturity rule that prevents
label leakage at every (re)training event.
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "profiling_output")
DOCS_DIR = os.path.join(PROJECT_DIR, "docs")

DATASET_FINAL = os.path.join(OUTPUT_DIR, "dataset_final.parquet")
BATCH_METADATA = os.path.join(OUTPUT_DIR, "batch_metadata.csv")
L2_THRESHOLDS_PATH = os.path.join(OUTPUT_DIR, "l2_thresholds.csv")
CLASS_BALANCE = os.path.join(OUTPUT_DIR, "class_balance.csv")

# ---------------------------------------------------------------------------
# Temporal split
# 20 batches: 2024-08-01 .. 2026-03-01.
# Initial training period: first 8 batches (2024-08-01 .. 2025-03-01).
# Evaluation: last 12 batches (2025-04-01 .. 2026-03-01) — one year of
# monthly evaluation batches, per the methodology (chap03 §3.2.2).
# ---------------------------------------------------------------------------
FIRST_EVAL_PP = pd.Timestamp("2025-04-01")

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
ROLLING_WINDOW_MONTHS = 6      # S3: covers P95 per-customer median IPI (~180d, Step 1 profiling)
PSI_THRESHOLD = 0.2            # S4 feature-drift trigger (Siddiqi 2006; thesis eq. 3.9)
AUC_DECLINE_THRESHOLD = 0.02   # S4 performance-decline trigger; sensitivity-tested in Phase 2
N_PSI_BINS = 10                # equal-frequency bins for PSI
PSI_EPSILON = 1e-6             # smoothing constant for empty bins (thesis eq. 3.9)
N_MONITORED_FEATURES = 10      # top-K features by gain importance to monitor
RECALL_AT_K_PCT = 0.10         # Recall@10% — SME intervention capacity

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
KEY_COLS = ["server_id", "shop_id", "company_id", "prediction_point"]

FEATURE_COLS = [
    # customer-level (14)
    "recency", "frequency", "monetary", "avg_monetary", "ipi_mean", "ipi_std",
    "recent_activity", "freq_decline_ratio", "unique_products", "avg_basket_size",
    "total_qty", "pay_ratio", "debt_frequency", "customer_tenure",
    # store-level (6)
    "store_txn_volume", "store_active_customers", "store_median_ipi",
    "recency_vs_store", "frequency_vs_store", "monetary_vs_store",
    # time-level (5)
    "month", "quarter", "is_cny_season", "is_double11", "is_spring_festival",
]

# Calendar features change deterministically with the prediction point, so they
# would register permanent "drift" — excluded from PSI monitoring.
MONITOR_EXCLUDE = ["month", "quarter", "is_cny_season", "is_double11", "is_spring_festival"]

# ---------------------------------------------------------------------------
# Provisional XGBoost hyperparameters.
# Used by step7_2 to derive feature-importance rankings for drift monitoring.
# Phase 2 tunes hyperparameters once via time-series CV on the initial
# training period, then freezes them; rerun step7_2 afterwards to regenerate
# the monitored-feature list under the tuned model.
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "objective": "binary:logistic",
    "tree_method": "hist",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


def label_close_dates(batch_meta: pd.DataFrame, l2_thresholds: pd.DataFrame) -> pd.DataFrame:
    """Per batch, the date at which each scheme's labels are fully observable.

    L1 closes at l1_label_end (prediction_point + 30 days).
    L2 closes at prediction_point + max(tau_c) over categories at that
    prediction point (tau_c is already capped at 90 days upstream).
    """
    out = pd.DataFrame({
        "prediction_point": pd.to_datetime(batch_meta["prediction_point"]),
        "l1_label_close": pd.to_datetime(batch_meta["l1_label_end"]),
    })
    tau = l2_thresholds.copy()
    tau["prediction_point"] = pd.to_datetime(tau["prediction_point"])
    max_tau = tau.groupby("prediction_point")["tau_c"].max()
    out["l2_label_close"] = out["prediction_point"] + pd.to_timedelta(
        out["prediction_point"].map(max_tau), unit="D"
    )
    return out


def usable_batches(t, scheme: str, close_dates: pd.DataFrame) -> list:
    """Prediction points whose labels are fully observable at time t.

    This is the no-leakage rule for every (re)training event: a model
    trained at time t may only use batches returned by this function.
    """
    if scheme == "L1":
        col = "l1_label_close"
    elif scheme == "L2":
        col = "l2_label_close"
    else:
        raise ValueError(f"Unknown scheme: {scheme!r} (expected 'L1' or 'L2')")
    ok = close_dates[close_dates[col] <= pd.Timestamp(t)]
    return sorted(ok["prediction_point"].tolist())
```

- [x] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_experiment_config.py -v`
Expected: 7 passed

- [x] **Step 6: Commit**

```bash
git add pipeline/experiment_config.py tests/conftest.py tests/test_experiment_config.py
git commit -m "feat: add experiment config with split constants and label-maturity rule"
```

---

### Task 3: `drift_utils.py` — equal-frequency binning + PSI (TDD)

**Files:**
- Create: `pipeline/drift_utils.py`
- Test: `tests/test_drift_utils.py`

- [x] **Step 1: Write the failing tests**

`tests/test_drift_utils.py`:
```python
import numpy as np
import pytest

from drift_utils import bin_proportions, compute_psi, make_bins


def test_make_bins_returns_interior_edges():
    values = np.arange(100, dtype=float)  # 0..99
    edges = make_bins(values, n_bins=10)
    assert len(edges) == 9               # 9 interior edges -> 10 bins
    assert edges[0] == pytest.approx(9.9)
    assert np.all(np.diff(edges) > 0)


def test_make_bins_collapses_duplicate_edges():
    # constant feature -> quantiles identical -> a single unique edge
    edges = make_bins(np.ones(50), n_bins=10)
    assert len(edges) == 1


def test_make_bins_ignores_nan():
    values = np.concatenate([np.arange(100, dtype=float), [np.nan] * 100])
    edges = make_bins(values, n_bins=10)
    assert not np.any(np.isnan(edges))
    assert len(edges) == 9


def test_bin_proportions_sum_to_one_and_bin0_is_missing():
    values = np.array([np.nan, np.nan, 1.0, 5.0, 9.0])
    edges = np.array([3.0, 7.0])         # 3 value bins + missing bin
    props = bin_proportions(values, edges)
    assert len(props) == 4               # bin0 missing + 3 value bins
    assert props.sum() == pytest.approx(1.0)
    assert props[0] == pytest.approx(2 / 5)   # 2 NaN of 5
    assert props[1] == pytest.approx(1 / 5)   # 1.0  <= 3
    assert props[2] == pytest.approx(1 / 5)   # 5.0 in (3, 7]
    assert props[3] == pytest.approx(1 / 5)   # 9.0  > 7


def test_psi_identical_distributions_is_zero():
    props = np.array([0.0, 0.25, 0.25, 0.5])
    assert compute_psi(props, props) == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_shift():
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, 10_000)
    cur = rng.normal(1.0, 1, 10_000)     # mean shift of 1 sd
    edges = make_bins(ref, n_bins=10)
    psi = compute_psi(bin_proportions(ref, edges), bin_proportions(cur, edges))
    assert psi > 0.2                     # clearly above the drift threshold


def test_psi_empty_bin_is_finite():
    ref = np.array([0.5, 0.5, 0.0])
    cur = np.array([0.0, 0.5, 0.5])
    psi = compute_psi(ref, cur, epsilon=1e-6)
    assert np.isfinite(psi)
    assert psi > 0
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_drift_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drift_utils'`

- [x] **Step 3: Write the implementation**

`pipeline/drift_utils.py`:
```python
"""
Drift-monitoring utilities: equal-frequency binning and PSI.
=============================================================
PSI follows thesis eq. 3.9: proportions are smoothed with a small epsilon
so empty bins never produce undefined logarithms.

Binning convention: bin 0 is a dedicated missing-value bin (ipi_mean is
47.5% NaN in the feature matrix, ipi_std 63.5% — missingness itself can
drift). Bins 1..len(edges)+1 partition the real line at the interior edges.
"""

import numpy as np


def make_bins(reference, n_bins: int = 10) -> np.ndarray:
    """Interior equal-frequency bin edges from reference values (NaN ignored).

    Returns up to n_bins - 1 strictly increasing edges; duplicate quantiles
    (low-cardinality features) are collapsed, so fewer edges may be returned.
    """
    reference = np.asarray(reference, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.nanquantile(reference, qs))


def bin_proportions(values, edges) -> np.ndarray:
    """Proportion of values in each bin. Index 0 = missing (NaN);
    index b (1-based) = the b-th interval defined by the interior edges,
    right-inclusive. Proportions sum to 1 over all values."""
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    n = len(values)
    missing = np.isnan(values)
    # right=True: value <= edge falls in the lower bin, matching quantile edges
    idx = np.digitize(values[~missing], edges, right=True) + 1
    counts = np.bincount(idx, minlength=len(edges) + 2).astype(float)
    counts[0] = missing.sum()
    return counts / n


def compute_psi(ref_props, cur_props, epsilon: float = 1e-6) -> float:
    """Population Stability Index between two binned distributions."""
    ref = np.asarray(ref_props, dtype=float) + epsilon
    cur = np.asarray(cur_props, dtype=float) + epsilon
    return float(np.sum((cur - ref) * np.log(cur / ref)))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_drift_utils.py -v`
Expected: 7 passed

- [x] **Step 5: Run the full test suite**

Run: `python3 -m pytest tests/ -v`
Expected: 14 passed

- [x] **Step 6: Commit**

```bash
git add pipeline/drift_utils.py tests/test_drift_utils.py
git commit -m "feat: add PSI and equal-frequency binning utilities with missing-value bin"
```

---

### Task 4: `step7_1_train_eval_split.py` — train/evaluation split (§7.1)

**Files:**
- Create: `pipeline/step7_1_train_eval_split.py`
- Output: `profiling_output/train_eval_split.csv`, `profiling_output/experiment_config.json`, `profiling_output/step7_1_split_summary.txt`

- [x] **Step 1: Write the script**

`pipeline/step7_1_train_eval_split.py`:
```python
"""
Step 7.1: Define Train/Evaluation Split
========================================
Assigns each of the 20 monthly batches a role:
  - initial_train: batches before FIRST_EVAL_PP (2025-04-01) — reserved for
    initial model training and Phase-2 hyperparameter tuning
  - evaluation:    2025-04-01 .. 2026-03-01 — 12 monthly prequential batches

Also records, per batch, the date at which each label scheme becomes fully
observable (label-maturity rule), and snapshots all experiment constants to
experiment_config.json for reproducibility.

Input:
  - profiling_output/batch_metadata.csv
  - profiling_output/l2_thresholds.csv
  - profiling_output/dataset_final.parquet   (row counts per role)

Output:
  - profiling_output/train_eval_split.csv
  - profiling_output/experiment_config.json
  - profiling_output/step7_1_split_summary.txt
"""

import json
import os
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq

from experiment_config import (
    AUC_DECLINE_THRESHOLD,
    BATCH_METADATA,
    CLASS_BALANCE,
    DATASET_FINAL,
    FEATURE_COLS,
    FIRST_EVAL_PP,
    L2_THRESHOLDS_PATH,
    MONITOR_EXCLUDE,
    N_MONITORED_FEATURES,
    N_PSI_BINS,
    OUTPUT_DIR,
    PSI_EPSILON,
    PSI_THRESHOLD,
    RECALL_AT_K_PCT,
    ROLLING_WINDOW_MONTHS,
    XGB_PARAMS,
    label_close_dates,
    usable_batches,
)


def main():
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)

    close = label_close_dates(batch_meta, l2_thresholds)
    close["role"] = close["prediction_point"].apply(
        lambda pp: "initial_train" if pp < FIRST_EVAL_PP else "evaluation"
    )
    close["eligible_customers"] = pd.to_datetime(
        batch_meta["prediction_point"]
    ).map(
        batch_meta.assign(pp=pd.to_datetime(batch_meta["prediction_point"]))
        .set_index("pp")["eligible_customers"]
    ).values

    # Row counts per role from the final dataset (reads one column only)
    pps = pq.read_table(DATASET_FINAL, columns=["prediction_point"]).to_pandas()
    counts = pps["prediction_point"].value_counts()
    close["n_rows"] = close["prediction_point"].map(counts).astype("Int64")

    split_path = os.path.join(OUTPUT_DIR, "train_eval_split.csv")
    close.to_csv(split_path, index=False)

    # Batches usable for the initial models (trained at FIRST_EVAL_PP)
    usable = {s: usable_batches(FIRST_EVAL_PP, s, close) for s in ("L1", "L2")}

    config = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "first_eval_pp": str(FIRST_EVAL_PP.date()),
        "n_batches_total": len(close),
        "n_batches_initial_train": int((close["role"] == "initial_train").sum()),
        "n_batches_evaluation": int((close["role"] == "evaluation").sum()),
        "initial_train_usable_L1": [str(p.date()) for p in usable["L1"]],
        "initial_train_usable_L2": [str(p.date()) for p in usable["L2"]],
        "rolling_window_months_S3": ROLLING_WINDOW_MONTHS,
        "psi_threshold_S4": PSI_THRESHOLD,
        "auc_decline_threshold_S4": AUC_DECLINE_THRESHOLD,
        "n_psi_bins": N_PSI_BINS,
        "psi_epsilon": PSI_EPSILON,
        "n_monitored_features": N_MONITORED_FEATURES,
        "monitor_exclude": MONITOR_EXCLUDE,
        "recall_at_k_pct": RECALL_AT_K_PCT,
        "feature_cols": FEATURE_COLS,
        "xgb_params_provisional": XGB_PARAMS,
        "class_balance_source": os.path.basename(CLASS_BALANCE),
    }
    config_path = os.path.join(OUTPUT_DIR, "experiment_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    lines = []
    w = lines.append
    w("=" * 70)
    w("STEP 7.1: TRAIN/EVALUATION SPLIT")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)
    w("")
    w(f"Total batches:            {len(close)}")
    w(f"Initial training period:  {config['n_batches_initial_train']} batches "
      f"({close['prediction_point'].min():%Y-%m-%d} .. "
      f"{close.loc[close['role'] == 'initial_train', 'prediction_point'].max():%Y-%m-%d})")
    w(f"Evaluation period:        {config['n_batches_evaluation']} batches "
      f"({FIRST_EVAL_PP:%Y-%m-%d} .. "
      f"{close['prediction_point'].max():%Y-%m-%d})")
    w("")
    w("LABEL MATURITY AT FIRST EVALUATION POINT")
    w("-" * 70)
    w(f"Batches usable for initial L1 model: {len(usable['L1'])} "
      f"(labels close 30 days after prediction point)")
    w(f"Batches usable for initial L2 model: {len(usable['L2'])} "
      f"(labels close up to max category tau, capped 90 days)")
    w("L2 trains on fewer matured batches than L1 by design: under the")
    w("prequential protocol, labels are revealed only after the label window")
    w("closes, and L2 windows are longer.")
    w("")
    w("PER-BATCH DETAIL")
    w("-" * 70)
    w(f"{'prediction_point':<18}{'role':<15}{'L1 close':<13}{'L2 close':<13}{'rows':>10}")
    for _, r in close.iterrows():
        w(f"{r['prediction_point']:%Y-%m-%d}        {r['role']:<15}"
          f"{r['l1_label_close']:%Y-%m-%d}   {r['l2_label_close']:%Y-%m-%d}   "
          f"{r['n_rows']:>8,}")
    w("")
    w("OUTPUT FILES")
    w("-" * 70)
    w(f"  {os.path.basename(split_path)}")
    w(f"  {os.path.basename(config_path)}")
    w("=" * 70)

    report = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "step7_1_split_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the script**

Run:
```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step7_1_train_eval_split.py
```
Expected output includes:
- `Total batches: 20`, `Initial training period: 8 batches (2024-08-01 .. 2025-03-01)`, `Evaluation period: 12 batches (2025-04-01 .. 2026-03-01)`
- Usable batches for L1 = 8; for L2 a smaller number (expected 6–7, depending on max tau per batch — verify each listed L2 close date ≤ 2025-04-01)
- Per-batch rows matching `class_balance.csv` `n_total` values (~290k–328k per batch)

- [x] **Step 3: Sanity-check the outputs**

Run:
```bash
python3 -c "
import json, pandas as pd
split = pd.read_csv('../profiling_output/train_eval_split.csv', parse_dates=['prediction_point','l1_label_close','l2_label_close'])
assert len(split) == 20
assert (split['role'] == 'initial_train').sum() == 8
assert (split['role'] == 'evaluation').sum() == 12
assert (split['l2_label_close'] >= split['l1_label_close']).all()
assert split['n_rows'].sum() == 6_340_667
cfg = json.load(open('../profiling_output/experiment_config.json'))
assert len(cfg['feature_cols']) == 25
assert all(pd.Timestamp(p) < pd.Timestamp('2025-04-01') or pd.Timestamp(p) == pd.Timestamp('2025-04-01') for p in cfg['initial_train_usable_L1'])
print('step7_1 outputs OK')
"
```
Expected: `step7_1 outputs OK`

- [x] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step7_1_train_eval_split.py
git commit -m "feat: add step 7.1 train/eval split with label-maturity accounting"
```

---

### Task 5: `step7_2_drift_baselines.py` — initial models + PSI baselines (§7.2)

**Files:**
- Create: `pipeline/step7_2_drift_baselines.py`
- Output: `profiling_output/initial_model_L1.json`, `initial_model_L2.json`, `feature_importance.csv`, `psi_reference.json`, `psi_batch_distributions.csv`, `step7_2_drift_summary.txt`

- [x] **Step 1: Write the script**

`pipeline/step7_2_drift_baselines.py`:
```python
"""
Step 7.2: Precompute Drift Monitoring Inputs
=============================================
For each label scheme (L1, L2):
  1. Train a provisional initial XGBoost model on all label-matured batches
     at the first evaluation point (leakage-safe via usable_batches).
  2. Rank features by gain importance; select top-K behavioral features to
     monitor (calendar features excluded — they change every batch by
     construction).
  3. Freeze equal-frequency bin edges on the reference batch (last matured
     training batch) and store reference bin proportions.
  4. Precompute bin proportions for every batch under the frozen edges, so
     the Phase-2 experiment can evaluate PSI between any reference/current
     pair as a lookup + sum.

NOTE: hyperparameters are provisional defaults (experiment_config.XGB_PARAMS).
After Phase-2 tuning, rerun this script to regenerate importances/baselines.

Input:
  - profiling_output/dataset_final.parquet
  - profiling_output/batch_metadata.csv
  - profiling_output/l2_thresholds.csv

Output:
  - profiling_output/initial_model_L1.json / initial_model_L2.json
  - profiling_output/feature_importance.csv
  - profiling_output/psi_reference.json
  - profiling_output/psi_batch_distributions.csv
  - profiling_output/step7_2_drift_summary.txt
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from drift_utils import bin_proportions, compute_psi, make_bins
from experiment_config import (
    BATCH_METADATA,
    DATASET_FINAL,
    FEATURE_COLS,
    FIRST_EVAL_PP,
    L2_THRESHOLDS_PATH,
    MONITOR_EXCLUDE,
    N_MONITORED_FEATURES,
    N_PSI_BINS,
    OUTPUT_DIR,
    PSI_EPSILON,
    XGB_PARAMS,
    label_close_dates,
    usable_batches,
)

SCHEMES = ("L1", "L2")


def main():
    df = pd.read_parquet(DATASET_FINAL)
    batch_meta = pd.read_csv(BATCH_METADATA)
    l2_thresholds = pd.read_csv(L2_THRESHOLDS_PATH)
    close = label_close_dates(batch_meta, l2_thresholds)
    all_pps = sorted(df["prediction_point"].unique())

    importance_rows = []
    dist_rows = []
    psi_reference = {}
    summary = []
    w = summary.append
    w("=" * 70)
    w("STEP 7.2: DRIFT MONITORING BASELINES")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    w("=" * 70)

    for scheme in SCHEMES:
        train_pps = usable_batches(FIRST_EVAL_PP, scheme, close)
        train = df[df["prediction_point"].isin(train_pps)]
        X = train[FEATURE_COLS]
        y = train[f"label_{scheme}"]
        spw = float((y == 0).sum() / (y == 1).sum())

        model = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model.fit(X, y)
        train_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

        model_path = os.path.join(OUTPUT_DIR, f"initial_model_{scheme}.json")
        model.save_model(model_path)

        gains = model.get_booster().get_score(importance_type="gain")
        imp = (
            pd.DataFrame(
                {"feature": FEATURE_COLS,
                 "gain": [gains.get(f, 0.0) for f in FEATURE_COLS]}
            )
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )
        imp["rank"] = imp.index + 1
        imp["scheme"] = scheme
        importance_rows.append(imp)

        monitored = (
            imp[~imp["feature"].isin(MONITOR_EXCLUDE)]
            .head(N_MONITORED_FEATURES)["feature"]
            .tolist()
        )

        ref_pp = max(train_pps)
        ref = df[df["prediction_point"] == ref_pp]
        feat_info = {}
        for feat in monitored:
            edges = make_bins(ref[feat].to_numpy(dtype=float), N_PSI_BINS)
            props = bin_proportions(ref[feat].to_numpy(dtype=float), edges)
            feat_info[feat] = {
                "edges": edges.tolist(),
                "reference_proportions": props.tolist(),
            }
        psi_reference[scheme] = {
            "reference_pp": f"{pd.Timestamp(ref_pp):%Y-%m-%d}",
            "train_pps": [f"{pd.Timestamp(p):%Y-%m-%d}" for p in train_pps],
            "scale_pos_weight": spw,
            "train_auc_in_sample": float(train_auc),
            "monitored_features": monitored,
            "features": feat_info,
        }

        # Bin proportions for every batch under the frozen edges + PSI vs ref
        psi_vs_ref = {feat: {} for feat in monitored}
        for pp in all_pps:
            cur = df[df["prediction_point"] == pp]
            for feat in monitored:
                edges = np.array(feat_info[feat]["edges"])
                props = bin_proportions(cur[feat].to_numpy(dtype=float), edges)
                psi_vs_ref[feat][pp] = compute_psi(
                    feat_info[feat]["reference_proportions"], props, PSI_EPSILON
                )
                for b, p in enumerate(props):
                    dist_rows.append({
                        "scheme": scheme,
                        "prediction_point": f"{pd.Timestamp(pp):%Y-%m-%d}",
                        "feature": feat,
                        "bin": b,
                        "proportion": p,
                    })

        w("")
        w(f"SCHEME {scheme}")
        w("-" * 70)
        w(f"  Training batches (matured at {FIRST_EVAL_PP:%Y-%m-%d}): "
          f"{len(train_pps)} ({psi_reference[scheme]['train_pps'][0]} .. "
          f"{psi_reference[scheme]['train_pps'][-1]})")
        w(f"  Training rows:       {len(train):,}")
        w(f"  scale_pos_weight:    {spw:.4f}")
        w(f"  In-sample AUC:       {train_auc:.4f}  (provisional hyperparameters)")
        w(f"  Reference batch:     {psi_reference[scheme]['reference_pp']}")
        w(f"  Monitored features:  {', '.join(monitored)}")
        w("")
        w("  Top 10 features by gain:")
        for _, r in imp.head(10).iterrows():
            w(f"    {r['rank']:>2}. {r['feature']:<25} {r['gain']:>12.2f}")
        w("")
        w(f"  PSI vs reference per evaluation batch (threshold 0.2), "
          f"max over monitored features:")
        for pp in all_pps:
            if pd.Timestamp(pp) < FIRST_EVAL_PP:
                continue
            worst = max(monitored, key=lambda f: psi_vs_ref[f][pp])
            w(f"    {pd.Timestamp(pp):%Y-%m-%d}  max PSI = "
              f"{psi_vs_ref[worst][pp]:.4f}  ({worst})")

    pd.concat(importance_rows).to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False
    )
    with open(os.path.join(OUTPUT_DIR, "psi_reference.json"), "w") as f:
        json.dump(psi_reference, f, indent=2)
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(OUTPUT_DIR, "psi_batch_distributions.csv"), index=False
    )

    w("")
    w("OUTPUT FILES")
    w("-" * 70)
    for name in ("initial_model_L1.json", "initial_model_L2.json",
                 "feature_importance.csv", "psi_reference.json",
                 "psi_batch_distributions.csv"):
        w(f"  {name}")
    w("=" * 70)

    report = "\n".join(summary)
    with open(os.path.join(OUTPUT_DIR, "step7_2_drift_summary.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the script**

Run:
```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step7_2_drift_baselines.py
```
Expected (takes a few minutes — trains two XGBoost models on ~1.9–2.5M rows each):
- L1 trains on 8 batches, L2 on fewer (matching step7_1's usable counts)
- In-sample AUC roughly 0.75–0.95 for both schemes (if ≈0.5 the labels/features are misaligned — stop and investigate; if ≈1.0 suspect leakage — stop and investigate)
- `recency` expected among the top features by gain
- PSI values mostly < 0.2 for near-in-time batches; values generally increasing with distance from the reference batch

- [x] **Step 3: Sanity-check the outputs**

Run:
```bash
python3 -c "
import json, pandas as pd
imp = pd.read_csv('../profiling_output/feature_importance.csv')
assert set(imp['scheme']) == {'L1', 'L2'}
assert len(imp) == 50  # 25 features x 2 schemes
ref = json.load(open('../profiling_output/psi_reference.json'))
for s in ('L1', 'L2'):
    assert len(ref[s]['monitored_features']) == 10
    assert not set(ref[s]['monitored_features']) & {'month','quarter','is_cny_season','is_double11','is_spring_festival'}
    for f, info in ref[s]['features'].items():
        assert abs(sum(info['reference_proportions']) - 1.0) < 1e-9
dist = pd.read_csv('../profiling_output/psi_batch_distributions.csv')
sums = dist.groupby(['scheme','prediction_point','feature'])['proportion'].sum()
assert (sums - 1.0).abs().max() < 1e-9
assert dist['prediction_point'].nunique() == 20
print('step7_2 outputs OK')
"
```
Expected: `step7_2 outputs OK`

- [x] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step7_2_drift_baselines.py
git commit -m "feat: add step 7.2 initial models, feature importances, and PSI baselines"
```

---

### Task 6: `step7_3_preprocessing_log.py` — consolidated decision log (§7.3)

**Files:**
- Create: `pipeline/step7_3_preprocessing_log.py`
- Output: `docs/preprocessing_log.md`

- [x] **Step 1: Write the script**

`pipeline/step7_3_preprocessing_log.py`:
```python
"""
Step 7.3: Consolidated Preprocessing Decision Log
==================================================
Assembles a single markdown document recording every preprocessing decision:
exclusion criteria and affected counts, window-length choices with profiling
justification, L2 tier proportions and threshold stability, class balance per
batch, the train/evaluation split, and drift-monitoring configuration.

This is the reproducibility record required by Data_Preprocessing_Plan.md
Step 7.3 and the thesis methodology chapter.

Input: all profiling_output artifacts from steps 2-7.2
Output: docs/preprocessing_log.md
"""

import json
import os
from datetime import datetime

import pandas as pd

from experiment_config import (
    CLASS_BALANCE,
    DATASET_FINAL,
    DOCS_DIR,
    FIRST_EVAL_PP,
    L2_THRESHOLDS_PATH,
    OUTPUT_DIR,
)

EXCLUSION_LOGS = [
    ("2.1 Extreme financial outliers", "exclusion_log_step2_1.csv"),
    ("2.2 System/test accounts", "exclusion_log_step2_2.csv"),
    ("2.3 Returns and corrections", "exclusion_log_step2_3.csv"),
    ("2.4 Orphan line items", "exclusion_log_step2_4.csv"),
]


def count_rows(path):
    """Row count of a possibly large CSV without loading it into memory."""
    with open(path) as f:
        return sum(1 for _ in f) - 1


def md_table(df, float_fmt="{:.4f}"):
    def fmt(v):
        if isinstance(v, float):
            return float_fmt.format(v)
        return str(v)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def main():
    lines = []
    w = lines.append

    w("# Preprocessing Decision Log")
    w("")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S} by "
      "`pipeline/step7_3_preprocessing_log.py`. Regenerate after any "
      "pipeline rerun — do not edit by hand.")
    w("")

    # --- 1. Data scope ----------------------------------------------------
    w("## 1. Data scope")
    w("")
    w("- Raw inputs: `srv_sales.csv` (2.1 GB), `srv_saleitems.csv` (5.1 GB), "
      "`industry_servereid.csv` (13 industry categories, 100% merchant coverage).")
    w("- Span: 2024-05-01 to 2026-05-31 (~25 months).")
    w("- Customer identity: composite key `(server_id, shop_id, company_id)` — "
      "ids are per-merchant auto-increments and are NOT globally unique.")
    w("")

    # --- 2. Exclusions ----------------------------------------------------
    w("## 2. Exclusion criteria and affected record counts (Step 2)")
    w("")
    rows = []
    for label, fname in EXCLUSION_LOGS:
        path = os.path.join(OUTPUT_DIR, fname)
        n = count_rows(path) if os.path.exists(path) else "MISSING"
        rows.append({"Cleaning step": label, "Log file": f"`{fname}`",
                     "Records excluded": f"{n:,}" if isinstance(n, int) else n})
    w(md_table(pd.DataFrame(rows)))
    w("")
    w("Per-step rationale is recorded in the corresponding "
      "`profiling_output/step2_*_summary.txt` / analysis reports.")
    w("")

    # --- 3. Window choices ------------------------------------------------
    w("## 3. Window-length choices and justification (Steps 3 & 5)")
    w("")
    w("| Parameter | Value | Justification |")
    w("|---|---|---|")
    w("| Observation window | 90 days | Methodology §4.2; covers P75 of "
      "per-customer median IPI (60d) with margin |")
    w("| L1 label window | 30 days | Fixed inactivity threshold per methodology |")
    w("| L2 label window | P95 of pooled category IPIs, capped at 90 days | "
      "Thesis eq. 3.6; cap preserves 12 evaluation batches |")
    w("| Batch cadence | Monthly (1st of month) | Matches SME review cycle |")
    w("| S3 rolling window | 6 months | Covers P95 per-customer median IPI "
      "(~180d, Step 1 profiling); sensitivity-tested in Phase 2 |")
    w("")

    # --- 4. Train/eval split ---------------------------------------------
    w("## 4. Train/evaluation split (Step 7.1)")
    w("")
    split = pd.read_csv(
        os.path.join(OUTPUT_DIR, "train_eval_split.csv"),
        parse_dates=["prediction_point", "l1_label_close", "l2_label_close"],
    )
    for col in ("prediction_point", "l1_label_close", "l2_label_close"):
        split[col] = split[col].dt.strftime("%Y-%m-%d")
    w(md_table(split))
    w("")
    cfg = json.load(open(os.path.join(OUTPUT_DIR, "experiment_config.json")))
    w(f"- First evaluation point: **{cfg['first_eval_pp']}** — "
      f"{cfg['n_batches_initial_train']} initial-training batches, "
      f"{cfg['n_batches_evaluation']} evaluation batches (one year).")
    w(f"- Label-maturity rule: batches usable for the initial L1 model: "
      f"{len(cfg['initial_train_usable_L1'])}; for L2: "
      f"{len(cfg['initial_train_usable_L2'])}. L2 labels close later "
      "(longer windows), so fewer batches are matured — enforced by "
      "`experiment_config.usable_batches` at every retraining event.")
    w("")

    # --- 5. L2 tiers and threshold stability ------------------------------
    w("## 5. L2 fallback tiers and threshold stability (Step 5)")
    w("")
    tiers = pd.read_parquet(DATASET_FINAL, columns=["prediction_point", "L2_tier"])
    tier_tbl = (
        tiers.groupby([tiers["prediction_point"].dt.strftime("%Y-%m-%d"), "L2_tier"])
        .size().unstack(fill_value=0)
    )
    tier_pct = (tier_tbl.div(tier_tbl.sum(axis=1), axis=0) * 100).round(2)
    tier_pct.index.name = "prediction_point"
    w("Tier share per batch (% of customers):")
    w("")
    w(md_table(tier_pct.reset_index()))
    w("")
    tau = pd.read_csv(L2_THRESHOLDS_PATH)
    stab = tau.groupby("category")["tau_c"].agg(["min", "max", "mean", "std"])
    stab["cv_pct"] = (stab["std"] / stab["mean"] * 100).round(1)
    stab = stab.round(1).reset_index()
    w("Category threshold (tau_c, days) stability across batches:")
    w("")
    w(md_table(stab))
    w("")

    # --- 6. Class balance -------------------------------------------------
    w("## 6. Class balance per batch per scheme (Step 6)")
    w("")
    bal = pd.read_csv(CLASS_BALANCE)
    w(md_table(bal))
    w("")
    w("`scale_pos_weight` is recomputed at each retraining from the pooled "
      "training data of that event (thesis eq. 3.8); the per-batch values "
      "above are diagnostics.")
    w("")

    # --- 7. Drift monitoring ----------------------------------------------
    w("## 7. Drift-monitoring configuration (Step 7.2)")
    w("")
    ref = json.load(open(os.path.join(OUTPUT_DIR, "psi_reference.json")))
    w(f"- PSI: {cfg['n_psi_bins']} equal-frequency bins + dedicated "
      f"missing-value bin, epsilon = {cfg['psi_epsilon']}, trigger threshold "
      f"= {cfg['psi_threshold_S4']}.")
    w(f"- AUC-decline trigger threshold = {cfg['auc_decline_threshold_S4']}.")
    w("- Calendar features excluded from monitoring (deterministic drift): "
      f"{', '.join(cfg['monitor_exclude'])}.")
    for s in ("L1", "L2"):
        w(f"- **{s}**: reference batch {ref[s]['reference_pp']}, in-sample AUC "
          f"{ref[s]['train_auc_in_sample']:.4f} (provisional hyperparameters), "
          f"monitored features: {', '.join(ref[s]['monitored_features'])}.")
    w("")

    # --- 8. Deviations ----------------------------------------------------
    w("## 8. Deviations from the original plan")
    w("")
    w("- **L2 tier design**: Data_Preprocessing_Plan.md §5.2 described a "
      "3-tier scheme (customer/store/category). The implemented scheme is the "
      "thesis's 2-tier category + global-fallback design (eq. 3.6–3.7), which "
      "pools IPIs at category level for statistical stability.")
    w("- **Zero bill_amt records retained** in Step 2.3 (only negative-amount "
      "returns removed); see `step2_3_returns_analysis.txt`.")
    w("- **Hyperparameters provisional**: initial models in Step 7.2 use "
      "documented defaults; Phase 2 tunes once via time-series CV on the "
      "initial training period, then rerun `step7_2_drift_baselines.py`.")
    w("")

    out_path = os.path.join(DOCS_DIR, "preprocessing_log.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the script**

Run:
```bash
cd /Users/phuongthaoduong/Project/Data_Processing/pipeline && python3 step7_3_preprocessing_log.py
```
Expected: `Wrote .../docs/preprocessing_log.md (...)`. Counting rows in the exclusion logs may take ~a minute if they are large.

- [x] **Step 3: Review the generated log**

Open `docs/preprocessing_log.md` and check:
- All four exclusion rows show numeric counts (no "MISSING")
- Tier table columns are the actual `L2_tier` values (expect `category` and possibly `global`), percentages per row sum to ~100
- Split table shows 20 rows; class-balance table shows 40 rows (20 batches × 2 schemes)

- [x] **Step 4: Commit**

```bash
cd /Users/phuongthaoduong/Project/Data_Processing
git add pipeline/step7_3_preprocessing_log.py docs/preprocessing_log.md
git commit -m "feat: add step 7.3 consolidated preprocessing decision log"
```

---

### Task 7: Final verification

- [x] **Step 1: Run the full test suite**

Run: `cd /Users/phuongthaoduong/Project/Data_Processing && python3 -m pytest tests/ -v`
Expected: 14 passed

- [x] **Step 2: Verify all Step 7 deliverables exist**

Run:
```bash
ls -lh profiling_output/train_eval_split.csv profiling_output/experiment_config.json \
  profiling_output/step7_1_split_summary.txt profiling_output/initial_model_L1.json \
  profiling_output/initial_model_L2.json profiling_output/feature_importance.csv \
  profiling_output/psi_reference.json profiling_output/psi_batch_distributions.csv \
  profiling_output/step7_2_drift_summary.txt docs/preprocessing_log.md
```
Expected: all 10 files listed, none zero-byte.

- [x] **Step 3: End-to-end leakage spot-check**

The critical invariant: no batch used to train an initial model has a label window extending past the first evaluation point.

Run:
```bash
python3 -c "
import json, pandas as pd
cfg = json.load(open('profiling_output/experiment_config.json'))
split = pd.read_csv('profiling_output/train_eval_split.csv', parse_dates=['prediction_point','l1_label_close','l2_label_close'])
first_eval = pd.Timestamp(cfg['first_eval_pp'])
for scheme, col in (('L1','l1_label_close'), ('L2','l2_label_close')):
    used = pd.to_datetime(cfg[f'initial_train_usable_{scheme}'])
    closes = split.set_index('prediction_point').loc[used, col]
    assert (closes <= first_eval).all(), f'{scheme} leakage!'
print('No label leakage in initial training sets')
"
```
Expected: `No label leakage in initial training sets`

- [x] **Step 4: Final commit**

```bash
git add -A
git status   # confirm only intended files are staged (no data files)
git commit -m "chore: complete step 7 experiment preparation" --allow-empty
```

---

## What Step 7 hands to the training phase

| Deliverable | Consumed by |
|---|---|
| `experiment_config.py` (`usable_batches`, constants) | All four strategies S1–S4 at every retraining event |
| `train_eval_split.csv` + `experiment_config.json` | Prequential loop: which batches to score vs train on |
| `initial_model_L1/L2.json` | S1 static baseline starting point; sanity reference |
| `feature_importance.csv`, `psi_reference.json` | S4 drift trigger: which features to watch, against what baseline |
| `psi_batch_distributions.csv` | Cheap PSI computation between any (reference, current) batch pair |
| `drift_utils.py` | S4 PSI recomputation when the reference resets after retraining |
| `docs/preprocessing_log.md` | Thesis chapter 3/4 writing; reproducibility record |

Out of scope for Step 7 (deliberately deferred to Phase 2): hyperparameter tuning via time-series CV, the LLM-guided-vs-RFM feature comparison, the S1–S4 strategy loop, and evaluation metrics (AUC/F1/Recall@10%, CV, Jaccard, Wilcoxon tests).

## Execution record

Executed 2026-08-09/10 via subagent-driven development. All tasks in this plan were carried out; every checkbox above is marked complete.

Deviations from the plan as written:

1. Task 4's `eligible_customers` join was simplified to `set_index().map()` on parsed dates, rather than the originally sketched merge approach.
2. Task 6's Section-2 exclusion table was corrected to report true removal counts: 2.2 excludes 26 accounts (160,947 transactions), parsed fail-fast from `test_account_detection_summary.txt`; 2.4 excludes 611,091 orphans, parsed fail-fast from `step2_4_orphan_summary.txt` (the exclusion log itself is only a 10,000-row sample and cannot be used to derive true counts).
3. Review-driven hardening commits were added beyond the original task list: `6b8aede` (fail loudly when L2 thresholds are missing for a batch), `d9763ed` (guard PSI against mismatched shapes and empty inputs), `db05c5e` (guard degenerate `scale_pos_weight` plus Phase-2 consumer notes), `ae59cb8` (match decision-log table precision to computed rounding).
4. Observed results worth carrying into the thesis narrative: L1 in-sample AUC 0.8544, L2 in-sample AUC 0.8580; the L2 global fallback threshold tier never fires in practice (100% of batches resolve at a more specific tier), which is worth calling out explicitly rather than treating as a silent no-op.
