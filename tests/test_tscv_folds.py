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
