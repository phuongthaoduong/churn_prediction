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
