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
