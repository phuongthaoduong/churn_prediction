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
