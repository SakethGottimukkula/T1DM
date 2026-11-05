"""Tests for evaluation metrics calculation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.metrics import summarize


def test_summarize_sample():
    """Test summarize() on hand-checked sample inputs."""
    refs = [100, 100, 100, 250, 200]
    preds = [100, 150, 210, 100, 60]
    res = summarize(refs, preds)

    assert res["n_rows"] == 5
    assert res["rmse"] == 106.49
    assert res["mae"] == 90.0
    assert res["zone_counts"] == {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1}
    assert res["zone_percentages"] == {
        "A": 20.0,
        "B": 20.0,
        "C": 20.0,
        "D": 20.0,
        "E": 20.0,
    }
    assert res["a_plus_b_pct"] == 40.0


def test_summarize_empty():
    """Test summarize() on empty inputs returns zeros without raising."""
    res = summarize([], [])

    assert res["n_rows"] == 0
    assert res["rmse"] == 0.0
    assert res["mae"] == 0.0
    assert res["zone_counts"] == {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    assert res["zone_percentages"] == {
        "A": 0.0,
        "B": 0.0,
        "C": 0.0,
        "D": 0.0,
        "E": 0.0,
    }
    assert res["a_plus_b_pct"] == 0.0
