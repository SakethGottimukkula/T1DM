"""Tests for Clarke Error Grid Analysis classification."""

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ega import classify_array, classify_point


class TestClassifyPoint:
    """Hand-checked Clarke EGA zone classifications."""

    # ---- Zone A ----

    def test_zone_a_both_below_70(self):
        """Both ref and pred ≤ 70 → A."""
        assert classify_point(60, 55) == "A"

    def test_zone_a_within_20pct(self):
        """pred within ±20% of ref → A."""
        assert classify_point(150, 140) == "A"
        assert classify_point(100, 115) == "A"

    def test_zone_a_exact_match(self):
        """Exact match → A."""
        assert classify_point(120, 120) == "A"

    # ---- Zone B ----

    def test_zone_b_above_20pct(self):
        """Outside 20% band but not in C/D/E → B."""
        assert classify_point(200, 155) == "B"

    def test_zone_b_moderate_error(self):
        assert classify_point(100, 75) == "B"

    # ---- Zone C ----

    def test_zone_c_upper(self):
        """ref in [70,290] and pred ≥ ref+110 → C."""
        assert classify_point(100, 215) == "C"

    def test_zone_c_lower(self):
        """ref in [130,180] and pred ≤ 7/5*ref - 182 → C."""
        # ref=150, threshold = 7/5*150 - 182 = 210 - 182 = 28
        assert classify_point(150, 25) == "C"

    # ---- Zone D ----

    def test_zone_d_high_ref_mid_pred(self):
        """ref ≥ 240 and 70 ≤ pred ≤ 180 → D."""
        assert classify_point(260, 150) == "D"

    def test_zone_d_low_ref_mid_pred(self):
        """ref ≤ 175/3 ≈ 58.33 and 70 ≤ pred ≤ 180 → D."""
        assert classify_point(50, 100) == "D"

    # ---- Zone E ----

    def test_zone_e_high_ref_low_pred(self):
        """ref ≥ 180 and pred ≤ 70 → E."""
        assert classify_point(200, 50) == "E"

    def test_zone_e_low_ref_high_pred(self):
        """ref ≤ 70 and pred ≥ 180 → E."""
        assert classify_point(60, 200) == "E"


class TestClassifyArray:
    """Test array-level classification and counts."""

    def test_basic_array(self):
        refs = [60, 150, 200]
        preds = [55, 140, 50]
        labels, counts = classify_array(refs, preds)
        assert labels[0] == "A"
        assert labels[1] == "A"
        assert labels[2] == "E"
        assert counts["A"] == 2
        assert counts["E"] == 1
        assert counts["B"] == 0

    def test_empty_array(self):
        labels, counts = classify_array([], [])
        assert labels == []
        assert all(v == 0 for v in counts.values())

    def test_mismatched_lengths(self):
        with pytest.raises(AssertionError):
            classify_array([100], [100, 200])
