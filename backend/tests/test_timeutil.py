"""Tests for backend/app/timeutil.py timestamp parsing."""

import pytest
import pandas as pd

from app.timeutil import parse_timestamps


def test_timeutil_same_datetime_across_formats():
    """Verify ISO 'T', ISO space, and Day-first parse to the same datetime."""
    s1 = parse_timestamps(["2020-01-15T08:00:00"])
    s2 = parse_timestamps(["2020-01-15 08:00:00"])
    s3 = parse_timestamps(["15-01-2020 08:00:00"])

    assert s1.iloc[0] == s2.iloc[0] == s3.iloc[0]
    assert s1.iloc[0] == pd.Timestamp("2020-01-15 08:00:00")


def test_timeutil_offsets_discarded_and_tz_naive():
    """Verify 'Z' and UTC offsets are discarded, wall-clock kept, and result is tz-naive."""
    s_z = parse_timestamps(["2020-01-15T08:00:00Z"])
    s_offset = parse_timestamps(["2020-01-15T08:00:00+05:30"])

    expected = pd.Timestamp("2020-01-15 08:00:00")
    assert s_z.iloc[0] == expected
    assert s_offset.iloc[0] == expected
    assert s_z.dt.tz is None
    assert s_offset.dt.tz is None


def test_timeutil_mixed_formats_raise_value_error():
    """Verify mixed ISO and Day-first formats raise ValueError with message."""
    with pytest.raises(ValueError, match="Mixed timestamp formats"):
        parse_timestamps(["2020-01-15T08:00:00", "15-01-2020 08:05:00"])


def test_timeutil_invalid_timestamps_raise_value_error():
    """Verify invalid date and invalid pattern raise ValueError naming accepted formats."""
    with pytest.raises(ValueError) as exc1:
        parse_timestamps(["2020-13-45T08:00:00"])
    assert "2020-13-45T08:00:00" in str(exc1.value)
    assert "ISO 8601" in str(exc1.value)
    assert "Day-first" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        parse_timestamps(["hello"])
    assert "hello" in str(exc2.value)
    assert "ISO 8601" in str(exc2.value)
    assert "Day-first" in str(exc2.value)
