"""Independent validation of the calendar-week MA family and the 52W low.

The optimized series builder must reproduce, bit for bit, the value obtained by
evaluating the scalar calendar-week MA at every session. A faster window walk is
only acceptable if it is numerically indistinguishable from the definition.
"""
import numpy as np
import pandas as pd
import pytest

from rs_stages.quant import (
    calendar_asof,
    high_52w,
    low_52w,
    ma_10w,
    ma_10w_series,
    ma_30w,
    ma_30w_series,
    ma_calendar_weeks,
    ma_calendar_weeks_series,
)


def _reference_scalar(close: pd.Series, end: pd.Timestamp, weeks: int) -> float:
    """Definition-level calendar-window mean, written independently."""
    s = close.sort_index().dropna()
    t = calendar_asof(s.index, end)
    start = calendar_asof(s.index, t - pd.Timedelta(weeks=weeks))
    window = s.loc[(s.index >= start) & (s.index <= t)]
    if len(window) < 2:
        raise ValueError("insufficient window")
    return float(window.mean())


def _reference_series(close: pd.Series, weeks: int) -> pd.Series:
    """Per-session loop over the scalar definition — the slow, obvious version.

    Loops over SESSIONS, meaning rows that carry a Close. This reference used to
    loop over every row the provider emitted, which encoded the same mistake the
    optimized builder made: a dated row with no close was given a position in the
    series, and ma_slope_pct steps back by position. On 31 Aug 2026 that made
    every one of 1,505 symbols fail its independent slope check. LOCKED_SPEC 3
    and sma_series both already say an empty row is not a session; this reference
    was the outlier, so it moved rather than the rule.
    """
    s = close.sort_index().dropna()
    values = []
    for t in s.index:
        try:
            values.append(_reference_scalar(s, t, weeks))
        except ValueError:
            values.append(np.nan)
    return pd.Series(values, index=s.index, dtype=float)


def _price_series(seed: int, periods: int, freq: str) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=periods, freq=freq)
    steps = rng.normal(0.0, 1.5, len(idx)).cumsum()
    return pd.Series(250.0 + steps, index=idx)


@pytest.mark.parametrize("weeks", [10, 30, 52])
def test_series_builder_is_bit_identical_to_the_definition(weeks):
    close = _price_series(seed=7, periods=520, freq="C")
    fast = ma_calendar_weeks_series(close, weeks)
    slow = _reference_series(close, weeks)
    assert fast.index.equals(slow.index)
    both = fast.notna() & slow.notna()
    assert both.sum() > 100
    assert (fast.isna() == slow.isna()).all()
    assert np.array_equal(fast[both].to_numpy(), slow[both].to_numpy())


@pytest.mark.parametrize("weeks", [10, 30])
def test_series_builder_matches_definition_when_history_has_gaps(weeks):
    close = _price_series(seed=11, periods=420, freq="C")
    close.iloc[5:9] = np.nan
    close.iloc[200] = np.nan
    close.iloc[-3] = np.nan
    fast = ma_calendar_weeks_series(close, weeks)
    slow = _reference_series(close, weeks)
    assert (fast.isna() == slow.isna()).all()
    both = fast.notna() & slow.notna()
    assert np.array_equal(fast[both].to_numpy(), slow[both].to_numpy())


def test_30w_helpers_are_unchanged_by_the_generic_refactor():
    close = _price_series(seed=3, periods=460, freq="C")
    end = close.index[-1]
    assert ma_30w(close, end) == ma_calendar_weeks(close, end, 30)
    assert np.isclose(ma_30w(close, end), _reference_scalar(close, end, 30), rtol=0, atol=1e-12)
    assert ma_30w_series(close).equals(ma_calendar_weeks_series(close, 30))


def test_10w_matches_independent_calendar_window():
    close = _price_series(seed=5, periods=300, freq="C")
    end = close.index[-1]
    assert np.isclose(ma_10w(close, end), _reference_scalar(close, end, 10), rtol=0, atol=1e-12)
    assert ma_10w_series(close).equals(ma_calendar_weeks_series(close, 10))


def test_10w_window_is_shorter_than_the_30w_window():
    """A 10-week window must average fewer sessions than a 30-week window."""
    idx = pd.bdate_range("2025-01-01", periods=300)
    close = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    end = idx[-1]
    s = close.sort_index()
    ten = s.loc[calendar_asof(s.index, end - pd.Timedelta(weeks=10)) :]
    thirty = s.loc[calendar_asof(s.index, end - pd.Timedelta(weeks=30)) :]
    assert len(ten) < len(thirty)
    # On a strictly rising series the shorter average must sit above the longer.
    assert ma_10w(close, end) > ma_30w(close, end)


def test_10w_requires_a_reference_session_before_the_window():
    """No session on or before ``end - 10 weeks`` is insufficiency, not a partial mean."""
    idx = pd.date_range("2025-01-01", "2025-02-05", freq="7D")
    close = pd.Series(100.0, index=idx)
    with pytest.raises(ValueError):
        ma_10w(close, idx[-1])



def test_52w_low_mirrors_the_52w_high_window_and_guard():
    end = pd.Timestamp("2026-07-31")
    idx = pd.bdate_range(end - pd.Timedelta(weeks=52), end)
    rng = np.random.default_rng(23)
    low = pd.Series(rng.uniform(50.0, 90.0, len(idx)), index=idx)
    high = low + 15.0
    assert low_52w(low, end) == float(low.min())
    assert high_52w(high, end) == float(high.max())
    assert low_52w(low, end) < high_52w(high, end)


def test_52w_low_requires_200_sessions():
    """199 sessions inside a complete 52-week window is explicit insufficiency."""
    end = pd.Timestamp("2026-07-31")
    recent = pd.bdate_range(end - pd.Timedelta(days=280), periods=198)
    idx = pd.DatetimeIndex([end - pd.Timedelta(weeks=60)]).append(recent)
    low = pd.Series(100.0, index=idx)
    with pytest.raises(ValueError, match="52W low"):
        low_52w(low, end)
