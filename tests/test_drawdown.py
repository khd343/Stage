"""Depth and texture of an advance, which the snapshot could not describe.

Two stocks both up 32% over twelve months are not the same holding if one fell
9% along the way and the other fell 35%. Relative strength ranks the first
number and says nothing about the second, so the screen could not separate a
smooth advance from a violent one -- and that is usually what decides whether
a winner is actually held.

Units follow the columns these sit beside: drawdowns are decimal fractions
like R3M, negative, rendered by fmt_return. The up-day share carries the _Pct
suffix this codebase uses for percentage points, like ATR_Pct.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rs_stages.quant import (max_drawdown, max_drawdowns, rs_returns, up_days_pct,
                             up_session_share)


def _series(values: list[float], end: str = "2026-09-17") -> pd.Series:
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def _long(days: int = 400) -> pd.Series:
    return _series(list(np.linspace(100.0, 160.0, days)))


# --- the core measurement ------------------------------------------------------

def test_the_worst_peak_to_trough_fall_is_measured_not_the_last_one():
    s = _series([100.0, 120.0, 90.0, 110.0, 105.0])
    assert max_drawdown(s, s.index[0], s.index[-1]) == pytest.approx(-0.25)


def test_an_advance_that_never_gave_anything_back_reads_zero():
    s = _series([100.0, 110.0, 120.0, 130.0])
    assert max_drawdown(s, s.index[0], s.index[-1]) == 0.0


def test_the_peak_must_lie_inside_the_window():
    """A crash that already happened before the window opened is not this
    window's drawdown. Measuring from an earlier peak would report a fall the
    holder of this window never lived through."""
    s = _series([500.0, 100.0, 105.0, 102.0, 108.0])
    assert max_drawdown(s, s.index[1], s.index[-1]) == pytest.approx(102.0 / 105.0 - 1.0)
    assert max_drawdown(s, s.index[0], s.index[-1]) == pytest.approx(-0.8)


def test_a_fall_after_the_window_closes_does_not_leak_in():
    s = _series([100.0, 105.0, 110.0, 20.0])
    assert max_drawdown(s, s.index[0], s.index[2]) == 0.0


def test_blank_sessions_are_closed_up_before_measuring():
    s = _series([100.0, np.nan, 120.0, np.nan, 90.0])
    assert max_drawdown(s, s.index[0], s.index[-1]) == pytest.approx(-0.25)


def test_a_window_too_short_to_hold_a_fall_is_unmeasurable():
    s = _series([100.0, 120.0])
    assert np.isnan(max_drawdown(s, s.index[-1], s.index[-1]))
    assert np.isnan(max_drawdown(pd.Series(dtype=float), pd.Timestamp("2026-01-01"),
                                 pd.Timestamp("2026-02-01")))


# --- the ladder, aligned to the returns it sits beside --------------------------

def test_the_drawdown_windows_are_the_return_windows():
    """They are read as a pair on screen, so they must describe one period."""
    s = _long()
    t = s.index[-1]
    assert sorted(max_drawdowns(s, t)) == sorted(rs_returns(s, t))


def test_no_window_ever_reports_a_positive_drawdown():
    values = max_drawdowns(_long(), _long().index[-1])
    assert all(v <= 0.0 for v in values.values())


def test_one_fall_has_one_depth_however_long_the_window():
    v = list(np.linspace(100.0, 160.0, 400))
    v[-20:] = list(np.linspace(160.0, 120.0, 20))
    s = _series(v)
    dd = max_drawdowns(s, s.index[-1])
    assert dd[3] == pytest.approx(-0.25, abs=0.01)
    assert dd[12] == pytest.approx(dd[3], abs=0.01)


def test_a_history_too_short_for_a_window_leaves_it_blank():
    """calendar_asof raises when no session predates the reference, exactly as
    it does for the returns, so a young name reports nothing rather than a
    twelve-month figure measured over six weeks."""
    dd = max_drawdowns(_series(list(np.linspace(100.0, 110.0, 30))), pd.Timestamp("2026-09-17"))
    assert np.isnan(dd[12]) and np.isnan(dd[9])


# --- texture: how the advance was delivered ------------------------------------

def test_the_up_day_share_counts_sessions_that_closed_higher():
    s = _series([100.0, 101.0, 100.0, 102.0, 103.0])
    assert up_session_share(s, s.index[0], s.index[-1]) == pytest.approx(75.0)


def test_the_share_is_percentage_points_not_a_fraction():
    """It sits beside ATR_Pct and Pct_From_52W_High, which are both points."""
    s = _series([100.0, 101.0, 102.0])
    assert up_session_share(s, s.index[0], s.index[-1]) == pytest.approx(100.0)


def test_a_flat_session_is_not_an_up_session():
    s = _series([100.0, 100.0, 101.0])
    assert up_session_share(s, s.index[0], s.index[-1]) == pytest.approx(50.0)


def test_a_window_with_nothing_to_compare_is_unmeasurable():
    s = _series([100.0])
    assert np.isnan(up_session_share(s, s.index[0], s.index[0]))


def test_the_published_share_looks_back_six_months():
    """Long enough to describe character, short enough to still be about the
    move on screen. It is the window Da, Gurun and Warachka used."""
    s = _long()
    t = s.index[-1]
    from rs_stages.quant import calendar_asof
    start = calendar_asof(s.index, t - pd.DateOffset(months=6))
    assert up_days_pct(s, t) == pytest.approx(up_session_share(s, start, t))


def test_the_published_share_is_blank_without_six_months_of_history():
    assert np.isnan(up_days_pct(_series(list(np.linspace(100.0, 110.0, 30))),
                                pd.Timestamp("2026-09-17")))


# --- the published row ----------------------------------------------------------

def _snapshot(values: list[float], decision: str = "2026-09-18"):
    from rs_stages.data import build_decision_snapshot
    idx = pd.bdate_range(end="2026-09-17", periods=len(values))
    close = pd.Series(values, index=idx, dtype=float)
    frame = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                          "Volume": np.full(len(values), 1e6)})
    return build_decision_snapshot(frame, pd.Timestamp(decision))


def test_every_return_on_the_row_has_a_drawdown_beside_it():
    from rs_stages.screener import analyze_universe
    out = analyze_universe({"X": _snapshot(list(np.linspace(100.0, 160.0, 400)))})
    for months in (3, 6, 9, 12):
        assert f"R{months}M" in out.columns
        assert f"MaxDD_{months}M" in out.columns, "a return without its cost is half the story"
    assert "Up_Days_Pct_6M" in out.columns


def test_the_published_drawdown_is_the_locked_function_not_a_copy():
    from rs_stages.screener import analyze_universe
    v = list(np.linspace(100.0, 160.0, 400))
    v[-20:] = list(np.linspace(160.0, 120.0, 20))
    snap = _snapshot(v)
    out = analyze_universe({"X": snap})
    expected = max_drawdowns(snap.data["Close"], snap.latest_completed_session)
    for months, value in expected.items():
        assert out.loc["X", f"MaxDD_{months}M"] == pytest.approx(value, nan_ok=True)
    assert out.loc["X", "Up_Days_Pct_6M"] == pytest.approx(
        up_days_pct(snap.data["Close"], snap.latest_completed_session))


def test_a_young_name_reports_blanks_rather_than_a_short_window():
    from rs_stages.screener import analyze_universe
    out = analyze_universe({"X": _snapshot(list(np.linspace(100.0, 110.0, 30)))})
    assert np.isnan(out.loc["X", "MaxDD_12M"])
    assert np.isnan(out.loc["X", "Up_Days_Pct_6M"])
