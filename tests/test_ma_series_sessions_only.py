"""A dated row with no Close is not a session, and must not shift the slope.

Yahoo publishes a dated row before it has that session's values. The calendar-MA
series used to carry a point for every such row -- holding an MA value repeated
from the previous real session. The values were correct; the POSITIONS were not,
and ma_slope_pct steps back a fixed number of positions, so one empty row inside
the trailing ten made "ten sessions ago" reach back only nine.

Measured live on 31 Aug 2026: every one of 1,505 symbols failed its independent
slope check, because 28 Aug arrived as a dated row with no close. The MA values
themselves matched throughout -- a calendar mean skips NaN either way -- which is
why only the slope, and the two Stages whose sign it flipped, disagreed.

LOCKED_SPEC 3 already settles which reading is right: the boundary is "the
latest session carrying a Close, not merely the latest row the provider
emitted". sma_series had always dropped empty rows first; this pins the calendar
series to the same rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rs_stages.quant import ma_30w_series, ma_slope_pct


def _closes(periods: int = 260) -> pd.Series:
    idx = pd.bdate_range(end=pd.Timestamp("2026-08-31"), periods=periods)
    rng = np.random.default_rng(4)
    return pd.Series(100.0 + rng.normal(0.2, 1.5, periods).cumsum(), index=idx).clip(lower=5.0)


def test_the_series_is_indexed_on_sessions_that_have_a_close():
    close = _closes()
    empty = close.index[-3]
    holed = close.copy()
    holed.loc[empty] = np.nan

    series = ma_30w_series(holed)

    assert empty not in series.index, "a dated row with no close is not a session"
    assert len(series) == len(close) - 1


def test_a_blank_row_behaves_exactly_as_if_it_were_absent():
    """The invariant, stated correctly.

    A first attempt asserted the slope was UNCHANGED by blanking a row outside
    the trailing ten. That was wrong: the 30-week window spans ~150 sessions, so
    removing a close legitimately moves the mean at both ends of the slope. What
    must hold is narrower and is the actual defect -- a dated row with no close
    must be indistinguishable from no row at all.
    """
    close = _closes()
    end = close.index[-1]

    for offset in (-3, -12, -40):
        blanked = close.copy()
        blanked.loc[close.index[offset]] = np.nan
        deleted = close.drop(close.index[offset])

        assert ma_slope_pct(ma_30w_series(blanked), end, sessions=10) == ma_slope_pct(
            ma_30w_series(deleted), end, sessions=10), f"offset {offset}"


def test_the_lookback_counts_sessions_not_rows():
    """Ten blank rows in the trailing window must not consume the lookback.

    This is the live 31 Aug 2026 failure in miniature: with blanks occupying
    positions, "ten sessions ago" reached back nine real sessions, and the whole
    universe's slope shifted at once.
    """
    close = _closes()
    end = close.index[-1]
    blanked = close.copy()
    for offset in range(-11, -1):
        blanked.loc[close.index[offset]] = np.nan

    assert ma_slope_pct(ma_30w_series(blanked), end, sessions=10) == ma_slope_pct(
        ma_30w_series(close.drop(close.index[-11:-1])), end, sessions=10)
