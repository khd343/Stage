"""Corporate actions masquerading as price moves.

NSE applies circuit bands of 2, 5, 10 or 20 percent to what it lists, so a
single session that moves 40 or 70 percent did not happen as a price move. It
is a split, a bonus or a demerger arriving in the price series as though the
money had evaporated.

Measured on this repo's own published panel (342 sessions, 750 symbols): six
such sessions, five of which match no clean split ratio and are therefore
demergers, the case yfinance's auto_adjust does not handle. On the snapshot of
17 Sep 2026 four names still carried one inside their 52-week window and every
one of them read "Stage 4 - Declining, SELL" with relative strength in the
bottom decile, having never declined.

These pin detection, classification, and the one predicate the forward record
depends on: whether a return spans a discontinuity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rs_stages import corporate_actions as ca


def _series(values: list[float], start: str = "2026-01-05") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


# --- what is and is not a price move -----------------------------------------

def test_a_move_inside_the_circuit_band_is_never_flagged():
    """20 percent is the widest ordinary band, so a 19 percent day is real."""
    found = ca.implausible_sessions(_series([100.0, 119.0, 96.4, 100.0]))
    assert found.empty


def test_a_halving_is_flagged_and_named():
    found = ca.implausible_sessions(_series([100.0, 101.0, 50.5, 51.0]))
    assert len(found) == 1
    row = found.iloc[0]
    assert row["Date"] == pd.Timestamp("2026-01-07"), "the session that moved, not its neighbour"
    assert row["Ratio"] == pytest.approx(0.5)
    assert row["Looks_Like"] == "1:2 split or 1:1 bonus"
    assert row["Kind"] == "split/bonus"


def test_an_unmatched_ratio_is_reported_as_a_possible_demerger():
    """A demerger leaves no clean ratio. Forcing it to the nearest split would
    invite a correction that corrupts the data further."""
    found = ca.implausible_sessions(_series([100.0, 100.0, 35.1]))
    assert len(found) == 1
    assert found.iloc[0]["Kind"] == "unclassified"
    assert "demerger" in found.iloc[0]["Looks_Like"]


def test_a_reverse_split_is_flagged_too():
    """The upward direction is rarer but it is the one that could pull a name
    INTO a cohort on a move that never happened."""
    found = ca.implausible_sessions(_series([100.0, 100.0, 500.0]))
    assert len(found) == 1
    assert found.iloc[0]["Looks_Like"] == "5:1 reverse split"


def test_gaps_in_the_series_are_closed_before_ratios_are_taken():
    """A blank session is not a 100 percent fall. Stage's histories carry NaN
    rows, and treating one as a price would flag every name that has one."""
    assert ca.implausible_sessions(_series([100.0, np.nan, 101.0, np.nan, 102.0])).empty


def test_a_series_too_short_to_have_a_ratio_yields_nothing():
    assert ca.implausible_sessions(_series([100.0])).empty
    assert ca.implausible_sessions(pd.Series(dtype=float)).empty


# --- across the universe -------------------------------------------------------

def test_scan_names_each_symbol_and_puts_the_worst_first():
    found = ca.scan({
        "CALM":  _series([100.0, 101.0, 102.0]),
        "HALVE": _series([100.0, 100.0, 50.0]),
        "CRASH": _series([100.0, 100.0, 21.0]),
    })
    assert list(found["Symbol"]) == ["CRASH", "HALVE"], "worst first, calm name absent"
    assert set(found.columns) >= {"Symbol", "Date", "Ratio", "Move_Pct", "Looks_Like", "Kind"}


def test_scan_can_be_limited_to_a_window():
    """The snapshot flag asks about the 52-week window, not all history."""
    s = _series([100.0, 40.0] + [40.0] * 8)
    assert len(ca.scan({"X": s})) == 1
    assert ca.scan({"X": s}, since=s.index[3]).empty


# --- the predicate the forward record depends on --------------------------------

def test_a_return_spanning_a_phantom_is_refused():
    s = _series([100.0] * 3 + [35.0] * 4)
    assert ca.spans_discontinuity(s, s.index[0], s.index[5])


def test_a_return_that_ends_before_the_phantom_is_clean():
    s = _series([100.0] * 3 + [35.0] * 4)
    assert not ca.spans_discontinuity(s, s.index[0], s.index[2])


def test_a_phantom_on_the_start_session_does_not_taint_the_return():
    """The split happened ON the snapshot day. Both endpoints are at or after
    it, so both are on the post-split basis and the return is sound. Refusing
    it would throw away a year of gradeable rows for every name that ever
    split on a Monday."""
    s = _series([100.0] * 3 + [50.0] * 4)
    phantom = s.index[3]
    assert ca.spans_discontinuity(s, s.index[0], phantom), "spanning it is bad"
    assert not ca.spans_discontinuity(s, phantom, s.index[6]), "starting on it is fine"


def test_an_unknown_window_is_treated_as_clean():
    """A symbol with no prices cannot be shown to be contaminated, and the
    absent-price case is already reported as a coverage hole."""
    assert not ca.spans_discontinuity(pd.Series(dtype=float), pd.Timestamp("2026-01-05"),
                                      pd.Timestamp("2026-02-05"))


# --- the mass-failure guard ------------------------------------------------------

def test_a_vendor_correcting_a_handful_of_names_does_not_trip_the_guard():
    """MEASURED, and it moved this threshold. On 2026-04-20 five illiquid names
    jumped in one session in BOTH directions -- a vendor unfreezing stale
    prices, not five corporate actions. The first ceiling was nine, which that
    real event came within four of. Aborting costs a session from the record
    permanently, and since these rows are now flagged, publishing them marked
    is strictly better than not publishing at all."""
    for count in (1, 3, 5, 20, 35):
        assert not ca.is_mass_failure(count, universe_size=1784), count


def test_an_event_large_enough_to_distort_the_cross_section_trips_the_guard():
    """RS_Score is a cross-sectional percentile, so a large enough block of
    fictional prices moves the rank of every OTHER name too. Flagging cannot
    repair that, which is what the guard is actually protecting."""
    assert ca.is_mass_failure(36, universe_size=1784)
    assert ca.is_mass_failure(600, universe_size=1784)


def test_the_guard_has_a_floor_so_a_tiny_universe_cannot_disarm_it():
    assert not ca.is_mass_failure(3, universe_size=20)
    assert ca.is_mass_failure(ca.MASS_EVENT_FLOOR, universe_size=20)
