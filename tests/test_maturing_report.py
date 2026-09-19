"""The maturing report is the engine's own verdict, restated -- never a second rule.

A young name is IN the universe already; the night it crosses 200 sessions its
RS, Stage and 52-week fields appear on their own. This report only says who is
next. It must therefore flag exactly the names whose High_52W the engine left
blank, and its threshold must be the engine's, not a copy that can drift.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from rs_stages.data import build_decision_snapshot
from rs_stages.quant import high_52w
from scripts.real_data_audit import MATURITY_SESSIONS, maturing_report

BOUNDARY = pd.Timestamp("2026-09-10")
DECISION = pd.Timestamp("2026-09-11")


def _snapshot(sessions: int):
    idx = pd.bdate_range(end=BOUNDARY, periods=sessions)
    close = pd.Series(np.linspace(100.0, 120.0, sessions), index=idx)
    frame = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                          "Volume": np.full(sessions, 1e5)})
    return build_decision_snapshot(frame, DECISION)


def test_the_threshold_is_the_engines_and_cannot_drift():
    assert MATURITY_SESSIONS == inspect.signature(high_52w).parameters["min_sessions"].default


def test_flag_follows_high_52w_not_a_recount():
    """A row the engine gave a 52-week high is never reported, whatever its count."""
    snaps = {"YOUNG": _snapshot(150), "MATURE": _snapshot(260)}
    result = pd.DataFrame({"High_52W": [np.nan, 123.4]}, index=["YOUNG", "MATURE"])
    out = maturing_report(result, snaps, BOUNDARY)
    assert list(out["Symbol"]) == ["YOUNG"]


def _sparse_snapshot(span_days: int = 400, every: int = 3):
    """Long enough history, too few sessions inside the window: a name with
    real trading gaps rather than a recent listing."""
    idx = pd.bdate_range(end=BOUNDARY, periods=span_days)[::every]
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    frame = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                          "Volume": np.full(len(idx), 1e5)})
    return build_decision_snapshot(frame, DECISION)


def test_a_recent_listing_is_waiting_on_its_history_not_on_a_session_count():
    """THE LIVE BUG. high_52w never reaches its min_sessions check for these:
    calendar_asof raises first, because no session exists on or before the
    window start. Reporting a session shortfall for a name that is only young
    describes a rule the engine does not have."""
    snaps = {"YOUNG": _snapshot(150)}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["YOUNG"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    assert row["Blocked_By"] == "history"
    first = pd.bdate_range(end=BOUNDARY, periods=150)[0]
    assert row["Matures_Around"] == (first + pd.Timedelta(weeks=52)).date()


def test_a_name_past_two_hundred_sessions_is_never_told_it_needs_more():
    """Measured live on 2026-09-18: 38 of 130 rows carried a count ABOVE 200
    beside "1 session to go" -- EUROPRATIK at 248. A reader comparing the two
    columns can only conclude the report is broken."""
    snaps = {"YOUNG": _snapshot(240)}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["YOUNG"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    assert row["Sessions_52W"] > MATURITY_SESSIONS
    assert row["Blocked_By"] == "history"
    assert row["Sessions_To_Go"] >= 1, "never a negative or zero countdown"


def test_a_gappy_name_with_enough_history_is_waiting_on_sessions():
    """The engine's OTHER condition, and the only one the old model described."""
    snaps = {"GAPPY": _sparse_snapshot()}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["GAPPY"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    assert row["Blocked_By"] == "sessions"
    assert row["Sessions_To_Go"] == MATURITY_SESSIONS - row["Sessions_52W"]


def test_the_report_names_the_condition_the_engine_actually_applies():
    """Both blockers are real and they are different questions. A report that
    knows only one of them must describe the wrong one for the other."""
    snaps = {"YOUNG": _snapshot(150), "GAPPY": _sparse_snapshot()}
    result = pd.DataFrame({"High_52W": [np.nan, np.nan]}, index=["YOUNG", "GAPPY"])
    out = maturing_report(result, snaps, BOUNDARY)
    assert set(out["Blocked_By"]) == {"history", "sessions"}


def test_the_docstring_no_longer_claims_the_invariant_that_was_false():
    """It asserted High_52W is NaN exactly when the session count is short.
    Live data disproved that on 38 names."""
    import scripts.real_data_audit as audit
    doc = audit.maturing_report.__doc__
    assert "calendar_asof" in doc or "52 weeks of history" in doc


def test_sorted_soonest_first_and_empty_is_a_frame_not_an_error():
    snaps = {"A": _snapshot(190), "B": _snapshot(120), "C": _snapshot(190)}
    result = pd.DataFrame({"High_52W": [np.nan] * 3}, index=["A", "B", "C"])
    out = maturing_report(result, snaps, BOUNDARY)
    assert list(out["Symbol"]) == ["A", "C", "B"], "ties broken by symbol, soonest first"
    empty = maturing_report(pd.DataFrame({"High_52W": [1.0]}, index=["X"]), {"X": _snapshot(260)}, BOUNDARY)
    assert len(empty) == 0
    assert list(empty.columns) == ["Symbol", "Sessions_52W", "Blocked_By",
                                  "Sessions_To_Go", "Matures_Around"]


def test_the_history_countdown_is_the_wait_for_the_listing_to_age():
    """Not a session shortfall borrowed from the other branch. It is the
    business days until the first session falls 52 weeks behind."""
    snaps = {"YOUNG": _snapshot(150)}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["YOUNG"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    expected = len(pd.bdate_range(BOUNDARY, pd.Timestamp(row["Matures_Around"]))) - 1
    assert row["Sessions_To_Go"] == expected >= 1


def test_a_first_session_exactly_on_the_window_start_has_enough_history():
    """The engine's own boundary: calendar_asof searches on-or-before, so a
    session sitting exactly on the window start is FOUND and high_52w proceeds
    to its count check. One day later it raises. Off by one here would
    misname the blocker for every name on its maturing day."""
    start = BOUNDARY - pd.Timedelta(weeks=52)
    idx = pd.DatetimeIndex([start]).append(pd.bdate_range(start, BOUNDARY)[::4][1:])
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    frame = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                          "Volume": np.full(len(idx), 1e5)})
    snaps = {"EDGE": build_decision_snapshot(frame, DECISION)}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["EDGE"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    assert row["Blocked_By"] == "sessions", "on the boundary, history is sufficient"


def test_the_count_is_the_window_not_the_whole_history():
    """Sessions_52W answers "how many sessions inside the 52 weeks", and a name
    with history reaching further back has more sessions than that. Counting
    everything would overstate the count and understate the wait."""
    snap = _sparse_snapshot(span_days=400, every=3)
    snaps = {"GAPPY": snap}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["GAPPY"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]

    close = snap.data["Close"].dropna()
    inside = int(close[(close.index > BOUNDARY - pd.Timedelta(weeks=52))
                       & (close.index <= BOUNDARY)].notna().sum())
    assert len(close) > inside, "fixture must reach back past the window"
    assert row["Sessions_52W"] == inside


def test_the_window_edge_convention_is_fixed_even_though_it_is_immaterial():
    """A session sitting EXACTLY 52 weeks back is excluded from the count.

    This is a convention, not a load-bearing rule: it moves Sessions_52W by at
    most one and cannot change Blocked_By, because high_52w's own window is a
    superset either way. It is pinned so it cannot drift silently, and labelled
    so nobody mistakes it for a rule with consequences.
    """
    start = BOUNDARY - pd.Timedelta(weeks=52)
    idx = pd.DatetimeIndex([start]).append(pd.bdate_range(start, BOUNDARY)[::4][1:])
    close = pd.Series(np.linspace(100.0, 120.0, len(idx)), index=idx)
    frame = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                          "Volume": np.full(len(idx), 1e5)})
    snaps = {"EDGE": build_decision_snapshot(frame, DECISION)}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["EDGE"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    assert row["Sessions_52W"] == len(idx) - 1, "the session on the edge is not counted"
