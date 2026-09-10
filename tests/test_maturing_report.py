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


def test_count_to_go_and_eta_are_consistent():
    snaps = {"YOUNG": _snapshot(150)}
    result = pd.DataFrame({"High_52W": [np.nan]}, index=["YOUNG"])
    row = maturing_report(result, snaps, BOUNDARY).iloc[0]
    assert row["Sessions_52W"] == 150
    assert row["Sessions_To_Go"] == MATURITY_SESSIONS - 150
    assert row["Reaches_200_Around"] == (BOUNDARY + pd.tseries.offsets.BDay(50)).date()


def test_sorted_soonest_first_and_empty_is_a_frame_not_an_error():
    snaps = {"A": _snapshot(190), "B": _snapshot(120), "C": _snapshot(190)}
    result = pd.DataFrame({"High_52W": [np.nan] * 3}, index=["A", "B", "C"])
    out = maturing_report(result, snaps, BOUNDARY)
    assert list(out["Symbol"]) == ["A", "C", "B"], "ties broken by symbol, soonest first"
    empty = maturing_report(pd.DataFrame({"High_52W": [1.0]}, index=["X"]), {"X": _snapshot(260)}, BOUNDARY)
    assert len(empty) == 0 and list(empty.columns) == ["Symbol", "Sessions_52W", "Sessions_To_Go", "Reaches_200_Around"]
