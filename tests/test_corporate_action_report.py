"""The audit's corporate-action report and the flag it puts on the snapshot.

The engine's 52-week fields and its relative-strength returns are both read
over the last 52 weeks, so an action inside that window distorts the row and
nothing else does. The flag therefore asks exactly that question, and the
report is written beside the snapshot the same way the maturing list is.

The report NEVER corrects a price. It says which rows are not to be believed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rs_stages.data import build_decision_snapshot
from rs_stages import corporate_actions as ca
from scripts.real_data_audit import (MAX_UNIVERSE_LOSS_PCT, corporate_action_report,
                                     flag_corporate_actions)

BOUNDARY = pd.Timestamp("2026-09-10")
DECISION = pd.Timestamp("2026-09-11")


def _snapshot(closes: list[float]):
    idx = pd.bdate_range(end=BOUNDARY, periods=len(closes))
    close = pd.Series(closes, index=idx, dtype=float)
    frame = pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                          "Volume": np.full(len(closes), 1e5)})
    return build_decision_snapshot(frame, DECISION)


def _calm(n: int = 300) -> list[float]:
    return list(np.linspace(100.0, 120.0, n))


def _with_action_at(position: int, n: int = 300, ratio: float = 0.5) -> list[float]:
    v = np.linspace(100.0, 120.0, n)
    v[position:] = v[position:] * ratio
    return list(v)


def test_a_calm_universe_produces_an_empty_report():
    report = corporate_action_report({"CALM": _snapshot(_calm())}, BOUNDARY, universe_size=1)
    assert report.empty
    assert list(report.columns)[:2] == ["Symbol", "Date"]


def test_an_action_inside_the_window_is_reported_with_its_classification():
    snaps = {"SPLIT": _snapshot(_with_action_at(250))}
    report = corporate_action_report(snaps, BOUNDARY, universe_size=1)
    assert len(report) == 1
    assert report.iloc[0]["Symbol"] == "SPLIT"
    assert report.iloc[0]["Kind"] == "split/bonus"


def test_an_action_older_than_the_window_is_not_reported():
    """It has aged out of every metric the snapshot publishes, so the row is
    sound again and flagging it would cry wolf forever."""
    snaps = {"OLD": _snapshot(_with_action_at(5, n=400))}
    assert corporate_action_report(snaps, BOUNDARY, universe_size=1).empty


def test_the_flag_marks_only_the_affected_rows():
    result = pd.DataFrame({"Close": [10.0, 20.0]}, index=pd.Index(["GOOD", "SPLIT"], name="Symbol"))
    report = corporate_action_report(
        {"GOOD": _snapshot(_calm()), "SPLIT": _snapshot(_with_action_at(250))},
        BOUNDARY, universe_size=2)
    flagged = flag_corporate_actions(result, report)
    assert list(flagged.columns) == ["Close", "Corporate_Action", "Corporate_Action_Date",
                                     "Corporate_Action_Kind"]
    assert "split" in flagged.loc["SPLIT", "Corporate_Action_Kind"]
    assert flagged.loc["SPLIT", "Corporate_Action"] is np.True_ or flagged.loc["SPLIT", "Corporate_Action"]
    assert not flagged.loc["GOOD", "Corporate_Action"]
    assert flagged.loc["GOOD", "Corporate_Action_Date"] == ""
    assert flagged.loc["SPLIT", "Corporate_Action_Date"] != ""


def test_the_flag_reports_the_most_recent_action_when_there_are_two():
    v = np.linspace(100.0, 120.0, 300)
    v[200:] = v[200:] * 0.5
    v[260:] = v[260:] * 0.5
    snaps = {"TWICE": _snapshot(list(v))}
    report = corporate_action_report(snaps, BOUNDARY, universe_size=1)
    assert len(report) == 2
    result = pd.DataFrame({"Close": [1.0]}, index=pd.Index(["TWICE"], name="Symbol"))
    latest = flag_corporate_actions(result, report).loc["TWICE", "Corporate_Action_Date"]
    assert latest == max(report["Date"]).date().isoformat()


def test_the_flag_is_written_even_when_nothing_was_found():
    """A column that appears only on bad days is a column every consumer has to
    guard. The schema must not depend on the weather."""
    result = pd.DataFrame({"Close": [10.0]}, index=pd.Index(["GOOD"], name="Symbol"))
    flagged = flag_corporate_actions(result, pd.DataFrame())
    assert not flagged["Corporate_Action"].any()
    assert (flagged["Corporate_Action_Date"] == "").all()
    assert (flagged["Corporate_Action_Kind"] == "").all()


def test_a_universe_wide_adjustment_failure_aborts_the_audit():
    """Circuit limits mean this cannot be a market event. Publishing it would
    freeze fiction into the archive, which first-write-wins makes permanent."""
    snaps = {f"S{i}": _snapshot(_with_action_at(299)) for i in range(12)}
    with pytest.raises(SystemExit, match="corporate-action"):
        corporate_action_report(snaps, BOUNDARY, universe_size=12)


def test_a_backlog_of_old_actions_never_aborts_the_audit():
    """The guard asks about THIS session only. A dozen names carrying an action
    from different weeks is ordinary and must still publish."""
    snaps = {f"S{i}": _snapshot(_with_action_at(250 + i)) for i in range(12)}
    report = corporate_action_report(snaps, BOUNDARY, universe_size=12)
    assert len(report) == 12


def test_the_two_tolerances_are_one_number():
    """The audit already refuses to publish when more than 2% of the universe
    is unusable. A block of fictional prices is unusable in exactly the same
    way, so it gets the same ceiling. Two numbers for one idea would drift."""
    assert ca.MASS_EVENT_PCT == MAX_UNIVERSE_LOSS_PCT
