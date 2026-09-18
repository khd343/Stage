"""The grader's command-line behaviour, and the one thing CI needs from it.

The publishing step runs `git add data/track_record.csv` under `set -e`. If the
grader wrote nothing, that path does not exist, `git add` fails, and the job
fails with it. Measured: the first manually dispatched run failed exactly
there, and the scheduled runs of 2 October and 2 November would have too,
because no horizon closes before 9 October.

So the record file is a promise, not a by-product: it exists from the first run
with its schema in place, empty until there is something to say. No consumer
then needs to handle its absence.
"""
from __future__ import annotations

import sys

import pandas as pd
import pytest

import scripts.grade_track_record as g
from rs_stages import track_record as tr


@pytest.fixture
def grader(tmp_path, monkeypatch):
    monkeypatch.setattr(g, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setattr(g, "RECORD", tmp_path / "track_record.csv")
    monkeypatch.setattr(sys, "argv", ["grade_track_record"])
    return g


def test_the_grader_leaves_a_record_behind_when_there_is_nothing_to_archive(grader):
    assert grader.main() == 0
    assert grader.RECORD.exists(), "CI stages this path unconditionally"
    back = pd.read_csv(grader.RECORD)
    assert list(back.columns) == tr.TRACK_COLUMNS
    assert back.empty


def test_a_second_run_changes_nothing(grader):
    """Append-only means a re-run is a no-op, and CI reports 'unchanged'
    rather than committing an identical file every month."""
    grader.main()
    first = grader.RECORD.read_bytes()
    grader.main()
    assert grader.RECORD.read_bytes() == first


def test_an_existing_record_is_never_emptied_by_a_quiet_run(grader):
    """The dangerous version of this fix: writing a blank record over real
    grades because this month had nothing new to add."""
    graded = pd.DataFrame([{**{c: None for c in tr.TRACK_COLUMNS},
                            "snapshot_date": "2026-09-11", "horizon_weeks": 4,
                            "cohort": "buy_star", "n": 1, "n_priced": 1,
                            "median_return_pct": 12.0}])
    graded.to_csv(grader.RECORD, index=False)
    assert grader.main() == 0
    back = pd.read_csv(grader.RECORD)
    assert len(back) == 1 and back.iloc[0]["median_return_pct"] == 12.0


def test_a_dry_run_never_writes(grader, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["grade_track_record", "--dry-run"])
    assert grader.main() == 0
    assert not grader.RECORD.exists()
