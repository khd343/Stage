import importlib.util
import sys
from pathlib import Path

import pandas as pd

_SPEC = importlib.util.spec_from_file_location(
    "real_data_audit", Path(__file__).resolve().parents[1] / "scripts" / "real_data_audit.py"
)
_AUDIT = importlib.util.module_from_spec(_SPEC)
sys.modules["real_data_audit"] = _AUDIT
_SPEC.loader.exec_module(_AUDIT)


def _result(date="2026-08-25", symbols=("AAA", "BBB"), close=100.0):
    return pd.DataFrame(
        {"Date": [date] * len(symbols), "Close": [close] * len(symbols), "RS_Score": [90, 40]},
        index=pd.Index(symbols, name="Symbol"),
    )


def test_archive_writes_one_file_per_snapshot_date(tmp_path):
    _AUDIT._archive_snapshot(_result("2026-08-25"), tmp_path)
    archived = tmp_path / "snapshots" / "research_2026-08-25.csv"
    assert archived.exists()
    back = pd.read_csv(archived)
    assert set(back["Symbol"]) == {"AAA", "BBB"}
    assert "Close" in back.columns, "Close is what makes forward returns computable"


def test_archive_refuses_to_overwrite_an_existing_date(tmp_path):
    """A re-run must never replace what was actually published that day.

    The audit re-runs -- a retry, a manual dispatch, a same-day fix. On 2026-08-24
    four separate commits carried that one snapshot date. If a later run could
    rewrite the dated file, the record's past would be mutable, and a forward record
    whose past can change proves nothing. First publication wins.
    """
    _AUDIT._archive_snapshot(_result("2026-08-25", close=100.0), tmp_path)
    _AUDIT._archive_snapshot(_result("2026-08-25", close=999.0), tmp_path)

    back = pd.read_csv(tmp_path / "snapshots" / "research_2026-08-25.csv")
    assert back["Close"].iloc[0] == 100.0, "the second run overwrote the first"
    assert len(list((tmp_path / "snapshots").glob("*.csv"))) == 1


def test_archive_uses_the_snapshots_own_date_not_today(tmp_path):
    """Naming by today's date would misfile every backfill and every late run."""
    _AUDIT._archive_snapshot(_result("2020-01-02"), tmp_path)
    assert (tmp_path / "snapshots" / "research_2020-01-02.csv").exists()


def test_archive_takes_the_latest_date_when_the_snapshot_is_split(tmp_path):
    """Providers settle unevenly, so a file can carry two dates at once.

    RS-Stages discloses the split rather than claiming one date. The archive takes
    the latest present, matching what the terminal reports -- inventing a single
    date for a split file would be the same dishonesty in reverse.
    """
    split = pd.DataFrame(
        {"Date": ["2026-08-24", "2026-08-25"], "Close": [10.0, 20.0]},
        index=pd.Index(["AAA", "BBB"], name="Symbol"),
    )
    _AUDIT._archive_snapshot(split, tmp_path)
    assert (tmp_path / "snapshots" / "research_2026-08-25.csv").exists()


def test_archive_is_silent_on_empty_or_dateless_input(tmp_path):
    """A failed run must not create a file that later reads as a real snapshot."""
    _AUDIT._archive_snapshot(pd.DataFrame(), tmp_path)
    _AUDIT._archive_snapshot(pd.DataFrame({"Close": [1.0]}), tmp_path)
    assert not (tmp_path / "snapshots").exists() or not list((tmp_path / "snapshots").glob("*.csv"))


def test_workflow_commits_the_archive_directory():
    """Writing the file on the runner is useless if the workflow never adds it."""
    wf = (Path(__file__).resolve().parents[1]
          / ".github" / "workflows" / "real_data_audit.yml").read_text(encoding="utf-8")
    assert "data/snapshots/" in wf, (
        "real_data_audit.yml must `git add data/snapshots/`, or the immutable "
        "record never leaves the runner")
