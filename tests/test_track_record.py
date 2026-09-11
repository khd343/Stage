"""The forward record: frozen, pre-registered, append-only, single-basis.

A forecast is evidence only if it was written before its outcome existed and
never edited after. These pin the four properties that make that true here,
and the one numerical trap (the cross-basis rule) that would quietly falsify
every return.
"""
from __future__ import annotations

import gzip
import pathlib
import tempfile
import uuid

import numpy as np
import pandas as pd
import pytest

from rs_stages import track_record as tr

ROOT = pathlib.Path(__file__).resolve().parents[1]


# --- pre-registration --------------------------------------------------------

def test_the_cohorts_are_pinned_verbatim():
    """Changing a rule is a new record, not an edit. This is the tripwire."""
    assert tr.COHORTS == {
        "universe":           "every published row (the control)",
        "buy_star":           "Action == 'BUY★'",
        "stage2_rs80":        "Stage starts with 'Stage 2' and RS_Score >= 80",
        "trend_template":     "Trend_Template_Pass is true",
        "breakout_confirmed": "Breakout_Confirmed is true",
    }
    assert tr.HORIZONS_WEEKS == (4, 8, 13)
    assert tr.rules_version() == tr.rules_version(), "deterministic"


def test_rules_version_moves_when_a_rule_moves(monkeypatch):
    before = tr.rules_version()
    monkeypatch.setitem(tr.COHORTS, "stage2_rs80", "Stage starts with 'Stage 2' and RS_Score >= 75")
    assert tr.rules_version() != before


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Symbol": ["A", "B", "C", "D", "E"],
        "Date": ["2026-09-10"] * 5,
        "Close": [100.0, 50.0, 20.0, 10.0, 5.0],
        "Action": ["BUY★", "BUY", "WATCH★", "SELL", "HOLD"],
        "Stage": ["Stage 2 — Advancing", "Stage 2 — Advancing", "Stage 1 — Basing", "Stage 2 — Advancing", ""],
        "RS_Score": [95.0, 80.0, 99.0, 79.9, np.nan],
        "Trend_Template_Score": [8, 6, 8, 2, 0],
        "Trend_Template_Pass": [True, "False", "True", False, "False"],
        "Above_MA_30W": [True] * 5,
        "Pct_From_52W_High": [-1.0, -5.0, -0.5, -30.0, -60.0],
        "VCP_Contractions": [3, 2, 0, 0, 0],
        "Breakout": [True, False, False, False, False],
        "Breakout_Confirmed": ["True", False, False, False, "False"],
        "Volume_Ratio": [1.5, 0.9, 1.1, 0.4, np.nan],
    })


def test_cohort_rules_are_exact_at_their_edges():
    f = _frame()
    assert list(tr.cohort_members(f, "universe")) == ["A", "B", "C", "D", "E"]
    assert list(tr.cohort_members(f, "buy_star")) == ["A"], "the star glyph, not BUY"
    assert list(tr.cohort_members(f, "stage2_rs80")) == ["A", "B"], "80 is in, 79.9 is out, Stage 1 is out"
    assert list(tr.cohort_members(f, "trend_template")) == ["A", "C"], "bool True and the string 'True' both count"
    assert list(tr.cohort_members(f, "breakout_confirmed")) == ["A"]
    with pytest.raises(KeyError, match="unregistered"):
        tr.cohort_members(f, "rs90")


# --- the archive --------------------------------------------------------------

def _tmpdir() -> pathlib.Path:
    d = pathlib.Path(tempfile.gettempdir()) / f"rs_snap_{uuid.uuid4().hex[:8]}"
    d.mkdir()
    return d


def test_archive_freezes_the_grading_columns_and_first_write_wins():
    d = _tmpdir()
    result = _frame().set_index("Symbol")            # the audit's frame is Symbol-indexed
    day = pd.Timestamp("2026-09-10")
    path = tr.archive_snapshot(result, d, day)
    assert path is not None and path.name == "2026-09-10.csv.gz"
    back = tr.read_snapshot(path)
    assert list(back.columns) == tr.GRADE_COLUMNS
    assert len(back) == 5 and set(back["Symbol"]) == {"A", "B", "C", "D", "E"}

    first = path.read_bytes()
    revised = result.copy(); revised["Close"] = 999.0
    assert tr.archive_snapshot(revised, d, day) is None, "a later run must not rewrite the past"
    assert path.read_bytes() == first


def test_archive_refuses_a_partial_record():
    d = _tmpdir()
    with pytest.raises(ValueError, match="refusing to archive a partial record"):
        tr.archive_snapshot(_frame().drop(columns=["RS_Score"]), d, pd.Timestamp("2026-09-10"))
    assert not list(d.glob("*.csv.gz"))


# --- returns: one basis, exact endpoints --------------------------------------

def _closes() -> pd.Series:
    idx = pd.bdate_range("2026-09-01", periods=80)
    return pd.Series(np.linspace(100.0, 140.0, 80), index=idx)


def test_forward_return_takes_both_ends_from_one_series():
    s = _closes()
    r = tr.forward_return(s, pd.Timestamp("2026-09-10"), 4)
    start = s.loc["2026-09-10"]; end = s.iloc[s.index.searchsorted(pd.Timestamp("2026-10-08"))]
    assert r == pytest.approx((end / start - 1) * 100)


def test_forward_return_is_nan_when_either_end_is_missing():
    s = _closes()
    assert np.isnan(tr.forward_return(s, pd.Timestamp("2026-09-12"), 4)), "a Saturday is not a session"
    assert np.isnan(tr.forward_return(s, pd.Timestamp("2026-12-10"), 13)), "the end lies past the series"
    assert np.isnan(tr.forward_return(pd.Series(dtype=float), pd.Timestamp("2026-09-10"), 4))


def test_the_archived_close_is_never_a_return_endpoint():
    """The cross-basis rule, checked structurally.

    The archive's Close is adjusted as of its day; the grading download as of
    today. A split between them puts the two on different bases, so no return
    may mix them. forward_return takes a series and nothing else.
    """
    import inspect
    params = list(inspect.signature(tr.forward_return).parameters)
    assert params == ["closes", "start", "weeks"]
    src = inspect.getsource(tr.grade_snapshot)
    assert 'frame["Close"]' not in src and "frame.Close" not in src


# --- grading and coverage ------------------------------------------------------

def test_gradeable_waits_for_the_settle_margin():
    cal = pd.bdate_range("2026-09-01", periods=40)                     # ends 2026-10-23
    assert tr.gradeable(pd.Timestamp("2026-09-10"), 4, cal), "4w + 3 sessions have passed"
    assert not tr.gradeable(pd.Timestamp("2026-09-24"), 4, cal), "4w lands 10-22; only 1 session after"
    assert not tr.gradeable(pd.Timestamp("2026-09-10"), 13, cal)


def test_grade_records_coverage_and_measures_against_the_control():
    f = _frame()
    idx = pd.bdate_range("2026-09-01", periods=60)
    closes = {
        "A": pd.Series(np.linspace(100, 130, 60), index=idx),   # +30%-ish
        "B": pd.Series(np.linspace(50, 55, 60), index=idx),     # +10%-ish
        "C": pd.Series(np.linspace(20, 18, 60), index=idx),     # negative
        "D": pd.Series(np.linspace(10, 10, 60), index=idx),     # flat
        # "E" absent: delisted -> counted in n, not in n_priced
    }
    bench = pd.Series(np.linspace(1000, 1050, 60), index=idx)
    out = tr.grade_snapshot(f, closes, bench, pd.Timestamp("2026-09-10"), 4, graded_on=pd.Timestamp("2026-11-01").date())
    assert list(out.columns) == tr.TRACK_COLUMNS
    uni = out[out.cohort == "universe"].iloc[0]
    assert uni.n == 5 and uni.n_priced == 4, "E is a visible hole, not a silent omission"
    star = out[out.cohort == "buy_star"].iloc[0]
    assert star.n == 1 and star.n_priced == 1
    assert star.excess_vs_universe_pp == pytest.approx(star.median_return_pct - uni.median_return_pct, abs=1e-3)
    assert star.excess_vs_benchmark_pp == pytest.approx(star.median_return_pct - uni.benchmark_return_pct, abs=1e-3)
    assert (out.rules_version == tr.rules_version()).all()
    assert (out.graded_on == "2026-11-01").all()


def test_append_only_never_touches_an_existing_row():
    existing = pd.DataFrame([{**{c: None for c in tr.TRACK_COLUMNS},
                              "snapshot_date": "2026-09-10", "horizon_weeks": 4, "cohort": "buy_star",
                              "n": 1, "n_priced": 1, "median_return_pct": 12.0, "rules_version": "old", "graded_on": "2026-10-10"}])
    new = existing.copy(); new["median_return_pct"] = 99.0; new["rules_version"] = "new"
    other = existing.copy(); other["cohort"] = "universe"; other["median_return_pct"] = 3.0
    merged = tr.append_only(existing, pd.concat([new, other]))
    assert len(merged) == 2
    kept = merged[merged.cohort == "buy_star"].iloc[0]
    assert kept.median_return_pct == 12.0 and kept.rules_version == "old", "recomputation must not overwrite"
    assert merged[merged.cohort == "universe"].iloc[0].median_return_pct == 3.0
    again = tr.append_only(merged, pd.concat([new, other]))
    pd.testing.assert_frame_equal(again, merged, check_dtype=False)


# --- the workflow --------------------------------------------------------------

def test_the_grading_workflow_only_ever_publishes_the_record():
    text = (ROOT / ".github" / "workflows" / "monthly_track_record.yml").read_text(encoding="utf-8")
    code = "\n".join(l.split("#", 1)[0] for l in text.splitlines())
    assert "git add data/track_record.csv" in code
    assert "data/snapshots" not in code and "latest_research" not in code, "the record job never touches the audit's outputs"
    assert "git pull --rebase" in code, "it pushes, so it must survive a moved main"
    assert "workflow_dispatch" in code
