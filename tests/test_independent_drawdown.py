"""The audit's second opinion on drawdown and texture.

Every published metric in this repo is re-derived by an implementation that
does not call the production one, and the run aborts when the two disagree.
A column that skipped that step would be the only number on the snapshot
nothing checks, which is exactly how a quiet arithmetic error survives.

These pin the second implementation against hand-computed values -- never
against the production function, because two names for one algorithm is not
a second opinion.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from rs_stages.quant import max_drawdowns, up_days_pct
from scripts.real_data_audit import independent_drawdowns, independent_up_days


def _series(values: list[float], end: str = "2026-09-17") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values)), dtype=float)


def test_the_second_opinion_is_not_the_first_one_renamed():
    """The whole value of the check is that it shares no code path."""
    for fn in (independent_drawdowns, independent_up_days):
        src = inspect.getsource(fn)
        for borrowed in ("max_drawdown", "up_days_pct", "up_session_share", "calendar_asof"):
            assert borrowed not in src, f"{fn.__name__} leans on production {borrowed}"


def test_it_finds_the_worst_fall_by_hand():
    """Rise to 160 over a year, then give back a quarter."""
    v = list(np.linspace(100.0, 160.0, 400))
    v[-20:] = list(np.linspace(160.0, 120.0, 20))
    got = independent_drawdowns(_series(v), pd.Timestamp("2026-09-17"))
    assert got[3] == pytest.approx(-0.25, abs=0.01)
    assert got[12] == pytest.approx(-0.25, abs=0.01)


def test_it_reads_zero_on_an_advance_that_never_retraced():
    got = independent_drawdowns(_series(list(np.linspace(100.0, 160.0, 400))),
                                pd.Timestamp("2026-09-17"))
    assert all(v == 0.0 for v in got.values())


def test_it_leaves_a_window_it_cannot_form_blank():
    got = independent_drawdowns(_series(list(np.linspace(100.0, 110.0, 30))),
                                pd.Timestamp("2026-09-17"))
    assert np.isnan(got[12])


def test_it_counts_up_sessions_in_the_right_direction():
    """Exact at either extreme whatever the window length works out to be."""
    rising = _series(list(np.linspace(100.0, 160.0, 400)))
    falling = _series(list(np.linspace(160.0, 100.0, 400)))
    assert independent_up_days(rising, pd.Timestamp("2026-09-17")) == pytest.approx(100.0)
    assert independent_up_days(falling, pd.Timestamp("2026-09-17")) == pytest.approx(0.0)


def test_the_two_implementations_agree_on_real_shapes():
    """Agreement is the point; this is the assertion the audit itself makes."""
    shapes = {
        "steady rise":   list(np.linspace(100.0, 160.0, 400)),
        "V recovery":    list(np.linspace(100.0, 60.0, 200)) + list(np.linspace(60.0, 130.0, 200)),
        "flat":          [100.0] * 400,
        "sawtooth":      list(100.0 + 20.0 * np.sin(np.linspace(0, 12 * np.pi, 400))),
        "late collapse": list(np.linspace(100.0, 160.0, 380)) + list(np.linspace(160.0, 70.0, 20)),
    }
    t = pd.Timestamp("2026-09-17")
    for name, values in shapes.items():
        s = _series(values)
        mine, theirs = independent_drawdowns(s, t), max_drawdowns(s, t)
        for months in (3, 6, 9, 12):
            assert mine[months] == pytest.approx(theirs[months], abs=1e-12, nan_ok=True), f"{name} {months}M"
        assert independent_up_days(s, t) == pytest.approx(up_days_pct(s, t), abs=1e-12, nan_ok=True), name


def test_a_blank_session_is_not_a_fall_in_either_implementation():
    s = _series([100.0, np.nan, 120.0, np.nan, 90.0] * 80)
    t = pd.Timestamp("2026-09-17")
    assert independent_drawdowns(s, t)[3] == pytest.approx(max_drawdowns(s, t)[3], abs=1e-12)


# --- the comparison the audit actually performs ---------------------------------

from scripts.real_data_audit import reconcile_texture  # noqa: E402


def _row(close: pd.Series, t: pd.Timestamp) -> pd.Series:
    values = {f"MaxDD_{m}M": v for m, v in max_drawdowns(close, t).items()}
    values["Up_Days_Pct_6M"] = up_days_pct(close, t)
    return pd.Series(values)


def test_agreement_produces_no_failures():
    s = _series(list(np.linspace(100.0, 160.0, 400)))
    t = pd.Timestamp("2026-09-17")
    assert reconcile_texture("X", _row(s, t), s, t) == []


def test_a_wrong_drawdown_is_named_and_fails_the_audit():
    s = _series(list(np.linspace(100.0, 160.0, 400)))
    t = pd.Timestamp("2026-09-17")
    row = _row(s, t)
    row["MaxDD_6M"] = -0.4
    failures = reconcile_texture("X", row, s, t)
    assert len(failures) == 1 and "MaxDD_6M" in failures[0] and "X" in failures[0]


def test_a_wrong_up_day_share_is_caught_too():
    s = _series(list(np.linspace(100.0, 160.0, 400)))
    t = pd.Timestamp("2026-09-17")
    row = _row(s, t)
    row["Up_Days_Pct_6M"] = 12.5
    assert any("Up_Days_Pct_6M" in f for f in reconcile_texture("X", row, s, t))


def test_a_column_that_stopped_being_published_is_a_mismatch():
    """Silently skipping an absent column would let the engine drop a metric
    and still pass its own audit."""
    s = _series(list(np.linspace(100.0, 160.0, 400)))
    t = pd.Timestamp("2026-09-17")
    row = _row(s, t).drop(labels=["MaxDD_12M"])
    assert any("MaxDD_12M" in f for f in reconcile_texture("X", row, s, t))


def test_both_sides_blank_is_agreement_not_a_failure():
    """The repo's convention everywhere else: NaN equals NaN here."""
    s = _series(list(np.linspace(100.0, 110.0, 30)))
    t = pd.Timestamp("2026-09-17")
    assert reconcile_texture("X", _row(s, t), s, t) == []


def test_the_audit_loop_actually_calls_it():
    """Existence is not reachability -- this repo has paid for that twice."""
    import scripts.real_data_audit as audit
    src = inspect.getsource(audit.main)
    assert "reconcile_texture(" in src
