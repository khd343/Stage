"""Sessions the vendor prints on days the exchange was shut.

Yahoo emits a row for NSE holidays: the previous close carried forward, volume
exactly zero. Measured on 14 Sep 2026, 19 of the 20 largest names carried one.

THE COVERAGE RULE CANNOT SEE THIS, and that is the whole reason this exists.
The information boundary admits a session at 98% coverage, but a day on which
the vendor carries every name forward looks BETTER covered than average. On
this repo's own published breadth history, 2026-05-28 and 2026-06-26 recorded
101% of a normal session's symbol count while nothing traded. No threshold on
completeness reaches that; only volume does.

Measured separation over 78 sessions of the live universe: real sessions never
fall below 99.16% of symbols trading, dead sessions sit at exactly 0.0%. The
floor is set inside that gap, far enough from the real side that dropping a
genuine session -- which would corrupt every metric for the whole universe on
that day -- is the failure this is tuned against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rs_stages.data import (MIN_SESSIONS_TO_JUDGE_RATE, MIN_SYMBOLS_TO_JUDGE,
                            TRADED_SHARE_FLOOR, non_trading_sessions,
                            normalize_session_index, without_sessions)

SESSIONS = pd.to_datetime(["2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15"])
HOLIDAY = pd.Timestamp("2026-09-14")


def _frame(volumes: list[float], closes: list[float] | None = None,
           index: pd.DatetimeIndex = SESSIONS) -> pd.DataFrame:
    closes = closes or [100.0] * len(index)
    return pd.DataFrame({"Close": closes, "High": closes, "Low": closes,
                         "Volume": volumes}, index=index)


def _universe(n: int = 60, holiday_volume: float = 0.0) -> dict[str, pd.DataFrame]:
    """A universe that traded on every session except the holiday."""
    return {f"S{i}": _frame([5e5, 6e5, holiday_volume, 7e5]) for i in range(n)}


# --- detection ------------------------------------------------------------------

def test_a_session_on_which_nothing_traded_is_found():
    assert non_trading_sessions(_universe()) == [HOLIDAY]


def test_an_ordinary_session_is_never_flagged():
    live = {f"S{i}": _frame([5e5, 6e5, 4e5, 7e5]) for i in range(60)}
    assert non_trading_sessions(live) == []


def test_the_handful_of_illiquid_names_that_never_trade_do_not_condemn_a_session():
    """Measured: on a real session the worst observed was 99.16% of symbols
    trading, so a few genuine zeros are normal and must not read as a closure."""
    universe = {f"S{i}": _frame([5e5, 6e5, 4e5, 7e5]) for i in range(60)}
    for i in range(3):                       # 5% of the universe, well above the real-world 0.84%
        universe[f"S{i}"] = _frame([5e5, 6e5, 0.0, 7e5])
    assert non_trading_sessions(universe) == []


def test_the_rule_is_a_share_not_a_count_so_universe_size_does_not_matter():
    assert non_trading_sessions(_universe(n=40)) == [HOLIDAY]
    assert non_trading_sessions(_universe(n=900)) == [HOLIDAY]


def test_too_few_observations_to_judge_are_left_alone():
    """A thin fetch is not evidence of a closure. Such a session cannot become
    the information boundary anyway, so silence is the safe answer."""
    thin = {f"S{i}": _frame([5e5, 6e5, 0.0, 7e5]) for i in range(MIN_SYMBOLS_TO_JUDGE - 1)}
    assert non_trading_sessions(thin) == []


def test_an_unknown_volume_is_not_a_zero_volume():
    """NaN means the vendor said nothing, which is not the same as saying the
    name did not trade. Counting it as zero would fabricate closures."""
    universe = {f"S{i}": _frame([5e5, 6e5, np.nan, 7e5]) for i in range(60)}
    assert non_trading_sessions(universe) == []


def test_a_symbol_with_no_volume_column_is_skipped_not_counted():
    universe = _universe(n=40)
    universe["NOVOL"] = pd.DataFrame({"Close": [1.0] * 4}, index=SESSIONS)
    assert non_trading_sessions(universe) == [HOLIDAY]


def test_sessions_are_judged_in_the_pipelines_own_date_space():
    """Histories arrive raw, so this has to name a session the way the rest of
    the pipeline will. Being independently "correct" about a timezone while
    normalize_session_index says otherwise would remove a row the snapshot
    never had and leave the one it did. The invariant is AGREEMENT, not a
    particular convention -- so this pins them together rather than pinning a
    date, and survives a change to either.

    Unreachable today: yfinance returns a tz-naive index. Verified, not assumed.
    """
    idx = SESSIONS.tz_localize("Asia/Kolkata")
    frame = _frame([5e5, 6e5, 0.0, 7e5], index=idx)
    dead = non_trading_sessions({f"S{i}": frame for i in range(60)})
    assert len(dead) == 1

    normalized = normalize_session_index(frame)
    # EXACT identity, not membership: a shifted date set still overlaps an
    # unshifted one, so "is it in there" is satisfiable by coincidence and let
    # a diverging convention through when this was first written.
    silent = list(normalized.index[normalized["Volume"] == 0])
    assert dead == silent, "must flag the row the pipeline itself sees as untraded"

    after = normalize_session_index(without_sessions({"X": frame}, dead)["X"])
    assert list(after.index) == [s for s in normalized.index if s not in silent]


def test_nothing_to_judge_yields_nothing():
    assert non_trading_sessions({}) == []
    assert non_trading_sessions({"X": pd.DataFrame()}) == []


def test_the_result_is_sorted_and_deterministic():
    idx = pd.to_datetime(["2026-05-28", "2026-06-26", "2026-09-14", "2026-09-15"])
    universe = {f"S{i}": _frame([0.0, 0.0, 0.0, 7e5], index=idx) for i in range(60)}
    found = non_trading_sessions(universe)
    assert found == sorted(found) and len(found) == 3


def test_flagging_an_implausible_share_of_the_window_aborts():
    """NSE closes about 15 days a year, so a two-year window holds roughly 6%
    dead sessions. A rule firing on far more has misfired, and dropping that
    many real sessions would corrupt every metric for the whole universe."""
    idx = pd.bdate_range("2026-01-01", periods=120)
    universe = {f"S{i}": pd.DataFrame(
        {"Close": [100.0] * 120, "High": [100.0] * 120, "Low": [100.0] * 120,
         "Volume": [0.0] * 40 + [5e5] * 80}, index=idx) for i in range(60)}
    with pytest.raises(SystemExit, match="non-trading"):
        non_trading_sessions(universe)


# --- removal ---------------------------------------------------------------------

def test_the_dead_session_leaves_every_symbol():
    cleaned = without_sessions(_universe(n=5), [HOLIDAY])
    for frame in cleaned.values():
        assert HOLIDAY not in pd.DatetimeIndex(frame.index)
        assert len(frame) == 3


def test_every_other_row_survives_untouched():
    before = _universe(n=1)["S0"]
    after = without_sessions({"S0": before}, [HOLIDAY])["S0"]
    kept = before.drop(index=HOLIDAY)
    pd.testing.assert_frame_equal(after, kept)


def test_a_symbol_that_never_had_the_session_is_unaffected():
    idx = pd.to_datetime(["2026-09-10", "2026-09-11", "2026-09-15"])
    frame = _frame([1.0, 2.0, 3.0], index=idx)
    after = without_sessions({"X": frame}, [HOLIDAY])["X"]
    pd.testing.assert_frame_equal(after, frame)


def test_removing_nothing_changes_nothing():
    before = _universe(n=3)
    after = without_sessions(before, [])
    assert list(after) == list(before)
    for symbol in before:
        pd.testing.assert_frame_equal(after[symbol], before[symbol])


def test_removal_matches_on_the_session_date_not_the_timestamp():
    idx = SESSIONS.tz_localize("Asia/Kolkata")
    after = without_sessions({"X": _frame([1.0, 2.0, 0.0, 3.0], index=idx)}, [HOLIDAY])["X"]
    assert len(after) == 3


def test_the_floor_sits_inside_the_measured_gap():
    """0.0% on a dead session, 99.16% worst case on a live one."""
    assert 0.0 < TRADED_SHARE_FLOOR < 0.9916


def test_a_rate_is_not_judged_over_a_handful_of_sessions():
    """One holiday in a four-session window is 25% and entirely ordinary. The
    ceiling is a rate, so it only applies once there are enough sessions for a
    rate to mean anything."""
    assert non_trading_sessions(_universe()) == [HOLIDAY]
    assert MIN_SESSIONS_TO_JUDGE_RATE > 4


# --- the audit must actually use it, and early enough ----------------------------

def test_the_audit_removes_dead_sessions_before_it_builds_anything():
    """Order is the whole point. Filtering after the snapshots are built would
    leave the phantom session inside every moving average, every positional
    lookback and the boundary itself. Existence is not reachability, and this
    repo has paid for that twice."""
    import inspect

    import scripts.real_data_audit as audit

    src = inspect.getsource(audit.main)
    assert "non_trading_sessions(" in src, "the audit never asks"
    assert "without_sessions(" in src, "the audit never acts on the answer"

    asked = src.index("non_trading_sessions(")
    acted = src.index("without_sessions(")
    built = src.index("build_universe_snapshots(")
    acquired = src.index("acquire_universe_histories(")
    assert acquired < asked < acted < built, (
        "the filter must sit between acquisition and snapshot construction"
    )


def test_the_audit_reports_what_it_removed():
    """A session silently vanishing from every history is exactly the kind of
    change that should never be invisible in the log."""
    import inspect

    import scripts.real_data_audit as audit

    src = inspect.getsource(audit.main)
    start = src.index("if closed:")
    branch = src[start:src.index("else:", start)]
    assert "print(" in branch, "the branch that REMOVES sessions must say so"
    assert "len(closed)" in branch, "and must state how many"
