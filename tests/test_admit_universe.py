"""Admission is by liquidity, never by history, and never touches existing rows.

The engine already refuses every metric a young symbol lacks history for, so
the universe file has exactly one job at the gate: keep out names that would
eat the coverage budget. These pin that rule and the merge's invariants --
sorted, unique, no new sector spelling, existing rows byte-identical.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts.admit_universe import (LIQUIDITY_MIN_SAMPLE, LIQUIDITY_RATIO, LIQUIDITY_WINDOW,
                                    candidates_from_list, liquid, merge, session_calendar)


def _sessions(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range(end="2026-09-09", periods=n)


def _closes(present: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.linspace(100, 110, len(present)), index=present)


def test_only_nse_rows_are_candidates_and_the_prefix_is_stripped():
    path = pathlib.Path(tempfile.gettempdir()) / "cand.csv"
    path.write_text("companyId,Name,Industry\n"
                    "NSE:ABC,ABC Limited,Healthcare\n"
                    "BSE:XYZ,XYZ Ltd,Healthcare\n"
                    "NSE:ABC,ABC Limited,Healthcare\n", encoding="utf-8")
    out = candidates_from_list(path)
    assert list(out["Symbol"]) == ["ABC"], "BSE dropped, duplicate collapsed, prefix removed"
    assert out.loc[0, "Company Name"] == "ABC Ltd", "house style is Ltd, not Limited"


def test_the_calendar_is_observed_not_assumed():
    """A holiday must not count against a name that could not have traded."""
    cal = _sessions(60)
    holiday = cal[-5]
    a = _closes(cal.drop(holiday))          # everyone was closed that day
    b = _closes(cal.drop(holiday))
    observed = session_calendar({"A": a, "B": b})
    assert holiday not in observed
    assert len(observed) == 59


def test_liquidity_is_presence_over_possible_sessions():
    cal = _sessions(LIQUIDITY_WINDOW)
    k = int(LIQUIDITY_RATIO * LIQUIDITY_WINDOW)          # 54 of 60
    closes = {
        "FULL": _closes(cal),
        "EDGE": _closes(cal[:k]),        # first-listed long ago, 54 of 60 -> exactly 90%
        "THIN": _closes(cal[:k - 1]),    # 53 of 60
        "DEAD": pd.Series(dtype=float),
    }
    admitted, counts = liquid(closes)
    assert admitted == {"FULL", "EDGE"}
    assert counts["THIN"] == (k - 1, LIQUIDITY_WINDOW) and counts["DEAD"] == (0, 0)


def test_young_but_liquid_is_admitted_under_the_DEFAULT_rule():
    """The 18 Aug 2026 migration batch: listed 18 sessions ago, traded all 18.

    The first version of this test passed `minimum=15` explicitly, which hid the
    fact that the default rule -- traded on 54 of a fixed 60 -- could never admit
    such a name. The dry run rejected the whole batch. A test that overrides the
    rule it exists to pin tests nothing.
    """
    cal = _sessions(LIQUIDITY_WINDOW)
    admitted, counts = liquid({"YOUNG": _closes(cal[-18:]), "FULL": _closes(cal)})
    assert "YOUNG" in admitted
    assert counts["YOUNG"] == (18, 18)


def test_a_handful_of_sessions_cannot_prove_liquidity():
    """2 of 2 is 100% and means nothing; the sample floor keeps it out."""
    cal = _sessions(LIQUIDITY_WINDOW)
    n = LIQUIDITY_MIN_SAMPLE - 1
    admitted, counts = liquid({"NEW": _closes(cal[-n:]), "FULL": _closes(cal)})
    assert "NEW" not in admitted
    assert counts["NEW"] == (n, n)


def test_absence_is_measured_against_the_sessions_the_name_could_have_traded():
    """Listed 30 sessions ago, missed 6 of them: 24/30 = 80% -> out."""
    cal = _sessions(LIQUIDITY_WINDOW)
    mine = cal[-30:]
    admitted, counts = liquid({"GAPPY": _closes(mine[:-6]), "FULL": _closes(cal)})
    # dropping the LAST six sessions keeps first-listed at cal[-30]; possible=30, traded=24
    assert counts["GAPPY"] == (24, 30)
    assert "GAPPY" not in admitted


def _universe() -> pd.DataFrame:
    return pd.DataFrame({
        "Company Name": ["Alpha Ltd", "Charlie Ltd"],
        "Industry": ["Healthcare", "Automobile"],
        "Symbol": ["ALPHA", "CHARLIE"],
        "Series": ["EQ", "EQ"],
    })


def test_merge_inserts_in_symbol_order_and_leaves_existing_rows_untouched():
    add = pd.DataFrame({"Symbol": ["BRAVO"], "Company Name": ["Bravo Ltd"], "Industry": ["Healthcare"]})
    out = merge(_universe(), add)
    assert list(out["Symbol"]) == ["ALPHA", "BRAVO", "CHARLIE"]
    assert list(out.columns) == ["Company Name", "Industry", "Symbol", "Series"]
    assert out.loc[out.Symbol == "BRAVO", "Series"].item() == "EQ"
    pd.testing.assert_frame_equal(out[out.Symbol != "BRAVO"].reset_index(drop=True), _universe())


def test_merge_refuses_a_new_sector_spelling():
    """An unknown value would silently become an 81st group in every sector view."""
    add = pd.DataFrame({"Symbol": ["BRAVO"], "Company Name": ["Bravo Ltd"], "Industry": ["Health care"]})
    with pytest.raises(ValueError, match="81st group"):
        merge(_universe(), add)


def test_merge_never_reclassifies_an_existing_name():
    """The candidate list disagreeing about a sector is a decision, not a side effect."""
    add = pd.DataFrame({"Symbol": ["ALPHA"], "Company Name": ["Alpha Ltd"], "Industry": ["Automobile"]})
    out = merge(_universe(), add)
    assert out.loc[out.Symbol == "ALPHA", "Industry"].item() == "Healthcare"
    assert len(out) == 2


def test_a_new_sector_is_admitted_only_by_explicit_declaration():
    """The universe inherited a source with no Banks category at all.

    Adding one is a vocabulary decision, so it must be named -- the same value
    that is refused as a typo is accepted once declared. Refusal stays the
    default so a misspelling can never ride in on the allowlist's coat-tails.
    """
    add = pd.DataFrame({"Symbol": ["HDFCBANK"], "Company Name": ["HDFC Bank Ltd"], "Industry": ["Banks"]})
    with pytest.raises(ValueError, match="81st group"):
        merge(_universe(), add)
    out = merge(_universe(), add, allow_sectors=frozenset({"Banks"}))
    assert out.loc[out.Symbol == "HDFCBANK", "Industry"].item() == "Banks"
    with pytest.raises(ValueError, match="81st group"):
        merge(_universe(), add.assign(Industry="Bank"), allow_sectors=frozenset({"Banks"}))


def test_the_tv_sector_map_never_lands_in_a_residual_by_measurement():
    """A residual bucket is not a sector, so co-occurrence may not choose it.

    The first version of the map let "Finance/Rental/Leasing" -- 64 names,
    essentially every NBFC -- resolve to Miscellaneous on a 7-name overlap, and
    BAJFINANCE was filed under Miscellaneous. Any industry whose measured
    plurality is the residual needs a hand decision, recorded as such.
    """
    root = pathlib.Path(__file__).resolve().parents[1]
    rows = list(__import__("csv").DictReader(open(root / "data" / "sector_map_tv.csv", encoding="utf-8")))
    assert rows, "the map exists and is non-empty"
    measured_into_residual = [r["tv_industry"] for r in rows
                              if r["basis"] == "measured" and r["sector"] == "Miscellaneous"]
    assert not measured_into_residual, measured_into_residual
    for r in rows:
        assert r["basis"] in {"measured", "hand"}, r
        assert r["reason"].strip(), f"{r['tv_industry']} has no recorded reason"


def test_every_mapped_sector_is_in_the_vocabulary_or_explicitly_new():
    """The map may only speak the universe's language, plus the two declared additions."""
    root = pathlib.Path(__file__).resolve().parents[1]
    csv = __import__("csv")
    known = {r["Industry"] for r in csv.DictReader(open(root / "data" / "ind_niftytotalmarket_list.csv", encoding="utf-8"))}
    declared_new = {"Banks", "Insurance"}
    mapped = {r["sector"] for r in csv.DictReader(open(root / "data" / "sector_map_tv.csv", encoding="utf-8"))}
    stray = mapped - known - declared_new
    assert not stray, f"map introduces undeclared sector value(s): {sorted(stray)}"
