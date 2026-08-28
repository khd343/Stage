"""LOCKED_SPEC 8.1 -- the information boundary is global.

The rule was written down and not implemented: every symbol stopped at its own
last close, so one published file held 968 symbols priced at 26 Aug 2026 and 536
at 27 Aug. RS_Score is a cross-sectional percentile, so that file ranked the
fresher cohort against the staler one and reported the gap as relative strength.

Measured against Yahoo on 28 Aug 2026 (150 random NSE symbols): a session is
carried for ~39% of the universe 16 hours after its close and ~100% by 40 hours.
NSE opens 17.75 hours after the previous close, so a pre-market run can never
see a complete latest session -- the boundary has to be chosen from coverage,
and no schedule change can substitute for that.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rs_stages.data import global_information_boundary, session_coverage

SESSIONS = pd.to_datetime(["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27"])
DECISION = pd.Timestamp("2026-08-28")


def _history(closes: list[float | None], index: pd.DatetimeIndex = SESSIONS) -> pd.DataFrame:
    return pd.DataFrame({"Close": closes}, index=index)


def _universe(fresh: int, stale: int) -> dict[str, pd.DataFrame]:
    """`fresh` symbols carry the 27th; `stale` ones have a dated row with no Close.

    The empty-but-dated row is the real Yahoo shape, not a simplification: the
    provider publishes the row when the session starts and the values later.
    """
    histories = {f"FRESH{i}": _history([10.0, 11.0, 12.0, 13.0]) for i in range(fresh)}
    histories.update({f"STALE{i}": _history([10.0, 11.0, 12.0, None]) for i in range(stale)})
    return histories


def test_a_dated_row_without_a_close_is_not_coverage():
    """The distinction the whole mechanism rests on.

    Counting rows instead of closes would report the newest session as fully
    covered hours before a single price for it existed, which is precisely the
    illusion that produced a split snapshot in the first place.
    """
    coverage = session_coverage(_universe(fresh=3, stale=7), DECISION)
    assert coverage[pd.Timestamp("2026-08-26")] == 10
    assert coverage[pd.Timestamp("2026-08-27")] == 3


def test_coverage_stops_strictly_before_the_decision_date():
    """A decision is made for session D; D itself has not happened."""
    coverage = session_coverage(_universe(fresh=2, stale=0), pd.Timestamp("2026-08-27"))
    assert pd.Timestamp("2026-08-27") not in coverage.index
    assert coverage.index.max() == pd.Timestamp("2026-08-26")


def test_one_unusable_frame_cannot_cost_the_whole_universe():
    """Frames the boundary primitives reject are excluded downstream anyway.

    Raising here would abort the count -- and therefore the run -- over a single
    symbol that was already going to be dropped with a reason.
    """
    histories = _universe(fresh=2, stale=0)
    histories["BADINDEX"] = pd.DataFrame({"Close": [1.0, 2.0]}, index=["not", "dates"])
    histories["DUPES"] = _history(
        [1.0, 2.0, 3.0, 4.0],
        pd.to_datetime(["2026-08-24", "2026-08-24", "2026-08-26", "2026-08-27"]),
    )
    histories["EMPTY"] = pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))
    coverage = session_coverage(histories, DECISION)
    assert coverage[pd.Timestamp("2026-08-27")] == 2


def test_the_boundary_is_the_newest_COVERED_session_not_the_newest_one():
    """The 28 Aug 2026 shape exactly: 39% carry the newest session, 100% the prior.

    Recency loses to coverage. Picking 27 Aug here is what shipped the split file.
    """
    boundary = global_information_boundary(
        _universe(fresh=39, stale=61), universe_size=100,
        decision_date=DECISION, min_coverage_pct=98.0,
    )
    assert boundary == pd.Timestamp("2026-08-26")


def test_the_newest_session_wins_once_it_is_covered():
    """The rule is not 'always fall back'; it is 'fall back only while thin'."""
    boundary = global_information_boundary(
        _universe(fresh=99, stale=1), universe_size=100,
        decision_date=DECISION, min_coverage_pct=98.0,
    )
    assert boundary == pd.Timestamp("2026-08-27")


def test_the_requirement_rounds_up():
    """98% of 100 is 98, and 98 symbols must not be 'nearly enough'.

    Rounding down would admit a session one symbol short of the very tolerance
    the publisher enforces a few lines later, so the run would pass the boundary
    check and then fail to publish.
    """
    assert global_information_boundary(
        _universe(fresh=98, stale=2), universe_size=100,
        decision_date=DECISION, min_coverage_pct=98.0,
    ) == pd.Timestamp("2026-08-27")
    assert global_information_boundary(
        _universe(fresh=97, stale=3), universe_size=100,
        decision_date=DECISION, min_coverage_pct=98.0,
    ) == pd.Timestamp("2026-08-26")


def test_a_universe_that_never_qualifies_fails_with_the_numbers_in_it():
    """Failing loudly is the point; failing uninformatively is not.

    The two causes -- the provider is still backfilling, or symbols have gone
    permanently stale -- need opposite responses (wait, versus fix the universe),
    and the message has to carry enough to tell them apart.
    """
    histories = {f"S{i}": _history([None, None, None, None]) for i in range(5)}
    histories["ONE"] = _history([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError) as excinfo:
        global_information_boundary(histories, universe_size=6,
                                    decision_date=DECISION, min_coverage_pct=98.0)
    message = str(excinfo.value)
    assert "2026-08-27" in message and "98.0%" in message and "covers 1" in message


def test_no_session_at_all_is_a_refusal_not_an_empty_answer():
    histories = {"A": _history([None, None, None, None])}
    with pytest.raises(ValueError, match="No session carrying a Close"):
        global_information_boundary(histories, universe_size=1,
                                    decision_date=DECISION, min_coverage_pct=98.0)


def test_an_empty_universe_cannot_define_a_boundary():
    """Guarding this keeps a zero denominator from making every session qualify."""
    with pytest.raises(ValueError, match="Universe size must be positive"):
        global_information_boundary(_universe(fresh=1, stale=0), universe_size=0,
                                    decision_date=DECISION, min_coverage_pct=98.0)


def test_the_28_aug_2026_split_end_to_end():
    """Choosing the boundary and applying it, on the shape that shipped broken.

    Selection and application are tested apart above; this is the join, because
    the defect only appears when both run. 1,505 symbols, 536 carrying the newest
    session and 968 not -- the file that actually published. The boundary lands
    one session back and every symbol comes out on it, including the 536 the
    provider had already updated.
    """
    from scripts.real_data_audit import MAX_UNIVERSE_LOSS_PCT, build_universe_snapshots

    histories = _universe(fresh=536, stale=968)
    histories["JBCHEPHARM"] = _history([10.0, None, None, None])  # long-stale, 1 symbol
    universe_size = len(histories)
    assert universe_size == 1505

    boundary = global_information_boundary(
        histories, universe_size, DECISION, 100.0 - MAX_UNIVERSE_LOSS_PCT
    )
    assert boundary == pd.Timestamp("2026-08-26"), "recency must lose to coverage"

    snapshots, unavailable = build_universe_snapshots(
        list(histories), histories, DECISION, boundary
    )

    assert {s.latest_completed_session for s in snapshots.values()} == {boundary}
    assert len(snapshots) == 1504
    assert [name for name, _ in unavailable] == ["JBCHEPHARM"]
    # And the loss it leaves must clear the publisher's own tolerance, since the
    # boundary was chosen from that very number.
    assert len(unavailable) / universe_size * 100 <= MAX_UNIVERSE_LOSS_PCT
