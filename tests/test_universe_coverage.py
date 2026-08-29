"""The audit must survive a bad symbol without publishing a distorted universe.

The run of 25 Aug 2026 aborted with "No completed market session exists before
decision date". Three tickers had timed out against the provider and one came
back with no rows, which raised out of the snapshot comprehension and killed the
whole 750-symbol audit before anything was published.

Two properties are pinned here, and they pull in opposite directions on purpose:
one symbol must not be able to fail the run, and a depleted universe must not be
able to publish quietly.
"""

import itertools
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from scripts.real_data_audit import (
    MAX_UNIVERSE_LOSS_PCT,
    build_universe_snapshots,
    enforce_universe_coverage,
)

NL = chr(10)
_TMP_COUNTER = itertools.count()


def tmp_snapshot(dates: list[str], header: str | None = None) -> pathlib.Path:
    """Write a throwaway snapshot CSV and return its path."""
    path = pathlib.Path(tempfile.gettempdir()) / f"rs_snap_{next(_TMP_COUNTER)}.csv"
    body = header if header is not None else "Date" + NL + "".join(d + NL for d in dates)
    path.write_text(body, encoding="utf-8")
    return path


DECISION = pd.Timestamp("2026-08-25")
BOUNDARY = pd.Timestamp("2026-08-24")


def _history(sessions: int = 30, last: str = "2026-08-24") -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=sessions)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, sessions),
            "High": np.linspace(101, 111, sessions),
            "Low": np.linspace(99, 109, sessions),
            "Close": np.linspace(100, 110, sessions),
            "Volume": np.full(sessions, 1_000.0),
        },
        index=idx,
    )


def test_a_symbol_with_no_rows_is_excluded_rather_than_fatal():
    """The exact shape of the failure: an empty frame from a failed download."""
    histories = {"GOOD": _history(), "BELRISE": _history().iloc[0:0]}

    snapshots, unavailable = build_universe_snapshots(
        ["GOOD", "BELRISE"], histories, DECISION, BOUNDARY
    )

    assert set(snapshots) == {"GOOD"}
    assert [name for name, _ in unavailable] == ["BELRISE"]


def test_a_symbol_absent_from_the_download_is_excluded_with_its_reason():
    snapshots, unavailable = build_universe_snapshots(
        ["GOOD", "NEVER_FETCHED"], {"GOOD": _history()}, DECISION, BOUNDARY
    )

    assert set(snapshots) == {"GOOD"}
    assert unavailable == [("NEVER_FETCHED", "the provider returned no history")]


def test_every_exclusion_carries_a_reason():
    """A silent drop is the failure mode this guard exists to prevent."""
    _, unavailable = build_universe_snapshots(
        ["A", "B"], {"A": _history().iloc[0:0]}, DECISION, BOUNDARY
    )

    assert len(unavailable) == 2
    assert all(reason.strip() for _, reason in unavailable)


def test_losing_a_stray_symbol_is_tolerated():
    """One delisting in a 750-name universe is 0.13% and must not stop a run."""
    enforce_universe_coverage(missing=1, total=750)


def test_losing_more_than_the_tolerance_refuses_to_publish():
    """RS_Score is cross-sectional: a depleted universe re-ranks the survivors."""
    missing = int(750 * MAX_UNIVERSE_LOSS_PCT / 100) + 1

    with pytest.raises(SystemExit, match="cross-sectional percentile"):
        enforce_universe_coverage(missing=missing, total=750)


def test_the_guard_holds_at_the_boundary():
    at_limit = int(750 * MAX_UNIVERSE_LOSS_PCT / 100)
    enforce_universe_coverage(missing=at_limit, total=750)

    with pytest.raises(SystemExit):
        enforce_universe_coverage(missing=at_limit + 1, total=750)


def test_an_empty_universe_does_not_divide_by_zero():
    enforce_universe_coverage(missing=0, total=0)


def test_a_symbol_the_provider_updated_early_is_truncated_not_ranked_ahead():
    """The half of the boundary that keeps fresh symbols IN, at the shared date.

    Yahoo settles the NSE universe unevenly, so on any given run some symbols
    already carry a session the rest do not. Excluding them would throw away
    good data; measuring them there would rank them on newer information than
    everyone else. Both are wrong. The data is cut at the boundary instead.
    """
    histories = {"NORMAL": _history(), "EARLY": _history(sessions=31, last="2026-08-25")}

    snapshots, unavailable = build_universe_snapshots(
        ["NORMAL", "EARLY"], histories, DECISION, BOUNDARY
    )

    assert unavailable == []
    assert snapshots["EARLY"].latest_completed_session == BOUNDARY
    assert snapshots["EARLY"].data.index.max() == BOUNDARY
    assert snapshots["NORMAL"].latest_completed_session == BOUNDARY


def test_a_symbol_the_provider_has_not_updated_is_excluded_not_slid_back():
    """The other half, and the one that produced the split file.

    Left to search its own history, build_decision_snapshot answers with the
    symbol's own last close -- a silent fallback that looks identical to a real
    measurement. It has to be an exclusion with a reason instead.
    """
    histories = {"NORMAL": _history(), "LAGGING": _history(sessions=25, last="2026-08-19")}

    snapshots, unavailable = build_universe_snapshots(
        ["NORMAL", "LAGGING"], histories, DECISION, BOUNDARY
    )

    assert set(snapshots) == {"NORMAL"}
    name, reason = unavailable[0]
    assert name == "LAGGING"
    assert "global boundary" in reason and "2026-08-24" in reason and "2026-08-19" in reason


def test_every_surviving_symbol_shares_one_date():
    """The invariant in one line: this is what LOCKED_SPEC 8.1 asks for."""
    histories = {
        "A": _history(),
        "B": _history(sessions=31, last="2026-08-25"),
        "C": _history(sessions=40),
    }

    snapshots, _ = build_universe_snapshots(list(histories), histories, DECISION, BOUNDARY)

    assert {s.latest_completed_session for s in snapshots.values()} == {BOUNDARY}


def test_the_published_boundary_is_read_from_the_snapshot_on_disk():
    """The floor that stops the record walking backwards."""
    import pandas as pd
    from scripts.real_data_audit import published_boundary

    path = tmp_snapshot(["2026-08-27", "2026-08-27", "2026-08-26"])
    assert published_boundary(path) == pd.Timestamp("2026-08-27")


def test_an_absent_or_unreadable_snapshot_imposes_no_floor():
    """A floor derived from a corrupt file is worse than no floor.

    The run is about to overwrite that file regardless, so refusing to publish
    on the strength of something unparseable would strand the record for good.
    """
    from pathlib import Path
    from scripts.real_data_audit import published_boundary

    assert published_boundary(Path("does-not-exist.csv")) is None
    assert published_boundary(tmp_snapshot([], header="Symbol\nAAA\n")) is None
    assert published_boundary(tmp_snapshot(["not-a-date"])) is None


def test_every_step_that_needs_artifacts_is_gated_on_the_audit_producing_them():
    """A correct no-op must not be reported as a broken audit.

    A run that refuses to move the record backwards writes no price panel, and
    `gh release upload` does not glob -- it fails with "no matches found". On
    29 Aug 2026 that turned a working refusal into a red X. Each step consuming
    an artifact must therefore ask whether one exists.
    """
    import pathlib as _pathlib

    text = (_pathlib.Path(__file__).resolve().parents[1]
            / ".github" / "workflows" / "real_data_audit.yml").read_text(encoding="utf-8")
    assert "id: audit" in text, "the audit step needs an id for later steps to read"
    consumers = [
        "Publish price panel as a release asset",
        "Publish validated research snapshot",
        "Upload research output",
    ]
    for name in consumers:
        head = text.index(name)
        window = text[head:head + 400]
        assert "steps.audit.outputs.published == 'true'" in window, (
            f"step {name!r} consumes an audit artifact but does not check one was produced")


def test_the_audit_signals_both_outcomes():
    """Signalling only success would leave the flag unset on a no-op.

    An unset output compares false, so gating would happen to work -- until
    someone inverts a condition. Both branches write the flag explicitly.
    """
    import pathlib as _pathlib

    src = (_pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "real_data_audit.py").read_text(encoding="utf-8")
    assert "signal_publication(False)" in src
    assert "signal_publication(True)" in src
