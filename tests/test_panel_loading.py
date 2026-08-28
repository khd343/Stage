"""The panel must load without Arrow, lazily, and never silently disagree.

The published panel lives outside the repository, so it can be missing, stale or
unreadable. Every one of those must degrade to an explicit message rather than a
wrong chart or a crashed app.
"""
import io

import numpy as np
import pandas as pd
import pytest

from rs_stages import data
from rs_stages.ui import loaders
from rs_stages.ui.loaders import PricePanel, _read_panel, load_price_panel, panel_matches

DATES = pd.bdate_range("2026-01-01", periods=40)
SYMBOLS = ("AAA", "BBB", "CCC")


def _grid(end=None) -> bytes:
    rng = np.random.default_rng(4)
    close = (100 + rng.normal(0, 1, (len(DATES), len(SYMBOLS))).cumsum(axis=0)).astype("float32")
    close[0, 2] = np.nan  # a symbol that started late
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        close=close,
        symbols=np.array(SYMBOLS, dtype="U32"),
        dates=pd.DatetimeIndex(DATES if end is None else end).to_numpy().astype("datetime64[D]"),
    )
    return buffer.getvalue()


def _research(date="2026-02-25") -> pd.DataFrame:
    return pd.DataFrame({"Symbol": list(SYMBOLS), "Date": [date] * len(SYMBOLS)})


def test_panel_reads_with_numpy_only():
    """No Arrow runtime is involved in reading a panel."""
    panel = _read_panel(io.BytesIO(_grid()))
    assert isinstance(panel, PricePanel)
    assert panel.symbols == SYMBOLS
    assert len(panel.dates) == len(DATES)
    assert panel.close.dtype == np.float32


def test_series_drops_missing_observations_rather_than_filling_them():
    panel = _read_panel(io.BytesIO(_grid()))
    late = panel.series("CCC")
    assert late is not None
    # The symbol has no observation on the first session; it is absent, not zero.
    assert len(late) == len(DATES) - 1
    assert late.index.min() == DATES[1]


def test_unknown_symbol_returns_none_not_an_empty_chart():
    panel = _read_panel(io.BytesIO(_grid()))
    assert panel.series("NOPE") is None


def test_tails_skip_symbols_without_enough_history():
    panel = _read_panel(io.BytesIO(_grid()))
    tails = panel.tails(1)
    # One observation cannot make a trend line.
    assert tails == {}
    assert set(panel.tails(10)) == set(SYMBOLS)


def test_terminal_session_is_the_last_date():
    panel = _read_panel(io.BytesIO(_grid()))
    assert panel.terminal_session == pd.Timestamp(DATES[-1])


def test_a_panel_from_a_different_session_is_rejected():
    """A chart and a table must never describe different sessions."""
    panel = _read_panel(io.BytesIO(_grid()))
    assert panel_matches(panel, _research(DATES[-1].strftime("%Y-%m-%d"))) is None
    reason = panel_matches(panel, _research("2026-03-10"))
    assert reason is not None
    assert "10 Mar 2026" in reason and "withheld" in reason


def test_local_file_is_preferred_over_the_network(tmp_path, monkeypatch):
    path = tmp_path / "price_panel.npz"
    path.write_bytes(_grid())
    monkeypatch.setattr(loaders, "PANEL_PATH", path)

    def explode(*a, **k):
        raise AssertionError("the network must not be touched when a local panel exists")

    monkeypatch.setattr(loaders.urllib.request, "urlopen", explode)
    panel, error = load_price_panel()
    assert error is None and panel is not None


def test_a_download_failure_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "PANEL_PATH", tmp_path / "absent.npz")

    def fail(*a, **k):
        raise loaders.urllib.error.URLError("no route to host")

    monkeypatch.setattr(loaders.urllib.request, "urlopen", fail)
    panel, error = load_price_panel()
    assert panel is None
    assert "could not be downloaded" in error
    # The rest of the terminal is explicitly said to be unaffected.
    assert "unaffected" in error


def test_a_corrupt_download_is_reported_not_raised(tmp_path, monkeypatch):
    monkeypatch.setattr(loaders, "PANEL_PATH", tmp_path / "absent.npz")

    class Response:
        def read(self):
            return b"this is not a numpy archive"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(loaders.urllib.request, "urlopen", lambda *a, **k: Response())
    panel, error = load_price_panel()
    assert panel is None
    assert "could not be read" in error


def test_an_unreadable_local_panel_is_reported_not_raised(tmp_path, monkeypatch):
    path = tmp_path / "price_panel.npz"
    path.write_bytes(b"garbage")
    monkeypatch.setattr(loaders, "PANEL_PATH", path)
    panel, error = load_price_panel()
    assert panel is None
    assert "local price panel could not be read" in error


def test_snapshot_no_longer_carries_the_panel():
    """The panel must not ride along in the value cache_data serialises."""
    assert "panel" not in {f for f in loaders.Snapshot.__dataclass_fields__}


def test_the_ui_universe_is_the_analytical_universe():
    """The header count and the analysed count must describe the same set.

    NSE reserves DUMMY-prefixed rows in the constituent CSV for corporate
    actions. The audit excludes them before computing anything, so a UI that
    read the CSV raw would advertise a universe larger than every figure
    beneath it.

    This once asserted the shipped CSV *contained* DUMMY rows, which was true
    only while the universe was NSE's own download. This repo's universe is
    built from a different source and carries none, so that assertion pinned
    the file rather than the behaviour and failed the moment the universe
    changed. The guarantee is restored below without depending on the contents
    of any particular universe file.
    """
    snapshot = loaders.load_snapshot()
    symbols = snapshot.universe["Symbol"].astype(str)
    assert not symbols.str.startswith("DUMMY").any()
    # Every analysed symbol is a member of the universe the header reports.
    assert set(snapshot.research["Symbol"].astype(str)).issubset(set(symbols))
    # Nothing is dropped for any reason OTHER than the exclusion: with no DUMMY
    # rows present, the raw row count and the analytical one must agree exactly.
    raw = pd.read_csv(loaders.UNIVERSE_PATH)
    reserved = int(raw["Symbol"].astype(str).str.startswith("DUMMY").sum())
    assert len(raw) - reserved == len(snapshot.universe)


def test_the_ui_reads_the_universe_through_the_excluding_loader():
    """The exclusion is a property of the loader, not of the shipped file.

    A universe file with no DUMMY rows cannot demonstrate that the exclusion
    still runs, so the check that survives a change of universe is that the UI
    calls the same locked loader the audit does. That loader's behaviour is
    pinned by tests/test_data.py; reading UNIVERSE_PATH with a bare
    pd.read_csv here instead would silently reintroduce the reserved rows.
    """
    assert loaders.load_nse_constituents_csv is data.load_nse_constituents_csv
