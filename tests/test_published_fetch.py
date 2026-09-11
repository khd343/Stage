"""The terminal reads its data from GitHub at runtime, atomically, with fallback.

The deployed checkout changes only on a container rebuild (D-2.1.7), which is
slow, sometimes skipped, and bypassed when the app wakes from sleep -- so a
terminal reading the checkout shows yesterday's session until something
restarts it. These pin the fix: remote reads happen only when asked, the three
audit outputs come from ONE source (never a mix of vintages), any remote
failure falls back to the checkout, and the app's caches are short-lived or
keyed on the snapshot's date so no process can hold stale data.
"""
from __future__ import annotations

import io
import pathlib
import re
import urllib.error

import pandas as pd

from rs_stages.ui import loaders

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve(mapping: dict[str, bytes]):
    """A fake urlopen serving bytes by URL tail, raising for anything else."""
    def urlopen(url, timeout=None):
        tail = url.rsplit("/", 1)[-1]
        if tail not in mapping:
            raise urllib.error.URLError(f"no {tail}")
        return _Response(mapping[tail])
    return urlopen


def _local_bytes(name: str) -> bytes:
    return (loaders.DATA_DIR / name).read_bytes()


def test_the_default_never_touches_the_network(monkeypatch):
    """Tests and offline work read the checkout; a remote call here is a bug."""
    def boom(*a, **k):
        raise AssertionError("network touched with remote=False")
    monkeypatch.setattr(loaders.urllib.request, "urlopen", boom)
    snap = loaders.load_snapshot()
    assert snap.source == "local"
    assert len(snap.research) > 0


def test_remote_reads_come_from_github_and_say_so(monkeypatch):
    served = {n: _local_bytes(n) for n in ("latest_research.csv", "previous_research.csv", "breadth_history.csv")}
    monkeypatch.setattr(loaders.urllib.request, "urlopen", _serve(served))
    snap = loaders.load_snapshot(remote=True)
    assert snap.source == "remote"
    assert len(snap.research) > 0 and snap.previous is not None and snap.breadth is not None


def test_a_remote_failure_falls_back_to_the_whole_local_set(monkeypatch):
    """One missing remote file must not produce a remote/local mix of vintages.

    Research fetched today beside a previous-session file from an older
    checkout would compare sessions the audit never paired.
    """
    served = {"latest_research.csv": _local_bytes("latest_research.csv")}   # previous, breadth absent
    monkeypatch.setattr(loaders.urllib.request, "urlopen", _serve(served))
    snap = loaders.load_snapshot(remote=True)
    assert snap.source == "local", "any remote failure sends the whole set to the checkout"
    assert snap.previous is not None


def test_remote_can_be_disabled_by_an_empty_url(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network touched with DATA_URL empty")
    monkeypatch.setattr(loaders.urllib.request, "urlopen", boom)
    monkeypatch.setattr(loaders, "DATA_URL", "")
    assert loaders.load_snapshot(remote=True).source == "local"


def test_remote_bytes_are_what_gets_read(monkeypatch):
    """Not merely 'remote succeeded': the frame must come from the served bytes."""
    df = pd.read_csv(loaders.DATA_DIR / "latest_research.csv")
    df = df.head(5).copy()
    df["Close"] = 99999.0
    served = {
        "latest_research.csv": df.to_csv(index=False).encode(),
        "previous_research.csv": _local_bytes("previous_research.csv"),
        "breadth_history.csv": _local_bytes("breadth_history.csv"),
    }
    monkeypatch.setattr(loaders.urllib.request, "urlopen", _serve(served))
    snap = loaders.load_snapshot(remote=True)
    assert len(snap.research) == 5 and (snap.research["Close"] == 99999.0).all()


# ---- the app's caches: short-lived, or keyed on the snapshot's date ----------

def test_the_snapshot_cache_is_short_lived_and_reads_remotely():
    m = re.search(r"@st\.cache_data\(ttl=(\w+)[^)]*\)\s*\ndef cached_snapshot\(\):(.*?)\n\n", APP, re.S)
    assert m, "cached_snapshot must be a cache_data with a ttl"
    ttl = m.group(1)
    seconds = int(ttl) if ttl.isdigit() else int(re.search(rf"^{ttl}\s*=\s*(\d+)", APP, re.M).group(1))
    assert seconds <= 600, f"snapshot ttl is {seconds}s; the point is freshness"
    assert "load_snapshot(remote=True)" in m.group(2)


def test_panel_and_sparkline_caches_are_keyed_on_the_snapshot_date_with_a_ttl():
    """A panel cached without a key or a ttl is held for the process's life.

    That is how every chart used to vanish after the snapshot moved: the old
    panel failed panel_matches against the new snapshot and nothing ever
    fetched another. The session is the key; the ttl bounds a stale process.
    """
    for name in ("_cached_panel", "_cached_sparklines"):
        m = re.search(rf"@st\.cache_resource\(([^)]*)\)\s*\ndef {name}\(([^)]*)\):", APP)
        assert m, f"{name} must be a cache_resource"
        assert "ttl=" in m.group(1), f"{name} has no ttl"
        assert "decision_key" in m.group(2), f"{name} is not keyed on the snapshot date"
    for wrapper in ("def cached_panel():", "def cached_sparklines("):
        assert wrapper in APP, "callers keep the unkeyed name; the wrapper supplies the key"
    assert "_decision_key()" in APP
