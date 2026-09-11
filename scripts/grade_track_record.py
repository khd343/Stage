"""Grade every archived snapshot that has aged enough, and append the results.

Run monthly. Idempotent: a (snapshot, horizon, cohort) already in the record
is never recomputed, so running twice writes nothing the second time. One
bulk price download covers every pending snapshot at once -- the union of
their symbols over the span from the oldest pending session to today -- so a
backlog of sixty snapshots costs one audit's worth of fetching, not sixty.

A symbol Yahoo no longer knows (delisted) is a coverage loss and is counted as
such; the strict acquisition path used by the audit would abort on it instead,
which here would mean never grading a cohort that contained a failure.

Usage:
    PYTHONPATH=. python scripts/grade_track_record.py [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from rs_stages.data import INDEX_TICKERS, download_index_history
from rs_stages.track_record import (COHORTS, HORIZONS_WEEKS, KEY, TRACK_COLUMNS, append_only,
                                    grade_snapshot, gradeable, read_snapshot, snapshot_date)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "data" / "snapshots"
RECORD = ROOT / "data" / "track_record.csv"
BENCHMARK = INDEX_TICKERS["NIFTY_500"]


def fetch_closes(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
                 batch: int = 100) -> dict[str, pd.Series]:
    """Adjusted closes per symbol, tolerating names the provider has dropped."""
    import yfinance as yf
    out: dict[str, pd.Series] = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download([s + ".NS" for s in chunk], start=start, end=end + pd.Timedelta(days=1),
                             auto_adjust=True, progress=False, threads=True)
        close = df["Close"] if "Close" in df else pd.DataFrame()
        for s in chunk:
            col = s + ".NS"
            out[s] = close[col].dropna() if col in close.columns else pd.Series(dtype=float)
    return out


def pending(record: pd.DataFrame | None, calendar: pd.DatetimeIndex) -> list[tuple[Path, int]]:
    """(snapshot file, horizon) pairs old enough to grade and not yet fully graded."""
    done: set[tuple[str, int]] = set()
    if record is not None and not record.empty:
        counts = record.groupby(["snapshot_date", "horizon_weeks"]).size()
        done = {(d, int(h)) for (d, h), n in counts.items() if n >= len(COHORTS)}
    todo = []
    for path in sorted(SNAPSHOTS.glob("*.csv.gz")):
        day = snapshot_date(path)
        for weeks in HORIZONS_WEEKS:
            if (day.date().isoformat(), weeks) in done:
                continue
            if gradeable(day, weeks, calendar):
                todo.append((path, weeks))
    return todo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    record = pd.read_csv(RECORD, dtype={"snapshot_date": str, "cohort": str}) if RECORD.exists() else None
    if not SNAPSHOTS.exists() or not any(SNAPSHOTS.glob("*.csv.gz")):
        print("No archived snapshots yet; nothing to grade.")
        return 0

    today = pd.Timestamp(date.today())
    oldest = min(snapshot_date(p) for p in SNAPSHOTS.glob("*.csv.gz"))
    benchmark = download_index_history(BENCHMARK, start=oldest - pd.Timedelta(days=7), end=today + pd.Timedelta(days=1))["Close"].astype(float)
    calendar = pd.DatetimeIndex(benchmark.dropna().index)

    todo = pending(record, calendar)
    if not todo:
        nxt = min((snapshot_date(p) + pd.Timedelta(weeks=min(HORIZONS_WEEKS)) for p in SNAPSHOTS.glob("*.csv.gz")), default=None)
        print(f"Nothing gradeable yet. Earliest horizon closes around {nxt.date() if nxt is not None else '-'}.")
        return 0

    frames = {path: read_snapshot(path) for path, _ in todo if path not in {}}
    symbols = sorted({s for f in frames.values() for s in f["Symbol"].astype(str)})
    span_start = min(snapshot_date(p) for p, _ in todo) - pd.Timedelta(days=3)
    print(f"Grading {len(todo)} snapshot-horizon pair(s) over {len(symbols)} symbols from {span_start.date()}")
    closes = fetch_closes(symbols, span_start, today)

    rows = [grade_snapshot(frames[path], closes, benchmark, snapshot_date(path), weeks, today.date())
            for path, weeks in todo]
    new = pd.concat(rows, ignore_index=True)
    merged = append_only(record, new)
    added = len(merged) - (0 if record is None else len(record))
    print(f"Appended {added} row(s); record now {len(merged)} rows.")
    for _, r in new[new.cohort != "universe"].iterrows():
        print(f"  {r.snapshot_date} {r.horizon_weeks:>2}w {r.cohort:<18} n={r.n:<4} priced={r.n_priced:<4} "
              f"median {r.median_return_pct:+.2f}%  vs universe {r.excess_vs_universe_pp:+.2f}pp  vs Nifty500 {r.excess_vs_benchmark_pp:+.2f}pp")
    if args.dry_run:
        print("dry run; record not written")
        return 0
    merged.to_csv(RECORD, index=False, lineterminator="\n")
    print(f"wrote {RECORD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
