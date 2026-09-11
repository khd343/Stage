"""Check the v2.3 detector against footprints published in the source.

The source records a handful of bases in its own shorthand — weeks, deepest
correction, tightest correction, contraction count. Two of those stocks are
still listed with history covering the period, so the detector can be measured
against a reading made by the method's author rather than against our own
fixtures.

Run locally, on demand, with network access:

    PYTHONPATH=. python scripts/validate_vcp_footprints.py

It once had its own workflow. That was removed on 2026-09-11: every detector it
checks belongs to an investigation the decision log closed (D-2.3.2, "five
attempts is enough"), so the run FAILS BY DESIGN, and a manual red button that
does not say so is a trap -- its first-ever run was mistaken for a production
defect. The script stays because the log says a sixth attempt should start from
evidence rather than from prose, and this is the evidence.

The breakout dates in the source are given to the month, so rather than guess a
session this scans candidate decision dates across the month and reports the
footprint at each. A match on any session in the stated month is a pass; the
source's reader was looking at a chart, not a specific close.
"""
from __future__ import annotations

import sys

import pandas as pd

from rs_stages.data import download_yfinance_history, normalize_session_index
from rs_stages.quant import footprint_label, vcp_footprint

#: symbol -> (breakout month per the source, expected footprint)
CASES = {
    "MELI": ("2007-12", {"weeks": 6, "deepest": 32, "tightest": 6, "contractions": 3}),
    "NFLX": ("2009-10", {"weeks": 27, "deepest": 27, "tightest": 7, "contractions": 3}),
}

#: How far a computed figure may sit from the source's and still count. Its
#: numbers were read off a chart by eye, so demanding exactness would be
#: measuring the wrong thing — but a wide tolerance would let anything pass.
TOL = {"weeks": 3.0, "deepest": 5.0, "tightest": 3.0, "contractions": 1}


def fetch(symbol: str, month: str) -> pd.DataFrame:
    """Daily history spanning the base and its breakout."""
    stop = pd.Timestamp(month) + pd.offsets.MonthEnd(1)
    start = stop - pd.Timedelta(weeks=90)          # 65-week base plus headroom
    import yfinance as yf

    frame = yf.download(
        symbol, start=start, end=stop + pd.Timedelta(days=5),
        auto_adjust=True, progress=False, actions=False,
    )
    if frame is None or frame.empty:
        raise ValueError(f"no history for {symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return normalize_session_index(frame)


#: Thresholds to sweep. A fixed 1.5% shattered a 27% decline into fragments on
#: real data, so the question is whether ANY fixed value reproduces the source's
#: counts, or whether the sensitivity has to tighten across the base as the
#: contractions themselves do.
SWEEP = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]


def sweep(symbol: str, month: str, want: dict, data: pd.DataFrame) -> None:
    """Report the footprint each threshold produces, to locate the structure."""
    sessions = [d for d in data.index if str(d)[:7] == month]
    label = f"{want['weeks']}W {want['deepest']}/{want['tightest']} {want['contractions']}T"
    print(f"  sweep against source {label}:")
    print(f"    {'thr':>5}  {'best session':>12}  {'footprint':>18}  {'wk':>5} {'deep':>6} {'tight':>6} {'T':>4}")
    for thr in SWEEP:
        best, best_score = None, None
        for t in sessions:
            fp = vcp_footprint(data["High"], data["Low"], t, thr)
            if not pd.notna(fp["Base_Weeks"]):
                continue
            score = abs(fp["Contractions"] - want["contractions"]) * 3 + abs(
                fp["Deepest_Pct"] - want["deepest"]
            )
            if best_score is None or score < best_score:
                best, best_score = (t, fp), score
        if best is None:
            print(f"    {thr:5.1f}  {'(no base)':>12}")
            continue
        t, fp = best
        mark = " <-" if fp["Contractions"] == want["contractions"] else ""
        print(
            f"    {thr:5.1f}  {str(t.date()):>12}  {footprint_label(fp):>18}  "
            f"{fp['Base_Weeks']:5.1f} {fp['Deepest_Pct']:6.1f} {fp['Tightest_Pct']:6.1f} "
            f"{fp['Contractions']:4d}{mark}"
        )


def main() -> None:
    failures: list[str] = []
    for symbol, (month, want) in CASES.items():
        label = f"{want['weeks']}W {want['deepest']}/{want['tightest']} {want['contractions']}T"
        print(f"\n=== {symbol} — source footprint {label}, breakout {month} ===")
        try:
            data = fetch(symbol, month)
        except Exception as exc:                      # noqa: BLE001
            failures.append(f"{symbol}: fetch failed ({type(exc).__name__}: {exc})")
            print(f"  FETCH FAILED: {exc}")
            continue

        sessions = [d for d in data.index if str(d)[:7] == month]
        print(f"  {len(data)} sessions fetched; scanning {len(sessions)} in {month}")
        sweep(symbol, month, want, data)
        print()

        best, best_score = None, None
        for t in sessions:
            fp = vcp_footprint(data["High"], data["Low"], t)
            if not pd.notna(fp["Base_Weeks"]):
                continue
            score = (
                abs(fp["Base_Weeks"] - want["weeks"])
                + abs(fp["Deepest_Pct"] - want["deepest"])
                + abs(fp["Tightest_Pct"] - want["tightest"])
                + abs(fp["Contractions"] - want["contractions"]) * 3
            )
            if best_score is None or score < best_score:
                best, best_score = (t, fp), score

        if best is None:
            failures.append(f"{symbol}: detector found no qualifying base in {month}")
            print("  NO BASE FOUND in the stated month")
            continue

        t, fp = best
        print(f"  closest session {t.date()} -> {footprint_label(fp)}")
        print(
            f"    weeks {fp['Base_Weeks']:.1f} (want {want['weeks']})  "
            f"deepest {fp['Deepest_Pct']:.1f} (want {want['deepest']})  "
            f"tightest {fp['Tightest_Pct']:.1f} (want {want['tightest']})  "
            f"contractions {fp['Contractions']} (want {want['contractions']})"
        )
        for key, got in (
            ("weeks", fp["Base_Weeks"]),
            ("deepest", fp["Deepest_Pct"]),
            ("tightest", fp["Tightest_Pct"]),
            ("contractions", fp["Contractions"]),
        ):
            if abs(got - want[key]) > TOL[key]:
                failures.append(
                    f"{symbol}: {key} {got:.1f} vs source {want[key]} "
                    f"(tolerance {TOL[key]})"
                )

    print("\n" + "=" * 62)
    if failures:
        print("MISMATCHES — the detector disagrees with the source's reading:")
        for f in failures:
            print(f"  - {f}")
        print("\nThis is information, not necessarily a defect: the source read")
        print("these off charts by eye. Report it; do not widen TOL to make it pass.")
        sys.exit(1)
    print("Detector reproduces every source footprint within tolerance.")


if __name__ == "__main__":
    main()
