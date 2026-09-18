"""Corporate actions arriving in the price series as though money evaporated.

NSE applies circuit bands of 2, 5, 10 or 20 percent to what it lists. A single
session that moves 40 or 70 percent therefore did not happen as a price move.
It is a split, a bonus issue, or a demerger.

Two causes, and only one of them is a data bug:

  SPLIT / BONUS     yfinance's auto_adjust is meant to restate the whole history
                    so no jump ever appears. A jump that appears anyway means
                    the adjustment did not reach this series.
  DEMERGER          auto_adjust does NOT handle these. The parent's price truly
                    falls because value has left it, but the holder received
                    shares in the new entity, so no economic loss occurred.

This module DETECTS and CLASSIFIES. It deliberately never rewrites a price.
Telling the two causes apart needs information the price series does not carry,
and a wrong correction applied silently is worse than a flagged anomaly. Every
consumer here either reports the flag or declines to compute a number across
it.

Measured on this repo's own published panel (342 sessions, 750 symbols): six
events, five of them matching no clean split ratio. On the 17 Sep 2026 snapshot
four still sat inside their 52-week window, and all four read "Stage 4 -
Declining, SELL" with relative strength in the bottom decile, having never
declined.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

#: Beyond this, one session is not a price move. NSE's widest ordinary band is
#: 20 percent; the threshold sits well clear of it so a violent but genuine
#: session is never called an action.
IMPLAUSIBLE_MOVE: float = 0.35

#: Ratio of today's close to yesterday's for the usual actions. A 1-for-5 split
#: leaves the price at a fifth, so the ratio is 0.2.
COMMON_ACTIONS: dict[float, str] = {
    0.1000: "1:10 split",
    0.1250: "1:8 split",
    0.2000: "1:5 split",
    0.2500: "1:4 split",
    0.3333: "1:3 split",
    0.4000: "2:5 split",
    0.5000: "1:2 split or 1:1 bonus",
    0.6667: "3:2 (1:2 bonus)",
    0.7500: "4:3 (1:3 bonus)",
    2.0000: "2:1 reverse split",
    3.0000: "3:1 reverse split",
    5.0000: "5:1 reverse split",
    10.000: "10:1 reverse split",
}

#: How close a ratio must sit to a clean action to be called one. Loose enough
#: to survive a day's genuine drift on top of the action, tight enough that an
#: arbitrary ratio is not labelled a split.
RATIO_TOLERANCE: float = 0.04

#: A universe-wide adjustment failure, not a market event.
#:
#: What this actually protects is the CROSS-SECTION. RS_Score is a percentile
#: over the universe, so a large enough block of fictional prices moves the
#: rank of every other name too, and flagging the affected rows cannot repair
#: that. A handful of bad rows, by contrast, is now marked and published, which
#: is strictly better than not publishing: aborting loses the session from the
#: record permanently, because the record never goes backwards.
#:
#: The ceiling is the audit's OWN existing tolerance for an unusable slice of
#: the universe (MAX_UNIVERSE_LOSS_PCT), pinned equal by a test, because one
#: idea should not carry two numbers. It was 0.5% until 2026-09-18, when a real
#: vendor correction unfroze five stale names in one session and came within
#: four of tripping it.
MASS_EVENT_FLOOR: int = 8
MASS_EVENT_PCT: float = 2.0

EVENT_COLUMNS = ["Symbol", "Date", "Move_Pct", "Ratio", "Prev_Close", "Close",
                 "Looks_Like", "Match_Gap", "Kind"]


def classify_ratio(ratio: float) -> tuple[str, float | None]:
    """Name the action a price ratio looks like, and how closely it matches.

    An unmatched ratio is reported as a possible demerger rather than forced
    into the nearest split: a demerger leaves no clean ratio, and mislabelling
    one as a split would invite a "fix" that corrupts the data further.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        return "unusable price", None
    label, gap = min(((name, abs(ratio - target) / target)
                      for target, name in COMMON_ACTIONS.items()),
                     key=lambda pair: pair[1])
    if gap <= RATIO_TOLERANCE:
        return label, gap
    return "unmatched - possible demerger or spin-off", gap


def _ratios(close: pd.Series) -> pd.Series:
    """Successive close-to-close ratios over the sessions that have a price.

    Blank rows are closed up first. Stage's histories carry NaN sessions, and
    treating one as a price would read every gap as a total loss.
    """
    clean = pd.to_numeric(pd.Series(close), errors="coerce").sort_index().dropna()
    if len(clean) < 2:
        return pd.Series(dtype=float)
    return clean / clean.shift(1)


def implausible_sessions(close: pd.Series, threshold: float = IMPLAUSIBLE_MOVE,
                         since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Sessions in one symbol whose move is too large to be a price move.

    `since` filters the EVENTS, not the input: a jump on the first session of
    the window still needs its predecessor to be measured at all.
    """
    ratio = _ratios(close)
    if ratio.empty:
        return pd.DataFrame(columns=[c for c in EVENT_COLUMNS if c != "Symbol"])
    clean = pd.to_numeric(pd.Series(close), errors="coerce").sort_index().dropna()
    flagged = ratio[(ratio - 1.0).abs() > threshold]
    rows = []
    for stamp, value in flagged.items():
        if not np.isfinite(value) or value <= 0:
            continue
        if since is not None and stamp < pd.Timestamp(since):
            continue
        label, gap = classify_ratio(float(value))
        position = clean.index.get_loc(stamp)
        rows.append({
            "Date": pd.Timestamp(stamp),
            "Move_Pct": round((float(value) - 1.0) * 100.0, 2),
            "Ratio": round(float(value), 6),
            "Prev_Close": round(float(clean.iloc[position - 1]), 4),
            "Close": round(float(clean.iloc[position]), 4),
            "Looks_Like": label,
            "Match_Gap": None if gap is None else round(gap, 4),
            "Kind": "split/bonus" if gap is not None and gap <= RATIO_TOLERANCE else "unclassified",
        })
    return pd.DataFrame(rows, columns=[c for c in EVENT_COLUMNS if c != "Symbol"])


def scan(closes: Mapping[str, pd.Series], threshold: float = IMPLAUSIBLE_MOVE,
         since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Every suspect session across many symbols, worst first."""
    frames = []
    for symbol in sorted(closes):
        found = implausible_sessions(closes[symbol], threshold=threshold, since=since)
        if not found.empty:
            found.insert(0, "Symbol", str(symbol))
            frames.append(found)
    if not frames:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    order = out["Move_Pct"].abs().sort_values(ascending=False, kind="stable").index
    return out.loc[order].reset_index(drop=True)[EVENT_COLUMNS]


def spans_discontinuity(close: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                        threshold: float = IMPLAUSIBLE_MOVE) -> bool:
    """Does a return measured from `start` to `end` cross a corporate action?

    The interval is half-open at the start on purpose. An action ON the start
    session leaves both endpoints on the post-action basis, so the return is
    sound; only an action strictly after it puts the two ends on different
    bases. An unknown series is reported clean, because a price that does not
    exist is already counted as a coverage hole rather than a bad number.
    """
    ratio = _ratios(close)
    if ratio.empty:
        return False
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    window = ratio[(ratio.index > start) & (ratio.index <= end)]
    return bool(((window - 1.0).abs() > threshold).any())


def is_mass_failure(count: int, universe_size: int) -> bool:
    """Too many actions on one session to be actions at all."""
    ceiling = max(MASS_EVENT_FLOOR, math.ceil(MASS_EVENT_PCT / 100.0 * max(universe_size, 0)))
    return int(count) >= ceiling
