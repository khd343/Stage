"""Admit NSE names to the universe by LIQUIDITY, never by history depth.

The 200-session rule lives in the engine (high_52w / low_52w, min_sessions=200)
and stays there: a symbol short of it gets NaN for every metric that needs it,
proven on 14-session listings (price present, everything else blank). So the
universe file does not need to enforce history -- it needs to enforce the one
thing the engine cannot see per symbol: whether admitting a name will eat the
coverage budget. The global boundary requires 98% of the universe to carry a
close, and the publisher refuses above 2% loss. An illiquid name spends that
budget every day; a young liquid name spends none.

Measured 10 Sep 2026 on 395 NSE names outside the universe: 339 traded on 59-60
of the last 60 sessions, costing 7.7 expected absentees a day on a 1,849-name
universe -- 0.42% against the 2.00% ceiling. 287 of them already had 200+
sessions and were missing only because they were never in the source file the
universe was first built from. That is the gap this tool closes, and keeps
closed: run it again with any candidate list and it admits what is liquid.

Usage:
    python scripts/admit_universe.py --list "stock list.csv" [--dry-run]

The candidate list IS the universe's boundary: nothing outside it is ever
admitted. It needs columns company_id (EXCHANGE:SYMBOL), name, industry, where
industry holds one of the universe's own sector values. BSE-prefixed rows are
ignored: one exchange, one session calendar, one cross-section.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "ind_niftytotalmarket_list.csv"

#: Liquidity is presence over the sessions a name COULD have traded: the last
#: LIQUIDITY_WINDOW sessions of the observed calendar, clipped to start at the
#: name's own first session. A name listed 18 sessions ago that traded all 18 is
#: perfectly liquid; measured against a fixed 60 it could never pass, and the
#: first dry run rejected the entire 18 Aug 2026 NSE migration batch that way.
#: LIQUIDITY_MIN_SAMPLE stops a two-session listing proving itself on 2 of 2.
LIQUIDITY_WINDOW = 60
LIQUIDITY_RATIO = 0.90
LIQUIDITY_MIN_SAMPLE = 10


def candidates_from_list(path: Path) -> pd.DataFrame:
    """NSE rows of a candidate list as [Symbol, Company Name, Industry].

    Accepts either header spelling the list has used (company_id/name/industry
    or companyId/Name/Industry). TradingView writes BAJAJ_AUTO for what NSE
    lists as BAJAJ-AUTO; the underscore is normalised.
    """
    raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    cols = {c.lower(): c for c in raw.columns}
    idc = cols.get("company_id") or cols.get("companyid")
    namec = cols.get("name")
    indc = cols.get("industry")
    if not (idc and namec and indc):
        raise ValueError("candidate list needs company_id, name and industry columns")
    nse = raw[raw[idc].str.strip().str.upper().str.startswith("NSE:")].copy()
    out = pd.DataFrame({
        "Symbol": nse[idc].str.slice(4).str.strip().str.upper().str.replace("_", "-", regex=False),
        "Company Name": nse[namec].str.strip().str.replace(r"\s+Limited$", " Ltd", regex=True),
        "Industry": nse[indc].str.strip(),
    })
    out = out[out["Symbol"] != ""].drop_duplicates("Symbol")
    return out.reset_index(drop=True)


def session_calendar(closes: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Every date on which ANY candidate traded: the NSE calendar, observed.

    Built from the data rather than from business days so that a holiday is
    not counted against a name that could not have traded on it.
    """
    dates = set()
    for s in closes.values():
        dates.update(pd.DatetimeIndex(s.dropna().index))
    return pd.DatetimeIndex(sorted(dates))


def liquid(closes: dict[str, pd.Series], window: int = LIQUIDITY_WINDOW,
           ratio: float = LIQUIDITY_RATIO, min_sample: int = LIQUIDITY_MIN_SAMPLE,
           ) -> tuple[set[str], dict[str, tuple[int, int]]]:
    """Names that traded on at least `ratio` of the sessions available to them.

    For each name the denominator is the recent calendar clipped at its first
    observed session, so youth is not penalised and only absence is. Returns the
    admitted set and every candidate's (traded, possible) pair, so a refusal can
    be reported with its numbers rather than as a bare omission.
    """
    calendar = session_calendar(closes)
    recent = calendar[-window:]
    counts: dict[str, tuple[int, int]] = {}
    admitted: set[str] = set()
    for sym, s in closes.items():
        have = pd.DatetimeIndex(s.dropna().index)
        if len(have) == 0:
            counts[sym] = (0, 0)
            continue
        possible = recent[recent >= have.min()]
        traded = int(have.isin(possible).sum())
        counts[sym] = (traded, len(possible))
        if len(possible) >= min_sample and traded >= ratio * len(possible):
            admitted.add(sym)
    return admitted, counts


def merge(universe: pd.DataFrame, additions: pd.DataFrame,
          allow_sectors: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Add rows to the universe without touching what is already there.

    Refuses an unknown sector loudly: a new spelling would silently become an
    81st group in every sector view. A sector that is genuinely new -- the
    universe inherited a source with no bank or insurance category at all -- is
    admitted only by naming it in `allow_sectors`, so a vocabulary change is a
    declared decision and never a typo that got through. Existing rows are never
    modified: if the candidate list disagrees about an existing name's sector,
    the universe's value stands.
    """
    known = set(universe["Industry"]) | set(allow_sectors)
    new = additions[~additions["Symbol"].isin(set(universe["Symbol"]))].copy()
    bad = sorted(set(new["Industry"]) - known)
    if bad:
        raise ValueError(f"unknown sector value(s), refusing to add an 81st group: {bad}")
    new["Series"] = "EQ"
    new = new[list(universe.columns)]
    out = pd.concat([universe, new], ignore_index=True)
    out = out.sort_values("Symbol", kind="stable").reset_index(drop=True)
    if out["Symbol"].duplicated().any():
        raise ValueError("duplicate symbol after merge")
    return out


def fetch_closes(symbols: list[str], days: int = 120, batch: int = 100) -> dict[str, pd.Series]:
    """Recent closes per symbol, tolerating names Yahoo has never heard of.

    Deliberately NOT rs_stages.data.download_yfinance_histories, which is strict
    by design -- one dead ticker aborts it. Here a dead ticker is an expected
    input that must be reported, not a failure.
    """
    import yfinance as yf
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=days)
    out: dict[str, pd.Series] = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        with contextlib.redirect_stderr(io.StringIO()):
            df = yf.download([s + ".NS" for s in chunk], start=start, end=end,
                             auto_adjust=True, progress=False, threads=True)
        close = df["Close"] if "Close" in df else pd.DataFrame()
        for s in chunk:
            col = s + ".NS"
            out[s] = close[col].dropna() if col in close.columns else pd.Series(dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="candidate CSV: company_id,name,industry")
    ap.add_argument("--allow-sector", nargs="*", default=[],
                    help="sector value(s) NEW to the universe, admitted by explicit declaration")
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    universe = pd.read_csv(args.universe, encoding="utf-8")
    cands = candidates_from_list(Path(args.list))
    fresh = cands[~cands["Symbol"].isin(set(universe["Symbol"]))]
    print(f"universe {len(universe)} | NSE candidates {len(cands)} | not yet in universe {len(fresh)}")

    closes = fetch_closes(list(fresh["Symbol"]))
    admitted, counts = liquid(closes)
    nodata = sorted(s for s, (t, p) in counts.items() if p == 0)
    thin = sorted(s for s, (t, p) in counts.items() if p and s not in admitted)
    print(f"no NSE data: {len(nodata)} | not liquid: {len(thin)} | admitted: {len(admitted)}")
    if thin:
        print("  not liquid (traded/possible): "
              + ", ".join(f"{s}({counts[s][0]}/{counts[s][1]})" for s in thin))

    add = fresh[fresh["Symbol"].isin(admitted)]
    merged = merge(universe, add, frozenset(args.allow_sector))
    print(f"result: {len(universe)} -> {len(merged)} rows, {merged['Industry'].nunique()} sectors")
    if args.dry_run:
        print("dry run; universe file untouched")
        return
    merged.to_csv(args.universe, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {args.universe}")


if __name__ == "__main__":
    sys.exit(main())
