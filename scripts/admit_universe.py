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
    python scripts/admit_universe.py --list "stocks list.csv" [--dry-run]

The candidate list needs columns companyId (EXCHANGE:SYMBOL), Name, Industry,
where Industry holds one of the universe's own sector values. BSE-prefixed rows
are ignored: one exchange, one session calendar, one cross-section.
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
    """NSE rows of a candidate list as [Symbol, Company Name, Industry]."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("companyId", "Name", "Industry"):
        if col not in raw.columns:
            raise ValueError(f"candidate list lacks required column {col!r}")
    nse = raw[raw["companyId"].astype(str).str.startswith("NSE:")].copy()
    out = pd.DataFrame({
        "Symbol": nse["companyId"].str.slice(4).str.strip(),
        "Company Name": nse["Name"].astype(str).str.strip()
                        .str.replace(r"\s+Limited$", " Ltd", regex=True),
        "Industry": nse["Industry"].astype(str).str.strip(),
    })
    out = out[out["Symbol"] != ""].drop_duplicates("Symbol")
    return out.reset_index(drop=True)


NSE_EQUITY_LIST = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
#: Main-board series admitted. BZ is the trade-to-trade segment for regulatory
#: defaulters; SME listings are on a separate board and a separate list. One
#: exchange, one board, one cross-section.
NSE_SERIES = ("EQ", "BE")
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Referer": "https://www.nseindia.com/",
}


def fetch_nse_equity_list(url: str = NSE_EQUITY_LIST) -> pd.DataFrame:
    """NSE's own complete main-board equity list: the authoritative candidate set.

    Every hand-maintained candidate list tried before this surfaced a different
    partial slice -- one found 503 names, three together found 839 of which 705
    were BSE codes needing resolution and 109 were dead tickers. This file is
    the complete answer, and it retires BSE-to-NSE resolution entirely: a BSE
    name with an NSE listing is already here under its NSE symbol.

    Returns [Symbol, Company Name, Series, Listed]. Header cells carry leading
    spaces in the source; they are stripped.
    """
    import io
    import urllib.request
    req = urllib.request.Request(url, headers=_NSE_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode("utf-8-sig", errors="replace")
    raw = pd.read_csv(io.StringIO(body), dtype=str)
    raw.columns = [c.strip() for c in raw.columns]
    for col in ("SYMBOL", "NAME OF COMPANY", "SERIES", "DATE OF LISTING"):
        if col not in raw.columns:
            raise ValueError(f"NSE equity list lacks expected column {col!r}")
    raw = raw.apply(lambda c: c.str.strip())
    keep = raw[raw["SERIES"].isin(NSE_SERIES)]
    return pd.DataFrame({
        "Symbol": keep["SYMBOL"],
        "Company Name": keep["NAME OF COMPANY"].str.replace(r"\s+Limited$", " Ltd", regex=True),
        "Series": keep["SERIES"],
        "Listed": keep["DATE OF LISTING"],
    }).drop_duplicates("Symbol").reset_index(drop=True)


def sector_lookup(sources: list[Path], tv_classification: Path | None,
                  tv_map: Path | None) -> dict[str, str]:
    """Symbol -> sector, from the user's own files first, then a mapped taxonomy.

    The universe's 80 sector names are the vocabulary. Anything that cannot be
    expressed in it is left OUT of the dict, so merge() refuses the name rather
    than inventing an 81st group. Order matters: a sector the user assigned by
    hand beats one inferred through a mapping.
    """
    out: dict[str, str] = {}
    for path in sources:
        frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
        idcol = next((c for c in frame.columns if c.lower() in ("company_id", "companyid", "symbol")), None)
        # "sector" wins over "industry" when a file carries both: the universe's
        # vocabulary IS sector-level, and Stock_cat.csv lists the finer industry first.
        seccol = next((c for c in frame.columns if c.lower() == "sector"), None)               or next((c for c in frame.columns if c.lower() == "industry"), None)
        if idcol is None or seccol is None:
            continue
        for cid, sec in zip(frame[idcol], frame[seccol]):
            ex, _, sym = str(cid).partition(":")
            sym = (sym or ex).strip().upper()
            if sec.strip() and sym not in out:
                out[sym] = sec.strip()
    if tv_classification and tv_map and tv_classification.exists() and tv_map.exists():
        mapping = pd.read_csv(tv_map, dtype=str).fillna("")
        tv_to_sector = dict(zip(mapping["tv_industry"].str.strip(), mapping["sector"].str.strip()))
        cls = pd.read_csv(tv_classification, dtype=str, encoding="utf-8-sig").fillna("")
        for sym, ind in zip(cls["Symbol"].str.strip().str.upper(), cls["TV_Industry"].str.strip()):
            sec = tv_to_sector.get(ind, "")
            if sec and sym not in out:
                out[sym] = sec
    return out


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
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--list", help="candidate CSV: companyId,Name,Industry")
    src.add_argument("--nse-list", action="store_true",
                     help="use NSE's official main-board equity list as the candidate set")
    ap.add_argument("--sectors", nargs="*", default=[],
                    help="CSV(s) giving symbol->sector in the universe's vocabulary (first match wins)")
    ap.add_argument("--tv-classification", help="Symbol,TV_Sector,TV_Industry CSV")
    ap.add_argument("--tv-map", help="tv_industry,sector CSV mapping into the universe's vocabulary")
    ap.add_argument("--allow-sector", nargs="*", default=[],
                    help="sector value(s) NEW to the universe, admitted by explicit declaration")
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    universe = pd.read_csv(args.universe, encoding="utf-8")
    if args.nse_list:
        listed = fetch_nse_equity_list()
        sectors = sector_lookup([Path(p) for p in args.sectors],
                                Path(args.tv_classification) if args.tv_classification else None,
                                Path(args.tv_map) if args.tv_map else None)
        listed["Industry"] = listed["Symbol"].map(sectors).fillna("")
        unsectored = listed[(listed["Industry"] == "") & ~listed["Symbol"].isin(set(universe["Symbol"]))]
        print(f"NSE main board: {len(listed)} | no sector in any source: {len(unsectored)} (not admitted)")
        cands = listed[listed["Industry"] != ""][["Symbol", "Company Name", "Industry"]].reset_index(drop=True)
    else:
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
