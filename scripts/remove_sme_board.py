"""One-time removal of SME-board listings from the universe. Reversible by git.

NSE runs two boards. The main board (EQUITY_L.csv, series EQ/BE/BZ) and the SME
platform (SME_EQUITY_L.csv) are separate lists with different lot sizes,
disclosure rules and liquidity regimes. The universe admitted 90 SME names on
10 Sep 2026 because they appeared in a hand-maintained candidate list and
traded daily -- the liquidity rule cannot see which board a name is on.

Going forward the candidate source is the main-board list only, so this cannot
recur. This script removes the ones already in. It reads the SME list live so
the decision is made against NSE's own definition, not a snapshot, and prints
every symbol it removes.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "ind_niftytotalmarket_list.csv"
SME_LIST = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
           "Referer": "https://www.nseindia.com/"}


def sme_symbols(url: str = SME_LIST) -> set[str]:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=30) as r:
        body = r.read().decode("utf-8-sig", errors="replace")
    frame = pd.read_csv(io.StringIO(body), dtype=str)
    frame.columns = [c.strip() for c in frame.columns]
    return set(frame["SYMBOL"].str.strip())


def main() -> int:
    dry = "--dry-run" in sys.argv
    uni = pd.read_csv(UNIVERSE, encoding="utf-8")
    sme = sme_symbols()
    gone = uni[uni["Symbol"].isin(sme)].sort_values("Symbol")
    print(f"universe {len(uni)} | on NSE SME board: {len(gone)}")
    for _, r in gone.iterrows():
        print(f"  - {r['Symbol']:<12} {r['Company Name']}")
    if dry:
        print("dry run; nothing written"); return 0
    kept = uni[~uni["Symbol"].isin(sme)]
    kept.to_csv(UNIVERSE, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {UNIVERSE}: {len(uni)} -> {len(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
