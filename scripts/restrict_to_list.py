"""Restrict the universe to the NSE names in one candidate list. Nothing else.

The universe is the user's list of stocks -- that is its definition, stated
twice. An expansion on 10 Sep 2026 admitted 478 names from NSE's complete
main-board list, including every bank and insurer; that was a wider universe
than the one asked for, and this removes what the list does not contain.

The admission TOOL keeps its NSE-list mode as an option; the universe's
boundary is the list. Existing rows inside the list are untouched. Prints every
symbol it removes.

Usage:
    python scripts/restrict_to_list.py --list "stock list.csv" [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "data" / "ind_niftytotalmarket_list.csv"


def nse_symbols_in(path: Path) -> set[str]:
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    idcol = next(c for c in frame.columns if c.lower() in ("company_id", "companyid"))
    ids = frame[idcol].str.strip()
    nse = ids[ids.str.upper().str.startswith("NSE:")].str.slice(4).str.strip().str.upper()
    # TradingView writes BAJAJ_AUTO for what NSE lists as BAJAJ-AUTO.
    return set(nse.str.replace("_", "-", regex=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    uni = pd.read_csv(args.universe, encoding="utf-8")
    allowed = nse_symbols_in(Path(args.list))
    gone = uni[~uni["Symbol"].isin(allowed)].sort_values("Symbol")
    print(f"universe {len(uni)} | list NSE names {len(allowed)} | not in list: {len(gone)}")
    for _, r in gone.iterrows():
        print(f"  - {r['Symbol']:<12} {r['Company Name'][:40]:<40} {r['Industry']}")
    if args.dry_run:
        print("dry run; nothing written"); return 0
    kept = uni[uni["Symbol"].isin(allowed)]
    kept.to_csv(args.universe, index=False, encoding="utf-8", lineterminator="\n")
    print(f"wrote {args.universe}: {len(uni)} -> {len(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
