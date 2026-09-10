# nse_tv_classification.csv

Copied 2026-09-10 from `Pareshking/Paresh` (MIT) — `data/nse_tv_classification.csv`,
TradingView's sector/industry tags for NSE symbols.

Used ONLY as a fallback sector source by `scripts/admit_universe.py`, translated into
this universe's own vocabulary through `sector_map_tv.csv`. It never overrides a sector
the user assigned by hand, and it never enters any calculation.

Refresh by re-copying from upstream; the mapping file is ours and stays.
