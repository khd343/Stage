"""Production data-boundary primitives for RS-Stages.

Acquisition is separated from quantitative calculations. The pre-market
information-set invariant is enforced before calculations consume the data.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DecisionSnapshot:
    """Market data snapshot permitted for a pre-market decision."""

    decision_date: pd.Timestamp
    latest_completed_session: pd.Timestamp
    data: pd.DataFrame


def normalize_session_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize market data to a sorted, unique tz-naive DatetimeIndex."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Market data must use a DatetimeIndex")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate market-data sessions detected")
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    if out.index.has_duplicates:
        raise ValueError("Duplicate sessions after timestamp normalization")
    return out.sort_index()


def latest_completed_session(index: pd.DatetimeIndex, decision_date: pd.Timestamp) -> pd.Timestamp:
    """Return the latest session strictly before decision session D."""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    decision = pd.Timestamp(decision_date)
    if decision.tzinfo is not None:
        decision = decision.tz_convert(None)
    decision = decision.normalize()
    pos = idx.searchsorted(decision, side="left") - 1
    if pos < 0:
        raise ValueError("No completed market session exists before decision date")
    return idx[pos]


def session_coverage(
    histories: dict[str, pd.DataFrame], decision_date: pd.Timestamp
) -> pd.Series:
    """How many symbols carry a usable Close at each session before decision D.

    Usable means the same thing it means to build_decision_snapshot: a dated row
    is not a session unless it has a Close. Yahoo publishes the row first and the
    values later, so counting rows would report a session as fully covered hours
    before any of its prices exist.

    A frame the boundary primitives would reject -- a non-datetime index,
    duplicate sessions -- is skipped rather than raised on. Those symbols are
    already excluded downstream with a reason; letting one of them abort the
    count would lose the whole universe to one bad frame.
    """
    decision = pd.Timestamp(decision_date).normalize()
    counts: Counter = Counter()
    for frame in histories.values():
        if frame is None or len(frame) == 0:
            continue
        try:
            data = normalize_session_index(frame)
        except (TypeError, ValueError):
            continue
        usable = data.index[data["Close"].notna()] if "Close" in data.columns else data.index
        counts.update(stamp for stamp in usable if stamp < decision)
    if not counts:
        return pd.Series(dtype="int64")
    return pd.Series(counts, dtype="int64").sort_index()


def global_information_boundary(
    histories: dict[str, pd.DataFrame],
    universe_size: int,
    decision_date: pd.Timestamp,
    min_coverage_pct: float,
) -> pd.Timestamp:
    """The newest session before D at which the universe can be measured together.

    LOCKED_SPEC 8.1: *the information boundary is global*. Letting each symbol
    stop at its own last close silently violates that, and the damage is not
    cosmetic -- RS_Score is a cross-sectional percentile, so a universe holding
    968 symbols priced on one session and 536 on the next ranks the fresher
    cohort against the staler one and reports the difference as relative
    strength. Measured 28 Aug 2026: Yahoo carries a session for roughly 39% of
    NSE names 16 hours after the close and ~100% by 40 hours, so a pre-market
    run can NEVER see a complete latest session. The boundary must therefore be
    chosen from coverage, not from recency, and no schedule can substitute for
    it.

    Returns the newest session meeting the threshold; typically one session
    older than the newest session that exists at all. That is the intended
    trade: a 30-week framework loses nothing to a day of latency, and gains a
    cross-section that is actually simultaneous.
    """
    if universe_size <= 0:
        raise ValueError("Universe size must be positive to compute coverage")
    coverage = session_coverage(histories, decision_date)
    if coverage.empty:
        raise ValueError(
            "No session carrying a Close exists before the decision date"
        )
    # ceil, not round: at 1,505 symbols and a 98% floor the requirement is 1,475
    # and not 1,474. A threshold that rounds down admits a session one symbol
    # short of the tolerance the publisher will later enforce.
    required = math.ceil(universe_size * min_coverage_pct / 100.0)
    qualifying = coverage.index[coverage.to_numpy() >= required]
    if len(qualifying) == 0:
        newest = coverage.index[-1]
        raise ValueError(
            f"No session before {pd.Timestamp(decision_date):%Y-%m-%d} covers "
            f"{min_coverage_pct:.1f}% of the {universe_size}-symbol universe "
            f"({required} symbols). The newest session, {newest:%Y-%m-%d}, "
            f"covers {int(coverage.iloc[-1])}. Either the provider is still "
            "backfilling and the run is simply too early, or enough symbols have "
            "gone permanently stale to breach the same tolerance the publisher "
            "enforces."
        )
    return pd.Timestamp(qualifying.max())


def build_decision_snapshot(market_data: pd.DataFrame, decision_date: pd.Timestamp) -> DecisionSnapshot:
    """Freeze only information available before the upcoming session opens.

    The boundary is the latest session that actually carries a Close, not merely
    the latest row the provider emitted. Yahoo publishes a dated row before the
    session's values are final, and a row with no Close is not a completed
    session: adopting it as the information boundary fabricates a boundary and
    §3 forbids that. Downstream the consequence was total — every calendar
    window would start and end one session late, so the 30-week average, its
    slope, Stage and every v2.2 field disagreed with an independent
    recalculation that had dropped the empty row first.

    Only the boundary is chosen this way. Interior rows are left untouched, so
    no history is discarded: a window mean skips a NaN and a window that never
    contained the row produce the same number.
    """
    data = normalize_session_index(market_data)
    decision = pd.Timestamp(decision_date)
    usable = data.index[data["Close"].notna()] if "Close" in data.columns else data.index
    latest = latest_completed_session(usable, decision)
    return DecisionSnapshot(
        decision_date=decision.normalize(),
        latest_completed_session=latest,
        data=data.loc[:latest].copy(),
    )


def validate_market_columns(frame: pd.DataFrame) -> None:
    """Require adjusted Close/High and raw share Volume columns."""
    required = {"Close", "High", "Volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required market columns: {sorted(missing)}")


def load_nse_constituents_csv(path: str | Path) -> pd.DataFrame:
    """Load the NSE universe and ignore symbols reserved for corporate actions.

    Symbols beginning with the literal prefix ``DUMMY`` are excluded from the
    analytical universe. The downloaded CSV itself remains unchanged so it is
    preserved as the official source file.
    """
    frame = pd.read_csv(path)
    required = {"Symbol", "Industry"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"NSE CSV missing required columns: {sorted(missing)}")
    if frame["Symbol"].isna().any() or frame["Industry"].isna().any():
        raise ValueError("NSE CSV contains missing Symbol or Industry values")
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip()
    if frame["Symbol"].eq("").any():
        raise ValueError("NSE CSV contains empty Symbol values")
    if frame["Symbol"].duplicated().any():
        raise ValueError("NSE CSV contains duplicate symbols")
    return frame.loc[~frame["Symbol"].str.startswith("DUMMY", na=False)].copy()


def yfinance_symbol(symbol: str) -> str:
    """Map an NSE CSV symbol to its Yahoo Finance NSE ticker without changing the universe."""
    symbol = str(symbol).strip()
    if not symbol:
        raise ValueError("Empty NSE symbol")
    return symbol if symbol.upper().endswith(".NS") else f"{symbol}.NS"


#: Benchmark indices. These are NOT part of the analytical universe and never
#: enter an RS ranking, a Stage classification or any locked signal. They exist
#: only as a reference line beside market breadth.
#:
#: Index tickers are passed to the provider verbatim: they carry no ``.NS``
#: suffix, so they must not go through :func:`yfinance_symbol`, which maps NSE
#: constituent symbols and is locked to that job.
INDEX_TICKERS = {
    "NIFTY_500": "^CRSLDX",
    "NIFTY_50": "^NSEI",
}


def download_index_history(
    ticker: str, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DataFrame:
    """Download a benchmark index using the same locked adjustment policy.

    Separate from :func:`download_yfinance_history` because an index ticker is
    not an NSE constituent symbol and must not be mapped as one.

    Since v2.2 this is no longer display-only: §4.1's RS line is computed from
    it and published. It remains outside every decision rule — no Stage, RS
    ranking, breakout test or Action label reads the index or anything derived
    from it — so a failed fetch degrades the RS line to unavailable and changes
    no signal.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for market-data acquisition") from exc

    frame = yf.download(
        str(ticker),
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        progress=False,
        actions=False,
        **yfinance_history_kwargs(),
    )
    if frame is None or frame.empty:
        raise ValueError(f"No history returned for index {ticker}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return normalize_session_index(frame)


def yfinance_history_kwargs() -> dict[str, bool]:
    """Return the locked yfinance adjustment policy."""
    return {"auto_adjust": True}


def _extract_bulk_ticker_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract one ticker from a yfinance bulk response without altering fields."""
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Bulk yfinance response did not return ticker-level columns")

    for level in range(frame.columns.nlevels):
        values = frame.columns.get_level_values(level)
        if ticker in values:
            out = frame.xs(ticker, axis=1, level=level, drop_level=True)
            if isinstance(out.columns, pd.MultiIndex):
                # A two-level provider response should collapse to OHLCV fields.
                out.columns = out.columns.get_level_values(-1)
            return out
    raise ValueError(f"No market data returned for {ticker}")


def download_yfinance_history(symbol: str, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    """Download one NSE symbol's history using the locked yfinance policy.

    ``end`` is exclusive per yfinance. The caller must still pass the result
    through ``build_decision_snapshot`` before signal calculations.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for market-data acquisition") from exc

    ticker = yfinance_symbol(symbol)
    frame = yf.download(
        ticker,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if frame.empty:
        raise ValueError(f"No market data returned for {ticker}")

    if isinstance(frame.columns, pd.MultiIndex):
        if ticker in frame.columns.get_level_values(-1):
            frame = frame.xs(ticker, axis=1, level=-1)
        elif ticker in frame.columns.get_level_values(0):
            frame = frame.xs(ticker, axis=1, level=0)
    frame = normalize_session_index(frame)
    validate_market_columns(frame)
    return frame


def download_yfinance_histories(
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    batch_size: int = 100,
) -> dict[str, pd.DataFrame]:
    """Download many NSE histories in bounded bulk batches.

    yfinance performs the provider requests in parallel inside ``download``.
    Batching avoids a single oversized request while reducing the per-symbol
    request overhead of the old sequential acquisition path.

    The function is intentionally strict: every requested symbol must return a
    non-empty, structurally valid OHLCV history. Partial acquisition raises an
    error rather than silently producing a partial quantitative universe.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for market-data acquisition") from exc

    normalized = [str(symbol).strip() for symbol in symbols]
    if any(not symbol for symbol in normalized):
        raise ValueError("Empty NSE symbol in bulk acquisition request")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate symbols in bulk acquisition request")

    histories: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    tickers = [yfinance_symbol(symbol) for symbol in normalized]

    for offset in range(0, len(tickers), batch_size):
        batch_tickers = tickers[offset : offset + batch_size]
        try:
            bulk = yf.download(
                tickers=batch_tickers,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=True,
                group_by="ticker",
            )
        except Exception as exc:
            for ticker in batch_tickers:
                failures[ticker] = f"{type(exc).__name__}: {exc}"
            continue

        for ticker in batch_tickers:
            symbol = ticker.removesuffix(".NS")
            try:
                frame = _extract_bulk_ticker_frame(bulk, ticker)
                if frame.empty:
                    raise ValueError(f"No market data returned for {ticker}")
                frame = normalize_session_index(frame)
                validate_market_columns(frame)
                histories[symbol] = frame
            except Exception as exc:
                failures[ticker] = f"{type(exc).__name__}: {exc}"

    if failures:
        raise RuntimeError(f"Bulk market-data acquisition failed for symbols: {failures}")

    return histories
