"""Run a reproducible real-data RS/Stage audit from the locked NSE universe."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rs_stages.actions import with_actions
from rs_stages.market import breadth_history_from_trends
from rs_stages.pipeline import acquire_universe_histories
from rs_stages.screener import analyze_universe, analyze_universe_with_trend
from rs_stages.quant import rs_blend, rs_returns, calendar_asof
from rs_stages.data import (
    INDEX_TICKERS,
    build_decision_snapshot,
    download_index_history,
    global_information_boundary,
    load_nse_constituents_csv,
    normalize_session_index,
)

#: Sessions of Close retained per symbol so the UI can draw price history and
#: recompute the locked moving averages for a single symbol without a download.
PANEL_SESSIONS = 420

#: Sessions of universe-wide participation retained for the breadth trend.
BREADTH_SESSIONS = 250

#: Benchmark plotted beside market breadth. It is reference data only: no RS
#: ranking, Stage classification or Action rule reads it. Note that it tracks
#: 500 companies while our breadth tracks the Nifty Total Market universe, so a
#: divergence between the two lines can be composition, not market behaviour.
BENCHMARK_KEY = "NIFTY_500"

#: Share of the universe that may be missing before the audit refuses to
#: publish. An engineering guard, not a quantity from any source: RS_Score is a
#: cross-sectional percentile, so a depleted universe silently re-ranks every
#: symbol that survives. A delisting or a stray provider timeout is expected and
#: reported; an outage is not something to publish through.
MAX_UNIVERSE_LOSS_PCT = 2.0


def build_universe_snapshots(
    symbols, histories: dict, decision: pd.Timestamp, boundary: pd.Timestamp
) -> tuple[dict, list[tuple[str, str]]]:
    """Snapshot every symbol at ONE shared information boundary.

    Returns the snapshots and the symbols that could not be snapshotted, each
    with its reason. A symbol with no completed session at the boundary carries
    no information there, so excluding it is the same explicit insufficiency the
    quant layer applies everywhere else.

    Two steps, and both are load-bearing. Truncating at the boundary stops a
    symbol the provider HAS already updated from being measured a session ahead
    of the rest. Requiring the result to land exactly on the boundary stops a
    symbol the provider has NOT updated from silently sliding back a session --
    which is what build_decision_snapshot does when left to search a symbol's own
    history, and is how one published file came to hold two different dates.
    """
    snapshots: dict = {}
    unavailable: list[tuple[str, str]] = []
    boundary = pd.Timestamp(boundary).normalize()
    for symbol in symbols:
        name = str(symbol)
        if name not in histories:
            unavailable.append((name, "the provider returned no history"))
            continue
        try:
            data = normalize_session_index(histories[name])
        except (TypeError, ValueError) as exc:
            unavailable.append((name, str(exc)))
            continue
        try:
            snapshot = build_decision_snapshot(data.loc[data.index <= boundary], decision)
        except ValueError as exc:
            unavailable.append((name, str(exc)))
            continue
        if snapshot.latest_completed_session != boundary:
            unavailable.append((
                name,
                f"no completed session at the global boundary "
                f"{boundary:%Y-%m-%d}; its latest is "
                f"{snapshot.latest_completed_session:%Y-%m-%d}",
            ))
            continue
        snapshots[name] = snapshot
    return snapshots, unavailable


def published_boundary(path: Path) -> pd.Timestamp | None:
    """The session the currently published snapshot describes, if any.

    Read so a later run can refuse to move the record backwards. Anything
    unreadable returns None: a floor derived from a corrupt file would be worse
    than no floor, and the run is about to overwrite that file anyway.
    """
    if not path.exists():
        return None
    try:
        dates = pd.to_datetime(pd.read_csv(path)["Date"], errors="coerce").dropna()
    except (OSError, ValueError, KeyError):
        return None
    return pd.Timestamp(dates.max()).normalize() if len(dates) else None


def enforce_universe_coverage(missing: int, total: int) -> None:
    """Refuse to publish a universe too depleted to rank against itself."""
    loss_pct = missing / total * 100 if total else 0.0
    if loss_pct > MAX_UNIVERSE_LOSS_PCT:
        raise SystemExit(
            f"{missing} of {total} symbols ({loss_pct:.1f}%) had no usable "
            f"history, above the {MAX_UNIVERSE_LOSS_PCT}% tolerance. RS_Score "
            "is a cross-sectional percentile, so publishing a universe this "
            "depleted would rank every remaining symbol against a set the "
            "snapshot does not disclose. Re-run once the provider recovers."
        )



def independent_calendar_asof(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    idx = pd.DatetimeIndex(index).sort_values().unique()
    pos = idx.searchsorted(pd.Timestamp(target), side="right") - 1
    if pos < 0:
        raise ValueError("No completed session exists on or before target date")
    return idx[pos]


def independent_rs(close: pd.Series, decision: pd.Timestamp) -> dict[int, float]:
    s = close.sort_index().dropna()
    t = independent_calendar_asof(s.index, decision)
    latest = float(s.loc[t])
    out = {}
    for m in (3, 6, 9, 12):
        ref = independent_calendar_asof(s.index, t - pd.DateOffset(months=m))
        out[m] = latest / float(s.loc[ref]) - 1.0
    return out


def independent_stage(close: pd.Series, decision: pd.Timestamp) -> tuple[float, float, str]:
    """Independently reproduce 30W MA, 10-session slope and Stage truth table."""
    s = close.sort_index().dropna()
    t = independent_calendar_asof(s.index, decision)
    start_ref = independent_calendar_asof(s.index, t - pd.Timedelta(weeks=30))
    window = s.loc[(s.index >= start_ref) & (s.index <= t)]
    if len(window) < 2:
        raise ValueError("Insufficient history for independent 30W MA")
    ma = float(window.mean())

    ma_values = []
    for point in s.index:
        try:
            point_t = independent_calendar_asof(s.index, point)
            point_start = independent_calendar_asof(s.index, point_t - pd.Timedelta(weeks=30))
            point_window = s.loc[(s.index >= point_start) & (s.index <= point_t)]
            if len(point_window) < 2:
                continue
            ma_values.append((point_t, float(point_window.mean())))
        except ValueError:
            continue
    ma_series = pd.Series(dict(ma_values)).sort_index().dropna()
    pos = ma_series.index.searchsorted(t, side="right") - 1
    if pos < 10:
        raise ValueError("Insufficient history for independent MA slope")
    prior = float(ma_series.iloc[pos - 10])
    if prior == 0:
        raise ValueError("Cannot calculate independent slope from zero prior MA")
    slope = (float(ma_series.iloc[pos]) / prior - 1.0) * 100.0

    # Independent truth table: do not call production classify_stage().
    above = float(s.loc[t]) > ma
    rising = slope > 0.0
    if above and rising:
        stage = "Stage 2 — Advancing"
    elif above and not rising:
        stage = "Stage 3 — Topping"
    elif not above and not rising:
        stage = "Stage 4 — Declining"
    else:
        stage = "Stage 1 — Basing"
    return ma, slope, stage


def independent_high52(high: pd.Series, decision: pd.Timestamp) -> float:
    s = high.sort_index().dropna()
    t = independent_calendar_asof(s.index, decision)
    start = independent_calendar_asof(s.index, t - pd.Timedelta(weeks=52))
    window = s.loc[(s.index >= start) & (s.index <= t)]
    if len(window) < 200:
        raise ValueError("Insufficient history for independent 52W high")
    return float(window.max())


def independent_ma_10w(close: pd.Series, decision: pd.Timestamp) -> float:
    """Independent 10-calendar-week mean, written without the quant helpers."""
    s = close.sort_index().dropna()
    t = independent_calendar_asof(s.index, decision)
    start = independent_calendar_asof(s.index, t - pd.Timedelta(weeks=10))
    window = s.loc[(s.index >= start) & (s.index <= t)]
    if len(window) < 2:
        raise ValueError("Insufficient history for independent 10W MA")
    return float(window.mean())


def independent_low52(low: pd.Series, decision: pd.Timestamp) -> float:
    s = low.sort_index().dropna()
    t = independent_calendar_asof(s.index, decision)
    start = independent_calendar_asof(s.index, t - pd.Timedelta(weeks=52))
    window = s.loc[(s.index >= start) & (s.index <= t)]
    if len(window) < 200:
        raise ValueError("Insufficient history for independent 52W low")
    return float(window.min())


def _finite(values: list[float]) -> list[float]:
    """Drop NaN from a POSITIONAL window, matching pandas' skipna semantics.

    The universe is downloaded in bulk, and yfinance indexes a bulk response by
    the UNION of every symbol's dates. A stock that did not trade on a day some
    other stock did therefore carries a NaN row: ARTEMISMED and ZODIAC have 247
    sessions against the market's 251, so four rows of theirs are NaN.

    Production reads those windows with pandas, whose .mean()/.max()/.min() skip
    NaN. These hand-written checks used Python's sum()/max()/min(), where NaN
    poisons a sum and makes max() return whatever the ordering happens to yield --
    max([nan, 1.0]) is nan, max([1.0, nan]) is 1.0. Two implementations that agree
    on clean data then disagree on any symbol with a trading gap.

    The window is sliced positionally FIRST and filtered after, deliberately.
    Dropping NaN before slicing would pull in older sessions to refill the window
    and silently measure a different period than production does.
    """
    return [v for v in values if v == v]


def independent_sma(close: pd.Series, decision: pd.Timestamp, sessions: int) -> float:
    """v2.2 §5.1 — session average by plain summation, no rolling window.

    Sessions without a Close are skipped, so this is the mean of the latest
    ``sessions`` closes that exist — the same definition production applies by
    dropping them before slicing, reached here by filtering the comprehension.
    """
    values = [
        float(v)
        for stamp, v in close.sort_index().items()
        if stamp <= decision and pd.notna(v)
    ]
    if len(values) < sessions:
        raise ValueError("insufficient history")
    window = values[-sessions:]
    window = _finite(window)
    if not window:
        raise ValueError('no finite observations in the SMA window')
    return sum(window) / len(window)


def independent_contraction(
    high: pd.Series, low: pd.Series, close: pd.Series, decision: pd.Timestamp
) -> tuple[int, float]:
    """v2.2 §10.5 — block ranges rebuilt by iteration rather than slicing."""
    rows = [
        (float(h), float(low.loc[stamp]), float(close.loc[stamp]))
        for stamp, h in high.sort_index().items()
        if stamp <= decision and stamp in low.index and stamp in close.index
    ]
    if len(rows) < 50:
        raise ValueError("insufficient history")
    base = rows[-50:]
    ranges = []
    for block in range(5):
        chunk = base[block * 10 : block * 10 + 10]
        highs = [r[0] for r in chunk]
        lows = [r[1] for r in chunk]
        closes = [r[2] for r in chunk]
        highs, lows, closes = _finite(highs), _finite(lows), _finite(closes)
        if not highs or not lows or not closes:
            raise ValueError("degenerate base: block has no finite observations")
        mean_close = sum(closes) / len(closes)
        if mean_close == 0:
            raise ValueError("degenerate base")
        ranges.append((max(highs) - min(lows)) / mean_close * 100.0)
    tightenings = sum(1 for a, b in zip(ranges, ranges[1:]) if b < a)
    if ranges[0] == 0:
        # A first block whose high equals its low: the stock did not move for ten
        # sessions. Production returns NaN here (quant.contraction_ratio guards
        # `blocks[0] == 0`); this copy divided unguarded and raised
        # ZeroDivisionError, which the caller does not catch -- it catches only
        # ValueError/KeyError -- so one flat microcap aborted the entire audit
        # after three hours of downloads.
        #
        # Raising ValueError is the idiom this module already uses to mean "this
        # symbol cannot be reconciled"; the caller skips it and every other symbol
        # is still checked. Returning NaN instead would silently report a match
        # against production's NaN and verify nothing.
        raise ValueError("degenerate base: first block has zero range")
    return tightenings, ranges[-1] / ranges[0]


def independent_volume_dryup(volume: pd.Series, decision: pd.Timestamp) -> float:
    """v2.2 §10.5 — the drought measure, summed by hand."""
    values = [float(v) for stamp, v in volume.sort_index().items() if stamp <= decision]
    if len(values) < 60:
        raise ValueError("insufficient history")
    recent = _finite(values[-10:])
    baseline = _finite(values[-60:-10])
    if not recent or not baseline:
        raise ValueError("no finite observations in the dry-up window")
    mean_baseline = sum(baseline) / len(baseline)
    if mean_baseline == 0:
        raise ValueError("degenerate baseline")
    return (sum(recent) / len(recent)) / mean_baseline


def independent_pivot(high: pd.Series, decision: pd.Timestamp) -> float:
    """v2.2 §10.6 — the base high, by max over an explicit list."""
    values = [float(v) for stamp, v in high.sort_index().items() if stamp <= decision]
    if len(values) < 50:
        raise ValueError("insufficient history")
    window = _finite(values[-50:])
    if not window:
        raise ValueError('no finite observations in the pivot window')
    return max(window)


def independent_volume_ratio(volume: pd.Series, decision: pd.Timestamp) -> float:
    s = volume.sort_index().astype(float)
    pos = s.index.searchsorted(pd.Timestamp(decision), side="right") - 1
    if pos < 50:
        raise ValueError("Insufficient history for independent volume ratio")
    baseline = float(s.iloc[pos - 50:pos].mean())
    latest = float(s.iloc[pos])
    if baseline == 0:
        return np.inf if latest > 0 else np.nan
    return latest / baseline


def independent_ud(close: pd.Series, volume: pd.Series, decision: pd.Timestamp) -> float:
    c, v = close.sort_index().align(volume.sort_index(), join="inner")
    pos = c.index.searchsorted(pd.Timestamp(decision), side="right") - 1
    if pos < 20:
        raise ValueError("Insufficient history for independent U/D")
    delta = c.diff()
    up_sum = float(v.where(delta > 0, 0.0).iloc[pos - 19:pos + 1].sum())
    down_sum = float(v.where(delta < 0, 0.0).iloc[pos - 19:pos + 1].sum())
    if down_sum == 0:
        return np.inf if up_sum > 0 else np.nan
    return up_sum / down_sum


#: NSE closes at 15:30 IST. A run after this hour may treat the session that
#: just finished as complete; a run before it may not.
NSE_CLOSE_HOUR_IST = 16


def default_decision_date(now_ist: pd.Timestamp) -> pd.Timestamp:
    """The session this run is deciding for.

    Decisions are pre-market for an upcoming session, and the boundary rule uses
    the latest completed session strictly before that date. So the decision date
    must be the *next* session for the run to use the freshest close.

    Run after the NSE close, today's session is complete, so the decision is for
    tomorrow and today's close is the terminal information date. Run before the
    close, today is still in progress and must not enter any calculation, so the
    decision is for today and yesterday's close is terminal.

    Without this, a run at 23:30 IST would set the decision to today and
    therefore use *yesterday's* close, discarding a completed session that has
    been available for eight hours.
    """
    today = now_ist.tz_localize(None).normalize()
    return today + pd.Timedelta(days=1) if now_ist.hour >= NSE_CLOSE_HOUR_IST else today


def resolve_dates(decision_arg: str | None, start_arg: str | None, end_arg: str | None) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    decision = pd.Timestamp(decision_arg) if decision_arg else default_decision_date(now_ist)
    start = pd.Timestamp(start_arg) if start_arg else decision - pd.Timedelta(days=500)
    end = pd.Timestamp(end_arg) if end_arg else decision + pd.Timedelta(days=1)
    if start >= end:
        raise ValueError("Audit start date must be before end date")
    return decision, start, end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--decision-date")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    decision, start, end = resolve_dates(args.decision_date, args.start, args.end)
    universe = load_nse_constituents_csv(args.universe)
    if universe["Symbol"].str.startswith("DUMMY", na=False).any():
        raise SystemExit("DUMMY symbols must never enter the analytical universe")

    # Market data is acquired once. Both decision dates are then derived from
    # the same download, so the previous-session snapshot cannot disagree with
    # the current one because of a provider revision between two calls.
    histories = acquire_universe_histories(args.universe, start, end)

    # A symbol the provider could not deliver is explicit insufficiency, not a
    # reason to lose the other 749. Three tickers timed out against Yahoo on the
    # run of 25 Aug 2026 and one came back with no rows at all, which raised out
    # of the comprehension this replaces and aborted the whole audit before
    # anything was published.
    #
    # Skipping is not silent, and it is bounded. RS_Score is a cross-sectional
    # percentile over the symbols actually analysed, so every dropped symbol
    # shifts the rank of every symbol that remains. A handful is immaterial; a
    # provider outage that removed a large slice would republish the whole
    # universe with quietly wrong ranks and nothing in the output would look
    # unusual. Past the tolerance the run therefore fails instead of publishing.
    #
    # The 2% ceiling is an engineering guard, not a quantity from any source,
    # and is named as such wherever it surfaces.
    # LOCKED_SPEC 8.1: the information boundary is GLOBAL. It is derived from
    # coverage rather than recency because Yahoo backfills the NSE universe over
    # roughly 16 to 40 hours after a close, while NSE reopens 17.75 hours after
    # it -- so a pre-market run that took "the newest session that exists" would
    # never see a complete one. Measured 28 Aug 2026: 39% coverage at 16 hours,
    # ~100% at 40. See DECISION_LOG D-2.2.16.
    #
    # The floor is the publisher's own tolerance, not a second opinion: a
    # boundary admitting more loss than enforce_universe_coverage permits would
    # pass here and abort a few lines later, so the two are one number.
    min_coverage_pct = 100.0 - MAX_UNIVERSE_LOSS_PCT
    try:
        boundary = global_information_boundary(
            histories, len(universe), decision, min_coverage_pct
        )
    except ValueError as exc:
        raise SystemExit(f"No global information boundary: {exc}")
    print(
        f"Global information boundary: {boundary.date()} "
        f"(newest session covering >= {min_coverage_pct:.1f}% of {len(universe)} symbols)"
    )

    # THE RECORD NEVER MOVES BACKWARDS. Coverage at the newest session varies with
    # the hour a run happens to start, so two runs on the same day can legitimately
    # choose different boundaries. Publishing the older one would walk the snapshot
    # back a session -- a terminal showing Thursday, then Wednesday, with both runs
    # reporting success. Re-publishing the SAME session is fine and expected (the
    # provider revises); only regression is refused, and it is a clean no-op rather
    # than a failure, because running at an hour the provider has not caught up to
    # is normal rather than broken.
    floor = published_boundary(Path(args.output))
    if floor is not None and boundary < floor:
        print(
            f"Published snapshot already describes {floor.date()}, which is newer "
            f"than the {boundary.date()} this run can cover. Nothing published: the "
            "record does not move backwards."
        )
        return

    snapshots, unavailable = build_universe_snapshots(
        universe["Symbol"], histories, decision, boundary
    )

    if unavailable:
        print(f"\nExcluded {len(unavailable)} of {len(universe)} symbols:")
        for name, reason in unavailable:
            print(f"  {name}: {reason}")

    enforce_universe_coverage(len(unavailable), len(universe))

    # The previous snapshot re-runs the identical pipeline with the information
    # boundary moved back one completed session. It is not a stored copy of an
    # earlier run, so both sides always come from the same pipeline version.
    #
    # It is chosen globally for the same reason the current one is: transitions
    # compare two snapshots, and a previous side assembled from each symbol's own
    # prior close would report a move between different pairs of days for
    # different symbols. Its absence is survivable -- day-over-day change is a
    # convenience, not a signal -- so a failure here degrades rather than aborts.
    previous_snapshots: dict = {}
    try:
        previous_boundary = global_information_boundary(
            histories, len(universe), boundary, min_coverage_pct
        )
        previous_snapshots, _ = build_universe_snapshots(
            list(snapshots), histories, boundary, previous_boundary
        )
        print(f"Previous information boundary: {previous_boundary.date()}")
    except ValueError as exc:
        print(f"No global boundary before {boundary.date()} ({exc}); "
              "day-over-day changes are published as unavailable")

    # Benchmark, fetched before the analysis because §4.1's RS line consumes it.
    # Over the full acquisition window rather than the breadth window: the RS
    # line needs 52 calendar weeks of stock/benchmark overlap, which the breadth
    # window does not guarantee. Reused for breadth alignment further down, so
    # this is one download, not two.
    #
    # A failure here must not fail the audit. The index is external and the RS
    # line degrades to unavailable; breadth is ours and computed either way.
    benchmark_ticker = INDEX_TICKERS[BENCHMARK_KEY]
    benchmark_close: pd.Series | None = None
    try:
        benchmark_close = download_index_history(
            benchmark_ticker, start=start, end=end
        )["Close"].astype(float)
        print(
            f"Benchmark {benchmark_ticker}: {len(benchmark_close)} sessions "
            f"{benchmark_close.index.min().date()} to {benchmark_close.index.max().date()}"
        )
    except (ImportError, ValueError, KeyError, OSError) as exc:
        print(
            f"Benchmark {benchmark_ticker} unavailable ({type(exc).__name__}); "
            "the RS line and the breadth index column are published as unavailable"
        )

    result, trends = analyze_universe_with_trend(
        snapshots, trend_sessions=PANEL_SESSIONS, benchmark=benchmark_close
    )
    previous_result = (
        analyze_universe(previous_snapshots, benchmark=benchmark_close)
        if previous_snapshots
        else pd.DataFrame()
    )

    # NaN == NaN IS AGREEMENT HERE. Both implementations return NaN for the same
    # honest reasons -- a zero volume baseline, a degenerate base, insufficient
    # history -- and np.isclose treats NaN as unequal to itself unless told
    # otherwise. Without equal_nan the checker reports a mismatch precisely when
    # the two sides agree that a value cannot be computed, which is the opposite
    # of what it exists to detect. U_D already carried equal_nan; every other
    # comparison was missed, and three zero-volume NSE listings (BRIGHT, KALYANI,
    # HERCULES) aborted the audit by agreeing correctly.
    failures = []
    checked_stage = checked_high = checked_volume = checked_ud = checked_liquidity = 0
    checked_ma_10w = checked_low = checked_trend = 0
    checked_sma = checked_contraction = checked_dryup = checked_pivot = 0
    for symbol, snap in snapshots.items():
        close = snap.data["Close"].astype(float)
        high = snap.data["High"].astype(float)
        volume = snap.data["Volume"].astype(float)
        t = snap.latest_completed_session
        try:
            expected = independent_rs(close, t)
            actual = rs_returns(close, t)
            for m in (3, 6, 9, 12):
                if not np.isclose(expected[m], actual[m], rtol=0, atol=1e-12, equal_nan=True):
                    failures.append(f"{symbol}: R{m}M mismatch")
            expected_blend = 0.40 * expected[3] + 0.20 * expected[6] + 0.20 * expected[9] + 0.20 * expected[12]
            if not np.isclose(expected_blend, float(result.loc[symbol, "RS_Blend"]), rtol=0, atol=1e-12, equal_nan=True):
                failures.append(f"{symbol}: RS blend mismatch")
        except ValueError:
            pass

        try:
            ma, slope, stage = independent_stage(close, t)
            checked_stage += 1
            row = result.loc[symbol]
            if not np.isclose(row["MA_30W"], ma, rtol=0, atol=1e-12, equal_nan=True): failures.append(f"{symbol}: 30W MA mismatch")
            if not np.isclose(row["MA_30W_Slope_10S_Pct"], slope, rtol=0, atol=1e-12, equal_nan=True): failures.append(f"{symbol}: MA slope mismatch")
            if row["Stage"] != stage: failures.append(f"{symbol}: Stage mismatch")
        except ValueError:
            pass

        try:
            expected = independent_high52(high, t)
            checked_high += 1
            if not np.isclose(result.loc[symbol, "High_52W"], expected, rtol=0, atol=1e-12, equal_nan=True): failures.append(f"{symbol}: 52W high mismatch")
        except ValueError:
            pass

        # --- v2.2 -----------------------------------------------------------
        for sessions, field in ((150, "SMA_150"), (200, "SMA_200")):
            try:
                expected = independent_sma(close, t, sessions)
                checked_sma += 1
                if not np.isclose(result.loc[symbol, field], expected, rtol=0, atol=1e-9, equal_nan=True):
                    failures.append(f"{symbol}: {field} mismatch")
            except (ValueError, KeyError):
                pass

        if "Low" in snap.data.columns:
            try:
                low_series = snap.data["Low"].astype(float)
                exp_count, exp_ratio = independent_contraction(high, low_series, close, t)
                checked_contraction += 1
                if int(result.loc[symbol, "VCP_Contractions"]) != exp_count:
                    failures.append(f"{symbol}: VCP_Contractions mismatch")
                if not np.isclose(
                    result.loc[symbol, "Contraction_Ratio"], exp_ratio,
                    rtol=0, atol=1e-9, equal_nan=True
                ):
                    failures.append(f"{symbol}: Contraction_Ratio mismatch")
            except (ValueError, KeyError):
                pass

        try:
            expected = independent_volume_dryup(volume, t)
            checked_dryup += 1
            if not np.isclose(result.loc[symbol, "Volume_DryUp"], expected, rtol=0, atol=1e-9, equal_nan=True):
                failures.append(f"{symbol}: Volume_DryUp mismatch")
        except (ValueError, KeyError):
            pass

        try:
            expected = independent_pivot(high, t)
            checked_pivot += 1
            if not np.isclose(result.loc[symbol, "VCP_Pivot"], expected, rtol=0, atol=1e-9, equal_nan=True):
                failures.append(f"{symbol}: VCP_Pivot mismatch")
        except (ValueError, KeyError):
            pass

        try:
            expected = independent_ma_10w(close, t)
            checked_ma_10w += 1
            if not np.isclose(result.loc[symbol, "MA_10W"], expected, rtol=0, atol=1e-12, equal_nan=True):
                failures.append(f"{symbol}: 10W MA mismatch")
        except ValueError:
            pass

        if "Low" in snap.data.columns:
            try:
                expected = independent_low52(snap.data["Low"].astype(float), t)
                checked_low += 1
                if not np.isclose(result.loc[symbol, "Low_52W"], expected, rtol=0, atol=1e-12, equal_nan=True):
                    failures.append(f"{symbol}: 52W low mismatch")
            except ValueError:
                pass

        try:
            expected = independent_volume_ratio(volume, t)
            checked_volume += 1
            if not np.isclose(result.loc[symbol, "Volume_Ratio"], expected, rtol=0, atol=1e-12, equal_nan=True): failures.append(f"{symbol}: volume ratio mismatch")
        except ValueError:
            pass

        try:
            expected = independent_ud(close, volume, t)
            checked_ud += 1
            if not np.isclose(result.loc[symbol, "U_D"], expected, rtol=0, atol=1e-12, equal_nan=True): failures.append(f"{symbol}: U/D mismatch")
        except ValueError:
            pass

        value = (close * volume).loc[:t].dropna()
        if len(value) >= 20:
            checked_liquidity += 1
            expected = float(value.iloc[-20:].mean())
            if not np.isclose(result.loc[symbol, "AvgValue20"], expected, rtol=0, atol=1e-12, equal_nan=True): failures.append(f"{symbol}: liquidity mismatch")
        elif not pd.isna(result.loc[symbol, "AvgValue20"]):
            failures.append(f"{symbol}: liquidity should be NaN")

    # The stored trend panel must agree with the row it belongs to. A panel that
    # disagreed with the snapshot would let the chart and the table tell two
    # different stories about the same session.
    for symbol, frame in trends.items():
        if frame.empty or symbol not in result.index:
            continue
        checked_trend += 1
        row = result.loc[symbol]
        last = frame.index.max()
        if last != row["Date"]:
            failures.append(f"{symbol}: trend panel ends at {last}, snapshot at {row['Date']}")
            continue
        if pd.notna(row["Close"]) and not np.isclose(
            float(frame["Close"].loc[last]), float(row["Close"]),
            rtol=0, atol=1e-12, equal_nan=True
        ):
            failures.append(f"{symbol}: trend panel Close disagrees with snapshot Close")
        for column in ("MA_10W", "MA_30W"):
            stored, reported = float(frame[column].loc[last]), row[column]
            if pd.notna(reported) and not np.isclose(stored, float(reported), rtol=0, atol=1e-12, equal_nan=True):
                failures.append(f"{symbol}: trend panel {column} disagrees with snapshot")

    result = result.join(universe.set_index("Symbol"), how="left", rsuffix="_NSE")
    result = with_actions(result)
    result.to_csv(args.output)

    output_dir = Path(args.output).resolve().parent

    # Previous-session snapshot: same columns, boundary moved back one session.
    previous_path = output_dir / "previous_research.csv"
    if not previous_result.empty:
        previous_result = previous_result.join(universe.set_index("Symbol"), how="left", rsuffix="_NSE")
        previous_result = with_actions(previous_result)
        previous_result.to_csv(previous_path)

    # Price panel: Close only. The moving averages are deliberately not stored —
    # the UI recomputes them for the one symbol it draws using the same locked
    # functions, so a chart line can never drift from the locked definition.
    #
    # Stored as a compressed NumPy grid rather than Parquet. Every symbol shares
    # the same completed-session calendar, so the panel is a dense
    # sessions x symbols matrix; that is smaller than Parquet (measured 0.88 MB
    # against 1.39 MB) and, more importantly, it is read with NumPy alone. The
    # presentation layer therefore needs no Arrow runtime to draw a chart.
    #
    # It is published as a release asset rather than committed: it is a
    # regenerated binary that changes completely every run, so Git cannot delta
    # it and would store a fresh blob per run, permanently. Measured history
    # cost: 1.43 MB/run committed against 0 MB/run as a replaced asset.
    panel_path = output_dir / "price_panel.npz"
    frames = {symbol: frame for symbol, frame in trends.items() if not frame.empty}
    symbols = sorted(frames)
    sessions = np.array(sorted({stamp for f in frames.values() for stamp in f.index}))

    closes = np.full((len(sessions), len(symbols)), np.nan, dtype="float32")
    position = {stamp: i for i, stamp in enumerate(sessions)}
    for column, symbol in enumerate(symbols):
        frame = frames[symbol]
        rows = np.fromiter((position[stamp] for stamp in frame.index), dtype=np.intp, count=len(frame))
        closes[rows, column] = frame["Close"].to_numpy(dtype="float32")

    np.savez_compressed(
        panel_path,
        close=closes,
        symbols=np.array(symbols, dtype="U32"),
        dates=pd.DatetimeIndex(sessions).to_numpy().astype("datetime64[D]"),
    )

    # The panel and the committed snapshot are published to different places, so
    # they could drift. Refuse to publish a panel whose terminal session
    # disagrees with the snapshot's decision date: a chart and a table must
    # never describe different sessions.
    panel_end = pd.Timestamp(sessions[-1]) if len(sessions) else pd.NaT
    snapshot_end = pd.Timestamp(pd.to_datetime(result["Date"]).max())
    if pd.isna(panel_end) or panel_end.normalize() != snapshot_end.normalize():
        failures.append(
            f"price panel ends at {panel_end} but the snapshot decision date is {snapshot_end}"
        )

    # Breadth history: point-in-time participation counts, one row per session.
    breadth_path = output_dir / "breadth_history.csv"
    breadth = breadth_history_from_trends(trends, sessions=BREADTH_SESSIONS)

    # Benchmark index, aligned onto the breadth session calendar. A failure to
    # fetch it must not fail the audit: breadth is ours and computed, the index
    # is an external convenience, so the column is simply absent and the chart
    # says so.
    if not breadth.empty:
        if benchmark_close is not None:
            aligned = benchmark_close.reindex(pd.DatetimeIndex(breadth["Date"]))
            breadth["Benchmark_Close"] = aligned.to_numpy()
            breadth["Benchmark_Ticker"] = benchmark_ticker
            covered = int(breadth["Benchmark_Close"].notna().sum())
            print(f"Benchmark {benchmark_ticker}: {covered} of {len(breadth)} sessions aligned")
        else:
            print("Benchmark absent; breadth published without the index column")

        breadth.to_csv(breadth_path, index=False)

    # THE SNAPSHOT MUST CARRY EXACTLY ONE DATE. Before the global boundary this
    # was a printed diagnostic captioned "provider lag, not an error" -- and it
    # WAS the error: a file holding 968 symbols at one session and 536 at the
    # next ranks two populations against each other and calls the difference
    # relative strength. Every path that could reintroduce it now fails here
    # instead of narrating itself, which is the whole point of the change.
    published = pd.to_datetime(result["Date"], errors="coerce").dt.normalize()
    distinct = sorted(published.dropna().unique())
    if len(distinct) != 1:
        counts = published.value_counts()
        breakdown = ", ".join(
            f"{pd.Timestamp(d).date()}={int(counts[d])}" for d in reversed(distinct)
        ) or "none"
        failures.append(
            f"the snapshot spans {len(distinct)} sessions ({breakdown}); "
            "LOCKED_SPEC 8.1 requires one global information boundary"
        )
    elif pd.Timestamp(distinct[0]) != boundary:
        failures.append(
            f"the snapshot is dated {pd.Timestamp(distinct[0]).date()} but the "
            f"global boundary is {boundary.date()}"
        )

    if failures:
        raise SystemExit("Independent research-output reconciliation failures:\n" + "\n".join(failures[:100]))

    print(f"Decision date: {decision.date()}")
    print(f"Yahoo history: {start.date()} to {end.date()} exclusive")
    print(f"Universe rows after DUMMY exclusion: {len(universe)}")
    print(f"Research rows: {len(result)}")
    print(f"Independent checks: stage={checked_stage}, high52={checked_high}, volume={checked_volume}, ud={checked_ud}, liquidity={checked_liquidity}")
    print(f"Independent checks (v2.1): ma10w={checked_ma_10w}, low52={checked_low}, trend_panel={checked_trend}")
    print(
        f"Independent checks (v2.2): sma={checked_sma}, contraction={checked_contraction}, "
        f"dryup={checked_dryup}, pivot={checked_pivot}"
    )
    rs_line_rows = int(result["RS_Line"].notna().sum()) if "RS_Line" in result.columns else 0
    print(f"RS line computed for {rs_line_rows} of {len(result)} symbols")
    print(f"Previous-session rows: {len(previous_result)}")
    sessions_n, symbols_n = closes.shape
    print(f"Price panel grid: {sessions_n} sessions x {symbols_n} symbols")
    print(f"Price panel size: {panel_path.stat().st_size / 1e6:.2f} MB (published as a release asset)")
    print(f"Breadth history sessions: {len(breadth)}")
    print(f"Action counts:\n{result['Action'].value_counts().to_string()}")
    print(f"Stage counts:\n{result['Stage'].value_counts(dropna=False).to_string()}")
    print(f"Sufficient RS rows: {result['RS_Blend'].notna().sum()}")
    print(f"Sufficient 52W rows: {result['High_52W'].notna().sum()}")
    print(f"Sufficient liquidity rows: {result['AvgValue20'].notna().sum()}")


if __name__ == "__main__":
    main()
