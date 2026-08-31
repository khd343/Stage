"""Pure reference calculations for RS-Stages.

No Streamlit/UI/data-download code belongs here. Functions are deterministic
quantitative primitives used by tests and later application code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def latest_completed_session(index: pd.DatetimeIndex, decision_date: pd.Timestamp) -> pd.Timestamp:
    """Return the latest observed session strictly before a pre-market decision date."""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    pos = idx.searchsorted(pd.Timestamp(decision_date), side="left") - 1
    if pos < 0:
        raise ValueError("No completed session is available before decision date")
    return idx[pos]


def calendar_asof(index: pd.DatetimeIndex, target: pd.Timestamp) -> pd.Timestamp:
    """Return the last observed session on or before a calendar reference date."""
    idx = pd.DatetimeIndex(index).sort_values().unique()
    pos = idx.searchsorted(pd.Timestamp(target), side="right") - 1
    if pos < 0:
        raise ValueError("No session exists on or before calendar reference date")
    return idx[pos]


def rs_returns(close: pd.Series, latest: pd.Timestamp) -> dict[int, float]:
    """Calculate 3/6/9/12 calendar-month simple returns."""
    close = close.sort_index().dropna()
    t = calendar_asof(close.index, pd.Timestamp(latest))
    latest_close = float(close.loc[t])
    out: dict[int, float] = {}
    for months in (3, 6, 9, 12):
        ref = calendar_asof(close.index, t - pd.DateOffset(months=months))
        out[months] = latest_close / float(close.loc[ref]) - 1.0
    return out


def rs_blend(returns: dict[int, float]) -> float:
    return 0.40 * returns[3] + 0.20 * returns[6] + 0.20 * returns[9] + 0.20 * returns[12]


def rs_score(blend: pd.Series) -> pd.Series:
    """Cross-sectional RS score using rank(pct=True, method='min') × 98 + 1."""
    valid = blend.dropna()
    pct = valid.rank(pct=True, method="min")
    result = pd.Series(np.nan, index=blend.index, dtype=float)
    result.loc[valid.index] = np.rint(pct * 98.0 + 1.0)
    return result


def calendar_window(series: pd.Series, end: pd.Timestamp, weeks: int) -> pd.Series:
    """Return observations from the calendar start reference session through end."""
    s = series.sort_index()
    t = calendar_asof(s.index, pd.Timestamp(end))
    start_ref = calendar_asof(s.index, t - pd.Timedelta(weeks=weeks))
    return s.loc[(s.index >= start_ref) & (s.index <= t)]


def ma_calendar_weeks(close: pd.Series, end: pd.Timestamp, weeks: int) -> float:
    """Simple moving average over every valid session in a calendar-week window.

    This is the single locked moving-average definition. The window starts at
    the last observed session on or before ``end - weeks`` and ends at the last
    observed session on or before ``end``. It is deliberately *not* a fixed
    trading-day row count, so the number of observations varies with holidays.
    """
    s = close.sort_index().dropna()
    t = calendar_asof(s.index, pd.Timestamp(end))
    start_ref = calendar_asof(s.index, t - pd.Timedelta(weeks=weeks))
    window = s.loc[(s.index >= start_ref) & (s.index <= t)]
    if len(window) < 2:
        raise ValueError(f"Insufficient history for {weeks}W MA window")
    return float(window.mean())


def ma_calendar_weeks_series(close: pd.Series, weeks: int) -> pd.Series:
    """Calendar-window MA at each session, evaluated as of that session.

    Values are identical to calling :func:`ma_calendar_weeks` at every session;
    window boundaries are resolved by position instead of by repeated sorting so
    the per-symbol cost is linear rather than quadratic. Sessions without a
    complete reference window yield NaN rather than a partial-window average.

    INDEXED ON REAL SESSIONS ONLY, like :func:`sma_series`. Yahoo emits a dated
    row before it has that session's values, and this series used to carry a
    point for each such row -- an MA value repeated from the previous real
    session. The values were right; the POSITIONS were not, and ma_slope_pct
    steps back a fixed number of positions. One empty row inside the trailing
    ten therefore made "ten sessions ago" reach back only nine, and the whole
    universe's slope silently shifted. Measured 31 Aug 2026: every one of 1,505
    symbols failed its independent slope check because 28 Aug arrived as a dated
    row with no close. §3 already settles which reading is right -- a row with no
    Close is not a completed session.
    """
    clean = close.sort_index().dropna()
    if clean.empty:
        return pd.Series(np.nan, index=clean.index, dtype=float)

    session_index = pd.DatetimeIndex(clean.index)
    ends = np.arange(len(session_index))
    starts = clean.index.searchsorted(session_index - pd.Timedelta(weeks=weeks), side="right") - 1

    values = np.full(len(session_index), np.nan, dtype=float)
    for position, (start, end) in enumerate(zip(starts, ends)):
        if start < 0 or end < 0 or (end - start + 1) < 2:
            continue
        values[position] = float(clean.iloc[start : end + 1].mean())
    return pd.Series(values, index=session_index, dtype=float)


def ma_30w(close: pd.Series, end: pd.Timestamp) -> float:
    """30-calendar-week simple moving average using calendar start as-of session."""
    return ma_calendar_weeks(close, end, 30)


def ma_30w_series(close: pd.Series) -> pd.Series:
    """Calendar-window 30W MA at each session where a reference session exists."""
    return ma_calendar_weeks_series(close, 30)


def ma_10w(close: pd.Series, end: pd.Timestamp) -> float:
    """10-calendar-week simple moving average.

    Adopted in locked-spec v2.1 as the shorter trend reference. It uses exactly
    the same calendar-window construction as the 30-week MA so the two lines are
    directly comparable; it is not a 50-row trading-day average.
    """
    return ma_calendar_weeks(close, end, 10)


def ma_10w_series(close: pd.Series) -> pd.Series:
    """Calendar-window 10W MA at each session where a reference session exists."""
    return ma_calendar_weeks_series(close, 10)


def ma_slope_pct(ma: pd.Series, end: pd.Timestamp, sessions: int = 10) -> float:
    """10-session percentage change in the 30W MA."""
    s = ma.sort_index().dropna()
    pos = s.index.searchsorted(pd.Timestamp(end), side="right") - 1
    if pos < sessions:
        raise ValueError("Insufficient history for slope")
    prior = float(s.iloc[pos - sessions])
    if prior == 0:
        raise ValueError("Cannot calculate slope from zero prior MA")
    return (float(s.iloc[pos]) / prior - 1.0) * 100.0


def classify_stage(close: float, ma: float, slope_pct: float) -> str:
    """Classify the locked 30W-MA stage using strict comparisons."""
    values = (close, ma, slope_pct)
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("Stage classification requires finite Close, MA and slope")
    above = close > ma
    rising = slope_pct > 0.0
    if above and rising:
        return "Stage 2 — Advancing"
    if above and not rising:
        return "Stage 3 — Topping"
    if not above and not rising:
        return "Stage 4 — Declining"
    return "Stage 1 — Basing"


def high_52w(close_high: pd.Series, end: pd.Timestamp, min_sessions: int = 200) -> float:
    """Maximum adjusted High in a 52-calendar-week window."""
    s = close_high.sort_index().dropna()
    t = calendar_asof(s.index, pd.Timestamp(end))
    start_ref = calendar_asof(s.index, t - pd.Timedelta(weeks=52))
    window = s.loc[(s.index >= start_ref) & (s.index <= t)]
    if len(window) < min_sessions:
        raise ValueError("Insufficient history for 52W high")
    return float(window.max())


def low_52w(close_low: pd.Series, end: pd.Timestamp, min_sessions: int = 200) -> float:
    """Minimum adjusted Low in a 52-calendar-week window.

    Mirrors :func:`high_52w` exactly — same calendar window, same minimum
    observation count — so the pair defines a symmetric 52-week range. It is a
    presentation/range input only; no locked signal consumes it.
    """
    s = close_low.sort_index().dropna()
    t = calendar_asof(s.index, pd.Timestamp(end))
    start_ref = calendar_asof(s.index, t - pd.Timedelta(weeks=52))
    window = s.loc[(s.index >= start_ref) & (s.index <= t)]
    if len(window) < min_sessions:
        raise ValueError("Insufficient history for 52W low")
    return float(window.min())


def near_52w_high(close: float, high52: float, threshold: float = 0.03) -> bool:
    return bool(close >= (1.0 - threshold) * high52)


def volume_ratio(volume: pd.Series, end: pd.Timestamp) -> float:
    """Latest completed-session volume / preceding 50-session average."""
    v = volume.sort_index().astype(float)
    pos = v.index.searchsorted(pd.Timestamp(end), side="right") - 1
    if pos < 50:
        raise ValueError("Insufficient history for prior-50 volume baseline")
    baseline = float(v.iloc[pos - 50 : pos].mean())
    if baseline == 0:
        return np.inf if float(v.iloc[pos]) > 0 else np.nan
    return float(v.iloc[pos]) / baseline


def up_down_ratio(close: pd.Series, volume: pd.Series, end: pd.Timestamp) -> float:
    """20-session U/D ending at the latest completed session."""
    c, v = close.sort_index().align(volume.sort_index(), join="inner")
    c, v = c.astype(float), v.astype(float)
    pos = c.index.searchsorted(pd.Timestamp(end), side="right") - 1
    if pos < 20:
        raise ValueError("Insufficient history for 20-session U/D")
    delta = c.diff()
    up = v.where(delta > 0, 0.0)
    down = v.where(delta < 0, 0.0)
    up_sum = float(up.iloc[pos - 19 : pos + 1].sum())
    down_sum = float(down.iloc[pos - 19 : pos + 1].sum())
    if down_sum == 0:
        return np.inf if up_sum > 0 else np.nan
    return up_sum / down_sum


def ud_classification(ud: float) -> str:
    """Apply locked U/D thresholds with Heavy Distribution taking precedence."""
    if np.isnan(ud):
        return "Undefined"
    if ud < 0.6:
        return "Heavy Distribution"
    if ud < 0.7:
        return "Distribution Warning"
    if ud <= 1.3:
        return "Neutral"
    if ud <= 1.5:
        return "Accumulating"
    return "Strong Accumulation"


def breakout(stage: str, close: float, high52: float, vol_ratio: float) -> bool:
    return stage == "Stage 2 — Advancing" and near_52w_high(close, high52) and vol_ratio > 1.5


def breakout_confirmed(stage: str, close: float, high52: float, vol_ratio: float, ud: float) -> bool:
    return breakout(stage, close, high52, vol_ratio) and ud > 1.3


# --- v2.2: pre-breakout structure -------------------------------------------
#
# Every threshold below is a single named constant, because §5.1 records that
# the trend-template figures are transcribed from a published source and must be
# verifiable against it, and §10.5 records that the contraction thresholds are
# ours rather than the book's. Neither claim survives if the numbers are buried
# in expressions.

#: §5.1 — Minervini trend template. Transcribed, not derived.
TREND_TEMPLATE_LOW_MULTIPLE = 1.30      # TT6: >= 30% above the 52-week low
TREND_TEMPLATE_HIGH_FRACTION = 0.75     # TT7: within 25% of the 52-week high
TREND_TEMPLATE_MIN_RS = 70.0            # TT8
TREND_TEMPLATE_RISING_SESSIONS = 21     # TT3: "at least one month"

#: §4.1 — RS line.
RS_LINE_HIGH_TOLERANCE = 0.005          # "at a new high" without float equality
RS_LINE_PRICE_GAP_PCT = -5.0            # ours: price demonstrably off its high
RS_LINE_MIN_OVERLAP = 200               # sessions of stock/benchmark overlap

#: §10.4 / §10.5 / §10.6 — volatility, contraction, dry-up, pivot. All ours.
ATR_SESSIONS = 14
VCP_BASE_SESSIONS = 50
VCP_BLOCKS = 5
VCP_CONTRACTION_RATIO_MAX = 0.60
VCP_VOLUME_DRYUP_MAX = 0.80
VCP_MIN_CONTRACTIONS = 2

#: §10.5.1 — the source puts a VCP at two to six contractions. This is not a
#: display cap: a base yielding more than six has not been read badly, it is not
#: the pattern. Enforcing it is a structural claim, not a tuned parameter.
VCP_MAX_CONTRACTIONS = 6

#: §10.5.1 — depth bounds, from the source. It states that most constructive
#: setups correct between 10% and 35%, and that a stock down 60% or more is
#: rarely worth buying because the overhead supply above it is punishing. The
#: upper gate is the constructive bound; the rejection bound is recorded
#: separately because the two carry different weight in the source.
VCP_MAX_BASE_DEPTH_PCT = 35.0
VCP_REJECT_BASE_DEPTH_PCT = 60.0
VOLUME_DRYUP_RECENT = 10
VOLUME_DRYUP_BASELINE = 50

#: §11.1 — Stage 1 readiness.
STAGE1_SLOPE_MIN = -0.10
STAGE1_RS_MIN = 50.0
STAGE1_CONTRACTION_MAX = 0.70
STAGE1_DRYUP_MAX = 0.90


def _position(index: pd.DatetimeIndex, end: pd.Timestamp) -> int:
    """Index position of the latest session at or before ``end``."""
    return index.searchsorted(pd.Timestamp(end), side="right") - 1


def sma(close: pd.Series, end: pd.Timestamp, sessions: int) -> float:
    """Session-based simple moving average ending at ``end`` inclusive.

    Deliberately distinct from :func:`ma_calendar_weeks`. §5 locks the 30-week
    average as a calendar-week construction; §5.1's criteria are stated by their
    author in trading sessions. Thirty calendar weeks is not 150 sessions, and
    collapsing the two would restate one author's rule in another's units.

    Sessions without a Close are dropped before the window is taken, so this is
    the mean of the latest ``sessions`` closes that exist. Averaging whatever
    survives inside a fixed slice would report the mean of 199 observations as a
    200-session average, which §3 forbids: the shortfall would be invisible.
    A calendar-window average has no such problem — its bounds are dates, so
    skipping a gap changes nothing — which is why only the session-count
    averages need this.
    """
    c = close.sort_index().astype(float).dropna()
    pos = _position(c.index, end)
    if pos + 1 < sessions:
        raise ValueError(f"Insufficient history for a {sessions}-session average")
    return float(c.iloc[pos + 1 - sessions : pos + 1].mean())


def sma_series(close: pd.Series, sessions: int) -> pd.Series:
    """The same average at every session, for slope tests and charting.

    Missing closes are dropped first, for the reason given in :func:`sma`.
    """
    valid = close.sort_index().astype(float).dropna()
    return valid.rolling(sessions, min_periods=sessions).mean()


def sma_rising(close: pd.Series, end: pd.Timestamp, sessions: int, over: int) -> bool:
    """True when the ``sessions``-session average is higher than it was ``over`` ago."""
    series = sma_series(close, sessions).dropna()
    if series.empty:
        raise ValueError(f"Insufficient history for a {sessions}-session average")
    pos = _position(series.index, end)
    if pos < over:
        raise ValueError("Insufficient history to measure the average's direction")
    return bool(series.iloc[pos] > series.iloc[pos - over])


def trend_template(
    close: float,
    sma_50: float,
    sma_150: float,
    sma_200: float,
    sma_200_rising: bool,
    low_52w: float,
    high_52w: float,
    rs: float,
) -> dict[str, bool | int]:
    """§5.1 — the eight criteria, each reported separately.

    A count alone cannot distinguish a stock failing only on RS from one failing
    on six, so every criterion is published and the count is derived from them.
    Any non-finite input fails its own criterion rather than poisoning the rest.
    """
    def ok(value: float) -> bool:
        return bool(np.isfinite(value))

    tt = {
        "TT1_Above_150_200": ok(close) and ok(sma_150) and ok(sma_200)
        and close > sma_150 and close > sma_200,
        "TT2_150_Above_200": ok(sma_150) and ok(sma_200) and sma_150 > sma_200,
        "TT3_200_Rising": bool(sma_200_rising),
        "TT4_50_Above_150_200": ok(sma_50) and ok(sma_150) and ok(sma_200)
        and sma_50 > sma_150 and sma_50 > sma_200,
        "TT5_Above_50": ok(close) and ok(sma_50) and close > sma_50,
        "TT6_Above_52W_Low": ok(close) and ok(low_52w)
        and close >= low_52w * TREND_TEMPLATE_LOW_MULTIPLE,
        "TT7_Near_52W_High": ok(close) and ok(high_52w)
        and close >= high_52w * TREND_TEMPLATE_HIGH_FRACTION,
        "TT8_RS": ok(rs) and rs >= TREND_TEMPLATE_MIN_RS,
    }
    result: dict[str, bool | int] = {k: bool(v) for k, v in tt.items()}
    result["Trend_Template_Score"] = int(sum(result.values()))
    result["Trend_Template_Pass"] = bool(result["Trend_Template_Score"] == len(tt))
    return result


def rs_line(close: pd.Series, benchmark: pd.Series) -> pd.Series:
    """§4.1 — Close / Benchmark_Close on the sessions the two actually share.

    Inner join, never a fill: a benchmark session the stock did not trade, or a
    stock session the index did not, is dropped. Manufacturing either side's
    price to complete the calendar would invent the very quantity being measured.
    """
    c, b = close.sort_index().astype(float).align(
        benchmark.sort_index().astype(float), join="inner"
    )
    valid = c.notna() & b.notna() & (b != 0)
    return (c[valid] / b[valid]).astype(float)


def rs_line_high_52w(line: pd.Series, end: pd.Timestamp, min_sessions: int = RS_LINE_MIN_OVERLAP) -> float:
    """Highest RS line value across the trailing 52 calendar weeks."""
    window = calendar_window(line, end, 52)
    if len(window) < min_sessions:
        raise ValueError("Insufficient stock/benchmark overlap for a 52-week RS line high")
    return float(window.max())


def rs_line_at_high(line_value: float, line_high: float) -> bool:
    """Within tolerance of the 52-week RS line high."""
    if not (np.isfinite(line_value) and np.isfinite(line_high)) or line_high == 0:
        return False
    return bool(line_value >= line_high * (1.0 - RS_LINE_HIGH_TOLERANCE))


def rs_line_nh_before_price(line_value: float, line_high: float, pct_from_52w_high: float) -> bool:
    """§4.1 — relative strength at a new high while price is not.

    The ordering is the signal. A stock making new price highs is already
    advancing and is O'Neil's breakout, not his leading tell; the case worth
    naming is strength leading price out of a base.
    """
    if not np.isfinite(pct_from_52w_high):
        return False
    return rs_line_at_high(line_value, line_high) and bool(
        pct_from_52w_high <= RS_LINE_PRICE_GAP_PCT
    )


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range; undefined on the first session, which has no prior close."""
    h, l = high.sort_index().astype(float), low.sort_index().astype(float)
    c = close.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    h, c = h.align(c, join="inner")
    l = l.reindex(h.index)
    prev = c.shift(1)
    return pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)


def atr_pct(high: pd.Series, low: pd.Series, close: pd.Series, end: pd.Timestamp) -> float:
    """§10.4 — ATR(14) as a percentage of the closing price."""
    tr = true_range(high, low, close).dropna()
    pos = _position(tr.index, end)
    if pos + 1 < ATR_SESSIONS:
        raise ValueError("Insufficient history for ATR(14)")
    atr = float(tr.iloc[pos + 1 - ATR_SESSIONS : pos + 1].mean())
    c = close.sort_index().astype(float)
    last = float(c.iloc[_position(c.index, end)])
    if not np.isfinite(last) or last == 0:
        return float("nan")
    return atr / last * 100.0


def range_blocks(
    high: pd.Series, low: pd.Series, close: pd.Series, end: pd.Timestamp
) -> list[float]:
    """§10.5 — each block's high-low range as a percentage of its mean close.

    Five consecutive ten-session blocks across the fifty-session base, oldest
    first. Fixed blocks rather than detected swings: swing detection needs its
    own tunable definition of a swing, and a second undocumented parameter set
    inside a pattern the source already leaves qualitative is exactly what §1
    warns against.
    """
    h, l = high.sort_index().astype(float), low.sort_index().astype(float)
    c = close.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    c = c.reindex(h.index)
    pos = _position(h.index, end)
    if pos + 1 < VCP_BASE_SESSIONS:
        raise ValueError("Insufficient history for the contraction base")
    size = VCP_BASE_SESSIONS // VCP_BLOCKS
    start = pos + 1 - VCP_BASE_SESSIONS
    out: list[float] = []
    for block in range(VCP_BLOCKS):
        lo = start + block * size
        window_h, window_l = h.iloc[lo : lo + size], l.iloc[lo : lo + size]
        mean_close = float(c.iloc[lo : lo + size].mean())
        if not np.isfinite(mean_close) or mean_close == 0:
            return [float("nan")] * VCP_BLOCKS
        out.append((float(window_h.max()) - float(window_l.min())) / mean_close * 100.0)
    return out


def vcp_contractions(blocks: list[float]) -> int:
    """How many times the range tightened against the block before it."""
    return int(
        sum(
            1
            for a, b in zip(blocks, blocks[1:])
            if np.isfinite(a) and np.isfinite(b) and b < a
        )
    )


def contraction_ratio(blocks: list[float]) -> float:
    """Final block's range against the first: below 1 means the base is tightening."""
    if len(blocks) < 2 or not np.isfinite(blocks[0]) or blocks[0] == 0:
        return float("nan")
    if not np.isfinite(blocks[-1]):
        return float("nan")
    return blocks[-1] / blocks[0]


def volume_dryup(volume: pd.Series, end: pd.Timestamp) -> float:
    """§10.5 — recent volume against the longer baseline preceding it.

    The opposite instrument to :func:`volume_ratio`, which compares one session
    to a baseline and so detects the breakout spike. This compares a sustained
    window to a longer one and detects the drought that precedes it.
    """
    v = volume.sort_index().astype(float)
    pos = _position(v.index, end)
    needed = VOLUME_DRYUP_RECENT + VOLUME_DRYUP_BASELINE
    if pos + 1 < needed:
        raise ValueError("Insufficient history for the volume dry-up baseline")
    recent_start = pos + 1 - VOLUME_DRYUP_RECENT
    recent = float(v.iloc[recent_start : pos + 1].mean())
    baseline = float(v.iloc[recent_start - VOLUME_DRYUP_BASELINE : recent_start].mean())
    if not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return recent / baseline


def base_depth_pct(high: pd.Series, low: pd.Series, end: pd.Timestamp) -> float:
    """§10.5.1 — the base's peak-to-trough correction, as a percentage.

    Measured across the base window from its highest High to its lowest Low.
    This is the "how far did it correct" the source gates on, and it is not
    ``Pct_From_52W_High``: a stock can sit close to its 52-week high while its
    base still cut deeply, and vice versa.
    """
    h, l = high.sort_index().astype(float), low.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    pos = _position(h.index, end)
    if pos + 1 < VCP_BASE_SESSIONS:
        raise ValueError("Insufficient history for the base depth")
    window_h = h.iloc[pos + 1 - VCP_BASE_SESSIONS : pos + 1]
    window_l = l.iloc[pos + 1 - VCP_BASE_SESSIONS : pos + 1]
    peak = float(window_h.max())
    trough = float(window_l.min())
    if not np.isfinite(peak) or peak <= 0:
        return float("nan")
    return (peak - trough) / peak * 100.0


def vcp_setup(
    ratio: float,
    dryup: float,
    contractions: int,
    stage: str | None,
    depth_pct: float,
) -> bool:
    """§10.5.1 — a tradeable setup, not merely a contracting range.

    The measurements this composes stay pure: ``Contraction_Ratio``,
    ``Volume_DryUp`` and ``VCP_Contractions`` describe the price structure
    wherever it occurs. This function answers the different question of whether
    the structure is worth acting on, which the source gates two further ways.

    **Stage 2 is required.** The source is emphatic that a base is only
    tradeable inside an established uptrend, and that an attractive base within
    a downtrend is the classic error — the structure looks identical and the
    context inverts its meaning. Without this gate the screen presented
    declining stocks as coiling: 42% of it, on the snapshot that exposed this.

    **The base must not have cut too deep.** A deeply corrected stock carries
    overhead supply that caps the advance a breakout could produce, so the
    source bounds constructive corrections and rejects the deepest outright.
    """
    if not (np.isfinite(ratio) and np.isfinite(dryup)):
        return False
    if not str(stage).startswith("Stage 2"):
        return False
    if not np.isfinite(depth_pct) or depth_pct > VCP_MAX_BASE_DEPTH_PCT:
        return False
    return bool(
        ratio <= VCP_CONTRACTION_RATIO_MAX
        and dryup <= VCP_VOLUME_DRYUP_MAX
        and contractions >= VCP_MIN_CONTRACTIONS
    )


def vcp_pivot(high: pd.Series, end: pd.Timestamp) -> float:
    """§10.6 — the highest high of the base: the buy point at its top."""
    h = high.sort_index().astype(float)
    pos = _position(h.index, end)
    if pos + 1 < VCP_BASE_SESSIONS:
        raise ValueError("Insufficient history for the base pivot")
    return float(h.iloc[pos + 1 - VCP_BASE_SESSIONS : pos + 1].max())


def pct_to_pivot(close: float, pivot: float) -> float:
    """Distance still to travel; zero at the pivot and negative once through it."""
    if not (np.isfinite(close) and np.isfinite(pivot)) or close == 0:
        return float("nan")
    return (pivot / close - 1.0) * 100.0


def stage1_readiness(
    slope_pct: float, rs: float, ratio: float, dryup: float, close: float, ma_10w: float
) -> int:
    """§11.1 — how ready a base looks, counted 0-5.

    Ranking only. No locked signal reads this, and a Stage 1 stock scoring five
    still carries the Stage 1 action.
    """
    checks = (
        np.isfinite(slope_pct) and slope_pct >= STAGE1_SLOPE_MIN,
        np.isfinite(rs) and rs >= STAGE1_RS_MIN,
        np.isfinite(ratio) and ratio <= STAGE1_CONTRACTION_MAX,
        np.isfinite(dryup) and dryup <= STAGE1_DRYUP_MAX,
        np.isfinite(close) and np.isfinite(ma_10w) and close > ma_10w,
    )
    return int(sum(bool(c) for c in checks))


# --- v2.3: swing detection and the technical footprint ----------------------
#
# §10.5.1 replaces §10.5's fixed-block detector with the source's own
# definitions. Contractions are peak-to-trough pullbacks, which means the code
# must first decide what counts as a swing rather than noise.

#: The one number in §10.5.1 that is ours. The source reads swings by eye and
#: never states a reversal threshold. Its tightest worked contraction is 2%, so
#: the threshold has to sit below that or the final contraction — the one that
#: forms the pivot — becomes invisible.
SWING_REVERSAL_PCT = 1.5


def swing_points(
    high: pd.Series, low: pd.Series, threshold_pct: float = SWING_REVERSAL_PCT
) -> list[tuple[pd.Timestamp, float, str]]:
    """Alternating swing highs and lows, by percentage reversal.

    A running extreme is carried forward until price reverses against it by
    ``threshold_pct``; at that point the extreme is confirmed as a swing and the
    direction flips. Peaks are taken from High and troughs from Low, because a
    contraction is measured peak to trough and both live intraday.

    Returned as ``(timestamp, price, kind)`` with ``kind`` in ``{"H", "L"}``,
    strictly alternating. The final running extreme is included as a
    provisional swing: the rightmost contraction is the one that forms the
    pivot, so excluding it until it reverses would hide precisely the structure
    the pattern exists to find.
    """
    h = high.sort_index().astype(float)
    l = low.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    frame = pd.DataFrame({"h": h, "l": l}).dropna()
    if len(frame) < 2:
        return []

    factor = threshold_pct / 100.0
    stamps = list(frame.index)
    highs = frame["h"].to_numpy()
    lows = frame["l"].to_numpy()

    # Direction is unknown at the first bar. Carry both candidate extremes until
    # one of them is broken by the threshold; guessing instead would plant a
    # spurious swing at the left edge of every series.
    direction: str | None = None
    hi_price, hi_at = highs[0], 0
    lo_price, lo_at = lows[0], 0
    out: list[tuple[pd.Timestamp, float, str]] = []

    for i in range(1, len(frame)):
        if direction is None:
            hi_price, hi_at = (highs[i], i) if highs[i] > hi_price else (hi_price, hi_at)
            lo_price, lo_at = (lows[i], i) if lows[i] < lo_price else (lo_price, lo_at)
            if lows[i] <= hi_price * (1 - factor):
                out.append((stamps[hi_at], float(hi_price), "H"))
                direction, lo_price, lo_at = "down", lows[i], i
            elif highs[i] >= lo_price * (1 + factor):
                out.append((stamps[lo_at], float(lo_price), "L"))
                direction, hi_price, hi_at = "up", highs[i], i
            continue

        if direction == "up":
            if highs[i] > hi_price:
                hi_price, hi_at = highs[i], i
            elif lows[i] <= hi_price * (1 - factor):
                out.append((stamps[hi_at], float(hi_price), "H"))
                direction, lo_price, lo_at = "down", lows[i], i
        else:
            if lows[i] < lo_price:
                lo_price, lo_at = lows[i], i
            elif highs[i] >= lo_price * (1 + factor):
                out.append((stamps[lo_at], float(lo_price), "L"))
                direction, hi_price, hi_at = "up", highs[i], i

    if direction == "up":
        out.append((stamps[hi_at], float(hi_price), "H"))
    elif direction == "down":
        out.append((stamps[lo_at], float(lo_price), "L"))
    return out


def contraction_depths(
    swings: list[tuple[pd.Timestamp, float, str]]
) -> list[tuple[pd.Timestamp, float]]:
    """§10.5.1 — peak-to-trough depth of each contraction, oldest first.

    A contraction is a swing high followed by a swing low. Returned as
    ``(swing-high timestamp, depth percent)`` so a caller can locate the
    contraction that formed the pivot, not merely count how many there were.
    """
    out: list[tuple[pd.Timestamp, float]] = []
    for (stamp, peak, kind), (_, trough, next_kind) in zip(swings, swings[1:]):
        if kind == "H" and next_kind == "L" and peak > 0:
            out.append((stamp, (peak - trough) / peak * 100.0))
    return out


#: §10.5.1 — cascade parameters. The source describes each contraction as
#: roughly half the previous, so the sensitivity used to find contraction i+1 is
#: derived from the depth of contraction i rather than fixed. A sweep against
#: two of the source's own footprints showed no single fixed threshold can work:
#: NFLX's deepest leg (27%) needs ~15% sensitivity while its tightest (7%) needs
#: ~8% or finer, and the contraction count needs something in between.
CASCADE_INITIAL_PCT = 15.0
CASCADE_RATIO = 0.25
CASCADE_FLOOR_PCT = 1.5


#: §10.5.1 — the base runs 3 to 65 weeks. Outside that the structure is not a
#: VCP: too short has not digested supply, too long has stopped being a pause.
VCP_MIN_BASE_WEEKS = 3
VCP_MAX_BASE_WEEKS = 65


def vcp_footprint(
    high: pd.Series,
    low: pd.Series,
    end: pd.Timestamp,
    initial_pct: float = CASCADE_INITIAL_PCT,
    ratio: float = CASCADE_RATIO,
) -> dict:
    """§10.5.1 — the base's technical footprint, in the source's own terms.

    Returns ``Base_Weeks``, ``Deepest_Pct``, ``Tightest_Pct``, ``Contractions``
    and ``VCP_Pivot``, or an all-unavailable dict when no base qualifies.

    The base begins at the absolute high the stock came off — not at a fixed
    number of sessions back. That single change is what §10.5 got most wrong:
    a window fixed at ten weeks cannot represent a six-week base or a
    forty-week one, and the source's own worked examples span both.

    The pivot is the high that opens the final, tightest contraction. It is not
    the highest point of the base, which is where the base *started*.
    """
    absent = {
        "Base_Weeks": float("nan"),
        "Deepest_Pct": float("nan"),
        "Tightest_Pct": float("nan"),
        "Contractions": 0,
        "VCP_Pivot": float("nan"),
    }

    h = high.sort_index().astype(float)
    l = low.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    stop = pd.Timestamp(end)
    start = stop - pd.Timedelta(weeks=VCP_MAX_BASE_WEEKS)
    h, l = h.loc[(h.index > start) & (h.index <= stop)], l.loc[(l.index > start) & (l.index <= stop)]
    if len(h) < 2:
        return dict(absent)

    found = cascading_contractions(
        h, l, stop, initial_pct=initial_pct, ratio=ratio
    )
    if not found or len(found) > VCP_MAX_CONTRACTIONS:
        return dict(absent)

    base_start = found[0][0]
    weeks = (stop - base_start).days / 7.0
    if not (VCP_MIN_BASE_WEEKS <= weeks <= VCP_MAX_BASE_WEEKS):
        return dict(absent)

    # The pivot is the high opening the final, tightest contraction — not the
    # top of the base, which is where the base began.
    pivot_stamp, pivot_price, tightest = found[-1]
    return {
        "Base_Weeks": float(weeks),
        "Deepest_Pct": float(max(d for _, _, d in found)),
        "Tightest_Pct": float(tightest),
        "Contractions": int(len(found)),
        "VCP_Pivot": float(pivot_price),
    }


def footprint_label(fp: dict) -> str:
    """Render a footprint the way the source writes it: ``40W 31/3 4T``."""
    weeks, deep, tight = fp.get("Base_Weeks"), fp.get("Deepest_Pct"), fp.get("Tightest_Pct")
    if not all(np.isfinite(float(v)) for v in (weeks, deep, tight) if v is not None):
        return "—"
    if any(v is None for v in (weeks, deep, tight)):
        return "—"
    return f"{round(weeks)}W {round(deep)}/{round(tight)} {int(fp['Contractions'])}T"


def cascading_contractions(
    high: pd.Series,
    low: pd.Series,
    end: pd.Timestamp,
    initial_pct: float = CASCADE_INITIAL_PCT,
    ratio: float = CASCADE_RATIO,
    floor_pct: float = CASCADE_FLOOR_PCT,
) -> list[tuple[pd.Timestamp, float, float]]:
    """Contractions found with sensitivity that tightens as they shrink.

    Returns ``(peak timestamp, peak price, depth percent)`` oldest first.

    A fixed-threshold zigzag cannot read this pattern. Coarse enough to keep a
    27% leg intact, it steps over the 3% leg that forms the pivot; fine enough
    to see the 3%, it shatters the 27% into a dozen fragments — which inflates
    the count and deflates the deepest reading at the same time. Measured
    against the source's own footprints, every fixed value failed differently.

    So the threshold adapts: the search for each contraction is scaled to the
    depth of the one before it, which is how the source describes reading them.
    The first contraction uses ``initial_pct``; each subsequent search uses the
    previous depth times ``ratio``, floored so a base of ever-tinier wiggles
    cannot run away.
    """
    h = high.sort_index().astype(float)
    l = low.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    stop = pd.Timestamp(end)
    window = pd.DataFrame({"h": h, "l": l}).loc[:stop].dropna()
    if len(window) < 3:
        return []

    stamps = list(window.index)
    highs = window["h"].to_numpy()
    lows = window["l"].to_numpy()

    # The base opens at the absolute high the stock comes off.
    start = int(highs.argmax())
    if start >= len(highs) - 2:
        return []

    out: list[tuple[pd.Timestamp, float, float]] = []
    peak, peak_at = highs[start], start
    trough = lows[start]
    threshold = initial_pct
    seeking_trough = True

    for j in range(start + 1, len(window)):
        if seeking_trough:
            if lows[j] < trough:
                trough = lows[j]
            if peak > 0 and highs[j] >= trough * (1 + threshold / 100.0):
                depth = (peak - trough) / peak * 100.0
                # The pattern is defined by contractions that shrink. A deeper
                # correction than the one before it is not a continuation of
                # this base — it is a new, deeper base beginning, so the
                # sequence restarts there rather than accumulating.
                if out and depth >= out[-1][2]:
                    out = [(stamps[peak_at], float(peak), float(depth))]
                else:
                    out.append((stamps[peak_at], float(peak), float(depth)))
                threshold = max(floor_pct, depth * ratio)
                seeking_trough = False
                peak, peak_at = highs[j], j
        else:
            if highs[j] > peak:
                peak, peak_at = highs[j], j
            if lows[j] <= peak * (1 - threshold / 100.0):
                seeking_trough = True
                trough = lows[j]

    # The contraction in progress at T is the one that forms the pivot, so it is
    # reported rather than withheld until it resolves.
    if seeking_trough and peak > trough > 0:
        depth = (peak - trough) / peak * 100.0
        if out and depth >= out[-1][2]:
            out = [(stamps[peak_at], float(peak), float(depth))]
        else:
            out.append((stamps[peak_at], float(peak), float(depth)))
    return out


# --- v2.3: recovery-gated contractions --------------------------------------
#
# Attempts one through four all asked "how large must a reversal be to count?"
# and answered with a threshold — fixed, then cascading, then bounded. All four
# over-segmented real bases in the same direction: 26 and 44 contractions where
# the source reads 3. Over-counting in one direction is not a tuning error, it
# is the wrong question.
#
# Inside a single 27% correction there are dozens of small counter-rallies. A
# reversal detector sees each as a swing. They are all sub-structure of ONE
# contraction, because price never climbed back near the prior high in between.
# What separates one contraction from the next is not the size of a reversal but
# RECOVERY: price must retrace most of the decline before a new contraction can
# begin. That is a structural rule about where a contraction ends, and it acts
# on the segmentation itself rather than filtering its output afterwards — which
# is why attempt four's structural bounds could only reject a base that had
# already been shattered into 44 pieces.

#: Fraction of a decline price must retrace before the next contraction starts.
#: Ours: the source reads contractions by eye and states no such number. Set
#: from its own description of bases whose rallies recover most of each decline.
VCP_RECOVERY_FRACTION = 0.70

#: A correction shallower than this is noise, not a contraction. Below the
#: tightest contraction the source reports, so it cannot hide a real leg.
VCP_MIN_CONTRACTION_PCT = 1.0


def recovery_contractions(
    high: pd.Series,
    low: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    recovery: float = VCP_RECOVERY_FRACTION,
    min_pct: float = VCP_MIN_CONTRACTION_PCT,
) -> list[tuple[pd.Timestamp, float, float, float]]:
    """Segment a base into contractions by recovery rather than by sensitivity.

    Returns one tuple per completed contraction: the peak's session, the peak,
    the trough, and the depth as a percentage of the peak. A contraction is
    open from its peak until price retraces ``recovery`` of the decline; only
    then can the next one begin, so counter-rallies inside a decline cannot
    fragment it.
    """
    h = high.sort_index().astype(float)
    l = low.sort_index().astype(float)
    h, l = h.align(l, join="inner")
    window = (h.index >= pd.Timestamp(start)) & (h.index <= pd.Timestamp(end))
    h, l = h[window], l[window]
    if len(h) < 2:
        return []

    stamps = h.index
    out: list[tuple[pd.Timestamp, float, float, float]] = []
    peak, peak_at = float(h.iloc[0]), 0
    trough = float(l.iloc[0])

    for i in range(1, len(h)):
        hi, lo = float(h.iloc[i]), float(l.iloc[i])
        depth = (peak - trough) / peak * 100.0 if peak > 0 else 0.0

        # Still extending the high with no meaningful decline behind it: this is
        # the same leg reaching further, not a new one.
        if hi >= peak and depth < min_pct:
            peak, peak_at, trough = hi, i, lo
            continue

        if lo < trough:
            trough = lo

        depth = (peak - trough) / peak * 100.0 if peak > 0 else 0.0
        if depth < min_pct:
            continue

        # Price has retraced enough of the decline for this contraction to be
        # complete. The next one starts here.
        if hi >= trough + recovery * (peak - trough):
            out.append((stamps[peak_at], peak, trough, depth))
            peak, peak_at, trough = hi, i, lo

    return out


def final_contraction_pivot(
    high: pd.Series,
    low: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, float]:
    """§10.6 — the pivot at the top of the final contraction, and its depth.

    The source places the pivot at the last contraction's high, not the base's.
    Those differ whenever the base high sits in an early leg. Returns
    ``(nan, nan)`` when no contraction completes, so the caller withholds the
    reading rather than substituting the base high.
    """
    legs = recovery_contractions(high, low, start, end)
    if not legs:
        return float("nan"), float("nan")
    _, peak, _, depth = legs[-1]
    return float(peak), float(depth)
