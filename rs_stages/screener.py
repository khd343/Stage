"""Universe-level RS/Stage calculations on validated pre-market snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DecisionSnapshot
from .quant import (
    atr_pct,
    base_depth_pct,
    breakout,
    breakout_confirmed,
    classify_stage,
    high_52w,
    low_52w,
    ma_10w_series,
    ma_30w,
    ma_30w_series,
    ma_slope_pct,
    max_drawdowns,
    up_days_pct,
    contraction_ratio,
    pct_to_pivot,
    range_blocks,
    rs_blend,
    rs_line,
    rs_line_at_high,
    rs_line_high_52w,
    rs_line_nh_before_price,
    rs_returns,
    rs_score,
    sma,
    sma_rising,
    stage1_readiness,
    trend_template,
    up_down_ratio,
    vcp_contractions,
    vcp_pivot,
    vcp_setup,
    volume_dryup,
    volume_ratio,
    TREND_TEMPLATE_RISING_SESSIONS,
)

#: Columns the presentation layer may read without recomputing anything.
TREND_COLUMNS = ("Close", "MA_10W", "MA_30W", "Volume")

#: The five trend-health conditions, in display order. Each is a locked field or
#: a strict comparison between locked fields; none introduces new methodology.
TREND_HEALTH_CONDITIONS = (
    ("Above_MA_30W", "Close is above the 30-week line"),
    ("MA_30W_Rising", "30-week line is rising"),
    ("MA10W_Above_MA30W", "10-week line is above the 30-week line"),
    ("Above_MA_10W", "Close is above the 10-week line"),
    ("RS_Not_Lagging", "Relative strength is not lagging (RS ≥ 50)"),
)

#: Minervini's eight trend-template conditions (§5.1), in the source's own
#: order. Five are structural comparisons between locked fields; the last
#: three carry numeric thresholds transcribed from the source and labelled
#: provisional in the UI, not retuned or replaced here. See DECISION_LOG
#: D-2.2.2 and FORMULAS.md's "Trend template" section for the exact values.
TREND_TEMPLATE_CONDITIONS = (
    ("TT1_Above_150_200", "Close is above the 150- and 200-session averages"),
    ("TT2_150_Above_200", "150-session average is above the 200-session average"),
    ("TT3_200_Rising", "200-session average is rising"),
    ("TT4_50_Above_150_200", "50-session average is above the 150- and 200-session averages"),
    ("TT5_Above_50", "Close is above the 50-session average"),
    ("TT6_Above_52W_Low", "At least 30% above the 52-week low — provisional threshold"),
    ("TT7_Near_52W_High", "Within 25% of the 52-week high — provisional threshold"),
    ("TT8_RS", "Relative strength is 70 or better — provisional threshold"),
)


def _analyze_symbol(
    symbol: str,
    snap: DecisionSnapshot,
    trend_sessions: int | None,
    benchmark: pd.Series | None = None,
) -> tuple[dict, pd.DataFrame | None]:
    """Calculate one symbol's locked fields and, optionally, its trend series.

    The trend series is a by-product of calculations the row already performs,
    so collecting it costs no additional market data and no additional moving
    average passes.
    """
    data = snap.data
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    volume = data["Volume"].astype(float)
    t = snap.latest_completed_session
    row: dict = {"Symbol": symbol, "Date": t}

    # Latest completed-session close. Every price-derived presentation field
    # below is traceable to this single observation.
    try:
        row["Close"] = float(close.loc[t])
    except (KeyError, TypeError, ValueError):
        row["Close"] = float("nan")

    try:
        returns = rs_returns(close, t)
        row.update({f"R{m}M": returns[m] for m in returns})
        row["RS_Blend"] = rs_blend(returns)
    except (ValueError, KeyError):
        row["RS_Blend"] = float("nan")

    # What each of those returns COST. The same windows deliberately: a +32%
    # that gave back 35% on the way is a different holding from a +32% that
    # gave back 9%, and nothing else on the row separates them. Display and
    # sort only -- neither feeds RS_Blend, the Stage, or the Action.
    row.update({f"MaxDD_{m}M": v for m, v in max_drawdowns(close, t).items()})
    row["Up_Days_Pct_6M"] = up_days_pct(close, t)

    ma_30w_full = ma_10w_full = None
    try:
        ma = ma_30w(close, t)
        ma_30w_full = ma_30w_series(close.loc[:t])
        slope = ma_slope_pct(ma_30w_full, t, sessions=10)
        row["MA_30W"] = ma
        row["MA_30W_Slope_10S_Pct"] = slope
        row["Stage"] = classify_stage(float(close.loc[t]), ma, slope)
    except (ValueError, KeyError):
        row["MA_30W"] = float("nan")
        row["MA_30W_Slope_10S_Pct"] = float("nan")
        row["Stage"] = None

    # 10-calendar-week MA — locked-spec v2.1. Same calendar-window construction
    # as the 30-week line so the two are directly comparable.
    try:
        ma_10w_full = ma_10w_series(close.loc[:t])
        row["MA_10W"] = float(ma_10w_full.loc[t])
        if not np.isfinite(row["MA_10W"]):
            row["MA_10W"] = float("nan")
    except (ValueError, KeyError):
        row["MA_10W"] = float("nan")

    try:
        h52 = high_52w(high, t)
        row["High_52W"] = h52
        row["Near_52W_High"] = bool(close.loc[t] >= 0.97 * h52)
    except (ValueError, KeyError):
        row["High_52W"] = float("nan")
        row["Near_52W_High"] = False

    # 52-week low mirrors the 52-week high window exactly. It is a range input
    # only; when the provider frame carries no Low column the field stays NaN
    # rather than being substituted with Close.
    if "Low" in data.columns:
        try:
            row["Low_52W"] = low_52w(data["Low"].astype(float), t)
        except (ValueError, KeyError):
            row["Low_52W"] = float("nan")
    else:
        row["Low_52W"] = float("nan")

    try:
        row["Volume_Ratio"] = volume_ratio(volume, t)
    except (ValueError, KeyError):
        row["Volume_Ratio"] = float("nan")

    try:
        row["U_D"] = up_down_ratio(close, volume, t)
    except (ValueError, KeyError):
        row["U_D"] = float("nan")

    value = (close * volume).loc[:t].dropna()
    row["AvgValue20"] = float(value.iloc[-20:].mean()) if len(value) >= 20 else np.nan

    try:
        close_50 = close.loc[:t].dropna()
        row["SMA_50"] = float(close_50.iloc[-50:].mean()) if len(close_50) >= 50 else np.nan
        row["Below_50DMA"] = bool(close.loc[t] < row["SMA_50"]) if pd.notna(row["SMA_50"]) else False
    except (ValueError, KeyError):
        row["SMA_50"] = np.nan
        row["Below_50DMA"] = False

    # Ext_Pct is the displayed extension above the 30-week line. The locked
    # Extended_20Pct condition keeps its specified form (Close > 1.20 × MA_30W)
    # rather than being re-derived from Ext_Pct: the two are algebraically
    # equivalent but not bit-identical in floating point, and the locked
    # comparison is the authority.
    row["Ext_Pct"] = (
        (row["Close"] / row["MA_30W"] - 1.0) * 100.0
        if pd.notna(row["Close"]) and pd.notna(row["MA_30W"]) and row["MA_30W"] != 0
        else np.nan
    )
    row["Extended_20Pct"] = bool(
        pd.notna(row["MA_30W"]) and float(close.loc[t]) > 1.20 * float(row["MA_30W"])
    )
    row["Pct_From_52W_High"] = (
        (row["Close"] / row["High_52W"] - 1.0) * 100.0
        if pd.notna(row["Close"]) and pd.notna(row["High_52W"]) and row["High_52W"] != 0
        else np.nan
    )

    row["Above_MA_30W"] = bool(pd.notna(row["Close"]) and pd.notna(row["MA_30W"]) and row["Close"] > row["MA_30W"])
    row["Above_MA_10W"] = bool(pd.notna(row["Close"]) and pd.notna(row["MA_10W"]) and row["Close"] > row["MA_10W"])
    row["MA10W_Above_MA30W"] = bool(
        pd.notna(row["MA_10W"]) and pd.notna(row["MA_30W"]) and row["MA_10W"] > row["MA_30W"]
    )
    row["MA_30W_Rising"] = bool(
        pd.notna(row["MA_30W_Slope_10S_Pct"]) and row["MA_30W_Slope_10S_Pct"] > 0.0
    )

    row["Distribution"] = bool(pd.notna(row["U_D"]) and row["U_D"] < 0.7)
    row["Heavy_Distribution"] = bool(pd.notna(row["U_D"]) and row["U_D"] < 0.6)

    row["Breakout"] = (
        breakout(
            row.get("Stage"),
            float(close.loc[t]),
            row.get("High_52W", float("nan")),
            row.get("Volume_Ratio", float("nan")),
        )
        if row.get("Stage")
        else False
    )
    row["Breakout_Confirmed"] = (
        breakout_confirmed(
            row.get("Stage"),
            float(close.loc[t]),
            row.get("High_52W", float("nan")),
            row.get("Volume_Ratio", float("nan")),
            row.get("U_D", float("nan")),
        )
        if row.get("Stage")
        else False
    )

    row.update(_prebreakout(data, close, high, volume, t, benchmark, row.get("Stage")))
    row["Pct_To_Pivot"] = pct_to_pivot(row["Close"], row["VCP_Pivot"])

    trend = None
    if trend_sessions is not None:
        frame = pd.DataFrame(
            {
                "Close": close.loc[:t],
                "MA_10W": ma_10w_full if ma_10w_full is not None else np.nan,
                "MA_30W": ma_30w_full if ma_30w_full is not None else np.nan,
                "Volume": volume.loc[:t],
            }
        )
        trend = frame.tail(trend_sessions) if trend_sessions > 0 else frame

    return row, trend


def _prebreakout(
    data: pd.DataFrame,
    close: pd.Series,
    high: pd.Series,
    volume: pd.Series,
    t: pd.Timestamp,
    benchmark: pd.Series | None,
    stage: str | None = None,
) -> dict:
    """v2.2 §4.1, §5.1, §10.4-10.6 fields that need no cross-sectional input.

    Every field degrades on its own. A symbol without enough history for a
    200-session average still reports its contraction, and one whose provider
    frame carries no Low still reports its trend template. Nothing here
    substitutes a value it could not compute.
    """
    row: dict = {}

    # Session-based averages. Deliberately not MA_30W: see §5.1.
    for sessions, field in ((150, "SMA_150"), (200, "SMA_200")):
        try:
            row[field] = sma(close, t, sessions)
        except (ValueError, KeyError):
            row[field] = float("nan")
    try:
        row["SMA_200_Rising"] = sma_rising(close, t, 200, TREND_TEMPLATE_RISING_SESSIONS)
    except (ValueError, KeyError):
        row["SMA_200_Rising"] = False

    # Volatility and contraction both need Low, which §9.1 treats as optional.
    low = data["Low"].astype(float) if "Low" in data.columns else None
    if low is not None:
        try:
            row["ATR_Pct"] = atr_pct(high, low, close, t)
        except (ValueError, KeyError):
            row["ATR_Pct"] = float("nan")
        try:
            blocks = range_blocks(high, low, close, t)
            row["VCP_Contractions"] = vcp_contractions(blocks)
            row["Contraction_Ratio"] = contraction_ratio(blocks)
        except (ValueError, KeyError):
            row["VCP_Contractions"] = 0
            row["Contraction_Ratio"] = float("nan")
        try:
            row["Base_Depth_Pct"] = base_depth_pct(high, low, t)
        except (ValueError, KeyError):
            row["Base_Depth_Pct"] = float("nan")
    else:
        row["ATR_Pct"] = float("nan")
        row["VCP_Contractions"] = 0
        row["Contraction_Ratio"] = float("nan")
        row["Base_Depth_Pct"] = float("nan")

    try:
        row["Volume_DryUp"] = volume_dryup(volume, t)
    except (ValueError, KeyError):
        row["Volume_DryUp"] = float("nan")

    row["VCP_Setup"] = vcp_setup(
        row["Contraction_Ratio"],
        row["Volume_DryUp"],
        row["VCP_Contractions"],
        stage,
        row["Base_Depth_Pct"],
    )

    try:
        row["VCP_Pivot"] = vcp_pivot(high, t)
    except (ValueError, KeyError):
        row["VCP_Pivot"] = float("nan")

    # The RS line needs a benchmark. Absent one the fields are unavailable
    # rather than defaulted: a stock has no relative strength against nothing.
    row["RS_Line"] = float("nan")
    row["RS_Line_High_52W"] = float("nan")
    row["RS_Line_At_High"] = False
    if benchmark is not None:
        try:
            line = rs_line(close.loc[:t], benchmark)
            if not line.empty:
                row["RS_Line"] = float(line.iloc[-1])
                row["RS_Line_High_52W"] = rs_line_high_52w(line, t)
                row["RS_Line_At_High"] = rs_line_at_high(
                    row["RS_Line"], row["RS_Line_High_52W"]
                )
        except (ValueError, KeyError, IndexError):
            pass
    return row


def _finalize(result: pd.DataFrame) -> pd.DataFrame:
    """Apply cross-sectional fields and the trend-health count.

    RS is cross-sectional, so it can only be scored once every symbol's blend
    exists. Trend health depends on RS and is therefore also finalized here.
    """
    if result.empty:
        return result
    result["RS_Score"] = rs_score(result["RS_Blend"])
    result["Liquid_UI_Filter"] = result["AvgValue20"] > 5e7
    result["RS_Not_Lagging"] = (result["RS_Score"] >= 50).astype(bool)
    result["Trend_Health"] = (
        sum(result[field].astype(bool).astype(int) for field, _ in TREND_HEALTH_CONDITIONS)
    ).astype(int)

    # §5.1 and §11.1 both read RS, so like trend health they can only be
    # resolved after the cross-sectional score exists.
    template = pd.DataFrame(
        [
            trend_template(
                close=r.get("Close", float("nan")),
                sma_50=r.get("SMA_50", float("nan")),
                sma_150=r.get("SMA_150", float("nan")),
                sma_200=r.get("SMA_200", float("nan")),
                sma_200_rising=bool(r.get("SMA_200_Rising", False)),
                low_52w=r.get("Low_52W", float("nan")),
                high_52w=r.get("High_52W", float("nan")),
                rs=r.get("RS_Score", float("nan")),
            )
            for r in result.to_dict("records")
        ],
        index=result.index,
    )
    for column in template.columns:
        result[column] = template[column]

    # §4.1 — the divergence is the ordering of two facts already published.
    result["RS_Line_NH_Before_Price"] = [
        rs_line_nh_before_price(line, high, pct)
        for line, high, pct in zip(
            result["RS_Line"], result["RS_Line_High_52W"], result["Pct_From_52W_High"]
        )
    ]

    # §11.1 — defined only for Stage 1, and unavailable rather than zero
    # elsewhere: a score of 0 would read as "a bad base" for a stock that has
    # no base at all.
    # str() at the point of use, not astype(str) on the column: pandas 3.0
    # preserves missing values through astype(str) rather than rendering them
    # "nan", so a symbol with too little history to classify arrives here as a
    # float and Stage.astype(str) does not make it safe to call .startswith on.
    readiness = [
        stage1_readiness(slope, rs, ratio, dryup, close, ma10)
        if str(label).startswith("Stage 1")
        else float("nan")
        for label, slope, rs, ratio, dryup, close, ma10 in zip(
            result["Stage"],
            result["MA_30W_Slope_10S_Pct"],
            result["RS_Score"],
            result["Contraction_Ratio"],
            result["Volume_DryUp"],
            result["Close"],
            result["MA_10W"],
        )
    ]
    result["Stage1_Readiness"] = pd.Series(readiness, index=result.index, dtype="float64")
    return result


def analyze_universe(
    snapshots: dict[str, DecisionSnapshot], benchmark: pd.Series | None = None
) -> pd.DataFrame:
    """Calculate RS/Stage fields before optional liquidity filtering.

    The guide interpretation layer consumes only fields produced here or by
    the validated stock-history path. Missing history remains explicit.
    """
    rows = [
        _analyze_symbol(symbol, snap, None, benchmark)[0]
        for symbol, snap in snapshots.items()
    ]
    return _finalize(pd.DataFrame(rows).set_index("Symbol"))


def analyze_universe_with_trend(
    snapshots: dict[str, DecisionSnapshot],
    trend_sessions: int = 260,
    benchmark: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the locked snapshot plus each symbol's trailing trend series.

    Identical row output to :func:`analyze_universe`; the trend series is the
    per-session Close, 10-week MA, 30-week MA and Volume already computed while
    building the row. It exists so the presentation layer can draw price history
    without re-downloading market data or recomputing locked averages.
    """
    rows: list[dict] = []
    trends: dict[str, pd.DataFrame] = {}
    for symbol, snap in snapshots.items():
        row, trend = _analyze_symbol(symbol, snap, trend_sessions, benchmark)
        rows.append(row)
        if trend is not None:
            trends[str(symbol)] = trend
    return _finalize(pd.DataFrame(rows).set_index("Symbol")), trends
