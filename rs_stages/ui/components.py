"""HTML component builders for the RS-Stages terminal.

Each function returns a self-contained HTML fragment. The component geometry —
the avatar-tile row, the rank-bar cell, the 52-week range track, the stacked
posture bar, the shelf list and the stat card — is adopted from the WealthStar
reference terminal, which is the design source for this project. The data poured
into them is entirely ours.

Every value that originates in data is escaped before it reaches the markup.
Components never invent a value: an unavailable number renders as an em dash.
"""
from __future__ import annotations

import html
import math
from urllib.parse import quote
from typing import Any, Iterable, Sequence

import pandas as pd

from .theme import (
    DASH,
    NEGATIVE,
    POSITIVE,
    action_style,
    fmt_pct,
    fmt_price,
    fmt_return,
    fmt_rs,
    initials,
    signed_color,
    stage_color,
    stage_display,
    stage_key,
    to_float,
)


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def query_href(**params: Any) -> str:
    """Build an in-app link from query parameters.

    Values are URL-encoded before being HTML-escaped. html.escape alone is not
    enough: it turns "&" into "&amp;" for the attribute but leaves it a
    parameter separator for the browser, so an industry named
    "Metals & Mining" would arrive truncated at the ampersand and match nothing.
    """
    query = "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items() if v is not None)
    return html.escape(f"?{query}", quote=True)


def stock_href(symbol: Any) -> str:
    """Link into the Stock view for a symbol."""
    return query_href(view="Stock", symbol=symbol)


# --- small parts ------------------------------------------------------------


def dot(color: str, size: int = 8) -> str:
    return f'<span class="ws-dot" style="width:{size}px;height:{size}px;background:{color}"></span>'


def sparkline(values: Sequence[float], width: int = 84, height: int = 26) -> str:
    """Inline SVG trend line, coloured by net direction over the window.

    Returns an em dash when there is not enough history to draw a line, rather
    than drawing a flat line that would imply a price series we do not have.
    """
    points = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if len(points) < 2:
        return f'<span style="color:var(--faint)">{DASH}</span>'
    low, high = min(points), max(points)
    span = high - low
    step = width / (len(points) - 1)
    if span <= 0:
        coords = " ".join(f"{i * step:.1f},{height / 2:.1f}" for i in range(len(points)))
    else:
        coords = " ".join(
            f"{i * step:.1f},{(height - 2) - ((v - low) / span) * (height - 4):.1f}"
            for i, v in enumerate(points)
        )
    color = POSITIVE if points[-1] >= points[0] else NEGATIVE
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'preserveAspectRatio="none" aria-hidden="true">'
        f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def range_track(low: Any, close: Any, high: Any, width: int | None = 96, caps: bool = True) -> str:
    """52-week range with the close positioned inside it.

    ``width=None`` lets the track fill its container, for the wider stock-page
    variant. Returns an em dash when the range cannot be bounded, rather than
    drawing a dot at an arbitrary position.
    """
    low_v, close_v, high_v = to_float(low), to_float(close), to_float(high)
    if not all(math.isfinite(v) for v in (low_v, close_v, high_v)) or high_v <= low_v:
        return f'<span style="color:var(--faint)">{DASH}</span>'
    position = min(100.0, max(0.0, (close_v - low_v) / (high_v - low_v) * 100.0))
    sizing = f"width:{width}px" if width else "flex:1"
    cap_left = '<span class="ws-range-cap">L</span>' if caps else ""
    cap_right = '<span class="ws-range-cap">H</span>' if caps else ""
    return (
        f'<span class="ws-range" style="{sizing}">{cap_left}'
        f'<span class="ws-range-track">'
        f'<span class="ws-range-dot" style="left:{position:.2f}%"></span></span>'
        f"{cap_right}</span>"
    )


def rs_cell(rs: Any) -> str:
    """RS number over a proportional rank bar."""
    value = to_float(rs)
    if not math.isfinite(value):
        return f'<span style="color:var(--faint)">{DASH}</span>'
    color = stage_color("Stage 2") if value >= 80 else ("#2D6CDF" if value >= 50 else "#9aa1ac")
    return (
        f'<span class="ws-rs num">{fmt_rs(value)}</span>'
        f'<span class="ws-rs-bar bar-grow" style="width:{min(100.0, value):.0f}%;background:{color}"></span>'
    )


def action_chip(action: Any) -> str:
    color, background = action_style(action)
    label = esc(action) if action is not None and str(action) != "nan" else DASH
    return f'<span class="ws-chip" style="background:{background};color:{color}">{label}</span>'


def stage_cell(stage: Any) -> str:
    key = stage_key(stage)
    if key not in {"Stage 1", "Stage 2", "Stage 3", "Stage 4"}:
        return f'<span style="color:var(--faint)">{DASH}</span>'
    return f'<span class="ws-stage">{dot(stage_color(stage))}{esc(stage_display(stage))}</span>'


def symbol_cell(symbol: Any, subtitle: Any, stage: Any) -> str:
    """Avatar tile plus symbol and industry, linking into the Stock view."""
    color = stage_color(stage)
    return (
        f'<a class="ws-sym-cell" target="_self" href="{stock_href(symbol)}">'
        f'<span class="ws-tile" style="background:{color}14;color:{color}">{esc(initials(symbol))}</span>'
        f'<span style="display:flex;flex-direction:column;min-width:0">'
        f'<span class="ws-sym">{esc(symbol)}</span>'
        f'<span class="ws-sym-sub col-hide-sm">{esc(subtitle) if subtitle and str(subtitle) != "nan" else ""}</span>'
        f"</span></a>"
    )


def pill_link(symbol: Any, meta: Any = None) -> str:
    meta_html = f'<span class="meta num">{esc(meta)}</span>' if meta not in (None, "") else ""
    return (
        f'<a class="ws-pill-link pop" target="_self" href="{stock_href(symbol)}">'
        f'<span class="sym">{esc(symbol)}</span>{meta_html}</a>'
    )


def pill_row(items: Iterable[tuple[Any, Any]], empty: str = "Nothing in this group today.") -> str:
    pills = "".join(pill_link(symbol, meta) for symbol, meta in items)
    if not pills:
        return f'<div class="ws-note">{esc(empty)}</div>'
    return f'<div class="ws-chips">{pills}</div>'


# --- cards and shelves ------------------------------------------------------


def card(inner: str, extra_class: str = "", style: str = "") -> str:
    classes = f"ws-card {extra_class}".strip()
    style_attr = f' style="{style}"' if style else ""
    return f'<div class="{classes}"{style_attr}>{inner}</div>'


def stat_card(label: str, value: str, suffix: str = "", note: str = "", color: str = "var(--ink)") -> str:
    suffix_html = f"<small>{esc(suffix)}</small>" if suffix else ""
    note_html = f'<div class="ws-stat-note num">{esc(note)}</div>' if note else ""
    return (
        f'<div class="ws-stat lift"><div class="ws-stat-label">{esc(label)}</div>'
        f'<div class="ws-stat-value num" style="color:{color}">{esc(value)}{suffix_html}</div>'
        f"{note_html}</div>"
    )


def stat_row(cards: Iterable[str]) -> str:
    return f'<div class="ws-stat-row">{"".join(cards)}</div>'


def posture_bar(stage_counts: dict[str, int], title: str = "Stage posture") -> str:
    """Legend plus a stacked bar of the four locked Stages.

    Widths are shares of the classified universe; stocks whose Stage could not
    be classified are excluded from the bar and from its denominator.
    """
    order = ["Stage 2", "Stage 3", "Stage 4", "Stage 1"]
    total = sum(int(stage_counts.get(key, 0)) for key in order)
    legend = "".join(
        f'<span class="item">{dot(stage_color(key))}{esc(stage_display(key).split(" · ")[1])}'
        f'<b class="num">{int(stage_counts.get(key, 0)):,}</b></span>'
        for key in order
    )
    if total <= 0:
        segments = '<div style="width:100%;background:var(--track)"></div>'
    else:
        segments = "".join(
            f'<div class="bar-grow" style="width:{int(stage_counts.get(key, 0)) / total * 100:.4f}%;'
            f'background:{stage_color(key)}"></div>'
            for key in order
        )
    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'flex-wrap:wrap;gap:8px;margin-bottom:2px">'
        f'<span style="font-size:12.5px;font-weight:600;color:var(--sub)">{esc(title)}</span>'
        f'<span class="ws-legend">{legend}</span></div>'
        f'<div class="ws-segbar">{segments}</div>'
    )


def shelf(
    label: str,
    count: int,
    color: str,
    items: Iterable[tuple[Any, Any]],
    limit: int = 12,
    more_href: str = "",
) -> str:
    """One 'what changed' row: label, count badge, and member pills.

    When the group is larger than the row can hold, the overflow is stated
    rather than silently dropped, with a link to where the rest lives.
    """
    all_items = list(items)
    members = all_items[:limit]
    pills = "".join(pill_link(symbol, meta) for symbol, meta in members)
    hidden = len(all_items) - len(members)
    if hidden > 0:
        target = f' href="{more_href}"' if more_href else ""
        tag = "a" if more_href else "span"
        pills += (
            f'<{tag} class="ws-pill-link pop" target="_self"{target} '
            f'style="color:var(--sub)"><span class="sym">+{hidden} more</span></{tag}>'
        )
    return (
        f'<div class="ws-shelf">'
        f'<span class="ws-shelf-label" style="color:{color}">{esc(label)}</span>'
        f'<span class="ws-shelf-count num" style="color:{color};background:{color}18">{int(count)}</span>'
        f'<div class="ws-chips" style="flex:1 1 200px;min-width:0">{pills}</div></div>'
    )


def checklist(items: Sequence[tuple[str, bool]]) -> str:
    """Evidence checklist with a pass/fail mark per condition."""
    rows = []
    for label, passed in items:
        color, background = (POSITIVE, "var(--up-bg)") if passed else (NEGATIVE, "var(--down-bg)")
        mark = "✓" if passed else "✕"
        rows.append(
            f'<div class="ws-check">'
            f'<span class="ws-check-mark" style="background:{background};color:{color}">{mark}</span>'
            f'<span>{esc(label)}</span></div>'
        )
    return "".join(rows)


def kv_grid(items: Sequence[tuple[str, str, str, str]]) -> str:
    """Grid of label / value / note tiles. Items are (label, value, note, colour)."""
    tiles = "".join(
        f'<div><div class="ws-kv-label">{esc(label)}</div>'
        f'<div class="ws-kv-value num" style="color:{color}">{esc(value)}</div>'
        f'<div class="ws-kv-note">{esc(note)}</div></div>'
        for label, value, note, color in items
    )
    return f'<div class="ws-kv">{tiles}</div>'


#: Tone for a boolean condition. "good" means True is favourable evidence,
#: "warn" means True is a caution, "neutral" means True carries no valence.
STATE_TONES = {
    "good": (POSITIVE, "var(--up-bg)"),
    "warn": ("#B5781A", "var(--amber-bg)"),
    "bad": (NEGATIVE, "var(--down-bg)"),
    "neutral": ("var(--sub)", "var(--track)"),
}


def state_pill(value: Any, tone: str = "good", yes: str = "Yes", no: str = "No") -> str:
    """Render a boolean condition as a pill coloured by what it means.

    A flat "Yes" loses the distinction between confirmation and a warning: a
    confirmed breakout and a distribution warning are not the same kind of Yes.
    Only the meaningful state is coloured; its absence stays neutral.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f'<span class="ws-state" style="background:var(--track);color:var(--faint)">{DASH}</span>'
    active = bool(value)
    color, background = STATE_TONES[tone if active else "neutral"]
    return f'<span class="ws-state" style="background:{background};color:{color}">{esc(yes if active else no)}</span>'


def evidence_card(title: str, rows: Sequence[tuple[str, str, str]]) -> str:
    """A titled block of label / value rows, each with its definition beneath.

    Replaces a flat field-value dump: the reader can see which measure a number
    belongs to, and what it means, without consulting the methodology page.
    """
    body = "".join(
        f'<div class="ws-ev"><div class="ws-ev-label">{esc(label)}'
        f'<div class="ws-ev-note">{esc(note)}</div></div>'
        f'<div class="ws-ev-value num">{value}</div></div>'
        for label, value, note in rows
    )
    return card(f'<div class="ws-card-title">{esc(title)}</div>{body}')


def evidence_grid(cards: Iterable[str]) -> str:
    return f'<div class="ws-ev-grid">{"".join(cards)}</div>'


#: Status marks for the threshold table, in the guide's own vocabulary.
STATUS_MARKS = {
    "met": ("✓", POSITIVE, "var(--up-bg)"),
    "unmet": ("✕", NEGATIVE, "var(--down-bg)"),
    "caution": ("■", "#B5781A", "var(--amber-bg)"),
    "neutral": ("·", "var(--sub)", "var(--track)"),
}


def signal_line(label: str, value: str, tone: str = "neutral") -> str:
    """One line of the Signal Card: what the measure is, and what it reads."""
    color = {"good": POSITIVE, "warn": "#B5781A", "bad": NEGATIVE}.get(tone, "var(--ink)")
    return (
        f'<div class="ws-sigline"><span class="ws-sigline-label">{esc(label)}</span>'
        f'<span class="ws-sigline-value" style="color:{color}">{esc(value)}</span></div>'
    )


def signal_note(kind: str, text: str) -> str:
    """A WAIT / conflict / caution note. Absent notes render nothing at all."""
    if not text:
        return ""
    palette = {
        "wait": ("#B5781A", "var(--amber-bg)", "Waiting on"),
        "conflict": ("#2D6CDF", "var(--blue-bg)", "Conflict"),
        "caution": ("#C2562F", "var(--slip-bg)", "Caution"),
        "source": ("var(--sub)", "var(--track)", "Source"),
    }
    color, background, title = palette[kind]
    return (
        f'<div class="ws-signote" style="background:{background};border-color:{color}33">'
        f'<span class="ws-signote-title" style="color:{color}">{title}</span>'
        f'<span class="ws-signote-body">{esc(text)}</span></div>'
    )


def threshold_table(rows: Sequence[Any]) -> str:
    """The guide's Signal / Value / Threshold / Status table, as boxes."""
    body = "".join(
        (
            lambda mark: (
                f'<div class="ws-thr">'
                f'<div class="ws-thr-signal">{esc(r.signal)}'
                f'<div class="ws-thr-note">{esc(r.note)}</div></div>'
                f'<div class="ws-thr-value num">{esc(r.value)}</div>'
                f'<div class="ws-thr-rule">{esc(r.threshold)}</div>'
                f'<div class="ws-thr-status"><span class="ws-state" '
                f'style="background:{mark[2]};color:{mark[1]}">{mark[0]}</span></div></div>'
            )
        )(STATUS_MARKS.get(r.status, STATUS_MARKS["neutral"]))
        for r in rows
    )
    head = (
        '<div class="ws-thr ws-thr-head"><div class="ws-thr-signal">Signal</div>'
        '<div class="ws-thr-value">Value</div><div class="ws-thr-rule">Threshold</div>'
        '<div class="ws-thr-status">Met</div></div>'
    )
    return card(head + body, style="padding:6px 16px 10px")


def missing_notice(title: str, detail: str) -> str:
    """Explicit unavailability. Never a zero, never a placeholder number."""
    return f'<div class="ws-missing"><b>{esc(title)}</b><br>{esc(detail)}</div>'


def corporate_action_notice(flagged: object, when: str, looks_like: str) -> str:
    """Warn that a row's history contains a corporate action, not a price move.

    The engine reports the prices it was given, and yfinance restates splits
    but not demergers. So a demerged parent reads as a catastrophic decline:
    Stage 4, SELL, relative strength in the bottom decile, down 60% from a
    52-week high that is the pre-demerger price. Every one of those numbers is
    arithmetically correct and none of them means what it appears to mean.

    Returns empty for a clean row, and for a row from a snapshot frozen before
    the flag existed -- the app reads published CSVs it did not write.
    """
    if flagged is None or flagged is False:
        return ""
    if isinstance(flagged, float) and flagged != flagged:  # NaN
        return ""
    if flagged is not True and str(flagged).strip().lower() not in {"true", "1"}:
        return ""
    stamp = f" on {esc(str(when))}" if str(when).strip() else ""
    guess = f" It resembles a {esc(str(looks_like))}." if str(looks_like).strip() else ""
    return (
        '<div class="ws-missing"><b>Corporate action in this history'
        f'{stamp}.</b><br>A single session moved more than a circuit limit allows, '
        "so it is not a price move.{guess} The stage, the returns and the 52-week "
        "high are measured across it and are not a decline in the business. "
        "Nothing here has been corrected."
        "</div>"
    ).replace("{guess}", guess)


# --- the dense screener table ----------------------------------------------

#: Column key -> (header label, alignment, hide-on-mobile)
SCREENER_COLUMNS = {
    "symbol": ("Stock", "left", False),
    "trend": ("3M trend", "left", True),
    "rs": ("RS", "left", False),
    "stage": ("Stage", "left", False),
    "action": ("Action", "left", False),
    "ud": ("U/D", "right", True),
    "ext": ("Ext %", "right", True),
    "range": ("52W range", "left", True),
    "r3m": ("3M", "right", True),
    # --- v2.2 pre-breakout structure ---
    "evidence": ("Evidence", "left", False),
    "rsline": ("RS line", "left", False),
    "contraction": ("Contraction", "right", True),
    "dryup": ("Vol dry-up", "right", True),
    "pivot": ("To pivot", "right", False),
    "template": ("Template", "right", True),
    "readiness": ("Readiness", "right", False),
    "atr": ("ATR %", "right", True),
}

#: The pre-breakout view's columns. Deliberately omits Action: §11 assigns every
#: Stage 1 stock the same label, so showing it beside a readiness ranking would
#: imply the label were varying with the score. It is not.
SETUP_COLUMNS = (
    "symbol", "evidence", "trend", "rs", "stage",
    "rsline", "contraction", "dryup", "pivot", "readiness",
)

#: The four published pre-breakout conditions, in the order the Setups view
#: lists them. Evidence is how many a stock satisfies at once.
SETUP_CONDITIONS = (
    ("Trend template", "Trend_Template_Pass"),
    ("RS leading price", "RS_Line_NH_Before_Price"),
    ("Contracting base", "VCP_Setup"),
    ("Stage 1 ready", "Stage1_Readiness"),
)


def _cell(key: str, row: pd.Series, trend: Sequence[float] | None) -> str:
    if key == "symbol":
        return symbol_cell(row.get("Symbol"), row.get("Industry"), row.get("Stage"))
    if key == "trend":
        return sparkline(trend) if trend is not None else f'<span style="color:var(--faint)">{DASH}</span>'
    if key == "rs":
        return rs_cell(row.get("RS_Score"))
    if key == "stage":
        return stage_cell(row.get("Stage"))
    if key == "action":
        return action_chip(row.get("Action"))
    if key == "ud":
        value = to_float(row.get("U_D"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        text = "∞" if math.isinf(value) else f"{value:.2f}×"
        color = POSITIVE if value > 1.3 else (NEGATIVE if value < 0.7 else "var(--sub)")
        return f'<span class="num" style="color:{color};font-weight:600">{text}</span>'
    if key == "ext":
        value = to_float(row.get("Ext_Pct"))
        color = "#B5781A" if math.isfinite(value) and value > 20 else signed_color(value)
        return f'<span class="num" style="color:{color};font-weight:600">{fmt_pct(value)}</span>'
    if key == "range":
        return range_track(row.get("Low_52W"), row.get("Close"), row.get("High_52W"))
    if key == "r3m":
        value = row.get("R3M")
        return (
            f'<span class="num" style="color:{signed_color(to_float(value) )};font-weight:600">'
            f"{fmt_return(value)}</span>"
        )

    # --- v2.2 ---------------------------------------------------------------
    if key == "rsline":
        # The divergence is the point, not the ratio's magnitude: a raw
        # Close/Index number means nothing to a reader on its own.
        if bool(row.get("RS_Line_NH_Before_Price")):
            return (
                f'<span style="color:{POSITIVE};font-weight:700">Leading</span>'
                '<div class="ws-sub">new high before price</div>'
            )
        if bool(row.get("RS_Line_At_High")):
            return '<span style="color:var(--ink);font-weight:600">At high</span>'
        if math.isnan(to_float(row.get("RS_Line"))):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        return '<span style="color:var(--sub)">—</span>'
    if key == "contraction":
        value = to_float(row.get("Contraction_Ratio"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        color = POSITIVE if value <= 0.60 else ("var(--sub)" if value < 1.0 else NEGATIVE)
        return f'<span class="num" style="color:{color};font-weight:600">{value:.2f}×</span>'
    if key == "dryup":
        value = to_float(row.get("Volume_DryUp"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        color = POSITIVE if value <= 0.80 else ("var(--sub)" if value < 1.2 else NEGATIVE)
        return f'<span class="num" style="color:{color};font-weight:600">{value:.2f}×</span>'
    if key == "pivot":
        value = to_float(row.get("Pct_To_Pivot"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        if value <= 0:
            return f'<span class="num" style="color:{POSITIVE};font-weight:600">through</span>'
        color = POSITIVE if value <= 3.0 else "var(--sub)"
        return f'<span class="num" style="color:{color};font-weight:600">+{value:.1f}%</span>'
    if key == "template":
        value = to_float(row.get("Trend_Template_Score"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        color = POSITIVE if value == 8 else ("var(--ink)" if value >= 6 else "var(--sub)")
        return f'<span class="num" style="color:{color};font-weight:600">{int(value)}/8</span>'
    if key == "evidence":
        value = to_float(row.get("Setup_Evidence"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        filled = int(value)
        # Bars rather than the readiness dots: both are counts and they sit in
        # the same row, so they must not be mistaken for one another.
        bars = "".join(
            f'<span style="display:inline-block;width:4px;height:12px;margin-right:2px;'
            f'border-radius:1px;background:{POSITIVE if i < filled else "var(--line)"}"></span>'
            for i in range(len(SETUP_CONDITIONS))
        )
        color = POSITIVE if filled >= 2 else "var(--sub)"
        return (
            f'<span class="num" style="font-weight:600;color:{color}" '
            f'title="{filled} of {len(SETUP_CONDITIONS)} setup conditions met">'
            f'{bars} {filled}</span>'
        )
    if key == "readiness":
        value = to_float(row.get("Stage1_Readiness"))
        if math.isnan(value):
            # Unavailable outside Stage 1, which is not the same as zero.
            return f'<span style="color:var(--faint)">{DASH}</span>'
        filled = int(value)
        dots = "".join(
            f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;'
            f'margin-right:3px;background:{POSITIVE if i < filled else "var(--line)"}"></span>'
            for i in range(5)
        )
        return f'<span class="num" style="font-weight:600">{dots} {filled}/5</span>'
    if key == "atr":
        value = to_float(row.get("ATR_Pct"))
        if math.isnan(value):
            return f'<span style="color:var(--faint)">{DASH}</span>'
        return f'<span class="num" style="color:var(--sub);font-weight:600">{value:.1f}%</span>'
    return DASH


def screener_table(
    frame: pd.DataFrame,
    trends: dict[str, Sequence[float]] | None = None,
    columns: Sequence[str] | None = None,
    sorted_by: str | None = None,
) -> str:
    """Render the dense stock table.

    ``trends`` maps symbol to a short close series for the sparkline column; a
    symbol without one shows an em dash rather than a fabricated line.
    """
    keys = list(columns or SCREENER_COLUMNS)
    header = "".join(
        '<th class="{cls}">{label}</th>'.format(
            cls=" ".join(
                filter(
                    None,
                    [
                        "right" if SCREENER_COLUMNS[k][1] == "right" else "",
                        "col-hide-sm" if SCREENER_COLUMNS[k][2] else "",
                        "active" if sorted_by == k else "",
                    ],
                )
            ),
            label=esc(SCREENER_COLUMNS[k][0]) + (" ▼" if sorted_by == k else ""),
        )
        for k in keys
    )

    body_rows = []
    for position, (_, row) in enumerate(frame.iterrows()):
        trend = (trends or {}).get(str(row.get("Symbol")))
        cells = "".join(
            '<td class="{cls}">{html}</td>'.format(
                cls=" ".join(
                    filter(
                        None,
                        [
                            "right" if SCREENER_COLUMNS[k][1] == "right" else "",
                            "col-hide-sm" if SCREENER_COLUMNS[k][2] else "",
                        ],
                    )
                ),
                html=_cell(k, row, trend),
            )
            for k in keys
        )
        delay = min(position, 12) * 18
        body_rows.append(f'<tr class="rfade" style="animation-delay:{delay}ms">{cells}</tr>')

    if not body_rows:
        return missing_notice(
            "No stocks match these filters.",
            "Filters change presentation only — widen them to see more of the same validated snapshot.",
        )
    return (
        '<div class="ws-table-wrap"><div class="ws-scroll"><table class="ws-table">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div></div>"
    )


def _plural(count: Any, noun: str) -> str:
    """'1 stock' / '12 stocks' — a count of one is not written as a plural."""
    value = to_float(count)
    if not math.isfinite(value):
        return f"{DASH} {noun}s"
    number = int(value)
    return f"{number:,} {noun}" if number == 1 else f"{number:,} {noun}s"


def industry_table(frame: pd.DataFrame) -> str:
    """Ranked industry rows: rank, name, median-RS bar, participation, 3M."""
    if frame.empty:
        return missing_notice(
            "No industry data in this snapshot.",
            "Industry is the NSE constituent CSV's Industry field; it is not remapped.",
        )
    head = (
        '<div class="ws-irow-head"><span class="ws-irank">#</span>'
        '<span class="ws-iname">Industry</span>'
        '<span class="ws-irs">Strength (median RS)</span>'
        '<span class="ws-inum col-hide-sm">Partic.</span>'
        '<span class="ws-inum">3M</span></div>'
    )
    rows = []
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        rs = to_float(row.get("Median_RS"))
        width = min(100.0, max(0.0, rs)) if math.isfinite(rs) else 0.0
        color = POSITIVE if rs >= 80 else ("#2D6CDF" if rs >= 50 else "#9aa1ac")
        participation = to_float(row.get("Participation_Pct"))
        rows.append(
            f'<div class="ws-irow">'
            f'<span class="ws-irank num">{rank}</span>'
            f'<div class="ws-iname"><a target="_self" '
            f'href="{query_href(view="Industries", industry=row.get("Industry"))}">'
            f'{esc(row.get("Industry"))}</a>'
            f'<div class="sub num">{_plural(to_float(row.get("Stocks")), "stock")}</div></div>'
            f'<div class="ws-irs"><div class="ws-irs-track col-hide-sm">'
            f'<div class="ws-irs-fill bar-grow" style="width:{width:.1f}%;background:{color}"></div></div>'
            f'<span class="ws-irs-value num">{fmt_rs(rs)}</span></div>'
            f'<span class="ws-inum num col-hide-sm" style="color:var(--sub)">'
            f'{fmt_pct(participation, digits=0, signed=False)}</span>'
            f'<span class="ws-inum num" style="color:{signed_color(to_float(row.get("Median_R3M")))};font-weight:600">'
            f'{fmt_return(row.get("Median_R3M"))}</span></div>'
        )
    return card(head + "".join(rows), extra_class="", style="padding:6px 16px 10px")


def transition_rows(frame: pd.DataFrame, kind: str = "flag") -> str:
    """Compact rows for a Movers group: symbol, change, RS, extension."""
    rows = []
    for _, row in frame.iterrows():
        if kind == "stage":
            middle = (
                f'<span>{esc(stage_display(row.get("Stage_From")))}</span>'
                f'<span style="color:var(--faint)">→</span>'
                f'<span style="color:{stage_color(row.get("Stage_To"))};font-weight:600">'
                f'{esc(stage_display(row.get("Stage_To")))}</span>'
            )
        elif kind == "action":
            middle = (
                f'<span>{esc(row.get("Action_From"))}</span>'
                f'<span style="color:var(--faint)">→</span>'
                f'<span style="color:{action_style(row.get("Action_To"))[0]};font-weight:700">'
                f'{esc(row.get("Action_To"))}</span>'
            )
        else:
            middle = f'<span>{esc(row.get("Industry"))}</span>'
        rows.append(
            f'<a class="ws-shelf" target="_self" href="{stock_href(row.get("Symbol"))}" '
            f'style="text-decoration:none;color:var(--ink)">'
            f'<span style="font-weight:700;font-size:14px;flex:0 0 auto;min-width:110px">'
            f'{esc(row.get("Symbol"))}</span>'
            f'<span style="font-size:12.5px;color:var(--sub);display:flex;align-items:center;'
            f'flex-wrap:wrap;gap:6px;flex:1 1 150px;min-width:0">{middle}</span>'
            f'<span class="num" style="font-size:13px;color:var(--sub);flex:0 0 auto">RS '
            f'<b style="color:var(--ink)">{fmt_rs(row.get("RS_Score"))}</b></span>'
            f'<span class="num" style="font-size:12.5px;color:var(--faint);flex:0 0 auto;'
            f'text-align:right">{fmt_pct(row.get("Ext_Pct"))} ext</span></a>'
        )
    return "".join(rows)


def price_chart_payload(frame: pd.DataFrame) -> list[dict]:
    """Serialize a trend frame for the charting library."""
    payload = []
    for stamp, row in frame.iterrows():
        entry = {"time": pd.Timestamp(stamp).strftime("%Y-%m-%d")}
        for key in ("Close", "MA_10W", "MA_30W"):
            value = to_float(row.get(key))
            if math.isfinite(value):
                entry[key] = value
        payload.append(entry)
    return payload
