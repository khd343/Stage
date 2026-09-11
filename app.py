"""RS-Stages — quantitative research terminal.

Presentation layer only. Every number rendered here comes from the validated
repository snapshot published by the Real Data Research Audit; this module
performs no locked calculation of its own. Filters change what is displayed,
never what was computed.

The visual system and component layout follow the WealthStar reference terminal,
which is this project's design source. The universe, industry classification,
Stage vocabulary, RS bands and the nine-label guide Action framework are
RS-Stages' own, as specified in docs/LOCKED_SPEC.md.
"""
from __future__ import annotations

import html
import math

import pandas as pd
import streamlit as st

from rs_stages.market import breadth_snapshot, industry_leadership
from rs_stages.movers import rs_movers, transitions
from rs_stages.quant import ma_10w_series, ma_30w_series
from rs_stages import signal_card
from rs_stages.screener import TREND_HEALTH_CONDITIONS, TREND_TEMPLATE_CONDITIONS
from rs_stages.ui import charts
from rs_stages.ui import components as ui
from rs_stages.ui import theme
from rs_stages.ui.loaders import load_price_panel, load_snapshot, panel_matches
from rs_stages.ui.theme import (
    CAUTION,
    NEGATIVE,
    POSITIVE,
    fmt_date,
    fmt_inr,
    fmt_number,
    fmt_pct,
    fmt_price,
    fmt_return,
    fmt_ratio,
    fmt_rs,
    rs_band,
    stage_color,
    stage_display,
    to_float,
)

st.set_page_config(
    page_title="RS-Stages — relative strength, stages and guide actions",
    page_icon="◧",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(theme.stylesheet(), unsafe_allow_html=True)

VIEWS = ["Dashboard", "Setups", "Screener", "Industries", "Market", "Movers", "Stock", "Methodology"]
STAGE_ORDER = ["Stage 2", "Stage 3", "Stage 4", "Stage 1"]
ACTION_ORDER = ["BUY★", "BUY", "HOLD", "WAIT", "WATCH★", "WATCH", "REDUCE", "SELL", "AVOID"]
PAGE_SIZE = 50
#: Transition rows shown inline before the rest moves behind an expander.
MOVERS_INLINE = 25


def write(markup: str) -> None:
    """Inject an HTML fragment as one block.

    Blank lines are stripped because CommonMark would otherwise end the HTML
    block there and render the remainder of the fragment as literal text.
    """
    st.markdown(theme.collapse_blank_lines(markup), unsafe_allow_html=True)


# FRESHNESS, NOT JUST CACHING. The deployed checkout only changes on a container
# rebuild (D-2.1.7), and that rebuild is not ours to trigger reliably -- so the
# snapshot is read from GitHub at runtime with the checkout as fallback, and
# every cache below is either short-lived or KEYED ON THE SNAPSHOT'S DATE, so a
# process can never hold yesterday's data beside today's. See D-2.1.9.
SNAPSHOT_TTL_SECONDS = 600


@st.cache_data(ttl=SNAPSHOT_TTL_SECONDS, show_spinner="Reading the validated snapshot…")
def cached_snapshot():
    """The published snapshot, refreshed from GitHub within ten minutes."""
    return load_snapshot(remote=True)


def _decision_key() -> str:
    """The snapshot's session, as the cache key everything derived from it uses."""
    stamp = cached_snapshot().decision_date
    return "" if stamp is None else stamp.strftime("%Y-%m-%d")


@st.cache_resource(ttl=3600, show_spinner="Loading price history…")
def _cached_panel(decision_key: str):
    """The price panel for ONE snapshot session, held by reference.

    cache_resource, not cache_data: the panel is by far the largest artifact and
    cache_data would serialise it on every rerun. The session is the key, so a
    new snapshot fetches a new panel instead of reusing one that panel_matches
    would reject -- which used to leave every chart withheld until a restart.
    """
    panel, error = load_price_panel()
    if panel is not None:
        mismatch = panel_matches(panel, cached_snapshot().research)
        if mismatch:
            return None, mismatch
    return panel, error


def cached_panel():
    return _cached_panel(_decision_key())


@st.cache_resource(ttl=3600, show_spinner=False)
def _cached_sparklines(sessions: int, decision_key: str):
    panel, _ = _cached_panel(decision_key)
    return {} if panel is None else panel.tails(sessions)


def cached_sparklines(sessions: int = 63):
    """Trailing closes per symbol for the sparkline column."""
    return _cached_sparklines(sessions, _decision_key())


SNAP = cached_snapshot()
DATA = SNAP.research
BREADTH = breadth_snapshot(DATA)


# --- navigation -------------------------------------------------------------
def _initial(param: str, allowed: list[str] | None, fallback: str) -> str:
    value = st.query_params.get(param)
    if value and (allowed is None or value in allowed):
        return value
    return fallback


def _top_symbol() -> str:
    ordered = DATA.sort_values("RS_Score", ascending=False)["Symbol"].dropna().astype(str)
    return ordered.iloc[0] if len(ordered) else ""


if "symbol" not in st.session_state:
    requested = st.query_params.get("symbol")
    known = set(DATA["Symbol"].astype(str))
    st.session_state["symbol"] = requested if requested in known else _top_symbol()
_INDUSTRIES = sorted(DATA["Industry"].dropna().astype(str).unique().tolist())
_REQUESTED_INDUSTRY = _initial("industry", _INDUSTRIES, "")
if "screener_industry" not in st.session_state:
    st.session_state["screener_industry"] = _REQUESTED_INDUSTRY or "All"
if "industry_pick" not in st.session_state:
    st.session_state["industry_pick"] = _REQUESTED_INDUSTRY or "None"


# --- header -----------------------------------------------------------------
decision = SNAP.decision_date
coverage = SNAP.date_coverage

# Date is set per symbol from that symbol's own latest completed session, not
# a shared clock (screener.py) — the provider updates its feed asynchronously,
# larger names first, so a fraction of the universe can lag by one session on
# any given run. decision_date is the newest date present; the stamp must not
# let that read as every symbol's date when it is not.
if coverage.is_split:
    stamp_detail = (
        f'<span style="color:var(--faint);font-weight:500">· Nifty Total Market '
        f'({len(SNAP.universe):,}) · {ui.dot("#B5781A", size=6)} '
        f'<span style="color:#B5781A">{coverage.current_count:,} of '
        f'{len(SNAP.universe):,} as of this date, {coverage.lagging_count:,} '
        f'one session behind (provider lag)</span></span>'
    )
else:
    stamp_detail = (
        f'<span style="color:var(--faint);font-weight:500">· Nifty Total Market '
        f'({len(SNAP.universe):,})</span>'
    )
write(
    '<div class="ws-header"><div class="ws-header-inner">'
    '<a class="ws-brand" target="_self" href="?view=Dashboard">'
    '<span class="ws-mark">RS</span><span><div class="ws-wordmark">RS-Stages</div>'
    '<div class="ws-tagline">Relative strength · stages · guide actions</div></span></a>'
    f'<div class="ws-stamp">{ui.dot(POSITIVE)}Validated snapshot '
    f'<span class="num" style="color:var(--ink)">{fmt_date(decision)}</span>'
    f'{stamp_detail}</div></div></div>'
)

st.write("")
selected = st.segmented_control(
    "Section",
    VIEWS,
    default=_initial("view", VIEWS, "Dashboard"),
    key="nav",
    label_visibility="collapsed",
)
VIEW = selected or _initial("view", VIEWS, "Dashboard")


def heading(title: str, subtitle: str) -> None:
    write(f'<div class="ws-page-title">{ui.esc(title)}</div><div class="ws-page-sub">{subtitle}</div>')
    st.write("")


def missing(key: str) -> bool:
    """Render the explicit unavailability notice for an artifact, if missing."""
    detail = cached_panel()[1] if key == "panel" else SNAP.missing.get(key)
    if not detail:
        return False
    titles = {
        "panel": "Price history is not published yet",
        "previous": "Day-over-day comparison is unavailable",
        "breadth": "Participation history is not published yet",
        "v21_fields": "This snapshot predates locked-spec v2.1",
        "v22_fields": "The pre-breakout structure is not published yet",
    }
    write(ui.missing_notice(titles.get(key, "Unavailable"), detail))
    return True


# --- shared fragments -------------------------------------------------------
def regime_card(link: bool = True) -> str:
    color = {"Broad": POSITIVE, "Mixed": CAUTION, "Narrow": NEGATIVE}.get(BREADTH["regime"], "#9aa1ac")
    link_html = (
        '<a target="_self" href="?view=Market" style="margin-left:auto;font-size:12.5px;'
        'color:var(--ink);font-weight:600;text-decoration:none;'
        'border-bottom:1.5px solid var(--rule-strong)">Market detail →</a>'
        if link
        else ""
    )
    return ui.card(
        '<div class="ws-eyebrow">Market regime</div>'
        '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">'
        f'<span style="font-size:19px;font-weight:800;color:{color}">{ui.esc(BREADTH["regime"])}</span>'
        f'<span class="num" style="font-size:13px;color:var(--sub)">'
        f'<b style="color:var(--ink)">{fmt_pct(BREADTH["pct_above_ma_30w"], 0, signed=False)}</b>'
        " of the universe above its 30-week line</span>"
        f"{link_html}</div>"
        f'<div style="font-size:12.5px;color:var(--sub);margin-top:8px;line-height:1.5">'
        f'{ui.esc(BREADTH["regime_description"])} Breadth is a description of participation '
        "in completed sessions. It is not a call on direction.</div>",
        extra_class="lift",
    )


def action_distribution(frame: pd.DataFrame) -> str:
    counts = frame["Action"].value_counts()
    total = int(counts.sum())
    if not total:
        return ""
    legend, segments = [], []
    for label in ACTION_ORDER:
        count = int(counts.get(label, 0))
        if not count:
            continue
        color = theme.action_style(label)[0]
        legend.append(
            f'<span class="item">{ui.dot(color)}{ui.esc(label)}<b class="num">{count:,}</b></span>'
        )
        segments.append(
            f'<div class="bar-grow" style="width:{count / total * 100:.4f}%;background:{color}"></div>'
        )
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'flex-wrap:wrap;gap:8px;margin-bottom:2px">'
        '<span style="font-size:12.5px;font-weight:600;color:var(--sub)">Action distribution</span>'
        f'<span class="ws-legend">{"".join(legend)}</span></div>'
        f'<div class="ws-segbar">{"".join(segments)}</div>'
    )


def group_shelf_card(title: str, description: str, rows: pd.DataFrame, meta: str = "rs") -> str:
    if rows.empty:
        body = f'<div class="ws-note">No stock meets this condition in the current snapshot.</div>'
    else:
        items = [
            (
                row["Symbol"],
                f"RS {fmt_rs(row.get('RS_Score'))}"
                if meta == "rs"
                else fmt_pct(row.get("Ext_Pct")),
            )
            for _, row in rows.head(14).iterrows()
        ]
        body = ui.pill_row(items)
    return ui.card(
        '<div style="display:flex;align-items:baseline;gap:9px;flex-wrap:wrap;margin-bottom:4px">'
        f'<span style="font-size:14.5px;font-weight:700">{ui.esc(title)}</span>'
        f'<span class="num" style="font-size:12.5px;color:var(--faint)">{len(rows):,}</span></div>'
        f'<div class="ws-note" style="margin-bottom:11px">{ui.esc(description)}</div>{body}',
        extra_class="lift",
    )


# --- Dashboard --------------------------------------------------------------
def page_dashboard() -> None:
    heading(
        "Today’s briefing",
        "A top-down read of the validated snapshot — regime, industry leadership, what changed "
        "since the previous completed session, and where names sit today. This page orients; "
        "the Screener selects.",
    )

    coverage = SNAP.date_coverage
    if coverage.is_split:
        lagging = DATA[
            pd.to_datetime(DATA["Date"], errors="coerce").dt.normalize() != coverage.latest
        ].copy()
        lagging["Date"] = pd.to_datetime(lagging["Date"]).dt.strftime("%d %b")
        lagging = lagging.sort_values("Symbol")
        items = list(zip(lagging["Symbol"], lagging["Date"]))
        write(
            ui.card(
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">'
                f'{ui.dot("#B5781A", size=8)}<div class="ws-eyebrow" style="margin:0;color:#B5781A">'
                "Not every stock is on today's close</div></div>"
                f'<div class="ws-note" style="margin-bottom:10px">'
                f"The price provider updates its feed asynchronously — larger, more liquid names "
                f"first. <strong>{coverage.current_count:,} of {len(DATA):,}</strong> stocks reflect "
                f"{fmt_date(coverage.latest)}; <strong>{coverage.lagging_count:,}</strong> "
                f"({coverage.lagging_pct:.0f}%) still carry their prior session, listed below with "
                f"the date each is actually on. Every field for a lagging stock is internally "
                f"consistent for its own date — nothing here is fabricated — but a ranking that "
                f"compares it to a same-day peer is comparing two different sessions. This is not "
                f"withheld: waiting for every stock to agree would let one thin name delay the "
                f"other {len(DATA) - 1:,}."
                "</div>"
                + ui.pill_row(items[:25])
            )
        )
        if len(items) > 25:
            with st.expander(f"{len(items) - 25:,} more lagging symbols"):
                write(ui.pill_row(items[25:]))
        st.write("")

    write(regime_card())
    st.write("")
    write(
        ui.stat_row(
            [
                ui.stat_card("Valid RS", f"{BREADTH.get('valid_rs', 0):,}", note="scored against the universe"),
                ui.stat_card(
                    "Stage 2 — Advancing",
                    f"{BREADTH['stages']['Stage 2']:,}",
                    note=f"{BREADTH['stages']['Stage 2'] / max(BREADTH['classified'], 1) * 100:.0f}% of classified stocks",
                    color=POSITIVE,
                ),
                ui.stat_card(
                    "Confirmed breakouts",
                    f"{BREADTH['breakout_confirmed']:,}",
                    note=f"of {BREADTH['breakout']:,} breakout setups",
                    color=POSITIVE if BREADTH["breakout_confirmed"] else "var(--ink)",
                ),
                ui.stat_card(
                    "Distribution warnings",
                    f"{BREADTH['distribution']:,}",
                    note="U/D below 0.7",
                    color=NEGATIVE if BREADTH["distribution"] else "var(--ink)",
                ),
            ]
        )
    )

    st.write("")
    write(ui.card(ui.posture_bar(BREADTH["stages"]) + "<div style='height:14px'></div>" + action_distribution(DATA)))

    # Leading industries
    industries = industry_leadership(DATA)
    if not industries.empty:
        top = industries.head(6)
        chips = "".join(
            f'<a class="ws-pill-link pop" target="_self" '
            f'href="{ui.query_href(view="Industries", industry=row["Industry"])}">'
            f'<span class="sym">{ui.esc(row["Industry"])}</span>'
            f'<span class="meta num">RS {fmt_rs(row["Median_RS"])}</span></a>'
            for _, row in top.iterrows()
        )
        st.write("")
        write(
            ui.card(
                '<div style="display:flex;align-items:center;justify-content:space-between;'
                'margin-bottom:10px;gap:8px;flex-wrap:wrap">'
                '<div class="ws-eyebrow" style="margin:0">Leading industries</div>'
                '<a target="_self" href="?view=Industries" style="font-size:12.5px;color:var(--ink);'
                'font-weight:600;text-decoration:none;border-bottom:1.5px solid var(--rule-strong)">'
                "All industries →</a></div>"
                f'<div class="ws-chips">{chips}</div>'
                '<div class="ws-note" style="margin-top:10px">Industry is the NSE constituent '
                "field, ranked by median RS so one extreme constituent cannot define the group.</div>",
                extra_class="lift",
            )
        )

    # What changed since the previous completed session
    st.write("")
    write('<div class="ws-eyebrow" style="margin-bottom:8px">What changed since the previous close</div>')
    if SNAP.previous is None:
        missing("previous")
    else:
        groups = transitions(DATA, SNAP.previous)
        if not groups:
            write(
                ui.card(
                    '<div class="ws-note">No stock changed Stage, breakout state, 10-week '
                    "position, distribution state or Action between the two completed sessions.</div>"
                )
            )
        else:
            shelves = []
            for label, payload in list(groups.items())[:8]:
                rows = payload["rows"]
                color = (
                    POSITIVE
                    if any(word in label for word in ("Entered Stage 2", "New breakout", "Newly confirmed", "Reclaimed", "cleared", "within 3%"))
                    else NEGATIVE
                    if any(word in label for word in ("Stage 4", "Left Stage 2", "Lost", "New distribution"))
                    else CAUTION
                )
                shelves.append(
                    ui.shelf(
                        label,
                        len(rows),
                        color,
                        [(r["Symbol"], f"RS {fmt_rs(r.get('RS_Score'))}") for _, r in rows.iterrows()],
                        more_href="?view=Movers",
                    )
                )
            write(
                ui.card(
                    "".join(shelves)
                    + '<div class="ws-note" style="margin-top:10px">Each group is a set difference '
                    "between two completed-session snapshots produced by the same pipeline. "
                    '<a target="_self" href="?view=Movers" style="color:var(--ink);font-weight:600">'
                    "All movers →</a></div>"
                )
            )

    # Leaders grouped by state — our locked conditions, not invented shelves.
    st.write("")
    write('<div class="ws-eyebrow" style="margin-bottom:8px">Where names sit today</div>')
    ordered = DATA.sort_values("RS_Score", ascending=False)
    stage2 = ordered[ordered["Stage_Label"] == "Stage 2"]
    left, right = st.columns(2, gap="medium")
    with left:
        write(
            group_shelf_card(
                "RS leadership",
                "Relative-strength rank 80 or higher — the locked v2 leadership band.",
                ordered[ordered["RS_Score"] >= 80],
            )
        )
        write(
            group_shelf_card(
                "Breakout setups",
                "Stage 2, within 3% of the 52-week high, on volume above 1.5× the prior-50 baseline — "
                "confirmation (U/D > 1.3) not yet present.",
                stage2[stage2["Breakout"].fillna(False).astype(bool) & ~stage2["Breakout_Confirmed"].fillna(False).astype(bool)],
            )
        )
    with right:
        write(
            group_shelf_card(
                "Confirmed breakouts",
                "Breakout setup that also carries U/D above 1.3.",
                ordered[ordered["Breakout_Confirmed"].fillna(False).astype(bool)],
            )
        )
        write(
            group_shelf_card(
                "Timing warnings in leadership",
                "Stage 2 with RS 80+ that the guide holds back: extended beyond 20% above the "
                "30-week line, or below the 50-session average.",
                stage2[
                    (stage2["RS_Score"] >= 80)
                    & (
                        stage2.get("Extended_20Pct", pd.Series(False, index=stage2.index)).fillna(False).astype(bool)
                        | stage2.get("Below_50DMA", pd.Series(False, index=stage2.index)).fillna(False).astype(bool)
                    )
                ],
            )
        )


# --- Screener ---------------------------------------------------------------
SORTS = {
    "RS (high to low)": ("RS_Score", False, "rs"),
    "RS (low to high)": ("RS_Score", True, "rs"),
    "3-month return": ("R3M", False, "r3m"),
    "Extension above 30-week": ("Ext_Pct", False, "ext"),
    "U/D ratio": ("U_D", False, "ud"),
    "Symbol (A–Z)": ("Symbol", True, "symbol"),
}


#: Canonical screens. Each composes published fields only; none introduces a
#: rule, and the Action column remains the guide's own interpretation.
SCREENER_PRESETS = {
    "None": ("", lambda f: f),
    "Buy candidates": (
        "Stage 2, RS 80 or better, not extended beyond 20% above the 30-week line, and "
        "holding the 50-session average — the guide's own entry conditions, composed.",
        lambda f: f[
            (f["Stage_Label"] == "Stage 2")
            & (pd.to_numeric(f["RS_Score"], errors="coerce") >= 80)
            & (~f["Extended_20Pct"].fillna(False).astype(bool))
            & (~f["Below_50DMA"].fillna(False).astype(bool))
        ],
    ),
    "Coiling": (
        "Contracting range and drying volume across the 50-session base (§10.5).",
        lambda f: f[f["VCP_Setup"].fillna(False).astype(bool)],
    ),
    "RS leading price": (
        "Relative strength at a 52-week high while price is still 5% or more below its own (§4.1).",
        lambda f: f[f["RS_Line_NH_Before_Price"].fillna(False).astype(bool)],
    ),
    "Template pass": (
        "All eight Minervini trend-template criteria satisfied (§5.1). Thresholds provisional.",
        lambda f: f[f["Trend_Template_Pass"].fillna(False).astype(bool)],
    ),
    "Exit now": (
        "Stage 4, or Stage 3 with volume confirming distribution — where the guide says reduce.",
        lambda f: f[
            (f["Stage_Label"] == "Stage 4")
            | ((f["Stage_Label"] == "Stage 3") & f["Distribution"].fillna(False).astype(bool))
        ],
    ),
}


def page_screener() -> None:
    heading(
        "Screener",
        "The full validated universe with its evidence and the guide Action as the decision column. "
        "Filters are presentation only — they never recompute the cross-sectional RS ranking.",
    )

    industries = ["All"] + sorted(DATA["Industry"].dropna().astype(str).unique().tolist())
    row1 = st.columns([2, 2, 2])
    query = row1[0].text_input("Search", placeholder="Symbol, company or industry", key="screener_query")
    industry = row1[1].selectbox("Industry", industries, key="screener_industry")
    sort_label = row1[2].selectbox("Sort by", list(SORTS), key="screener_sort")

    # Canonical screens as one click. Each is a composition of published fields
    # in the source books' own terms — nothing here is a new rule, and the
    # filters remain presentation only.
    preset = st.segmented_control(
        "Preset",
        list(SCREENER_PRESETS),
        default="None",
        key="screener_preset",
        help="Presets compose existing published fields; they never recompute the RS ranking.",
    )

    stages = st.segmented_control(
        "Stage",
        ["All stages"] + [stage_display(s) for s in STAGE_ORDER],
        default="All stages",
        key="screener_stage",
    )
    actions = st.segmented_control(
        "Action", ACTION_ORDER, selection_mode="multi", key="screener_action"
    )
    row2 = st.columns([3, 2])
    rs_low, rs_high = row2[0].slider("RS band", 1, 99, (1, 99), key="screener_rs")
    liquid_only = row2[1].toggle(
        "Liquid only (20-session traded value above ₹5 Cr)", key="screener_liquid"
    )
    prebreakout = st.toggle(
        "Show pre-breakout evidence instead of momentum",
        key="screener_prebreakout",
        help="Swaps the U/D, extension, 52-week range and 3-month columns for the v2.2 "
             "contraction, volume dry-up, pivot distance and trend-template columns.",
    )

    view = DATA.copy()
    if preset and preset != "None":
        description, predicate = SCREENER_PRESETS[preset]
        try:
            view = predicate(view)
        except KeyError:
            # A preset reading v2.2 fields against a v2.1 snapshot.
            view = view.iloc[0:0]
            description += " — unavailable until the audit republishes with v2.2 fields."
        write(f'<div class="ws-note" style="margin:2px 0 8px">{ui.esc(description)}</div>')
    if industry != "All":
        view = view[view["Industry"].astype(str) == industry]
    if stages and stages != "All stages":
        view = view[view["Stage_Label"] == stages.split(" · ")[0]]
    if actions:
        view = view[view["Action"].isin(list(actions))]
    score = pd.to_numeric(view["RS_Score"], errors="coerce")
    view = view[score.between(rs_low, rs_high)]
    if liquid_only and "Liquid_UI_Filter" in view.columns:
        view = view[view["Liquid_UI_Filter"].fillna(False).astype(bool)]
    if query and query.strip():
        needle = query.strip().upper()
        haystack = (
            view["Symbol"].astype(str).str.upper()
            + " "
            + view["Industry"].astype(str).str.upper()
            + " "
            + view.get("Company Name", pd.Series("", index=view.index)).astype(str).str.upper()
        )
        view = view[haystack.str.contains(needle, na=False, regex=False)]

    column, ascending, sort_key = SORTS[sort_label]
    if column in view.columns:
        view = view.sort_values(column, ascending=ascending, na_position="last")

    total = len(view)
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = 1
    if pages > 1:
        page = st.columns([1, 4])[0].number_input(
            f"Page (1–{pages})", min_value=1, max_value=pages, value=1, step=1, key="screener_page"
        )
    window = view.iloc[(int(page) - 1) * PAGE_SIZE : int(page) * PAGE_SIZE]

    shown = f"{len(window):,} of {total:,}" if pages > 1 else f"{total:,}"
    write(
        f'<div class="ws-note" style="margin:2px 0 10px">Showing <b style="color:var(--ink)">{shown}</b> '
        f"stocks · sorted by {ui.esc(sort_label.lower())} · Action is the guide interpretation of the "
        "evidence to its left, never a substitute for it.</div>"
    )
    columns = (
        ("symbol", "trend", "rs", "stage", "action",
         "rsline", "contraction", "dryup", "pivot", "template")
        if prebreakout
        else None
    )
    write(ui.screener_table(window, cached_sparklines(), columns=columns, sorted_by=sort_key))
    st.download_button(
        "Download these results (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="rs-stages-screener.csv",
        mime="text/csv",
        key="screener_csv",
        help=f"All {total:,} filtered rows, not just this page.",
    )
    if cached_panel()[0] is None:
        st.write("")
        missing("panel")


# --- Industries -------------------------------------------------------------
INDUSTRY_SORTS = {
    "Median RS (high to low)": ("Median_RS", False),
    "Median RS (low to high)": ("Median_RS", True),
    "Participation": ("Participation_Pct", False),
    "Median 3-month return": ("Median_R3M", False),
    "Stocks (most first)": ("Stocks", False),
    "Industry (A–Z)": ("Industry", True),
}


def page_industries() -> None:
    heading(
        "Industry strength",
        "Where is leadership concentrated? Industry is exactly the NSE constituent CSV's Industry "
        "field — it is never remapped. Aggregates are descriptive; the Action label is always "
        "produced at stock level.",
    )
    industries = industry_leadership(DATA)
    if industries.empty:
        write(ui.missing_notice("No industry data.", "The snapshot carries no Industry field."))
        return

    leaders = ", ".join(industries.head(3)["Industry"].astype(str))
    write(
        ui.card(
            f'Leadership sits in <b>{ui.esc(leaders)}</b>, ranked by median RS across '
            f'<span class="num">{len(industries)}</span> industries covering '
            f'<span class="num">{int(industries["Stocks"].sum()):,}</span> stocks.',
            extra_class="lift",
            style="font-size:13.5px",
        )
    )
    st.write("")

    controls = st.columns([2, 2])
    sort_label = controls[0].selectbox("Sort by", list(INDUSTRY_SORTS), key="industry_sort")
    names = ["None"] + industries["Industry"].astype(str).tolist()
    selected = controls[1].selectbox("Show stocks in", names, key="industry_pick")

    column, ascending = INDUSTRY_SORTS[sort_label]
    if column in industries.columns:
        industries = industries.sort_values(column, ascending=ascending, na_position="last")
    write(ui.industry_table(industries.reset_index(drop=True)))

    # Drill-down: the constituents of one industry, without leaving the page.
    if selected and selected != "None":
        members = DATA[DATA["Industry"].astype(str) == selected]
        members = members.sort_values("RS_Score", ascending=False, na_position="last")
        st.write("")
        stage_counts = {
            f"Stage {n}": int((members["Stage_Label"] == f"Stage {n}").sum()) for n in (1, 2, 3, 4)
        }
        write(
            f'<div class="ws-eyebrow" style="margin-bottom:8px">{ui.esc(selected)} · '
            f'{len(members):,} stocks</div>'
        )
        write(ui.card(ui.posture_bar(stage_counts, f"Stage posture in {selected}")))
        st.write("")
        write(
            ui.screener_table(
                members.head(PAGE_SIZE), cached_sparklines(), sorted_by="rs"
            )
        )
        if len(members) > PAGE_SIZE:
            write(
                f'<div class="ws-note" style="margin-top:8px">Showing the '
                f'{PAGE_SIZE} highest-RS names. '
                f'<a target="_self" href="{ui.query_href(view="Screener", industry=selected)}" '
                f'style="color:var(--ink);font-weight:600">'
                f"Open all {len(members):,} in the Screener →</a></div>"
            )


# --- Market -----------------------------------------------------------------
def page_market() -> None:
    heading(
        "Market regime",
        "How broad is participation across the Nifty Total Market universe? Every measure below is "
        "a count of the same locked per-stock fields the Screener shows.",
    )
    write(regime_card(link=False))
    st.write("")

    cards = [
        ui.stat_card(
            "Above the 30-week line",
            fmt_pct(BREADTH["pct_above_ma_30w"], 0, signed=False).rstrip("%"),
            suffix="%",
            note=f"{BREADTH['above_ma_30w']:,} of {BREADTH['classified']:,} classified",
            color=POSITIVE if BREADTH["pct_above_ma_30w"] >= 60 else "var(--ink)",
        )
    ]
    if BREADTH["has_ma_10w"]:
        cards.append(
            ui.stat_card(
                "Above the 10-week line",
                fmt_pct(BREADTH["pct_above_ma_10w"], 0, signed=False).rstrip("%"),
                suffix="%",
                note="shorter-term participation",
            )
        )
    cards.append(
        ui.stat_card(
            "Within 3% of the 52-week high",
            f"{BREADTH['near_52w_high']:,}",
            note=f"{BREADTH['breakout_confirmed']:,} breakouts confirmed",
            color=POSITIVE,
        )
    )
    cards.append(
        ui.stat_card(
            "RS leadership / lagging",
            f"{BREADTH.get('rs_leaders', 0):,}",
            suffix=f" / {BREADTH.get('rs_lagging', 0):,}",
            note="RS 80+ versus RS below 50",
        )
    )
    write(ui.stat_row(cards))

    st.write("")
    write(ui.card(ui.posture_bar(BREADTH["stages"], "Stage posture across the universe")))

    st.write("")
    if SNAP.breadth is None or SNAP.breadth.empty:
        missing("breadth")
    else:
        history = SNAP.breadth

        def series(column):
            return [
                {"time": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"), "value": float(r[column])}
                for _, r in history.iterrows()
                if column in history.columns and pd.notna(r.get(column))
            ]

        benchmark = series("Benchmark_Close")
        ticker = (
            str(history["Benchmark_Ticker"].dropna().iloc[-1])
            if "Benchmark_Ticker" in history.columns and history["Benchmark_Ticker"].notna().any()
            else ""
        )
        legend = (
            '<span class="item"><span style="width:14px;height:2.5px;background:#2D6CDF"></span>'
            "Above 30-week</span>"
            '<span class="item"><span style="width:14px;height:2.5px;background:#9DBDF0"></span>'
            "Above 10-week</span>"
        )
        if benchmark:
            legend += (
                '<span class="item"><span style="width:14px;height:0;border-top:2px dashed '
                f'#1a1d21"></span>Nifty 500{" · " + ui.esc(ticker) if ticker else ""}</span>'
            )
        detail = (
            f"{len(history)} completed sessions, each counted as of that session. The series is a "
            "stack of point-in-time counts, not one snapshot projected backwards."
        )
        if benchmark:
            detail += (
                " Breadth is a percentage on the left axis; the index is a price level on the "
                "right. The index tracks 500 companies while breadth tracks the whole Nifty Total "
                "Market universe, so a divergence can be composition rather than market behaviour."
            )
        write(
            ui.card(
                '<div style="display:flex;align-items:center;justify-content:space-between;'
                'flex-wrap:wrap;gap:8px;margin-bottom:6px">'
                '<span style="font-size:13px;font-weight:700">Participation trend</span>'
                f'<span class="ws-legend">{legend}</span></div>'
                f'<div class="ws-note">{detail}</div>'
            )
        )
        _line_chart(series("Pct_Above_MA_30W"), series("Pct_Above_MA_10W"), benchmark)
        if not benchmark:
            st.write("")
            write(
                ui.missing_notice(
                    "No benchmark index in this snapshot",
                    "The breadth history carries no index column, so only participation is drawn. "
                    "Re-run the Real Data Research Audit to publish it alongside breadth.",
                )
            )


def _line_chart(
    series_30w: list[dict], series_10w: list[dict], benchmark: list[dict] | None = None
) -> None:
    """Participation trend, with the benchmark index on its own axis.

    Breadth is a percentage and an index is a price level; sharing one scale
    would flatten whichever has the smaller range into a straight line, so the
    index gets the right-hand axis and breadth keeps the left.
    """
    series = [
        {"data": series_30w, "color": "#2D6CDF", "last_value": True, "scale": "left"},
        {"data": series_10w, "color": "#9DBDF0", "scale": "left"},
    ]
    if benchmark:
        series.append(
            {"data": benchmark, "color": "#1a1d21", "dashed": True, "last_value": True,
             "scale": "right"}
        )
    st.iframe(
        charts.line_chart(
            series,
            element_id="breadth",
            height=280,
            unavailable="The participation trend could not be drawn in this environment.",
        ),
        height=296,
    )


# --- Movers -----------------------------------------------------------------
def page_movers() -> None:
    previous_date = SNAP.previous_date
    heading(
        "What changed",
        f"Structural changes between {ui.esc(fmt_date(previous_date))} and "
        f"{ui.esc(fmt_date(SNAP.decision_date))}. Every group is a set difference between two "
        "completed-session snapshots run through the identical pipeline."
        if previous_date is not None
        else "Structural changes between the two most recent completed sessions.",
    )
    if missing("previous"):
        return

    groups = transitions(DATA, SNAP.previous)
    if not groups:
        write(ui.card('<div class="ws-note">Nothing changed state between the two sessions.</div>'))
        return

    # Group labels carry our locked Stage vocabulary; they are not lower-cased,
    # because "stage 2 — advancing" is not the name of anything in this system.
    parts = [
        f'<b class="num">{len(payload["rows"])}</b> {ui.esc(label)}'
        for label, payload in groups.items()
    ]
    write(
        ui.card(
            f'Of <span class="num">{len(DATA):,}</span> stocks: ' + " · ".join(parts) + ".",
            extra_class="lift",
            style="font-size:13.5px",
        )
    )

    for label, payload in groups.items():
        rows = payload["rows"]
        kind = "stage" if "Stage_To" in rows.columns else ("action" if "Action_To" in rows.columns else "flag")
        color = (
            POSITIVE
            if any(w in label for w in ("Entered Stage 2", "New breakout", "Newly confirmed", "Reclaimed", "cleared", "within 3%"))
            else NEGATIVE
            if any(w in label for w in ("Stage 4", "Left Stage 2", "Lost", "New distribution"))
            else CAUTION
        )
        st.write("")
        # Every member is reachable. The first page is inline; the remainder
        # goes behind an expander so a group of a hundred names does not bury
        # the groups beneath it.
        head, tail = rows.head(MOVERS_INLINE), rows.iloc[MOVERS_INLINE:]
        write(
            ui.card(
                '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;flex-wrap:wrap">'
                f'{ui.dot(color, 9)}<span style="font-size:14.5px;font-weight:700">{ui.esc(label)}</span>'
                f'<span class="num" style="font-size:12.5px;color:var(--faint)">{len(rows)}</span>'
                f'<span style="margin-left:auto;font-size:12px;color:var(--sub)">'
                f'{ui.esc(payload["description"])}</span></div>'
                + ui.transition_rows(head, kind),
                style="padding:14px 16px 8px",
            )
        )
        if not tail.empty:
            with st.expander(f"Show the remaining {len(tail):,} in “{label}”"):
                write(ui.card(ui.transition_rows(tail, kind), style="padding:4px 16px 8px"))

    movers = rs_movers(DATA, SNAP.previous, count=12)
    if not movers.empty:
        st.write("")
        rows = "".join(
            f'<a class="ws-shelf" target="_self" href="{ui.stock_href(r["Symbol"])}" '
            f'style="text-decoration:none;color:var(--ink)">'
            f'<span style="font-weight:700;font-size:14px;min-width:110px">{ui.esc(r["Symbol"])}</span>'
            f'<span class="num" style="font-size:12.5px;color:var(--sub);flex:1 1 120px">'
            f'{fmt_rs(r["RS_Previous"])} → <b style="color:var(--ink)">{fmt_rs(r["RS_Score"])}</b></span>'
            f'<span class="num" style="font-weight:700;font-size:13px;'
            f'color:{theme.signed_color(r["RS_Change"])}">{r["RS_Change"]:+.0f}</span></a>'
            for _, r in movers.iterrows()
        )
        write(
            ui.card(
                '<div style="font-size:14.5px;font-weight:700;margin-bottom:2px">Largest RS rank changes</div>'
                '<div class="ws-note" style="margin-bottom:4px">RS is a cross-sectional percentile, so a '
                "change is a change in standing against the universe, not a return.</div>" + rows,
                style="padding:14px 16px 8px",
            )
        )


# --- Stock ------------------------------------------------------------------
def page_stock() -> None:
    symbols = DATA.sort_values("RS_Score", ascending=False)["Symbol"].dropna().astype(str).tolist()
    if not symbols:
        write(ui.missing_notice("No symbols in the snapshot.", "The research snapshot is empty."))
        return
    st.selectbox("Symbol", symbols, key="symbol")
    symbol = st.session_state["symbol"]
    row = DATA.loc[DATA["Symbol"] == symbol].iloc[0]

    stage = row.get("Stage")
    color = stage_color(stage)
    band, band_color = rs_band(row.get("RS_Score"))
    company = row.get("Company Name")

    price_html = (
        f'<span class="num" style="font-size:18px;font-weight:800">{fmt_price(row.get("Close"))}</span>'
        if pd.notna(row.get("Close"))
        else ""
    )
    metrics = [
        ("RS rank", fmt_rs(row.get("RS_Score")), band, band_color, False),
        (
            "Trend health",
            f"{int(row['Trend_Health'])}/5" if pd.notna(row.get("Trend_Health")) else theme.DASH,
            "conditions met",
            "var(--ink)",
            True,
        ),
        (
            "From 52-week high",
            fmt_pct(row.get("Pct_From_52W_High")),
            "adjusted high",
            theme.signed_color(row.get("Pct_From_52W_High")),
            True,
        ),
    ]
    metric_html = "".join(
        f'<div class="ws-metric"><div class="label">{ui.esc(label)}</div>'
        f'<div class="value{" small" if small else ""} num" style="color:{c}">{ui.esc(value)}</div>'
        f'<div class="label" style="margin-top:3px">{ui.esc(note)}</div></div>'
        for label, value, note, c, small in metrics
    )

    write(
        '<div style="display:flex;align-items:flex-end;justify-content:space-between;'
        'flex-wrap:wrap;gap:14px;margin:6px 0 16px">'
        '<div style="display:flex;align-items:center;gap:14px">'
        f'<span class="ws-tile" style="width:44px;height:44px;border-radius:12px;font-size:16px;'
        f'background:{color}14;color:{color}">{ui.esc(theme.initials(symbol))}</span><div>'
        f'<div style="font-size:24px;font-weight:800;letter-spacing:-.5px">{ui.esc(symbol)}</div>'
        f'<div style="display:flex;align-items:baseline;gap:8px;margin-top:3px">{price_html}'
        f'<span style="font-size:12.5px;color:var(--faint)">{ui.esc(company) if pd.notna(company) else ""}</span></div>'
        f'<div style="display:flex;align-items:center;gap:7px;font-size:13px;color:var(--sub);margin-top:3px">'
        f'{ui.dot(color)}<span>{ui.esc(stage_display(stage))}</span>'
        f'<span style="color:var(--faint)">· {ui.esc(row.get("Industry"))} · as of '
        f'{ui.esc(fmt_date(row.get("Date")))}</span></div></div></div>'
        f'<div style="display:flex;gap:22px;align-items:flex-end;flex-wrap:wrap">{metric_html}</div></div>'
    )

    # Action ribbon — the decision, stated with its exact reason.
    action = str(row.get("Action", theme.DASH))
    action_color, action_bg = theme.action_style(action)
    write(
        f'<div class="ws-ribbon" style="background:{action_bg};border:1px solid {action_color}22;'
        f'margin-bottom:16px">{ui.dot(action_color, 7)}'
        f'<span style="font-weight:800;color:{action_color};font-size:14px">{ui.esc(action)}</span>'
        f'<span style="color:var(--sub)">{ui.esc(row.get("Action_Reason", ""))}</span></div>'
    )

    _stock_chart(symbol, row)

    left, right = st.columns(2, gap="medium")
    with left:
        returns = "".join(
            f'<div style="text-align:center"><div class="ws-kv-label">{label}</div>'
            f'<div class="num" style="font-size:15px;font-weight:700;'
            f'color:{theme.signed_color(to_float(row.get(key)))}">{fmt_return(row.get(key))}</div></div>'
            for label, key in (("3M", "R3M"), ("6M", "R6M"), ("9M", "R9M"), ("12M", "R12M"))
        )
        range_block = ""
        if pd.notna(row.get("Low_52W")) and pd.notna(row.get("High_52W")):
            range_block = (
                '<div class="ws-card-title" style="margin:20px 0 12px">52-week range</div>'
                '<div style="display:flex;align-items:center;gap:10px">'
                f'<span class="num" style="font-size:12px;color:var(--sub)">{fmt_price(row.get("Low_52W"))}</span>'
                + ui.range_track(
                    row.get("Low_52W"), row.get("Close"), row.get("High_52W"), width=None, caps=False
                )
                + f'<span class="num" style="font-size:12px;color:var(--sub)">{fmt_price(row.get("High_52W"))}</span></div>'
                f'<div class="ws-note" style="margin-top:7px">{fmt_pct(row.get("Pct_From_52W_High"))} '
                "from its 52-week adjusted high.</div>"
            )
        write(
            ui.card(
                '<div class="ws-card-title">Calendar-month returns</div>'
                f'<div style="display:flex;justify-content:space-between">{returns}</div>'
                '<div class="ws-note" style="margin-top:10px">RS blends these as '
                "0.40×3M + 0.20×6M + 0.20×9M + 0.20×12M, then ranks cross-sectionally.</div>"
                + range_block
            )
        )
    with right:
        weinstein_conditions = [
            (label, bool(row.get(field, False)))
            for field, label in TREND_HEALTH_CONDITIONS
            if field in row.index
        ]
        write(_author_box("Weinstein", weinstein_conditions, signal_card.weinstein_line(row)))

    # Extension and structural risk
    ext = to_float(row.get("Ext_Pct"))
    ma30 = to_float(row.get("MA_30W"))
    close = to_float(row.get("Close"))
    risk = (ma30 / close - 1.0) * 100.0 if math.isfinite(ma30) and math.isfinite(close) and close else math.nan
    st.write("")
    write(
        ui.card(
            '<div class="ws-card-title">Extension and structural risk</div>'
            + ui.kv_grid(
                [
                    (
                        "Above the 30-week line",
                        fmt_pct(ext),
                        "beyond 20% is a guide timing warning" if math.isfinite(ext) else "",
                        CAUTION if math.isfinite(ext) and ext > 20 else "var(--ink)",
                    ),
                    ("30-week line", fmt_price(ma30), "locked calendar-window average", "var(--ink)"),
                    (
                        "10-week line",
                        fmt_price(row.get("MA_10W")),
                        "shorter reference (v2.1)",
                        "var(--ink)",
                    ),
                    (
                        "Return to the 30-week line",
                        fmt_pct(risk),
                        "if price retraced there",
                        NEGATIVE if math.isfinite(risk) and risk < 0 else "var(--ink)",
                    ),
                ]
            )
            + '<div class="ws-note" style="margin-top:12px">These are levels read from the completed '
            "price series, not targets and not a buy or sell call.</div>"
        )
    )

    # Three authorities, three boxes. Weinstein's sits above in the header
    # row; O'Neil's and Minervini's follow here, each an itemized checklist
    # plus that authority's own conclusion sentence and nothing else's —
    # separating them is the point, so a reader can see exactly which
    # criteria one book's method is satisfying without another's evidence
    # folded into the same card.
    st.write("")
    write('<div class="ws-eyebrow" style="margin-bottom:8px">O\'Neil</div>')
    write(_author_box("O'Neil", signal_card.oneil_checklist(row), signal_card.oneil_line(row)))

    # v2.2 — Minervini's box, deliberately its own section below: these
    # fields describe what has *not* happened yet, and mixing them into the
    # evidence for the current Action would blur the two.
    _prebreakout_section(row)

    # Cross-authority notes. Section 6 of the guide: when the signals
    # disagree, or a WAIT is waiting on something specific, or the label
    # alone misses a timing risk, none of that belongs to one author alone —
    # it is a statement about how two readings interact, so it sits after
    # all three boxes rather than inside any single one of them.
    interaction_notes = "".join(
        [
            ui.signal_note("wait", signal_card.wait_note(row, action)),
            ui.signal_note("conflict", signal_card.conflict_note(row)),
            ui.signal_note("caution", signal_card.caution_note(row)),
        ]
    )
    if interaction_notes:
        st.write("")
        write('<div class="ws-eyebrow" style="margin-bottom:8px">Where the readings interact</div>')
        write(ui.card(interaction_notes))

    # Bottom line — the same Action label and reason already shown in the
    # ribbon at the top, restated now that all three boxes have been seen.
    # Nothing new is computed here: it is the one locked decision, quoted
    # alongside each authority's own conclusion sentence. Minervini's line
    # is shown for context, not as a vote — action_for() in actions.py never
    # reads a v2.2 field, so a trend-template pass or a contraction setup
    # can sit beside any Action label at all, including WAIT or SELL.
    author_lines = [
        line
        for line in [
            signal_card.weinstein_line(row),
            signal_card.oneil_line(row),
            signal_card.minervini_line(row),
        ]
        if line
    ]
    if author_lines:
        st.write("")
        write('<div class="ws-eyebrow" style="margin-bottom:8px">Bottom line</div>')
        quotes = "".join(
            f'<div class="ws-note" style="margin-top:6px">{ui.esc(line)}</div>' for line in author_lines
        )
        write(
            ui.card(
                f'<div class="ws-ribbon" style="background:{action_bg};'
                f'border:1px solid {action_color}22;margin:0 0 2px">'
                f'{ui.dot(action_color, 7)}<span style="font-weight:800;color:{action_color};'
                f'font-size:14px">{ui.esc(action)}</span>'
                f'<span style="color:var(--sub)">{ui.esc(row.get("Action_Reason", ""))}</span></div>'
                + quotes
                + '<div class="ws-note" style="margin-top:10px">Action is Weinstein\'s stage gated by '
                "O'Neil's strength and volume — the two authorities whose evidence decides it. "
                "Minervini's reading above is context, not a vote: no Stage, RS ranking, breakout "
                "test or Action label reads a v2.2 field.</div>"
            )
        )

    # The remaining locked fields the Action spec requires exposed, grouped by
    # the measure each belongs to rather than dumped as one flat list.
    st.write("")
    write('<div class="ws-eyebrow" style="margin-bottom:8px">Calculation detail</div>')
    write(
        ui.evidence_grid(
            [
                ui.evidence_card(
                    "Leadership",
                    [
                        ("RS blend", fmt_number(row.get("RS_Blend"), 4),
                         "0.40×3M + 0.20×6M + 0.20×9M + 0.20×12M, before ranking."),
                        ("3M / 6M", " / ".join(fmt_return(row.get(k)) for k in ("R3M", "R6M")),
                         "Calendar-month returns to the latest completed session."),
                        ("9M / 12M", " / ".join(fmt_return(row.get(k)) for k in ("R9M", "R12M")),
                         "Referenced to the last session on or before each calendar date."),
                    ],
                ),
                ui.evidence_card(
                    "Trend structure",
                    [
                        ("30-week line", fmt_price(row.get("MA_30W")),
                         "Average of every valid session in a 30-calendar-week window."),
                        ("10-week line", fmt_price(row.get("MA_10W")),
                         "Same construction, shorter window. Not a Stage input."),
                        ("50-session average", fmt_price(row.get("SMA_50")),
                         "Mean close over the latest 50 completed sessions."),
                        ("Below the 50-session average", ui.state_pill(row.get("Below_50DMA"), "warn"),
                         "A timing warning; it does not reclassify Stage."),
                        ("30-week slope", fmt_pct(row.get("MA_30W_Slope_10S_Pct")),
                         "Change in the 30-week line over the latest 10 completed sessions."),
                    ],
                ),
                ui.evidence_card(
                    "Volume and momentum",
                    [
                        ("Volume ratio", fmt_ratio(row.get("Volume_Ratio")),
                         "Latest session's volume against the prior-50 baseline."),
                        ("U/D ratio", fmt_ratio(row.get("U_D")),
                         "Up-day volume over down-day volume; below 0.7 is distribution."),
                        ("Near the 52-week high", ui.state_pill(row.get("Near_52W_High"), "good"),
                         "Within 3% of the 52-week high — part of O'Neil's breakout setup."),
                    ]
                    + (
                        [
                            ("RS line", fmt_number(row.get("RS_Line"), 4),
                             "Close ÷ Nifty 500 on the sessions both actually traded."),
                            ("52-week RS line high", fmt_number(row.get("RS_Line_High_52W"), 4),
                             "Maximum RS line over the trailing 52 calendar weeks."),
                        ]
                        if "RS_Line" in row.index
                        else []
                    ),
                ),
                ui.evidence_card(
                    "52-week range",
                    [
                        ("52-week high", fmt_price(row.get("High_52W")),
                         "Maximum adjusted High over 52 calendar weeks, 200 sessions minimum."),
                        ("52-week low", fmt_price(row.get("Low_52W")),
                         "Minimum adjusted Low over the same window."),
                        ("Breakout setup", ui.state_pill(row.get("Breakout"), "good"),
                         "Stage 2, within 3% of the high, volume ratio above 1.5."),
                        ("Confirmed", ui.state_pill(row.get("Breakout_Confirmed"), "good"),
                         "The setup, plus U/D above 1.3. Never collapsed into one state."),
                    ],
                ),
                ui.evidence_card(
                    "Liquidity",
                    [
                        ("20-session traded value", fmt_inr(row.get("AvgValue20")),
                         "Mean of close × raw volume over the latest 20 completed sessions."),
                        ("Liquid", ui.state_pill(row.get("Liquid_UI_Filter"), "neutral"),
                         "Above ₹5 crore. A screener filter only; it never re-ranks RS."),
                        ("Decision date", fmt_date(row.get("Date")),
                         "The latest completed session before the upcoming decision session."),
                    ],
                ),
            ]
        )
    )
    write(
        '<div class="ws-note" style="margin-top:12px">Action interprets this evidence; it never '
        "replaces or hides it. Stage describes where price sits against its own 30-week line. "
        "Relative strength describes how the stock ranks against the rest of the universe. A "
        "stock can turn up structurally while still lagging, and the card says so.</div>"
    )


def _author_box(title: str, conditions: list[tuple[str, bool]], conclusion: str) -> str:
    """One authority's evidence: its own itemized checklist plus its own conclusion.

    Each authority gets a separate box rather than one flat signal card mixing
    all three, so a reader can see exactly which criteria one book's method
    is satisfying without the others' evidence in between. The conclusion is
    not free text: it is built in signal_card.py from the row's own values
    (weinstein_line / oneil_line / minervini_line), so it can only ever name a
    criterion the checklist above it already shows as met.
    """
    met = sum(1 for _, passed in conditions if passed)
    body = (
        ui.checklist(conditions)
        if conditions
        else '<div class="ws-note">The snapshot does not carry the checklist fields.</div>'
    )
    footer = (
        f'<div class="ws-note" style="margin-top:10px">{ui.esc(conclusion)}</div>'
        if conclusion
        else ""
    )
    count = f'<span class="num" style="color:var(--faint);font-weight:600">{met}/{len(conditions)}</span>' if conditions else ""
    return ui.card(
        f'<div class="ws-card-title" style="display:flex;align-items:center;gap:6px">'
        f'{ui.esc(title)}{count}</div>{body}{footer}'
    )


def _prebreakout_section(row: pd.Series) -> None:
    """v2.2 §4.1, §5.1, §10.4-10.6, §11.1 for one stock.

    Rendered only when the snapshot actually carries these fields. A snapshot
    published before v2.2 shows nothing here rather than a grid of em dashes,
    which would suggest the stock lacks the structure rather than the audit
    lacking the columns.
    """
    if not SNAP.has("Contraction_Ratio", "Trend_Template_Score"):
        return

    st.write("")
    write('<div class="ws-eyebrow" style="margin-bottom:8px">Minervini</div>')
    write(
        '<div class="ws-note" style="margin-bottom:10px">Before the move — everything else on '
        "this page measures what has already happened; this is the structure Minervini looks "
        "for before a stock breaks out.</div>"
    )

    def num(value, suffix="", digits=2, absent="—"):
        v = pd.to_numeric(value, errors="coerce")
        return absent if pd.isna(v) else f"{v:,.{digits}f}{suffix}"

    ratio = pd.to_numeric(row.get("Contraction_Ratio"), errors="coerce")
    dryup = pd.to_numeric(row.get("Volume_DryUp"), errors="coerce")
    to_pivot = pd.to_numeric(row.get("Pct_To_Pivot"), errors="coerce")
    score = pd.to_numeric(row.get("Trend_Template_Score"), errors="coerce")
    readiness = pd.to_numeric(row.get("Stage1_Readiness"), errors="coerce")

    cards = [
        ui.evidence_card(
            "Contraction and volume",
            [
                ("Range vs base start", num(ratio, "×"),
                 "good" if pd.notna(ratio) and ratio <= 0.60 else "neutral"),
                ("Volume dry-up", num(dryup, "×"),
                 "good" if pd.notna(dryup) and dryup <= 0.80 else "neutral"),
                ("Successive contractions", num(row.get("VCP_Contractions"), digits=0), "neutral"),
                ("ATR", num(row.get("ATR_Pct"), "%", 1), "neutral"),
                ("Setup", "Yes" if bool(row.get("VCP_Setup")) else "No",
                 "good" if bool(row.get("VCP_Setup")) else "neutral"),
            ],
        ),
        ui.evidence_card(
            "Pivot and template",
            [
                ("Base pivot", num(row.get("VCP_Pivot")), "neutral"),
                (
                    "Distance to pivot",
                    "Through it" if pd.notna(to_pivot) and to_pivot <= 0 else num(to_pivot, "%", 1),
                    "good" if pd.notna(to_pivot) and to_pivot <= 3.0 else "neutral",
                ),
                (
                    "Trend template",
                    "—" if pd.isna(score) else f"{int(score)} of 8",
                    "good" if pd.notna(score) and score == 8 else "neutral",
                ),
                (
                    "Stage 1 readiness",
                    "Not applicable" if pd.isna(readiness) else f"{int(readiness)} of 5",
                    "good" if pd.notna(readiness) and readiness >= 4 else "neutral",
                ),
            ],
        ),
    ]
    write(ui.evidence_grid(cards))

    template_conditions = [
        (label, bool(row.get(field, False)))
        for field, label in TREND_TEMPLATE_CONDITIONS
        if field in row.index
    ]
    if template_conditions:
        met = sum(1 for _, passed in template_conditions if passed)
        st.write("")
        write(
            ui.card(
                '<div class="ws-card-title" style="display:flex;align-items:center;gap:6px">'
                "Minervini trend-template checklist"
                f'<span class="num" style="color:var(--faint);font-weight:600">{met}/8</span></div>'
                + ui.checklist(template_conditions)
            )
        )

    minervini_conclusion = signal_card.minervini_line(row)
    if minervini_conclusion:
        write(f'<div class="ws-note" style="margin-top:8px">{ui.esc(minervini_conclusion)}</div>')

    write(
        '<div class="ws-note" style="margin:10px 0 0">None of these is a decision rule. '
        "No Stage, RS ranking, breakout test or Action label reads any field in this block, "
        "and a Stage 1 stock scoring 5 of 5 still carries the Stage 1 action. "
        "<b style=\"color:var(--ink)\">Three of the trend template's eight thresholds are "
        "provisional</b> — the 52-week-low, 52-week-high and RS cut-offs are transcribed from "
        "the published template and await verification against NSE history; the other five are "
        "structural comparisons and carry no invented number, as recorded in §5.1.</div>"
    )


def _stock_chart(symbol: str, row: pd.Series) -> None:
    """Price with the locked 10- and 30-week lines, drawn from the price panel.

    The moving averages are recomputed here from the published Close series
    using the same locked functions the audit used, so a drawn line can never
    drift from the definition it claims to show.
    """
    panel = cached_panel()[0]
    history = None if panel is None else panel.series(symbol)
    if history is None or len(history) < 40:
        missing("panel")
        return
    frame = pd.DataFrame(
        {"Close": history, "MA_10W": ma_10w_series(history), "MA_30W": ma_30w_series(history)}
    ).tail(260)
    points = ui.price_chart_payload(frame)
    write(
        ui.card(
            '<div class="ws-legend" style="margin-bottom:2px">'
            '<span class="item"><span style="width:14px;height:2.5px;background:var(--ink);'
            'border-radius:2px"></span>Close</span>'
            '<span class="item"><span style="width:14px;height:2px;background:#9DBDF0;'
            'border-radius:2px"></span>10-week line</span>'
            '<span class="item"><span style="width:14px;border-top:2px dashed #2D6CDF"></span>'
            "30-week line</span>"
            f'<span class="item num" style="margin-left:auto;color:var(--faint)">{len(frame)} '
            "completed sessions</span></div>",
            style="padding:14px 16px 6px",
        )
    )
    st.iframe(
        charts.line_chart(
            [
                {
                    "data": [{"time": p["time"], "value": p["Close"]} for p in points if "Close" in p],
                    "color": "#1a1d21",
                    "last_value": True,
                },
                {
                    "data": [{"time": p["time"], "value": p["MA_10W"]} for p in points if "MA_10W" in p],
                    "color": "#9DBDF0",
                },
                {
                    "data": [{"time": p["time"], "value": p["MA_30W"]} for p in points if "MA_30W" in p],
                    "color": "#2D6CDF",
                    "dashed": True,
                },
            ],
            element_id="price",
            height=340,
            unavailable="The price chart could not be drawn in this environment.",
        ),
        height=356,
    )


# --- Methodology ------------------------------------------------------------
def page_methodology() -> None:
    heading(
        "Methodology",
        "Plain-language definitions of every calculation behind the tables, the information "
        "boundary they respect, and the interpretation layer that sits on top without replacing them.",
    )
    sections = [
        (
            "The universe",
            "The official Nifty Total Market constituent CSV is authoritative; the live count is read "
            "from the file rather than hard-coded. Symbols NSE reserves for corporate actions (the "
            "DUMMY prefix) are excluded before anything is computed, so the analytical universe is "
            f"{len(SNAP.universe):,} constituents in this snapshot, of which {len(DATA):,} carry "
            "sufficient history to classify. Industry is exactly the CSV's Industry field, never "
            "remapped. There is no F&O filter.",
        ),
        (
            "Before the move — v2.2",
            "Everything else on this site measures what has already happened: a breakout is "
            "confirmed after it breaks out. Locked-spec v2.2 adds the structure that precedes "
            "it, sourced from Minervini alongside Weinstein and O'Neil rather than invented. "
            "The RS line is Close ÷ Nifty 500 on the sessions both actually traded; when it "
            "reaches a 52-week high while price is still 5% or more below its own, relative "
            "strength is leading price — O'Neil's tell. Contraction splits the 50-session base "
            "into five blocks and compares the last block's high-low range against the first; "
            "volume dry-up compares the last ten sessions against the fifty before them, which "
            "is the opposite instrument to the volume ratio used for breakouts. The pivot is "
            "the base's highest high. Stage 1 readiness counts five conditions so the largest "
            "and most undifferentiated bucket can be ranked. None of it is a decision rule: no "
            "Stage, RS ranking, breakout test or Action label reads any v2.2 field. "
            "Three of the trend template's eight thresholds are provisional — the 52-week-low, "
            "52-week-high and RS cut-offs are transcribed from the published template and await "
            "verification against NSE history; the other five are structural comparisons and "
            "carry no invented number.",
        ),
        (
            "Information boundary",
            "Decisions are pre-market for the upcoming session. For decision session D only information "
            "through the latest completed NSE session T may be used; no upcoming or incomplete session "
            "enters any calculation. Missing history produces explicit insufficiency, never a "
            "fabricated value.",
        ),
        (
            "Relative strength",
            "Returns are measured over 3, 6, 9 and 12 calendar months, each referenced to the last "
            "session on or before the calendar date. They blend as 0.40×3M + 0.20×6M + 0.20×9M + "
            "0.20×12M with no skip month. The cross-sectional score is rank(blend, pct, method='min') "
            "× 98 + 1, rounded to an integer 1–99. Bands: 80–99 leadership, 50–79 adequate, below 50 "
            "lagging. Ranking happens before any liquidity filter, so UI filters never change a rank.",
        ),
        (
            "Stage",
            "The 30-week line is a simple average over every valid session in a 30-calendar-week window "
            "ending at T — not a fixed 150-row trading-day average. Slope is the 10-session percentage "
            "change in that average. Stage 2 Advancing: close above the line, slope positive. Stage 3 "
            "Topping: above, slope not positive. Stage 4 Declining: at or below, slope not positive. "
            "Stage 1 Basing: at or below, slope positive. Stage is categorical and is never treated "
            "as a number.",
        ),
        (
            "The 10-week line",
            "Adopted in locked-spec v2.1, built with exactly the same calendar-window construction as "
            "the 30-week line so the two are directly comparable. It is a trend reference and a "
            "checklist input; it does not reclassify Stage and no locked signal depends on it.",
        ),
        (
            "52-week range, volume and accumulation",
            "The 52-week high is the maximum adjusted High over the preceding 52 calendar weeks, "
            "requiring at least 200 valid sessions; the 52-week low mirrors it exactly. Proximity means "
            "close at or above 97% of that high. Volume ratio divides the latest completed session's "
            "volume by the mean of the 50 sessions before it, which excludes the session itself. U/D "
            "sums up-volume against down-volume over the 20 completed sessions ending at T; unchanged "
            "closes count as neither. Bands: above 1.5 strong accumulation, above 1.3 accumulating, "
            "0.7–1.3 neutral, below 0.7 distribution warning, below 0.6 heavy distribution.",
        ),
        (
            "Breakout and confirmation",
            "A breakout setup requires Stage 2, a close within 3% of the 52-week high and a volume "
            "ratio above 1.5. Confirmation additionally requires U/D above 1.3. The two states are "
            "reported separately and are never collapsed into one.",
        ),
        (
            "Timing warnings",
            "Extended is close above 1.20 × the 30-week line. Below-50DMA is close under the 50 "
            "completed-session average. Both are timing warnings that gate the Action; neither "
            "reclassifies Stage. The guide's pullback-with-volume-drying condition was held open "
            "through v2.1 for want of a precise definition; v2.2 closes it by sourcing the "
            "concept from Minervini rather than inventing one. See \u201cBefore the move\u201d above.",
        ),
        (
            "The Action layer",
            "Stage 4 is always SELL. Stage 3 is SELL below RS 50, otherwise REDUCE. Stage 1 is WATCH★ "
            "at RS 80+, WATCH at 50–79, AVOID below 50. In Stage 2, distribution gives REDUCE; RS "
            "below 50 gives WAIT; RS 50–79 gives WAIT on a breakout and otherwise HOLD; at RS 80+ an "
            "extension or 50DMA warning gives WAIT, a confirmed breakout gives BUY★, an unconfirmed "
            "setup gives BUY, and otherwise HOLD. Stage takes precedence over RS wherever the two "
            "conflict. Action is an interpretation of the evidence beside it and never modifies, "
            "replaces or hides the underlying calculation.",
        ),
        (
            "Liquidity",
            "The 20-session mean of close × raw volume; liquid means above ₹5 crore. This is a "
            "presentation filter only and never re-ranks the RS universe.",
        ),
        (
            "Attribution and limits",
            "Stage structure and the 30-week average follow Stan Weinstein; relative-strength "
            "leadership and breakout principles follow William O'Neil; the trend template and the "
            "volatility-contraction pattern added in v2.2 follow Mark Minervini, whose numeric "
            "criteria are implemented verbatim while the contraction detector is ours, and whose "
            "template thresholds remain provisional pending verification against the source text. "
            "No Action label reads any v2.2 field. The nine-label mechanical "
            "mapping is this project's specification adopted from the supplied NSE Signal "
            "Interpretation Guide, not a verbatim rule from either book. The visual system and layout "
            "follow the WealthStar reference terminal as a design source. Prices are end-of-day from "
            "yfinance with auto-adjustment; volume is unadjusted. Figures are for study, not "
            "real-time execution.",
        ),
    ]
    for title, body in sections:
        write(ui.card(f'<div class="ws-card-title">{ui.esc(title)}</div><div class="ws-note">{ui.esc(body)}</div>'))

    st.write("")
    write('<div class="ws-eyebrow" style="margin-bottom:8px">This snapshot</div>')
    st.json(
        {
            "decision_date": None if decision is None else decision.strftime("%Y-%m-%d"),
            "previous_session": None
            if SNAP.previous_date is None
            else SNAP.previous_date.strftime("%Y-%m-%d"),
            "universe_rows": int(len(SNAP.universe)),
            "research_rows": int(len(DATA)),
            "valid_rs": int(BREADTH.get("valid_rs", 0)),
            "stages": BREADTH["stages"],
            "above_30_week_pct": round(float(BREADTH["pct_above_ma_30w"]), 2),
            "confirmed_breakouts": int(BREADTH["breakout_confirmed"]),
            "price_panel_available": cached_panel()[0] is not None,
            "breadth_history_sessions": 0 if SNAP.breadth is None else int(len(SNAP.breadth)),
            "unavailable": SNAP.missing,
        }
    )


# --- Setups -----------------------------------------------------------------
#: v2.2 §4.1, §10.5, §10.6, §11.1. Every other view answers "what has already
#: happened"; this one answers "what is coiling". They are different questions
#: and the evidence columns differ accordingly.
SETUP_SORTS = {
    "Evidence (most first)": ("Setup_Evidence", False),
    "Readiness (best first)": ("Stage1_Readiness", False),
    "Closest to pivot": ("Pct_To_Pivot", True),
    "Tightest contraction": ("Contraction_Ratio", True),
    "Driest volume": ("Volume_DryUp", True),
    "RS (high to low)": ("RS_Score", False),
    "Template score": ("Trend_Template_Score", False),
}


def page_setups() -> None:
    heading(
        "Setups",
        "Stocks that have not moved yet. Every other view ranks what the market has already "
        "done; this one ranks the evidence that precedes it — relative strength leading price, "
        "a contracting range, drying volume, and distance still to travel to the pivot.",
    )

    if missing("v22_fields"):
        return

    view = DATA.copy()
    total_universe = len(view)

    # How many of the four published conditions a stock satisfies at once.
    # Composed from fields already in the snapshot — this introduces no new
    # rule, in the same way the Screener presets do not. The sections are not
    # alternatives to choose between: they sit at different distances from the
    # same move, so a name meeting several is carrying more evidence than a
    # name meeting one, and that was invisible while each section was its own
    # separate list.
    def _met(frame: pd.DataFrame, field: str) -> pd.Series:
        if field not in frame.columns:
            return pd.Series(False, index=frame.index)
        if field == "Stage1_Readiness":
            return pd.to_numeric(frame[field], errors="coerce") >= 4
        return frame[field].fillna(False).astype(bool)

    view["Setup_Evidence"] = sum(
        _met(view, field).astype(int) for _, field in ui.SETUP_CONDITIONS
    )

    groups = st.segmented_control(
        "Setup",
        ["Trend template", "RS leading price", "Contracting base", "Stage 1 ready",
         "Stacked (2+)", "All"],
        default="Trend template",
        key="setups_group",
    )
    row = st.columns([2, 2, 2])
    sort_label = row[0].selectbox("Sort by", list(SETUP_SORTS), key="setups_sort")
    industries = ["All"] + sorted(DATA["Industry"].dropna().astype(str).unique().tolist())
    industry = row[1].selectbox("Industry", industries, key="setups_industry")
    liquid_only = row[2].toggle(
        "Liquid only", value=True, key="setups_liquid",
        help="20-session traded value above ₹5 Cr. On by default here: a base that cannot be "
             "bought in size is not an opportunity.",
    )

    explain = {
        "Trend template": (
            "All eight of Minervini's trend-template criteria satisfied: price above the "
            "150- and 200-session averages, those averages correctly stacked and the 200 "
            "rising, price above the 50-session average, well off the 52-week low, near the "
            "52-week high, and RS 70 or better. Three of the eight thresholds are the "
            "source's stated values for a different market and era — they are provisional "
            "here and have not been validated against NSE history.",
            lambda f: f[f["Trend_Template_Pass"].fillna(False).astype(bool)],
        ),
        "RS leading price": (
            "Relative strength at a 52-week high while price is at least 5% below its own. "
            "O'Neil's leading tell: the stock is outperforming from inside its base.",
            lambda f: f[f["RS_Line_NH_Before_Price"].fillna(False).astype(bool)],
        ),
        "Contracting base": (
            "Range tightening to 0.60× or less across the 50-session base, volume drying to "
            "0.80× or less, and at least two successive contractions. Minervini's pattern; "
            "the detector is ours, stated in §10.5.",
            lambda f: f[f["VCP_Setup"].fillna(False).astype(bool)],
        ),
        "Stage 1 ready": (
            "Stage 1 stocks scoring 4 or 5 of the readiness count: the decline has stopped, "
            "RS is no longer lagging, the range is tightening, volume is drying and the "
            "10-week line has been reclaimed.",
            lambda f: f[pd.to_numeric(f["Stage1_Readiness"], errors="coerce") >= 4],
        ),
        "Stacked (2+)": (
            "Names satisfying two or more of the four conditions at once. The sections above "
            "are the same move seen at different distances — earliest and least certain "
            "through to latest and most confirmed — so a stock appearing in several is "
            "carrying more evidence than one appearing in a single list.",
            lambda f: f[f["Setup_Evidence"] >= 2],
        ),
        "All": (
            "The whole validated universe with the pre-breakout evidence columns, unfiltered.",
            lambda f: f,
        ),
    }
    label = groups if groups in explain else "Trend template"
    description, predicate = explain[label]

    if liquid_only and "Liquid_UI_Filter" in view.columns:
        view = view[view["Liquid_UI_Filter"].fillna(False).astype(bool)]
    if industry != "All":
        view = view[view["Industry"].astype(str) == industry]
    view = predicate(view)

    column, ascending = SETUP_SORTS[sort_label]
    if column in view.columns:
        sortable = view[pd.to_numeric(view[column], errors="coerce").notna()]
        unsortable = view[pd.to_numeric(view[column], errors="coerce").isna()]
        view = pd.concat(
            [sortable.sort_values(column, ascending=ascending), unsortable]
        )

    write(f'<div class="ws-note" style="margin:2px 0 10px">{ui.esc(description)}</div>')

    if view.empty:
        write(
            ui.missing_notice(
                f"No stock currently matches “{ui.esc(label)}”.",
                "This is a real reading of today's snapshot, not a missing artifact: the "
                f"condition was evaluated across all {total_universe:,} constituents and none "
                "met it. Setups of this kind are intermittent by nature — a market with no "
                "coiling leaders is information, not an error.",
            )
        )
        return

    write(
        f'<div class="ws-note" style="margin:2px 0 10px"><b style="color:var(--ink)">{len(view):,}</b> '
        f"of {total_universe:,} constituents · sorted by {ui.esc(sort_label.lower())}</div>"
    )
    write(
        ui.screener_table(
            view.head(PAGE_SIZE), cached_sparklines(), columns=ui.SETUP_COLUMNS
        )
    )
    if len(view) > PAGE_SIZE:
        with st.expander(f"The remaining {len(view) - PAGE_SIZE:,}"):
            write(
                ui.screener_table(
                    view.iloc[PAGE_SIZE:], cached_sparklines(), columns=ui.SETUP_COLUMNS
                )
            )

    st.download_button(
        "Download this list (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name=f"rs-stages-setups-{label.lower().replace(' ', '-')}.csv",
        mime="text/csv",
        key="setups_csv",
    )

    write(
        '<div class="ws-note" style="margin:14px 0 0">A setup is not a signal. None of these '
        "conditions is a locked Action input: every stock above still carries whatever Action "
        "its Stage, RS and volume produced, and Stage 1 remains Stage 1 however ready its base "
        "looks. This view ranks what to watch, not what to buy.</div>"
    )


PAGES = {
    "Dashboard": page_dashboard,
    "Setups": page_setups,
    "Screener": page_screener,
    "Industries": page_industries,
    "Market": page_market,
    "Movers": page_movers,
    "Stock": page_stock,
    "Methodology": page_methodology,
}
PAGES.get(VIEW, page_dashboard)()
write(theme.FOOTER)
