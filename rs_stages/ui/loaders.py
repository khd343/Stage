"""Loading of the published research artifacts for the presentation layer.

The UI reads only what the audit published. Nothing here recalculates a locked
field, and nothing substitutes a value for one that is absent: when an artifact
has not been generated yet, the loader says so and the page renders an explicit
notice instead of a plausible-looking number.
"""
from __future__ import annotations

import io
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..actions import with_actions
from ..data import load_nse_constituents_csv

DATA_DIR = Path("data")
RESEARCH_PATH = DATA_DIR / "latest_research.csv"
PREVIOUS_PATH = DATA_DIR / "previous_research.csv"
UNIVERSE_PATH = DATA_DIR / "ind_niftytotalmarket_list.csv"
PANEL_PATH = DATA_DIR / "price_panel.npz"

#: The price panel is published as a release asset rather than committed: it is
#: a regenerated binary that Git cannot delta, so committing it would add a
#: fresh ~1.4 MB blob to history on every audit run, permanently.
#: Default points at THIS repository's own release. A fork that keeps the
#: upstream URL draws its charts from someone else's panel while its CSVs come
#: from its own audit -- two different vintages on one screen, and the failure is
#: silent until the upstream release is deleted or made private, at which point
#: every chart disappears at once. Override with RS_STAGES_PANEL_URL if the panel
#: is ever hosted elsewhere.
PANEL_URL = os.environ.get(
    "RS_STAGES_PANEL_URL",
    "https://github.com/khd343/Stage/releases/download/data-latest/price_panel.npz",
)
PANEL_TIMEOUT_SECONDS = 30
BREADTH_PATH = DATA_DIR / "breadth_history.csv"

#: Fields introduced by locked-spec v2.1. A snapshot published before that
#: revision simply lacks them; the pages that need them degrade explicitly.
V21_FIELDS = ("Close", "MA_10W", "Low_52W", "Ext_Pct", "Pct_From_52W_High", "Trend_Health")

#: Fields introduced by locked-spec v2.2 — the pre-breakout structure. A
#: snapshot published before that revision lacks them entirely, so the Setups
#: view reports that the audit has not been re-run rather than rendering an
#: empty table that looks like "no setups found". The two readings are opposite
#: and a reader cannot tell them apart from an empty table alone.
V22_FIELDS = (
    "RS_Line",
    "RS_Line_NH_Before_Price",
    "Contraction_Ratio",
    "Volume_DryUp",
    "VCP_Setup",
    "Pct_To_Pivot",
    "Trend_Template_Score",
    "Stage1_Readiness",
)

REGENERATE_HINT = (
    "Run the Real Data Research Audit workflow to publish it. Until then this "
    "section stays empty rather than showing a value the snapshot cannot support."
)


@dataclass
class DateCoverage:
    """How the universe splits across information dates within one snapshot.

    ``Date`` is set per symbol from that symbol's own latest completed session
    (screener.py), not from one shared clock. The provider updates its feed
    asynchronously — larger, more liquid names first — so on any given run a
    fraction of the universe can still carry the previous session while the
    rest has moved on. 24-25 Aug 2026 split roughly 305/445 this way, with the
    lagging group's median 20-session traded value about a third of the
    leading group's: a liquidity effect, not a random one.

    This is not corrected by waiting for the slowest symbol before publishing,
    which is a defect of a different, worse kind: one thin, illiquid stock
    could then hold back the other 749 indefinitely. It is corrected by never
    describing a split universe as one date.
    """

    latest: pd.Timestamp | None
    counts: dict[pd.Timestamp, int]

    @property
    def is_split(self) -> bool:
        return len(self.counts) > 1

    @property
    def current_count(self) -> int:
        return self.counts.get(self.latest, 0) if self.latest is not None else 0

    @property
    def lagging_count(self) -> int:
        return sum(self.counts.values()) - self.current_count

    @property
    def lagging_pct(self) -> float:
        total = sum(self.counts.values())
        return (self.lagging_count / total * 100.0) if total else 0.0


@dataclass
class Snapshot:
    """Everything the UI is allowed to read, plus what is missing and why."""

    research: pd.DataFrame
    universe: pd.DataFrame
    previous: pd.DataFrame | None = None
    breadth: pd.DataFrame | None = None
    missing: dict[str, str] = field(default_factory=dict)

    @property
    def decision_date(self) -> pd.Timestamp | None:
        stamp = pd.to_datetime(self.research.get("Date"), errors="coerce").max()
        return None if pd.isna(stamp) else stamp

    @property
    def date_coverage(self) -> DateCoverage:
        dates = pd.to_datetime(self.research.get("Date"), errors="coerce").dropna()
        if dates.empty:
            return DateCoverage(latest=None, counts={})
        counts = dates.dt.normalize().value_counts().to_dict()
        return DateCoverage(latest=dates.max().normalize(), counts=counts)

    @property
    def previous_date(self) -> pd.Timestamp | None:
        if self.previous is None:
            return None
        stamp = pd.to_datetime(self.previous.get("Date"), errors="coerce").max()
        return None if pd.isna(stamp) else stamp

    def has(self, *columns: str) -> bool:
        return all(column in self.research.columns for column in columns)


def _read_research(path: Path, universe: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Symbol"] = frame["Symbol"].astype(str).str.strip()

    # The universe CSV is authoritative for Industry and Company Name; the
    # snapshot may already carry them from the audit's own join.
    columns = ["Symbol"] + [c for c in ("Industry", "Company Name") if c in universe.columns]
    merged = frame.merge(
        universe[columns].drop_duplicates("Symbol"), on="Symbol", how="left", suffixes=("", "_u")
    )
    for column in ("Industry", "Company Name"):
        alias = f"{column}_u"
        if alias in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].fillna(merged[alias])
            else:
                merged[column] = merged[alias]
            merged = merged.drop(columns=[alias])

    merged["Stage_Label"] = merged["Stage"].map(lambda v: str(v).split(" — ", 1)[0])
    # Action is recomputed from the published columns with the same deterministic
    # function the audit used, so the table and the snapshot cannot disagree.
    return with_actions(merged)


@dataclass(frozen=True)
class PricePanel:
    """Completed-session closes as a dense sessions x symbols grid.

    Every symbol shares the same session calendar, so the panel is a matrix
    rather than a long table. Reading it needs NumPy only — no Arrow runtime is
    involved in drawing a chart.
    """

    dates: pd.DatetimeIndex
    symbols: tuple[str, ...]
    close: "np.ndarray"

    def series(self, symbol: str) -> pd.Series | None:
        """Close series for one symbol, or None if it is not in the panel."""
        try:
            column = self.symbols.index(str(symbol))
        except ValueError:
            return None
        values = self.close[:, column]
        series = pd.Series(values, index=self.dates, dtype="float64").dropna()
        return series if not series.empty else None

    def tails(self, sessions: int) -> dict[str, list[float]]:
        """Trailing closes per symbol, for the sparkline column."""
        window = self.close[-sessions:, :]
        out: dict[str, list[float]] = {}
        for column, symbol in enumerate(self.symbols):
            values = window[:, column]
            values = values[~np.isnan(values)]
            if len(values) >= 2:
                out[symbol] = values.tolist()
        return out

    @property
    def terminal_session(self) -> pd.Timestamp | None:
        return None if len(self.dates) == 0 else pd.Timestamp(self.dates[-1])


def _read_panel(source) -> PricePanel:
    with np.load(source, allow_pickle=False) as payload:
        return PricePanel(
            dates=pd.DatetimeIndex(payload["dates"]),
            symbols=tuple(str(s) for s in payload["symbols"]),
            close=payload["close"],
        )


def load_price_panel() -> tuple[PricePanel | None, str | None]:
    """Return the price panel, or None plus the reason it is unavailable.

    A local file wins when present, so a developer can work from a panel they
    generated themselves and the app needs no network in that case. Otherwise
    the published release asset is downloaded. Nothing is fabricated when both
    fail: the caller renders an explicit notice.

    This is deliberately separate from :func:`load_snapshot`. The panel is by
    far the largest artifact, and the pages that do not draw price history must
    never pay to load it.
    """
    if PANEL_PATH.exists():
        try:
            return _read_panel(PANEL_PATH), None
        except (OSError, ValueError, KeyError) as exc:
            return None, f"The local price panel could not be read ({type(exc).__name__})."

    if not PANEL_URL:
        return None, "No price panel is available and no panel URL is configured."

    try:
        with urllib.request.urlopen(PANEL_URL, timeout=PANEL_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, (
            f"The published price panel could not be downloaded ({type(exc).__name__}), so "
            "price history, trend lines and sparklines are unavailable. Everything else on "
            "this page comes from the committed snapshot and is unaffected."
        )

    try:
        return _read_panel(io.BytesIO(payload)), None
    except (OSError, ValueError, KeyError) as exc:
        return None, f"The downloaded price panel could not be read ({type(exc).__name__})."


def panel_matches(panel: PricePanel, research: pd.DataFrame) -> str | None:
    """Return a reason string when the panel disagrees with the snapshot.

    The panel and the snapshot are published to different places, so they can
    drift. A panel from a different decision date is rejected rather than drawn:
    a chart and a table must never describe different sessions.
    """
    decision = pd.to_datetime(research["Date"], errors="coerce").max()
    terminal = panel.terminal_session
    if pd.isna(decision) or terminal is None:
        return None
    if terminal.normalize() == pd.Timestamp(decision).normalize():
        return None
    return (
        f"The published price panel ends at {terminal:%d %b %Y} but this snapshot's decision "
        f"date is {pd.Timestamp(decision):%d %b %Y}. Price history is withheld rather than "
        "drawn against a different session. Re-run the Real Data Research Audit so both are "
        "published from the same run."
    )


def load_snapshot() -> Snapshot:
    """Read every published artifact, recording whatever is unavailable."""
    missing: dict[str, str] = {}
    # The same locked loader the audit uses, so the UI's universe *is* the
    # analytical universe. Reading the CSV raw here counted the DUMMY rows NSE
    # reserves for corporate actions, so the header advertised 752 constituents
    # while every figure beneath it was computed over 750.
    universe = load_nse_constituents_csv(UNIVERSE_PATH)
    research = _read_research(RESEARCH_PATH, universe)

    absent = [column for column in V21_FIELDS if column not in research.columns]
    if absent:
        missing["v21_fields"] = (
            "This snapshot predates locked-spec v2.1, so it carries no "
            + ", ".join(absent)
            + ". "
            + REGENERATE_HINT
        )

    absent_v22 = [column for column in V22_FIELDS if column not in research.columns]
    if absent_v22:
        missing["v22_fields"] = (
            "This snapshot predates locked-spec v2.2, so it carries none of the "
            "pre-breakout structure — no RS line, contraction, volume dry-up, "
            "pivot distance, trend template or Stage 1 readiness. Setups cannot "
            "be listed from it, and an empty list would read as 'no setups "
            "found' rather than 'not yet computed'. " + REGENERATE_HINT
        )

    previous = None
    if PREVIOUS_PATH.exists():
        try:
            previous = _read_research(PREVIOUS_PATH, universe)
        except (OSError, ValueError, KeyError) as exc:
            missing["previous"] = f"The previous-session snapshot could not be read ({type(exc).__name__})."
    else:
        missing["previous"] = (
            "No previous-session snapshot has been published, so day-over-day changes "
            "cannot be computed. " + REGENERATE_HINT
        )

    breadth = None
    if BREADTH_PATH.exists():
        try:
            breadth = pd.read_csv(BREADTH_PATH)
            breadth["Date"] = pd.to_datetime(breadth["Date"], errors="coerce")
            breadth = breadth.dropna(subset=["Date"]).sort_values("Date")
        except (OSError, ValueError) as exc:
            missing["breadth"] = f"The breadth history could not be read ({type(exc).__name__})."
            breadth = None
    else:
        missing["breadth"] = (
            "No breadth history has been published, so the participation trend cannot "
            "be drawn. " + REGENERATE_HINT
        )

    return Snapshot(
        research=research,
        universe=universe,
        previous=previous,
        breadth=breadth,
        missing=missing,
    )
