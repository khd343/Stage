"""The forward record: frozen snapshots, pre-registered cohorts, append-only grades.

Nothing this system publishes could ever be graded, because every run overwrote
the last. This module is the answer, and it is built around one idea: a forecast
is evidence only if it was written down before its outcome existed and never
edited after. So:

  * ARCHIVE     -- each validated snapshot is frozen under the session it
                   describes, first write wins, never rewritten.
  * COHORTS     -- the signals to be graded are named HERE, in advance, and
                   pinned by a test. Changing one is a new record, not an edit.
  * GRADE       -- forward returns are read from ONE fresh price series per
                   symbol, never from the archived Close: the archive was
                   adjusted as of its day and a later download as of today, and
                   a split in between would put the two endpoints on different
                   bases (the cross-basis rule).
  * APPEND ONLY -- a grade, once written, is never recomputed. Coverage is
                   recorded so a symbol that died is a visible hole, not a
                   silent omission.

The archive is deliberately WIDE and the rules are deliberately NARROW. Keeping
a column costs kilobytes; not keeping it costs the question, permanently. But a
wide archive must not become a wide set of claims -- archiving evidence and
pre-registering a hypothesis are different acts, and only the second one is
graded. So ARCHIVE_EXCLUDE may shrink freely while COHORTS stays fixed, and
rules_version() ignores the archive schema entirely.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from rs_stages.corporate_actions import spans_discontinuity

#: The minimum a snapshot must carry to be gradeable at all: identity, plus
#: every column a pre-registered cohort reads. Archiving is refused without
#: these, because a snapshot frozen without them can never be graded and first
#: write wins, so the loss is permanent. A test pins this against the cohort
#: rules themselves.
REQUIRED_COLUMNS = [
    "Symbol", "Date", "Close", "Action", "Stage", "RS_Score",
    "Trend_Template_Pass", "Breakout_Confirmed",
]

#: Everything else the engine produced is archived too. Curating the list now
#: means guessing which question matters in 2027, and this repo has a standing
#: lesson about confident predictions of what the data will show. A column not
#: written is unanswerable forever; a column written and never used costs
#: kilobytes.
#:
#: These three are the exception, and only because they are genuinely free to
#: recover: they never change, and they already live in the version-controlled
#: universe file, so any historical snapshot can be joined back to them.
ARCHIVE_EXCLUDE = frozenset({"Company Name", "Industry", "Series"})

#: Floats are rounded on the way in. Four decimals on a rupee price is a
#: hundredth of a paisa, and nothing reads the archived Close as a number --
#: the cross-basis rule forbids it. Measured over the live 1,784-row snapshot:
#: 63 columns raw cost 123 MB/year, rounded 63 MB/year.
ARCHIVE_DECIMALS = 4

#: Calendar weeks forward at which each snapshot is graded.
HORIZONS_WEEKS = (4, 8, 13)

#: Sessions that must exist past D + H before that horizon is graded, so the
#: end close has settled at the provider (it has not, at 16 hours -- measured).
SETTLE_SESSIONS = 3

#: PRE-REGISTERED. These are the questions the record exists to answer, fixed
#: before the first snapshot was archived. `universe` is the control every
#: other cohort is measured against. A test pins this dict verbatim; the
#: rules_version() hash rides in every graded row so a change is visible in the
#: data itself. Do not tune a threshold after seeing a result -- that is the
#: one act that makes a forward record worthless.
COHORTS: dict[str, str] = {
    "universe":           "every published row (the control)",
    "buy_star":           "Action == 'BUY★'",
    "stage2_rs80":        "Stage starts with 'Stage 2' and RS_Score >= 80",
    "trend_template":     "Trend_Template_Pass is true",
    "breakout_confirmed": "Breakout_Confirmed is true",
}

TRACK_COLUMNS = [
    "snapshot_date", "horizon_weeks", "cohort", "n", "n_priced",
    "median_return_pct", "mean_return_pct", "hit_rate",
    "universe_median_pct", "excess_vs_universe_pp",
    "benchmark_return_pct", "excess_vs_benchmark_pp",
    "rules_version", "graded_on",
]


def rules_version() -> str:
    """A fingerprint of everything that defines a grade. Changes when the rules do."""
    blob = json.dumps({"cohorts": COHORTS, "horizons": HORIZONS_WEEKS,
                       "required": REQUIRED_COLUMNS, "settle": SETTLE_SESSIONS}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _truthy(values: pd.Series) -> pd.Series:
    """True for bool True and for the strings a CSV round-trip produces."""
    return values.astype(str).str.strip().str.lower().isin({"true", "1"})


def cohort_members(frame: pd.DataFrame, cohort: str) -> pd.Index:
    """Symbols in `cohort`, by the pre-registered rule. Unknown cohort raises."""
    if cohort not in COHORTS:
        raise KeyError(f"unregistered cohort {cohort!r}; the record only answers pre-registered questions")
    f = frame
    if cohort == "universe":
        mask = pd.Series(True, index=f.index)
    elif cohort == "buy_star":
        mask = f["Action"].astype(str).str.strip() == "BUY★"
    elif cohort == "stage2_rs80":
        rs = pd.to_numeric(f["RS_Score"], errors="coerce")
        mask = f["Stage"].astype(str).str.startswith("Stage 2") & (rs >= 80)
    elif cohort == "trend_template":
        mask = _truthy(f["Trend_Template_Pass"])
    else:  # breakout_confirmed
        mask = _truthy(f["Breakout_Confirmed"])
    return pd.Index(f.loc[mask.fillna(False), "Symbol"].astype(str))


def archive_snapshot(result: pd.DataFrame, snapshots_dir: Path, boundary: pd.Timestamp) -> Path | None:
    """Freeze the published snapshot under its session. First write wins.

    Keyed on the SESSION, not the run: several runs a day can publish the same
    boundary and they all describe one session. A later run never rewrites an
    earlier file, even if the provider has revised -- a record whose past can
    change proves nothing, and a revision arriving after the fact is exactly
    the edit that would make a stale call look prescient.
    """
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    path = snapshots_dir / f"{pd.Timestamp(boundary):%Y-%m-%d}.csv.gz"
    if path.exists():
        return None
    frame = result.reset_index() if "Symbol" not in result.columns else result
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"snapshot lacks grading column(s) {missing}; refusing to archive a partial record")
    frozen = frame[[c for c in frame.columns if c not in ARCHIVE_EXCLUDE]].copy()
    for column in frozen.columns:
        if pd.api.types.is_float_dtype(frozen[column]):
            frozen[column] = frozen[column].round(ARCHIVE_DECIMALS)
    payload = frozen.to_csv(index=False).encode("utf-8")
    path.write_bytes(gzip.compress(payload, compresslevel=9))
    return path


def read_snapshot(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, dtype={"Symbol": str})


def snapshot_date(path: Path) -> pd.Timestamp:
    return pd.Timestamp(path.name.split(".", 1)[0])


def forward_return(closes: pd.Series, start: pd.Timestamp, weeks: int,
                   guard_corporate_actions: bool = True) -> float:
    """Close-to-close return, percent, both ends from ONE series.

    The start is the session on `start` itself (the snapshot's session); the
    end is the first session on or after start + weeks. Either missing -> NaN,
    never a substituted neighbour. The archived Close is deliberately NOT the
    start: it was adjusted as of its day, this series as of today, and one
    split between them would make the return a fiction.

    ONE SERIES IS NOT ENOUGH; IT MUST ALSO BE CONTINUOUS. A demerger inside the
    window moves the two endpoints onto different bases even though both came
    from this download, and yfinance's auto_adjust does not restate demergers.
    Measured on this universe: roughly one such event per two gradings of a
    selective cohort. A refused return becomes a coverage hole, counted in `n`
    and absent from `n_priced`, which is the honest report -- the record is
    append-only, so a fiction written once can never be taken back.
    """
    s = closes.sort_index().dropna()
    if s.empty:
        return float("nan")
    start = pd.Timestamp(start).normalize()
    if start not in s.index:
        return float("nan")
    target = start + pd.Timedelta(weeks=weeks)
    pos = s.index.searchsorted(target, side="left")
    if pos >= len(s):
        return float("nan")
    a, b = float(s.loc[start]), float(s.iloc[pos])
    if not (a > 0):
        return float("nan")
    if guard_corporate_actions and spans_discontinuity(s, start, s.index[pos]):
        return float("nan")
    return (b / a - 1.0) * 100.0


def gradeable(snapshot_day: pd.Timestamp, weeks: int, calendar: pd.DatetimeIndex) -> bool:
    """True once SETTLE_SESSIONS sessions exist past D + H on the given calendar."""
    target = pd.Timestamp(snapshot_day).normalize() + pd.Timedelta(weeks=weeks)
    after = calendar[calendar >= target]
    return len(after) > SETTLE_SESSIONS


def grade_snapshot(frame: pd.DataFrame, closes: dict[str, pd.Series], benchmark: pd.Series,
                   snapshot_day: pd.Timestamp, weeks: int, graded_on: date | None = None) -> pd.DataFrame:
    """One row per cohort for one snapshot at one horizon.

    `n` is the cohort's size on the day; `n_priced` how many could be graded.
    The gap between them is coverage lost to delistings and provider gaps, and
    it is recorded rather than hidden -- a cohort whose losers vanished would
    otherwise grade as a winner.
    """
    graded_on = graded_on or date.today()
    day = pd.Timestamp(snapshot_day).normalize()
    bench = forward_return(benchmark, day, weeks)
    rows = []
    per_cohort: dict[str, np.ndarray] = {}
    for cohort in COHORTS:
        members = cohort_members(frame, cohort)
        rets = np.array([forward_return(closes[s], day, weeks) if s in closes else float("nan")
                         for s in members], dtype=float)
        per_cohort[cohort] = rets
    uni = per_cohort["universe"]
    uni_med = float(np.nanmedian(uni)) if np.isfinite(uni).any() else float("nan")
    for cohort, rets in per_cohort.items():
        priced = rets[np.isfinite(rets)]
        med = float(np.median(priced)) if len(priced) else float("nan")
        mean = float(np.mean(priced)) if len(priced) else float("nan")
        hit = float(np.mean(priced > 0)) if len(priced) else float("nan")
        rows.append({
            "snapshot_date": day.date().isoformat(),
            "horizon_weeks": int(weeks),
            "cohort": cohort,
            "n": int(len(rets)),
            "n_priced": int(len(priced)),
            "median_return_pct": round(med, 4),
            "mean_return_pct": round(mean, 4),
            "hit_rate": round(hit, 4),
            "universe_median_pct": round(uni_med, 4),
            "excess_vs_universe_pp": round(med - uni_med, 4) if np.isfinite(med) and np.isfinite(uni_med) else float("nan"),
            "benchmark_return_pct": round(bench, 4),
            "excess_vs_benchmark_pp": round(med - bench, 4) if np.isfinite(med) and np.isfinite(bench) else float("nan"),
            "rules_version": rules_version(),
            "graded_on": graded_on.isoformat(),
        })
    return pd.DataFrame(rows, columns=TRACK_COLUMNS)


KEY = ["snapshot_date", "horizon_weeks", "cohort"]


def append_only(existing: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Add rows whose key is absent. Existing rows are returned byte-for-byte.

    A grade is a statement about what was known when it was made. Recomputing
    it later -- with revised prices, a new rule, a fixed bug -- would make the
    record describe today's opinion of the past rather than the past.
    """
    if existing is None or existing.empty:
        merged = new.copy()
    else:
        have = set(map(tuple, existing[KEY].astype(str).to_numpy()))
        fresh = new[[tuple(map(str, k)) not in have for k in new[KEY].to_numpy()]]
        merged = pd.concat([existing, fresh], ignore_index=True)
    merged["horizon_weeks"] = merged["horizon_weeks"].astype(int)
    return merged.sort_values(KEY, kind="stable").reset_index(drop=True)[TRACK_COLUMNS]
