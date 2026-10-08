"""The recipe fingerprint: a grade must name the engine that produced it.

`rules_version()` was documented as "a fingerprint of everything that defines a
grade". It was not. It hashed the grading APPARATUS -- cohort descriptions,
horizons, required columns, the settle rule -- and nothing that decides WHO
LANDS IN A COHORT. `action_for()`, the RS blend weights and the stage
thresholds all sat outside it.

So an edit to the recipe mid-record would leave every grade before and after
carrying the SAME hash, and the record would pool two different systems with
nothing in the data to show it. The guard looked like it covered this, which is
why it survived: existence is not reachability.

Found 2026-10-08, with `track_record.csv` still header-only -- the cheapest
moment there will ever be.

WHAT IS AND IS NOT NORMALISED, and why the line sits there:

  * docstrings and comments are stripped -- prose churn would train a reader to
    ignore the hash, and the whole value is that a change is worth looking at.
  * IDENTIFIERS ARE KEPT. Stripping them would make the hash blind to swapping
    two variables (`rs >= 80 and vol >= 1.5` vs `vol >= 80 and rs >= 1.5`),
    which is a behaviour change. A rename therefore churns the hash; that cost
    is deliberate, and pinned below so it cannot be "fixed" without reading
    this.

Over-detection is the safe direction. A spurious change costs one glance at a
diff; a missed one silently voids the record.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rs_stages import track_record as tr

BASE = """
THRESHOLD = 70

def decide(rs, vol):
    if rs >= THRESHOLD and vol >= 1.5:
        return "BUY"
    return "WAIT"
"""


# --- the normaliser -------------------------------------------------------------

def test_the_fingerprint_is_deterministic():
    assert tr._source_fingerprint(BASE) == tr._source_fingerprint(BASE)


def test_a_changed_threshold_moves_the_fingerprint():
    """The case the whole file exists for."""
    assert tr._source_fingerprint(BASE.replace("= 70", "= 80")) != tr._source_fingerprint(BASE)


def test_a_changed_comparison_moves_the_fingerprint():
    assert tr._source_fingerprint(BASE.replace(">= 1.5", "> 1.5")) != tr._source_fingerprint(BASE)


def test_a_changed_output_label_moves_the_fingerprint():
    """Labels are data the cohort rules match on, not prose."""
    assert tr._source_fingerprint(BASE.replace('"BUY"', '"BUY*"')) != tr._source_fingerprint(BASE)


@pytest.mark.parametrize("prose", [
    '"""A docstring that says nothing about behaviour."""\n' + BASE,
    BASE.replace("def decide(rs, vol):", 'def decide(rs, vol):\n    """Pick an action."""'),
    BASE.replace('    return "WAIT"', '    # a comment\n    return "WAIT"'),
    BASE.replace("\n\ndef decide", "\n\n\n\ndef decide"),
    BASE + "\n\n",
])
def test_prose_and_formatting_do_not_move_the_fingerprint(prose):
    assert tr._source_fingerprint(prose) == tr._source_fingerprint(BASE)


def test_swapping_two_variables_moves_the_fingerprint():
    """THE REASON IDENTIFIERS ARE KEPT. Normalising names away would make these
    two functions hash alike while one of them is wrong."""
    swapped = BASE.replace("if rs >= THRESHOLD and vol >= 1.5:",
                           "if vol >= THRESHOLD and rs >= 1.5:")
    assert tr._source_fingerprint(swapped) != tr._source_fingerprint(BASE)


def test_a_rename_moves_the_fingerprint_and_that_is_the_accepted_cost():
    """Documented, not accidental. See the module docstring before loosening."""
    assert tr._source_fingerprint(BASE.replace("vol", "volume")) != tr._source_fingerprint(BASE)


def test_unparseable_source_is_loud():
    """A recipe file that does not parse must never fingerprint as empty."""
    with pytest.raises(SyntaxError):
        tr._source_fingerprint("def broken(:\n")


# --- the real recipe ------------------------------------------------------------

def test_the_recipe_version_is_a_stable_short_hash():
    got = tr.recipe_version()
    assert got == tr.recipe_version()
    assert len(got) == 12 and all(c in "0123456789abcdef" for c in got)


def test_the_recipe_reads_every_named_source_and_none_is_empty():
    """Teeth. A list that silently resolved to nothing would hash a constant
    and pass every other test in this file."""
    assert len(tr.RECIPE_SOURCES) >= 3
    for name in tr.RECIPE_SOURCES:
        text = tr._recipe_source(name)
        assert len(text) > 500, f"{name} came back suspiciously small"


def test_a_missing_recipe_file_is_loud_not_skipped():
    """The silent-skip class: a renamed module must break the build, not drop
    quietly out of the fingerprint.

    MATCHED ON THE MESSAGE, and that is the point. `Path.read_text` raises
    FileNotFoundError on its own, so asserting the exception type alone passes
    with the explicit guard deleted -- a test with no teeth. What the guard
    buys is a reader who is told WHY the build broke and where to look, so
    that is what is pinned.
    """
    with pytest.raises(FileNotFoundError, match="RECIPE_SOURCES"):
        tr._recipe_source("no_such_module.py")


def test_every_file_that_decides_a_cohort_is_fingerprinted():
    """THE GAP-CLOSER, and the reason this is structural rather than a list.

    Finds every module that WRITES a column a pre-registered cohort reads, and
    requires it to be inside the fingerprint. Move `Action` to a new module and
    this fails instead of the record going quietly stale.
    """
    import ast
    from pathlib import Path

    identity = {"Symbol", "Date", "Close"}
    decided = set(tr.REQUIRED_COLUMNS) - identity
    assert decided, "no decision columns to check -- REQUIRED_COLUMNS changed shape"

    pkg = Path(tr.__file__).resolve().parent
    writes: dict[str, set[str]] = {}
    for path in sorted(pkg.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for t in targets:
                if (isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                        and t.slice.value in decided):
                    writes.setdefault(t.slice.value, set()).add(path.name)

    missing = sorted(decided - set(writes))
    assert not missing, f"no module writes {missing} -- the scan found nothing, so it proves nothing"

    for column, files in sorted(writes.items()):
        outside = sorted(files - set(tr.RECIPE_SOURCES))
        assert not outside, (
            f"{column} is decided in {outside}, which is not fingerprinted; "
            f"add it to RECIPE_SOURCES or the forward record cannot see it change"
        )


# --- how it reaches the record --------------------------------------------------

def test_the_rules_version_moves_when_the_recipe_moves(monkeypatch):
    """The defect, stated as a test."""
    before = tr.rules_version()
    monkeypatch.setattr(tr, "recipe_version", lambda: "deadbeefcafe")
    assert tr.rules_version() != before


def test_the_rules_version_still_moves_when_a_cohort_moves(monkeypatch):
    """The old guarantee must survive the new one."""
    before = tr.rules_version()
    monkeypatch.setitem(tr.COHORTS, "universe", "redefined")
    assert tr.rules_version() != before


def test_the_recipe_version_is_its_own_column():
    """Two hashes, because the two have different change policies: a cohort
    edit breaks the pre-registration, a recipe edit is a new engine version.
    One combined hash cannot tell a reader which of those happened."""
    assert "recipe_version" in tr.TRACK_COLUMNS
    assert tr.TRACK_COLUMNS.index("recipe_version") == tr.TRACK_COLUMNS.index("rules_version") + 1


def test_a_graded_row_carries_both_hashes():
    frame = pd.DataFrame([
        {"Symbol": "A", "Date": "2026-09-11", "Close": 100.0, "Action": "BUY★",
         "Stage": "Stage 2 — Advancing", "RS_Score": 90.0,
         "Trend_Template_Pass": True, "Breakout_Confirmed": False},
    ])
    closes = {"A": pd.Series([100.0, 110.0],
                             index=pd.to_datetime(["2026-09-11", "2026-10-09"]))}
    bench = pd.Series([200.0, 210.0], index=pd.to_datetime(["2026-09-11", "2026-10-09"]))
    out = tr.grade_snapshot(frame, closes, bench, pd.Timestamp("2026-09-11"),
                            4, date(2026, 10, 14))
    assert not out.empty
    assert (out.recipe_version == tr.recipe_version()).all()
    assert (out.rules_version == tr.rules_version()).all()

def test_growing_the_schema_never_destroys_existing_grades():
    """The path this very change walks: TRACK_COLUMNS gained a column while a
    record already existed. Old rows must keep their numbers and carry a BLANK
    new hash -- they genuinely had none -- and must never be dropped or
    recomputed. Unreachable today (the record is empty) but reachable the next
    time a column is added, which is when it would silently corrupt history.
    """
    old_schema = [c for c in tr.TRACK_COLUMNS if c != "recipe_version"]
    existing = pd.DataFrame([{**{c: None for c in old_schema},
                              "snapshot_date": "2026-09-11", "horizon_weeks": 4,
                              "cohort": "universe", "median_return_pct": 12.0,
                              "rules_version": "old", "graded_on": "2026-10-20"}])
    fresh = pd.DataFrame([{**{c: None for c in tr.TRACK_COLUMNS},
                           "snapshot_date": "2026-09-15", "horizon_weeks": 4,
                           "cohort": "universe", "median_return_pct": 7.0,
                           "rules_version": "new", "recipe_version": "abc123abc123",
                           "graded_on": "2026-11-02"}])
    out = tr.append_only(existing, fresh)

    assert list(out.columns) == tr.TRACK_COLUMNS
    assert len(out) == 2, "a schema change must not drop a row"
    kept = out[out.snapshot_date == "2026-09-11"].iloc[0]
    assert kept.median_return_pct == 12.0 and kept.rules_version == "old"
    assert pd.isna(kept.recipe_version), "an absent hash is blank, never invented"
    added = out[out.snapshot_date == "2026-09-15"].iloc[0]
    assert added.recipe_version == "abc123abc123"
