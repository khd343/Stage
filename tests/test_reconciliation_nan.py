"""NaN == NaN is agreement in the audit's reconciliation, not a mismatch."""

import re
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "real_data_audit.py"


def test_numpy_treats_nan_as_unequal_without_the_flag():
    """The premise. If this ever changes, the fix below becomes unnecessary."""
    assert not np.isclose(np.nan, np.nan, rtol=0, atol=1e-9)
    assert np.isclose(np.nan, np.nan, rtol=0, atol=1e-9, equal_nan=True)


def test_every_reconciliation_comparison_passes_equal_nan():
    """Both implementations return NaN for the same honest reasons.

    A zero volume baseline, a degenerate base, insufficient history -- these are
    agreements that a value cannot be computed. Without equal_nan the checker
    reports a mismatch exactly when the two sides agree, which is the opposite of
    what it exists to detect.

    Three zero-volume NSE listings (BRIGHT, KALYANI, HERCULES) aborted a full
    audit this way: both sides correctly returned NaN, and the run died calling it
    a disagreement.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    # Reconciliation comparisons are the ones pinned to an absolute tolerance.
    calls = re.findall(r"np\.isclose\((?:[^()]|\([^()]*\))*\)", src, re.S)
    graded = [c for c in calls if "rtol=0" in c]
    assert graded, "no reconciliation comparisons found; has the checker moved?"
    missing = [" ".join(c.split()) for c in graded if "equal_nan" not in c]
    assert not missing, (
        "these comparisons treat NaN as a mismatch, so two implementations that "
        "agree a value is uncomputable would fail the audit:\n  "
        + "\n  ".join(m[:110] for m in missing)
    )
