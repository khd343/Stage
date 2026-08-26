"""The independent contraction check must not crash on a flat base."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "real_data_audit", Path(__file__).resolve().parents[1] / "scripts" / "real_data_audit.py"
)
_AUDIT = importlib.util.module_from_spec(_SPEC)
sys.modules["real_data_audit"] = _AUDIT
_SPEC.loader.exec_module(_AUDIT)

from rs_stages.quant import contraction_ratio


def _series(values, idx):
    return pd.Series(values, index=idx, dtype=float)


def test_a_flat_first_block_raises_valueerror_not_zerodivision():
    """A stock that did not move for ten sessions must skip, not abort the audit.

    The caller wraps this in `except (ValueError, KeyError)`. A ZeroDivisionError
    escapes that and kills the whole run -- which is exactly what happened: one
    flat microcap ended a three-hour audit at the reconciliation step, after every
    download had already completed.
    """
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    # `base` is the LAST 50 rows, so block 0 is positions 10-19. Those are the
    # ten flat sessions; everything else moves normally.
    high = [12.0] * 10 + [10.0] * 10 + [12.0] * 40
    low = [11.0] * 10 + [10.0] * 10 + [11.0] * 40
    close = [11.5] * 10 + [10.0] * 10 + [11.5] * 40

    with pytest.raises(ValueError, match="degenerate base"):
        _AUDIT.independent_contraction(
            _series(high, idx), _series(low, idx), _series(close, idx), idx[-1]
        )


def test_production_returns_nan_for_the_same_case():
    """Production already guards it; the independent copy did not.

    Pinning both sides together: if production ever starts dividing unguarded,
    this test fails alongside the one above rather than leaving the two
    implementations to drift apart silently.
    """
    assert pd.isna(contraction_ratio([0.0, 1.0, 2.0]))
    assert contraction_ratio([2.0, 1.0]) == 0.5


def test_a_normal_base_still_reconciles():
    """The guard must not swallow the ordinary case it sits beside."""
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    # Ranges narrow block by block but never reach zero -- an ordinary tightening
    # base, which is the case the guard must not swallow.
    width = [5.0] * 10 + [4.0] * 10 + [3.0] * 10 + [2.0] * 10 + [1.5] * 10 + [1.0] * 10
    high = [15.0 + w / 2 for w in width]
    low = [15.0 - w / 2 for w in width]
    close = [15.0] * 60
    count, ratio = _AUDIT.independent_contraction(
        _series(high, idx), _series(low, idx), _series(close, idx), idx[-1]
    )
    assert isinstance(count, int)
    assert ratio > 0
