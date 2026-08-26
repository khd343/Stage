"""The hand-written checks must skip NaN exactly as pandas does."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "real_data_audit", Path(__file__).resolve().parents[1] / "scripts" / "real_data_audit.py"
)
_AUDIT = importlib.util.module_from_spec(_SPEC)
sys.modules["real_data_audit"] = _AUDIT
_SPEC.loader.exec_module(_AUDIT)

from rs_stages import quant


def test_python_builtins_mishandle_nan_which_is_why_finite_exists():
    """The premise, stated so the fix cannot look arbitrary later.

    max() is not merely wrong on NaN, it is ORDER-DEPENDENT wrong: every NaN
    comparison is False, so the answer depends on where the NaN sits.
    """
    assert np.isnan(max([float("nan"), 1.0, 3.0]))
    assert max([1.0, float("nan"), 3.0]) == 3.0
    assert np.isnan(sum([1.0, float("nan")]))
    assert pd.Series([1.0, np.nan, 3.0]).max() == 3.0


def test_finite_filters_nan_and_keeps_order():
    assert _AUDIT._finite([1.0, float("nan"), 3.0]) == [1.0, 3.0]
    assert _AUDIT._finite([float("nan")] * 3) == []
    assert _AUDIT._finite([2.0, 1.0]) == [2.0, 1.0]


def _vol(values):
    idx = pd.date_range("2026-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_dryup_matches_production_when_the_window_holds_nan():
    """A bulk yfinance response pads a non-trading symbol with NaN rows.

    ARTEMISMED and ZODIAC have 247 sessions against the market's 251, so four of
    their rows are NaN. Production skips them; these checks used to propagate
    them, and the audit aborted on two symbols out of 1,505.
    """
    values = [1000.0] * 60 + [500.0] * 10
    values[5] = float("nan")
    values[-3] = float("nan")
    v = _vol(values)
    t = v.index[-1]

    mine = _AUDIT.independent_volume_dryup(v, t)
    theirs = quant.volume_dryup(v, t)
    assert np.isclose(mine, theirs, rtol=0, atol=1e-12), f"{mine} != {theirs}"


def test_a_window_that_is_entirely_nan_raises_rather_than_returning_a_number():
    """The caller skips on ValueError. Returning a number from no data would
    reconcile a fabricated value against production and report a false match."""
    v = _vol([float("nan")] * 10 + [1000.0] * 60)
    with pytest.raises(ValueError):
        _AUDIT.independent_volume_dryup(v, v.index[9])
