"""Where depth and texture appear on screen.

The number only helps at the moment of choosing, so it has to reach two
places: the table, where a name is picked out of hundreds, and the detail
page, where the pick is examined. On the detail page it sits directly under
the return for the SAME window, because a return and what it cost are one
fact read together, and separating them invites reading the gain alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rs_stages.ui import components as ui
from rs_stages.ui import theme


def _rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"Symbol": "SMOOTH", "Industry": "Steel", "Stage": "Stage 2 — Advancing",
         "RS_Score": 95.0, "Action": "BUY", "R3M": 0.12, "MaxDD_12M": -0.09},
        {"Symbol": "VIOLENT", "Industry": "Steel", "Stage": "Stage 2 — Advancing",
         "RS_Score": 94.0, "Action": "BUY", "R3M": 0.12, "MaxDD_12M": -0.35},
    ])


# --- the table ------------------------------------------------------------------

def test_the_table_carries_a_drawdown_column():
    assert "maxdd" in ui.SCREENER_COLUMNS
    label, align, _ = ui.SCREENER_COLUMNS["maxdd"]
    assert align == "right", "a number reads against the right edge"
    assert "12M" in label and "DD" in label.upper(), "the window must be on the label"


def test_the_two_advances_are_told_apart_on_screen():
    """Same relative strength, same three-month return, very different holding."""
    html = ui.screener_table(_rows(), columns=["symbol", "rs", "maxdd"])
    assert "-9.0%" in html and "-35.0%" in html


def test_a_row_without_the_column_shows_a_dash_not_a_zero():
    """Snapshots published before this existed, and young names, have no value.
    A zero would read as 'never fell', which is the opposite of unknown."""
    frame = _rows().drop(columns=["MaxDD_12M"])
    html = ui.screener_table(frame, columns=["symbol", "maxdd"])
    assert theme.DASH in html
    assert "0.0%" not in html


def test_a_blank_drawdown_shows_a_dash_not_a_zero():
    frame = _rows()
    frame["MaxDD_12M"] = [np.nan, -0.35]
    html = ui.screener_table(frame, columns=["symbol", "maxdd"])
    assert theme.DASH in html and "-35.0%" in html


def test_a_flat_advance_reads_zero_and_that_is_real():
    """Zero drawdown is a genuine measurement, distinct from unknown."""
    frame = _rows()
    frame["MaxDD_12M"] = [0.0, -0.35]
    html = ui.screener_table(frame, columns=["symbol", "maxdd"])
    assert "0.0%" in html


# --- the pairing ------------------------------------------------------------------

def test_the_detail_pairs_each_return_with_its_own_window():
    row = {"R3M": 0.12, "R6M": 0.28, "R9M": 0.31, "R12M": 0.32,
           "MaxDD_3M": -0.09, "MaxDD_6M": -0.17, "MaxDD_9M": -0.17, "MaxDD_12M": -0.17,
           "Up_Days_Pct_6M": 52.4}
    html = ui.return_and_cost(row)
    for window in ("3M", "6M", "9M", "12M"):
        assert f">{window}<" in html, f"{window} column missing"
    assert "+12.0%" in html and "-9.0%" in html
    assert "+32.0%" in html and "-17.0%" in html


def test_the_pairing_says_what_the_second_number_is():
    """Two stacked percentages with no caption is a guessing game."""
    html = ui.return_and_cost({"R3M": 0.12, "MaxDD_3M": -0.09}).lower()
    assert "worst" in html or "fall" in html or "drawdown" in html


def test_the_up_day_share_is_stated_as_a_sentence_not_a_bare_number():
    """A labelled 52.4 means nothing. A sentence needs no legend."""
    html = ui.return_and_cost({"R3M": 0.12, "MaxDD_3M": -0.09, "Up_Days_Pct_6M": 52.4})
    assert "52.4%" in html
    assert "session" in html.lower()


def test_the_share_is_omitted_rather_than_guessed_when_absent():
    html = ui.return_and_cost({"R3M": 0.12, "MaxDD_3M": -0.09})
    assert "session" not in html.lower()
    html_nan = ui.return_and_cost({"R3M": 0.12, "Up_Days_Pct_6M": float("nan")})
    assert "session" not in html_nan.lower()


def test_a_row_with_no_history_at_all_still_renders():
    html = ui.return_and_cost({})
    assert theme.DASH in html
