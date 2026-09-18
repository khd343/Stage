"""The banner that stops a demerger being read as a collapse.

On 17 Sep 2026 four names read "Stage 4 - Declining, SELL" with relative
strength in the bottom decile, and not one of them had fallen. The engine is
reporting the prices it was given, so the row is not wrong so much as
unreadable without this sentence. The banner supplies the sentence.
"""
from __future__ import annotations

import pandas as pd

from rs_stages.ui import components as ui


def test_a_clean_row_gets_no_banner():
    """Silence is the normal case. A banner that appears on every row would be
    read as decoration and then ignored on the day it matters."""
    assert ui.corporate_action_notice(False, "", "") == ""
    assert ui.corporate_action_notice(None, "", "") == ""


def test_a_flagged_row_names_the_date_and_what_it_looks_like():
    html = ui.corporate_action_notice(True, "2026-04-30", "unmatched - possible demerger or spin-off")
    assert "2026-04-30" in html
    assert "demerger" in html


def test_the_banner_says_the_fall_is_not_a_fall():
    """The whole point. A reader who takes nothing else from it must take
    this: the stage, the returns and the 52-week high are not to be trusted."""
    html = ui.corporate_action_notice(True, "2026-04-30", "1:2 split or 1:1 bonus").lower()
    assert "not a price" in html or "not a decline" in html
    assert "52-week" in html


def test_a_flag_with_no_date_still_warns():
    """The flag is the finding; the date is context. Losing the date must not
    silently lose the warning."""
    assert ui.corporate_action_notice(True, "", "") != ""


def test_the_banner_escapes_what_it_is_given():
    html = ui.corporate_action_notice(True, "<script>x</script>", "<b>y</b>")
    assert "<script>" not in html and "<b>y</b>" not in html


def test_a_missing_flag_reads_as_clean():
    """Snapshots frozen before this shipped have no such column, and the app
    reads published CSVs it did not write."""
    assert ui.corporate_action_notice(pd.NA, "", "") == ""
    assert ui.corporate_action_notice(float("nan"), "", "") == ""
