"""The banner that stops a demerger being read as a collapse.

On 17 Sep 2026 four names read "Stage 4 - Declining, SELL" with relative
strength in the bottom decile, and not one of them had fallen. The engine is
reporting the prices it was given, so the row is not wrong so much as
unreadable without this sentence. The banner supplies the sentence.
"""
from __future__ import annotations

import pandas as pd

from rs_stages import corporate_actions as ca
from rs_stages.ui import components as ui


def test_a_clean_row_gets_no_banner():
    """Silence is the normal case. A banner that appears on every row would be
    read as decoration and then ignored on the day it matters."""
    assert ui.corporate_action_notice(False, "", "") == ""
    assert ui.corporate_action_notice(None, "", "") == ""


def test_a_flagged_row_names_the_date_and_what_it_looks_like():
    html = ui.corporate_action_notice(True, "2026-04-30", ca.classify_ratio(0.351)[0])
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


# --- the table, where names are actually picked --------------------------------

def _rows() -> pd.DataFrame:
    """MBECL is the real case: RS 99, Stage 2, and a +242% session that no
    circuit limit allows. Scanning the table for leaders finds it first."""
    return pd.DataFrame([
        {"Symbol": "CLEAN", "Industry": "Steel", "Stage": "Stage 2 — Advancing",
         "RS_Score": 98.0, "Action": "BUY", "Corporate_Action": False},
        {"Symbol": "MBECL", "Industry": "Steel", "Stage": "Stage 2 — Advancing",
         "RS_Score": 99.0, "Action": "BUY", "Corporate_Action": True},
    ])


def test_a_flagged_leader_is_marked_in_the_table():
    """The banner on the detail page is too late: the pick happens here, and
    the dangerous direction is an UPWARD phantom that manufactures strength."""
    html = ui.screener_table(_rows(), columns=["symbol", "rs", "stage", "action"])
    before, _, after = html.partition("MBECL")
    assert ui.CORPORATE_ACTION_MARK in after[:400], "the mark rides with the flagged symbol"
    assert ui.CORPORATE_ACTION_MARK not in before, "and not with the clean one"


def test_the_table_is_unmarked_when_nothing_is_flagged():
    frame = _rows()
    frame["Corporate_Action"] = False
    html = ui.screener_table(frame, columns=["symbol", "rs", "stage", "action"])
    assert ui.CORPORATE_ACTION_MARK not in html


def test_a_frame_without_the_column_still_renders():
    """The app reads published snapshots it did not write, including ones
    frozen before this column existed."""
    frame = _rows().drop(columns=["Corporate_Action"])
    html = ui.screener_table(frame, columns=["symbol", "rs"])
    assert "MBECL" in html and ui.CORPORATE_ACTION_MARK not in html


def test_the_mark_explains_itself_on_hover():
    html = ui.screener_table(_rows(), columns=["symbol"])
    assert "title=" in html
    assert "corporate action" in html.lower()


def test_the_string_false_from_a_csv_round_trip_is_not_a_flag():
    """The app reads published CSVs, where a boolean column can arrive as the
    strings "True"/"False". Treating any non-empty value as truthy would mark
    every row in the universe and make the mark meaningless."""
    frame = _rows()
    frame["Corporate_Action"] = ["False", "True"]
    html = ui.screener_table(frame, columns=["symbol"])
    before, _, after = html.partition("MBECL")
    assert ui.CORPORATE_ACTION_MARK in after[:400]
    assert ui.CORPORATE_ACTION_MARK not in before, "the string 'False' is not a flag"
    assert ui.corporate_action_notice("False", "", "") == ""
    assert ui.corporate_action_notice("True", "2026-09-01", "") != ""


# --- the sentence itself, as a reader meets it ----------------------------------

def test_the_warning_never_claims_a_decline():
    """Found by looking at the live page. MBECL's phantom session was +242%,
    and the banner told the reader the numbers were 'not a decline in the
    business' -- true, irrelevant, and pointing the wrong way. The distortion
    runs in both directions and the wording has to work for both."""
    html = ui.corporate_action_notice(True, "2026-09-01", "1:2 split or 1:1 bonus").lower()
    assert "decline" not in html
    assert "fall" not in html


def test_the_label_reads_as_a_noun_phrase_in_the_sentence():
    """Also found on the live page: 'It resembles a unmatched - possible
    demerger or spin-off.' A label that cannot follow an article is a label
    the sentence cannot use."""
    html = ui.corporate_action_notice(True, "2026-09-01",
                                      ca.classify_ratio(0.351)[0]).lower()
    assert "a unmatched" not in html and "an unmatched" not in html
    assert "demerger" in html


def test_the_classification_is_never_embedded_in_a_sentence():
    """THE ROOT CAUSE, not the symptom. The label is a data string; a sentence
    template that puts it after an article is an implicit grammar contract
    between the engine and the copy, and it broke the first time a label was
    not a noun phrase. It also breaks for labels already frozen in published
    snapshots, which no code change can reach. So the classification gets its
    own line and the prose never bends around it."""
    for label in ("1:2 split or 1:1 bonus", "5:1 reverse split",
                  ca.classify_ratio(0.351)[0],
                  "unmatched - possible demerger or spin-off"):  # a retired label, still in the archive
        html = ui.corporate_action_notice(True, "2026-09-01", label)
        assert label in html, "the classification is still reported"
        assert f"a {label}" not in html, f"no article may precede {label!r}"
        assert "resembles" not in html.lower()
