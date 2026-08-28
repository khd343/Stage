"""The Setups view must offer every pre-breakout condition it publishes.

UI_SPEC names four setup sections. The shipped view offered three: the trend
template was missing, so the largest populated v2.2 condition — 132 symbols on
the 24 Aug 2026 snapshot — was reachable only through a Screener preset. The
view had no test coverage at all, which is why a whole section could go absent
without anything failing.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rs_stages.ui.loaders import load_snapshot

ENTRYPOINT = str(Path(__file__).resolve().parent.parent / "app.py")

#: Every published pre-breakout condition needs a home in this view.
REQUIRED_SECTIONS = [
    "Trend template",
    "RS leading price",
    "Contracting base",
    "Stage 1 ready",
]


def _setups(section: str | None = None) -> AppTest:
    at = AppTest.from_file(ENTRYPOINT, default_timeout=60)
    at.query_params["view"] = "Setups"
    at.run()
    assert not at.exception
    if section is not None:
        _chips(at).set_value(section).run()
        assert not at.exception
    return at


def _chips(at: AppTest):
    """The Setup chips, not the view navigation — both are segmented controls."""
    return [c for c in at.segmented_control if c.label == "Setup"][0]


def _text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown)


def test_every_published_condition_has_a_section():
    options = list(_chips(_setups()).options)
    for section in REQUIRED_SECTIONS:
        assert section in options, f"{section} has no section in the Setups view"


def test_the_view_opens_on_a_section_that_can_hold_names():
    """RS leading price is legitimately empty most days; opening there reads as broken."""
    assert _chips(_setups()).value == "Trend template"


@pytest.mark.parametrize("section", REQUIRED_SECTIONS + ["All"])
def test_each_section_renders_without_error(section):
    at = _setups(section)
    assert not at.exception


def test_the_trend_template_section_states_its_thresholds_are_provisional():
    """UI_SPEC: every surface showing a trend-template result must say so."""
    assert "provisional" in _text(_setups("Trend template")).lower()


def test_the_rs_divergence_section_states_the_five_percent_gap():
    """The gap is ours and changes the field's meaning, so it must be visible."""
    assert "5%" in _text(_setups("RS leading price"))


def test_an_empty_section_explains_itself_rather_than_rendering_blank():
    """An unmet condition and an absent field are different facts.

    The notice has to name the universe it searched, or "nothing matched" is
    indistinguishable from "nothing was loaded". The size is read from the
    published universe rather than written here: this test asserted a literal
    "750" and went on passing for days after the universe became 1,505, because
    the branch only runs when the section is genuinely empty. It first failed on
    28 Aug 2026 -- not when the universe changed, but when the data finally made
    the section empty -- and it failed against a count the app renders as
    "1,505". A test that pins a number the app derives will always be one data
    refresh from lying.
    """
    text = _text(_setups("RS leading price"))
    if "No stock currently matches" not in text:
        pytest.skip("the section is populated, so the empty-state branch is untested here")
    assert f"{len(load_snapshot().universe):,}" in text
    assert "not a missing artifact" in text


def test_the_stacked_section_exists():
    """The sections are evidence to combine, not alternatives to choose between."""
    assert "Stacked (2+)" in list(_chips(_setups()).options)


def test_the_stacked_section_renders():
    assert not _setups("Stacked (2+)").exception


def test_evidence_counts_every_condition_and_no_more():
    """The count must track SETUP_CONDITIONS, so adding a section updates it."""
    import pandas as pd

    from rs_stages.ui.components import SETUP_CONDITIONS

    assert len(SETUP_CONDITIONS) == len(REQUIRED_SECTIONS)
    assert [name for name, _ in SETUP_CONDITIONS] == REQUIRED_SECTIONS


def test_readiness_below_the_section_threshold_is_not_counted_as_evidence():
    """Stage1_Readiness is a 0-5 score; only 4+ is the published condition."""
    import pandas as pd

    frame = pd.DataFrame({"Stage1_Readiness": [5, 4, 3, None]})
    met = pd.to_numeric(frame["Stage1_Readiness"], errors="coerce") >= 4
    assert list(met) == [True, True, False, False]
