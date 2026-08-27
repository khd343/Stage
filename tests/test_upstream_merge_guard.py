"""Every file this repo owns must survive a merge from upstream.

This is a fork. Upstream regenerates the same research CSVs on the same
schedule, and still runs the universe-refresh workflow that was deleted here,
so a plain `git merge upstream/main` would either conflict on data nobody wants
from the other side or — worse, on the universe file — quietly replace the
whole ticker list with NSE's.

`.gitattributes` declares `merge=ours` for those paths. The failure mode this
test exists for is a NEW owned file being added later and nobody remembering to
list it: the merge would then succeed, look clean, and overwrite it. Adding a
path to OWNED without adding it to `.gitattributes` fails here instead.

Note `merge=ours` names a driver git does not ship. Every clone must run
`git config merge.ours.driver true` once; without it the merge falls back to a
normal conflict — noisy, but never silent and never destructive. That is why
this test checks the committed declaration rather than the local config: the
declaration is the part that can be shared.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Paths this repo regenerates or authored itself, and must never take from
#: upstream. Keep in step with .gitattributes.
OWNED = (
    "data/latest_research.csv",
    "data/previous_research.csv",
    "data/breadth_history.csv",
    "data/snapshots/**",
    "data/price_panel.npz",
    "data/ind_niftytotalmarket_list.csv",
)


def _declared() -> dict[str, set[str]]:
    """Map each pattern in .gitattributes to the attributes set on it."""
    declared: dict[str, set[str]] = {}
    for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        pattern, *attrs = re.split(r"\s+", line)
        declared.setdefault(pattern, set()).update(attrs)
    return declared


def test_every_owned_path_is_declared_merge_ours() -> None:
    declared = _declared()
    missing = [p for p in OWNED if "merge=ours" not in declared.get(p, set())]
    assert not missing, (
        "these paths are regenerated here but would be taken from upstream on a "
        f"merge: {missing}. Add `<path> merge=ours` to .gitattributes."
    )


def test_the_universe_file_is_guarded() -> None:
    """Called out separately because it is the one that fails silently.

    The research CSVs are rewritten by the next audit run, so taking upstream's
    copy costs a day. The universe file is static here — nothing regenerates
    it — so upstream's 750-symbol list would simply become the universe, and
    the audit would keep passing on it.
    """
    assert "merge=ours" in _declared().get("data/ind_niftytotalmarket_list.csv", set())
