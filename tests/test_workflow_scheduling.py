"""Scheduling and publishing invariants for the workflows that write to main.

Two scheduled workflows commit to the repository: the weekday research audit
and the Friday universe refresh. They were both set to fire at 18:00 UTC, which
is not a coincidence that costs nothing — the audit runs for about three
minutes and the refresh for about twenty seconds, so every Friday on which the
constituent list changed, the refresh would land first and the audit's push
would be rejected non-fast-forward at its final step. The audit replaces the
price panel on the release *before* it commits, so that failure leaves a panel
one session ahead of the committed snapshot: exactly the drift the loader
refuses to draw through, i.e. a live terminal with no charts until someone
re-runs by hand.

These tests pin the two properties that prevent it: the schedules are ordered
and disjoint, and neither push can die on a race it could rebase past.

Parsed with the standard library only. PyYAML is not in requirements.txt and is
absent from the resolved production environment, so importing it here would
pass locally and fail in CI.
"""
from __future__ import annotations

import pathlib
import re

WORKFLOWS = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: Minutes in a week, keyed from Monday 00:00 UTC, so two schedules can be
#: compared as sets rather than by eyeballing two cron strings.
WEEK_MINUTES = 7 * 24 * 60


def _field(spec: str, low: int, high: int) -> set[int]:
    """Expand one cron field into the values it matches."""
    values: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, _, raw_step = part.partition("/")
            step = int(raw_step)
        if part == "*":
            start, end = low, high
        elif "-" in part:
            start_raw, _, end_raw = part.partition("-")
            start, end = int(start_raw), int(end_raw)
        else:
            start = end = int(part)
        values.update(range(start, end + 1, step))
    assert values <= set(range(low, high + 1)), f"out-of-range cron field {spec!r}"
    return values


def fire_minutes(cron: str) -> set[int]:
    """Minutes-into-the-week at which a five-field cron fires.

    Only day-of-week is honoured for the day, which is all these schedules use;
    a day-of-month restriction would make the answer depend on the calendar and
    is rejected rather than silently ignored.
    """
    minute, hour, dom, month, dow = cron.split()
    assert dom == "*" and month == "*", f"unsupported day-of-month/month in {cron!r}"
    # cron day-of-week is Sunday-based; the week here starts on Monday.
    days = {(day + 6) % 7 for day in _field(dow, 0, 6)}
    return {
        day * 24 * 60 + hour_value * 60 + minute_value
        for day in days
        for hour_value in _field(hour, 0, 23)
        for minute_value in _field(minute, 0, 59)
    }


def _workflows() -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(WORKFLOWS.glob("*.yml"))}


def _pushing_workflows() -> dict[str, str]:
    """Workflows that write commits back to the repository."""
    return {name: text for name, text in _workflows().items() if "git push" in text}


def _crons(text: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"-\s*cron:\s*['\"](.+?)['\"]", text)]


def test_only_the_audit_publishes():
    """A second pushing workflow would need its own place in this ordering.

    update_nse_universe.yml was REMOVED: the universe is now this repo's own
    ticker list, and that job downloaded the NSE index constituents and REPLACED
    the file wholesale -- a successful run, no error, and the universe silently
    gone every Friday.
    """
    assert set(_pushing_workflows()) == {"real_data_audit.yml"}


def test_no_two_publishing_workflows_share_a_fire_minute():
    """Concurrent runs race for the push; the loser is rejected non-fast-forward."""
    schedules = {
        name: set().union(*(fire_minutes(c) for c in _crons(text))) if _crons(text) else set()
        for name, text in _pushing_workflows().items()
    }
    names = sorted(schedules)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = schedules[first] & schedules[second]
            assert not overlap, (
                f"{first} and {second} both fire at minute(s) {sorted(overlap)} of the week; "
                "both push to main, so one push is rejected."
            )


def test_no_workflow_can_overwrite_the_universe_file():
    """The universe is this repo's own ticker list; nothing may regenerate it.

    Deleted rather than merely disabled, deliberately: a commented-out
    `- cron:` still matches the regex in _crons(), so commenting the schedule
    would leave every test here passing while GitHub ran nothing -- a green suite
    describing a workflow that does not exist.
    """
    universe = "ind_niftytotalmarket_list.csv"
    for name, text in _workflows().items():
        assert not ("curl" in text and universe in text), (
            f"{name} downloads over {universe}; the universe would be replaced")


def test_the_audit_has_a_catch_up_schedule():
    """One scheduled slot is not a schedule; it is a hope.

    GitHub's `schedule` event is best-effort. Measured on this repo: the 26 Aug
    run fired 106 minutes late, and the 27 Aug run never fired, so that session
    was never published and nothing reported a failure -- a dropped run leaves
    no trace in the Actions list at all. A second slot hours later turns a
    dropped run into a late one, and the publish step already no-ops when the
    regenerated files are unchanged, so the normal-day cost is one idle run.

    Deliberately a second CRON and not a watchdog workflow. Upstream solved the
    same problem by having a second workflow re-dispatch the audit, which needs
    a personal access token because GitHub blocks the automatic token from
    dispatching workflows. That token expires, and a safety net that dies when
    a credential lapses -- reporting it only as a red tick in a tab nobody is
    watching -- is worse than one extra idle run on a free public runner.
    """
    crons = _crons(_workflows()["real_data_audit.yml"])
    assert len(crons) >= 2, "the audit needs a catch-up slot; a dropped run is silent"
    hours = {int(cron.split()[1]) for cron in crons}
    assert len(hours) >= 2, f"catch-up must sit in a different hour, got {sorted(hours)}"


def test_scheduled_minutes_avoid_the_contended_slots():
    """GitHub's own advice: do not schedule on the busy minutes of the hour.

    The queue is deepest at :00 and the quarter-hours, because that is where
    everyone schedules. This is the one lever that costs nothing and measurably
    reduces both delay and drops.
    """
    contended = {0, 15, 30, 45}
    for name, text in _workflows().items():
        for cron in _crons(text):
            minute = int(cron.split()[0])
            assert minute not in contended, (
                f"{name} fires at :{minute:02d}, one of the most contended minutes of "
                "the hour; pick an off-beat minute to reduce delays and drops."
            )


def _slots(text: str) -> list[tuple[int, int, set[int]]]:
    """(minute, hour, weekdays) for every cron in a workflow, UTC."""
    out = []
    for cron in _crons(text):
        minute, hour, _, _, dow = cron.split()
        out.append((int(minute), int(hour), _field(dow, 0, 6)))
    return out


#: IST is UTC+5:30. NSE trades 09:15-15:30 IST.
IST_OFFSET_MINUTES = 330
NSE_OPEN_IST = 9 * 60 + 15
NSE_CLOSE_IST = 15 * 60 + 30


def test_every_audit_run_is_pre_market_in_ist():
    """A snapshot published after the open is no longer a pre-market decision.

    The audit moved to the morning AFTER each session so the price provider has
    time to post the close (it had not done so even ten hours later: the 26 Aug
    run saw a 26 Aug bar for 2 symbols of 1,505). Moving later trades against
    the open, and this is the wall it must not cross -- the locked spec's whole
    boundary rule is that a decision is made before the session it is for.
    """
    for minute, hour, _ in _slots(_workflows()["real_data_audit.yml"]):
        ist = (hour * 60 + minute + IST_OFFSET_MINUTES) % (24 * 60)
        assert ist < NSE_OPEN_IST, (
            f"cron {hour:02d}:{minute:02d} UTC is {ist // 60:02d}:{ist % 60:02d} IST, "
            "at or after the 09:15 open; the snapshot would not be pre-market"
        )


def test_every_audit_run_gives_the_provider_time_to_settle():
    """The failure this schedule exists to fix was asking Yahoo too early.

    Eight hours was not enough three sessions running. Twelve is the floor that
    keeps a future edit from quietly walking the schedule back toward the close
    and reintroducing a bug whose only symptom is a snapshot one session stale.
    """
    minimum_hours = 12
    for minute, hour, _ in _slots(_workflows()["real_data_audit.yml"]):
        ist = (hour * 60 + minute + IST_OFFSET_MINUTES) % (24 * 60)
        # The run is the morning after, so settling spans the prior close to
        # midnight, then midnight to the run.
        settled = (24 * 60 - NSE_CLOSE_IST) + ist
        assert settled >= minimum_hours * 60, (
            f"cron {hour:02d}:{minute:02d} UTC leaves only {settled / 60:.1f}h after the "
            f"previous close; at least {minimum_hours}h is required"
        )


def test_the_audit_days_are_shifted_one_past_the_sessions_they_publish():
    """Mon-Fri sessions, read the next morning, need Tue-Sat runs.

    Getting this wrong is silent in both directions: a Monday run has no
    preceding session to fetch and republishes Friday's, and dropping Saturday
    loses every Friday close.
    """
    days = set().union(*(dow for _, _, dow in _slots(_workflows()["real_data_audit.yml"])))
    # cron day-of-week is Sunday-based: 2..6 is Tuesday through Saturday.
    assert days == {2, 3, 4, 5, 6}, f"expected Tue-Sat runs, got cron days {sorted(days)}"


def test_the_audit_cannot_run_beside_itself():
    """Two slots means a late primary can still be running when the catch-up fires.

    Both push the same branch, so a race discards one entire run. The queue must
    hold the second rather than cancel either: a half-finished audit publishes
    nothing, so there is no partial result worth preferring over a completed one.
    """
    text = _workflows()["real_data_audit.yml"]
    assert "concurrency:" in text, "two scheduled slots can overlap; serialise them"
    assert "cancel-in-progress: false" in text, (
        "cancelling a running audit to start a newer one throws away a complete "
        "result for an unfinished one"
    )
