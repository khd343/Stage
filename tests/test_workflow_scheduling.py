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


def test_the_audit_is_the_only_workflow_that_writes_published_data():
    """Two workflows push; only one may touch the record.

    update_nse_universe.yml was REMOVED: the universe is now this repo's own
    ticker list, and that job downloaded the NSE index constituents and REPLACED
    the file wholesale -- a successful run, no error, and the universe silently
    gone every Friday.

    keepalive.yml also pushes, but only ever a timestamp. It exists because
    GitHub disables schedules after 60 days without COMMITS, and this repo's
    only automatic commits are the audit's -- so a stalled audit would disable
    the schedule that revives it. If it ever learns to write under data/, it
    becomes a second publisher and this ordering no longer holds.
    """
    assert set(_pushing_workflows()) == {"real_data_audit.yml", "keepalive.yml"}

    # Comments stripped first. An earlier version of this check matched the bare
    # string and failed on the workflow's own comment explaining that the AUDIT
    # writes data/ -- a scan that reads prose tests the prose, not the code.
    code = chr(10).join(line.split("#", 1)[0] for line in
                        _workflows()["keepalive.yml"].splitlines())
    assert "data/" not in code, "the keepalive must never write published data"
    assert "git add .keepalive" in code, "the keepalive must stage only its own timestamp"


def _weekly_minutes(text: str) -> set[int]:
    """Minutes-into-the-week for crons that repeat weekly (day-of-month '*')."""
    weekly = [c for c in _crons(text) if c.split()[2] == "*"]
    return set().union(*(fire_minutes(c) for c in weekly)) if weekly else set()


def _clock_slots(text: str) -> set[tuple[int, int]]:
    """(hour, minute) for EVERY cron, including day-of-month restricted ones.

    fire_minutes deliberately refuses a day-of-month cron -- expanding one into
    minutes-of-week would depend on the calendar. A monthly schedule can still
    collide with a daily one, so it is compared on the clock instead, which is
    conservative: same hour and minute is a necessary condition for any overlap.
    """
    return {(int(c.split()[1]), int(c.split()[0])) for c in _crons(text)}


def test_no_two_publishing_workflows_can_fire_together():
    """Concurrent runs race for the push; the loser is rejected non-fast-forward.

    Both pushers now rebase and retry, so a collision is survivable rather than
    fatal. This keeps them from colliding in the first place.
    """
    names = sorted(_pushing_workflows())
    texts = _pushing_workflows()
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap = _weekly_minutes(texts[first]) & _weekly_minutes(texts[second])
            assert not overlap, (
                f"{first} and {second} both fire at minute(s) {sorted(overlap)} of the week")
            clash = _clock_slots(texts[first]) & _clock_slots(texts[second])
            assert not clash, (
                f"{first} and {second} share clock slot(s) {sorted(clash)}; a monthly "
                "schedule can still land on a daily one")


def test_every_pushing_workflow_can_survive_a_moved_main():
    """A bare `git push` throws the whole run away when main moved underneath it."""
    for name, text in _pushing_workflows().items():
        assert "git pull --rebase" in text, (
            f"{name} pushes without a rebase path; a concurrent commit would discard it")


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


#: Cron day-of-week values on which NSE never trades (Sunday=0, Saturday=6).
NON_TRADING_DAYS = {0, 6}


def _trades_on_a_session_day(days: set[int]) -> bool:
    """False for a cron that only ever fires on a weekend.

    A weekend run cannot collide with a live session and has no "previous
    session" relationship to defend, so the session-hours and day-family rules
    below simply do not apply to it.
    """
    return bool(days - NON_TRADING_DAYS)


def test_no_scheduled_run_lands_inside_a_trading_session():
    """09:15-15:30 IST is the one window where a run can accomplish nothing.

    Mid-session the newest bar is incomplete and still moving. The boundary
    would refuse it and fall back a session, so this is not a correctness
    hazard -- it is a guaranteed no-op, an audit burning seven minutes to
    republish what it already had.

    This replaces a 12-hour "settling" floor written the same day and derived
    from a provider model that did not survive it: 3.4 hours proved sufficient
    on the evening of 28 Aug 2026 while 15.8 hours had not been that morning.
    Age does not predict coverage, so the schedule no longer pretends to. The
    per-run coverage threshold does that job with measurement instead.
    """
    for minute, hour, days in _slots(_workflows()["real_data_audit.yml"]):
        if not _trades_on_a_session_day(days):
            continue
        ist = (hour * 60 + minute + IST_OFFSET_MINUTES) % (24 * 60)
        assert not (NSE_OPEN_IST <= ist <= NSE_CLOSE_IST), (
            f"cron {hour:02d}:{minute:02d} UTC is {ist // 60:02d}:{ist % 60:02d} IST, "
            "inside the trading session; the newest bar is still moving"
        )


def test_every_run_day_can_actually_reach_a_session():
    """Two families, and each needs its own days.

    A run BEFORE the open publishes the previous day's session, so it needs
    Tue-Sat -- a Monday morning has no preceding session and would republish
    Friday's. A run AFTER the close publishes that same day's session, so it
    needs Mon-Fri. Getting either wrong is silent: the run succeeds and simply
    republishes what was already there.
    """
    for minute, hour, days in _slots(_workflows()["real_data_audit.yml"]):
        if not _trades_on_a_session_day(days):
            continue
        ist = (hour * 60 + minute + IST_OFFSET_MINUTES) % (24 * 60)
        label = f"cron {hour:02d}:{minute:02d} UTC ({ist // 60:02d}:{ist % 60:02d} IST)"
        if ist < NSE_OPEN_IST:
            # cron day-of-week is Sunday-based: 2..6 is Tuesday through Saturday.
            assert days == {2, 3, 4, 5, 6}, (
                f"{label} runs before the open, so it publishes the PREVIOUS "
                f"session and needs Tue-Sat; got {sorted(days)}")
        else:
            assert days == {1, 2, 3, 4, 5}, (
                f"{label} runs after the close, so it publishes THAT day's "
                f"session and needs Mon-Fri; got {sorted(days)}")


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
