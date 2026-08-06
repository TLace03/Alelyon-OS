"""Which sessions are waiting on the owner, and which are still working.

The problem this exists for, in the owner's words: *"I have 18 sessions running
in unison. Having to keep an eye on all of the sessions that need me to give
them input is pretty rough."* Eighteen terminals, and the only way to find the
one that stopped to ask a question is to read all eighteen.

Every other fleet reader answers a question about the **work**: which areas are
contested, which branches are stacked, what the queue holds. This one answers a
question about the **owner's time**, and it is the only reading in the fleet
surface whose subject is a person.

Why this could not be said before, and what changed
---------------------------------------------------
`fleet_chat` declined to render presence, and gave the reason::

    an agent session that is thinking, or waiting on a tool, has no outstanding
    edits and is not idle. `sessions_live` is offered instead and named for what
    it measures.

That is exactly right about the evidence it was looking at. The **mesh** sees
uncommitted edits, and a session composing a sentence, a session blocked on a
permission prompt, and a session that exited an hour ago all have none. No
threshold over that signal separates them, and a green dot drawn from it would
have been an invention.

This module does not overturn that finding — it reads a different record. The
harness writes every turn of every session to a transcript, and the **shape of
the last turn** distinguishes the three cases the mesh cannot:

* an assistant turn that called **no tool** ended the turn. The model is not
  running, nothing is pending, and the next thing that happens in that window is
  something the owner types.
* an assistant turn that **called a tool** is mid-work: the harness runs the
  tool and feeds the result back.
* an assistant turn that called `AskUserQuestion` or `ExitPlanMode` is blocked on
  the owner *explicitly*, and says so in a field the model cannot fake after the
  fact.

`session_activity` has parsed all of this since it was written. It keeps the last
`RECENT_TURNS` turns with their tool names on `SessionRun.recent`, and its own
docstring already claims the property this depends on — *"that a turn happened,
when, on which model, with which tool names… the harness wrote every one of
those; the model cannot edit them after the fact."* The reading was never taken.
This module is that reading and nothing else: it observes nothing new, and every
field it returns is composed from `session_activity` and `session_records`.

One detail in `session_activity._Accumulator.absorb` is load-bearing and worth
naming, because a later refactor could remove it without any test noticing:
a `user` record carrying a tool *result* is deliberately not appended to
`recent`. That is what makes "the last entry in `recent`" mean "the last thing
the model did" rather than "the last thing that happened". If tool results ever
start being appended, `WAITING` silently becomes unreachable — so
`test_a_tool_result_does_not_end_a_turn` pins it.

What it refuses to say
----------------------
**A pending tool call and a permission prompt are the same record.** When the
harness stops to ask "allow this command?", it writes nothing — the transcript
holds an issued tool call and no result, which is byte-for-byte what a command
that is still running looks like. So `STALLED` is reported as *"a tool call has
been outstanding for N minutes"* and names all three things that can produce it
(a slow command, a permission prompt, a dead session), rather than guessing at
one. On this repository the distinction is not academic: the runtime suite runs
for twenty minutes, and a reader who saw that as "blocked" would go and
interrupt a healthy run.

**`WAITING` is not proof that a person is needed.** It is proof that nothing is
running. A session that finished its work and stopped looks precisely like one
that stopped to ask — which is the honest answer, because in both cases the next
move is the owner's and in neither case is the machine doing anything.

**A crashed session is `WAITING` or `STALLED` too.** `session_records` already
says why: *"a crashed session leaves a transcript that stopped being written
exactly like an idle one"*. Age is the only discriminator offered, and it is
offered as age rather than as a verdict.

**Only Claude Code writes these transcripts.** Codex, Copilot, Cursor and
Antigravity sessions are absent from this board rather than reported as quiet —
`session_records.LIMITS` carries the same caveat for the same reason.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from alelyon.runtime.common.session_activity import (
    ActivityIndex, SessionRun, redact,
)

ATTENTION_SCHEMA = "alelyon.session-attention/0.1"

#: The turn ended with no tool call. Nothing is running.
WAITING = "waiting"
#: The turn ended on a tool whose whole purpose is to ask the owner something.
ASKING = "asking"
#: A tool call is outstanding and recent.
WORKING = "working"
#: A tool call is outstanding and has been for a while. Three causes, and this
#: cannot tell them apart — see the module docstring.
STALLED = "stalled"
#: Nothing has been written for a long time.
DORMANT = "dormant"
#: No timestamp could be read. "Cannot tell" is not "idle".
UNKNOWN = "unknown"

#: Tools that exist to put a question to the owner and wait. A turn ending on
#: one of these is blocked on a person by construction, not by inference.
ASKING_TOOLS = frozenset({"AskUserQuestion", "ExitPlanMode"})

#: How long a tool call may be outstanding before it is reported as `STALLED`.
#: Deliberately generous: `tests/runtime` runs for twenty minutes on this
#: repository, and calling that "stuck" would send a reader to interrupt a
#: healthy run. Raise it, do not lower it, when in doubt.
DEFAULT_STALL_MINUTES = 12.0

#: Quiet for longer than this and a session is reported as dormant rather than
#: as waiting, so an afternoon's finished windows do not crowd out the one that
#: asked a question four minutes ago.
DEFAULT_DORMANT_MINUTES = 180.0

#: Enough windows for a fleet this size, and the cap is always reported.
DEFAULT_MAX_SESSIONS = 60

LIMITS: tuple[str, ...] = (
    "A pending tool call and a permission prompt are the same record. The "
    "harness writes nothing when it stops to ask 'allow this command?', so a "
    "command that is still running and a prompt waiting for a click are "
    "indistinguishable here. STALLED names all three causes -- slow command, "
    "permission prompt, dead session -- and picks none of them.",
    "WAITING means nothing is running, not that a person is required. A "
    "session that finished its work and stopped is reported exactly like one "
    "that stopped to ask a question, because in both cases the next move is "
    "the owner's.",
    "A crashed session leaves a transcript that stopped being written exactly "
    "like an idle one. Age is the only discriminator offered, and it is "
    "offered as an age and never as a verdict.",
    "Only Claude Code writes the transcripts this reads. Codex, Copilot, "
    "Cursor and Antigravity sessions are ABSENT from this board rather than "
    "reported as quiet, so 'nothing else is running' is never implied.",
    "The state is read from the last turn the transcript records. A session "
    "whose transcript is being written as this is read can be reported one "
    "turn behind, which shows up as a state that is stale by seconds.",
    "Text quoted from a turn is the model's own words. It is QUOTED and never "
    "checked -- a session can describe itself as blocked while running, and "
    "the state beside the quote is the derived answer, not the prose.",
)


@dataclass(frozen=True)
class Attention:
    """One session, and whether it is waiting on the owner."""

    session_id: str
    state: str
    #: Minutes since the transcript was last written, or None if unreadable.
    idle_minutes: float | None
    cwd: str = ""
    branch: str = ""
    model: str = ""
    #: The tool the last turn called, empty when it called none. The whole
    #: classification turns on this field.
    last_tool: str = ""
    #: What that call named -- a path, a pattern, a redacted command.
    last_target: str = ""
    #: The model's own last words, truncated. QUOTED, never checked.
    summary: str = ""
    turns: int = 0
    #: Fleet members this session still has running, which is a reason not to
    #: interrupt it even when it looks quiet.
    running_agents: int = 0
    #: The rule that produced this state, written so a reader can disagree.
    evidence: str = ""

    @property
    def needs_owner(self) -> bool:
        """Whether the next move in this window is the owner's.

        `STALLED` is excluded on purpose. It *may* be a permission prompt --
        which would need the owner -- and it may equally be a twenty-minute
        test run, and rounding an ambiguity up into a demand for attention is
        how a board like this stops being read.
        """
        return self.state in (ASKING, WAITING)

    @property
    def short_id(self) -> str:
        return self.session_id[:8]

    @property
    def waited(self) -> str:
        """The wait as a reader says it out loud."""
        if self.idle_minutes is None:
            return "unknown"
        if self.idle_minutes < 1.0:
            return "just now"
        if self.idle_minutes < 60.0:
            return f"{int(self.idle_minutes)}m"
        hours = self.idle_minutes / 60.0
        if hours < 24.0:
            return f"{hours:.1f}h"
        return f"{hours / 24.0:.1f}d"


@dataclass(frozen=True)
class Board:
    """Every session this repository can see, sorted by who is waiting longest."""

    schema: str = ATTENTION_SCHEMA
    read_at: int = 0
    records_root: str = ""
    sessions: tuple[Attention, ...] = ()
    limits: tuple[str, ...] = LIMITS
    #: Anything that limited the reading -- an unreadable records directory, or
    #: a cap that hid transcripts. Never empty merely because nothing is wrong.
    notes: tuple[str, ...] = field(default_factory=tuple)

    def _in(self, *states: str) -> tuple[Attention, ...]:
        chosen = [s for s in self.sessions if s.state in states]
        chosen.sort(key=lambda s: -(s.idle_minutes or 0.0))
        return tuple(chosen)

    @property
    def asking(self) -> tuple[Attention, ...]:
        """Sessions blocked on the owner explicitly, longest wait first."""
        return self._in(ASKING)

    @property
    def waiting(self) -> tuple[Attention, ...]:
        return self._in(WAITING)

    @property
    def needs_owner(self) -> tuple[Attention, ...]:
        """The queue, in the order a person should work it.

        Asking first regardless of age: a session that stopped to put a
        question has already been told what to do next and cannot proceed,
        while one that merely ended a turn may need nothing at all.
        """
        return self.asking + self.waiting

    @property
    def working(self) -> tuple[Attention, ...]:
        return self._in(WORKING)

    @property
    def stalled(self) -> tuple[Attention, ...]:
        return self._in(STALLED)

    @property
    def dormant(self) -> tuple[Attention, ...]:
        return self._in(DORMANT, UNKNOWN)

    @property
    def live(self) -> tuple[Attention, ...]:
        """Everything that is not dormant: the windows open right now."""
        return self._in(ASKING, WAITING, WORKING, STALLED)


def classify(role: str, tools: tuple[str, ...], idle_minutes: float | None, *,
             stall_minutes: float = DEFAULT_STALL_MINUTES,
             dormant_minutes: float = DEFAULT_DORMANT_MINUTES,
             ) -> tuple[str, str]:
    """The whole decision, as a pure function of the last turn's shape.

    Separated from the reading so it can be tested without a transcript, and so
    a reader can check the rule against the four inputs it actually uses.
    Returns the state and the sentence that justifies it.
    """
    if idle_minutes is None:
        return UNKNOWN, ("the transcript carries no readable timestamp, so how "
                         "long this has been so cannot be said")
    if not role:
        return UNKNOWN, ("no turn was recorded in the window read, so there is "
                         "nothing to read a state from")

    asked = [name for name in tools if name in ASKING_TOOLS]
    if asked:
        return ASKING, (
            f"the last turn called {asked[0]}, a tool whose only purpose is to "
            f"put a question to you and wait for the answer")

    if not tools:
        if idle_minutes >= dormant_minutes:
            return DORMANT, (
                f"the last turn ended without a tool call and nothing has been "
                f"written for {idle_minutes / 60.0:.1f}h; a finished session "
                f"and a crashed one look the same here")
        return WAITING, (
            "the last turn ended without calling a tool, so nothing is running "
            "and the next thing that happens here is something you type")

    if idle_minutes >= dormant_minutes:
        return DORMANT, (
            f"a {tools[0]} call was issued and nothing has been written for "
            f"{idle_minutes / 60.0:.1f}h; this session is almost certainly gone")
    if idle_minutes >= stall_minutes:
        return STALLED, (
            f"a {tools[0]} call has been outstanding for {int(idle_minutes)}m. "
            f"That is a slow command, a permission prompt waiting for a click, "
            f"or a session that died mid-call -- the transcript records the "
            f"same thing for all three")
    return WORKING, (
        f"a {tools[0]} call is outstanding and the transcript was written "
        f"{int(idle_minutes)}m ago; this session is busy")


def _last_turn(session: SessionRun):
    """The last turn the model took, or None.

    `recent` holds turns in order and excludes `user` records carrying tool
    results, so its tail is the last thing the MODEL did rather than the last
    thing that happened. That is what makes WAITING reachable at all.
    """
    for turn in reversed(session.recent):
        if turn.role == "assistant":
            return turn
    return None


def attention_of(session: SessionRun, *, now: float | None = None,
                 stall_minutes: float = DEFAULT_STALL_MINUTES,
                 dormant_minutes: float = DEFAULT_DORMANT_MINUTES,
                 ) -> Attention:
    """Read one session's state off its last turn."""
    moment = now if now is not None else time.time()
    idle = None if session.last_at is None else max(
        0.0, (moment - session.last_at) / 60.0)
    turn = _last_turn(session)
    tools = turn.tools if turn else ()
    state, evidence = classify(
        turn.role if turn else "", tools, idle,
        stall_minutes=stall_minutes, dormant_minutes=dormant_minutes)
    summary = ""
    if turn is not None:
        summary = redact(turn.text or turn.thinking, 200)
    return Attention(
        session_id=session.session_id,
        state=state,
        idle_minutes=idle,
        cwd=session.cwd,
        branch=session.branch,
        model=session.model,
        last_tool=tools[0] if tools else "",
        last_target=(turn.targets[0] if turn and turn.targets else ""),
        summary=summary,
        turns=session.turns,
        running_agents=len(session.running_agents),
        evidence=evidence)


def read_board(cwd: str | Path | None = None, *,
               records_root=None,
               index: ActivityIndex | None = None,
               max_sessions: int = DEFAULT_MAX_SESSIONS,
               stall_minutes: float = DEFAULT_STALL_MINUTES,
               dormant_minutes: float = DEFAULT_DORMANT_MINUTES,
               now: float | None = None) -> Board:
    """Every session for `cwd`, and whether each is waiting on you.

    `index` is accepted so a caller polling this on a timer can reuse one
    `ActivityIndex` and get its incremental cursor, rather than re-reading
    every transcript from byte zero each time.
    """
    moment = now if now is not None else time.time()
    reader = index if index is not None else ActivityIndex(records_root)
    activity = reader.read(cwd=cwd, max_sessions=max_sessions, now=moment)
    sessions = tuple(
        attention_of(s, now=moment, stall_minutes=stall_minutes,
                     dormant_minutes=dormant_minutes)
        for s in activity.sessions)
    return Board(
        schema=ATTENTION_SCHEMA, read_at=int(moment),
        records_root=activity.records_root, sessions=sessions,
        notes=activity.notes)


# ── the second question: which of them is stuck on a decision ────────────────
#
# The board above answers "who has stopped". It cannot answer "and why", because
# the transcript records that a turn ended and never records what it was waiting
# for. The fleet bus does: a session that cannot proceed publishes a `blocked-on`
# finding saying so.
#
# Joining the two is worth doing and worth being careful about, because the two
# halves are graded differently and a reader who forgets that will over-trust the
# result. The state is DERIVED — the harness wrote it. The blocker is DECLARED —
# a session typed it, and `worktree_bus` says so in as many words: *"a finding's
# body is self-reported. Nothing here checks that it is true, accurate, or still
# current, and a stale finding looks exactly like a fresh one except for its
# timestamp."*
#
# Nothing on the bus records a blocker being RESOLVED. There is no such row and
# adding one here would be inventing a protocol nobody publishes to. So
# `later_activity` is offered instead: whether the same session has published
# anything at all since. It is weak evidence and it is the honest kind — a
# session that published a `landed` finding after saying it was stuck has
# probably moved on, and "probably" is the whole claim.

#: How far back to read the bus for blockers. A blocked-on finding older than
#: this is not reported: a session blocked for days has a different problem and
#: is not what a "what should I look at now" board is for.
DEFAULT_BLOCKER_HOURS = 24.0


@dataclass(frozen=True)
class Blocker:
    """A session's own statement that it cannot proceed. DECLARED throughout."""

    session_id: str
    at: int
    #: The finding's body, truncated. QUOTED and never checked.
    summary: str
    age_minutes: float
    #: Whether this session has published anything SINCE. Weak evidence that it
    #: moved on, offered because nothing records a blocker being resolved.
    later_activity: bool = False

    @property
    def stale(self) -> bool:
        return self.later_activity


def blockers(bus, *, limit: int = 400, hours: float = DEFAULT_BLOCKER_HOURS,
             now: float | None = None) -> dict:
    """The newest unresolved `blocked-on` finding per session.

    Reads the bus and writes nothing. Returns a plain mapping so a caller can
    join it onto a `Board` without this module needing to know about one.
    """
    from alelyon.runtime.common.worktree_bus import KIND_BLOCKED

    moment = now if now is not None else time.time()
    try:
        found = bus.findings(limit=limit)
    except Exception:                                             # noqa: BLE001
        # A bus that cannot be read is a missing reading, never an empty one.
        # Returning {} would render as "nobody is blocked", which is a claim.
        return {}

    newest: dict = {}
    published_after: dict = {}
    for finding in found:
        author = getattr(finding, "from_session", "")
        if not author:
            continue
        at = int(getattr(finding, "at", 0) or 0)
        age = max(0.0, (moment - at) / 60.0)
        if getattr(finding, "kind", "") != KIND_BLOCKED:
            published_after[author] = max(published_after.get(author, 0), at)
            continue
        if age > hours * 60.0:
            continue
        if author in newest and newest[author].at >= at:
            continue
        newest[author] = Blocker(
            session_id=author, at=at, summary=redact(
                str(getattr(finding, "body", "")), 240), age_minutes=age)

    return {author: Blocker(
        session_id=block.session_id, at=block.at, summary=block.summary,
        age_minutes=block.age_minutes,
        later_activity=published_after.get(author, 0) > block.at)
        for author, block in newest.items()}
