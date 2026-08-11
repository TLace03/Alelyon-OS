"""The orchestration chain, read from the harness's own transcripts.

`session_records.py` answers *which sessions exist*. This answers *what they are
doing*: a session, the fleets it launched, every agent in each fleet, the model
each one is running, what it has touched, and how far it has got.

The chain is a directory layout, not an inference
-------------------------------------------------
Claude Code writes it, and the agent chooses none of it::

    <records_root>/<project>/<session-uuid>.jsonl          the session
    <records_root>/<project>/<session-uuid>/subagents/
        agent-<id>.jsonl                                   spawned directly
        agent-<id>.meta.json                               its type and depth
        workflows/wf_<run>/journal.jsonl                   one fleet's ledger
        workflows/wf_<run>/agent-<id>.jsonl                that fleet's members

So "this agent belongs to that fleet, which that session launched" is read off a
path the harness built, in the same way `session_records` reads `cwd` off a field
the harness wrote. That is the whole reason this is `DERIVED` and not a
self-report — see `CLAIMS.md` §2.3.

What is derived and what is merely quoted
-----------------------------------------
The distinction matters more here than anywhere else in the fleet surface, and it
is carried on the dataclasses rather than left to a reader:

* **DERIVED** — that a turn happened, when, on which model, with which tool
  names, how many tokens it cost, and which files its tool calls named. The
  harness wrote every one of those; the model cannot edit them after the fact.
* **QUOTED** — the text of an assistant turn and its thinking block. That is the
  model's own output. It is displayed because it is the only window into *why* a
  fleet did what it did, and it is labelled at every read, because nothing here
  checks it and a confident paragraph is not evidence.

`Turn.text` and `Turn.thinking` are the only content fields in this module. Every
other field is structure.

Privacy: content is read here, and that is a change
---------------------------------------------------
`AGENTS.md` §9 names chat history sensitive and forbids printing, copying,
summarising or diffing it "unless the task requires the minimum necessary fields
and the owner authorized that access". **The owner authorized it on 2026-08-03**,
asking that a session's and a fleet's reasoning be recorded and displayed in the
Fleet view. This module is the minimum that satisfies that:

* only `text` and `thinking` blocks of **assistant** turns are read, truncated to
  `EXCERPT_CHARS`;
* **tool results are never read.** That is where file contents, positions, fills
  and broker output land, and none of it is needed to say what an agent is doing;
* a `Bash` command and a file path are kept because they are what "working on"
  means, and both pass through `redact()` first;
* nothing is written, cached to disk, logged, or sent anywhere. The excerpts live
  in memory for as long as a panel is showing them.

`session_records.py` keeps its structure-only guarantee and its test. This is a
separate module precisely so that guarantee did not have to be weakened to add
this one.

Cheap enough to poll
--------------------
The largest transcript in this repository is 96 MB, so a reader that parsed
whole files could not run on a timer. `ActivityIndex` keeps a byte cursor per
file and reads only what was appended since the last pass; a file whose size and
mtime are unchanged is not opened at all. The session transcript itself is read
from its **tail** — identity comes from the head, exactly as `session_records`
does it, and current activity is by definition at the end.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import time
from typing import NamedTuple

from alelyon.runtime.common.session_records import (
    DEFAULT_RECORDS_ROOT, MAX_LINE_BYTES, UNATTRIBUTED, same_directory,
)

ACTIVITY_SCHEMA = "alelyon.session-activity/0.1"

#: How much of one assistant turn is kept. Enough to read the point of a
#: paragraph, far short of the transcript. The cap is the "minimum necessary"
#: half of the §9 authorization and is applied at parse time, not at render
#: time — an excerpt that existed in memory whole has already been copied.
EXCERPT_CHARS = 900

#: How many recent turns are retained per agent and per session. A fleet view
#: shows what is happening now; the transcript is the archive and stays on disk.
#: Counted in RESPONSES, so the window reaches further back than the record
#: count suggests — a response that made three tool calls occupies one slot.
RECENT_TURNS = 24

#: Bytes of a session transcript's tail to read for current activity.
TAIL_BYTES = 512 * 1024

#: Minutes without a write, past which an agent is no longer called running.
#: Deliberately short: a fleet member that has not written for two minutes has
#: almost certainly finished or died, and calling it "running" would inflate
#: every count on the screen.
RUNNING_MINUTES = 2.0

#: Statuses. A closed vocabulary because the status IS the claim.
RUNNING = "RUNNING"        # its transcript was written within RUNNING_MINUTES
SETTLED = "SETTLED"        # the fleet's journal recorded a result for it
STOPPED = "STOPPED"        # the journal recorded it stopped or errored
QUIET = "QUIET"            # no recent write and no recorded outcome
STATUSES = (RUNNING, SETTLED, STOPPED, QUIET)

#: Fleet kinds.
FLEET_WORKFLOW = "workflow"     # a Workflow run: wf_<id>, with a journal
FLEET_DIRECT = "direct"         # agents the session spawned itself

#: Patterns scrubbed out of any command or path before it is kept. Not a
#: security boundary — a secret in an unusual shape survives it — but the common
#: shapes must not reach a screen that may be screenshotted.
_SECRET = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9]{20,}|"
    r"(?:api[_-]?key|token|secret|password|passwd|bearer)\s*[=:]\s*\S+)")

_ROLE_ASSISTANT = "assistant"
_ROLE_USER = "user"

#: An absolute path, in either convention a tool call can carry: a Windows drive
#: (`C:/…`, and `\\\\server\\share` once separators are normalised) or a POSIX
#: root. Matched on the slash-normalised form, so only forward slashes appear.
_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:/|/)")

LIMITS: tuple[str, ...] = (
    "An agent's reasoning is the model's own output. It is quoted, never "
    "checked: a confident paragraph here is exactly as unverified as a "
    "confident paragraph anywhere else, and an agent can be wrong at length.",
    "Status is inferred from when a transcript was last written and from what a "
    "workflow journal recorded. A crashed agent stops being written exactly "
    "like a finished one, so RUNNING means 'written to recently' and QUIET "
    "means 'not written to recently' - neither is a report from the agent.",
    "Files are taken from the paths tool calls NAMED, not from the filesystem. "
    "An agent that read a file it did not edit appears to be working on it, and "
    "an edit made through a shell command this cannot parse does not appear at "
    "all.",
    "This is Claude Code's convention alone. A fleet run by another vendor's "
    "tool writes nothing here and is absent rather than reported as empty.",
    "Tool RESULTS are never read, so what an agent was told back - including "
    "whether its command failed - is not visible here. Only what it did.",
    "Token counts are the harness's own usage figures for turns this pass "
    "read. A session whose transcript was truncated or rotated reports less "
    "than it spent.",
    "A turn here is one model RESPONSE, not one transcript record. The harness "
    "writes a response's blocks as several records and repeats that response's "
    "usage on each, so counting records would bill one response several times. "
    "Tool CALL counts are not deduplicated, because a response that called "
    "three tools really did call three.",
)


def redact(text: str, limit: int = 220) -> str:
    """Scrub the common secret shapes and cap the length."""
    cleaned = _SECRET.sub("[REDACTED]", str(text or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit - 1] + "…"


@dataclass(frozen=True)
class Turn:
    """One model RESPONSE, assembled from the records the harness wrote for it.

    Not one record. The harness writes a response's content blocks as separate
    records — the thinking on one, each tool call on the next — and stamps the
    response's whole `usage` on every one of them. A Turn per record would
    therefore repeat a response's tokens once per block, so the records of one
    response are merged into one Turn here, carrying its tokens once and all of
    its tool calls in order.

    `at` is the first of those records; `input_tokens` and `output_tokens` are
    the response's, not a share of them.

    Everything but `text` and `thinking` is structure the model did not write.
    """

    at: int
    role: str
    model: str
    #: The model's own words, truncated. QUOTED, never checked.
    text: str = ""
    #: The model's thinking block, truncated. QUOTED, never checked.
    thinking: str = ""
    #: Tool names this turn called, in order.
    tools: tuple[str, ...] = ()
    #: What those calls named — a path, a pattern, a redacted command.
    targets: tuple[str, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    #: How many thinking blocks this turn carried, redacted ones included.
    #: DERIVED, not QUOTED: it counts blocks the harness wrote rather than
    #: quoting what they said, and it is the only way to tell a turn that
    #: reasoned behind a redaction from a turn that did not reason.
    thinking_blocks: int = 0
    #: How many of those carried text after redaction. Never larger than
    #: `thinking_blocks`; the difference is what was withheld.
    thinking_readable: int = 0

    @property
    def has_content(self) -> bool:
        return bool(self.text or self.thinking)


@dataclass(frozen=True)
class TurnScope:
    """What the turns that named one exact set of files spent.

    The harness writes a turn's `usage` and its `tool_use` inputs on the same
    record, so the tokens a turn spent and the files it named are available
    together. Everything else in this module sums the two apart — a total bill
    and a union of files — which throws the pairing away. This keeps it, and it
    is the only thing standing between "the fleet spent 4M tokens and touched
    these 300 files" and "this area cost at least *this* much".

    Turns are grouped by the exact set of files they named, never divided
    between them. A turn that read two files spent its tokens on both and no
    record says how much on each, so a reader wanting a per-area figure gets a
    lower and an upper bound out of these groups rather than an invented split.
    """

    #: Repository-relative, sorted, deduplicated. Empty when these turns named
    #: no placeable file.
    files: tuple[str, ...] = ()
    #: Repository paths these turns named inside a shell COMMAND, verified
    #: against the index. A weaker class of evidence than `files` and kept
    #: apart for that reason: `files` comes from a field the harness recorded,
    #: while this is a parse of text the model wrote. Both say "named", neither
    #: says "changed" — but only one of them was structured when it arrived.
    commanded: tuple[str, ...] = ()
    #: How many of the paths these turns named fell outside the repository. Non-
    #: zero with an empty `files` means the turns worked entirely outside the
    #: tree, which is NOT the same fact as naming no file at all.
    outside: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def named_nothing(self) -> bool:
        """True when these turns named nothing this repository contains.

        Thinking, a search, a plain answer, a shell command that mentioned no
        tracked file — real spend with nothing to hang it on. Reported as its
        own quantity and never spread across the files a neighbouring turn
        happened to name.
        """
        return not self.files and not self.commanded and not self.outside

    @property
    def placeable(self) -> tuple[str, ...]:
        """`files` widened by `commanded`, for a caller that wants both.

        Offered as a property rather than folded into `files` so that a reader
        choosing the wider set does so deliberately and can still see which
        half it came from.
        """
        return tuple(sorted({*self.files, *self.commanded}))


@dataclass(frozen=True)
class AgentRun:
    """One member of a fleet."""

    agent_id: str
    #: `attributionAgent`/`agentType` from the harness — general-purpose,
    #: workflow-subagent, Explore, and so on.
    agent_type: str
    #: The fleet it belongs to: a workflow run id, or FLEET_DIRECT.
    fleet_id: str
    session_id: str
    path: str
    #: Every model seen on this agent's turns, most-used first. A fleet member
    #: can be re-driven on a different model, and reporting only the first would
    #: hide that.
    models: tuple[str, ...] = ()
    started_at: int | None = None
    last_at: int | None = None
    turns: int = 0
    tool_counts: tuple[tuple[str, int], ...] = ()
    files: tuple[str, ...] = ()
    #: Tokens against the files the same turn named. Covers every turn read,
    #: not only the `recent` window.
    scopes: tuple[TurnScope, ...] = ()
    #: The instruction it was given, truncated. Written by whatever spawned it.
    brief: str = ""
    recent: tuple[Turn, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    #: Thinking blocks over the whole run, redacted ones included, and how many
    #: carried text. DERIVED. Complete rather than windowed: `recent` is capped
    #: at `RECENT_TURNS`, so summing these from it would under-report every run
    #: longer than that.
    thinking_blocks: int = 0
    thinking_readable: int = 0
    status: str = QUIET
    status_evidence: str = ""
    #: Where this agent actually ran, as the harness stamped it on the
    #: transcript. Carried on the AGENT and not only on the session because an
    #: agent given `isolation: worktree` runs on a branch of its own, and
    #: inheriting the parent session's branch would attribute its work to the
    #: wrong one. `gitBranch` is written per record by the harness, so it is the
    #: branch that was checked out AT THE TIME rather than whatever is checked
    #: out when something reads the transcript back.
    cwd: str = ""
    branch: str = ""

    @property
    def model(self) -> str:
        return self.models[0] if self.models else UNATTRIBUTED

    @property
    def repo_files(self) -> tuple[str, ...]:
        """`files`, as repository-relative paths, using this agent's own `cwd`.

        The agent is the right root to use and the session is not: an agent given
        `isolation: worktree` runs in a checkout of its own, which is why `cwd`
        is carried per agent rather than inherited. A worktree path resolves to
        the file it is a checkout of either way — see `repo_relative`.

        Files outside the repository are dropped, so this is shorter than
        `files` whenever an agent touched a scratchpad. Compare the two lengths
        rather than assuming this placed everything.
        """
        roots = [r for r in (repo_root_of(self.cwd), self.cwd) if r]
        return repo_paths_of(self.files, roots=roots)

    @property
    def elapsed_seconds(self) -> int | None:
        if self.started_at is None or self.last_at is None:
            return None
        return max(0, self.last_at - self.started_at)

    @property
    def last_reasoning(self) -> str:
        """The most recent thing it said or thought. QUOTED."""
        for turn in reversed(self.recent):
            if turn.thinking:
                return turn.thinking
            if turn.text:
                return turn.text
        return ""

    @property
    def doing(self) -> str:
        """One line: what its latest turn actually did."""
        for turn in reversed(self.recent):
            if turn.tools:
                target = turn.targets[0] if turn.targets else ""
                return f"{turn.tools[0]} {target}".strip()
        return "no tool call in the turns read"


@dataclass(frozen=True)
class Fleet:
    """A set of agents one session launched together."""

    fleet_id: str
    kind: str
    session_id: str
    path: str
    agents: tuple[AgentRun, ...] = ()
    #: From the workflow journal: how many agents it started and settled. The
    #: journal is the harness's own ledger, so this is the one place a fleet's
    #: own account of itself can be compared with its members' transcripts.
    started: int = 0
    settled: int = 0
    started_at: int | None = None
    last_at: int | None = None

    @property
    def running(self) -> tuple[AgentRun, ...]:
        return tuple(a for a in self.agents if a.status == RUNNING)

    @property
    def models(self) -> tuple[tuple[str, int], ...]:
        counts: dict[str, int] = {}
        for agent in self.agents:
            counts[agent.model] = counts.get(agent.model, 0) + 1
        return tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    @property
    def output_tokens(self) -> int:
        return sum(a.output_tokens for a in self.agents)

    @property
    def label(self) -> str:
        return self.fleet_id if self.kind == FLEET_WORKFLOW else "direct spawns"


@dataclass(frozen=True)
class SessionRun:
    """One session, its fleets, and what it is doing itself.

    This is the orchestrator in the user's sense: a session that has launched
    fleets is *behaving as* a project manager. The role is derived from what it
    did, and is never a declared identity — nothing in a transcript says "I am
    the manager", and this module does not invent one.
    """

    session_id: str
    cwd: str
    path: str
    branch: str = ""
    models: tuple[str, ...] = ()
    started_at: int | None = None
    last_at: int | None = None
    turns: int = 0
    tool_counts: tuple[tuple[str, int], ...] = ()
    recent: tuple[Turn, ...] = ()
    fleets: tuple[Fleet, ...] = ()
    #: Tokens against the files the same turn named — the session's own turns
    #: only. Its fleets' scopes hang off their own `AgentRun`s, for the reason
    #: the module header gives for never summing the two.
    scopes: tuple[TurnScope, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    #: Thinking blocks the harness recorded for this session, redacted ones
    #: included, and how many carried text. DERIVED.
    #:
    #: **Complete over what was READ, and a session is read from its TAIL.**
    #: `AgentRun` carries the same pair and there the total really is the whole
    #: run, because an agent transcript is absorbed entire. Here it is not: on
    #: its first pass `_session` absorbs at most the last `TAIL_BYTES` of the
    #: file, so a long session's earlier blocks were never seen and cannot be
    #: counted. A reading that needs the whole session — the reasoning corpus
    #: does — has to make its own pass rather than sum these.
    #:
    #: What they ARE complete over is that window: they are accumulated per
    #: record as it is absorbed, not summed from `recent`, which is capped at
    #: `RECENT_TURNS` and would under-report every run longer than that.
    thinking_blocks: int = 0
    thinking_readable: int = 0
    #: True when the transcript's tail was reached, i.e. `recent` really is the
    #: most recent activity rather than an arbitrary window.
    tail_read: bool = True
    notes: tuple[str, ...] = ()

    @property
    def model(self) -> str:
        return self.models[0] if self.models else UNATTRIBUTED

    @property
    def agents(self) -> tuple[AgentRun, ...]:
        return tuple(a for fleet in self.fleets for a in fleet.agents)

    @property
    def orchestrating(self) -> bool:
        """Whether this session has launched anything. The manager role."""
        return bool(self.agents)

    @property
    def running_agents(self) -> tuple[AgentRun, ...]:
        return tuple(a for a in self.agents if a.status == RUNNING)

    @property
    def active(self) -> bool:
        if self.last_at is None:
            return False
        return (time.time() - self.last_at) / 60.0 <= RUNNING_MINUTES

    @property
    def last_reasoning(self) -> str:
        for turn in reversed(self.recent):
            if turn.role != _ROLE_ASSISTANT:
                continue
            if turn.thinking:
                return turn.thinking
            if turn.text:
                return turn.text
        return ""


@dataclass(frozen=True)
class Activity:
    """Every session the scan covered, with its chain."""

    schema: str
    read_at: int
    records_root: str
    sessions: tuple[SessionRun, ...] = ()
    limits: tuple[str, ...] = LIMITS
    notes: tuple[str, ...] = ()

    @property
    def fleets(self) -> tuple[Fleet, ...]:
        return tuple(f for s in self.sessions for f in s.fleets)

    @property
    def agents(self) -> tuple[AgentRun, ...]:
        return tuple(a for f in self.fleets for a in f.agents)

    @property
    def running(self) -> tuple[AgentRun, ...]:
        return tuple(a for a in self.agents if a.status == RUNNING)

    def session(self, session_id: str) -> SessionRun | None:
        return next((s for s in self.sessions if s.session_id == session_id), None)

    @property
    def headline(self) -> str:
        live = [s for s in self.sessions if s.active]
        return (f"{len(self.sessions)} session(s), {len(live)} active · "
                f"{len(self.fleets)} fleet(s) · {len(self.agents)} agent(s), "
                f"{len(self.running)} running")


# ── parsing ──────────────────────────────────────────────────────────────────
def _stamp(value) -> int | None:
    """A transcript timestamp is ISO-8601 with a Z. Unknown stays None."""
    text = str(value or "")
    if not text:
        return None
    try:
        import datetime as _dt
        return int(_dt.datetime.fromisoformat(
            text.replace("Z", "+00:00")).timestamp())
    except (ValueError, TypeError, OverflowError):
        return None


def _blocks(message: dict) -> tuple[str, str, list[str], list[str], int, int]:
    """(text, thinking, tools, targets, blocks, readable) out of one message.

    `blocks` and `readable` are the reason this returns six things instead of
    four. The joined `thinking` string cannot distinguish a turn that carried
    three REDACTED thinking blocks from a turn that did no thinking at all: a
    redacted block contributes the empty string and disappears into the join.
    Those two are opposite facts about a model, and on this machine the first
    is the common case — measured 2026-08-10, 3,971 of 4,988 thinking blocks in
    three days carried a signature and an empty body.

    So the count is taken here, where the blocks are still separate, and it is
    DERIVED in this module's sense: the harness wrote the block, and a model
    cannot delete one after the fact. No additional content is read to produce
    it.

    Tool *results* are skipped by name rather than by omission: they are the
    field this module must not read, and a reader of this function should see
    the decision rather than infer it from what is missing.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tools: list[str] = []
    targets: list[str] = []
    content = message.get("content")
    blocks = readable = 0
    if isinstance(content, str):
        return redact(content, EXCERPT_CHARS), "", [], [], 0, 0
    if not isinstance(content, list):
        return "", "", [], [], 0, 0
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            body = str(block.get("thinking") or block.get("text") or "")
            blocks += 1
            if body.strip():
                readable += 1
            thinking_parts.append(body)
        elif kind == "tool_use":
            name = str(block.get("name") or "")
            tools.append(name)
            target = _target_of(name, block.get("input"))
            if target:
                targets.append(target)
        elif kind == "tool_result":
            continue  # deliberately unread — see the module docstring
    return (redact("\n".join(text_parts), EXCERPT_CHARS),
            redact("\n".join(thinking_parts), EXCERPT_CHARS),
            tools, targets, blocks, readable)


class MessageBlocks(NamedTuple):
    """What one harness message contained, named rather than positional."""

    #: Assistant prose, joined, redacted and capped at `EXCERPT_CHARS`.
    text: str
    #: Reasoning, joined, redacted and capped the same way. **Redacted blocks
    #: contribute the empty string and vanish into the join**, which is why the
    #: two counts below exist and why this string alone must never be read as
    #: evidence that a model did or did not reason.
    thinking: str
    tools: tuple[str, ...]
    #: What each tool call named — a path, a pattern or a command. Not what any
    #: of them returned: tool results are never read.
    targets: tuple[str, ...]
    #: Thinking blocks in this message, redacted ones included. DERIVED.
    blocks: int
    #: How many of those carried text. Never larger than `blocks`.
    readable: int


def message_blocks(message: dict) -> MessageBlocks:
    """`_blocks` under a public name, for readers outside this module.

    There is one transcript-block parser in this repository and this is it. A
    second module that needs reasoning out of a harness message — the reasoning
    corpus does — calls this rather than walking `content` itself, because a
    second walk is how the weaker one survives: it would be the copy that
    forgets `tool_result` is deliberately unread, or that a redacted block is a
    block.

    It delegates rather than reimplements, so the two can never disagree.
    """
    text, thinking, tools, targets, blocks, readable = _blocks(message)
    return MessageBlocks(text=text, thinking=thinking, tools=tuple(tools),
                         targets=tuple(targets), blocks=blocks,
                         readable=readable)


def _target_of(tool: str, payload) -> str:
    """What one tool call is aimed at, redacted.

    A path or a pattern is the answer to "what is it working on", which is the
    question the whole view exists for. A command is kept because for `Bash` it
    is the only answer there is, and it is the field most likely to carry
    something that should not be on a screen — so it goes through `redact`.
    """
    if not isinstance(payload, dict):
        return ""
    for key in ("file_path", "path", "notebook_path", "pattern", "query",
                "description", "command"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return redact(value, 160)
    return ""


#: A path-shaped run of characters inside a shell command. Deliberately loose:
#: it produces CANDIDATES, and the index decides which of them are real.
_COMMAND_TOKEN = re.compile(r"[A-Za-z0-9_./\\-]{4,}")


def _command_candidates(payload) -> list[str]:
    """Path-shaped tokens in a tool call's command string.

    Candidates only. `Bash` carries the largest block of spend that names no
    file in a structured field — measured at 10.4% of unscoped output on this
    repository — and the paths are usually right there in the command
    (`pytest tests/x.py`, `python tools/y.py`). Nothing here decides they are
    paths; `_commanded_files` asks the repository's index, and a token the
    index does not know is dropped.

    A token must contain a separator. A bare `main.py` is a word as often as a
    path, and matching one against the index by name alone would place spend on
    a file the turn never mentioned.
    """
    if not isinstance(payload, dict):
        return []
    command = payload.get("command")
    if not isinstance(command, str) or not command.strip():
        return []
    out: list[str] = []
    for token in _COMMAND_TOKEN.findall(command):
        candidate = _posix(token).lstrip("./")
        if "/" in candidate:
            out.append(candidate)
    return out


def _commanded_files(candidates: Iterable[str],
                     tracked: frozenset) -> tuple[str, ...]:
    """The candidates this repository actually tracks, sorted and deduplicated.

    An empty `tracked` returns nothing at all. That is the absence of a
    membership test, not a licence to accept every candidate — see
    `worktree_areas.tracked_files`.
    """
    if not tracked:
        return ()
    return tuple(sorted({c for c in (candidates or ()) if c in tracked}))


def _files_in(payload) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out = []
    for key in ("file_path", "path", "notebook_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return out


#: The segment a git worktree lives under. A worktree is a second checkout of
#: the same repository, so a file inside one is a file in the repository and its
#: subject is the path BELOW this marker — `alelyon/x.py`, not
#: `.claude/worktrees/wt-9/alelyon/x.py`, which matches no area rule at all.
_WORKTREE_MARK = "/.claude/worktrees/"


def _posix(path: str) -> str:
    """Separators normalised, quotes and stray whitespace removed."""
    return str(path or "").strip().strip('"').strip("'").replace("\\", "/")


def repo_root_of(path: str) -> str:
    """The main checkout a path belongs to, when it is inside a worktree.

    A worktree of this repository lives at `<root>/.claude/worktrees/<name>`, so
    the root is simply the text before that marker. Returns "" when the path
    names no worktree, which is not the same as "not in a repository" — it means
    this rule had nothing to say and the caller should use a root it already
    knows.
    """
    posix = _posix(path)
    cut = posix.find(_WORKTREE_MARK)
    return posix[:cut] if cut > 0 else ""


def repo_relative(path: str, *, roots: Iterable[str] = ()) -> str:
    """One raw tool-payload path as a repository-relative POSIX path, or "".

    `AgentRun.files` holds the strings tool calls actually carried, which are
    very often absolute and Windows-shaped —
    `C:\\Users\\...\\famMain\\alelyon\\runtime\\common\\blueprint.py`. Every area
    rule in `worktree_areas` is a repository-RELATIVE prefix (`alelyon/runtime/`),
    and `area_of` only normalises separators, so an absolute path matches no rule
    and resolves to `UNMAPPED`. Measured over 1256 agents and 6852 recorded
    files, 28 resolved to an area — 0.4% — while 6216 of those files were inside
    the repository. This is the conversion that was missing.

    An empty string is returned for anything that cannot be shown to be inside
    one of `roots`, and that is deliberate: an agent's scratchpad file and a
    transcript under `~/.claude/projects` are genuinely not repository work, and
    guessing them into an area would replace a blind edge with a wrong one.

    A path inside a worktree is reported as the file it is a checkout OF, so two
    agents editing the same module from different worktrees agree on the subject.

    What this does not do: it never touches the filesystem, so it cannot tell a
    path that exists from one that does not, and a relative path is trusted to be
    relative to the repository rather than resolved against anything.
    """
    posix = _posix(path)
    if not posix:
        return ""

    inside = repo_root_of(posix)
    if inside:
        # Inside a worktree, whatever root the caller had in mind.
        posix = posix[len(inside) + len(_WORKTREE_MARK):]
        cut = posix.find("/")
        return posix[cut + 1:] if cut >= 0 else ""

    absolute = bool(_ABSOLUTE.match(posix))
    if absolute:
        best = ""
        for root in roots:
            candidate = _posix(root).rstrip("/")
            if not candidate:
                continue
            head = posix[:len(candidate)]
            if head.casefold() == candidate.casefold() and \
                    posix[len(candidate):len(candidate) + 1] == "/" and \
                    len(candidate) > len(best):
                best = candidate
        if not best:
            return ""
        posix = posix[len(best) + 1:]

    # A relative path that climbs out of the tree describes something this
    # cannot place, and `..` never appears in a repository-relative path.
    if not posix or posix.startswith("../") or posix == "..":
        return ""
    return posix.lstrip("/")


def repo_paths_of(paths: Iterable[str], *,
                  roots: Iterable[str] = ()) -> tuple[str, ...]:
    """`repo_relative` over many paths: sorted, deduplicated, blanks dropped.

    The count of what fell out is not reported here. A caller that needs to say
    "n of m files were placed" should compare against the input, because a
    silently shorter tuple is exactly how a coverage claim becomes wrong.
    """
    roots = tuple(roots)
    return tuple(sorted({rel for rel in
                         (repo_relative(p, roots=roots) for p in (paths or ()))
                         if rel}))


def _roots_for(accumulator, extra: Iterable[str] = ()) -> tuple[str, ...]:
    """The checkouts a transcript's paths may be relative to.

    Its own `cwd` first, then whatever the caller knows about. Order does not
    decide the match — `repo_relative` takes the LONGEST root that fits, so a
    worktree nested under the main checkout still wins over the checkout.
    """
    own = (repo_root_of(accumulator.cwd), accumulator.cwd)
    return tuple(r for r in (*own, *(extra or ())) if r)


def _turn_scopes(raw: dict, *, roots: Iterable[str] = (),
                 tracked: Iterable[str] = ()) -> tuple[TurnScope, ...]:
    """`_Accumulator.scopes` with its paths placed in the repository.

    Groups are not merged after placement even when two of them land on the
    same repository-relative set, because merging would make `outside` — a
    count of paths, not of turns — stop meaning anything. The folds downstream
    sum across groups anyway.
    """
    roots = tuple(roots)
    tracked = frozenset(tracked or ())
    out: list[TurnScope] = []
    for (paths, spoken), (turns, input_tokens, output_tokens) in raw.items():
        placed = repo_paths_of(paths, roots=roots)
        # Counted per input path rather than as `len(paths) - len(placed)`:
        # two paths can place onto one repository-relative file (a worktree
        # copy and the main checkout), and that subtraction would report a
        # file that WAS placed as one that fell outside.
        outside = sum(1 for p in paths if not repo_relative(p, roots=roots))
        # A path already named in a structured field is not also reported as
        # commanded: the stronger evidence stands, and listing it twice would
        # make the weaker class look bigger than it is.
        commanded = tuple(c for c in _commanded_files(spoken, tracked)
                          if c not in placed)
        out.append(TurnScope(
            files=placed, commanded=commanded, outside=outside, turns=turns,
            input_tokens=input_tokens, output_tokens=output_tokens))
    out.sort(key=lambda s: (-s.output_tokens, s.files))
    return tuple(out)


@dataclass
class _Accumulator:
    """Mutable running totals for one transcript, kept across incremental reads."""

    models: dict[str, int] = field(default_factory=dict)
    tools: dict[str, int] = field(default_factory=dict)
    files: set = field(default_factory=set)
    #: The set of files a turn named -> [turns, input, output] for every turn
    #: that named exactly that set. Raw paths, because placing them needs `cwd`
    #: and `cwd` may arrive on a later record than the first turn does.
    scopes: dict = field(default_factory=dict)
    turns: int = 0
    started_at: int | None = None
    last_at: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    #: Thinking blocks over the WHOLE run, not only the `recent` window. The
    #: window is capped at `RECENT_TURNS`, so a total taken from it would
    #: under-report every run longer than that; these are accumulated per
    #: record as it is read and are complete.
    thinking_blocks: int = 0
    thinking_readable: int = 0
    recent: list = field(default_factory=list)
    brief: str = ""
    cwd: str = ""
    session_id: str = ""
    branch: str = ""
    agent_type: str = ""
    #: The response id the last record belonged to. One model response that
    #: made several tool calls is written as several records, each carrying
    #: that ONE response's `usage` in full — see `_priced`.
    last_message_id: str = ""
    #: The response the newest entry in `recent` belongs to. A later record of
    #: that same response merges into it instead of appending a second Turn.
    turn_of_message: str = ""
    #: The response being assembled: [turns, input, output, {files}]. Held open
    #: until a different response arrives, so every file the response named is
    #: charged to its tokens together instead of to whichever record came first.
    pending: list = field(default_factory=list)

    def absorb(self, record: dict) -> None:
        kind = record.get("type")
        at = _stamp(record.get("timestamp"))
        if at is not None:
            self.started_at = at if self.started_at is None else min(
                self.started_at, at)
            self.last_at = at if self.last_at is None else max(self.last_at, at)
        if not self.cwd and record.get("cwd"):
            self.cwd = str(record["cwd"])
        if not self.session_id and record.get("sessionId"):
            self.session_id = str(record["sessionId"])
        if not self.branch and record.get("gitBranch"):
            self.branch = str(record["gitBranch"])
        if not self.agent_type:
            for key in ("attributionAgent", "agentType"):
                if record.get(key):
                    self.agent_type = str(record[key])
                    break
        if kind not in (_ROLE_ASSISTANT, _ROLE_USER):
            return
        message = record.get("message")
        if not isinstance(message, dict):
            return

        model = str(message.get("model") or "")
        text, thinking, tools, targets, blocks, readable = _blocks(message)
        if kind == _ROLE_USER:
            # The first user record of an agent transcript is its brief: the
            # instruction whatever spawned it wrote. Later user records are tool
            # results, which are not read.
            if not self.brief and text:
                self.brief = text
            if not tools:
                return

        usage = message.get("usage")
        input_tokens = output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens") or 0) + int(
                usage.get("cache_read_input_tokens") or 0) + int(
                usage.get("cache_creation_input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)

        # One model response that called three tools is written as three
        # records, and the harness stamps that response's whole `usage` on each
        # of them. Charging every record would bill the response three times,
        # so tokens, turns and models are counted once per response id and only
        # the tool calls — which really did happen three times — are not.
        # An id-less record is charged on its own: it cannot be shown to be a
        # repeat, and treating unknown as duplicate would silently drop spend.
        message_id = str(message.get("id") or "")
        repeat = bool(message_id) and message_id == self.last_message_id
        self.last_message_id = message_id
        if not repeat:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if model:
                self.models[model] = self.models.get(model, 0) + 1
        # Counted per RECORD rather than per response, and deliberately not
        # gated on `repeat`: the blocks of one response are split across its
        # records, each block appearing on exactly one of them, so a per-record
        # sum is the response's true total while a `not repeat` guard would keep
        # only whichever record happened to come first.
        self.thinking_blocks += blocks
        self.thinking_readable += readable
        for name in tools:
            self.tools[name] = self.tools.get(name, 0) + 1
        named: list[str] = []
        spoken: list[str] = []
        for block in (message.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                for path in _files_in(block.get("input")):
                    self.files.add(path)
                    named.append(path)
                spoken.extend(_command_candidates(block.get("input")))

        # The pairing, kept where it exists: a response's tokens against the
        # files that same response named. A response is held open until the
        # next one begins, because its files arrive across several records and
        # charging the first record's file alone would credit one file with
        # work done on three.
        if repeat:
            if self.pending:
                self.pending[3].update(named)
                self.pending[4].update(spoken)
        else:
            self._flush_pending()
            self.pending = [1, input_tokens, output_tokens, set(named),
                            set(spoken)]

        if kind == _ROLE_ASSISTANT and not repeat:
            self.turns += 1
        if text or thinking or tools:
            # One response is ONE Turn. Its blocks arrive on several records —
            # the thinking on one, each tool call on the next — and appending a
            # Turn per record would repeat the response's usage once per block.
            # `session_spend` sums `Turn.output_tokens` over this window for its
            # per-model split and its burn series, so a Turn per record inflates
            # both by exactly the parallel-tool-call rate. Merging also makes
            # the window agree with `turns`, which counts responses.
            if repeat and self.recent and self.turn_of_message == message_id:
                previous = self.recent[-1]
                self.recent[-1] = replace(
                    previous,
                    text=previous.text or text,
                    thinking=previous.thinking or thinking,
                    tools=previous.tools + tuple(tools),
                    targets=previous.targets + tuple(targets),
                    # Summed rather than kept-first: `thinking` above keeps the
                    # first non-empty body, but the COUNTS are of blocks spread
                    # across this response's records and every one of them is
                    # real.
                    thinking_blocks=previous.thinking_blocks + blocks,
                    thinking_readable=previous.thinking_readable + readable)
            else:
                self.recent.append(Turn(
                    at=at or 0, role=str(kind), model=model or UNATTRIBUTED,
                    text=text, thinking=thinking, tools=tuple(tools),
                    targets=tuple(targets), input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    thinking_blocks=blocks, thinking_readable=readable))
                self.turn_of_message = message_id
            # Trimming drops from the FRONT, so the Turn a later record of this
            # response would merge into is never the one removed.
            if len(self.recent) > RECENT_TURNS:
                del self.recent[:-RECENT_TURNS]

    def _flush_pending(self) -> None:
        """Settle the response being assembled into the scope groups."""
        if not self.pending:
            return
        turns, input_tokens, output_tokens, files, spoken = self.pending
        key = (tuple(sorted(files)), tuple(sorted(spoken)))
        scope = self.scopes.setdefault(key, [0, 0, 0])
        scope[0] += turns
        scope[1] += input_tokens
        scope[2] += output_tokens
        self.pending = []

    def settled_scopes(self) -> dict:
        """`scopes` with the open response folded in.

        Called by the readers rather than left to `absorb`, because the last
        response of an incremental pass is not finished — its next record may
        arrive in the next poll, and closing it early would split one response
        across two groups.
        """
        if not self.pending:
            return self.scopes
        turns, input_tokens, output_tokens, files, spoken = self.pending
        merged = {k: list(v) for k, v in self.scopes.items()}
        key = (tuple(sorted(files)), tuple(sorted(spoken)))
        scope = merged.setdefault(key, [0, 0, 0])
        scope[0] += turns
        scope[1] += input_tokens
        scope[2] += output_tokens
        return merged

    def ranked_models(self) -> tuple[str, ...]:
        return tuple(name for name, _count in sorted(
            self.models.items(), key=lambda kv: (-kv[1], kv[0])))

    def ranked_tools(self) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(self.tools.items(), key=lambda kv: (-kv[1], kv[0])))


def _iter_records(path: Path, *, start: int = 0,
                  tail_bytes: int | None = None) -> tuple[list, int]:
    """(records, new cursor) from `path`, reading only what is needed.

    `start` resumes from a byte offset — the incremental path, used on every
    poll. `tail_bytes` reads only the end of a large file and drops the first
    (partial) line, which is how a 96 MB session transcript is read at all.
    """
    records: list = []
    try:
        size = path.stat().st_size
    except OSError:
        return records, start
    if start > size:
        start = 0  # the file was rotated or truncated; begin again
    offset = start
    if tail_bytes is not None and size - start > tail_bytes:
        offset = size - tail_bytes
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            raw = handle.read()
    except OSError:
        return records, start
    if offset > start:
        # Dropped a partial line to get here: never parse half a record.
        cut = raw.find(b"\n")
        raw = raw[cut + 1:] if cut >= 0 else b""
    for line in raw.split(b"\n"):
        if not line.strip() or len(line) > MAX_LINE_BYTES:
            continue
        try:
            record = json.loads(line.decode("utf-8", "replace"))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records, size


def _agent_status(last_at: int | None, journal: dict, agent_id: str,
                  now: float) -> tuple[str, str]:
    outcome = journal.get(agent_id)
    if last_at is not None and (now - last_at) / 60.0 <= RUNNING_MINUTES:
        return RUNNING, (f"its transcript was written within "
                         f"{RUNNING_MINUTES:g} minute(s)")
    if outcome == "result":
        return SETTLED, "the fleet's journal recorded a result for it"
    if outcome in ("stopped", "error", "failed"):
        return STOPPED, f"the fleet's journal recorded {outcome!r}"
    if last_at is None:
        return QUIET, "its transcript carries no timestamp this pass could read"
    minutes = (now - last_at) / 60.0
    return QUIET, (f"no write for {minutes:.0f} minute(s) and no outcome in a "
                   f"journal; a crashed agent looks exactly like this")


def _read_journal(path: Path) -> dict:
    """agentId → last event type, from a workflow's own ledger."""
    outcomes: dict[str, str] = {}
    records, _cursor = _iter_records(path)
    for record in records:
        agent_id = str(record.get("agentId") or "")
        if agent_id:
            outcomes[agent_id] = str(record.get("type") or "")
    return outcomes


class ActivityIndex:
    """Incremental reader. Hold one and call `read()` on a timer.

    Every file it has seen carries a `(size, mtime, cursor)`. A file that has not
    grown is not opened; one that has is read from its cursor. That is what makes
    a two-second poll affordable against a directory holding a gigabyte of
    transcripts.
    """

    def __init__(self, records_root: str | Path | None = None) -> None:
        self.records_root = Path(records_root) if records_root is not None \
            else DEFAULT_RECORDS_ROOT
        self._cursors: dict[str, tuple[int, float, int]] = {}
        self._state: dict[str, _Accumulator] = {}
        self._journals: dict[str, dict] = {}

    # ── one pass ─────────────────────────────────────────────────────────────
    def read(self, *, cwd: str | Path | None = None,
             max_sessions: int = 12, now: float | None = None,
             roots: Iterable[str] = (),
             tracked: Iterable[str] = ()) -> Activity:
        """Read every session for `cwd`, most recently written first.

        `roots` are additional checkouts of this repository, for placing the
        paths tool calls named. A session's own `cwd` is always a root, which
        covers a session working inside the worktree it was started in; it does
        NOT cover the common case of a session started in the main checkout
        that does its editing in a worktree elsewhere on disk. Only the mesh
        knows where this repository's worktrees actually are, so it passes them
        in rather than this module guessing at path conventions — measured on
        this repository, 20.1% of read output named paths that no root here
        could place, against 20.1% that landed on an area.
        """
        moment = now if now is not None else time.time()
        notes: list[str] = []
        try:
            projects = [d for d in sorted(self.records_root.iterdir())
                        if d.is_dir()]
        except OSError:
            return Activity(schema=ACTIVITY_SCHEMA, read_at=int(moment),
                            records_root=str(self.records_root),
                            notes=("the harness records directory could not be "
                                   "read; no session activity is available",))

        candidates: list[tuple[float, Path]] = []
        for project in projects:
            for transcript in project.glob("*.jsonl"):
                try:
                    stat = transcript.stat()
                except OSError:
                    continue
                candidates.append((stat.st_mtime, transcript))
        candidates.sort(key=lambda pair: -pair[0])

        sessions: list[SessionRun] = []
        examined = 0
        for _mtime, transcript in candidates:
            if len(sessions) >= max_sessions:
                notes.append(
                    f"{len(candidates) - examined} older transcript(s) were not "
                    f"read: the newest {max_sessions} are shown")
                break
            examined += 1
            session = self._session(transcript, moment, roots=roots,
                                    tracked=tracked)
            if session is None:
                continue
            if cwd is not None and not same_directory(session.cwd, str(cwd)):
                continue
            sessions.append(session)

        return Activity(schema=ACTIVITY_SCHEMA, read_at=int(moment),
                        records_root=str(self.records_root),
                        sessions=tuple(sessions), notes=tuple(notes))

    # ── one session ──────────────────────────────────────────────────────────
    def _session(self, transcript: Path, now: float, *,
                 roots: Iterable[str] = (),
                 tracked: Iterable[str] = ()) -> SessionRun | None:
        session_id = transcript.stem
        accumulator, changed = self._absorb(transcript, tail_bytes=TAIL_BYTES)
        if accumulator is None:
            return None
        fleets = self._fleets(transcript.with_suffix("") , session_id, now,
                              roots=roots, tracked=tracked)
        return SessionRun(
            session_id=accumulator.session_id or session_id,
            cwd=accumulator.cwd,
            path=str(transcript),
            branch=accumulator.branch,
            models=accumulator.ranked_models(),
            started_at=accumulator.started_at,
            last_at=accumulator.last_at,
            turns=accumulator.turns,
            tool_counts=accumulator.ranked_tools(),
            recent=tuple(accumulator.recent),
            fleets=fleets,
            scopes=_turn_scopes(accumulator.settled_scopes(),
                                roots=_roots_for(accumulator, roots),
                                tracked=tracked),
            input_tokens=accumulator.input_tokens,
            output_tokens=accumulator.output_tokens,
            thinking_blocks=accumulator.thinking_blocks,
            thinking_readable=accumulator.thinking_readable,
            notes=() if changed else (),
        )

    def _fleets(self, session_dir: Path, session_id: str,
                now: float, *, roots: Iterable[str] = (),
                tracked: Iterable[str] = ()) -> tuple[Fleet, ...]:
        subagents = session_dir / "subagents"
        if not subagents.is_dir():
            return ()
        fleets: list[Fleet] = []

        direct = sorted(subagents.glob("agent-*.jsonl"))
        if direct:
            agents = tuple(self._agent(p, FLEET_DIRECT, session_id, {}, now,
                                       roots=roots, tracked=tracked)
                           for p in direct)
            agents = tuple(a for a in agents if a is not None)
            if agents:
                fleets.append(self._fleet(FLEET_DIRECT, FLEET_DIRECT, session_id,
                                          str(subagents), agents, {}))

        workflows = subagents / "workflows"
        if workflows.is_dir():
            for run in sorted(workflows.iterdir()):
                if not run.is_dir():
                    continue
                journal = self._journal(run / "journal.jsonl")
                agents = tuple(a for a in (
                    self._agent(p, run.name, session_id, journal, now,
                                roots=roots, tracked=tracked)
                    for p in sorted(run.glob("agent-*.jsonl"))) if a is not None)
                if agents:
                    fleets.append(self._fleet(run.name, FLEET_WORKFLOW,
                                              session_id, str(run), agents,
                                              journal))
        fleets.sort(key=lambda f: -(f.last_at or 0))
        return tuple(fleets)

    @staticmethod
    def _fleet(fleet_id: str, kind: str, session_id: str, path: str,
               agents: tuple[AgentRun, ...], journal: dict) -> Fleet:
        stamps = [a.last_at for a in agents if a.last_at is not None]
        starts = [a.started_at for a in agents if a.started_at is not None]
        return Fleet(
            fleet_id=fleet_id, kind=kind, session_id=session_id, path=path,
            agents=agents,
            started=len(journal) or len(agents),
            settled=len([v for v in journal.values() if v == "result"]),
            started_at=min(starts) if starts else None,
            last_at=max(stamps) if stamps else None)

    def _agent(self, path: Path, fleet_id: str, session_id: str, journal: dict,
               now: float, *, roots: Iterable[str] = (),
               tracked: Iterable[str] = ()) -> AgentRun | None:
        accumulator, _changed = self._absorb(path)
        if accumulator is None:
            return None
        agent_id = path.stem[len("agent-"):] if path.stem.startswith("agent-") \
            else path.stem
        agent_type = accumulator.agent_type or self._meta_type(path)
        status, evidence = _agent_status(accumulator.last_at, journal, agent_id,
                                         now)
        return AgentRun(
            agent_id=agent_id,
            agent_type=agent_type or UNATTRIBUTED,
            fleet_id=fleet_id,
            session_id=accumulator.session_id or session_id,
            path=str(path),
            models=accumulator.ranked_models(),
            started_at=accumulator.started_at,
            last_at=accumulator.last_at,
            turns=accumulator.turns,
            tool_counts=accumulator.ranked_tools(),
            files=tuple(sorted(accumulator.files)),
            scopes=_turn_scopes(accumulator.settled_scopes(),
                                roots=_roots_for(accumulator, roots),
                                tracked=tracked),
            brief=accumulator.brief,
            recent=tuple(accumulator.recent),
            input_tokens=accumulator.input_tokens,
            output_tokens=accumulator.output_tokens,
            thinking_blocks=accumulator.thinking_blocks,
            thinking_readable=accumulator.thinking_readable,
            cwd=accumulator.cwd,
            branch=accumulator.branch,
            status=status, status_evidence=evidence)

    def _journal(self, path: Path) -> dict:
        """A fleet's ledger, cached on (size, mtime) like every other file.

        A workflow run's journal is small, but a session can hold sixteen of
        them and the poll is every two seconds. Re-reading them all each pass
        was the one place the incremental design leaked, and a test now fails if
        it leaks again.
        """
        key = str(path)
        try:
            stat = path.stat()
        except OSError:
            return self._journals.get(key, (0, 0.0, {}))[2]
        cached = self._journals.get(key)
        if cached is not None and cached[0] == stat.st_size \
                and cached[1] == stat.st_mtime:
            return cached[2]
        outcomes = _read_journal(path)
        self._journals[key] = (stat.st_size, stat.st_mtime, outcomes)
        return outcomes

    @staticmethod
    def _meta_type(path: Path) -> str:
        meta = path.with_suffix(".meta.json")
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(payload.get("agentType") or "") if isinstance(payload, dict) \
            else ""

    # ── the incremental core ─────────────────────────────────────────────────
    def _absorb(self, path: Path, *,
                tail_bytes: int | None = None) -> tuple[_Accumulator | None, bool]:
        """Fold whatever is new in `path` into its accumulator.

        Returns (accumulator, changed). A file whose size and mtime are both
        unchanged is not opened, which is the entire performance argument for
        polling this at all.
        """
        key = str(path)
        try:
            stat = path.stat()
        except OSError:
            return self._state.get(key), False
        previous = self._cursors.get(key)
        if previous is not None:
            size, mtime, cursor = previous
            if size == stat.st_size and mtime == stat.st_mtime:
                return self._state.get(key), False
        else:
            cursor = 0

        records, new_cursor = _iter_records(path, start=cursor,
                                            tail_bytes=tail_bytes)
        accumulator = self._state.get(key)
        if accumulator is None:
            accumulator = _Accumulator()
            self._state[key] = accumulator
        for record in records:
            accumulator.absorb(record)
        self._cursors[key] = (stat.st_size, stat.st_mtime, new_cursor)
        return accumulator, bool(records)

    def forget(self) -> None:
        """Drop every cursor and every excerpt held in memory."""
        self._cursors.clear()
        self._state.clear()
        self._journals.clear()


def read_activity(cwd: str | Path | None = None, *,
                  records_root: str | Path | None = None,
                  max_sessions: int = 12,
                  now: float | None = None,
                  roots: Iterable[str] = (),
                  tracked: Iterable[str] = ()) -> Activity:
    """One-shot read. `ActivityIndex` is the one to hold for polling."""
    return ActivityIndex(records_root).read(cwd=cwd, max_sessions=max_sessions,
                                            now=now, roots=roots,
                                            tracked=tracked)


__all__ = [
    "ACTIVITY_SCHEMA", "Activity", "ActivityIndex", "AgentRun", "EXCERPT_CHARS",
    "FLEET_DIRECT", "FLEET_WORKFLOW", "Fleet", "LIMITS", "MessageBlocks",
    "QUIET", "RECENT_TURNS",
    "RUNNING", "RUNNING_MINUTES", "SETTLED", "STATUSES", "STOPPED", "SessionRun",
    "TAIL_BYTES", "Turn", "TurnScope", "message_blocks",
    "read_activity", "redact", "repo_paths_of",
    "repo_relative",
    "repo_root_of",
]
