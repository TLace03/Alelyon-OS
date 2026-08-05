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
)


def redact(text: str, limit: int = 220) -> str:
    """Scrub the common secret shapes and cap the length."""
    cleaned = _SECRET.sub("[REDACTED]", str(text or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned if len(cleaned) <= limit else cleaned[:limit - 1] + "…"


@dataclass(frozen=True)
class Turn:
    """One turn, as the harness recorded it.

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

    @property
    def has_content(self) -> bool:
        return bool(self.text or self.thinking)


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
    #: The instruction it was given, truncated. Written by whatever spawned it.
    brief: str = ""
    recent: tuple[Turn, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
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
    input_tokens: int = 0
    output_tokens: int = 0
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


def _blocks(message: dict) -> tuple[str, str, list[str], list[str]]:
    """(text, thinking, tools, targets) out of one message's content.

    Tool *results* are skipped by name rather than by omission: they are the
    field this module must not read, and a reader of this function should see
    the decision rather than infer it from what is missing.
    """
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    tools: list[str] = []
    targets: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        return redact(content, EXCERPT_CHARS), "", [], []
    if not isinstance(content, list):
        return "", "", [], []
    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            text_parts.append(str(block.get("text") or ""))
        elif kind == "thinking":
            thinking_parts.append(str(block.get("thinking")
                                      or block.get("text") or ""))
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
            tools, targets)


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


@dataclass
class _Accumulator:
    """Mutable running totals for one transcript, kept across incremental reads."""

    models: dict[str, int] = field(default_factory=dict)
    tools: dict[str, int] = field(default_factory=dict)
    files: set = field(default_factory=set)
    turns: int = 0
    started_at: int | None = None
    last_at: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    recent: list = field(default_factory=list)
    brief: str = ""
    cwd: str = ""
    session_id: str = ""
    branch: str = ""
    agent_type: str = ""

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
        text, thinking, tools, targets = _blocks(message)
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
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        if model:
            self.models[model] = self.models.get(model, 0) + 1
        for name in tools:
            self.tools[name] = self.tools.get(name, 0) + 1
        for block in (message.get("content") or []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                for path in _files_in(block.get("input")):
                    self.files.add(path)

        if kind == _ROLE_ASSISTANT:
            self.turns += 1
        if text or thinking or tools:
            self.recent.append(Turn(
                at=at or 0, role=str(kind), model=model or UNATTRIBUTED,
                text=text, thinking=thinking, tools=tuple(tools),
                targets=tuple(targets), input_tokens=input_tokens,
                output_tokens=output_tokens))
            if len(self.recent) > RECENT_TURNS:
                del self.recent[:-RECENT_TURNS]

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
             max_sessions: int = 12, now: float | None = None) -> Activity:
        """Read every session for `cwd`, most recently written first."""
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
            session = self._session(transcript, moment)
            if session is None:
                continue
            if cwd is not None and not same_directory(session.cwd, str(cwd)):
                continue
            sessions.append(session)

        return Activity(schema=ACTIVITY_SCHEMA, read_at=int(moment),
                        records_root=str(self.records_root),
                        sessions=tuple(sessions), notes=tuple(notes))

    # ── one session ──────────────────────────────────────────────────────────
    def _session(self, transcript: Path, now: float) -> SessionRun | None:
        session_id = transcript.stem
        accumulator, changed = self._absorb(transcript, tail_bytes=TAIL_BYTES)
        if accumulator is None:
            return None
        fleets = self._fleets(transcript.with_suffix("") , session_id, now)
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
            input_tokens=accumulator.input_tokens,
            output_tokens=accumulator.output_tokens,
            notes=() if changed else (),
        )

    def _fleets(self, session_dir: Path, session_id: str,
                now: float) -> tuple[Fleet, ...]:
        subagents = session_dir / "subagents"
        if not subagents.is_dir():
            return ()
        fleets: list[Fleet] = []

        direct = sorted(subagents.glob("agent-*.jsonl"))
        if direct:
            agents = tuple(self._agent(p, FLEET_DIRECT, session_id, {}, now)
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
                    self._agent(p, run.name, session_id, journal, now)
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
               now: float) -> AgentRun | None:
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
            brief=accumulator.brief,
            recent=tuple(accumulator.recent),
            input_tokens=accumulator.input_tokens,
            output_tokens=accumulator.output_tokens,
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
                  now: float | None = None) -> Activity:
    """One-shot read. `ActivityIndex` is the one to hold for polling."""
    return ActivityIndex(records_root).read(cwd=cwd, max_sessions=max_sessions,
                                            now=now)


__all__ = [
    "ACTIVITY_SCHEMA", "Activity", "ActivityIndex", "AgentRun", "EXCERPT_CHARS",
    "FLEET_DIRECT", "FLEET_WORKFLOW", "Fleet", "LIMITS", "QUIET", "RECENT_TURNS",
    "RUNNING", "RUNNING_MINUTES", "SETTLED", "STATUSES", "STOPPED", "SessionRun",
    "Turn", "read_activity", "redact", "repo_paths_of", "repo_relative",
    "repo_root_of",
]
