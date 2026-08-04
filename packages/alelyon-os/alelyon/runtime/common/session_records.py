"""Derive which agent sessions are in a checkout, from records they did not write.

This closes the gap [DYNAMIC-CACHE.md](../../../docs/features/DYNAMIC-CACHE.md)
§9.3 leaves open. That section asks "whether any vendor writes a per-session
marker *inside* a shared checkout that could be derived instead of declared" and
says that, and only that, would move the participation protocol off the critical
path. The answer found here is *beside* the checkout rather than inside it, but
it answers the same question: **Claude Code writes one transcript file per
session, keyed to the directory the session started in, and the agent authors
neither the filename nor the `cwd` field.**

Why that is worth having
------------------------
`worktree.py` derives a session from the worktree *path*, which works because
some tools put a session id there. It deliberately exempts the primary checkout
([worktree.py](worktree.py) — "the repository's own checkout belongs to no
session"), and that exemption is correct: the primary checkout's path carries no
session id, so reading one out of it would attribute the owner's own tree to
whichever agent happened to look.

The cost is that the busiest directory in the repository has no derived identity
at all, and therefore no inbox — `tools/fleet.py inbox` refuses outright there.
Findings published about work in the primary checkout reach nobody, which has
already happened in this repository.

This module does not read the worktree path, so it does not disturb that rule. It
reads a record written by the harness, about the session, at a location the
session did not choose.

Where this sits on the evidence ladder
--------------------------------------
OBSERVED in the sense the rest of the mesh uses: the file exists, its name is a
session id, and its `cwd` field was written by Claude Code rather than by the
model running inside it. `CLAIMS.md` §2.3 asks that attribution be validated
against an independently-held invariant rather than against the shape of what the
writer emitted, and a transcript index the agent cannot address is such an
invariant.

It is **not** authentication, and three separate things stop it short:

* `cwd` is where a session **started**, not where it is editing. A session that
  launches in the primary checkout and then works in a worktree it created still
  records the primary checkout here. This *over-approximates* occupancy — it
  names more sessions than are really in the directory, never fewer.
* liveness is inferred from file modification time. A crashed session leaves a
  file that stops being written exactly like an idle one does.
* it is Claude Code's convention alone. Codex, Copilot, Cursor and Antigravity
  write nothing this module can read, and for them it returns an empty result
  rather than a guess.

An over-approximate answer is still the right trade here, and the asymmetry is
the reason: a mailbox that reaches you along with somebody else is a working
mailbox, while no mailbox reaches nobody. The over-approximation is stated at
every read so a caller cannot mistake it for occupancy.

Ambiguity is refused by name, not resolved by guessing
------------------------------------------------------
Where exactly one session's records point at a directory, that session is
derived. Where several do, `identify()` returns `UNATTRIBUTED` **and the
candidate set**, rather than picking the most recently written one — which would
be a coin flip between two sessions typing at the same moment, and would be wrong
silently.

What a caller does with the candidate set is the actual improvement.
`tools/fleet.py --session` previously accepted any string, so a session declared
an identity out of nothing. Validated against these records it can only *select*
one the harness already wrote, which is a closed vocabulary rather than free
text. That is still a declaration and is still labelled one — but a declaration
constrained by an independently-held record is a different and better thing than
a declaration constrained by nothing.

Privacy: this reads structure, never content
--------------------------------------------
A transcript is chat history, which `AGENTS.md` §9 names sensitive. The parser
extracts three keys — `sessionId`, `cwd`, `timestamp` — and discards every other
field without inspecting it, stopping as soon as it has what it needs. Message
content is never read into memory beyond the line buffer that carried it, never
returned, and never logged. That is redaction by construction rather than after
printing, per §4 rule 8, and `test_session_records.py` asserts it by feeding the
parser a record whose content would be recognisable in any output.

Read-only and side-effect free: opens files for reading, writes nothing, and
creates nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

#: Returned wherever a session could not be derived. A value, never a blank —
#: the same contract `worktree.UNATTRIBUTED` carries, and deliberately the same
#: spelling so a caller can compare the two without a translation table.
UNATTRIBUTED = "UNATTRIBUTED"

#: Where Claude Code keeps one directory per project and one transcript per
#: session. A parameter everywhere below rather than a constant read at call
#: time, so tests never depend on the developer's real home directory.
DEFAULT_RECORDS_ROOT = Path.home() / ".claude" / "projects"

#: Only these keys are read out of a transcript record. Everything else is
#: discarded unexamined — see the privacy note above.
_WANTED_KEYS = ("sessionId", "cwd", "timestamp")

#: How far into a transcript to look for the `cwd` field before giving up.
#:
#: Measured rather than guessed: across the 39 transcripts in this repository's
#: project directory on 2026-08-03, `cwd` appeared at line 0 or line 2 in every
#: single one, and the lines before it are queue bookkeeping that carries no
#: `cwd` at all. Fifty is far enough above the observed maximum to absorb a
#: format change and far below the size of these files — the largest is 19 MB,
#: and reading one to its end to answer "which directory" would be absurd.
MAX_HEAD_LINES = 50

#: A line longer than this is not a record this module can use, and reading it
#: whole would pull an arbitrary amount of conversation into memory. Skipped.
MAX_LINE_BYTES = 1 << 20

#: Minutes since a transcript was last written, within which its session is
#: treated as live. Not a proof of liveness and not presented as one — see
#: `LIMITS`. Thirty minutes is long enough to span a model's thinking time and
#: an operator reading output, and short enough that yesterday's sessions do not
#: crowd the answer.
DEFAULT_ACTIVE_MINUTES = 30.0

LIMITS: tuple[str, ...] = (
    "This derives where a session STARTED, not where it is editing. A session "
    "that launched in this checkout and then worked in a worktree it created is "
    "still counted here, so the answer over-approximates occupancy: it names "
    "more sessions than are really in the directory, never fewer.",
    "Liveness is inferred from a file's modification time. A crashed session "
    "leaves a transcript that stopped being written exactly like an idle one, "
    "so 'active' means 'written to recently' and nothing stronger.",
    "This is Claude Code's convention alone. Codex, Copilot, Cursor and "
    "Antigravity write no record this can read, and sessions from those tools "
    "are absent rather than reported as UNATTRIBUTED - absent from this answer "
    "is not absent from the repository.",
    "A transcript index is not authentication. It is a record the agent did not "
    "author, which is better evidence than a self-report and is still not proof "
    "that the session it names is the one asking.",
    "Where more than one session points at a directory, no identity is derived. "
    "The candidate set is returned instead, because choosing the most recently "
    "written one is a coin flip between two sessions working at the same moment.",
)


@dataclass(frozen=True)
class SessionRecord:
    """One agent session, as the harness recorded it rather than as it claims."""

    session_id: str
    #: The directory the session started in, verbatim from the record. Windows
    #: drive-letter case varies between records for one directory, so compare
    #: with `same_directory()` and never with `==`.
    cwd: str
    tool_family: str
    #: Unix seconds the transcript was last written, or None when unreadable.
    #: Never 0, which would read as 1970 rather than as unknown.
    last_written_at: int | None
    #: The rule that produced this record, so a reader can disagree with it.
    evidence: str

    def idle_minutes(self, now: float | None = None) -> float | None:
        """Minutes since the transcript was last written, or None if unknown."""
        if self.last_written_at is None:
            return None
        moment = now if now is not None else time.time()
        return max(0.0, (moment - self.last_written_at) / 60.0)

    def active(self, *, within_minutes: float = DEFAULT_ACTIVE_MINUTES,
               now: float | None = None) -> bool:
        """Whether the transcript was written recently enough to look live.

        An unknown write time is NOT active: "cannot tell" and "idle" are
        different answers, and only one of them is this one.
        """
        idle = self.idle_minutes(now)
        return idle is not None and idle <= within_minutes


def same_directory(left: str, right: str) -> bool:
    """Whether two path strings name one directory.

    Case-insensitively on Windows, and after normalising separators, because the
    harness records the drive letter in whichever case the process was launched
    with. Both `c:\\Repos\\example` and `C:\\Repos\\example` occur for one
    directory in real records, so a `==` comparison silently halves the answer.

    The example is deliberately synthetic. This module ships in the public
    `alelyon-os` wheel, and the real path this was written from named a
    developer's home directory — a docstring is published source, so an
    illustrative path is someone's username unless it is invented.
    """
    if not left or not right:
        return False
    try:
        lhs, rhs = Path(left).resolve(), Path(right).resolve()
    except OSError:
        lhs, rhs = Path(left), Path(right)
    left_text, right_text = str(lhs), str(rhs)
    if os.name == "nt" or left_text[:1].isalpha() and left_text[1:2] == ":":
        return left_text.casefold() == right_text.casefold()
    return left_text == right_text


def _read_head(path: Path) -> dict:
    """The structural fields of a transcript, from a bounded head read.

    Returns only `_WANTED_KEYS`. Every other field of every record is discarded
    without being examined, and the read stops as soon as a `cwd` is found.
    """
    found: dict = {}
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= MAX_HEAD_LINES:
                    break
                if len(line) > MAX_LINE_BYTES:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(record, dict):
                    continue
                for key in _WANTED_KEYS:
                    value = record.get(key)
                    if value and key not in found:
                        found[key] = value
                if "cwd" in found:
                    break
    except OSError:
        return {}
    return found


def _records_in_dir(directory: Path) -> list[SessionRecord]:
    out: list[SessionRecord] = []
    try:
        entries = sorted(directory.glob("*.jsonl"))
    except OSError:
        return out
    for entry in entries:
        head = _read_head(entry)
        cwd = head.get("cwd")
        if not cwd:
            continue
        # The filename stem is a session id and so is the `sessionId` field.
        # They agreed in all 39 transcripts measured, so a disagreement is a
        # format change rather than a routine case, and the filename wins: it is
        # the one the harness uses to address the file.
        stem = entry.stem
        declared = head.get("sessionId")
        note = ("" if not declared or declared == stem
                else f"; record names {declared!r} but the file is {stem!r}")
        try:
            written = int(entry.stat().st_mtime)
        except OSError:
            written = None
        out.append(SessionRecord(
            session_id=stem,
            cwd=str(cwd),
            tool_family="claude-code",
            last_written_at=written,
            evidence=("Claude Code transcript index; the harness wrote the "
                      "filename and the cwd field, not the agent" + note),
        ))
    return out


def sessions_in(directory: str | Path, *,
                records_root: str | Path | None = None,
                active_within_minutes: float | None = DEFAULT_ACTIVE_MINUTES,
                now: float | None = None) -> tuple[SessionRecord, ...]:
    """Sessions whose harness record points at `directory`, most recent first.

    Scans every project directory rather than computing the one whose name
    encodes this path. The encoding replaces every non-alphanumeric character
    with a dash, which is lossy — `\\.claude\\` and `--claude-` are
    indistinguishable afterwards — and it preserves the drive letter's case, so
    one directory can own two differently-named project folders. Reading the
    `cwd` each transcript states avoids reconstructing a name that was never
    invertible.

    `active_within_minutes=None` returns every session ever recorded for the
    directory rather than only the live ones.
    """
    root = Path(records_root) if records_root is not None else DEFAULT_RECORDS_ROOT
    target = str(directory)
    try:
        project_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    except OSError:
        return ()

    matches: list[SessionRecord] = []
    for project in project_dirs:
        for record in _records_in_dir(project):
            if not same_directory(record.cwd, target):
                continue
            if active_within_minutes is not None and not record.active(
                    within_minutes=active_within_minutes, now=now):
                continue
            matches.append(record)

    # Most recently written first; an unknown write time sorts last rather than
    # sorting as 1970 and displacing a real answer.
    matches.sort(key=lambda r: (r.last_written_at is None,
                                -(r.last_written_at or 0), r.session_id))
    return tuple(matches)


def identify(directory: str | Path, *,
             records_root: str | Path | None = None,
             active_within_minutes: float | None = DEFAULT_ACTIVE_MINUTES,
             now: float | None = None) -> tuple[str, str]:
    """(session, evidence) for a directory, or UNATTRIBUTED and why not.

    Derives an identity only where exactly one session is live in the directory.
    Two live sessions is not a tie to break — it is the case this whole module
    exists to make visible — so the answer names both and derives neither.
    """
    found = sessions_in(directory, records_root=records_root,
                        active_within_minutes=active_within_minutes, now=now)
    if not found:
        return UNATTRIBUTED, ("no Claude Code session record points at this "
                              "directory; another tool leaves no record here")
    if len(found) == 1:
        only = found[0]
        return only.session_id, only.evidence
    names = ", ".join(record.session_id for record in found)
    return UNATTRIBUTED, (
        f"{len(found)} sessions are live in this directory ({names}); the "
        f"harness records cannot say which one is asking, and picking the most "
        f"recent would be a guess between sessions working at the same moment")


def candidates(directory: str | Path, *,
               records_root: str | Path | None = None,
               active_within_minutes: float | None = DEFAULT_ACTIVE_MINUTES,
               now: float | None = None) -> tuple[str, ...]:
    """The session ids a declaration for this directory may select from.

    A caller that must declare an identity should validate it against this
    closed vocabulary. The result is still a declaration and must still be
    labelled one — but it is a declaration the agent can no longer invent, which
    is the difference between self-reporting and selecting.
    """
    return tuple(record.session_id for record in sessions_in(
        directory, records_root=records_root,
        active_within_minutes=active_within_minutes, now=now))
