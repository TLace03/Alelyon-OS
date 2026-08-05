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

* the `cwd` a record states is **one reading of a field that changes**, and this
  module reads the first one it finds. See "Two answers" below — the honest
  summary is that a stated `cwd` over-approximates and a *head-read* one can
  also under-approximate, which is why it is no longer the only signal.
* liveness is inferred from file modification time. A crashed session leaves a
  file that stops being written exactly like an idle one does.
* it is Claude Code's convention alone. Codex, Copilot, Cursor and Antigravity
  write nothing this module can read, and for them it returns an empty result
  rather than a guess.

An over-approximate answer is still the right trade here, and the asymmetry is
the reason: a mailbox that reaches you along with somebody else is a working
mailbox, while no mailbox reaches nobody. The over-approximation is stated at
every read so a caller cannot mistake it for occupancy.

Two answers, because the harness writes two things down
-------------------------------------------------------
A transcript states a `cwd`, and the harness also *files* the transcript in a
project directory whose name is that directory with every non-alphanumeric
character replaced by a dash. Those are different facts, and this module used
only the first.

The `cwd` field is written on **every record, not once**, and it changes as the
session's working directory changes. Reading only the head therefore does not
report "where the session started" as a deliberate simplification — it reports
*the oldest surviving reading of a moving value*, which is a different and worse
thing. Measured on this machine at `9533bde`: **4 of 9 project directories name
a directory that no transcript inside them states a `cwd` for.** All four state
the primary checkout, because all four sessions were relocated into a worktree
after their first record was written.

The consequence was not an over-approximation. It was a **hole**: each of those
worktrees derived *zero* sessions, so `tools/fleet.py inbox` refused there, and
`--session` fell back to free text in exactly the directories where a closed
vocabulary was wanted. The claim this module used to make — "names more sessions
than are really in the directory, never fewer" — was false for the directory a
session moved *to*. It held only for the directory it moved *from*.

So filing is read as well, and the rule where they disagree is:

    the directory the harness FILED the transcript under is where the session
    is; the `cwd` a record states is where it has been.

Filing wins on disagreement because it is the harness's current answer while the
head `cwd` is its stalest one. Where filing and the stated `cwd` agree — every
session that never moved — nothing changes at all, which is why this is additive
rather than a reinterpretation of existing records.

The slug is **not invertible**: a directory named `.claude` and one named
`-claude` beside it collapse to the same dashes, so a directory name cannot be
turned back into a path. It does not
need to be. Encoding is total even where decoding is ambiguous, so the target
directory is encoded and the two names compared — and because every separator
maps to a dash, the comparison is indifferent to `/` versus `\\`. Two genuinely
different directories *can* collide onto one slug; that is an over-approximation
of the kind named above, is labelled in the evidence string, and is preferred to
the hole it replaces.

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

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
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

#: Every character Claude Code replaces with a dash when it names the project
#: directory a transcript is filed in. Deliberately the whole complement of
#: `[A-Za-z0-9]` rather than a list of separators: the drive colon, the path
#: separator, the dot of `.claude` and the dashes already in a directory name
#: are all mapped the same way, which is exactly why the result cannot be
#: decoded back into a path.
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]")

#: Minutes since a transcript was last written, within which its session is
#: treated as live. Not a proof of liveness and not presented as one — see
#: `LIMITS`. Thirty minutes is long enough to span a model's thinking time and
#: an operator reading output, and short enough that yesterday's sessions do not
#: crowd the answer.
DEFAULT_ACTIVE_MINUTES = 30.0

LIMITS: tuple[str, ...] = (
    "A session is counted for the directory its transcript is FILED under, and "
    "for any directory a record in it states as the working directory. Both "
    "over-approximate occupancy - a session that started here and moved on is "
    "still named here, and two directories can collide onto one project-folder "
    "name - so this names more sessions than are really in the directory. It "
    "is not a claim that they are editing it.",
    "The directory a transcript is filed under is read as the session's "
    "current location and the cwd it states as a place it has been, because "
    "the cwd field is written on every record and only the first is read. "
    "Where the two disagree the filing wins, which is a rule about which "
    "reading is FRESHER and not about which is true.",
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
    #: Name of the project directory the harness filed this transcript in — a
    #: directory path with every non-alphanumeric character replaced by a dash.
    #: Not a path and not convertible to one; compare with `filed_under()`.
    #: Empty for a record built by a caller rather than read off disk, which is
    #: why the filing rule never fires on one.
    filed_under: str = ""

    @property
    def moved(self) -> bool:
        """Whether the harness filed this somewhere the record does not state.

        True exactly for a session that was relocated after its first record was
        written — the case that used to leave a worktree with no derived session
        at all. False where nothing is filed and False where the two agree, so
        it is never true merely because the answer is unknown.
        """
        if not self.filed_under or not self.cwd:
            return False
        return not filed_under(self.cwd, self.filed_under)

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


def project_slug(directory: str | Path) -> str:
    """A directory as Claude Code names the project folder it files it under.

    Every non-alphanumeric character becomes a dash, so this is **lossy and one
    way**: `c:\\repo\\.claude` and `c:\\repo\\-claude` produce one name and
    cannot be told apart afterwards. That is why nothing here tries to turn a
    folder name back into a path, and why a caller compares encodings instead.

    Separators do not survive, which makes the comparison indifferent to `/`
    versus `\\` for free rather than by normalising first.
    """
    if not directory:
        return ""
    try:
        text = str(Path(directory).resolve())
    except OSError:
        text = str(directory)
    return _NON_ALNUM.sub("-", text)


def filed_under(directory: str | Path, project: str) -> bool:
    """Whether `project` is the folder name the harness would file `directory` in.

    Case-insensitive for the same reason `same_directory` is: the drive letter
    is recorded in whichever case the process was launched with, and it survives
    into the folder name. Both the resolved and the literal spelling are tried,
    because a directory that no longer exists cannot be resolved and must still
    be comparable.
    """
    if not directory or not project:
        return False
    wanted = project.casefold()
    if project_slug(directory).casefold() == wanted:
        return True
    return _NON_ALNUM.sub("-", str(directory)).casefold() == wanted


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
            filed_under=directory.name,
        ))
    return out


#: Why a record was counted for a directory. Kept apart because they are not
#: equally strong and a reader must be able to tell which one fired.
_BY_FILING = ("Claude Code filed this session's transcript in the project "
              "folder for this directory, which is where the harness "
              "currently considers the session to be working; the harness "
              "chose that folder name, not the agent")
_BY_MOVE = ("Claude Code filed this session's transcript under a DIFFERENT "
            "directory, so the cwd this record states is a place the session "
            "has been and not where it is; the harness wrote both, not the "
            "agent")


def _member(record: SessionRecord, target: str) -> tuple[bool, str]:
    """Whether `record` counts for `target`, and the rule that decided it.

    Three cases, and the third is the one this function exists for:

    * the transcript is filed under `target` — counted, on the harness's own
      current answer;
    * it is filed elsewhere, or nowhere, and *agrees* with the `cwd` it states —
      the original rule, unchanged;
    * it is filed elsewhere and DISAGREES with the `cwd` it states — the session
      moved, so the stated `cwd` no longer places it. Without this the session
      is counted for a directory it left and for none it went to.
    """
    if record.filed_under and filed_under(target, record.filed_under):
        return True, _BY_FILING
    if record.moved:
        return False, _BY_MOVE
    return same_directory(record.cwd, target), record.evidence


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
            counts, rule = _member(record, target)
            if not counts:
                continue
            if active_within_minutes is not None and not record.active(
                    within_minutes=active_within_minutes, now=now):
                continue
            # The evidence travels with the record, because two records in one
            # answer can have been placed here by different rules.
            matches.append(record if rule == record.evidence
                           else replace(record, evidence=rule))

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
