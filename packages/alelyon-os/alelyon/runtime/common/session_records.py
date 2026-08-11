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
extracts four top-level keys — `type`, `sessionId`, `cwd`, `timestamp` — and
stops before a later object or array can lead into message content. Only those
bounded structural scalar tokens reach JSON decoding; content is never returned
or logged. That is redaction by construction rather than after printing, per
§4 rule 8, and `test_session_records.py` asserts it with decoder canaries.

Read-only and side-effect free: opens files for reading, writes nothing, and
creates nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import ntpath
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
_WANTED_KEYS = ("type", "sessionId", "cwd", "timestamp")
_WANTED_KEY_SET = frozenset(_WANTED_KEYS)

#: What a record must carry to be USABLE, as opposed to what this reader would
#: like to have. The two are not the same set, and conflating them is what made
#: a record without a `timestamp` weigh exactly as much as no record at all.
#:
#: `timestamp` is the repository-incarnation privacy boundary for the
#: content-bearing activity reader. Identity does not need it: `_records_in_dir`
#: requires only `cwd`, takes the session id from the filename, and already
#: passes `head.get("timestamp")` through `_timestamp_seconds`, which answers
#: None for an absent one. So an absent timestamp was always modelled downstream
#: as NOT ESTABLISHED -- it was only the parser that treated it as disqualifying,
#: and it discarded the whole record rather than the one field it was missing.
_REQUIRED_KEYS = frozenset({"type", "sessionId", "cwd"})
_STRUCTURAL_VALUE_BYTES = 8192

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
    #: Original transcript timestamp, used only to refuse records older than a
    #: selected repository incarnation. It is structural metadata from the
    #: bounded head read, never message content.
    original_at: int | None = None

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


def project_slug_spellings(directory: str | Path) -> tuple[str, ...]:
    """Finite folder spellings Claude Code may use for one directory.

    The harness preserves drive-letter case and sometimes records a literal
    spelling after the path has disappeared.  Those alternatives are derived
    from the selected directory itself; callers never need to enumerate the
    projects store to discover them.
    """
    raw = str(directory)
    candidates = [project_slug(directory), _NON_ALNUM.sub("-", raw)]
    if len(raw) >= 2 and raw[0].isalpha() and raw[1] == ":":
        for drive in (raw[0].lower(), raw[0].upper()):
            candidates.append(_NON_ALNUM.sub("-", drive + raw[1:]))
    return tuple(dict.fromkeys(value for value in candidates if value))


def _project_directories(root: Path, directory: str | Path) -> tuple[Path, ...]:
    """Existing directly-addressed project folders for ``directory``."""
    out: list[Path] = []
    seen: set[str] = set()
    for slug in project_slug_spellings(directory):
        candidate = root / slug
        try:
            if not candidate.is_dir():
                continue
            key = os.path.normcase(str(candidate.resolve(strict=False)))
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return tuple(out)


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
    original_left, original_right = str(left), str(right)
    try:
        lhs, rhs = Path(left).resolve(), Path(right).resolve()
    except OSError:
        lhs, rhs = Path(left), Path(right)
    left_text, right_text = str(lhs), str(rhs)
    windows_like = (os.name == "nt" or "\\" in original_left
                    or "\\" in original_right
                    or (original_left[:1].isalpha()
                        and original_left[1:2] == ":")
                    or (original_right[:1].isalpha()
                        and original_right[1:2] == ":"))
    if windows_like:
        def normalized(value: str) -> str:
            text = value.replace("/", "\\")
            folded = text.casefold()
            if folded.startswith("\\\\?\\unc\\"):
                text = "\\\\" + text[8:]
            elif folded.startswith("\\\\?\\"):
                text = text[4:]
            return ntpath.normpath(text).casefold()

        return normalized(left_text) == normalized(right_text)
    return left_text == right_text


def directory_key(value: object) -> str:
    """Closed normalized equality key matching :func:`same_directory`."""
    text = str(value or "")
    if not text or "\x00" in text or "\n" in text or "\r" in text:
        return ""
    try:
        resolved = str(Path(text).resolve(strict=False))
    except (OSError, ValueError):
        resolved = text
    windows_like = (os.name == "nt" or "\\" in text
                    or (text[:1].isalpha() and text[1:2] == ":"))
    if not windows_like:
        return resolved
    normalized = resolved.replace("/", "\\")
    folded = normalized.casefold()
    if folded.startswith("\\\\?\\unc\\"):
        normalized = "\\\\" + normalized[8:]
    elif folded.startswith("\\\\?\\"):
        normalized = normalized[4:]
    return ntpath.normpath(normalized).casefold()


def _string_end(raw: bytes, start: int) -> int | None:
    if start >= len(raw) or raw[start] != ord('"'):
        return None
    escaped = False
    for position in range(start + 1, len(raw)):
        byte = raw[position]
        if escaped:
            escaped = False
        elif byte == ord("\\"):
            escaped = True
        elif byte == ord('"'):
            return position + 1
    return None


def _space(raw: bytes, position: int) -> int:
    while position < len(raw) and raw[position] in b" \t\r\n":
        position += 1
    return position


def _scalar_end(raw: bytes, start: int) -> int | None:
    if start >= len(raw) or raw[start] in b"{[":
        return None
    if raw[start] == ord('"'):
        return _string_end(raw, start)
    position = start
    while position < len(raw) and raw[position] not in b",}":
        position += 1
    return position if position > start else None


#: The only bytes that can change composite-skipping state. Everything else in
#: a transcript — which is almost all of it, since the composite being skipped
#: is usually a message body — is stepped over inside `re`, in C, rather than
#: one byte at a time in Python. A message composite here runs to 145 KB.
_STRUCTURAL_BYTES = re.compile(rb'["\\{}\[\]]')


def _composite_end(raw: bytes, start: int) -> int | None:
    """End of one JSON object/array, found LEXICALLY without decoding it.

    This is what lets an unrelated composite be stepped over rather than
    traversed. Nothing inside it is decoded, kept, or looked at for structural
    fields: the scan counts brackets, respecting strings and their escapes, and
    returns the offset just past the close. A record whose composite never
    closes on this line is refused, because guessing where it ended is how a
    parser starts reading content it was told not to read.

    It jumps between structurally significant bytes instead of visiting each
    one. A naive per-byte loop made the head read 5.6x slower than the
    `json.loads` reader this parser replaced, and put ~49% of a 2s poll inside
    this function, because the composite it skips is the message body and the
    body is nearly all ordinary text. Same answer, same refusals; the states
    that matter are only ever reached at one of six bytes.
    """
    if start >= len(raw) or raw[start] not in b"{[":
        return None
    stack: list[int] = []
    quoted = False
    escaped_at = -1
    pairs = {ord("}"): ord("{"), ord("]"): ord("[")}
    for match in _STRUCTURAL_BYTES.finditer(raw, start):
        position = match.start()
        if position == escaped_at:
            # The byte after a backslash inside a string is literal, whatever
            # it is. Skipping it here is what keeps \\" from closing a string
            # and \\\\ from escaping the quote that follows it.
            continue
        byte = raw[position]
        if quoted:
            if byte == ord("\\"):
                escaped_at = position + 1
            elif byte == ord('"'):
                quoted = False
            continue
        if byte == ord('"'):
            quoted = True
        elif byte in b"{[":
            stack.append(byte)
        elif byte in b"}]":
            if not stack or stack[-1] != pairs.get(byte):
                return None
            stack.pop()
            if not stack:
                return position + 1
        # A backslash OUTSIDE a string is an ordinary byte and means nothing
        # here. It is matched only because the same pattern has to catch it
        # inside one; falling through to the close-bracket branch would refuse
        # records this has always accepted. A fuzz differential against the
        # per-byte reference caught exactly that, on inputs like b'{\\}'.
    return None


def _decode_scalar(raw: bytes) -> object | None:
    if len(raw) > _STRUCTURAL_VALUE_BYTES:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


def _structural_record(raw: bytes) -> dict:
    """Extract one record's top-level structural prefix only.

    An object/array belonging to an unrelated key is STEPPED OVER lexically by
    `_composite_end` and never traversed: no byte inside it is decoded or
    examined for structural fields. The privacy property that motivated this
    parser is therefore intact — message content is still never read — while
    the four wanted keys remain findable wherever the writer put them.

    Refusing such a row outright, which is what this did before, made the
    parser depend on key ORDER. The harness serialises `message` at key #4 and
    `cwd` at #12, so every content-bearing record was refused and `_read_head`
    returned {} for every transcript on disk. A composite belonging to an
    unrelated key is not evidence about the record; it is a field this reader
    has no business in, and stepping over it is the whole fix.

    A composite that does not close on this line is still refused: guessing
    where it ended is how a parser starts reading what it was told not to.

    Completeness is judged against `_REQUIRED_KEYS`, not `_WANTED_KEYS`. The
    early return below still fires only when all four are in hand, because that
    is an optimisation — there is nothing left to look for — but reaching the
    end of a record with three of them is a record, not a failure. Demanding
    the fourth discarded any row the writer left a `timestamp` off, and it
    discarded the whole row rather than the field, so a session that was
    perfectly identifiable became invisible to `candidates()` and every caller
    downstream of it refused for want of an id it actually had.
    """
    position = _space(raw, 0)
    if position >= len(raw) or raw[position] != ord("{"):
        return {}
    position += 1
    found: dict = {}
    while position < len(raw):
        position = _space(raw, position)
        if position < len(raw) and raw[position] == ord("}"):
            return found if _REQUIRED_KEYS.issubset(found) else {}
        key_end = _string_end(raw, position)
        if key_end is None or key_end - position > 256:
            return {}
        key = _decode_scalar(raw[position:key_end])
        if not isinstance(key, str):
            return {}
        position = _space(raw, key_end)
        if position >= len(raw) or raw[position] != ord(":"):
            return {}
        position = _space(raw, position + 1)
        if key not in _WANTED_KEY_SET and position < len(raw) \
                and raw[position] in b"{[":
            # Not ours: step over it without decoding a byte of it. A wanted
            # key holding a composite is still refused below, because a `cwd`
            # that is an object is malformed rather than merely uninteresting.
            value_end = _composite_end(raw, position)
        else:
            value_end = _scalar_end(raw, position)
        if value_end is None:
            return {}
        if key in _WANTED_KEY_SET and key not in found:
            value = _decode_scalar(raw[position:value_end])
            if value is None:
                return {}
            found[key] = value
            if _WANTED_KEY_SET.issubset(found):
                return found
        position = _space(raw, value_end)
        if position >= len(raw):
            return {}
        if raw[position] == ord(","):
            position += 1
            continue
        if raw[position] == ord("}"):
            return found if _REQUIRED_KEYS.issubset(found) else {}
        return {}
    return {}


def _read_head(path: Path) -> dict:
    """The structural fields of a transcript, from a bounded head read.

    Returns only `_WANTED_KEYS`. The read stops once all four structural keys
    are found in one top-level prefix, before any later composite value. The
    timestamp is the repository-incarnation privacy boundary used by the
    content-bearing activity reader.
    """
    try:
        with path.open("rb") as handle:
            for _index in range(MAX_HEAD_LINES):
                line = handle.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_LINE_BYTES:
                    while line and not line.endswith(b"\n"):
                        line = handle.readline(MAX_LINE_BYTES + 1)
                    continue
                found = _structural_record(line)
                if found:
                    return found
    except OSError:
        return {}
    return {}


def _timestamp_seconds(value) -> int | None:
    """Parse one structural transcript timestamp without coercive guessing."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value >= 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        parsed = datetime.fromisoformat(
            raw[:-1] + "+00:00" if raw.endswith("Z") else raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


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
            original_at=_timestamp_seconds(head.get("timestamp")),
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
                now: float | None = None,
                not_before: int | None = None,
                exact_cwd: bool = False) -> tuple[SessionRecord, ...]:
    """Sessions whose harness record points at `directory`, most recent first.

    Directly probes the finite resolved, literal, and drive-case encodings of
    this selected directory. The encoding is lossy, so transcripts within an
    addressed collision folder are still resolved by their exact structural
    `cwd`; unrelated project folders are never enumerated or opened.

    `active_within_minutes=None` returns every session ever recorded for the
    directory rather than only the live ones.
    """
    root = Path(records_root) if records_root is not None else DEFAULT_RECORDS_ROOT
    target = str(directory)
    project_dirs = _project_directories(root, target)

    matches: list[SessionRecord] = []
    for project in project_dirs:
        for record in _records_in_dir(project):
            if exact_cwd:
                counts, rule = same_directory(record.cwd, target), record.evidence
            else:
                counts, rule = _member(record, target)
            if not counts:
                continue
            if not_before is not None and (
                    record.original_at is None
                    or record.original_at < int(not_before)):
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
             now: float | None = None,
             not_before: int | None = None,
             exact_cwd: bool = False) -> tuple[str, str]:
    """(session, evidence) for a directory, or UNATTRIBUTED and why not.

    Derives an identity only where exactly one session is live in the directory.
    Two live sessions is not a tie to break — it is the case this whole module
    exists to make visible — so the answer names both and derives neither.
    """
    found = sessions_in(directory, records_root=records_root,
                        active_within_minutes=active_within_minutes, now=now,
                        not_before=not_before, exact_cwd=exact_cwd)
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
               now: float | None = None,
               not_before: int | None = None,
               exact_cwd: bool = False) -> tuple[str, ...]:
    """The session ids a declaration for this directory may select from.

    A caller that must declare an identity should validate it against this
    closed vocabulary. The result is still a declaration and must still be
    labelled one — but it is a declaration the agent can no longer invent, which
    is the difference between self-reporting and selecting.
    """
    return tuple(record.session_id for record in sessions_in(
        directory, records_root=records_root,
        active_within_minutes=active_within_minutes, now=now,
        not_before=not_before, exact_cwd=exact_cwd))
