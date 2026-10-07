"""Persistent analyst-chat threads — the conversation survives the process.

Until now the AI Analyst was a `QTextEdit` that died with the app. Ask it
something on Monday, and on Tuesday there is no record that you asked, what it
answered, or — the part that matters here — **which figures it was shown**.

That last point is why this stores far more than prose. Every assistant turn
persists its `facts`: the tool results the answer was built from, each with its
own source and as-of stamp. A saved conversation that kept only the text would
be a transcript of assertions; keeping the facts means a thread re-opened in
three months still shows *what the number was and when it was true*. If a figure
was quoted from a Tuesday close, the record says so forever.

Layout — `globals/analyst_chat/`:
    index.json          thread list (id, title, timestamps, turn count)
    <id>.jsonl          one turn per line, append-only
    evicted/index.json  the threads the cap moved out, each with `evicted_at`
                        and `file`
    evicted/<file>      their transcripts, as they were

A directory rather than one file, so a torn write in one thread cannot take the
others down with it, and deleting a thread is an unlink rather than a rewrite of
everybody's history.

**The cap archives; only the reader's Delete deletes.** The list holds at most
`MAX_THREADS` threads. When one more would make it longer, the least recently
used thread leaves the list, and its transcript is MOVED into `evicted/`, never
unlinked, under the same lock that writes the index. Nothing here reads
`evicted/` back: loading, listing and the capped index ignore it. (Until
2026-10-02 the oldest thread's file was unlinked, and the only record of a
conversation was gone the moment a new one was started.) `delete_thread` is the
reader asking for a thread to be gone, and is a real delete.

The archive is written so that it never disagrees with the disk after a failed
write: an archive row exists only for a transcript that was moved (or for a
thread that had none), a move that cannot be completed is taken back, and so is
a move whose thread-index write fails. A crash in the middle can still leave a
row whose file has not moved yet; the next attempt completes it, and a Delete of
that thread drops it.

**Reading `evicted/index.json`.** A reader must resolve a transcript only
through its row's `file`, never by building `<id>.jsonl` from the id:
  * `file` is the name inside `evicted/`. It is `<id>.jsonl`, or `<id>-2.jsonl`
    and so on when an earlier archive already holds that name, so ids repeat
    across rows and `(id, created)` is what identifies a conversation.
  * `file` is "" for a thread that never had a turn: the row is its whole record.
  * A row whose file is missing is tolerated, not an error: it is a crash's
    leftover, or a transcript the reader removed from the directory by hand.
  * An archive index that cannot be read is moved aside, byte for byte, as
    `evicted/index.corrupt-<time>.json`, and archiving continues in a fresh
    `index.json`. The transcripts are untouched, so a reader can still list the
    directory.
This applies to every store, not only Lattice's: the markets analyst's and the
workspace CLI's stores archive past the cap too. Nothing prunes `evicted/`.

**Superseded turns stay in the record.** Regenerating an answer or editing a
question replaces what follows it on screen, but the lines are not deleted:
`supersede` marks them, `load_thread` leaves them out unless asked, and so does
the model's context. A thread read back later can still show what was said
before it was replaced.

**More than one process may write a store.** The desktop window and the Lattice
service can run at once against the same directory, and the index is a
read-modify-write. Every write therefore holds an operating-system lock on the
store's `.store.lock` file as well as the in-process lock.

**Named stores.** Every function takes an optional `store` name. The default
store is the Financial Markets analyst's, at the path above, and is what an
unqualified call gets — this is the whole compatibility guarantee, and it is why
`_DIR`/`_INDEX` remain module globals rather than moving inside a class. A named
store is one registered with `register_store`, and it is a different directory
with the same format. Lattice registers its own: two products asking questions
on one machine must not share a thread list, and a markets thread carries book
figures that have no business appearing in a general assistant's history.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from alelyon.runtime.common.paths import GLOBALS_DIR

_log = logging.getLogger(__name__)

_DIR = GLOBALS_DIR / "analyst_chat"
_INDEX = _DIR / "index.json"

#: Named stores, beyond the default one the globals above describe.
_ROOTS: Dict[str, Any] = {}

# Bounds. Threads are cheap, but an unbounded index turns the picker into a
# scrolling graveyard and an unbounded thread makes re-open slow. The thread
# bound limits the LIST: a thread past it is archived under `evicted/`.
MAX_THREADS = 60
MAX_TURNS = 400

#: The directory, inside a store, that evicted threads are moved into.
EVICTED_DIR = "evicted"

# One lock across every store. Contention between two products' chat panels is
# a few file writes a minute; a lock per store would be four more objects whose
# lifetimes have to be reasoned about for no measurable gain.
_LOCK = threading.Lock()


def register_store(name: str, root) -> None:
    """Point a named store at its own directory. Idempotent."""
    from pathlib import Path
    name = str(name or "").strip()
    if not name:
        raise ValueError("a named store needs a name; '' is the default store")
    _ROOTS[name] = Path(root)


def _root(store: str = ""):
    """The directory a call operates in.

    An UNREGISTERED name falls back to the default store rather than raising.
    A typo would otherwise lose a user's conversation at the moment they pressed
    send; landing in the default list is visible and recoverable.
    """
    return _ROOTS.get(str(store or ""), _DIR)


def _index_path(store: str = ""):
    root = _root(store)
    # The default store reads the module global so `_set_root_for_tests` and any
    # caller that swaps `_INDEX` keep working exactly as before.
    return _INDEX if root is _DIR else root / "index.json"


#: How long a writer waits for another process's lock before writing anyway.
#: Losing a reader's message to a stuck peer is worse than the race the lock
#: exists to close, so the wait is bounded and the write proceeds after it.
LOCK_WAIT_S = 10.0


@contextlib.contextmanager
def _store_lock(store: str = "") -> Iterator[None]:
    """Hold the in-process lock and the store's operating-system lock.

    The in-process lock is taken first, so threads in one process queue on it
    rather than on the file. If the file lock cannot be taken within
    `LOCK_WAIT_S` the write still happens under the in-process lock alone,
    which is exactly the protection every write had before this lock existed.
    """
    with _LOCK:
        root = _root(store)
        handle = None
        locked = False
        try:
            root.mkdir(parents=True, exist_ok=True)
            handle = open(root / ".store.lock", "a+b")
            locked = _os_lock(handle)
        except Exception:  # noqa: BLE001 - a lock failure must not lose a write
            locked = False
        try:
            yield
        finally:
            if handle is not None:
                try:
                    if locked:
                        _os_unlock(handle)
                finally:
                    handle.close()


def _os_lock(handle) -> bool:
    deadline = time.monotonic() + LOCK_WAIT_S
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.02)
    import fcntl
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)


def _os_unlock(handle) -> None:
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"


@dataclass
class Turn:
    """One message. Assistant turns carry the evidence, not just the prose."""
    id: str
    ts: float
    role: str
    text: str
    # Assistant-only provenance. Empty on user turns.
    tools: List[str] = field(default_factory=list)       # tool names actually run
    facts: List[Dict[str, Any]] = field(default_factory=list)   # serialised Facts
    unsupported: List[str] = field(default_factory=list)  # figures the facts did not back
    provider: str = ""                                    # which LLM answered
    error: str = ""
    # True when the decoder was grammar-constrained to the desks' own rendered
    # figures. Persisted because it is a different, stronger claim than
    # `grounded`, and a thread re-read later must not conflate them.
    constrained: bool = False
    # The generation stopped early. Persisted, and not merely shown once, because
    # a half answer read back in three months is indistinguishable from a short
    # one — and the difference is whether the model had finished its thought.
    truncated: bool = False
    # The reader stopped it themselves. Recorded separately from `truncated`: one
    # is a failure and the other is a decision.
    cancelled: bool = False
    # Token accounting for the exchange that produced this turn, exactly as the
    # wire reported it. None means the backend reported nothing — UNMEASURED —
    # and must never be rendered as zero: a reader summing a thread's cost has
    # to know which turns were counted and which were not.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    # Replaced by a later version: an answer regenerated, or a question edited
    # together with everything after it. Kept in the file; see `supersede`.
    superseded: bool = False

    @property
    def grounded(self) -> bool:
        """True when every figure in the prose traces to a fact. A turn with no
        figures at all is trivially grounded — it made no numeric claim."""
        return not self.unsupported


@dataclass
class Thread:
    id: str
    title: str
    created: float
    updated: float
    turns: int = 0
    # The provider this thread reopens with — "auto"/"local"/"cloud"/
    # "endpoint:<id>", or "" for none. PINNED by the reader's choice, not derived
    # from the turns: the choice can be made before the first turn exists, and a
    # thread every turn of which was answered by Auto still deserves to reopen on
    # what the reader picked. A stale pin (endpoint since removed) is the picker's
    # problem to fall back from, not this store's to validate.
    pinned_provider: str = ""


def _now() -> float:
    return time.time()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _thread_path(thread_id: str, store: str = ""):
    # Ids are generated here, but a caller could pass anything; never let one
    # escape the directory.
    safe = re.sub(r"[^0-9a-zA-Z_-]", "", str(thread_id or ""))
    return (_root(store) / f"{safe}.jsonl") if safe else None


def auto_title(text: str, limit: int = 48) -> str:
    """A thread's name is its first question, trimmed. Naming a conversation is
    a chore nobody does, and 'New chat 4' is not a name."""
    t = " ".join(str(text or "").split())
    if not t:
        return "New thread"
    return t if len(t) <= limit else t[: limit - 1].rstrip() + "…"


# ── index ────────────────────────────────────────────────────────────────────
def _read_index_locked(store: str = "") -> List[Thread]:
    try:
        raw = json.loads(_index_path(store).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    out: List[Thread] = []
    for d in raw if isinstance(raw, list) else []:
        try:
            out.append(Thread(id=str(d["id"]), title=str(d.get("title", "")),
                              created=float(d.get("created", 0.0)),
                              updated=float(d.get("updated", 0.0)),
                              turns=int(d.get("turns", 0)),
                              pinned_provider=str(d.get("pinned_provider", ""))))
        except Exception:  # noqa: BLE001
            continue
    return out


def _evicted_dir(store: str = ""):
    return _root(store) / EVICTED_DIR


#: A Windows replace of, or onto, a file another handle has open fails with a
#: PermissionError that clears in milliseconds, when the other side is a reader
#: that opens, reads a few kilobytes and closes. The wait is bounded, and the
#: error is raised after it, so a handle that is held for good still fails.
_REPLACE_TRIES = 6
_REPLACE_WAIT_S = 0.02


def _replace(src, dst) -> None:
    """`os.replace`, tried again briefly on a sharing violation."""
    for attempt in range(_REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_TRIES - 1:
                raise
            time.sleep(_REPLACE_WAIT_S)


#: `_read_evicted_index_locked` found a file that is there, was read, and is not
#: a list of rows. Distinct from None (could not be read) because the two are
#: handled differently: only a file known to be damaged is set aside.
_CORRUPT = object()


def _read_evicted_index_locked(store: str = ""):
    """The archive's index rows as stored: a list, `[]` when there is none yet,
    None when one exists and could not be read just now, or `_CORRUPT` when it
    was read and is not a list of rows.

    The differences matter. Rewriting an unreadable index from an empty list
    would erase the only list of what the archive holds, so None means "leave
    everything alone"; a sharing violation or a permission error is that case,
    and may clear. `_CORRUPT` is a file that will never be a valid index again.
    """
    path = _evicted_dir(store) / "index.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except UnicodeDecodeError:
        return _CORRUPT
    except Exception:  # noqa: BLE001
        return None
    try:
        rows = json.loads(raw)
    except ValueError:
        return _CORRUPT
    return rows if isinstance(rows, list) else _CORRUPT


def _write_evicted_index_locked(rows: list, store: str = "") -> None:
    """Atomically, like `index.json`: the old archive index or the new one."""
    index = _evicted_dir(store) / "index.json"
    tmp = index.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    _replace(tmp, index)


def _set_aside_evicted_index_locked(store: str = "") -> bool:
    """Move a damaged archive index out of the way under a name of its own.

    Not deleted and not rewritten: every byte stays on disk for whoever repairs
    it, and the next archive starts a fresh `index.json`. Without this an
    unreadable index stopped the cap for good, because rewriting it from
    nothing would have erased the list of the archive and the alternative was
    to archive nothing. The transcripts in `evicted/` are not touched either
    way. True once the file is out of the way.
    """
    directory = _evicted_dir(store)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(_now()))
    for n in range(1, 1000):
        suffix = "" if n == 1 else f"-{n}"
        target = directory / f"index.corrupt-{stamp}{suffix}.json"
        if target.exists():
            continue
        try:
            _replace(directory / "index.json", target)
        except Exception:  # noqa: BLE001
            return False
        _log.warning("[history] the archive index in %s was damaged; kept as "
                     "%s and a new one started", directory, target.name)
        return True
    return False


def _free_archive_name(thread_id: str, taken: set, directory) -> str:
    """A file name in `evicted/` that no earlier archive holds or is promised.

    An id can come back (a late save re-creates a thread file that was already
    archived once), and `os.replace` would overwrite the earlier archive, so the
    name is chosen to be free: `<id>.jsonl`, then `<id>-2.jsonl`, and so on.
    """
    for n in range(1, 10_000):
        name = f"{thread_id}.jsonl" if n == 1 else f"{thread_id}-{n}.jsonl"
        if name not in taken and not (directory / name).exists():
            return name
    raise OSError("no free archive name")


@dataclass
class _Move:
    """One archive step, kept so that it can be taken back."""
    entry: dict
    added: bool                  # this call added the row (a retry reuses one)
    source: Any
    target: Any                  # where the transcript went; None: no transcript


def _archive_one_locked(row: Thread, entries: list, store: str,
                        moves: List[_Move]) -> bool:
    """Move one evicted thread into `evicted/`. True once it is safely there.

    The archive index row is written BEFORE the move, so a crash between the
    two leaves an archive row for a file that is still in place; the retry finds
    the row by (id, created), brings it up to date (the transcript may have
    grown) and finishes the move. A move that FAILS takes its own row back out,
    so a row always means "this was moved", apart from that crash window. The
    thread file is never unlinked and never overwritten.
    """
    directory = _evicted_dir(store)
    source = _thread_path(row.id, store)
    has_transcript = source is not None and source.exists()
    entry = next((e for e in entries if isinstance(e, dict)
                  and e.get("id") == row.id
                  and e.get("created") == row.created), None)
    if entry is not None and has_transcript and (
            not entry.get("file")
            or (directory / Path(str(entry["file"])).name).exists()):
        # The earlier archive of this id is complete, and a transcript now
        # sits at the id again. This one is archived separately.
        entry = None
    added = entry is None
    if added:
        taken = {str(e.get("file")) for e in entries if isinstance(e, dict)}
        name = (_free_archive_name(source.stem, taken, directory)
                if has_transcript else "")
        entry = {**asdict(row), "evicted_at": _now(), "file": name}
        entries.append(entry)
    else:
        entry.update(title=row.title, updated=row.updated, turns=row.turns,
                     pinned_provider=row.pinned_provider, evicted_at=_now())
    try:
        _write_evicted_index_locked(entries, store)
    except Exception:  # noqa: BLE001
        if added:
            entries[:] = [e for e in entries if e is not entry]
        raise
    name = str(entry.get("file") or "")
    target = directory / Path(name).name if name and has_transcript else None
    if target is not None:
        try:
            if target.exists():
                raise FileExistsError(str(target))     # never overwrite one
            _replace(source, target)
        except Exception:  # noqa: BLE001
            if added:
                _forget_rows_locked(entries, [entry], store)
            raise
    moves.append(_Move(entry, added, source, target))
    return True


def _forget_rows_locked(entries: list, gone: list, store: str) -> None:
    """Take rows back out of the archive index. Best effort: the index it
    writes is only ever shorter, and a failure leaves the rows as they were."""
    entries[:] = [e for e in entries if not any(e is g for g in gone)]
    try:
        _write_evicted_index_locked(entries, store)
    except Exception:  # noqa: BLE001
        pass


def _undo_archive_locked(moves: List[_Move], entries: list, store: str) -> None:
    """Take a batch of archive moves back: transcripts to where they were, the
    rows this batch added out of the archive. A transcript that cannot be put
    back keeps its row, because the row is then the only record of where it is.
    """
    forget = []
    for move in reversed(moves):
        restored = True
        if move.target is not None:
            try:
                if move.target.exists() and not move.source.exists():
                    _replace(move.target, move.source)
            except Exception:  # noqa: BLE001
                restored = False
        if restored and move.added:
            forget.append(move.entry)
    if forget:
        _forget_rows_locked(entries, forget, store)


def _archive_evicted_locked(stale: List[Thread], store: str = ""):
    """Archive threads the cap moved out. Returns `(kept, undo)`: the threads it
    could NOT archive, and a callable that takes this batch back.

    A thread that could not be archived stays in the index, past the cap, and
    is tried again on the next write. A list that is briefly longer than the
    cap is the cheaper failure: dropping the row would leave the transcript
    unreachable. The caller runs `undo` when the thread index it then writes
    cannot be written, so that the archive and the index land together or not
    at all.
    """
    nothing = (lambda: None)
    if not stale:
        return [], nothing
    entries = _read_evicted_index_locked(store)
    if entries is _CORRUPT:
        if not _set_aside_evicted_index_locked(store):
            return list(stale), nothing
        entries = []
    if entries is None:
        return list(stale), nothing
    try:
        _evicted_dir(store).mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        return list(stale), nothing
    kept: List[Thread] = []
    moves: List[_Move] = []
    for row in stale:
        try:
            if not _archive_one_locked(row, entries, store, moves):
                kept.append(row)
        except Exception:  # noqa: BLE001
            kept.append(row)
    return kept, (lambda: _undo_archive_locked(moves, entries, store))


def _forget_pending_archive_locked(row: Thread, store: str = "") -> None:
    """Drop the archive rows a crash left for a thread the reader is deleting.

    A pending row is one whose file is not in `evicted/`: the thread was about
    to be archived and was not. The reader's Delete means the conversation is
    to be gone, and the row's title is its first question. A row whose file is
    there is a finished archive and stays, whatever shares its id.
    """
    entries = _read_evicted_index_locked(store)
    if not isinstance(entries, list) or not entries:
        return
    directory = _evicted_dir(store)
    pending = [e for e in entries if isinstance(e, dict)
               and e.get("id") == row.id and e.get("created") == row.created
               and e.get("file")
               and not (directory / Path(str(e["file"])).name).exists()]
    if pending:
        _forget_rows_locked(entries, pending, store)


def _write_index_locked(rows: List[Thread], store: str = "") -> None:
    index = _index_path(store)
    _root(store).mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda t: t.updated, reverse=True)
    # Past the cap a thread is archived, not deleted (see the module docstring).
    # Every caller holds `_store_lock`, so the move and the index write are one
    # critical section, and a failed index write takes the moves back.
    over, undo = _archive_evicted_locked(rows[MAX_THREADS:], store)
    rows = rows[:MAX_THREADS] + over
    try:
        tmp = index.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(r) for r in rows], indent=1),
                       encoding="utf-8")
        _replace(tmp, index)
    except Exception:
        undo()
        raise


def list_threads(store: str = "") -> List[Thread]:
    """Most recently touched first."""
    with _LOCK:
        return _read_index_locked(store)


def new_thread(title: str = "", store: str = "") -> Thread:
    t = Thread(id=_new_id(), title=title or "New thread",
               created=_now(), updated=_now(), turns=0)
    with _store_lock(store):
        rows = _read_index_locked(store)
        rows.append(t)
        _write_index_locked(rows, store)
    return t


def rename(thread_id: str, title: str, store: str = "") -> bool:
    title = " ".join(str(title or "").split())[:80]
    if not title:
        return False
    with _store_lock(store):
        rows = _read_index_locked(store)
        for r in rows:
            if r.id == thread_id:
                r.title = title
                _write_index_locked(rows, store)
                return True
    return False


def pin_provider(thread_id: str, provider: str, store: str = "") -> bool:
    """Pin the provider a thread reopens with; "" clears the pin.

    Only the index row is touched — not `updated`, because pinning is not a new
    message and must not reorder the thread list under the reader's hand. Returns
    False for an unknown thread rather than creating one: a pin with no thread to
    hold it is a caller bug, not a thread.
    """
    provider = str(provider or "").strip()
    with _store_lock(store):
        rows = _read_index_locked(store)
        for r in rows:
            if r.id == thread_id:
                r.pinned_provider = provider
                _write_index_locked(rows, store)
                return True
    return False


def touch(thread_id: str, store: str = "") -> bool:
    """Count a thread as used now, without adding a turn.

    For work done in a thread that is not a new message: regenerating an answer
    rewrites what follows a question and then waits for a model, and meanwhile
    the thread's `updated` is as old as its last message, which is exactly what
    the cap reads to choose the thread it archives. Without this a thread in
    the middle of an answer can be the least recently used one. False for an
    unknown thread, which is not created.
    """
    with _store_lock(store):
        rows = _read_index_locked(store)
        for r in rows:
            if r.id == thread_id:
                r.updated = _now()
                _write_index_locked(rows, store)
                return True
    return False


def delete_thread(thread_id: str, store: str = "") -> bool:
    """The reader's own Delete: a real delete, nothing is archived.

    The one thing kept is a finished archive of an earlier life of the same id.
    A row the cap left half-written for this thread (file not yet moved) goes
    with it, because its title is the first question of a chat that is now to be
    gone.
    """
    with _store_lock(store):
        rows = _read_index_locked(store)
        keep = [r for r in rows if r.id != thread_id]
        if len(keep) == len(rows):
            return False
        _write_index_locked(keep, store)
        for gone in (r for r in rows if r.id == thread_id):
            try:
                _forget_pending_archive_locked(gone, store)
            except Exception:  # noqa: BLE001 - a Delete must still delete
                pass
        p = _thread_path(thread_id, store)
        try:
            if p is not None and p.exists():
                p.unlink()
        except Exception:  # noqa: BLE001
            pass
    return True


# ── turns ────────────────────────────────────────────────────────────────────
def _count(value) -> Optional[int]:
    """A token count the record actually stated, or None. Never coerces an
    absence into a zero — the two are different facts."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _parse_turn(d: dict) -> Optional[Turn]:
    try:
        return Turn(
            id=str(d.get("id") or _new_id()),
            ts=float(d.get("ts", 0.0)),
            role=str(d.get("role", ROLE_USER)),
            text=str(d.get("text", "")),
            tools=[str(x) for x in (d.get("tools") or [])],
            facts=[x for x in (d.get("facts") or []) if isinstance(x, dict)],
            unsupported=[str(x) for x in (d.get("unsupported") or [])],
            provider=str(d.get("provider", "")),
            error=str(d.get("error", "")),
            constrained=bool(d.get("constrained", False)),
            # Absent in every record written before streaming existed, which is
            # exactly right: those answers all arrived whole.
            truncated=bool(d.get("truncated", False)),
            cancelled=bool(d.get("cancelled", False)),
            prompt_tokens=_count(d.get("prompt_tokens")),
            completion_tokens=_count(d.get("completion_tokens")),
            # Only a literal true supersedes. A record of any other shape
            # keeps the turn visible, so a malformed line cannot hide one.
            superseded=d.get("superseded") is True,
        )
    except Exception:  # noqa: BLE001
        return None


def _heal_torn_tail(path) -> None:
    """Terminate a partial final line before appending.

    A crash mid-write leaves a record with no trailing newline. Appending after
    it concatenates the two on one line, so the reader loses BOTH — the torn
    record *and* the good one written after it, and every turn thereafter. The
    parser already tolerates one unreadable line; this makes sure the damage
    stops at one.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb+") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
    except Exception:  # noqa: BLE001
        pass


def append(thread_id: str, turn: Turn, store: str = "") -> bool:
    """Append one turn and touch the index. Returns False if nothing was
    written — a caller must not show a message as saved when it is not."""
    p = _thread_path(thread_id, store)
    if p is None:
        return False
    if not turn.id:
        turn.id = _new_id()
    if not turn.ts:
        turn.ts = _now()
    try:
        with _store_lock(store):
            _root(store).mkdir(parents=True, exist_ok=True)
            _heal_torn_tail(p)
            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(turn)) + "\n")
            rows = _read_index_locked(store)
            known = {r.id for r in rows}
            if thread_id not in known:
                rows.append(Thread(id=thread_id, title=auto_title(turn.text),
                                   created=turn.ts, updated=turn.ts, turns=1))
            else:
                for r in rows:
                    if r.id == thread_id:
                        r.updated = turn.ts
                        r.turns += 1
                        # The first user message names the thread.
                        if turn.role == ROLE_USER and (
                                not r.title or r.title == "New thread"):
                            r.title = auto_title(turn.text)
                        break
            _write_index_locked(rows, store)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_thread(thread_id: str, limit: int = MAX_TURNS,
                store: str = "", *,
                include_superseded: bool = False) -> List[Turn]:
    """Oldest first — this is a transcript, and reading it backwards is wrong.

    Superseded turns are left out unless `include_superseded` is set, so the
    transcript on screen and the model's context both see the current version.
    """
    p = _thread_path(thread_id, store)
    if p is None:
        return []
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return []
    out: List[Turn] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            t = _parse_turn(json.loads(line))
        except Exception:  # noqa: BLE001
            continue                 # a torn line must not hide the rest
        if t is not None and (include_superseded or not t.superseded):
            out.append(t)
    return out[-max(1, int(limit)):]


def supersede(thread_id: str, from_turn_id: str, store: str = "") -> int:
    """Mark one turn and every later turn superseded. Returns how many.

    This is how an answer is regenerated or a question edited: what follows
    the replaced turn stops being part of the conversation, and stays in the
    file. Turns already superseded are left as they are, and an unknown or
    already-superseded `from_turn_id` marks nothing.

    The thread file is rewritten through a temporary sibling and replaced, so
    a crash leaves the old transcript or the new one, never a mixture. A line
    that does not parse is written back byte for byte: a rewrite must not
    destroy the one copy of a damaged record.
    """
    p = _thread_path(thread_id, store)
    wanted = str(from_turn_id or "")
    if p is None or not wanted:
        return 0
    with _store_lock(store):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return 0
        records: List[tuple] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except Exception:  # noqa: BLE001
                parsed = None
            records.append((stripped,
                            parsed if isinstance(parsed, dict) else None))
        start = next((
            index for index, (_line, record) in enumerate(records)
            if record is not None and str(record.get("id", "")) == wanted
            and record.get("superseded") is not True), None)
        if start is None:
            return 0
        marked = 0
        lines: List[str] = []
        visible = 0
        for index, (line, record) in enumerate(records):
            if (record is not None and index >= start
                    and record.get("superseded") is not True):
                record = {**record, "superseded": True}
                line = json.dumps(record)
                marked += 1
            lines.append(line)
            if record is not None and record.get("superseded") is not True:
                visible += 1
        temporary = p.with_name(p.name + ".tmp")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(p)
        rows = _read_index_locked(store)
        for row in rows:
            if row.id == thread_id:
                row.turns = visible
                break
        _write_index_locked(rows, store)
    return marked


def recent_exchanges(thread_id: str, pairs: int = 4,
                     store: str = "") -> List[Turn]:
    """The tail of the transcript, for conversational follow-ups ('and its
    peers?'). Bounded on purpose: the whole thread would blow the context of a
    local model and drag stale figures into a fresh question."""
    turns = load_thread(thread_id, store=store)
    return turns[-max(0, int(pairs) * 2):] if pairs > 0 else []


def clear_all(store: str = "") -> bool:
    """Wipe every thread in ONE store.

    A store name is required to reach anything but the default, so clearing
    Lattice's history cannot take the markets analyst's with it.

    This wipes the listed threads. It does not reach `evicted/`: archived
    threads are not deleted by anything but the reader's own decision, and no
    caller yet carries one for them. A "clear history" control that means the
    archive too must say so and remove `evicted/` itself.
    """
    root, index = _root(store), _index_path(store)
    try:
        with _store_lock(store):
            if root.exists():
                for p in root.glob("*.jsonl"):
                    try:
                        p.unlink()
                    except Exception:  # noqa: BLE001
                        pass
                if index.exists():
                    index.unlink()
        return True
    except Exception:  # noqa: BLE001
        return False


def _set_root_for_tests(path) -> None:
    """Point the store at a tmp dir. Tests must never touch real history."""
    global _DIR, _INDEX
    from pathlib import Path
    _DIR = Path(path)
    _INDEX = _DIR / "index.json"
