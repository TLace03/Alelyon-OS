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
    index.json      thread list (id, title, timestamps, turn count)
    <id>.jsonl      one turn per line, append-only

A directory rather than one file, so a torn write in one thread cannot take the
others down with it, and deleting a thread is an unlink rather than a rewrite of
everybody's history.

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
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from alelyon.runtime.common.paths import GLOBALS_DIR

_DIR = GLOBALS_DIR / "analyst_chat"
_INDEX = _DIR / "index.json"

#: Named stores, beyond the default one the globals above describe.
_ROOTS: Dict[str, Any] = {}

# Bounds. Threads are cheap, but an unbounded index turns the picker into a
# scrolling graveyard and an unbounded thread makes re-open slow.
MAX_THREADS = 60
MAX_TURNS = 400

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


def _write_index_locked(rows: List[Thread], store: str = "") -> None:
    index = _index_path(store)
    _root(store).mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda t: t.updated, reverse=True)
    for stale in rows[MAX_THREADS:]:
        p = _thread_path(stale.id, store)
        try:
            if p is not None and p.exists():
                p.unlink()
        except Exception:  # noqa: BLE001
            pass
    rows = rows[:MAX_THREADS]
    tmp = index.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8")
    tmp.replace(index)


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


def delete_thread(thread_id: str, store: str = "") -> bool:
    with _store_lock(store):
        rows = _read_index_locked(store)
        keep = [r for r in rows if r.id != thread_id]
        if len(keep) == len(rows):
            return False
        _write_index_locked(keep, store)
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
