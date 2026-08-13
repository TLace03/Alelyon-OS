"""The collective cache every worktree writes into, and what it is safe to show.

`worktree.observe()` answers "what is true right now". This module remembers, so
the questions that need two points in time become answerable: which worktree
changed what, when it started, whether it is still moving, and which sessions are
working near each other. That is the substrate the Dynamic Worktree Cache in
`docs/features/DYNAMIC-CACHE.md` is built on.

Derived and declared are stored apart and never merged
-------------------------------------------------------
Two kinds of record go in, and conflating them would undo the whole design:

* **Observed** rows are written from `worktree.observe()`, i.e. derived from git's
  own records. Nothing an agent writes can change them.
* **Declared** rows are written by a session about itself — its id, its model, what
  it thinks it is doing. That is genuinely useful (it is the only route to model
  identity at all) and it is *self-reported*, so it lives in its own table, is
  labelled at every read, and never overwrites an observed fact.

Where the two disagree, both are kept and `disagreements()` names it. A session
claiming a worktree that git says belongs to another path is exactly the event this
cache exists to make visible.

`occupancy_conflicts()` names the other disagreement, and it is the one the
observer cannot reach alone. A worktree path carries the session that CREATED the
directory; nothing in it changes when a second session enters and works there, so
that session's edits are attributed to the creator. `find_contentions()` cannot
correct this either — it pairs worktrees against one another, and two sessions in
one worktree are one record with one merged set of touched paths. Only a
declaration can say who is actually in a tree, which makes this the one question
here that a self-report answers better than a derivation, and the guarantee is
one-sided to match: a conflict named is real, a silent worktree is unproven.

Colour capacity is measured, not assumed
-----------------------------------------
Colour is how a reader tells two worktrees apart at a glance, so the palette was
validated rather than chosen: `scripts/validate_palette.js` from the dataviz
skill, run on the reference categorical ramp with `--pairs all`, because a reader
compares *any* two worktrees and not merely adjacent ones.

The result is `COLOUR_SLOTS` — **three** hues, the largest set that clears every
gate in BOTH light and dark. The full eight-hue ramp fails badly under all-pairs
(worst normal-vision ΔE 7.1, protan 3.2), and even four fails in dark, where
violet and blue collapse to ΔE 1.9 for a protan reader.

So colour is an accelerator and never the identity. Every worktree also carries a
stable `sigil`, and a worktree past capacity gets no hue at all rather than a
generated one — a fourth "colour" nobody can distinguish from the first is worse
than an honest blank, and the label is doing the work either way.

Slots are assigned first-seen and **persisted**, so colour follows the entity and
not its rank: a new worktree never repaints the ones already on screen.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import functools
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import time
from typing import Iterable

from alelyon.runtime.common import sqlite_wal as _DB
from alelyon.runtime.common import toolpath
from alelyon.runtime.common.worktree import (
    UNATTRIBUTED, WorktreeMesh, observe, session_for_path,
)

SCHEMA_VERSION = 1

#: How long one SQLite call may wait for a lock, and how many times a BUSY is
#: retried before it becomes a refusal.
#:
#: This was 1.0s with no retry at all, on a path that opens with BEGIN IMMEDIATE
#: — a write lock — while the store it takes it on is shared by every session in
#: the repository. Under the rollback journal those stores ran on, one writer
#: excluded every reader, so a second later the caller raised
#: `RepositoryScopeUnavailable` and the views that depend on it did not load.
#: A fail-fast budget turns ordinary contention into an outage.
#:
#: The numbers are proportionate rather than generous: worst case is
#: `_SQLITE_BUSY_ATTEMPTS` waits of at most 0.8s of backoff plus the 5s lock
#: wait each, and it is BOUNDED — a fixed count, a capped delay, then the
#: original error by name.
_SQLITE_TIMEOUT_SECONDS = 5.0
_SQLITE_BUSY_TIMEOUT_MS = int(_SQLITE_TIMEOUT_SECONDS * 1000)
_SQLITE_BUSY_ATTEMPTS = 4

#: The shared bus itself. Already 10s; now also the explicit busy_timeout, so
#: "how long may this block" is one number rather than two that can drift.
_BUS_TIMEOUT_SECONDS = 10.0

# A linked worktree's ``.git`` marker is a short ``gitdir: ...`` pointer. Its
# contents are part of the checkout incarnation: rewriting that pointer in
# place must invalidate cached repository identity even when the file's inode
# is unchanged. Read only a bounded prefix; a valid pointer's operative first
# line is far smaller than this budget.
_GIT_POINTER_IDENTITY_BYTES = 4096

#: Validated with `node scripts/validate_palette.js "<hex,…>" --mode <mode> --pairs all`
#: (dataviz skill). Both modes ALL CHECKS PASS at three slots:
#:   dark  — worst all-pairs CVD ΔE 9.4 (deutan), normal-vision ΔE 20.9
#:   light — worst all-pairs CVD ΔE 9.2 (deutan), normal-vision ΔE 24.0
#: Adding a fourth fails: light survives violet, dark does not (ΔE 1.9 protan
#: against blue). Three is the number that holds in both, so three is the number.
COLOUR_SLOTS: tuple[tuple[str, str, str], ...] = (
    ("blue", "#2a78d6", "#3987e5"),
    ("orange", "#eb6834", "#d95926"),
    ("aqua", "#1baf7a", "#199e70"),
)

#: What a worktree past capacity gets. Not a hue: see the module docstring.
UNSLOTTED = "unslotted"

#: Light-mode aqua measures 2.74:1 against the reference surface, under the 3:1
#: bar. The dataviz rule is that a contrast WARN obligates relief rather than
#: being dismissable, so a renderer must label the mark or offer a table view —
#: it must not rely on the swatch alone.
LOW_CONTRAST_LIGHT = frozenset({"aqua"})


@dataclass(frozen=True)
class WorktreeIdentity:
    """A worktree's durable identity in the cache."""

    key: str            # stable hash of the repo-relative path
    path: str
    label: str
    sigil: str          # short stable text badge — the identity colour only hints at
    colour_slot: str    # a COLOUR_SLOTS name, or UNSLOTTED
    first_seen: int
    last_seen: int
    tool_family: str

    def colour(self, *, dark: bool = True) -> str | None:
        """The hex for this worktree, or None when it holds no slot.

        None is the honest answer past capacity. A renderer must fall back to the
        sigil, never to a generated hue.
        """
        for name, light_hex, dark_hex in COLOUR_SLOTS:
            if name == self.colour_slot:
                return dark_hex if dark else light_hex
        return None

    @property
    def needs_contrast_relief(self) -> bool:
        """True when the light-mode swatch is below 3:1 and needs a label."""
        return self.colour_slot in LOW_CONTRAST_LIGHT


@dataclass(frozen=True)
class Operation:
    """One observed change to one worktree, between two snapshots."""

    key: str
    at: int
    kind: str           # "appeared" | "touched" | "settled" | "advanced" | "vanished"
    detail: str
    paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class Declaration:
    """What a session says about itself. Self-reported; never a derived fact."""

    key: str
    at: int
    session_id: str
    model: str
    note: str


#: A worktree whose declared occupant is not the session its path names.
FOREIGN_OCCUPANT = "foreign-occupant"
#: A worktree more than one session has declared work in.
SHARED_OCCUPANCY = "shared-occupancy"


@dataclass(frozen=True)
class Occupancy:
    """Who is working in one worktree, where that is not just its creator.

    The path names the session that CREATED the directory and nothing rewrites it
    when somebody else walks in, so this record exists to hold the difference
    rather than to resolve it. Both sides are kept: `derived_session` is what the
    location says, `declared_sessions` is what sessions said about themselves,
    and neither overwrites the other.
    """

    key: str
    path: str
    label: str
    kind: str                            # FOREIGN_OCCUPANT | SHARED_OCCUPANCY
    derived_session: str                 # from the path; the creator
    declared_sessions: tuple[str, ...]   # self-reported; the occupants
    last_declared_at: int

    @property
    def message(self) -> str:
        """One line a reader can act on. ASCII only: this is printed by
        `tools/worktree_report.py`, and a Windows console codepage cannot encode
        an em dash — a warning that raises on the way out is not a warning."""
        others = ", ".join(repr(s) for s in self.declared_sessions)
        if self.kind == SHARED_OCCUPANCY:
            return (f"{self.label}: {len(self.declared_sessions)} sessions have "
                    f"declared work in one worktree ({others}); their changes "
                    f"share a single tree and cannot contend as a pair")
        return (f"{self.label}: declared by {others}, but the path names "
                f"{self.derived_session!r}. Edits made here are attributed to "
                f"the creator, not the occupant")


_DDL = (
    """CREATE TABLE IF NOT EXISTS meta (
           name TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS worktree (
           key TEXT PRIMARY KEY,
           path TEXT NOT NULL,
           label TEXT NOT NULL,
           sigil TEXT NOT NULL,
           colour_slot TEXT NOT NULL,
           first_seen INTEGER NOT NULL,
           last_seen INTEGER NOT NULL,
           tool_family TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS snapshot (
           key TEXT NOT NULL,
           at INTEGER NOT NULL,
           head TEXT NOT NULL,
           touched TEXT NOT NULL,
           present INTEGER NOT NULL,
           PRIMARY KEY (key, at))""",
    """CREATE TABLE IF NOT EXISTS operation (
           key TEXT NOT NULL, at INTEGER NOT NULL, kind TEXT NOT NULL,
           detail TEXT NOT NULL, paths TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS declaration (
           key TEXT NOT NULL, at INTEGER NOT NULL, session_id TEXT NOT NULL,
           model TEXT NOT NULL, note TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS operation_key_at ON operation(key, at)",
    "CREATE INDEX IF NOT EXISTS snapshot_key_at ON snapshot(key, at)",
)


def _key_for(repo_root: str, path: str) -> str:
    """A stable identity for a worktree, independent of where the repo lives.

    Keyed on the path RELATIVE to the repository when it is inside it, so a
    checkout that moves keeps its history. An outside path keys on its own
    absolute form, which is the best available and is stated as such.
    """
    normalized = path.replace("\\", "/").rstrip("/")
    root = repo_root.replace("\\", "/").rstrip("/")
    if normalized.lower().startswith(root.lower() + "/"):
        normalized = normalized[len(root) + 1:]
    elif normalized.lower() == root.lower():
        normalized = "."
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=8).hexdigest()


def _sigil_for(key: str, label: str) -> str:
    """A short, stable, human-sayable badge.

    The identity a reader can rely on when colour has run out or cannot be seen.
    Derived from the key so it never changes, and prefixed with a letter of the
    label so it stays recognisable beside the name it belongs to.
    """
    head = "".join(ch for ch in label.upper() if ch.isalnum())[:2] or "WT"
    return f"{head}-{key[:4]}"


class WorktreeCache:
    """The durable collective store. One SQLite file, opened per operation."""

    def __init__(self, database: str | Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(self._connect()) as conn, conn:
            for statement in _DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO meta(name, value) VALUES('schema', ?)",
                (str(SCHEMA_VERSION),))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database), timeout=_BUS_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        # WAL. Measured 2026-08-11 this file was `journal_mode=delete` at 91 MB
        # with ten sessions live in one checkout: every `publish`, `claim` and
        # `status` took an EXCLUSIVE whole-database lock and every other session
        # waited for it. `db.try_set_wal` carries the cold-open retry subtlety
        # and declines rather than refuses where WAL is unavailable.
        _DB.try_set_wal(conn)
        conn.execute(f"PRAGMA busy_timeout = {int(_BUS_TIMEOUT_SECONDS * 1000)}")
        return conn

    # ── writing ─────────────────────────────────────────────────────────────

    def record(self, mesh: WorktreeMesh) -> tuple[Operation, ...]:
        """Fold one observation into the cache and return what changed.

        The diff against the previous snapshot is what makes per-worktree change
        tracking possible: a single observation is a state, and two are a history.
        """
        operations: list[Operation] = []
        with contextlib.closing(self._connect()) as conn, conn:
            seen_keys = set()
            for tree in mesh.worktrees:
                key = _key_for(mesh.repo_root, tree.path)
                seen_keys.add(key)
                self._upsert_identity(conn, key, tree, mesh.observed_at)
                touched = sorted(tree.touched_paths)
                previous = conn.execute(
                    "SELECT head, touched, present FROM snapshot WHERE key=? "
                    "ORDER BY at DESC LIMIT 1", (key,)).fetchone()
                operations.extend(
                    self._diff(key, previous, tree, touched, mesh.observed_at))
                conn.execute(
                    "INSERT OR REPLACE INTO snapshot(key, at, head, touched, present)"
                    " VALUES(?,?,?,?,?)",
                    (key, mesh.observed_at, tree.head, json.dumps(touched),
                     int(tree.present)))
            # A worktree the cache knows and this observation does not is gone.
            for row in conn.execute(
                    "SELECT key, label FROM worktree WHERE last_seen < ?",
                    (mesh.observed_at,)):
                if row["key"] not in seen_keys:
                    already = conn.execute(
                        "SELECT 1 FROM operation WHERE key=? AND kind='vanished' "
                        "ORDER BY at DESC LIMIT 1", (row["key"],)).fetchone()
                    if not already:
                        operations.append(Operation(
                            key=row["key"], at=mesh.observed_at, kind="vanished",
                            detail=f"{row['label']} is no longer listed by git"))
            for operation in operations:
                conn.execute(
                    "INSERT INTO operation(key, at, kind, detail, paths)"
                    " VALUES(?,?,?,?,?)",
                    (operation.key, operation.at, operation.kind,
                     operation.detail, json.dumps(list(operation.paths))))
        return tuple(operations)

    def _diff(self, key, previous, tree, touched, at) -> list[Operation]:
        if previous is None:
            return [Operation(key=key, at=at, kind="appeared",
                              detail=f"first seen at {tree.path}",
                              paths=tuple(touched))]
        out: list[Operation] = []
        was = json.loads(previous["touched"])
        if previous["head"] != tree.head:
            out.append(Operation(key=key, at=at, kind="advanced",
                                 detail=f"HEAD moved to {tree.head[:12]}"))
        gained = sorted(set(touched) - set(was))
        if gained:
            out.append(Operation(key=key, at=at, kind="touched",
                                 detail=f"{len(gained)} path(s) newly changed",
                                 paths=tuple(gained)))
        released = sorted(set(was) - set(touched))
        if released:
            out.append(Operation(key=key, at=at, kind="settled",
                                 detail=f"{len(released)} path(s) no longer changed",
                                 paths=tuple(released)))
        return out

    def _upsert_identity(self, conn, key: str, tree, at: int) -> None:
        row = conn.execute("SELECT key FROM worktree WHERE key=?", (key,)).fetchone()
        if row:
            # Colour and sigil are NEVER reassigned. That is the point of storing
            # them: a worktree keeps its identity across sessions, and adding a
            # new one cannot repaint the ones a reader has already learned.
            conn.execute(
                "UPDATE worktree SET last_seen=?, path=?, label=?, tool_family=? "
                "WHERE key=?", (at, tree.path, tree.label, tree.tool_family, key))
            return
        taken = {r["colour_slot"] for r in
                 conn.execute("SELECT colour_slot FROM worktree")}
        slot = next((name for name, _l, _d in COLOUR_SLOTS if name not in taken),
                    UNSLOTTED)
        conn.execute(
            "INSERT INTO worktree(key, path, label, sigil, colour_slot,"
            " first_seen, last_seen, tool_family) VALUES(?,?,?,?,?,?,?,?)",
            (key, tree.path, tree.label, _sigil_for(key, tree.label), slot,
             at, at, tree.tool_family))

    def declare(self, *, repo_root: str, path: str, session_id: str,
                model: str = "", note: str = "", at: int | None = None) -> Declaration:
        """Record what a session says about itself.

        SELF-REPORTED. Stored apart from observed facts, never overwriting one,
        and labelled at every read. It is the only route to model identity there
        is, and it is a claim.
        """
        key = _key_for(repo_root, path)
        moment = int(at if at is not None else time.time())
        with contextlib.closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO declaration(key, at, session_id, model, note)"
                " VALUES(?,?,?,?,?)", (key, moment, session_id, model, note))
        return Declaration(key=key, at=moment, session_id=session_id,
                           model=model, note=note)

    # ── reading ─────────────────────────────────────────────────────────────

    def identities(self) -> tuple[WorktreeIdentity, ...]:
        with contextlib.closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM worktree ORDER BY first_seen, key").fetchall()
        return tuple(WorktreeIdentity(
            key=r["key"], path=r["path"], label=r["label"], sigil=r["sigil"],
            colour_slot=r["colour_slot"], first_seen=r["first_seen"],
            last_seen=r["last_seen"], tool_family=r["tool_family"]) for r in rows)

    def history(self, key: str, *, limit: int = 200) -> tuple[Operation, ...]:
        """Everything this one worktree has done, newest first.

        The per-worktree track: what a reader follows when they want one
        worktree's story instead of the whole mesh's.
        """
        with contextlib.closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM operation WHERE key=? ORDER BY at DESC, rowid DESC"
                " LIMIT ?", (key, limit)).fetchall()
        return tuple(Operation(key=r["key"], at=r["at"], kind=r["kind"],
                               detail=r["detail"],
                               paths=tuple(json.loads(r["paths"]))) for r in rows)

    def recent(self, *, limit: int = 100) -> tuple[Operation, ...]:
        """The collective feed across every worktree, newest first."""
        with contextlib.closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT * FROM operation ORDER BY at DESC, rowid DESC LIMIT ?",
                (limit,)).fetchall()
        return tuple(Operation(key=r["key"], at=r["at"], kind=r["kind"],
                               detail=r["detail"],
                               paths=tuple(json.loads(r["paths"]))) for r in rows)

    def declarations(self, key: str | None = None) -> tuple[Declaration, ...]:
        with contextlib.closing(self._connect()) as conn, conn:
            if key is None:
                rows = conn.execute(
                    "SELECT * FROM declaration ORDER BY at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM declaration WHERE key=? ORDER BY at DESC",
                    (key,)).fetchall()
        return tuple(Declaration(key=r["key"], at=r["at"],
                                 session_id=r["session_id"], model=r["model"],
                                 note=r["note"]) for r in rows)

    def occupancy_conflicts(self) -> tuple[Occupancy, ...]:
        """Worktrees whose occupants are not simply the session that made them.

        `find_contentions()` pairs worktrees against each other, so it is blind
        by construction to two sessions inside ONE worktree: there is a single
        record, its `touched_paths` already merges both, and the pair it would
        need does not exist. That arrangement is not hypothetical — it is how one
        session's uncommitted work gets absorbed by a peer's next commit — and it
        is the only place the mesh reports a confident attribution that is wrong
        rather than an honest UNATTRIBUTED.

        Two shapes are reported, and only declarations can reveal either, because
        nothing observable changes when a second session enters a directory:

        * FOREIGN_OCCUPANT — the declared occupant is not the session the path
          names. The path still says the creator; this says who is actually
          there.
        * SHARED_OCCUPANCY — two or more distinct sessions declared the same
          worktree. Reported whether or not one of them is the creator, since
          what matters to a reader is the count of hands in one tree.

        Derived and declared are carried side by side and neither is corrected
        against the other. A session that never declares is invisible here, which
        is a real hole and not a solved problem: this measures participation, so
        the guarantee is one-sided — a conflict named is real, a silent tree is
        unproven rather than clear. `WorktreeMesh.limits` says so too.
        """
        with contextlib.closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT d.key AS key, w.path AS path, w.label AS label,"
                "       d.session_id AS session_id, MAX(d.at) AS at"
                "  FROM declaration d JOIN worktree w ON w.key = d.key"
                " GROUP BY d.key, d.session_id").fetchall()

        by_key: dict[str, list] = {}
        for row in rows:
            by_key.setdefault(row["key"], []).append(row)

        out: list[Occupancy] = []
        for key, group in by_key.items():
            declared = tuple(sorted({r["session_id"] for r in group}))
            derived = session_for_path(group[0]["path"])
            last_at = max(int(r["at"]) for r in group)
            if len(declared) > 1:
                kind = SHARED_OCCUPANCY
            elif derived != UNATTRIBUTED and declared[0] != derived:
                kind = FOREIGN_OCCUPANT
            else:
                # One session, and either it made the tree or the path carries no
                # id to disagree with. Nothing to report: an UNATTRIBUTED path is
                # an absent comparison, never a passed one.
                continue
            out.append(Occupancy(
                key=key, path=group[0]["path"], label=group[0]["label"],
                kind=kind, derived_session=derived,
                declared_sessions=declared, last_declared_at=last_at))
        # Most hands first, then stable by label so a reader can diff two runs.
        out.sort(key=lambda o: (-len(o.declared_sessions), o.label, o.key))
        return tuple(out)

    def disagreements(self) -> tuple[str, ...]:
        """Where a declaration names a worktree the observer has never seen.

        A session claiming a path git does not list is the event this cache
        exists to surface, so it is reported rather than reconciled.
        """
        with contextlib.closing(self._connect()) as conn, conn:
            rows = conn.execute(
                "SELECT DISTINCT d.key, d.session_id FROM declaration d "
                "LEFT JOIN worktree w ON w.key = d.key WHERE w.key IS NULL"
            ).fetchall()
        return tuple(
            f"session {r['session_id']!r} declared work on worktree {r['key']}, "
            f"which git has never listed in this repository" for r in rows)

    def colour_capacity(self) -> tuple[int, int]:
        """(slots held, slots available). Past capacity, colour stops helping."""
        with contextlib.closing(self._connect()) as conn, conn:
            held = conn.execute(
                "SELECT COUNT(*) c FROM worktree WHERE colour_slot != ?",
                (UNSLOTTED,)).fetchone()["c"]
        return int(held), len(COLOUR_SLOTS)


class _RepositoryIdentityUnavailable(Exception):
    """Internal signal whose exception semantics prevent negative caching."""


class RepositoryDatabaseAmbiguity(RuntimeError):
    """More than one fallback store claims one recovered Git repository."""


class RepositoryDatabaseUnavailable(RuntimeError):
    """A Git repository's durable Fleet database identity is unavailable."""


class RepositoryScopeUnavailable(RuntimeError):
    """A selected Git checkout's privacy boundary could not be established."""


def _anchor_cache_marker(anchor: str) -> tuple[int, ...]:
    """Cheap incarnation stamp preventing stale path-to-repository cache hits.

    Device/inode/mode/birth identity, and deliberately NOT modification time.

    This key used to carry ``st_mtime_ns`` and ``st_ctime_ns`` for the selected
    root and its ``.git``, which made it a cache key that could not hit: a
    directory's mtime moves whenever Git writes inside it, and one ordinary
    ``git status`` in the primary checkout is enough to change it. Measured on
    this repository at 287 worktrees, the identity LRU behind this marker took
    ONE hit per 287 lookups and re-spawned ``git rev-parse`` for the other 286.

    What these caches have to notice is an *unrelated checkout arriving at the
    same path* or a linked checkout being rebound to another common directory.
    Root/``.git`` filesystem identity handles the first. Bounded digests of the
    administrative ``commondir`` and reverse ``gitdir`` pointers handle the
    second without making ordinary source edits invalidate the cache.
    """
    marker = _root_incarnation_marker(anchor)
    try:
        root = Path(anchor).expanduser().resolve()
    except (OSError, RuntimeError):
        return marker + (-1, -1)
    admin = _linked_admin_directory(root / ".git")
    if admin is None:
        return marker + (-1, -1)
    return marker + (
        _bounded_identity_digest(admin / "commondir"),
        _bounded_identity_digest(admin / "gitdir"),
    )


def _bounded_identity_payload(path: Path) -> bytes | None:
    """Read one small structural Git file for a cache identity only."""
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) \
                or metadata.st_size > _GIT_POINTER_IDENTITY_BYTES:
            return None
        with path.open("rb") as handle:
            payload = handle.read(_GIT_POINTER_IDENTITY_BYTES + 1)
    except (OSError, RuntimeError):
        return None
    return payload if len(payload) <= _GIT_POINTER_IDENTITY_BYTES else None


def _bounded_identity_digest(path: Path) -> int:
    payload = _bounded_identity_payload(path)
    if payload is None:
        return -1
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _linked_admin_directory(dot_git: Path) -> Path | None:
    """Resolve only a bounded ``gitdir:`` marker; run no Git process."""
    payload = _bounded_identity_payload(dot_git)
    if payload is None:
        return None
    try:
        value = payload.decode("utf-8", errors="strict").strip()
    except UnicodeError:
        return None
    prefix = "gitdir:"
    if len(value.splitlines()) != 1 \
            or not value.lower().startswith(prefix):
        return None
    target = value[len(prefix):].strip()
    if not target:
        return None
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = dot_git.parent / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_dir() else None


def _root_incarnation_marker(anchor: str) -> tuple[int, ...]:
    """Stable root/.git identity, including a linked-worktree binding.

    Directory metadata is deliberately insensitive to ordinary repository
    edits. A linked worktree is different: its ``.git`` file binds the visible
    directory to an administrative Git directory, and that file can be
    rewritten without changing its inode. A bounded digest of its operative
    prefix prevents a stale cached identity from surviving that reassignment.
    """
    values: list[int] = []
    try:
        root = Path(anchor).expanduser().resolve()
    except (OSError, RuntimeError):
        return (-1,) * 10
    for entry in (root, root / ".git"):
        try:
            metadata = entry.lstat()
        except (OSError, RuntimeError):
            values.extend((-1, -1, -1, -1, -1))
            continue
        birth = getattr(metadata, "st_birthtime_ns", None)
        if birth is None and os.name == "nt":
            birth = getattr(metadata, "st_ctime_ns", None)
        values.extend((
            int(metadata.st_dev), int(metadata.st_ino),
            int(metadata.st_mode), int(birth or -1)))
        pointer_digest = -1
        if entry.name == ".git" and stat.S_ISREG(metadata.st_mode):
            pointer_digest = _bounded_identity_digest(entry)
        values.append(pointer_digest)
    return tuple(values)


@functools.lru_cache(maxsize=32)
def _known_repository_identity(anchor: str,
                               anchor_marker: tuple[int, ...]) -> Path:
    """The exact shared Git directory for ``anchor``.

    `git rev-parse --git-common-dir` is the whole point: every linked worktree
    answers with the SAME path, while sibling submodules answer with distinct
    paths below ``.git/modules``. The complete path is the repository identity;
    taking only its parent would collapse those submodules onto one store.

    Successful identities are cached because resolving a selected project
    otherwise starts a subprocess per reading. Failures raise rather than
    return a cached sentinel: a transient Git refusal must be retried instead
    of pinning the process to a fallback namespace until restart. The selected
    root/.git incarnation participates in the cache key, so reassigning a
    linked-worktree path to an unrelated repository cannot reuse a successful
    identity cached for its predecessor.
    """
    del anchor_marker
    try:
        found = subprocess.run(
            toolpath.argv("git", "-C", anchor, "rev-parse",
                          "--path-format=absolute", "--git-common-dir"),
            capture_output=True, text=True, timeout=30,
            **toolpath.no_window())
    except (OSError, subprocess.SubprocessError):
        raise _RepositoryIdentityUnavailable from None
    if found.returncode != 0 or not found.stdout.strip():
        raise _RepositoryIdentityUnavailable
    # `--path-format=absolute` because the bare form answers '.git' from the
    # repository root and an absolute path from a linked worktree, which is a
    # difference this caller would otherwise have to undo by hand. Same idiom as
    # `pr_relay.default_database`.
    return Path(found.stdout.strip()).resolve()


def _repository_identity(anchor: str) -> Path | None:
    """Return a positively cached Git identity, retrying every prior refusal."""
    try:
        return _known_repository_identity(anchor, _anchor_cache_marker(anchor))
    except _RepositoryIdentityUnavailable:
        return None


# A few hermetic callers clear path caches after monkeypatching Git. Preserve
# that small private testing seam while keeping the actual cache positive-only.
setattr(_repository_identity, "cache_clear",
        _known_repository_identity.cache_clear)


@functools.lru_cache(maxsize=32)
def _known_repository_globals(
        anchor: str, marker: tuple[int, ...]) -> Path | None:
    """The ordinary primary checkout's legacy ``globals`` directory.

    A normal repository and all of its linked worktrees share ``<primary>/.git``.
    A submodule or a repository using a separated Git directory does not have
    that shape; guessing ``common_dir.parent`` for either is what made sibling
    submodules collide. Those repositories use the hashed namespace below.
    """
    identity = _repository_identity(anchor)
    if identity is None or identity.name != ".git":
        return None
    primary = identity.parent
    return primary / "globals" if primary.is_dir() else None


def _repository_globals(anchor: str) -> Path | None:
    """Return legacy globals without trusting a stale path assignment."""
    return _known_repository_globals(anchor, _anchor_cache_marker(anchor))


# Preserve the private test seam used by hermetic Git-path fixtures.
setattr(_repository_globals, "cache_clear",
        _known_repository_globals.cache_clear)


def _selected_repository_state_root() -> Path:
    """Per-user state root for repositories that have no legacy Fleet bus.

    The path module already owns the platform convention. This selected-project
    case is deliberately different in a source checkout, where ``GLOBALS_DIR``
    points back into the source repository. An explicit ``ALELYON_HOME`` remains
    authoritative, as it is for every other runtime path.

    SINGLE-VALUED, and that is the whole change
    -------------------------------------------
    Both branches now end in the same ``globals/`` component, which is what
    ``paths`` uses for every branch of its own resolution. They did not: the
    ``ALELYON_HOME`` branch returned ``<home>/globals`` and the other returned
    ``<state root>`` with no component at all, so ONE machine wrote two stores
    depending only on whether ``runtime_env.bootstrap()`` had run in that
    process. That is not a hypothesis — measured on this workstation 2026-08-11,
    ``fleet_repository_paths.sqlite3`` exists at both, 106,496 B at the top level
    and 40,960 B under ``globals/``, with divergent contents.

    Fixing ``paths`` could not reach this function, because it does not read
    ``globals_dir()``: in a source checkout that answer is inside the repository,
    which is the one place this state must not go. ``paths.user_state_home()`` is
    the per-user answer WITH the component, so the two branches agree again.

    **Nothing here migrates, merges, moves or deletes the store that leaves.**
    Which of the two existing files is truth is an owner decision under
    AGENTS.md §6, and adopting one silently is the failure this refuses to
    commit. What it does instead is make the abandoned one VISIBLE — see
    :func:`superseded_selected_state`, which every caller of this function can
    report and none of them may act on.
    """
    from alelyon.runtime.common import paths
    if os.environ.get("ALELYON_HOME"):
        return Path(paths.GLOBALS_DIR)
    return Path(paths.user_state_home())


def _superseded_selected_repository_state_root() -> Path | None:
    """Where this function used to answer, when that is a DIFFERENT directory.

    ``None`` when the two coincide, which is the ``ALELYON_HOME`` branch: it
    already carried the ``globals/`` component and did not move.
    """
    from alelyon.runtime.common import paths
    if os.environ.get("ALELYON_HOME"):
        return None
    return Path(paths._user_state_dir())


#: Files this module used to keep at the superseded root. Named rather than
#: globbed: a glob would also sweep up whatever else a user put there, and this
#: list exists to report OUR stores, not to inventory somebody's directory.
_SELECTED_STATE_NAMES = ("fleet_repository_paths.sqlite3", "fleet_repositories")


def superseded_selected_state() -> tuple[Path, ...]:
    """Selected-repository stores at the OLD root that still exist on disk.

    A named, visible degraded state rather than a silent adoption. When this
    returns a non-empty tuple, those files hold coordination history that this
    build no longer reads and will never read: resolution moved by one
    ``globals/`` component so that it stops producing two answers on one machine.

    It is deliberately NOT a repair. Reading the old store would adopt one side
    of a divergence nobody has adjudicated; deleting it would destroy the other
    side of it; merging them is a state migration and an owner decision under
    AGENTS.md §6. So this reports, and an operator decides.

    Returns an empty tuple both when nothing was left behind and when the old
    root IS the new one. Those are the same fact for a caller — there is nothing
    stranded — which is why they are not distinguished here.
    """
    old = _superseded_selected_repository_state_root()
    if old is None:
        return ()
    new = _selected_repository_state_root()
    try:
        if old.resolve() == new.resolve():
            return ()
    except OSError:
        return ()
    stranded: list[Path] = []
    for name in _SELECTED_STATE_NAMES:
        candidate = old / name
        try:
            if candidate.exists():
                stranded.append(candidate)
        except OSError:
            # An unreadable candidate is not evidence of absence, and saying
            # "nothing stranded" on a failed stat is the one answer that would
            # be worse than saying nothing.
            stranded.append(candidate)
    return tuple(stranded)


def _namespace_metadata(path: Path) -> os.stat_result:
    """Injectable filesystem identity read for namespace falsifiers."""
    return path.stat()


def _repository_namespace(identity: str | Path, *, git_common: bool) -> str:
    """Opaque name for one filesystem incarnation (path only as fallback).

    Per-user state outlives a checkout. A path alone would therefore let an
    unrelated repository recreated at the same location inherit the retired
    checkout's session ids, notes and operation history. Device/file identity
    separates those incarnations while every linked worktree still shares the
    one Git-common directory marker.
    """
    raw = Path(identity)
    if not raw.is_absolute():
        raise RepositoryScopeUnavailable(
            "repository namespace identity must be an absolute path")
    try:
        resolved = raw.resolve()
    except (OSError, RuntimeError) as exc:
        raise RepositoryScopeUnavailable(
            "repository namespace identity is unavailable") from exc
    normalized = os.path.normcase(str(resolved))
    try:
        metadata = _namespace_metadata(resolved)
        device = int(metadata.st_dev)
        inode = int(metadata.st_ino)
        birth = getattr(metadata, "st_birthtime_ns", None)
        if birth is None and os.name == "nt":
            birth = getattr(metadata, "st_ctime_ns", None)
        # Device number and timestamps are shared/coarse. A nonzero inode is
        # the per-object component that licenses a path-independent namespace.
        usable = inode not in (-1, 0)
        incarnation = (f"{device}:{inode}:{int(birth or -1)}"
                       if usable else "metadata-unavailable")
    except (OSError, RuntimeError):
        # Callers use existing selected roots/common dirs. If metadata becomes
        # unreadable between validation and this read, keep the path scoped;
        # downstream opening will refuse rather than fall back to another DB.
        incarnation = "metadata-unavailable"
    kind = "git-common-dir" if git_common else "selected-root"
    # A strong filesystem identity is deliberately path-independent: moving a
    # checkout on the same volume preserves its Fleet history. Only the
    # degraded metadata-unavailable case falls back to the resolved path.
    identity_key = (incarnation if incarnation != "metadata-unavailable"
                    else f"{normalized}\0{incarnation}")
    return hashlib.sha256(
        f"{kind}\0{identity_key}".encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=64)
def _known_repository_context(
        anchor: str, marker: tuple[int, ...]) -> str:
    normalized = os.path.normcase(str(Path(anchor).expanduser().resolve()))
    # Device and birth time alone are not per-object identities: every path on
    # one filesystem can share the former and several entries can share the
    # latter. Require a root/.git inode or the bounded linked-worktree pointer
    # digest. Otherwise include the normalized absolute path and make the
    # degraded scope local to that directory instead of collapsing a volume.
    usable = _repository_context_marker_is_strong(marker)
    material = repr(marker) if usable else f"{normalized}\0{marker!r}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _repository_context_marker_is_strong(marker: tuple[int, ...]) -> bool:
    # marker: root(dev, ino, mode, birth, pointer), then .git in the same form.
    # Only .git can carry a pointer digest, at slot 9.
    return any(marker[index] not in (-1, 0)
               for index in (1, 6, 9) if index < len(marker))


def _repository_context_stamp(anchor: str | Path) -> tuple[str, bool]:
    """Return one content-free checkout id and whether it is path-independent."""
    root = str(Path(anchor).expanduser().resolve())
    marker = _root_incarnation_marker(root)
    return (_known_repository_context(root, marker),
            _repository_context_marker_is_strong(marker))


def repository_context_id(anchor: str | Path) -> str:
    """Opaque identity for the selected checkout incarnation at `anchor`.

    Unlike a resolved path string, this changes when an unrelated checkout is
    placed at the same path. Unlike a directory mtime, it does not change when
    an ordinary source file is added or removed.
    """
    return _repository_context_stamp(anchor)[0]


def _created_at(path: Path) -> int | None:
    """Best available filesystem-incarnation time for one stable marker."""
    try:
        metadata = path.lstat()
    except OSError:
        return None
    birth = getattr(metadata, "st_birthtime_ns", None)
    if birth is not None:
        return int(birth) // 1_000_000_000
    if os.name == "nt":
        return int(metadata.st_ctime_ns) // 1_000_000_000
    # POSIX exposes metadata-change time rather than birth time. Git's stable
    # template markers below are chosen precisely because at least one normally
    # retains its repository-creation ctime for the checkout's lifetime.
    return int(metadata.st_ctime_ns) // 1_000_000_000


def _git_pointer(path: Path, *, prefix: str = "") -> Path | None:
    """Read one bounded Git administrative path file without guessing."""
    try:
        if _is_reparse_like(path) or path.stat().st_size > 4096:
            return None
        value = path.read_text(encoding="utf-8", errors="strict").strip()
    except (OSError, UnicodeError):
        return None
    if prefix:
        if not value.lower().startswith(prefix.lower()):
            return None
        value = value[len(prefix):].strip()
    if not value or len(value) > 4096:
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = path.parent / candidate
    try:
        return candidate.resolve(strict=True)
    except OSError:
        return None


def _linked_worktree_arrival(root: Path, common: Path) -> int | None:
    """When Git last bound this linked worktree to its current path.

    A linked worktree's own ``.git`` file moves with the directory and therefore
    retains its original birth time. Git separately rewrites the admin
    ``worktrees/<name>/gitdir`` pointer when ``git worktree move`` changes the
    checkout path. That bounded structural file is the independent arrival
    signal needed on a first-ever Lattice selection; without it, a transcript
    from the retired occupant of the destination path could look newer than an
    older worktree moved in there.
    """
    dot_git = root / ".git"
    if not dot_git.is_file():
        return None
    admin = _git_pointer(dot_git, prefix="gitdir:")
    if admin is None:
        raise RepositoryScopeUnavailable(
            "the linked-worktree administrative pointer could not be read")
    worktrees = (common / "worktrees").resolve()
    if admin.parent != worktrees:
        # A separated Git directory is not the standard linked-worktree shape;
        # repository inception remains the available lower boundary.
        return None
    reverse = admin / "gitdir"
    pointed_back = _git_pointer(reverse)
    try:
        same_checkout = pointed_back == dot_git.resolve(strict=True)
        metadata = reverse.stat()
    except OSError:
        same_checkout = False
    if not same_checkout:
        raise RepositoryScopeUnavailable(
            "the linked-worktree path binding could not be validated")
    # Transcript timestamps are truncated to whole seconds. The file could have
    # been rewritten at any point inside this second, so the first defensible
    # accepted timestamp is the following second.
    return int(metadata.st_mtime_ns) // 1_000_000_000 + 1


def _binds_to_common(root: Path, common: Path) -> bool | None:
    """Whether ``root`` is a checkout of ``common``, from Git's own pointers.

    True or False when the administrative files answer outright; **None** when
    they do not, which is the caller's signal to fall back to the
    ``git rev-parse --git-common-dir`` probe. None is a real third answer here:
    a separated Git directory and another repository's worktree look alike
    through these files, and only one of them is a refusal.

    This is the binding ``_linked_worktree_arrival`` already validates, read for
    a different purpose. It exists because the probe is a subprocess and its
    callers ask this question once per worktree. Measured on this repository at
    287 worktrees, re-probing cost 286 spawns and 9.5-12.7 s inside
    ``validated_roots`` on every two-second activity tick, and another 287
    inside ``database_for``. Git had already written the answer down; deriving
    it again per path was re-asking a question we had been handed.

    The reverse pointer is what makes this a binding rather than a claim. A
    ``gitdir:`` file left behind by a checkout that has since moved still names
    an administrative directory, but that directory no longer points back.
    """
    dot_git = root / ".git"
    if dot_git.is_dir():
        # A primary checkout: its own ``.git`` IS the shared common directory.
        try:
            return dot_git.resolve() == common
        except OSError:
            return None
    if not dot_git.is_file():
        return False
    admin = _git_pointer(dot_git, prefix="gitdir:")
    if admin is None:
        return None
    try:
        worktrees = (common / "worktrees").resolve()
    except OSError:
        return None
    if admin.parent != worktrees:
        return None
    pointed_back = _git_pointer(admin / "gitdir")
    if pointed_back is None:
        return None
    try:
        return pointed_back == dot_git.resolve(strict=True)
    except OSError:
        return None


def _belongs_to_common(root: Path, common: Path) -> bool:
    """``_binds_to_common``, falling back to the Git probe where it abstains."""
    bound = _binds_to_common(root, common)
    if bound is None:
        return _repository_identity(str(root)) == common
    return bound


def _linked_worktree_binding_marker(root: Path) -> tuple[int, ...]:
    """Cheap revision token for a linked worktree's path binding.

    Moving a linked worktree away and back preserves both the checkout root and
    its ``.git`` file identity. Git does rewrite the reverse administrative
    pointer, however, and that rewrite is the arrival evidence consumed by
    :func:`_linked_worktree_arrival`. Include both bounded structural files in
    the inception cache key so that evidence cannot be hidden by an ABA move.
    """
    dot_git = root / ".git"
    if not dot_git.is_file():
        return ()

    values: list[int] = []

    def extend(path: Path) -> None:
        try:
            metadata = path.lstat()
        except OSError:
            values.extend((-1, -1, -1, -1, -1))
            return
        values.extend((
            int(metadata.st_dev), int(metadata.st_ino),
            int(getattr(metadata, "st_size", 0)),
            int(getattr(metadata, "st_mtime_ns", 0)),
            int(getattr(metadata, "st_ctime_ns", 0)),
        ))

    extend(dot_git)
    admin = _git_pointer(dot_git, prefix="gitdir:")
    if admin is None:
        values.extend((-2, -2, -2, -2, -2))
    else:
        extend(admin / "gitdir")
    return tuple(values)


def _repository_inception_marker(anchor: str | Path) -> tuple[int, ...]:
    root = Path(anchor).expanduser().resolve()
    return (_root_incarnation_marker(str(root))
            + _linked_worktree_binding_marker(root))


@functools.lru_cache(maxsize=64)
def _known_repository_inception(
        anchor: str, marker: tuple[int, ...]) -> int | None:
    del marker
    root = Path(anchor).expanduser().resolve()
    common = _repository_identity(str(root))
    if common is None:
        # A plain directory has no repository-incarnation boundary to apply.
        # A directory carrying Git metadata is different: a transient Git
        # refusal must not become permission to project exact-cwd transcripts.
        if (root / ".git").exists():
            raise RepositoryScopeUnavailable(
                "the selected Git checkout identity could not be read")
        return None
    candidates = (
        common / "description",
        common / "hooks",
        common / "info" / "exclude",
        common / "refs" / "tags",
        common / "objects" / "info",
    )
    times = [stamp for stamp in (_created_at(path) for path in candidates)
             if stamp is not None]
    common_started = min(times) if times else _created_at(common)
    git_marker = root / ".git"
    linked_started = _created_at(git_marker) if git_marker.is_file() else None
    linked_arrived = _linked_worktree_arrival(root, common)
    known = [stamp for stamp in (common_started, linked_started, linked_arrived)
             if stamp is not None]
    if not known:
        raise RepositoryScopeUnavailable(
            "the selected Git checkout inception could not be established")
    return max(known)


def repository_inception(anchor: str | Path) -> int | None:
    """Lower time boundary for activity belonging to this checkout instance."""
    root = str(Path(anchor).expanduser().resolve())
    return _known_repository_inception(root, _repository_inception_marker(root))


_PATH_CONTEXT_SCHEMA = "alelyon.fleet-selected-path-context/0.1"


def _path_context_database() -> Path:
    return _selected_repository_state_root() / "fleet_repository_paths.sqlite3"


def _path_context_key(anchor: str | Path) -> str:
    resolved = Path(anchor).expanduser().resolve()
    normalized = os.path.normcase(str(resolved))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _path_context_boundary(anchor: str | Path, context: str, *,
                           observed_at: int,
                           database: str | Path | None = None) -> int | None:
    """Persist a content-free path assignment and return its arrival floor.

    Repository state is path-independent so moving one checkout preserves its
    Fleet history. Transcript scoping needs the complementary fact: when a
    *different* checkout arrives at a path, records stamped for the retired
    occupant must not follow it. The ledger stores only a path hash, opaque
    context id, and second-resolution observation time -- never the path,
    session ids, or transcript content.

    First observation preserves the repository-inception boundary. A known
    context transition returns the first acceptable whole second after the
    transition was observed. SQLite's immediate transaction makes concurrent
    readers monotonic, and any persistence failure is a refusal rather than a
    fail-open content read.
    """
    target = Path(database) if database is not None else _path_context_database()
    key = _path_context_key(anchor)

    def attempt() -> int | None:
        connection = sqlite3.connect(target, timeout=_SQLITE_TIMEOUT_SECONDS)
        try:
            with connection:
                # WAL first: under the rollback journal this ran on, BEGIN
                # IMMEDIATE below takes an EXCLUSIVE whole-database lock and
                # every reader of this file blocks behind it.
                _DB.try_set_wal(connection)
                connection.execute(
                    f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS selected_path_context ("
                    "path_hash TEXT PRIMARY KEY, context_id TEXT NOT NULL, "
                    "observed_at INTEGER NOT NULL, transitioned INTEGER NOT NULL "
                    "CHECK (transitioned IN (0, 1)), schema TEXT NOT NULL)")
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT context_id, observed_at, transitioned, schema "
                    "FROM selected_path_context WHERE path_hash = ?", (key,)
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO selected_path_context "
                        "(path_hash, context_id, observed_at, transitioned, schema) "
                        "VALUES (?, ?, ?, 0, ?)",
                        (key, context, int(observed_at), _PATH_CONTEXT_SCHEMA))
                    return None
                previous_context, previous_at, transitioned, schema = row
                if schema != _PATH_CONTEXT_SCHEMA:
                    raise RepositoryScopeUnavailable(
                        "the selected-path context ledger schema is unsupported")
                if previous_context == context:
                    return int(previous_at) + 1 if int(transitioned) else None
                transition_at = max(int(previous_at), int(observed_at))
                connection.execute(
                    "UPDATE selected_path_context SET context_id = ?, "
                    "observed_at = ?, transitioned = 1, schema = ? "
                    "WHERE path_hash = ?",
                    (context, transition_at, _PATH_CONTEXT_SCHEMA, key))
                return transition_at + 1
        finally:
            connection.close()

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Bounded retry on SQLITE_BUSY, and bounded is the load-bearing word: a
        # fixed attempt count with capped backoff, never a spin. The whole block
        # is the retry unit because the `with` rolls its transaction back before
        # the next attempt starts, so re-running it from the top is safe.
        return _DB.with_busy_retry(attempt, attempts=_SQLITE_BUSY_ATTEMPTS)
    except RepositoryScopeUnavailable:
        raise
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        raise RepositoryScopeUnavailable(
            "the selected-path context ledger could not be updated") from exc


_PathContextRevision = tuple[str, int, bool]


def _path_context_revision(
        anchor: str | Path, *, database: str | Path | None = None,
        required: bool = False) -> _PathContextRevision | None:
    """Read one content-free durable path revision without creating state."""
    target = Path(database) if database is not None else _path_context_database()
    key = _path_context_key(anchor)

    def attempt():
        uri = target.resolve(strict=True).as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True,
                                     timeout=_SQLITE_TIMEOUT_SECONDS)
        try:
            with connection:
                connection.execute("PRAGMA query_only = ON")
                connection.execute(
                    f"PRAGMA busy_timeout = {_SQLITE_BUSY_TIMEOUT_MS}")
                return connection.execute(
                    "SELECT context_id, observed_at, transitioned, schema "
                    "FROM selected_path_context WHERE path_hash = ?", (key,)
                ).fetchone()
        finally:
            connection.close()

    try:
        row = _DB.with_busy_retry(attempt, attempts=_SQLITE_BUSY_ATTEMPTS)
    except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
        if not required and not target.exists():
            return None
        raise RepositoryScopeUnavailable(
            "the selected-path context ledger could not be read") from exc
    if row is None:
        if not required:
            return None
        raise RepositoryScopeUnavailable(
            "the selected-path context ledger has no selected-path row")
    context, observed_at, transitioned, schema = row
    try:
        observed = int(observed_at)
        transition_flag = int(transitioned)
    except (TypeError, ValueError) as exc:
        raise RepositoryScopeUnavailable(
            "the selected-path context ledger row is malformed") from exc
    if schema != _PATH_CONTEXT_SCHEMA or transition_flag not in (0, 1):
        raise RepositoryScopeUnavailable(
            "the selected-path context ledger schema is unsupported")
    return str(context), observed, bool(transition_flag)


class RepositoryScopeCache:
    """Per-reader repository inception cache with incarnation invalidation.

    Fleet can expose hundreds of linked worktrees. A small process-global LRU
    thrashes at that cardinality and would spawn one Git probe per root on every
    two-second activity tick. This cache is owned by the composite activity
    index, is unbounded only by that index's explicit mesh roots, and rechecks a
    cheap filesystem incarnation token before reusing each boundary.
    """

    def __init__(self, *, context_database: str | Path | None = None) -> None:
        self._entries: dict[
            str,
            tuple[str, int | None, tuple[int, ...], _PathContextRevision],
        ] = {}
        self._context_database = (Path(context_database)
                                  if context_database is not None else None)

    def inception(self, anchor: str | Path, *, now: float | None = None) -> int | None:
        root = str(Path(anchor).expanduser().resolve())
        context = repository_context_id(root)
        marker = _repository_inception_marker(root)
        cached = self._entries.get(root)
        if (cached is not None and cached[0] == context
                and cached[2] == marker):
            revision = _path_context_revision(
                root, database=self._context_database, required=True)
            if revision == cached[3]:
                return cached[1]
            if revision is not None and revision[0] == context:
                arrival = revision[1] + 1 if revision[2] else None
                boundary = cached[1]
                if arrival is not None:
                    boundary = (arrival if boundary is None
                                else max(boundary, arrival))
                self._entries[root] = (
                    context, boundary, marker, revision)
                return boundary
        inception = repository_inception(root)
        if inception is None:
            if (Path(root) / ".git").exists():
                # Keep this check here as well as in the default probe: tests
                # and embedders can inject the repository clock, but ``None``
                # must never become a valid boundary for a known Git checkout.
                raise RepositoryScopeUnavailable(
                    "the selected Git checkout inception is unavailable")
            # Non-Git directory scopes are supported by the standalone parser
            # for compatibility, but are not cached as repository truth. A path
            # that becomes a Git checkout on the next poll must be reprobed.
            return None
        arrival = _path_context_boundary(
            root, context, observed_at=int(time.time() if now is None else now),
            database=self._context_database)
        boundary = max(inception, arrival) if arrival is not None else inception
        revision = _path_context_revision(
            root, database=self._context_database, required=True)
        if revision is None or revision[0] != context:
            raise RepositoryScopeUnavailable(
                "the selected-path context changed during repository scoping")
        self._entries[root] = (context, boundary, marker, revision)
        return boundary

    def validated_roots(self, selected: str | Path | None,
                        candidates: Iterable[str | Path]) -> tuple[
                            tuple[str, ...], tuple[str, ...]]:
        """Current Git checkouts from ``candidates`` belonging to ``selected``.

        A worktree mesh is a point-in-time reading. Between that read and an
        activity poll, one of its paths can be removed or reassigned. Treating
        an ordinary directory (or an unrelated Git checkout) as another exact
        cwd would revive retired transcripts. Validation is strict only when
        the selected scope is itself Git; standalone non-Git parser callers
        retain their historical exact-directory behavior.
        """
        values = tuple(dict.fromkeys(
            str(Path(value).expanduser().resolve())
            for value in candidates if str(value)))
        if not selected:
            return values, ()
        selected_root = Path(selected).expanduser().resolve()
        if not (selected_root / ".git").exists():
            return values, ()
        common = _repository_identity(str(selected_root))
        if common is None:
            raise RepositoryScopeUnavailable(
                "the selected Git checkout identity could not be read")
        accepted: list[str] = []
        refused = 0
        for value in values:
            root = Path(value)
            # `_belongs_to_common` answers from Git's own administrative
            # pointers and only spawns a probe where those cannot decide. The
            # probe used to run per candidate against a 32-entry process-global
            # LRU, so at this cardinality it missed on essentially every
            # lookup -- one Git process per mesh root, every activity tick.
            if not (root / ".git").exists() \
                    or not _belongs_to_common(root, common):
                refused += 1
                continue
            accepted.append(value)
        notes = () if not refused else (
            f"{refused} stale or unrelated mesh root(s) were withheld from "
            "activity because they are not current Git checkouts of the "
            "selected repository.",)
        return tuple(accepted), notes

    def clear(self) -> None:
        self._entries.clear()


def _fallback_database(root: Path) -> Path:
    # A selected-root fallback must follow the complete checkout context, not
    # only the root directory inode. ``git init`` deliberately leaves that
    # inode in place, so a root-only namespace would let the newly-created Git
    # repository inherit coordination rows written before it was a repository.
    # The context includes the structural ``.git`` incarnation and remains
    # stable across transient Git probe failures and ordinary path moves.
    context = repository_context_id(root)
    namespace = hashlib.sha256(
        f"selected-root-context\0{context}".encode("utf-8")).hexdigest()
    return (_selected_repository_state_root() / "fleet_repositories" /
            namespace / "worktree_cache.db")


def _repository_member_roots(common: Path, current: Path) -> tuple[Path, ...]:
    """Known checkout roots sharing ``common``, primary first.

    Git stores a bounded ``gitdir`` pointer for each linked worktree. Reading
    those structural path files lets a recovered Git probe find a fallback DB
    established by any sibling without opening project content.
    """
    members: list[Path] = []
    if common.name == ".git":
        members.append(common.parent.resolve())
    current = current.resolve()
    if current not in members:
        members.append(current)
    worktrees = common / "worktrees"
    try:
        entries = sorted(entry for entry in worktrees.iterdir()
                         if entry.is_dir() and not _is_reparse_like(entry))
    except OSError:
        entries = []
    for entry in entries:
        pointer = entry / "gitdir"
        try:
            if _is_reparse_like(pointer) or pointer.stat().st_size > 4096:
                continue
            with pointer.open("r", encoding="utf-8", errors="strict") as handle:
                value = handle.read(4097).strip()
            if not value or len(value) > 4096:
                continue
            dot_git = Path(value).expanduser().resolve()
        except (OSError, UnicodeError):
            continue
        member = dot_git.parent if dot_git.name == ".git" else None
        if (member is not None and member not in members
                and _belongs_to_common(member, common)):
            members.append(member)
    return tuple(members)


def _established_fallback(common: Path, current: Path) -> Path | None:
    """The one existing fallback, refusing split histories explicitly."""
    candidates = tuple(dict.fromkeys(
        candidate for member in _repository_member_roots(common, current)
        if (candidate := _fallback_database(member)).exists()))
    if len(candidates) > 1:
        raise RepositoryDatabaseAmbiguity(
            "multiple repository fallback stores exist after Git recovery; "
            "Fleet refuses to choose one and hide the other")
    return candidates[0] if candidates else None


def _is_reparse_like(path: Path) -> bool:
    """Whether ``path`` can redirect traversal to another filesystem path.

    ``is_symlink`` covers POSIX links and Windows symbolic links; Python 3.12's
    ``is_junction`` and the Windows file-attribute check cover junctions and
    other reparse points. An entry that cannot be inspected is unsafe to adopt.
    """
    try:
        metadata = path.lstat()
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        if is_junction is not None and is_junction():
            return True
    except OSError:
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _resolved_path_within(path: Path, boundary: Path) -> bool:
    """Whether two existing paths resolve with ``path`` beneath ``boundary``."""
    try:
        path.resolve(strict=True).relative_to(boundary.resolve(strict=True))
    except (OSError, ValueError):
        return False
    return True


def _safe_legacy_database(legacy_globals: Path) -> Path | None:
    """Return an adoptable repository-local legacy database, else ``None``.

    A selected repository controls these path entries. It may supply an ordinary
    existing database for compatibility, but it may not redirect Lattice through
    a symlink, junction, or other reparse point to a database outside the selected
    checkout.
    """
    database = legacy_globals / "worktree_cache.db"
    if not database.is_file():
        return None
    primary = legacy_globals.parent
    if _is_reparse_like(legacy_globals) or _is_reparse_like(database):
        return None
    if not _resolved_path_within(legacy_globals, primary):
        return None
    if not _resolved_path_within(database, primary):
        return None
    # A filename is not a compatibility contract. Read the marker and complete
    # table vocabulary through a query-only connection before allowing normal
    # Fleet collection to mutate this file.
    # This probe used to keep its own 1.0s budget, from before the shared-store
    # budgets were raised. It follows them now, and the reason is its FAILURE
    # MODE rather than its cost: every exception below returns None, `database_
    # for` then hands back the scoped path instead of this one, and the caller
    # writes its findings and claims to a DIFFERENT FILE than its peers without
    # anything saying so. That is the stranded-bus outcome `default_database`
    # documents -- twelve private databases holding 70 findings nobody could
    # see -- reached this time by a transient lock rather than by a wrong path.
    # A loud refusal at one second would be a defensible trade; a silent
    # divergence at one second is not.
    #
    # MEASURED 2026-08-11, on a fixture holding the lock for 3s. A reader is not
    # blocked by an ordinary writer at all: `BEGIN IMMEDIATE` takes RESERVED,
    # which is compatible with the SHARED lock this takes, and the probe
    # returned in 0.03-0.05s at either budget. What blocks it is EXCLUSIVE --
    # taken while a commit writes back, and by checkpoint and VACUUM. Under a
    # held `BEGIN EXCLUSIVE` the 1.0s budget DECLINED adoption after 1.76s while
    # 5.0s adopted correctly after 2.62s.
    #
    # The bounded retry that `_read_scope_row` wraps around its writes is NOT
    # copied here. `timeout` already re-tries a BUSY internally for the whole
    # budget, so an outer `_SQLITE_BUSY_ATTEMPTS` loop would multiply this to
    # ~20s on a path a mesh read waits for, to answer a question whose wrong
    # answer is corrected at the next incarnation. The budget is the fix; a
    # retry on top of it is not proportionate.
    #
    # The busy_timeout is derived from the same constant rather than written
    # out, so "how long may this block" cannot drift into two numbers.
    try:
        uri = database.resolve(strict=True).as_uri() + "?mode=ro"
        with sqlite3.connect(uri, uri=True,
                             timeout=_SQLITE_TIMEOUT_SECONDS) as connection:
            connection.execute(
                f"PRAGMA busy_timeout = {int(_SQLITE_TIMEOUT_SECONDS * 1000)}")
            connection.execute("PRAGMA query_only = ON")
            tables = {str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
            cache_schema = (connection.execute(
                "SELECT value FROM meta WHERE name = 'schema'").fetchone()
                if "meta" in tables else None)
            bus_schema = (connection.execute(
                "SELECT value FROM bus_meta WHERE name = 'schema'").fetchone()
                if "bus_meta" in tables else None)
    except (OSError, sqlite3.Error, ValueError):
        return None
    cache_tables = {"meta", "worktree", "snapshot", "operation", "declaration"}
    cache_ok = (cache_schema is not None
                and str(cache_schema[0]) == str(SCHEMA_VERSION)
                and cache_tables <= tables)
    try:
        from alelyon.runtime.common import worktree_bus
        bus_version = int(bus_schema[0]) if bus_schema is not None else 0
        bus_tables = {"bus_meta", "finding", "delivery", "claim"}
        bus_ok = (1 <= bus_version <= worktree_bus.BUS_SCHEMA_VERSION
                  and bus_tables <= tables
                  and (bus_version < 2
                       or {"channel", "membership"} <= tables))
    except (ImportError, TypeError, ValueError):
        bus_ok = False
    if not (cache_ok or bus_ok):
        return None

    # A tracked database is source, not runtime state. Opening a selected
    # arbitrary repository must never dirty one of its committed files merely
    # because it reused Alelyon's historical filename.
    try:
        relative = database.resolve(strict=True).relative_to(
            primary.resolve(strict=True)).as_posix()
        tracked = subprocess.run(
            toolpath.argv("git", "-C", str(primary), "ls-files",
                          "--error-unmatch", "--", relative),
            capture_output=True, text=True, timeout=30,
            **toolpath.no_window())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if tracked.returncode == 0:
        return None
    if tracked.returncode != 1:
        # Only Git's specific "path is not tracked" result licenses legacy
        # adoption. A fatal/refusal code is missing evidence, not evidence that
        # the file is safe runtime state.
        return None
    return database


def database_for(anchor: str | Path) -> Path:
    """The shared fleet database for the repository containing `anchor`.

    Unlike `default_database`, this does not inherit the process/import
    checkout. Packaged Lattice supplies the repository the user selected, so
    two open projects cannot silently share coordination rows. An existing
    ordinary ``<primary>/globals/worktree_cache.db`` remains authoritative so a
    packaged application can see the repository's current Fleet bus. Otherwise
    the exact Git common directory -- or exact selected root when Git refuses --
    selects an opaque per-user namespace. This function only computes a path;
    it creates no directory or database.
    """
    root = Path(anchor).expanduser().resolve()
    fallback_database = _fallback_database(root)
    common = _repository_identity(str(root))
    if common is None:
        if fallback_database.exists():
            # Compatibility for a fallback established by an older release.
            # New fallbacks are not created while a Git checkout's common-dir
            # identity is unavailable: an established scoped store may exist
            # under that unknown identity, and choosing another DB would split
            # one repository's coordination history.
            return fallback_database
        if (root / ".git").exists():
            raise RepositoryDatabaseUnavailable(
                "Git refused the selected repository identity; Fleet will not "
                "create a second coordination database")
        return fallback_database
    established = _established_fallback(common, root)
    if established is not None:
        return established
    identity = common if common is not None else root
    namespace = _repository_namespace(identity, git_common=common is not None)
    scoped_database = (_selected_repository_state_root() / "fleet_repositories" /
                       namespace / "worktree_cache.db")
    # Once this repository has created its scoped store, that path is its stable
    # identity. A legacy file appearing later must not silently switch the
    # running project's history to a different database.
    if scoped_database.exists():
        return scoped_database

    legacy_globals = _repository_globals(str(root)) if common is not None else None
    if legacy_globals is not None:
        legacy_database = _safe_legacy_database(legacy_globals)
        if legacy_database is not None:
            return legacy_database

    return scoped_database


def default_database() -> Path:
    """The fleet bus, anchored on the REPOSITORY rather than on this worktree.

    `GLOBALS_DIR` is "the directory holding `pyproject.toml`", which is right
    for state a checkout owns -- and wrong for this file, because a linked
    worktree passes that test too. It is a real checkout with its own
    `pyproject.toml`, so every session working in one got a PRIVATE bus.

    That silently defeats the thing this database is for. Findings, claims and
    deliveries are cross-session by definition, and working in a worktree is the
    documented pattern, so the recommended workflow was the one that guaranteed
    nobody could hear you. Measured 2026-08-05: 12 worktree-local databases held
    70 findings and 22 claims invisible to the shared bus, while `publish`
    reported "reached N session(s)" to each of their authors.

    The repository is the correct scope and git already knows it: every linked
    worktree reports the same `--git-common-dir`. An installed wheel has no git
    and no checkout, so it keeps the per-user `GLOBALS_DIR` answer -- that case
    was never the broken one.

    **This is not a new decision.** `pr_relay.default_database` already anchors
    on `--git-common-dir` for exactly this reason and says so: "Deliberately not
    `GLOBALS_DIR`, which resolves to `<repo>/globals` and therefore gives each
    worktree its own file." The rule was written down; the bus never got it,
    because it inherited this module's location and this module's OTHER schemas
    -- colour slots, occupancy declarations -- genuinely are per-checkout.

    This does NOT migrate the stranded rows. A finding is a claim about a moment
    and some of those are long closed; folding twelve private histories into the
    shared one would resurrect them as live work. `stranded_buses()` reports
    them so the choice is somebody's rather than this function's.
    """
    from alelyon.runtime.common.paths import GLOBALS_DIR
    shared = _repository_globals(str(Path(GLOBALS_DIR).resolve()))
    if shared is not None:
        return shared / "worktree_cache.db"
    return Path(GLOBALS_DIR) / "worktree_cache.db"


def stranded_buses(repo_root: str | Path = ".") -> tuple:
    """Per-worktree databases holding rows the shared bus cannot see.

    Reported rather than merged, and never emptied: this says what was lost, so
    that a session which spent an afternoon talking to itself can find out.
    """
    shared = database_for(repo_root).resolve()
    out = []
    for tree in observe(repo_root).worktrees:
        candidate = (Path(tree.path) / "globals" / "worktree_cache.db").resolve()
        if candidate == shared or not candidate.exists():
            continue
        try:
            with sqlite3.connect(f"file:{candidate}?mode=ro", uri=True) as conn:
                tables = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                if "finding" not in tables:
                    continue
                findings = conn.execute("SELECT COUNT(*) FROM finding").fetchone()[0]
                claims = (conn.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
                          if "claim" in tables else 0)
        except sqlite3.Error:
            continue
        if findings or claims:
            out.append((str(candidate), int(findings), int(claims)))
    return tuple(sorted(out))


def record_now(repo_root: str | Path = ".", *, database: str | Path | None = None,
               mainline: str = "origin/main") -> tuple[WorktreeCache, WorktreeMesh,
                                                       tuple[Operation, ...]]:
    """Observe and fold into the cache in one call."""
    cache = WorktreeCache(database or database_for(repo_root))
    mesh = observe(repo_root, mainline=mainline)
    return cache, mesh, cache.record(mesh)


__all__ = [
    "COLOUR_SLOTS", "FOREIGN_OCCUPANT", "LOW_CONTRAST_LIGHT", "SCHEMA_VERSION",
    "SHARED_OCCUPANCY", "UNATTRIBUTED", "UNSLOTTED", "Declaration", "Occupancy",
    "Operation", "RepositoryDatabaseAmbiguity", "RepositoryDatabaseUnavailable",
    "RepositoryScopeCache",
    "RepositoryScopeUnavailable",
    "WorktreeCache",
    "WorktreeIdentity",
    "database_for", "default_database", "record_now", "repository_context_id",
    "repository_inception", "stranded_buses", "superseded_selected_state",
]
