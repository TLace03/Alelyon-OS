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

from dataclasses import dataclass
import functools
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

from alelyon.runtime.common import toolpath
from alelyon.runtime.common.worktree import (
    UNATTRIBUTED, WorktreeMesh, observe, session_for_path,
)

SCHEMA_VERSION = 1

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
        with self._connect() as conn:
            for statement in _DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO meta(name, value) VALUES('schema', ?)",
                (str(SCHEMA_VERSION),))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    # ── writing ─────────────────────────────────────────────────────────────

    def record(self, mesh: WorktreeMesh) -> tuple[Operation, ...]:
        """Fold one observation into the cache and return what changed.

        The diff against the previous snapshot is what makes per-worktree change
        tracking possible: a single observation is a state, and two are a history.
        """
        operations: list[Operation] = []
        with self._connect() as conn:
            seen_keys = set()
            for tree in mesh.worktrees:
                key = _key_for(mesh.repo_root, tree.path)
                seen_keys.add(key)
                identity = self._upsert_identity(conn, key, tree, mesh.observed_at)
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
                del identity
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
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO declaration(key, at, session_id, model, note)"
                " VALUES(?,?,?,?,?)", (key, moment, session_id, model, note))
        return Declaration(key=key, at=moment, session_id=session_id,
                           model=model, note=note)

    # ── reading ─────────────────────────────────────────────────────────────

    def identities(self) -> tuple[WorktreeIdentity, ...]:
        with self._connect() as conn:
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
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM operation WHERE key=? ORDER BY at DESC, rowid DESC"
                " LIMIT ?", (key, limit)).fetchall()
        return tuple(Operation(key=r["key"], at=r["at"], kind=r["kind"],
                               detail=r["detail"],
                               paths=tuple(json.loads(r["paths"]))) for r in rows)

    def recent(self, *, limit: int = 100) -> tuple[Operation, ...]:
        """The collective feed across every worktree, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM operation ORDER BY at DESC, rowid DESC LIMIT ?",
                (limit,)).fetchall()
        return tuple(Operation(key=r["key"], at=r["at"], kind=r["kind"],
                               detail=r["detail"],
                               paths=tuple(json.loads(r["paths"]))) for r in rows)

    def declarations(self, key: str | None = None) -> tuple[Declaration, ...]:
        with self._connect() as conn:
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
        with self._connect() as conn:
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
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT d.key, d.session_id FROM declaration d "
                "LEFT JOIN worktree w ON w.key = d.key WHERE w.key IS NULL"
            ).fetchall()
        return tuple(
            f"session {r['session_id']!r} declared work on worktree {r['key']}, "
            f"which git has never listed in this repository" for r in rows)

    def colour_capacity(self) -> tuple[int, int]:
        """(slots held, slots available). Past capacity, colour stops helping."""
        with self._connect() as conn:
            held = conn.execute(
                "SELECT COUNT(*) c FROM worktree WHERE colour_slot != ?",
                (UNSLOTTED,)).fetchone()["c"]
        return int(held), len(COLOUR_SLOTS)


@functools.lru_cache(maxsize=8)
def _repository_globals(anchor: str) -> Path | None:
    """`<primary checkout>/globals` for the repository `anchor` sits in, or None.

    `git rev-parse --git-common-dir` is the whole point: every linked worktree
    answers with the SAME `.git`, which is exactly the scope this database is
    supposed to have. `--git-dir` would answer per-worktree and reintroduce the
    split.

    Cached because `default_database()` is called per operation and this is a
    subprocess. None means "not a checkout" -- an installed wheel -- and the
    caller falls back.
    """
    try:
        found = subprocess.run(
            toolpath.argv("git", "-C", anchor, "rev-parse",
                          "--path-format=absolute", "--git-common-dir"),
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if found.returncode != 0 or not found.stdout.strip():
        return None
    # `--path-format=absolute` because the bare form answers '.git' from the
    # repository root and an absolute path from a linked worktree, which is a
    # difference this caller would otherwise have to undo by hand. Same idiom as
    # `pr_relay.default_database`.
    root = Path(found.stdout.strip()).resolve().parent
    return root / "globals" if root.is_dir() else None


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
    shared = _repository_globals(str(GLOBALS_DIR))
    if shared is not None:
        return shared / "worktree_cache.db"
    return Path(GLOBALS_DIR) / "worktree_cache.db"


def stranded_buses(repo_root: str | Path = ".") -> tuple:
    """Per-worktree databases holding rows the shared bus cannot see.

    Reported rather than merged, and never emptied: this says what was lost, so
    that a session which spent an afternoon talking to itself can find out.
    """
    shared = default_database().resolve()
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
    cache = WorktreeCache(database or default_database())
    mesh = observe(repo_root, mainline=mainline)
    return cache, mesh, cache.record(mesh)


__all__ = [
    "COLOUR_SLOTS", "FOREIGN_OCCUPANT", "LOW_CONTRAST_LIGHT", "SCHEMA_VERSION",
    "SHARED_OCCUPANCY", "UNATTRIBUTED", "UNSLOTTED", "Declaration", "Occupancy",
    "Operation", "WorktreeCache", "WorktreeIdentity",
    "default_database", "record_now", "stranded_buses",
]
