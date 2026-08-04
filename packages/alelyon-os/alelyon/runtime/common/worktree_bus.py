"""Sessions telling each other what they found — and territory, so they stop
landing on each other.

Two agent sessions, each behaving correctly in isolation, produce work that
cannot be reconciled because neither could see the other. That is
[DYNAMIC-CACHE.md](../../../docs/features/DYNAMIC-CACHE.md) §2's stated failure,
and the worktree mesh already *observes* it. Observation is not communication:
the mesh can say two worktrees touch the same file, and cannot say **why**, or
what the other session already knows.

This module is the missing half. It carries two things over the same store.

Findings
--------
A session publishes what it learned; other sessions read it. The load-bearing
design decision is **which half is trusted**:

* the **body is DECLARED** — a session says it, nothing checks it, and it is
  labelled as self-reported at every read;
* the **routing is DERIVED** — who a finding reaches is computed from the mesh's
  observed touched-path sets, not from the publisher's opinion of who should
  care.

That split is §5's rule ("validate against an independently-held invariant, never
against the shape of what the writer emitted") applied to a message bus. A
publisher cannot address a session that is not demonstrably working in the area,
because the address is not something the publisher writes: it writes a *subject*,
and the mesh resolves subjects to sessions. Every delivery carries the rule that
produced it, so a reader can disagree with the routing rather than only with the
message.

The exception is `to_session=`, which is an explicit address and is recorded as
`DECLARED` routing. It exists because a session sometimes genuinely knows who it
is answering, and pretending otherwise would push people to fake a subject to
reach a person.

This is the case the feature was built from, observed on 2026-08-03: one session
refactored `assistant/tools.py` and `engine.py` into a domain seam while another
had verified work in flight over the same files. The second session's suite began
failing with `NameError: name '_route' is not defined` — a real symptom of a
half-applied refactor that was nobody's defect. A finding addressed *by subject*
would have reached exactly the session that needed it, without either knowing the
other existed.

Territory
---------
A claim on an `Area`. `open_areas()` reports the areas no live worktree is
touching and no session has claimed, which is what lets a session be told "work
where you are needed" and find somewhere real to start.

**A claim is advisory and this is not negotiable.** §6.4: *a cache is not a lock*.
Two sessions can claim the same area in the same second; both writes land, and
`contested()` reports it rather than either being silently refused. Anything that
presented a claim as exclusive would be inventing a guarantee the substrate
cannot make, and the first agent that never registers would break it anyway.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from alelyon.runtime.common.worktree_areas import (
    Area, area_of, areas_of, parse_area,
)
from alelyon.runtime.common.worktree import UNATTRIBUTED, WorktreeMesh

#: Bumped independently of `worktree_cache.SCHEMA_VERSION`: the two share a file
#: but not a lifecycle, and coupling their versions would force a migration of
#: one whenever the other changed.
BUS_SCHEMA_VERSION = 1

#: How a delivery was arrived at. Mirrors `worktree_graph.PROVENANCE` deliberately
#: — a reader who has learned one vocabulary should not need a second.
DERIVED = "DERIVED"
DECLARED = "DECLARED"

#: What kind of thing a finding is. A closed vocabulary, because an inbox that
#: sorts by urgency needs to know which of these outrank the others, and free
#: text cannot be ordered. Adding one is a deliberate act.
KIND_REFACTOR = "refactor-in-flight"   # I am mid-change; expect transient breakage
KIND_INTERFACE = "interface-changed"   # a signature/import others depend on moved
KIND_DEFECT = "defect-found"           # a real bug, possibly not mine to fix
KIND_CONVENTION = "convention"         # a rule or pattern others should follow
KIND_BLOCKED = "blocked-on"            # I cannot proceed until something changes
KIND_LANDED = "landed"                 # work reached a branch others can build on
KINDS = (KIND_REFACTOR, KIND_INTERFACE, KIND_DEFECT, KIND_CONVENTION,
         KIND_BLOCKED, KIND_LANDED)

#: Inbox ordering. `refactor-in-flight` and `interface-changed` outrank the rest
#: because they are the two that make another session's *correct* work start
#: failing — the reader needs them before they debug something that is not theirs.
_URGENCY = {KIND_REFACTOR: 0, KIND_INTERFACE: 0, KIND_BLOCKED: 1,
            KIND_DEFECT: 2, KIND_CONVENTION: 3, KIND_LANDED: 4}

#: A finding nobody acknowledged eventually stops being news. Read-time only:
#: nothing is deleted, because the history is the point (§9 answer 1).
DEFAULT_INBOX_AGE_DAYS = 14.0

_BUS_LIMITS: tuple[str, ...] = (
    "A finding's body is self-reported. Nothing here checks that it is true, "
    "accurate, or still current, and a stale finding looks exactly like a fresh "
    "one except for its timestamp.",
    "Routing reaches only sessions the mesh can see. A session working outside "
    "every known worktree convention receives nothing and is not counted as "
    "having been told.",
    "Two sessions sharing one checkout are one path and therefore one derived "
    "identity. Nothing in git separates them, so they reach each other only by "
    "claiming an area or declaring occupancy (both self-reports), and not at "
    "all if neither does.",
    "A claim is advisory. Two sessions can hold the same area at once; this "
    "records that and does not prevent it.",
    "Silence is not consent. An unacknowledged finding means nobody pressed the "
    "button, not that nobody was affected.",
)


@dataclass(frozen=True)
class Finding:
    """One thing a session learned and thought another would need."""

    id: str
    at: int
    kind: str
    #: Who published it. Self-reported unless it was derived from the publisher's
    #: worktree path, which `published_evidence` records either way.
    from_session: str
    from_evidence: str
    #: Free text. DECLARED — see the module docstring.
    body: str
    #: What the finding is ABOUT. Routing resolves these to sessions; they are
    #: repository-relative paths, and the areas are derived from them.
    subject_paths: tuple[str, ...] = ()
    #: An explicit address, when the publisher named one. Empty for subject
    #: routing, which is the normal case.
    to_session: str = ""
    #: Set when the publisher addressed an area rather than paths.
    to_area: str = ""
    broadcast: bool = False
    severity: str = "info"

    def areas_in(self, space=None) -> tuple[Area, ...]:
        """Where this finding lands in ``space``'s coordinate vocabulary.

        Routing is derived from `subject_paths`, so which rules place them
        decides who hears the finding. A bus over another repository must pass
        that repository's space; `areas` below keeps the process default for
        every caller standing in its own checkout.
        """
        if self.to_area:
            return (parse_area(self.to_area),)
        return areas_of(self.subject_paths, space)

    @property
    def areas(self) -> tuple[Area, ...]:
        """Where this finding lands, in the process-default space."""
        return self.areas_in(None)

    @property
    def urgency(self) -> int:
        return _URGENCY.get(self.kind, 9)


@dataclass(frozen=True)
class Delivery:
    """One finding reaching one session, and the rule that sent it there."""

    finding: Finding
    to_session: str
    provenance: str          # DERIVED | DECLARED
    #: Why this session. Written for a reader who wants to disagree with the
    #: routing: "you have uncommitted edits in <paths>", not "matched".
    reason: str
    acknowledged_at: int | None = None

    @property
    def acknowledged(self) -> bool:
        return self.acknowledged_at is not None


@dataclass(frozen=True)
class Claim:
    """A session's advisory hold on an area."""

    area: str
    session_id: str
    at: int
    note: str
    released_at: int | None = None

    @property
    def active(self) -> bool:
        return self.released_at is None


@dataclass(frozen=True)
class AreaState:
    """What the fleet is doing in one area — the answer to "is this free?"."""

    area: Area
    #: Sessions with outstanding edits in this area, derived from the mesh.
    working: tuple[str, ...] = ()
    #: Worktree labels touching it, for a reader whose sessions are UNATTRIBUTED.
    worktrees: tuple[str, ...] = ()
    #: Sessions holding an active claim. Self-reported.
    claimed_by: tuple[str, ...] = ()
    open_findings: int = 0

    @property
    def occupied(self) -> bool:
        """Anything at all is happening here."""
        return bool(self.working or self.worktrees or self.claimed_by)

    @property
    def contested(self) -> bool:
        """More than one session holds it — a claim is not a lock."""
        return len(set(self.claimed_by)) > 1


_DDL = (
    """CREATE TABLE IF NOT EXISTS bus_meta (
           name TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS finding (
           id TEXT PRIMARY KEY,
           at INTEGER NOT NULL,
           kind TEXT NOT NULL,
           from_session TEXT NOT NULL,
           from_evidence TEXT NOT NULL,
           body TEXT NOT NULL,
           subject_paths TEXT NOT NULL,
           to_session TEXT NOT NULL,
           to_area TEXT NOT NULL,
           broadcast INTEGER NOT NULL,
           severity TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS delivery (
           finding_id TEXT NOT NULL,
           to_session TEXT NOT NULL,
           provenance TEXT NOT NULL,
           reason TEXT NOT NULL,
           at INTEGER NOT NULL,
           acknowledged_at INTEGER,
           PRIMARY KEY (finding_id, to_session))""",
    """CREATE TABLE IF NOT EXISTS claim (
           area TEXT NOT NULL,
           session_id TEXT NOT NULL,
           at INTEGER NOT NULL,
           note TEXT NOT NULL,
           released_at INTEGER,
           PRIMARY KEY (area, session_id, at))""",
    "CREATE INDEX IF NOT EXISTS finding_at ON finding(at)",
    "CREATE INDEX IF NOT EXISTS delivery_session ON delivery(to_session)",
    "CREATE INDEX IF NOT EXISTS claim_area ON claim(area)",
)


def _finding_id(at: int, session: str, kind: str, body: str) -> str:
    """Content-derived, so republishing the same finding twice is detectable.

    Includes the timestamp: a session that genuinely learns the same thing again
    later has news, and collapsing that into the first report would hide a
    recurrence behind a duplicate check.
    """
    material = f"{at}\x1f{session}\x1f{kind}\x1f{body}".encode("utf-8")
    return hashlib.blake2b(material, digest_size=8).hexdigest()


def _now() -> int:
    return int(time.time())


class FleetBus:
    """Findings and territory over one SQLite file, opened per operation.

    Shares `worktree_cache`'s database by default — DYNAMIC-CACHE.md §1's "two
    caches, one substrate" — with its own tables and its own schema version.
    """

    def __init__(self, database: str | Path, *, space=None) -> None:
        """``space`` is the coordinate vocabulary of the repository being
        observed.

        Explicit rather than ambient, because the two can differ and the
        difference is silent. `worktree_areas` resolves a default from the
        checkout the PROCESS is standing in, which is right for a tool run
        inside the repository it is asking about and wrong the moment a caller
        points at another one -- `--repo` does exactly that, and the public CLI
        exists so users can point it at directories they select. A bus built
        over repository X must place X's paths with X's rules, or every path
        reads UNMAPPED and the fleet sees an empty repository.

        ``None`` keeps the process default, so every existing caller standing in
        its own checkout is unchanged.
        """
        self.database = Path(database)
        self.space = space
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for statement in _DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO bus_meta(name, value) VALUES('schema', ?)",
                (str(BUS_SCHEMA_VERSION),))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    @property
    def limits(self) -> tuple[str, ...]:
        """What this bus cannot establish. Carried so no reader can render its
        output without the caveats being available to them."""
        return _BUS_LIMITS

    def _space(self):
        """This bus's coordinate space, or the process default."""
        from alelyon.runtime.common.worktree_areas import default_space
        return self.space if self.space is not None else default_space()

    # ── publishing ──────────────────────────────────────────────────────────
    def publish(self, *, kind: str, body: str, from_session: str,
                from_evidence: str = "self-reported", mesh: WorktreeMesh | None = None,
                subject_paths=(), to_session: str = "", to_area: str = "",
                broadcast: bool = False, severity: str = "info",
                cache=None,
                at: int | None = None) -> tuple[Finding, tuple[Delivery, ...]]:
        """Record a finding and resolve who it reaches.

        Returns the finding and its deliveries, so a caller can tell the
        publisher *"this reached nobody"* — which is a real and common outcome
        worth saying out loud, rather than letting a message vanish into a table
        and look sent.
        """
        if kind not in KINDS:
            raise ValueError(f"unknown finding kind {kind!r}; expected one of "
                             f"{', '.join(KINDS)}")
        body = " ".join(str(body or "").split())
        if not body:
            raise ValueError("a finding with no body is not a finding")
        if not (subject_paths or to_session or to_area or broadcast):
            raise ValueError(
                "a finding needs an audience: subject paths, an area, an "
                "explicit session, or broadcast. Refusing rather than "
                "defaulting to broadcast, which would make every unaddressed "
                "note interrupt everybody.")
        at = _now() if at is None else int(at)
        paths = tuple(dict.fromkeys(
            str(p).replace("\\", "/") for p in (subject_paths or ()) if str(p).strip()))
        finding = Finding(
            id=_finding_id(at, from_session, kind, body), at=at, kind=kind,
            from_session=from_session or UNATTRIBUTED, from_evidence=from_evidence,
            body=body, subject_paths=paths, to_session=to_session,
            to_area=to_area, broadcast=bool(broadcast), severity=severity)
        deliveries = self._route(finding, mesh, cache)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO finding
                   (id, at, kind, from_session, from_evidence, body,
                    subject_paths, to_session, to_area, broadcast, severity)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (finding.id, finding.at, finding.kind, finding.from_session,
                 finding.from_evidence, finding.body,
                 json.dumps(list(finding.subject_paths)), finding.to_session,
                 finding.to_area, int(finding.broadcast), finding.severity))
            for delivery in deliveries:
                conn.execute(
                    """INSERT OR IGNORE INTO delivery
                       (finding_id, to_session, provenance, reason, at)
                       VALUES (?,?,?,?,?)""",
                    (finding.id, delivery.to_session, delivery.provenance,
                     delivery.reason, at))
        return finding, deliveries

    def _route(self, finding: Finding, mesh: WorktreeMesh | None,
               cache=None) -> tuple[Delivery, ...]:
        """Resolve a finding's subject to the sessions it reaches.

        The whole honesty property of this module lives in this method: with the
        single exception of an explicit `to_session`, the publisher does not get
        to choose the recipients. The mesh's observed touched-path sets do.
        """
        out: list[Delivery] = []
        wanted = set(finding.areas_in(self.space))
        if finding.to_session:
            # An explicit address is not checked against anything, which means a
            # typo or a half-remembered id reports "reached 1 session" for a
            # session that has never existed. Caught by making exactly that
            # mistake: an 8-character prefix padded with zeros was accepted,
            # counted as delivered, and sat in the table addressed to nobody.
            #
            # Not refused, because a real session can be invisible here -- it may
            # work outside every path convention the mesh knows. Named instead,
            # so the publisher can tell "delivered" from "filed against a string".
            unknown = ""
            if mesh is not None:
                known = {w.session for w in mesh.worktrees
                         if w.session != UNATTRIBUTED}
                if finding.to_session not in known:
                    unknown = (". NOTE: this session id matches no worktree the "
                               "mesh can see, so it may be a typo, or a session "
                               "working outside every known convention")
            out.append(Delivery(
                finding, finding.to_session, DECLARED,
                "addressed explicitly by the publisher; not derived from any "
                "observed work" + unknown))

        # A claim doubles as a subscription, and it has to, because derived
        # routing has one blind spot big enough to make the feature useless
        # without it: **two sessions sharing the primary checkout**. Session
        # identity is derived from the worktree PATH, so two agents working in
        # one directory are one path and one identity — UNATTRIBUTED, if the
        # checkout sits at no session-carrying location. That is not a corner
        # case; it is what was actually observed in this repository on
        # 2026-08-03, and a live publish reached nobody because of it.
        #
        # There is no git record that separates them, so nothing here can derive
        # it. What a session CAN do is say where it is working, which is what a
        # claim already is. Routing therefore honours claims — labelled DECLARED,
        # never merged with observed work, and worth exactly what a self-report
        # is worth.
        for claim in self.active_claims():
            if claim.session_id == finding.from_session:
                continue
            if parse_area(claim.area) in wanted:
                out.append(Delivery(
                    finding, claim.session_id, DECLARED,
                    f"you claimed {claim.area}, which this finding is about; "
                    f"that is your own declaration, not observed work"))

        if mesh is None:
            return tuple(out)

        # Declared OCCUPANTS of a worktree, from `worktree_cache.declare()`.
        #
        # This is the better half of the shared-tree answer and it belongs here
        # rather than beside the claim above, because it is a *hybrid*: the
        # session identity is declared, but the paths it is matched on are the
        # tree's own observed edits. A claim is "I say I am working in this
        # area"; an occupancy declaration is "I say I am in this tree", and the
        # tree then says what it has actually changed. Half the evidence is
        # still independent of the writer, which is more than a claim offers.
        #
        # It stays DECLARED regardless, because the half that decides *who* is
        # self-reported and §5 grades a compound by its weakest link.
        occupants: dict[str, set[str]] = {}
        if cache is not None:
            try:
                paths = {i.key: i.path for i in cache.identities()}
                for declaration in cache.declarations():
                    location = paths.get(declaration.key)
                    if location and declaration.session_id:
                        occupants.setdefault(location, set()).add(
                            declaration.session_id)
            except Exception:  # noqa: BLE001 — a bus must not die on a cache
                occupants = {}
        for worktree in mesh.worktrees:
            touched_here = worktree.touched_paths
            for occupant in sorted(occupants.get(worktree.path, ())):
                if occupant == finding.from_session or not touched_here:
                    continue
                overlap = self._overlap(finding, touched_here, wanted, self.space)
                if overlap:
                    out.append(Delivery(
                        finding, occupant, DECLARED,
                        f"you declared work in {worktree.label}, and that tree "
                        f"has outstanding edits this finding is about "
                        f"({overlap.split(',')[0]}). The tree's edits are "
                        f"observed; that you are in it is your own report"))

            session = worktree.session
            if session == UNATTRIBUTED or session == finding.from_session:
                # A session cannot be told its own news, and a worktree whose
                # session could not be derived is NOT silently addressed by its
                # tool family — that would deliver to whoever happens to share a
                # vendor, which is not the same audience at all.
                continue
            touched = worktree.touched_paths
            if not touched:
                continue
            if finding.broadcast:
                out.append(Delivery(
                    finding, session, DERIVED,
                    f"broadcast; {worktree.label} has {len(touched)} path(s) "
                    f"outstanding, so this session is live"))
                continue
            overlap = self._overlap(finding, touched, wanted, self.space)
            if overlap:
                out.append(Delivery(finding, session, DERIVED, overlap))
        # One session may hold several worktrees, and may also have claimed the
        # area. Keep its STRONGEST reason once: observed work outranks anything
        # self-reported, so a session that is demonstrably editing the files is
        # told that, not "you claimed this".
        best: dict[str, Delivery] = {}
        for delivery in out:
            held = best.get(delivery.to_session)
            if held is None or (held.provenance == DECLARED
                                and delivery.provenance == DERIVED):
                best[delivery.to_session] = delivery
        return tuple(best.values())

    @staticmethod
    def _overlap(finding: Finding, touched, wanted, space=None) -> str:
        """Why this session is in the audience, in words a reader can dispute."""
        if finding.subject_paths:
            exact = sorted(set(finding.subject_paths) & set(touched))
            if exact:
                shown = ", ".join(exact[:3])
                more = f" (+{len(exact) - 3} more)" if len(exact) > 3 else ""
                return (f"you have outstanding edits to {shown}{more}, which "
                        f"this finding is about")
        hit = sorted({a for a in areas_of(touched, space) if a in wanted})
        if hit:
            names = ", ".join(str(a) for a in hit[:3])
            return (f"you have outstanding edits in {names}, the area this "
                    f"finding is about, but no individual file matched, so this is "
                    f"an area-level match and may not concern you")
        return ""

    # ── reading ─────────────────────────────────────────────────────────────
    def inbox(self, session_id: str, *, include_acknowledged: bool = False,
              max_age_days: float = DEFAULT_INBOX_AGE_DAYS,
              now: int | None = None) -> tuple[Delivery, ...]:
        """Findings routed to one session, most urgent first, then newest.

        Age-filtered at READ time only. Nothing is deleted: §9's answer 1 keeps
        history because that is what a reader wants after the fact, and a
        finding that scrolled out of an inbox is still evidence of what was
        known when.
        """
        now = _now() if now is None else int(now)
        floor = now - int(max_age_days * 86_400)
        clause = "" if include_acknowledged else " AND d.acknowledged_at IS NULL"
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT f.*, d.provenance, d.reason, d.acknowledged_at
                    FROM delivery d JOIN finding f ON f.id = d.finding_id
                    WHERE d.to_session = ? AND f.at >= ?{clause}""",
                (session_id, floor)).fetchall()
        out = [Delivery(self._finding(r), session_id, r["provenance"],
                        r["reason"], r["acknowledged_at"]) for r in rows]
        out.sort(key=lambda d: (d.finding.urgency, -d.finding.at))
        return tuple(out)

    @staticmethod
    def _finding(row) -> Finding:
        return Finding(
            id=row["id"], at=row["at"], kind=row["kind"],
            from_session=row["from_session"], from_evidence=row["from_evidence"],
            body=row["body"],
            subject_paths=tuple(json.loads(row["subject_paths"] or "[]")),
            to_session=row["to_session"], to_area=row["to_area"],
            broadcast=bool(row["broadcast"]), severity=row["severity"])

    def acknowledge(self, finding_id: str, session_id: str,
                    at: int | None = None) -> bool:
        """Mark one delivery read. False when it was not addressed to them."""
        with self._connect() as conn:
            changed = conn.execute(
                """UPDATE delivery SET acknowledged_at = ?
                   WHERE finding_id = ? AND to_session = ?
                     AND acknowledged_at IS NULL""",
                (_now() if at is None else int(at), finding_id, session_id))
            return changed.rowcount > 0

    def findings(self, *, limit: int = 100) -> tuple[Finding, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM finding ORDER BY at DESC LIMIT ?",
                (int(limit),)).fetchall()
        return tuple(self._finding(r) for r in rows)

    def undelivered(self, *, limit: int = 50) -> tuple[Finding, ...]:
        """Findings that reached nobody.

        Worth a surface of its own. A publisher who believes they warned the
        fleet, and did not, is in a worse position than one who never tried —
        and the mesh knows the difference.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT f.* FROM finding f
                   LEFT JOIN delivery d ON d.finding_id = f.id
                   WHERE d.finding_id IS NULL
                   ORDER BY f.at DESC LIMIT ?""", (int(limit),)).fetchall()
        return tuple(self._finding(r) for r in rows)

    # ── territory ───────────────────────────────────────────────────────────
    def claim(self, area, session_id: str, *, note: str = "",
              at: int | None = None) -> Claim:
        """Take an advisory hold. Never refuses — see `contested`."""
        text = str(area)
        record = Claim(area=text, session_id=session_id,
                       at=_now() if at is None else int(at), note=note)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO claim
                   (area, session_id, at, note, released_at) VALUES (?,?,?,?,NULL)""",
                (record.area, record.session_id, record.at, record.note))
        return record

    def release(self, area, session_id: str, at: int | None = None) -> bool:
        with self._connect() as conn:
            changed = conn.execute(
                """UPDATE claim SET released_at = ?
                   WHERE area = ? AND session_id = ? AND released_at IS NULL""",
                (_now() if at is None else int(at), str(area), session_id))
            return changed.rowcount > 0

    def active_claims(self) -> tuple[Claim, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM claim WHERE released_at IS NULL ORDER BY at"
            ).fetchall()
        return tuple(Claim(area=r["area"], session_id=r["session_id"],
                           at=r["at"], note=r["note"],
                           released_at=r["released_at"]) for r in rows)

    def contested(self) -> tuple[str, ...]:
        """Areas held by more than one session at once.

        Not an error condition and not prevented. Two project managers can both
        think they own a workstream; the useful thing a system can do is say so.
        """
        held: dict[str, set[str]] = {}
        for claim in self.active_claims():
            held.setdefault(claim.area, set()).add(claim.session_id)
        return tuple(sorted(a for a, who in held.items() if len(who) > 1))

    def survey(self, mesh: WorktreeMesh) -> tuple[AreaState, ...]:
        """What the fleet is doing, area by area.

        Merges three sources and keeps them distinguishable: derived work from
        the mesh, self-reported claims from the bus, and open findings. A caller
        deciding where to work needs all three and needs to know which is which.
        """
        working: dict[Area, set[str]] = {}
        trees: dict[Area, set[str]] = {}
        for worktree in mesh.worktrees:
            for area in areas_of(worktree.touched_paths, self.space):
                trees.setdefault(area, set()).add(worktree.label)
                if worktree.session != UNATTRIBUTED:
                    working.setdefault(area, set()).add(worktree.session)
        claimed: dict[Area, set[str]] = {}
        for claim in self.active_claims():
            claimed.setdefault(parse_area(claim.area), set()).add(claim.session_id)
        counts: dict[Area, int] = {}
        for finding in self.findings(limit=500):
            for area in finding.areas_in(self.space):
                counts[area] = counts.get(area, 0) + 1

        out = []
        for area in sorted(set(working) | set(trees) | set(claimed) | set(counts)):
            out.append(AreaState(
                area=area,
                working=tuple(sorted(working.get(area, ()))),
                worktrees=tuple(sorted(trees.get(area, ()))),
                claimed_by=tuple(sorted(claimed.get(area, ()))),
                open_findings=counts.get(area, 0)))
        return tuple(out)

    def open_areas(self, mesh: WorktreeMesh, *, candidates,
                   include_tier3: bool = False) -> tuple[Area, ...]:
        """Areas from `candidates` that no session is working in or holding.

        `candidates` is required rather than defaulted to "every pillar", and the
        reason is honesty about coverage: this module cannot enumerate the areas
        of a repository it has not been shown. A caller passes the areas it
        derived from real paths, and gets back the free subset — it never gets
        back an area that exists only because a table listed a pillar.

        Tier 3 areas are excluded by default. AGENTS.md §3 makes them capital,
        destructive, TRUST or release authority, and §1 forbids a probationary
        model from changing them at all — so offering one as free work to a
        session that asked "where am I needed?" routes an agent straight at
        authority it does not have. That is not only the order paths: the
        verifier, the release packaging and the native certified-width kernel
        are all on the list.
        """
        occupied = {state.area for state in self.survey(mesh) if state.occupied}
        out = []
        for area in candidates:
            if area in occupied or not area.mapped:
                continue
            if self._space().tier3(area) and not include_tier3:
                continue
            out.append(area)
        return tuple(sorted(set(out)))


def default_database() -> Path:
    """The same file `worktree_cache` uses. One substrate, two schemas."""
    from alelyon.runtime.common.worktree_cache import default_database as _db
    return _db()
