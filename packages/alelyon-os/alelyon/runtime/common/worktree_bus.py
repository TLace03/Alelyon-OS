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

Channels and threads
--------------------
The findings half above is a *notification* system: a session publishes, routing
decides who hears it, and there the exchange ends. Sessions had no way to hold a
conversation — `mail.py ack --note` publishes a reply as a brand-new finding with
no link to what it answers, so a thread of five messages is five unrelated rows
and a reader cannot reassemble it.

Channels and threads close that, and they are deliberately built on the *claim*
precedent rather than as a new trust model. A Slack channel is a **declared
subscription**, which is the exact opposite of the derived routing this module
exists to protect — so joining a channel is treated as what it is: the same kind
of self-report a claim already is, labelled `DECLARED` at every read, carrying
the same "that is your own declaration, not observed work" reason.

The consequence is a property Slack does not have and cannot have. A message
posted to `#runtime-common` naming `worktree_bus.py` reaches:

* everyone who **joined** the channel — `DECLARED`, because they said so; and
* everyone the mesh can see **editing that file**, whether or not they ever
  joined — `DERIVED`, because the repository says so.

A session cannot miss a message about the file it is demonstrably editing merely
because it was not in the room. Derived routing is not weakened by the channel;
the channel is an additional audience laid over it.

A **thread** is a `reply_to` pointing at the message being answered. A reply
reaches the parent's audience, and each recipient keeps **the provenance their
parent delivery had** — an inherited audience is graded by the link that
assembled it, never promoted by being replied to (§5's weakest-link rule).

Mentions (`@here`, `@channel`, `@<session-prefix>`) are `DECLARED` without
exception: they are a publisher naming an address, which is the one thing a
publisher is never trusted to do on its own.

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
#: 2 added channels, threads and the `message` kind. The upgrade is additive —
#: two new tables and two defaulted columns — so a v1 database opens, migrates in
#: place and keeps every row. There is no downgrade: an older build opening a
#: migrated file sees columns it does not select and works unchanged, which is
#: the only compatibility direction this file needs.
BUS_SCHEMA_VERSION = 2

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
KIND_MESSAGE = "message"               # conversation; carries no operational claim
KINDS = (KIND_REFACTOR, KIND_INTERFACE, KIND_DEFECT, KIND_CONVENTION,
         KIND_BLOCKED, KIND_LANDED, KIND_MESSAGE)

#: Inbox ordering. `refactor-in-flight` and `interface-changed` outrank the rest
#: because they are the two that make another session's *correct* work start
#: failing — the reader needs them before they debug something that is not theirs.
#:
#: `message` sorts LAST, below even `landed`, and that placement is the whole
#: safety argument for adding chat to an operational bus. Conversation is far
#: more frequent than incident reporting; ranking the two together would let a
#: busy channel push an `interface-changed` off the top of an inbox, and the one
#: thing this bus exists to deliver on time is the finding that stops a session
#: debugging somebody else's half-applied refactor. Chat may never outrank it.
_URGENCY = {KIND_REFACTOR: 0, KIND_INTERFACE: 0, KIND_BLOCKED: 1,
            KIND_DEFECT: 2, KIND_CONVENTION: 3, KIND_LANDED: 4,
            KIND_MESSAGE: 5}

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
    "A channel is not private and not access-controlled. Every session that can "
    "open the database can read every channel and join any of them; membership "
    "organises attention, it does not restrict who can see what.",
    "Nobody is obliged to be in a channel. A room's members are the sessions "
    "that joined, so posting to a quiet channel can reach nobody at all, which "
    "is why a message that matters should still name the paths it is about.",
    "A mention is resolved by prefix against sessions something else already "
    "knows about. One that matches nothing, or matches two sessions at once, "
    "delivers to NOBODY rather than guessing.",
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
    #: The channel this was posted to, without the `#`. Empty for a finding
    #: published the original way, which is every row written before schema 2.
    channel: str = ""
    #: The finding this answers, making the two a thread. Empty for a root
    #: message. Not a foreign key: a reply must survive its parent ageing out of
    #: an inbox, and a dangling `reply_to` is readable evidence that it did.
    reply_to: str = ""

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
class Channel:
    """A named room. Created by whoever first posts to or joins it.

    There is no admin, no private channel and no invite. Every session that can
    open the database can read every channel, and pretending otherwise would be
    a security claim this substrate cannot make: the file is a local SQLite
    database with filesystem permissions and nothing else (§9's "no silent
    security claims"). A channel organises attention. It does not contain
    anything.
    """

    name: str
    at: int
    created_by: str
    topic: str = ""

    @property
    def display(self) -> str:
        return f"#{self.name}"


@dataclass(frozen=True)
class Membership:
    """A session's declared subscription to a channel.

    The same shape as `Claim` and for the same reason — see the module docstring.
    Joining is a self-report; nothing verifies that a member has any business in
    the channel, and nothing needs to.
    """

    channel: str
    session_id: str
    at: int
    left_at: int | None = None

    @property
    def active(self) -> bool:
        return self.left_at is None


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
    """CREATE TABLE IF NOT EXISTS channel (
           name TEXT PRIMARY KEY,
           at INTEGER NOT NULL,
           created_by TEXT NOT NULL,
           topic TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS membership (
           channel TEXT NOT NULL,
           session_id TEXT NOT NULL,
           at INTEGER NOT NULL,
           left_at INTEGER,
           PRIMARY KEY (channel, session_id))""",
    "CREATE INDEX IF NOT EXISTS finding_at ON finding(at)",
    "CREATE INDEX IF NOT EXISTS delivery_session ON delivery(to_session)",
    "CREATE INDEX IF NOT EXISTS claim_area ON claim(area)",
    "CREATE INDEX IF NOT EXISTS membership_session ON membership(session_id)",
)

#: Indexes over columns that schema 2 ADDS, so they cannot live in `_DDL`: those
#: statements run against a v1 table where `finding.channel` does not exist yet,
#: and `CREATE INDEX IF NOT EXISTS` on a missing column is a hard error, not a
#: no-op. Applied by `_migrate` immediately after the column it indexes — caught
#: by opening a copy of this repository's real 278-finding database, which is the
#: only shape that reproduces it. A fresh database never does.
_INDEXES_V2 = (
    "CREATE INDEX IF NOT EXISTS finding_channel ON finding(channel, at)",
    "CREATE INDEX IF NOT EXISTS finding_reply ON finding(reply_to)",
)

#: Columns added to `finding` after schema 1 shipped, with the default a v1 row
#: gets. Applied by `_migrate` rather than declared in `_DDL`, because
#: `CREATE TABLE IF NOT EXISTS` is a no-op against an existing table and would
#: silently leave a populated database on the old shape — which is the failure
#: this list exists to prevent. The live bus in this repository held 278 findings
#: and 1303 deliveries when schema 2 was written; none of them may be lost, and
#: a v1 row is not wrong, it simply predates channels.
_FINDING_COLUMNS_V2 = (
    ("channel", "TEXT NOT NULL DEFAULT ''"),
    ("reply_to", "TEXT NOT NULL DEFAULT ''"),
)


def _finding_id(at: int, session: str, kind: str, body: str,
                channel: str = "", reply_to: str = "") -> str:
    """Content-derived, so republishing the same finding twice is detectable.

    Includes the timestamp: a session that genuinely learns the same thing again
    later has news, and collapsing that into the first report would hide a
    recurrence behind a duplicate check.

    Channel and parent are in the material because `publish` writes with
    `INSERT OR REPLACE`: without them, one session posting the same sentence to
    two channels inside the same second produces one id, and the second post
    silently overwrites the first. Chat makes that ordinary — "done" and "ack"
    land in several rooms a second apart — where for findings it was rare enough
    to have never been hit.
    """
    material = (f"{at}\x1f{session}\x1f{kind}\x1f{body}"
                f"\x1f{channel}\x1f{reply_to}").encode("utf-8")
    return hashlib.blake2b(material, digest_size=8).hexdigest()


def _now() -> int:
    return int(time.time())


#: Channel names are lowercase, dash-separated, and bounded. Not cosmetic: a
#: channel is created by being posted to, so `#Runtime-Common`, `#runtime common`
#: and `#runtime-common` would otherwise become three rooms holding one
#: conversation, and nobody would see the other two.
_CHANNEL_OK = "abcdefghijklmnopqrstuvwxyz0123456789-_."
MAX_CHANNEL_NAME = 48

#: `@here` and `@channel` both mean everyone; Slack users type both and would not
#: thank us for a distinction we cannot enforce.
EVERYONE_MENTIONS = ("here", "channel", "everyone")


def normalise_channel(name: str) -> str:
    """Canonical form of a channel name, or `""` for no channel.

    Refuses rather than mangles when nothing survives normalisation: a name that
    reduces to the empty string would silently post to no channel, and the
    author would believe they had spoken to a room.
    """
    given = str(name or "").strip()
    if not given:
        return ""          # no channel asked for, which is not the same thing
                           # as one asked for badly -- see below
    text = given.lstrip("#").strip().lower()
    if not text:
        # `###` reduces to nothing, and returning "" here would report it as "no
        # channel given" -- the caller then fails with "a finding needs an
        # audience", which names the wrong cause and sends the author looking at
        # their subject paths instead of at what they typed.
        raise ValueError(
            f"channel name {name!r} is only '#' characters; a channel needs a "
            f"name after the hash")
    text = "-".join(text.split())
    kept = "".join(c for c in text if c in _CHANNEL_OK).strip("-")
    if not kept:
        raise ValueError(
            f"channel name {name!r} contains no usable characters; expected "
            f"letters, digits, dash, dot or underscore")
    return kept[:MAX_CHANNEL_NAME]


def parse_mentions(body: str) -> tuple[tuple[str, ...], bool]:
    """`@handles` in a message body, and whether it addressed everyone.

    Deliberately literal: it reads the text and reports what it found. It
    resolves nothing, because resolving is routing's job and routing is the half
    of this module that is not allowed to trust the writer.
    """
    handles: list[str] = []
    everyone = False
    for token in str(body or "").split():
        if not token.startswith("@") or len(token) < 2:
            continue
        handle = token[1:].strip(".,;:!?)(<>[]\"'").lower()
        if not handle:
            continue
        if handle in EVERYONE_MENTIONS:
            everyone = True
        elif handle not in handles:
            handles.append(handle)
    return tuple(handles), everyone


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
            self._migrate(conn)
            conn.execute(
                "INSERT OR IGNORE INTO bus_meta(name, value) VALUES('schema', ?)",
                (str(BUS_SCHEMA_VERSION),))
            conn.execute(
                "UPDATE bus_meta SET value = ? WHERE name = 'schema' "
                "AND CAST(value AS INTEGER) < ?",
                (str(BUS_SCHEMA_VERSION), BUS_SCHEMA_VERSION))

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> tuple[str, ...]:
        """Bring an existing `finding` table up to the current column set.

        Idempotent and additive: every column is added with a constant default,
        so SQLite rewrites no rows and an interrupted run resumes correctly. It
        reads the table's ACTUAL shape rather than trusting the recorded schema
        version, because those two disagree in exactly the case that matters — a
        database written by a build that had the version bumped and the migration
        half-applied. The columns are the truth; `bus_meta` is a label.

        Returns what it added, so a caller can log a migration that really
        happened instead of one that was merely attempted.
        """
        have = {row["name"] for row in conn.execute("PRAGMA table_info(finding)")}
        added = []
        for column, declaration in _FINDING_COLUMNS_V2:
            if column in have:
                continue
            conn.execute(f"ALTER TABLE finding ADD COLUMN {column} {declaration}")
            added.append(column)
        for statement in _INDEXES_V2:
            conn.execute(statement)
        return tuple(added)

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
                cache=None, channel: str = "", reply_to: str = "",
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
        channel = normalise_channel(channel)
        reply_to = str(reply_to or "").strip()
        # `@here` is an audience expansion, not decoration: it means everyone the
        # mesh can see working, which is exactly what `broadcast` already routes.
        # Resolved HERE rather than in `_route` so the stored row genuinely is a
        # broadcast — a finding whose deliveries say "broadcast" while its own
        # row says otherwise would misinform every later reader of the table.
        if parse_mentions(body)[1]:
            broadcast = True
        if not (subject_paths or to_session or to_area or broadcast
                or channel or reply_to):
            raise ValueError(
                "a finding needs an audience: subject paths, an area, an "
                "explicit session, a channel, a message to reply to, or "
                "broadcast. Refusing rather than "
                "defaulting to broadcast, which would make every unaddressed "
                "note interrupt everybody.")
        at = _now() if at is None else int(at)
        if reply_to:
            # Flattened at WRITE time, not merely resolved at read time. A reply
            # to a reply is stored against the root, so "a thread is one level
            # deep" is a property of the data rather than a convention every
            # reader has to reimplement -- and a reader that forgot would drop
            # the third message in a conversation and show nothing missing.
            # One hop suffices: by induction every stored parent is already a root.
            reply_to = self._root_of(reply_to)
        if channel:
            self.ensure_channel(channel, created_by=from_session, at=at)
        elif reply_to:
            # A reply inherits its parent's room, so answering from an inbox --
            # where the channel is not something the replier had to know -- keeps
            # the answer in the conversation instead of orphaning it.
            channel = self._channel_of(reply_to)
        paths = tuple(dict.fromkeys(
            str(p).replace("\\", "/") for p in (subject_paths or ()) if str(p).strip()))
        finding = Finding(
            id=_finding_id(at, from_session, kind, body, channel, reply_to),
            at=at, kind=kind,
            from_session=from_session or UNATTRIBUTED, from_evidence=from_evidence,
            body=body, subject_paths=paths, to_session=to_session,
            to_area=to_area, broadcast=bool(broadcast), severity=severity,
            channel=channel, reply_to=reply_to)
        deliveries = self._route(finding, mesh, cache)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO finding
                   (id, at, kind, from_session, from_evidence, body,
                    subject_paths, to_session, to_area, broadcast, severity,
                    channel, reply_to)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (finding.id, finding.at, finding.kind, finding.from_session,
                 finding.from_evidence, finding.body,
                 json.dumps(list(finding.subject_paths)), finding.to_session,
                 finding.to_area, int(finding.broadcast), finding.severity,
                 finding.channel, finding.reply_to))
            for delivery in deliveries:
                conn.execute(
                    """INSERT OR IGNORE INTO delivery
                       (finding_id, to_session, provenance, reason, at)
                       VALUES (?,?,?,?,?)""",
                    (finding.id, delivery.to_session, delivery.provenance,
                     delivery.reason, at))
        return finding, deliveries

    def _root_of(self, finding_id: str) -> str:
        """The root of the thread `finding_id` belongs to.

        Returns the id unchanged when it names nothing. A dangling `reply_to` is
        readable evidence that a parent aged out or was never written, and
        refusing the reply would lose the only message that still described the
        exchange.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT reply_to FROM finding WHERE id = ?",
                               (finding_id,)).fetchone()
        return (row["reply_to"] or finding_id) if row is not None else finding_id

    def _channel_of(self, finding_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT channel FROM finding WHERE id = ?",
                               (finding_id,)).fetchone()
        return (row["channel"] or "") if row is not None else ""

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

        # Channel members. The same shape as the claim below and justified the
        # same way: a subscription is a self-report, so it is DECLARED, it never
        # merges with observed work, and it is worth what a self-report is worth.
        #
        # What it must NOT do is replace derived routing. A channel post that
        # names subject paths still reaches whoever the mesh sees editing them,
        # joined or not — the two legs are additive, and the dedup at the bottom
        # keeps the DERIVED reason when a session qualifies both ways. That is
        # the property in the module docstring: you cannot be talked about in a
        # room you are not in without being told.
        if finding.channel:
            for member in self.members(finding.channel):
                if member.session_id == finding.from_session:
                    continue
                out.append(Delivery(
                    finding, member.session_id, DECLARED,
                    f"you joined #{finding.channel}, where this was posted; "
                    f"that is your own declaration, not observed work"))

        # A thread reaches the audience of the message it answers.
        #
        # Each recipient keeps THEIR OWN provenance from the parent delivery
        # rather than a fresh label for the reply. A reply cannot promote a
        # DECLARED audience to a DERIVED one — nothing new was observed, someone
        # merely spoke again — and §5 grades a compound by its weakest link.
        if finding.reply_to:
            for parent in self.deliveries_of(finding.reply_to):
                if parent.to_session == finding.from_session:
                    continue
                out.append(Delivery(
                    finding, parent.to_session, parent.provenance,
                    f"you received the message this replies to, which reached "
                    f"you because: {parent.reason}"))

        # Mentions. DECLARED without exception — this is a publisher naming an
        # address, the one thing a publisher is never trusted to do alone.
        #
        # A handle is resolved against sessions something else already knows
        # about: the mesh, the channel's members, and claim holders. An
        # unresolvable handle produces NO delivery, deliberately. Inventing one
        # addressed to a typo is the "filed against a string" failure the
        # explicit-address branch above was already caught by; the chat layer
        # re-parses the body and reports which mentions matched nobody.
        handles, _everyone = parse_mentions(finding.body)
        if handles:
            known = {claim.session_id for claim in self.active_claims()}
            if finding.channel:
                known |= {m.session_id for m in self.members(finding.channel)}
            if mesh is not None:
                known |= {w.session for w in mesh.worktrees
                          if w.session != UNATTRIBUTED}
            known.discard(finding.from_session)
            for handle in handles:
                matched = sorted(s for s in known if s.lower().startswith(handle))
                if len(matched) != 1:
                    # Zero is a typo or an invisible session; more than one is a
                    # prefix short enough to be ambiguous. Guessing between two
                    # sessions is worse than delivering to neither, because the
                    # author would never learn it went to the wrong one.
                    continue
                out.append(Delivery(
                    finding, matched[0], DECLARED,
                    f"the author wrote @{handle}, which matches your session id; "
                    f"an address the publisher chose, not derived from your work"))
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
            # Deduplicated on this path too. Channel membership and an @mention
            # routinely name the same session, and the delivery table collapses
            # them on its primary key -- so without this, `publish` REPORTS
            # reaching two sessions and stores one. Over-reporting reach is the
            # one lie this module is built to prevent.
            return self._strongest(out)

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
        return self._strongest(out)

    @staticmethod
    def _strongest(deliveries) -> tuple[Delivery, ...]:
        """One delivery per session, keeping its best-evidenced reason.

        A session may hold several worktrees, have claimed the area, have joined
        the channel and be mentioned by name in the body. Observed work outranks
        anything self-reported, so a session that is demonstrably editing the
        files is told that, not "you joined a room".
        """
        best: dict[str, Delivery] = {}
        for delivery in deliveries:
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
            broadcast=bool(row["broadcast"]), severity=row["severity"],
            # `keys()` rather than a bare subscript: a row selected by a build
            # mid-migration, or by a query written before schema 2, has no such
            # column, and a pre-channel finding is a valid finding.
            channel=(row["channel"] if "channel" in row.keys() else "") or "",
            reply_to=(row["reply_to"] if "reply_to" in row.keys() else "") or "")

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

    def deliveries_of(self, finding_id: str) -> tuple[Delivery, ...]:
        """Every session one finding reached, and which of them read it.

        The publisher's half of `inbox`, and it was missing. `publish` reports
        who a finding reached at the moment it is sent, and after that a
        publisher had no way to ask anything about it — so "silence is not
        consent" was true, unfalsifiable from the sending end, and stayed that
        way: measured on this repository's bus at 33378ae, 10 of 836 deliveries
        were acknowledged and all 162 `landed` findings had been acknowledged
        zero times.

        Acknowledged still means only that a session pressed a button. It does
        not mean read, understood or agreed, and an unacknowledged delivery is
        not evidence that nobody looked.
        """
        # Every delivery column is ALIASED. `finding` and `delivery` both have a
        # `to_session`, so `SELECT f.*, d.to_session` yields two columns of that
        # name and `sqlite3.Row` returns the first -- the finding's explicit
        # ADDRESSEE, which is empty for every path-routed finding. This function
        # therefore reported "" as the recipient of the deliveries the mesh had
        # derived, which is most of them, and `mail.py sent` printed "read by"
        # followed by nothing.
        #
        # It shipped that way because every test addressed its finding with
        # `to_session=`, where the two columns agree and the bug is invisible.
        # The one test that used path routing asserted an EMPTY result, so it
        # never read the field. `test_deliveries_of_names_a_path_routed_recipient`
        # is the falsifier that was missing.
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT f.*, d.to_session AS delivered_to,
                          d.provenance AS delivered_by,
                          d.reason AS delivered_because,
                          d.acknowledged_at AS delivered_read_at
                   FROM delivery d JOIN finding f ON f.id = d.finding_id
                   WHERE d.finding_id = ?
                   ORDER BY d.at""", (finding_id,)).fetchall()
        return tuple(Delivery(self._finding(r), r["delivered_to"],
                              r["delivered_by"], r["delivered_because"],
                              r["delivered_read_at"]) for r in rows)

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

    # ── channels ────────────────────────────────────────────────────────────
    def ensure_channel(self, name: str, *, created_by: str = "",
                       topic: str = "", at: int | None = None) -> Channel:
        """Create a channel if it does not exist; return it either way.

        Creation is implicit — posting to `#foo` makes `#foo` — because the
        alternative is a session discovering it must run a second command before
        it can speak, at the moment it has something to say. `INSERT OR IGNORE`
        keeps the original creator and topic when two sessions race to first
        post, so the record says who really opened the room.
        """
        name = normalise_channel(name)
        if not name:
            raise ValueError("a channel needs a name")
        at = _now() if at is None else int(at)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO channel (name, at, created_by, topic)
                   VALUES (?,?,?,?)""",
                (name, at, created_by or UNATTRIBUTED, topic))
            row = conn.execute(
                "SELECT * FROM channel WHERE name = ?", (name,)).fetchone()
        return Channel(name=row["name"], at=row["at"],
                       created_by=row["created_by"], topic=row["topic"])

    def set_topic(self, name: str, topic: str) -> bool:
        with self._connect() as conn:
            changed = conn.execute(
                "UPDATE channel SET topic = ? WHERE name = ?",
                (str(topic or ""), normalise_channel(name)))
            return changed.rowcount > 0

    def channels(self) -> tuple[Channel, ...]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM channel ORDER BY name").fetchall()
        return tuple(Channel(name=r["name"], at=r["at"],
                             created_by=r["created_by"], topic=r["topic"])
                     for r in rows)

    def join(self, name: str, session_id: str,
             at: int | None = None) -> Membership:
        """Subscribe a session to a channel. Never refuses — see `Channel`."""
        name = normalise_channel(name)
        self.ensure_channel(name, created_by=session_id, at=at)
        record = Membership(channel=name, session_id=session_id,
                            at=_now() if at is None else int(at))
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO membership
                   (channel, session_id, at, left_at) VALUES (?,?,?,NULL)""",
                (record.channel, record.session_id, record.at))
        return record

    def leave(self, name: str, session_id: str, at: int | None = None) -> bool:
        with self._connect() as conn:
            changed = conn.execute(
                """UPDATE membership SET left_at = ?
                   WHERE channel = ? AND session_id = ? AND left_at IS NULL""",
                (_now() if at is None else int(at),
                 normalise_channel(name), session_id))
            return changed.rowcount > 0

    def members(self, name: str) -> tuple[Membership, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM membership
                   WHERE channel = ? AND left_at IS NULL ORDER BY at""",
                (normalise_channel(name),)).fetchall()
        return tuple(Membership(channel=r["channel"], session_id=r["session_id"],
                                at=r["at"], left_at=r["left_at"]) for r in rows)

    def memberships_of(self, session_id: str) -> tuple[Membership, ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM membership
                   WHERE session_id = ? AND left_at IS NULL ORDER BY channel""",
                (session_id,)).fetchall()
        return tuple(Membership(channel=r["channel"], session_id=r["session_id"],
                                at=r["at"], left_at=r["left_at"]) for r in rows)

    def history(self, name: str, *, limit: int = 50,
                before: int | None = None) -> tuple[Finding, ...]:
        """A channel's messages, oldest first within the page.

        Reads the `finding` table directly rather than the reader's deliveries,
        so a session that joins late sees what was said before it arrived. That
        is the difference between a channel and an inbox, and it is why joining a
        room is worth anything: the conversation predates you.
        """
        clause = " AND at < ?" if before is not None else ""
        params: list = [normalise_channel(name)]
        if before is not None:
            params.append(int(before))
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                # `rowid`, not `id`. Ids are content hashes, so two messages in
                # the same second sort at random and a rapid exchange renders out
                # of order -- which for a conversation inverts question and
                # answer. The rowid is insertion order and needs no new column.
                f"""SELECT * FROM finding WHERE channel = ?{clause}
                    ORDER BY at DESC, rowid DESC LIMIT ?""", params).fetchall()
        return tuple(reversed([self._finding(r) for r in rows]))

    def channel_stats(self) -> dict[str, tuple[int, int]]:
        """`{channel: (message count, newest at)}` in one pass.

        A rail listing every room needs both numbers for each, and asking
        `history()` per channel to get them reads every message in the database
        to count them. One grouped query instead.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT channel, count(*) AS n, max(at) AS last
                   FROM finding WHERE channel <> '' GROUP BY channel""").fetchall()
        return {r["channel"]: (r["n"], r["last"]) for r in rows}

    def thread(self, finding_id: str) -> tuple[Finding, ...]:
        """A root message and its replies, oldest first.

        One level deep on purpose. Slack made the same choice, and the reason
        holds here: a reply to a reply is a tree, a tree needs indentation to
        read, and a terminal inbox that indents arbitrarily deep becomes
        unreadable at exactly the moment a conversation gets interesting. A reply
        addressed at a reply is recorded against the same root.
        """
        with self._connect() as conn:
            root = conn.execute(
                "SELECT * FROM finding WHERE id = ?", (finding_id,)).fetchone()
            if root is None:
                return ()
            if root["reply_to"]:
                parent = conn.execute("SELECT * FROM finding WHERE id = ?",
                                      (root["reply_to"],)).fetchone()
                if parent is not None:
                    root = parent
            replies = conn.execute(
                # `rowid` for the same reason `history` uses it: two replies in
                # one second must read in the order they were sent.
                "SELECT * FROM finding WHERE reply_to = ? ORDER BY at, rowid",
                (root["id"],)).fetchall()
        return tuple([self._finding(root)] + [self._finding(r) for r in replies])

    def search(self, text: str, *, channel: str = "",
               limit: int = 50) -> tuple[Finding, ...]:
        """Substring search over message bodies, newest first.

        `LIKE`, not FTS5. The corpus is one repository's fleet chatter, the
        largest observed table was under 300 rows, and a full-text index would be
        a second schema to migrate for a scan that costs nothing at this size.
        Revisit if a channel ever holds enough to notice.
        """
        needle = f"%{str(text or '').strip()}%"
        clause = " AND channel = ?" if channel else ""
        params: list = [needle]
        if channel:
            params.append(normalise_channel(channel))
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM finding WHERE body LIKE ?{clause}
                    ORDER BY at DESC LIMIT ?""", params).fetchall()
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
