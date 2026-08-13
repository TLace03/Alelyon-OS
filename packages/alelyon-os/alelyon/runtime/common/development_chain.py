"""Immutable desk -> team -> worker identity for development lanes.

This is an organisational coordinate, not the authority ladder in
``fleet_hierarchy``.  A desk owns repository areas, a team belongs to exactly
one desk, and a worker belongs to exactly one fully-qualified team.  The fixed
shape deliberately has no implicit parent: a missing team never collapses to
its desk and a missing worker never collapses to its team.

The snapshot is supplied to every operation.  Nothing here discovers a fleet,
opens a store, mutates git, or keeps a process-global registry.  A downstream
caller may pass ``DevelopmentDesk.areas`` to ``desk_lanes.derive``; this module
does not import that policy or reproduce its area-to-check rules.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Iterable


SCHEMA = "alelyon.development-hierarchy/v1"

ACTIVE = "active"
DISABLED = "disabled"
RETIRED = "retired"
STATES = (ACTIVE, DISABLED, RETIRED)

MALFORMED_RECORD = "malformed-record"
MALFORMED_KEY = "malformed-key"
MALFORMED_STATE = "malformed-state"
MALFORMED_ORDER = "malformed-order"
MALFORMED_REVISION = "malformed-revision"
MALFORMED_IDENTITY = "malformed-identity"
MALFORMED_BRANCH_BINDING = "malformed-branch-binding"
MISSING_DESK = "missing-desk"
MISSING_TEAM = "missing-team"
MISSING_WORKER = "missing-worker"
MISSING_BRANCH_BINDING = "missing-branch-binding"
DUPLICATE_IDENTITY = "duplicate-identity"
DUPLICATE_BRANCH_BINDING = "duplicate-branch-binding"
AMBIGUOUS_IDENTITY = "ambiguous-identity"
ORPHAN_TEAM = "orphan-team"
ORPHAN_WORKER = "orphan-worker"
CROSS_DESK_PARENT = "cross-desk-parent"
#: Reserved for a future shape with generic parent edges.  V1's distinct
#: desk/team/worker records make a cycle unrepresentable; the validator must
#: not fabricate this result for an orphan or a cross-desk reference.
CYCLIC_MEMBERSHIP = "cyclic-membership"
UNSUPPORTED_SCHEMA = "unsupported-schema"
STALE_SNAPSHOT_VERSION = "stale-snapshot-version"
STALE_SNAPSHOT_DIGEST = "stale-snapshot-digest"
DISABLED_MEMBER = "disabled-member"
RETIRED_MEMBER = "retired-member"

REASONS = (
    MALFORMED_RECORD,
    MALFORMED_KEY,
    MALFORMED_STATE,
    MALFORMED_ORDER,
    MALFORMED_REVISION,
    MALFORMED_IDENTITY,
    MALFORMED_BRANCH_BINDING,
    MISSING_DESK,
    MISSING_TEAM,
    MISSING_WORKER,
    MISSING_BRANCH_BINDING,
    DUPLICATE_IDENTITY,
    DUPLICATE_BRANCH_BINDING,
    AMBIGUOUS_IDENTITY,
    ORPHAN_TEAM,
    ORPHAN_WORKER,
    CROSS_DESK_PARENT,
    CYCLIC_MEMBERSHIP,
    UNSUPPORTED_SCHEMA,
    STALE_SNAPSHOT_VERSION,
    STALE_SNAPSHOT_DIGEST,
    DISABLED_MEMBER,
    RETIRED_MEMBER,
)

_KEY = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class HierarchyRevision:
    """The schema, caller-assigned version, and exact snapshot commitment.

    One snapshot cannot prove that versions increase or that a retired key was
    not reused in a later snapshot.  Callers retaining revision history own
    those cross-snapshot comparisons.
    """

    schema: str
    version: int
    snapshot: str


@dataclass(frozen=True)
class DevelopmentDesk:
    """One development desk and the repository areas its lane owns."""

    key: str
    label: str
    order: int
    areas: tuple[str, ...] = ()
    state: str = ACTIVE


@dataclass(frozen=True)
class DevelopmentTeam:
    """One team under exactly one desk.

    ``owned_paths`` is a declared within-desk split.  It is identity metadata,
    not branch routing and not a replacement for the repository area policy.
    """

    key: str
    label: str
    desk_key: str
    order: int
    owned_paths: tuple[str, ...] = ()
    #: Exact opaque branch labels owned by this fully-qualified team.  Names
    #: carry no ancestry; they are compared whole and never parsed.
    branch_labels: tuple[str, ...] = ()
    state: str = ACTIVE


@dataclass(frozen=True)
class DevelopmentWorker:
    """One individual under one fully-qualified ``(desk, team)`` parent."""

    key: str
    label: str
    desk_key: str
    team_key: str
    order: int
    state: str = ACTIVE


@dataclass(frozen=True, order=True)
class WorkerIdentity:
    """The only worker identity suitable for durable attribution."""

    desk_key: str
    team_key: str
    worker_key: str

    @property
    def canonical(self) -> str:
        return f"{self.desk_key}/{self.team_key}/{self.worker_key}"


@dataclass(frozen=True)
class HierarchySnapshot:
    """One immutable, self-committing organisational reading."""

    revision: HierarchyRevision
    desks: tuple[DevelopmentDesk, ...]
    teams: tuple[DevelopmentTeam, ...]
    workers: tuple[DevelopmentWorker, ...]


@dataclass(frozen=True)
class HierarchyRefusal:
    """A named refusal; ``detail`` is explanatory and never parsed as policy."""

    reason: str
    detail: str


class HierarchyError(ValueError):
    """Typed build/aggregate refusal rather than an unclassified ValueError."""

    def __init__(self, refusal: HierarchyRefusal) -> None:
        self.refusal = refusal
        self.reason = refusal.reason
        super().__init__(f"{refusal.reason}: {refusal.detail}")


@dataclass(frozen=True)
class HierarchyResolution:
    """A complete worker chain or one named refusal."""

    identity: WorkerIdentity | None = None
    desk: DevelopmentDesk | None = None
    team: DevelopmentTeam | None = None
    worker: DevelopmentWorker | None = None
    refusal: HierarchyRefusal | None = None

    def __post_init__(self) -> None:
        success = (self.refusal is None
                   and all(value is not None for value in (
                       self.identity, self.desk, self.team, self.worker)))
        refused = (self.refusal is not None
                   and all(value is None for value in (
                       self.identity, self.desk, self.team, self.worker)))
        if not (success or refused):
            raise HierarchyError(HierarchyRefusal(
                MALFORMED_RECORD,
                "a resolution must be either one complete chain or one "
                "refusal, never a partial mixture"))

    @property
    def ok(self) -> bool:
        return self.refusal is None and self.identity is not None


def _refusal(reason: str, detail: str) -> HierarchyRefusal:
    return HierarchyRefusal(reason=reason, detail=detail)


def _raise(refusal: HierarchyRefusal | None) -> None:
    if refusal is not None:
        raise HierarchyError(refusal)


def _valid_key(value: object) -> bool:
    return isinstance(value, str) and _KEY.fullmatch(value) is not None


def _valid_order(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_scopes(value: object) -> bool:
    if not isinstance(value, tuple):
        return False
    if not all(isinstance(item, str) and item and item == item.strip()
               for item in value):
        return False
    return len(set(value)) == len(value)


def _revision_syntax(revision: object) -> HierarchyRefusal | None:
    if not isinstance(revision, HierarchyRevision):
        return _refusal(MALFORMED_REVISION,
                        "the snapshot reference is not a HierarchyRevision")
    if not isinstance(revision.schema, str) or not revision.schema:
        return _refusal(MALFORMED_REVISION,
                        "the hierarchy schema is not a non-empty string")
    if not _valid_order(revision.version) or revision.version == 0:
        return _refusal(MALFORMED_REVISION,
                        "the hierarchy version must be a positive integer")
    if _DIGEST.fullmatch(str(revision.snapshot or "")) is None:
        return _refusal(MALFORMED_REVISION,
                        "the snapshot reference must be sha256:<64 lower-hex>")
    return None


def _records_syntax(desks: object, teams: object, workers: object, *,
                    canonical: bool = False) -> HierarchyRefusal | None:
    if not all(isinstance(records, tuple)
               for records in (desks, teams, workers)):
        return _refusal(MALFORMED_RECORD,
                        "desk, team, and worker collections must be tuples")

    groups = (
        ("desk", desks, DevelopmentDesk),
        ("team", teams, DevelopmentTeam),
        ("worker", workers, DevelopmentWorker),
    )
    for kind, records, expected in groups:
        for record in records:
            if not isinstance(record, expected):
                return _refusal(
                    MALFORMED_RECORD,
                    f"the {kind} collection contains {type(record).__name__}, "
                    f"not {expected.__name__}")
            keys = [record.key]
            if isinstance(record, DevelopmentTeam):
                keys.append(record.desk_key)
            elif isinstance(record, DevelopmentWorker):
                keys.extend((record.desk_key, record.team_key))
            if not all(_valid_key(key) for key in keys):
                return _refusal(
                    MALFORMED_KEY,
                    f"{kind} keys must match [a-z0-9][a-z0-9._-]*")
            if not isinstance(record.label, str) or not record.label:
                return _refusal(MALFORMED_RECORD,
                                f"{kind} {record.key!r} has no display label")
            if not _valid_order(record.order):
                return _refusal(MALFORMED_ORDER,
                                f"{kind} {record.key!r} has an invalid order")
            if record.state not in STATES:
                return _refusal(MALFORMED_STATE,
                                f"{kind} {record.key!r} has state "
                                f"{record.state!r}")
            scopes = record.areas if isinstance(record, DevelopmentDesk) \
                else record.owned_paths if isinstance(record, DevelopmentTeam) \
                else ()
            if not _valid_scopes(scopes):
                field = "areas" if isinstance(record, DevelopmentDesk) \
                    else "owned_paths"
                return _refusal(
                    MALFORMED_RECORD,
                    f"{kind} {record.key!r} has malformed or duplicate {field}")
            if canonical and scopes != tuple(sorted(scopes)):
                field = "areas" if isinstance(record, DevelopmentDesk) \
                    else "owned_paths"
                return _refusal(
                    MALFORMED_RECORD,
                    f"{kind} {record.key!r} {field} are not in canonical "
                    "order")
            if isinstance(record, DevelopmentTeam):
                labels = record.branch_labels
                if (not isinstance(labels, tuple)
                        or not all(isinstance(label, str) and label
                                   for label in labels)):
                    return _refusal(
                        MALFORMED_BRANCH_BINDING,
                        f"team {record.desk_key}/{record.key} has a non-string "
                        "or empty branch binding")
                if canonical and labels != tuple(sorted(labels)):
                    return _refusal(
                        MALFORMED_BRANCH_BINDING,
                        f"team {record.desk_key}/{record.key} branch bindings "
                        "are not in canonical order")
    if canonical:
        canonical_groups = (
            ("desk", desks,
             tuple(sorted(desks, key=lambda item: (item.order, item.key)))),
            ("team", teams,
             tuple(sorted(teams, key=lambda item: (
                 item.desk_key, item.order, item.key)))),
            ("worker", workers,
             tuple(sorted(workers, key=lambda item: (
                 item.desk_key, item.team_key, item.order, item.key)))),
        )
        for kind, records, ordered in canonical_groups:
            if records != ordered:
                return _refusal(
                    MALFORMED_RECORD,
                    f"the {kind} collection is not in canonical order")
    return None


def _duplicates(desks: tuple[DevelopmentDesk, ...],
                teams: tuple[DevelopmentTeam, ...],
                workers: tuple[DevelopmentWorker, ...],
                ) -> HierarchyRefusal | None:
    identities = (
        ("desk", [desk.key for desk in desks]),
        ("team", [(team.desk_key, team.key) for team in teams]),
        ("worker", [(worker.desk_key, worker.team_key, worker.key)
                    for worker in workers]),
    )
    for kind, values in identities:
        repeated = sorted(value for value, count in Counter(values).items()
                          if count > 1)
        if repeated:
            return _refusal(
                DUPLICATE_IDENTITY,
                f"duplicate fully-qualified {kind} identity {repeated[0]!r}")
    return None


def _memberships(desks: tuple[DevelopmentDesk, ...],
                 teams: tuple[DevelopmentTeam, ...],
                 workers: tuple[DevelopmentWorker, ...],
                 ) -> HierarchyRefusal | None:
    desk_keys = {desk.key for desk in desks}
    for team in sorted(teams, key=lambda item: (item.desk_key, item.key)):
        if team.desk_key not in desk_keys:
            return _refusal(
                ORPHAN_TEAM,
                f"team {team.desk_key}/{team.key} names no desk parent")

    team_pairs = {(team.desk_key, team.key) for team in teams}
    team_keys = {team.key for team in teams}
    for worker in sorted(
            workers,
            key=lambda item: (item.desk_key, item.team_key, item.key)):
        parent = (worker.desk_key, worker.team_key)
        if parent in team_pairs:
            continue
        identity = f"{worker.desk_key}/{worker.team_key}/{worker.key}"
        if worker.team_key in team_keys:
            return _refusal(
                CROSS_DESK_PARENT,
                f"worker {identity} names team {worker.team_key!r} under the "
                "wrong desk")
        return _refusal(
            ORPHAN_WORKER,
            f"worker {identity} names no fully-qualified team parent")
    return None


def _branch_bindings(
        teams: tuple[DevelopmentTeam, ...],
        ) -> HierarchyRefusal | None:
    owners: dict[str, list[tuple[str, str]]] = {}
    for team in teams:
        for label in team.branch_labels:
            owners.setdefault(label, []).append((team.desk_key, team.key))
    repeated = sorted(label for label, bound in owners.items()
                      if len(bound) > 1)
    if not repeated:
        return None
    label = repeated[0]
    parents = sorted(f"{desk}/{team}" for desk, team in owners[label])
    return _refusal(
        DUPLICATE_BRANCH_BINDING,
        f"branch {label!r} is bound more than once: {', '.join(parents)}")


def _canonical_payload(snapshot: HierarchySnapshot) -> dict:
    """The exact JSON value committed by ``HierarchyRevision.snapshot``.

    The digest field is intentionally absent.  Record arrays and set-like
    ownership tuples are sorted so input permutation does not change identity.
    Labels, states, orders, ownership, and every parent key remain covered.
    """
    return {
        "schema": snapshot.revision.schema,
        "version": snapshot.revision.version,
        "desks": [
            {
                "key": desk.key,
                "label": desk.label,
                "order": desk.order,
                "areas": sorted(desk.areas),
                "state": desk.state,
            }
            for desk in sorted(snapshot.desks,
                               key=lambda item: (item.order, item.key))
        ],
        "teams": [
            {
                "key": team.key,
                "label": team.label,
                "desk_key": team.desk_key,
                "order": team.order,
                "owned_paths": sorted(team.owned_paths),
                "branch_labels": sorted(team.branch_labels),
                "state": team.state,
            }
            for team in sorted(
                snapshot.teams,
                key=lambda item: (item.desk_key, item.order, item.key))
        ],
        "workers": [
            {
                "key": worker.key,
                "label": worker.label,
                "desk_key": worker.desk_key,
                "team_key": worker.team_key,
                "order": worker.order,
                "state": worker.state,
            }
            for worker in sorted(
                snapshot.workers,
                key=lambda item: (
                    item.desk_key, item.team_key, item.order, item.key))
        ],
    }


def canonical_bytes(snapshot: HierarchySnapshot) -> bytes:
    """Canonical UTF-8 JSON, compact and non-ASCII preserving."""
    return json.dumps(
        _canonical_payload(snapshot),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")


def _snapshot_digest(snapshot: HierarchySnapshot) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(snapshot)).hexdigest()


def _materialize(name: str, records: object) -> tuple:
    try:
        return tuple(records)  # type: ignore[arg-type]
    except TypeError as exc:
        raise HierarchyError(_refusal(
            MALFORMED_RECORD,
            f"the {name} collection is not iterable")) from exc


def build_snapshot(*, version: int,
                   desks: Iterable[DevelopmentDesk],
                   teams: Iterable[DevelopmentTeam],
                   workers: Iterable[DevelopmentWorker],
                   ) -> HierarchySnapshot:
    """Validate and build a canonical snapshot or raise ``HierarchyError``.

    Input iterables are copied to tuples and caller records are never mutated.
    Invalid or duplicate membership is refused before any sorting could make it
    look canonical.
    """
    if not _valid_order(version) or version == 0:
        raise HierarchyError(_refusal(
            MALFORMED_REVISION,
            "the hierarchy version must be a positive integer"))

    desk_values = _materialize("desk", desks)
    team_values = _materialize("team", teams)
    worker_values = _materialize("worker", workers)
    _raise(_records_syntax(desk_values, team_values, worker_values))
    _raise(_duplicates(desk_values, team_values, worker_values))
    _raise(_memberships(desk_values, team_values, worker_values))
    _raise(_branch_bindings(team_values))

    ordered_desks = tuple(sorted(
        (replace(desk, areas=tuple(sorted(desk.areas)))
         for desk in desk_values),
        key=lambda item: (item.order, item.key)))
    ordered_teams = tuple(sorted(
        (replace(team,
                 owned_paths=tuple(sorted(team.owned_paths)),
                 branch_labels=tuple(sorted(team.branch_labels)))
         for team in team_values),
        key=lambda item: (item.desk_key, item.order, item.key)))
    ordered_workers = tuple(sorted(
        worker_values,
        key=lambda item: (
            item.desk_key, item.team_key, item.order, item.key)))
    provisional = HierarchySnapshot(
        revision=HierarchyRevision(SCHEMA, version, "sha256:" + "0" * 64),
        desks=ordered_desks,
        teams=ordered_teams,
        workers=ordered_workers,
    )
    return HierarchySnapshot(
        revision=HierarchyRevision(
            SCHEMA, version, _snapshot_digest(provisional)),
        desks=ordered_desks,
        teams=ordered_teams,
        workers=ordered_workers,
    )


def validate_snapshot(
        snapshot: HierarchySnapshot,
        *,
        expected: HierarchyRevision | None = None,
        ) -> HierarchyRefusal | None:
    """Validate in one deterministic refusal order.

    Precedence is structural syntax, schema, expected version, self/expected
    digest, duplicate identity, parent membership, then branch uniqueness.
    Lookup-only missing and inactive states are evaluated later from desk to
    team to worker.
    """
    if not isinstance(snapshot, HierarchySnapshot):
        return _refusal(MALFORMED_RECORD,
                        "the value is not a HierarchySnapshot")
    refusal = _records_syntax(
        snapshot.desks, snapshot.teams, snapshot.workers, canonical=True)
    if refusal is not None:
        return refusal
    refusal = _revision_syntax(snapshot.revision)
    if refusal is not None:
        return refusal
    if snapshot.revision.schema != SCHEMA:
        return _refusal(
            UNSUPPORTED_SCHEMA,
            f"schema {snapshot.revision.schema!r} is not {SCHEMA!r}")

    if expected is not None:
        refusal = _revision_syntax(expected)
        if refusal is not None:
            return refusal
        if expected.schema != SCHEMA:
            return _refusal(
                UNSUPPORTED_SCHEMA,
                f"expected schema {expected.schema!r} is not {SCHEMA!r}")
        if expected.version != snapshot.revision.version:
            return _refusal(
                STALE_SNAPSHOT_VERSION,
                f"expected version {expected.version}, got "
                f"{snapshot.revision.version}")

    actual = _snapshot_digest(snapshot)
    if snapshot.revision.snapshot != actual:
        return _refusal(
            STALE_SNAPSHOT_DIGEST,
            "the snapshot content does not match its own digest")
    if expected is not None and expected.snapshot != snapshot.revision.snapshot:
        return _refusal(
            STALE_SNAPSHOT_DIGEST,
            "the snapshot digest differs from the expected revision")

    refusal = _duplicates(snapshot.desks, snapshot.teams, snapshot.workers)
    if refusal is not None:
        return refusal
    refusal = _memberships(snapshot.desks, snapshot.teams, snapshot.workers)
    if refusal is not None:
        return refusal
    return _branch_bindings(snapshot.teams)


def _parse_worker_identity(
        identity: WorkerIdentity | str,
        ) -> WorkerIdentity | HierarchyRefusal:
    if isinstance(identity, str):
        parts = identity.split("/")
        if len(parts) != 3:
            return _refusal(
                MALFORMED_IDENTITY,
                "worker identity must be exactly <desk>/<team>/<worker>")
        parsed = WorkerIdentity(*parts)
        if parsed.canonical != identity:
            return _refusal(
                MALFORMED_IDENTITY,
                "worker identity is not in canonical form")
    elif isinstance(identity, WorkerIdentity):
        parsed = identity
    else:
        return _refusal(MALFORMED_IDENTITY,
                        "worker identity has an unsupported type")
    if not all(_valid_key(value) for value in (
            parsed.desk_key, parsed.team_key, parsed.worker_key)):
        return _refusal(MALFORMED_IDENTITY,
                        "worker identity contains a non-canonical key")
    return parsed


def _inactive(record: object, kind: str) -> HierarchyRefusal | None:
    state = getattr(record, "state", ACTIVE)
    key = getattr(record, "key", "")
    if state == DISABLED:
        return _refusal(DISABLED_MEMBER,
                        f"{kind} {key!r} is disabled")
    if state == RETIRED:
        return _refusal(RETIRED_MEMBER,
                        f"{kind} {key!r} is retired")
    return None


def _resolve_team_valid(
        snapshot: HierarchySnapshot,
        desk_key: str,
        team_key: str,
        *,
        require_active: bool,
        ) -> tuple[DevelopmentTeam | None, HierarchyRefusal | None]:
    desk = next((item for item in snapshot.desks if item.key == desk_key), None)
    if desk is None:
        return None, _refusal(MISSING_DESK,
                              f"desk {desk_key!r} is not in the snapshot")
    if require_active:
        refusal = _inactive(desk, "desk")
        if refusal is not None:
            return None, refusal

    team = next((item for item in snapshot.teams
                 if (item.desk_key, item.key) == (desk_key, team_key)), None)
    if team is None:
        return None, _refusal(
            MISSING_TEAM,
            f"team {desk_key}/{team_key} is not in the snapshot")
    if require_active:
        refusal = _inactive(team, "team")
        if refusal is not None:
            return None, refusal
    return team, None


def resolve_team(
        snapshot: HierarchySnapshot,
        desk_key: str,
        team_key: str,
        *,
        expected: HierarchyRevision,
        require_active: bool = True,
        ) -> tuple[DevelopmentTeam | None, HierarchyRefusal | None]:
    """Resolve an exact team; no bare or parent fallback is attempted."""
    if not isinstance(expected, HierarchyRevision):
        return None, _refusal(
            MALFORMED_REVISION,
            "team resolution requires an expected HierarchyRevision")
    refusal = validate_snapshot(snapshot, expected=expected)
    if refusal is not None:
        return None, refusal
    if not _valid_key(desk_key) or not _valid_key(team_key):
        return None, _refusal(
            MALFORMED_IDENTITY,
            "team identity requires canonical desk and team keys")
    return _resolve_team_valid(
        snapshot, desk_key, team_key, require_active=require_active)


def resolve_worker(
        snapshot: HierarchySnapshot,
        identity: WorkerIdentity | str,
        *,
        expected: HierarchyRevision,
        require_active: bool = True,
        ) -> HierarchyResolution:
    """Resolve one fully-qualified worker against an exact revision."""
    if not isinstance(expected, HierarchyRevision):
        return HierarchyResolution(refusal=_refusal(
            MALFORMED_REVISION,
            "worker resolution requires an expected HierarchyRevision"))
    refusal = validate_snapshot(snapshot, expected=expected)
    if refusal is not None:
        return HierarchyResolution(refusal=refusal)
    parsed = _parse_worker_identity(identity)
    if isinstance(parsed, HierarchyRefusal):
        return HierarchyResolution(refusal=parsed)

    team, refusal = _resolve_team_valid(
        snapshot, parsed.desk_key, parsed.team_key,
        require_active=require_active)
    if refusal is not None:
        return HierarchyResolution(refusal=refusal)
    desk = next(item for item in snapshot.desks
                if item.key == parsed.desk_key)
    worker = next((item for item in snapshot.workers
                   if (item.desk_key, item.team_key, item.key) == (
                       parsed.desk_key, parsed.team_key, parsed.worker_key)),
                  None)
    if worker is None:
        return HierarchyResolution(refusal=_refusal(
            MISSING_WORKER,
            f"worker {parsed.canonical} is not in the snapshot"))
    if require_active:
        refusal = _inactive(worker, "worker")
        if refusal is not None:
            return HierarchyResolution(refusal=refusal)
    return HierarchyResolution(
        identity=parsed, desk=desk, team=team, worker=worker)


def resolve_bare_team(
        snapshot: HierarchySnapshot,
        team_key: str,
        *,
        expected: HierarchyRevision,
        require_active: bool = True,
        ) -> tuple[DevelopmentTeam | None, HierarchyRefusal | None]:
    """Migration/UI lookup by key; ambiguous reuse refuses.

    Durable attribution must use ``resolve_team`` with both keys.  Display
    labels never participate in this lookup.
    """
    if not isinstance(expected, HierarchyRevision):
        return None, _refusal(
            MALFORMED_REVISION,
            "bare team resolution requires an expected HierarchyRevision")
    refusal = validate_snapshot(snapshot, expected=expected)
    if refusal is not None:
        return None, refusal
    if not _valid_key(team_key):
        return None, _refusal(MALFORMED_IDENTITY,
                              "bare team key is not canonical")
    matches = [team for team in snapshot.teams if team.key == team_key]
    if not matches:
        return None, _refusal(MISSING_TEAM,
                              f"team key {team_key!r} is not in the snapshot")
    if len(matches) > 1:
        chains = sorted(f"{team.desk_key}/{team.key}" for team in matches)
        return None, _refusal(
            AMBIGUOUS_IDENTITY,
            f"team key {team_key!r} matches {', '.join(chains)}")
    only = matches[0]
    return _resolve_team_valid(
        snapshot, only.desk_key, only.key, require_active=require_active)


def resolve_bare_worker(
        snapshot: HierarchySnapshot,
        worker_key: str,
        *,
        expected: HierarchyRevision,
        require_active: bool = True,
    ) -> HierarchyResolution:
    """Migration/UI lookup by key; ambiguous reuse refuses."""
    if not isinstance(expected, HierarchyRevision):
        return HierarchyResolution(refusal=_refusal(
            MALFORMED_REVISION,
            "bare worker resolution requires an expected HierarchyRevision"))
    refusal = validate_snapshot(snapshot, expected=expected)
    if refusal is not None:
        return HierarchyResolution(refusal=refusal)
    if not _valid_key(worker_key):
        return HierarchyResolution(refusal=_refusal(
            MALFORMED_IDENTITY, "bare worker key is not canonical"))
    matches = [worker for worker in snapshot.workers
               if worker.key == worker_key]
    if not matches:
        return HierarchyResolution(refusal=_refusal(
            MISSING_WORKER,
            f"worker key {worker_key!r} is not in the snapshot"))
    if len(matches) > 1:
        chains = sorted(
            f"{worker.desk_key}/{worker.team_key}/{worker.key}"
            for worker in matches)
        return HierarchyResolution(refusal=_refusal(
            AMBIGUOUS_IDENTITY,
            f"worker key {worker_key!r} matches {', '.join(chains)}"))
    only = matches[0]
    return resolve_worker(
        snapshot,
        WorkerIdentity(only.desk_key, only.team_key, only.key),
        expected=expected,
        require_active=require_active,
    )


def resolve_branch_team(
        snapshot: HierarchySnapshot,
        branch: str,
        *,
        expected: HierarchyRevision,
        require_active: bool = True,
        ) -> tuple[DevelopmentTeam | None, HierarchyRefusal | None]:
    """Resolve an exact opaque branch label to its one owning team.

    No prefix, path, commit, display label, or worker can infer ownership.  A
    branch belongs to the team; operations on it remain attributable to an
    independently resolved ``WorkerIdentity``.
    """
    if not isinstance(expected, HierarchyRevision):
        return None, _refusal(
            MALFORMED_REVISION,
            "branch resolution requires an expected HierarchyRevision")
    refusal = validate_snapshot(snapshot, expected=expected)
    if refusal is not None:
        return None, refusal
    if not isinstance(branch, str) or not branch:
        return None, _refusal(
            MALFORMED_BRANCH_BINDING,
            "branch lookup requires one non-empty opaque string")
    matches = [team for team in snapshot.teams
               if branch in team.branch_labels]
    if not matches:
        return None, _refusal(
            MISSING_BRANCH_BINDING,
            f"branch {branch!r} has no authoritative team binding")
    if len(matches) > 1:  # defensive: a valid snapshot already refuses this
        return None, _refusal(
            DUPLICATE_BRANCH_BINDING,
            f"branch {branch!r} has more than one team binding")
    team = matches[0]
    return _resolve_team_valid(
        snapshot, team.desk_key, team.key, require_active=require_active)


def teams_for_desk(snapshot: HierarchySnapshot,
                   desk_key: str) -> tuple[DevelopmentTeam, ...]:
    """Deterministic aggregate, including disabled/retired history."""
    _raise(validate_snapshot(snapshot))
    if not any(desk.key == desk_key for desk in snapshot.desks):
        raise HierarchyError(_refusal(
            MISSING_DESK, f"desk {desk_key!r} is not in the snapshot"))
    return tuple(sorted(
        (team for team in snapshot.teams if team.desk_key == desk_key),
        key=lambda item: (item.order, item.key)))


def workers_for_team(snapshot: HierarchySnapshot, desk_key: str,
                     team_key: str) -> tuple[DevelopmentWorker, ...]:
    """Deterministic aggregate for one exact team, with no bare fallback."""
    _raise(validate_snapshot(snapshot))
    if not any(desk.key == desk_key for desk in snapshot.desks):
        raise HierarchyError(_refusal(
            MISSING_DESK, f"desk {desk_key!r} is not in the snapshot"))
    if not any((team.desk_key, team.key) == (desk_key, team_key)
               for team in snapshot.teams):
        raise HierarchyError(_refusal(
            MISSING_TEAM,
            f"team {desk_key}/{team_key} is not in the snapshot"))
    return tuple(sorted(
        (worker for worker in snapshot.workers
         if (worker.desk_key, worker.team_key) == (desk_key, team_key)),
        key=lambda item: (item.order, item.key)))


__all__ = [
    "ACTIVE", "AMBIGUOUS_IDENTITY", "CROSS_DESK_PARENT",
    "CYCLIC_MEMBERSHIP", "DISABLED", "DISABLED_MEMBER",
    "DUPLICATE_BRANCH_BINDING", "DUPLICATE_IDENTITY", "DevelopmentDesk",
    "DevelopmentTeam", "DevelopmentWorker", "HierarchyError",
    "HierarchyRefusal",
    "HierarchyResolution", "HierarchyRevision", "HierarchySnapshot",
    "MALFORMED_BRANCH_BINDING", "MALFORMED_IDENTITY", "MALFORMED_KEY",
    "MALFORMED_ORDER",
    "MALFORMED_RECORD", "MALFORMED_REVISION", "MALFORMED_STATE",
    "MISSING_BRANCH_BINDING", "MISSING_DESK", "MISSING_TEAM",
    "MISSING_WORKER", "ORPHAN_TEAM", "ORPHAN_WORKER", "REASONS", "RETIRED",
    "RETIRED_MEMBER", "SCHEMA",
    "STALE_SNAPSHOT_DIGEST", "STALE_SNAPSHOT_VERSION", "STATES",
    "UNSUPPORTED_SCHEMA", "WorkerIdentity", "build_snapshot",
    "canonical_bytes", "resolve_bare_team", "resolve_bare_worker",
    "resolve_branch_team", "resolve_team", "resolve_worker", "teams_for_desk",
    "validate_snapshot", "workers_for_team",
]
