"""What is dormant, what it was doing, and what restarting it would collide with.

A repository accumulates worktrees. Each one was a session working on something;
most of them stopped without finishing. Picking one up again means answering four
questions that are currently spread across four modules and one person's memory:

* **which ones are dormant**, and on what evidence;
* **what each was working on**, in the coordinate space the fleet already uses;
* **how important that work is**, from the blueprint queue rather than a guess;
* **who else it would collide with** if it woke up.

This module answers those four and stops. It composes `worktree`,
`worktree_cache`, `worktree_bus`, `blueprint_focus` and `fleet_hierarchy`; it
adds no new observation of its own.

IT DOES NOT RESUME ANYTHING, AND THAT IS DELIBERATE
----------------------------------------------------
There is no `start()` here and there should not be. Waking a worktree means
launching a coding agent, in a directory, with a brief, against a paid API — a
process this repository does not own, spending money, writing code, under
`AGENTS.md` §3 Tier 3 process control. `fleet_hierarchy` now makes placement
and measured standing mandatory inputs to the commanding layer's route while
remaining read-only. The same split applies here: this module supplies the
reviewable plan and launches nothing; the authorised process owner consumes the
plan and records a concrete total-cost reason for any override.

So the output is a **plan**: a fully specified, reviewable description of what
*would* be started, where, with what brief, at what layer, and what it would
contend with. Launching is a separate, explicitly authorised step that belongs to
whatever actually owns agent processes. That split is not timidity — the plan is
the artifact worth reviewing before eight agents start in parallel, and a
one-click launcher that skipped it would be the least useful ordering of the two.

That other half now exists: `worktree_launch` consumes a `ResumePlan` and is the
only module here permitted to start anything. This one is unchanged by its
arrival and must stay that way — `test_the_plan_starts_nothing` fails if a
`start`/`spawn`/`launch` ever appears in this file. The split is the design, not
a stage of it: `worktree_launch.admit()` decides every refusal purely, and it can
only refuse or admit a plan that was computed here.

"DORMANT" IS DERIVED FROM A COMMIT, NOT FROM A SESSION
------------------------------------------------------
`worktree.py` states the limit this module inherits and must not launder: *"A
clean worktree that a session is still actively reading is indistinguishable from
an abandoned one. Staleness measures its commit, not its session."*

So nothing here says "asleep". It says `DORMANT`, meaning *no commit for longer
than the threshold*, and it carries the evidence. Where the harness recorded a
session that is currently live in that directory, the answer is `ACTIVE` instead
— that is the one signal that can contradict commit age, and it comes from a
record the agent did not write. Where neither is available the answer is
`UNKNOWN`, which is a value, never a blank.

Waking something a live session is still holding is the expensive mistake this
distinction exists to prevent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence, Tuple
import time

from alelyon.runtime.common import fleet_hierarchy as HIER
from alelyon.runtime.common import worktree as W
from alelyon.runtime.common import worktree_areas as A

#: A worktree whose HEAD is older than this reads as dormant. Seven days is the
#: same threshold `WorktreeMesh.stale()` already uses; a second number here would
#: mean two answers to "is this old" and no rule for which wins.
DORMANT_AFTER_DAYS = 7.0

#: Dormancy verdicts. Three values, and the third is not a failure to compute —
#: it is the honest answer when a commit time could not be read.
DORMANT = "DORMANT"
ACTIVE = "ACTIVE"
UNKNOWN = "UNKNOWN"

#: How an overlap was established. Never merged, exactly as the mesh keeps
#: observed and declared apart everywhere else.
OBSERVED = "observed"
DECLARED = "declared"

LIMITS: Tuple[str, ...] = (
    "This module RESUMES NOTHING. It produces a reviewable plan; launching an "
    "agent is a separate, explicitly authorised step that this repository does "
    "not own.",
    "DORMANT means no commit for longer than the threshold. It does NOT mean "
    "nobody is in the directory - a clean worktree somebody is actively reading "
    "is indistinguishable from an abandoned one, and only a live harness record "
    "can contradict commit age.",
    "What a worktree 'was working on' is derived from paths it touched, not "
    "from anything it declared about its intent. A worktree that read widely "
    "and changed one file reports that one file.",
    "Overlap is computed from paths, not semantics. Two worktrees editing "
    "different functions in one file are reported as overlapping - a deliberate "
    "false positive rather than a missed one.",
    "Two sessions inside ONE worktree share a single record and cannot appear "
    "as an overlap here, however directly they collide. That is the arrangement "
    "this pairing cannot see.",
    "Blueprint rank is the position of a DOCUMENT the worktree contends for, "
    "joined by worktree label. A worktree holding no queued document has no "
    "rank, which means unranked rather than unimportant.",
    "The layer is a reading of the work's KIND, from a phrase rule over paths. "
    "It bounds authority, never competence, and it is a suggestion a caller may "
    "ignore.",
    "A handover names who COULD take the work. It does not transfer anything, "
    "notify anyone, or ask. The receiving session learns about it when somebody "
    "tells it - `worktree_bus.publish` is how, and that is a separate act.",
    "Path overlap means the OPPOSITE thing in a handover than it does in a "
    "plan: waking two overlapping worktrees reproduces their collision, while "
    "handing one's work to the other is the cleanest reassignment available.",
)


@dataclass(frozen=True)
class Overlap:
    """Another worktree whose outstanding work touches the same paths."""

    path: str
    label: str
    session: str
    shared_paths: Tuple[str, ...]
    #: OBSERVED (both have outstanding edits to these paths) or DECLARED (the
    #: other session claimed the area rather than being seen in it).
    provenance: str
    reason: str

    @property
    def count(self) -> int:
        return len(self.shared_paths)


@dataclass(frozen=True)
class Candidate:
    """One worktree, and everything a caller needs to decide about waking it."""

    path: str
    label: str
    branch: Optional[str]
    head: str
    #: The session that CREATED the directory, where a path convention carries
    #: one. Not the occupant — see `worktree.Worktree.session`.
    session: str
    session_evidence: str

    dormancy: str
    dormancy_evidence: str
    age_days: Optional[float]

    #: Where the outstanding work sits, in the fleet's coordinate space.
    areas: Tuple[A.Area, ...]
    touched_paths: Tuple[str, ...]

    #: Position in the blueprint focus queue (0 = most urgent), or None.
    rank: Optional[int]
    rank_reason: str
    #: The queued document this worktree is holding, when it holds one.
    document: str

    #: What rank of work this looks like, and therefore what class of model it
    #: would deserve. A reading, never an instruction.
    layer: str
    layer_evidence: str
    capability: str

    overlaps: Tuple[Overlap, ...] = ()

    @property
    def dormant(self) -> bool:
        return self.dormancy == DORMANT

    @property
    def contested(self) -> bool:
        return bool(self.overlaps)


@dataclass(frozen=True)
class Inheritor:
    """A live worktree that could absorb a dormant one's work.

    The alternative to waking something. Nothing is started; this names who is
    already awake, already in that part of the codebase, and already authorised
    for work of that rank.
    """

    path: str
    label: str
    session: str
    #: How it qualifies. Ordered strongest first — see `_FITNESS_ORDER`.
    fitness: str
    evidence: str
    #: The layer that supervises both this worktree's work and the dormant
    #: work's. Empty when they share no supervisor below the board.
    supervisor: str
    #: Already has outstanding edits to paths the dormant work touched.
    #: A QUALIFICATION here, not a warning — see `handover`.
    already_in_the_code: bool
    shared_paths: Tuple[str, ...] = ()

    @property
    def overlap_count(self) -> int:
        return len(self.shared_paths)


#: Fitness classes, strongest first. The order IS the ranking.
IN_THE_CODE = "already-in-the-code"
SAME_AREA = "same-area"
SAME_PILLAR = "same-pillar"
SAME_SUPERVISOR = "same-supervisor"
_FITNESS_ORDER = (IN_THE_CODE, SAME_AREA, SAME_PILLAR, SAME_SUPERVISOR)


@dataclass(frozen=True)
class Handover:
    """Who could take a dormant worktree's work, and who was ruled out and why."""

    dormant: "Candidate"
    inheritors: Tuple[Inheritor, ...] = ()
    #: Live worktrees that were considered and NOT offered, each with its
    #: reason. Carried rather than filtered away: "nobody can take this" and
    #: "three could but all are below the capability floor" are different
    #: answers and a caller needs to tell them apart.
    refusals: Tuple[str, ...] = ()

    @property
    def possible(self) -> bool:
        return bool(self.inheritors)

    @property
    def best(self) -> Optional[Inheritor]:
        return self.inheritors[0] if self.inheritors else None


@dataclass(frozen=True)
class ResumePlan:
    """What waking a chosen set would involve. Not an instruction to wake it."""

    repo_root: str
    observed_at: int
    selected: Tuple[Candidate, ...] = ()
    #: Overlaps BETWEEN the selected worktrees — the ones a caller creates by
    #: choosing this particular set, as distinct from each one's overlaps with
    #: the wider repository.
    internal_conflicts: Tuple[Overlap, ...] = ()
    notes: Tuple[str, ...] = ()
    limits: Tuple[str, ...] = LIMITS

    @property
    def contested(self) -> bool:
        return bool(self.internal_conflicts)

    def graph(self) -> dict:
        """Nodes and edges, for a caller that wants to draw the selection.

        Returned as plain data rather than a widget so the shape can be tested
        without a GUI, and so a caller may render it however it likes. Recomputed
        from the plan each call; there is no hidden state to go stale.
        """
        return graph_of(self.selected, self.internal_conflicts)


def _dormancy(tree: W.Worktree, live_sessions: Sequence[str],
              *, older_than_days: float, now: float) -> Tuple[str, str, Optional[float]]:
    """(verdict, evidence, age_days) for one worktree.

    A live harness record beats commit age, and is the only thing that does. It
    is a record the agent did not write, about a directory it did not name, so it
    is evidence rather than a claim — but it names where a session STARTED, so it
    over-approximates occupancy and the evidence string says so.
    """
    age = tree.age_days(now)
    if tree.session != W.UNATTRIBUTED and tree.session in live_sessions:
        return (ACTIVE,
                f"the harness records session {tree.session} as live in this "
                f"directory; that over-approximates occupancy (it names where a "
                f"session started) but it is the one signal that can contradict "
                f"commit age",
                age)
    if age is None:
        return (UNKNOWN,
                "HEAD's commit time could not be read, so its age is unknown - "
                "which is not the same answer as old",
                None)
    if age > older_than_days:
        return (DORMANT,
                f"HEAD was committed {age:.1f} days ago, beyond the "
                f"{older_than_days:g}-day threshold. This measures the COMMIT, "
                f"not the session: nobody is known to have left",
                age)
    return (ACTIVE,
            f"HEAD was committed {age:.1f} days ago, within the "
            f"{older_than_days:g}-day threshold",
            age)


def _rank_index(blueprint) -> Mapping[str, tuple]:
    """`{worktree label: (rank, document path, why)}` from the focus queue.

    The join is the blueprint's own: `Focus.contenders` names the worktree
    LABELS holding each queued document, so inverting it gives each worktree the
    position of the most urgent document it is contending for.

    Label, not path, because that is what the blueprint records. Two worktrees
    with the same directory name would collide here; that is a real limit of the
    blueprint's own vocabulary rather than something this module can fix, and a
    caller reading a rank should treat it as the document's urgency, not as a
    property of the directory.
    """
    out: dict = {}
    if blueprint is None:
        return out
    for position, entry in enumerate(getattr(blueprint, "focus", ()) or ()):
        for contender in getattr(entry, "contenders", ()) or ():
            label = str(getattr(contender, "label", "") or "")
            if label and label not in out:      # first == most urgent
                out[label] = (position, getattr(entry.document, "path", ""),
                              getattr(entry, "why", ""))
    return out


def _overlaps_for(tree: W.Worktree, mesh: W.WorktreeMesh,
                  claims: Mapping[str, Sequence[str]],
                  space: Optional[A.AreaSpace] = None) -> Tuple[Overlap, ...]:
    """Every other worktree this one would contend with, observed then declared."""
    out: list = []
    mine = tree.touched_paths
    if mine:
        for other in mesh.worktrees:
            if other.path == tree.path:
                continue
            shared = mine & other.touched_paths
            if shared:
                out.append(Overlap(
                    path=other.path, label=other.label, session=other.session,
                    shared_paths=tuple(sorted(shared)), provenance=OBSERVED,
                    reason=(f"both have outstanding work on {len(shared)} "
                            f"shared path(s)")))
    # Declared holds, kept separate and labelled. A claim is a self-report and
    # is worth exactly what a self-report is worth, but a session that claimed
    # an area this worktree is in is a collision worth naming before waking it.
    # THIS repository's space, not the process default. Reaching for the module
    # -level helper here placed the observed repository's paths with whatever
    # checkout the process happens to stand in, so every declared claim silently
    # failed to match and the declared half of this function returned nothing.
    my_areas = {str(a) for a in A.areas_of(mine, space)}
    seen = {o.session for o in out}
    for area, sessions in (claims or {}).items():
        if area not in my_areas:
            continue
        for session in sessions:
            if session in seen or session == tree.session:
                continue
            seen.add(session)
            out.append(Overlap(
                path="", label="", session=session, shared_paths=(),
                provenance=DECLARED,
                reason=(f"session {session} claimed {area}, which this "
                        f"worktree's outstanding work is in; that is their own "
                        f"declaration, not observed work")))
    out.sort(key=lambda o: (-o.count, o.provenance, o.label, o.session))
    return tuple(out)


def survey(repo_root: Optional[str] = None, *,
           mesh: Optional[W.WorktreeMesh] = None,
           blueprint=None,
           bus=None,
           live_sessions: Sequence[str] = (),
           older_than_days: float = DORMANT_AFTER_DAYS,
           now: Optional[float] = None,
           space: Optional[A.AreaSpace] = None) -> Tuple[Candidate, ...]:
    """Every agent worktree, with what it was doing and what it would collide with.

    Every dependency is injectable and every one is optional, so this can be
    exercised without git, without a blueprint corpus and without a bus. That is
    not test convenience for its own sake: the caller that draws this screen
    already holds a mesh and a blueprint, and re-observing would give it a second
    answer to compare against its first.

    Read-only. Nothing is written, started, or claimed.
    """
    at = float(now if now is not None else time.time())
    mesh = mesh if mesh is not None else W.observe(repo_root, now=at)
    space = space if space is not None else A.load(mesh.repo_root)
    ranks = _rank_index(blueprint)

    claims: dict = {}
    if bus is not None:
        try:
            for claim in bus.active_claims():
                claims.setdefault(str(claim.area), []).append(claim.session_id)
        except Exception:  # noqa: BLE001 - a bus that cannot answer is not fatal
            claims = {}

    out: list = []
    for tree in mesh.agent_worktrees:
        verdict, evidence, age = _dormancy(
            tree, live_sessions, older_than_days=older_than_days, now=at)
        touched = tuple(sorted(tree.touched_paths))
        areas = space.areas_of(touched)
        rank, document, why = ranks.get(tree.label, (None, "", ""))
        # The work's KIND is read from what it touched. `place()` takes prose, so
        # the paths are joined into a brief-shaped string -- a rule over words,
        # which is what it says it is.
        layer, layer_evidence = HIER.place(" ".join(touched[:40]))
        out.append(Candidate(
            path=tree.path, label=tree.label, branch=tree.branch, head=tree.head,
            session=tree.session, session_evidence=tree.session_evidence,
            dormancy=verdict, dormancy_evidence=evidence, age_days=age,
            areas=areas, touched_paths=touched,
            rank=rank,
            rank_reason=(why or "no queued document names this worktree, so it "
                                "is unranked - which is not the same as "
                                "unimportant"),
            document=document,
            layer=layer.key, layer_evidence=layer_evidence,
            capability=layer.capability,
            overlaps=_overlaps_for(tree, mesh, claims, space)))

    # Ranked work first, then the most stale, then by label so the order is
    # stable across observations rather than following git's listing.
    out.sort(key=lambda c: (0 if c.rank is not None else 1,
                            c.rank if c.rank is not None else 0,
                            -(c.age_days or 0.0), c.label))
    return tuple(out)


def dormant(candidates: Iterable[Candidate]) -> Tuple[Candidate, ...]:
    """Only the ones whose commit age says dormant. ACTIVE and UNKNOWN excluded.

    UNKNOWN is excluded deliberately: offering a worktree whose age could not be
    read, in a list whose whole premise is "these are not in use", would put the
    one case nothing was established about beside the cases that were.
    """
    return tuple(c for c in candidates if c.dormancy == DORMANT)


def plan(selected: Sequence[Candidate], *, repo_root: str = "",
         observed_at: Optional[int] = None) -> ResumePlan:
    """What waking `selected` together would involve.

    The interesting output is `internal_conflicts`: overlaps the caller CREATES
    by choosing this particular set. A worktree that contends with something
    nobody is waking is a note; two worktrees in the chosen set contending with
    each other is the reason to choose differently, and the two must not be
    presented as one list.
    """
    chosen = tuple(selected or ())
    paths = {c.path for c in chosen}
    # De-duplicate the symmetric pair: A-with-B and B-with-A are ONE conflict.
    # The key must therefore be the unordered PAIR of the two worktrees. Keying
    # it on the overlap alone deduplicated nothing, because A's record names B
    # and B's record names A, so the two keys differed and every conflict was
    # reported twice.
    internal: dict = {}
    for candidate in chosen:
        for overlap in candidate.overlaps:
            if overlap.provenance != OBSERVED or overlap.path not in paths:
                continue
            key = frozenset((candidate.path, overlap.path))
            if key in internal:
                continue
            internal[key] = Overlap(
                path=overlap.path, label=overlap.label,
                session=overlap.session, shared_paths=overlap.shared_paths,
                provenance=OBSERVED,
                reason=(f"{candidate.label} and {overlap.label} are BOTH in "
                        f"this selection and share "
                        f"{len(overlap.shared_paths)} path(s); waking both "
                        f"reproduces the collision that stopped them"))
    unique = sorted(internal.values(), key=lambda o: (-o.count, o.label))

    notes: list = []
    unranked = [c.label for c in chosen if c.rank is None]
    if unranked:
        notes.append(f"{len(unranked)} selected worktree(s) hold no queued "
                     f"document and are unranked: {', '.join(sorted(unranked))}")
    not_dormant = [c.label for c in chosen if c.dormancy != DORMANT]
    if not_dormant:
        notes.append(f"{len(not_dormant)} selected worktree(s) are not dormant "
                     f"({', '.join(sorted(not_dormant))}); waking one that is "
                     f"still held is the expensive mistake here")
    return ResumePlan(
        repo_root=repo_root, selected=chosen,
        observed_at=int(observed_at if observed_at is not None else time.time()),
        internal_conflicts=tuple(unique), notes=tuple(notes))


#: Capability classes as a floor comparison. LOCAL and SMALL sit at the same
#: height deliberately: they are different *kinds* of cheap, not different
#: amounts of it, and ordering one above the other would invent a ranking
#: `fleet_hierarchy` explicitly declines to make.
_CLASS_HEIGHT = {HIER.FRONTIER: 3, HIER.MID: 2, HIER.SMALL: 1, HIER.LOCAL: 1}


def _meets_floor(capability: str, layer_key: str) -> bool:
    target = HIER.layer(layer_key)
    if target is None:
        return False
    return _CLASS_HEIGHT.get(capability, 0) >= _CLASS_HEIGHT.get(
        target.capability, 0)


def _common_supervisor(left: str, right: str) -> str:
    """The nearest layer that supervises both, or "".

    "Under the same management" is a real relation in the layer space and it is
    computed rather than asserted: two layers share a supervisor when their
    escalation paths intersect, and the nearest one is the first element of the
    left path that also appears in the right.

    The board is excluded. Every path terminates there, so counting it would
    make every pair of layers "same management" and the relation would mean
    nothing — and it is the owner, who supervises everything by definition.
    """
    left_path = [layer for layer in (left, *HIER.escalation_path(left))
                 if layer and not HIER.BY_KEY[layer].human_only]
    right_path = {layer for layer in (right, *HIER.escalation_path(right))
                  if layer and not HIER.BY_KEY[layer].human_only}
    for layer in left_path:
        if layer in right_path:
            return layer
    return ""


def handover(dormant_candidate: Candidate,
             live: Sequence[Candidate]) -> Handover:
    """Who could take `dormant_candidate`'s work instead of waking it.

    The whole point of this function: **nothing is started.** A dormant
    worktree's work is handed to a session that is already awake, already in
    that part of the codebase, and already authorised for work of that rank.

    Overlap INVERTS its meaning here, and that is worth stating plainly because
    the same field means the opposite thing twenty lines away. In `plan()`, two
    worktrees sharing paths is a conflict — waking both reproduces the collision
    that stopped them. In a handover it is the strongest QUALIFICATION there is:
    the session already editing those files is the one that can absorb the work
    without a second checkout, a second context, or a merge.

    Three refusals, each named rather than filtered away:

    * not awake — a dormant worktree cannot inherit from a dormant worktree, and
      one whose state is UNKNOWN establishes nothing;
    * below the capability floor — `fleet_hierarchy.fits()` decides, so a LOCAL
      or SMALL model's worktree is not handed district-rank work;
    * unrelated — no shared paths, no shared area, no shared pillar, and no
      common supervisor below the board.
    """
    target_layer = dormant_candidate.layer
    dormant_paths = set(dormant_candidate.touched_paths)
    dormant_areas = set(dormant_candidate.areas)
    dormant_pillars = {a.pillar for a in dormant_candidate.areas}

    found: list = []
    refusals: list = []

    for other in live or ():
        if other.path == dormant_candidate.path:
            continue
        if other.dormancy != ACTIVE:
            refusals.append(
                f"{other.label}: {other.dormancy.lower()}, so it cannot take on "
                f"work — an inheritor must already be awake")
            continue

        # Authority before affinity. A worktree in exactly the right area is
        # still the wrong inheritor if the work outranks what it may act on.
        #
        # Compared class-to-class rather than through `fits()`, which takes a
        # MODEL NAME. Handing it a capability class happens to produce the right
        # boolean via the `measured` argument and produces a nonsense evidence
        # string with it -- "no entry rule matched 'MID'" -- and a refusal whose
        # stated reason is gibberish is the kind that gets overridden.
        if not _meets_floor(other.capability, target_layer):
            refusals.append(
                f"{other.label}: its work is {other.capability}-class and the "
                f"dormant work is {target_layer}-rank, which needs "
                f"{HIER.BY_KEY[target_layer].capability}")
            continue

        shared = tuple(sorted(dormant_paths & set(other.touched_paths)))
        other_areas = set(other.areas)
        supervisor = _common_supervisor(other.layer, target_layer)

        if shared:
            fitness = IN_THE_CODE
            evidence = (f"already has outstanding edits to {len(shared)} of the "
                        f"path(s) this work touches, so it can absorb the work "
                        f"without a second checkout")
        elif dormant_areas & other_areas:
            common = sorted(str(a) for a in (dormant_areas & other_areas))
            fitness = SAME_AREA
            evidence = f"working in the same area: {', '.join(common)}"
        elif dormant_pillars & {a.pillar for a in other_areas}:
            common = sorted(dormant_pillars & {a.pillar for a in other_areas})
            fitness = SAME_PILLAR
            evidence = f"working in the same pillar: {', '.join(common)}"
        elif supervisor:
            fitness = SAME_SUPERVISOR
            evidence = (f"different work, but {other.layer} and {target_layer} "
                        f"both escalate to {supervisor}")
        else:
            refusals.append(
                f"{other.label}: no shared paths, area or pillar, and "
                f"{other.layer} and {target_layer} share no supervisor below "
                f"the board")
            continue

        found.append(Inheritor(
            path=other.path, label=other.label, session=other.session,
            fitness=fitness, evidence=evidence, supervisor=supervisor,
            already_in_the_code=bool(shared), shared_paths=shared))

    found.sort(key=lambda i: (_FITNESS_ORDER.index(i.fitness),
                              -i.overlap_count, i.label))
    if not found and not refusals:
        refusals.append("no other worktree was considered; there is nobody else "
                        "in this repository to hand the work to")
    return Handover(dormant=dormant_candidate, inheritors=tuple(found),
                    refusals=tuple(refusals))


def handovers(candidates: Sequence[Candidate]) -> Tuple[Handover, ...]:
    """A handover for every dormant candidate, against the live ones."""
    pool = tuple(candidates or ())
    awake = tuple(c for c in pool if c.dormancy == ACTIVE)
    return tuple(handover(c, awake) for c in pool if c.dormancy == DORMANT)


def graph_of(candidates: Sequence[Candidate],
             conflicts: Sequence[Overlap] = ()) -> dict:
    """`{"nodes": [...], "edges": [...]}` for a caller that draws the selection.

    Plain data, deliberately. A dict can be asserted on in a test with no GUI,
    serialised to a panel, or diffed between two selections to animate the
    difference — which is what a screen that rebuilds as the user ticks boxes
    actually needs.

    Node kinds: `worktree`, `area`. Edge kinds: `works-in` (a worktree to an
    area it has outstanding work in) and `contends` (two worktrees over shared
    paths). Areas are nodes rather than labels because two worktrees in one area
    is the shape a reader is looking for, and an edge through a shared node
    shows it without the layout having to.
    """
    nodes: list = []
    edges: list = []
    seen_areas: set = set()

    for candidate in candidates or ():
        nodes.append({
            "id": candidate.path, "kind": "worktree", "label": candidate.label,
            "branch": candidate.branch or "", "session": candidate.session,
            "dormancy": candidate.dormancy, "age_days": candidate.age_days,
            "rank": candidate.rank, "document": candidate.document,
            "layer": candidate.layer, "capability": candidate.capability,
            "touched": len(candidate.touched_paths),
        })
        for area in candidate.areas:
            key = f"area:{area}"
            if key not in seen_areas:
                seen_areas.add(key)
                nodes.append({"id": key, "kind": "area", "label": str(area),
                              "pillar": area.pillar, "surface": area.surface})
            edges.append({"source": candidate.path, "target": key,
                          "kind": "works-in"})

    for conflict in conflicts or ():
        if not conflict.path:
            continue
        edges.append({"source": conflict.path, "target": conflict.path,
                      "kind": "contends", "weight": conflict.count,
                      "label": conflict.label, "reason": conflict.reason,
                      "paths": list(conflict.shared_paths)})
    return {"nodes": nodes, "edges": edges}


__all__ = [
    "ACTIVE", "DECLARED", "DORMANT", "DORMANT_AFTER_DAYS", "IN_THE_CODE",
    "LIMITS", "OBSERVED", "SAME_AREA", "SAME_PILLAR", "SAME_SUPERVISOR",
    "UNKNOWN", "Candidate", "Handover", "Inheritor", "Overlap", "ResumePlan",
    "dormant", "graph_of", "handover", "handovers", "plan", "survey",
]
