"""Which desks to open, which team in each, and what is deliberately not opened.

`fleet_dispatch` answers *which model should do this brief*. `development_chain`
answers *what is the organisation*. Neither answers the question a session
actually faces when it is handed a pile of work: **which desks should be running
right now, and which should stay shut.**

That question is the whole point, and the answer is almost always "fewer than you
could". A tool that can open twenty desks will open twenty desks, and twenty
desks produce two hundred commits a day against one another — work written over,
work never merged, work duplicated three desks apart. Capacity was never the
constraint in an engineering organisation and is not the constraint here. The
constraint is **coordination**, and an organisation's answer to it is that a CTO
opens a small number of desks on purpose and leaves the rest closed.

So this module is a **selector, and mostly a refuser**. It takes the work, the
org chart, who is already busy, and a budget; it returns the activations to make
and — with equal weight — every demand it withheld and why. A plan that returned
only its activations would be indistinguishable from a plan that lost half its
input, which is exactly how work goes missing in a fleet this size.

Six rules do the selecting, and each is here because collapsing it would make the
plan look more decisive than the evidence supports.

1. **A demand routes by AREA, never by name.** A desk owns repository areas;
   `worktree_areas` places a path in one. A demand that reaches no desk's areas
   is `UNROUTABLE` and is withheld — it is never given to a default desk, because
   "nobody owns this" and "the first desk owns this" are different facts and only
   one of them is true.

2. **A demand reaching several desks is CROSS-DESK and is labelled so.** It is
   assigned to the desk with the most coverage and the others are recorded as
   consulted. Silently filing it under one desk is how a change lands with half
   its reviewers absent, and refusing it outright would stall exactly the work
   that most needs coordinating.

3. **Within a desk, the team whose owned paths cover most of the demand takes
   it.** Ties break on the team's declared order, then its key, so two readings
   of one situation agree.

4. **The budget binds, and what it excluded is named with the number that
   excluded it.** This is the rule the module exists for. Desks are opened in
   demand-priority order until the budget is spent; everything after that is
   `WITHHELD-BUDGET` and says so.

5. **A desk already occupied is not opened again.** Occupancy is supplied by the
   caller — the mesh knows who is editing where — and a busy desk's new demands
   queue rather than doubling up. This is what stops a second team being sent
   into a lane a first team is mid-change in.

6. **A disabled or retired desk, team, or worker is never selected.** State is
   read, never inferred from absence. Work owned only by a closed desk is
   `DESK-CLOSED`, not `UNROUTABLE`.

**What this does not do, deliberately.** It does not spawn anything, start a
worktree, write to a store, or talk to git. It returns one mandatory planning
input; the commanding layer decides whether and how to act after combining it
with model standing, separability, reuse, risk, and evidence constraints.
`AGENTS.md` §18 governs what a session owes the fleet once it does. The lead
publishes the dispatch record; a spawned teammate publishes its own lane record
when its harness exposes coordination tools, and the lead relays only when
direct publication is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from alelyon.runtime.common import development_chain as DC
from alelyon.runtime.common import worktree_areas as AREAS

SCHEMA = "alelyon.desk-dispatch/v1"

#: Why a demand was not dispatched. Each is a different fact and they must not
#: collapse into "not scheduled" -- one is a coverage hole in the org chart, one
#: is a deliberate limit, and one is somebody else already working.
UNROUTABLE = "unroutable"
WITHHELD_BUDGET = "withheld-budget"
DESK_OCCUPIED = "desk-occupied"
DESK_CLOSED = "desk-closed"
NO_TEAM = "no-team"

WITHHELD_REASONS = (UNROUTABLE, WITHHELD_BUDGET, DESK_OCCUPIED, DESK_CLOSED,
                    NO_TEAM)

#: What a plan cannot establish. Printed by every surface that shows one, for
#: the same reason `desk_lanes.LANE_LIMITS` is.
DISPATCH_LIMITS: tuple[str, ...] = (
    "A plan is a READ-ONLY RECOMMENDATION. The commanding layer must consult "
    "it before desk or team activation. Nothing here spawns an agent, opens a "
    "worktree, or reserves anything, and a second session planning against the "
    "same snapshot will produce the same plan rather than a conflicting one -- "
    "which is a property of purity, not a lock.",
    "Routing is by declared AREA. A demand that names no path routes nowhere, "
    "and a demand whose paths are wrong routes confidently to the wrong desk. "
    "The paths and caller-supplied repository area space are the inputs this "
    "trusts and cannot check.",
    "Occupancy is supplied by the caller and is only as current as the reading "
    "behind it. A desk that became busy after the reading is planned against as "
    "though it were free.",
    "The budget bounds DESKS and teams, never the work. Withheld demands are "
    "still outstanding; nothing here queues, schedules, or remembers them.",
    "Demand weight is caller-supplied priority, not measured token cost or "
    "quality. This chooses ownership under a concurrency budget; model, layer, "
    "separability, context reuse, review, and expected rework remain part of "
    "the commanding layer's total-cost route.",
    "Cross-desk assignment names a primary and consults the rest. It does not "
    "establish that the primary desk can complete the change alone.",
)


@dataclass(frozen=True)
class Demand:
    """One piece of work wanting a desk.

    `paths` is what routes it. `weight` orders the competition for a bounded
    number of desks; equal weights break on `key` so the plan is stable.
    """

    key: str
    label: str
    paths: tuple[str, ...] = ()
    #: Higher goes first. The caller's priority, not a computed urgency.
    weight: int = 0


@dataclass(frozen=True)
class Activation:
    """One desk opened, with the team chosen and the work assigned to it."""

    desk_key: str
    team_key: str
    demands: tuple[str, ...]
    #: Desks that also own paths in these demands. Named, never merged in.
    consulted_desks: tuple[str, ...] = ()
    #: Why this team rather than another, in one line.
    rationale: str = ""

    @property
    def cross_desk(self) -> bool:
        return bool(self.consulted_desks)


@dataclass(frozen=True)
class Withheld:
    """One demand not dispatched, and which of the reasons applies."""

    demand_key: str
    reason: str
    detail: str = ""


@dataclass(frozen=True)
class DispatchBudget:
    """How much of the organisation may be running at once.

    `max_desks` bounds parallel desks. V1 chooses at most one primary team per
    desk, so `max_teams_per_desk=0` closes every new desk and any positive value
    permits that one team. This is the difference between an organisation and
    a stampede.
    """

    max_desks: int = 3
    max_teams_per_desk: int = 1

    def __post_init__(self) -> None:
        if self.max_desks < 0 or self.max_teams_per_desk < 0:
            raise ValueError("a dispatch budget cannot be negative")


@dataclass(frozen=True)
class DispatchPlan:
    """What to open, what was withheld, and what the plan cannot say."""

    schema: str = SCHEMA
    activations: tuple[Activation, ...] = ()
    withheld: tuple[Withheld, ...] = ()
    budget: DispatchBudget = field(default_factory=DispatchBudget)
    limits: tuple[str, ...] = DISPATCH_LIMITS

    @property
    def desks_opened(self) -> tuple[str, ...]:
        return tuple(sorted({a.desk_key for a in self.activations}))

    def withheld_by_reason(self) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for row in self.withheld:
            out.setdefault(row.reason, []).append(row.demand_key)
        return {reason: tuple(keys) for reason, keys in sorted(out.items())}


def _covers(owned: tuple[str, ...], path: str) -> bool:
    """Does a declared area or owned path prefix cover this path?

    Prefix matching on a path SEPARATOR boundary, so `alelyon/runtime/common`
    does not swallow `alelyon/runtime/common_extras`. That collision is the one
    a naive `startswith` makes, and it silently widens a desk's ownership.
    """
    for prefix in owned:
        prefix = prefix.rstrip("/")
        if not prefix:
            continue
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _holds_area(held: tuple[str, ...], area: str) -> bool:
    """Whether a desk holding ``held`` owns one fleet area coordinate."""
    for candidate in held:
        if area == candidate:
            return True
        if area.startswith(candidate) and area[len(candidate)] in "/.":
            return True
    return False


def _desk_coverage(
        snapshot: DC.HierarchySnapshot,
        demand: Demand,
        space: AREAS.AreaSpace,
        *,
        active_only: bool,
) -> dict[str, int]:
    """Desk key -> demand paths whose fleet areas that desk owns."""
    scores: dict[str, int] = {}
    for desk in snapshot.desks:
        if active_only and desk.state != DC.ACTIVE:
            continue
        if not active_only and desk.state == DC.ACTIVE:
            continue
        hits = sum(
            1 for path in demand.paths
            if _holds_area(desk.areas, str(space.area_of(path)))
        )
        if hits:
            scores[desk.key] = hits
    return scores


def _pick_team(snapshot: DC.HierarchySnapshot, desk_key: str,
               demands: tuple[Demand, ...]) -> tuple[str, str]:
    """(team key, rationale) for the team best covering these demands.

    A desk with no active team is a real state and returns an empty key rather
    than falling back to the desk itself. `development_chain`'s own rule is that
    a missing team never collapses to its desk, and inventing one here would
    reintroduce exactly that.
    """
    paths = tuple(path for demand in demands for path in demand.paths)
    # Rank each team by (most coverage, then declared order, then key) and take
    # the smallest tuple. Written as one sort key rather than as a running
    # comparison because the running version is where the tie-break quietly
    # stops being deterministic.
    ranked = sorted(
        ((-sum(1 for path in paths if _covers(team.owned_paths, path)),
          team.order, team.key)
         for team in snapshot.teams
         if team.desk_key == desk_key and team.state == DC.ACTIVE))
    if not ranked:
        return "", ""
    negative_hits, _order, best_key = ranked[0]
    best_hits = -negative_hits
    if best_hits <= 0:
        return best_key, (f"no team under {desk_key} declares a path in this "
                          f"work; taking the desk's first active team by "
                          f"declared order")
    return best_key, (f"covers {best_hits} of {len(paths)} path(s) this work "
                      f"names")


def plan(snapshot: DC.HierarchySnapshot, demands: tuple[Demand, ...], *,
         budget: DispatchBudget | None = None,
         occupied_desks: frozenset[str] = frozenset(),
         space: AREAS.AreaSpace | None = None) -> DispatchPlan:
    """Choose the desks to open. Pure: records in, records out.

    ``space`` must describe the repository from which ``Demand.paths`` came.
    Deterministic in the strong sense — two callers handed the same snapshot,
    demands, budget, occupancy and area space produce byte-identical plans.
    That is what lets a second session check a first session's dispatch instead
    of relitigating it.
    """
    budget = budget or DispatchBudget()
    space = space or AREAS.default_space()
    withheld: list[Withheld] = []

    # Route every demand first, so the budget is spent on work that has
    # somewhere to go. Routing before budgeting also means an UNROUTABLE demand
    # is reported as unroutable rather than as "no budget left" -- a coverage
    # hole in the org chart must not be able to hide behind a limit.
    routed: dict[str, list[Demand]] = {}
    consulted: dict[str, set[str]] = {}
    for demand in sorted(demands, key=lambda d: (-d.weight, d.key)):
        coverage = _desk_coverage(
            snapshot, demand, space, active_only=True)
        if not coverage:
            closed = _desk_coverage(
                snapshot, demand, space, active_only=False)
            if closed:
                states = {desk.key: desk.state for desk in snapshot.desks}
                detail = ", ".join(
                    f"{key} is {states[key]}" for key in sorted(closed))
                withheld.append(Withheld(
                    demand.key, DESK_CLOSED,
                    f"the work is owned only by a closed desk: {detail}"))
                continue
            withheld.append(Withheld(
                demand.key, UNROUTABLE,
                f"none of the {len(demand.paths)} path(s) it names falls in an "
                f"active desk's declared areas"))
            continue
        # Most coverage takes it; the key breaks a tie so the choice is stable.
        primary = min(coverage, key=lambda key: (-coverage[key], key))
        routed.setdefault(primary, []).append(demand)
        others = {key for key in coverage if key != primary}
        if others:
            consulted.setdefault(primary, set()).update(others)

    # Desks in the order their heaviest demand competes at, so the budget buys
    # the most important work rather than whichever desk sorted first.
    def desk_rank(desk_key: str) -> tuple[int, str]:
        best = max(routed[desk_key], key=lambda d: (d.weight, d.key))
        return (-best.weight, desk_key)

    activations: list[Activation] = []
    opened = 0
    for desk_key in sorted(routed, key=desk_rank):
        group = tuple(sorted(routed[desk_key], key=lambda d: (-d.weight, d.key)))
        if desk_key in occupied_desks:
            withheld.extend(Withheld(
                d.key, DESK_OCCUPIED,
                f"{desk_key} already has work in flight; this queues rather "
                f"than opening a second team into the same lane") for d in group)
            continue
        if opened >= budget.max_desks:
            withheld.extend(Withheld(
                d.key, WITHHELD_BUDGET,
                f"the budget opens {budget.max_desks} desk(s) at once and they "
                f"are taken; this is outstanding, not scheduled") for d in group)
            continue
        if budget.max_teams_per_desk == 0:
            withheld.extend(Withheld(
                d.key, WITHHELD_BUDGET,
                "the budget allows 0 team(s) per desk; this is outstanding, "
                "not scheduled") for d in group)
            continue
        team_key, rationale = _pick_team(snapshot, desk_key, group)
        if not team_key:
            withheld.extend(Withheld(
                d.key, NO_TEAM,
                f"{desk_key} has no active team, and a missing team does not "
                f"collapse to its desk") for d in group)
            continue
        activations.append(Activation(
            desk_key=desk_key, team_key=team_key,
            demands=tuple(d.key for d in group),
            consulted_desks=tuple(sorted(consulted.get(desk_key, ()))),
            rationale=rationale))
        opened += 1

    return DispatchPlan(
        activations=tuple(activations),
        withheld=tuple(sorted(withheld,
                              key=lambda w: (w.reason, w.demand_key))),
        budget=budget)


__all__ = [
    "Activation", "DESK_CLOSED", "DESK_OCCUPIED", "DISPATCH_LIMITS", "Demand",
    "DispatchBudget", "DispatchPlan", "NO_TEAM", "SCHEMA", "UNROUTABLE",
    "WITHHELD_BUDGET", "WITHHELD_REASONS", "Withheld", "plan",
]
