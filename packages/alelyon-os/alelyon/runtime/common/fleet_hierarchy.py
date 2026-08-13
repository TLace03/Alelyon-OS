"""The canonical layer space: which rank of the fleet may decide what.

A fleet today is flat. A session fans out twenty agents, every one of them runs
whatever model the caller happened to name, and nothing records that judging a
design and renaming a symbol are not the same job. This is the coordinate space
that fixes that — six layers, from the one that sets policy to the one that does
the routine work — and the rule that a layer's authority is bounded by what
`AGENTS.md` §3 already says about risk.

Two things this is NOT
----------------------
**Not a second risk taxonomy.** `max_tier` on every layer is an
`AGENTS.md` §3 tier, copied rather than reinvented, and a test fails if the two
drift. Inventing a parallel severity scale would mean two answers to "may this
change ship" and no rule for which wins.

**Not an autonomous scheduler.** Nothing here launches an agent or mutates a
dispatch, and nothing can. A layer assignment is data the commanding model must
read before choosing a model for an agent, in the same way `Domain` is data the
assistant reads before choosing tools. It answers "what rank is this work?" and
supplies a capability floor; the commanding model chooses the lowest-cost
admissible execution graph and remains accountable for that choice.

The board is not a model, and that is a design truth
----------------------------------------------------
`AGENTS.md` §3 requires **explicit owner authority** for a Tier 3 behaviour
change, and §1 forbids a probationary model from making one at all. So the top
layer is not a frontier model with a bigger budget — it is the **owner**, and a
model at the executive layer may only prepare material for that decision.
`Layer.human_only` carries it, and no promotion can ever set it False: a ratchet
that could promote a model into owner authority would be a ratchet that
dissolves the one gate the repository is built around.

Where a model starts, and where it ends up
------------------------------------------
`ENTRY_CLASS` maps a model name to a starting capability class. It is a
**declaration with a date on it**, not a measurement, and it exists only so a
model that has never run has somewhere to begin. What a model is actually
*worth* at a layer is decided by `fleet_ledger`, from runs that happened. The
table is the door; the ledger is the career.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

#: Bumped when a layer is added, removed, or its authority changes. Same
#: contract as `worktree_areas.AREA_SPACE_VERSION`: extending the space is a
#: version change, not a refactor, because every ledger row is keyed on it.
LAYER_SPACE_VERSION = 1

#: Returned wherever a layer could not be derived. A value, never a blank.
UNPLACED = "UNPLACED"

# ── capability classes ───────────────────────────────────────────────────────
#: What a model is *for*, coarsely. Four classes rather than a score, because a
#: score invites arithmetic between models that were never measured on one
#: scale, and this vocabulary only has to be good enough to say "do not send the
#: cheap one to referee a design".
FRONTIER = "FRONTIER"    # the largest available; reserved for judgement
MID = "MID"              # competent implementation and review
SMALL = "SMALL"          # mechanical, high-volume, cheap
LOCAL = "LOCAL"          # runs on this machine; no data leaves it
CLASSES = (FRONTIER, MID, SMALL, LOCAL)

#: Relative cost weight per class, for reporting only. Deliberately ordinal and
#: unitless: real prices change, differ per provider, and a figure that looks
#: like dollars would be read as dollars. `fleet_ledger` measures TOKENS, which
#: is a fact, and leaves money to whoever holds the invoice.
COST_WEIGHT = {FRONTIER: 8, MID: 3, SMALL: 1, LOCAL: 0}


@dataclass(frozen=True, order=True)
class Layer:
    """One rank of the hierarchy, and the bounds on what it may do."""

    rank: int
    key: str
    title: str
    #: What this layer exists to decide, in the vocabulary of the work.
    decides: str
    #: Highest AGENTS.md §3 risk tier this layer may act on. Copied from that
    #: document; never a second scale.
    max_tier: int
    #: The capability class its work is worth. Higher layers command better
    #: models; lower layers command cheaper ones.
    capability: str
    #: Whether a layer may *approve* another layer's work rather than only do
    #: its own. Approval is what makes a hierarchy more than a labelling.
    may_approve: bool
    #: Whether only a human may occupy this layer. True for the board alone.
    human_only: bool = False
    #: What it escalates to when the work exceeds its authority.
    escalates_to: str = ""

    def __str__(self) -> str:
        return self.key

    @property
    def cost_weight(self) -> int:
        return COST_WEIGHT[self.capability]

    def may_act_on(self, tier: int) -> bool:
        """Whether work at this risk tier is within the layer's authority."""
        return int(tier) <= self.max_tier


#: The space. Ordered from the top down; `rank` is the identity and the order.
LAYERS: tuple[Layer, ...] = (
    Layer(rank=0, key="board", title="Board",
          decides="the charter itself: what the layers are, what each may "
                  "approve, and every Tier 3 authorisation",
          max_tier=3, capability=FRONTIER, may_approve=True, human_only=True,
          escalates_to=""),
    Layer(rank=1, key="executive", title="Executive",
          decides="cross-pillar design, refereeing another layer's conclusion, "
                  "and accepting or refusing a promotion",
          max_tier=2, capability=FRONTIER, may_approve=True,
          escalates_to="board"),
    Layer(rank=2, key="regional", title="Regional",
          decides="one pillar's architecture and the specifications under it",
          max_tier=2, capability=FRONTIER, may_approve=True,
          escalates_to="executive"),
    Layer(rank=3, key="district", title="District",
          decides="one subsystem: a multi-file change and the tests that would "
                  "falsify it",
          max_tier=2, capability=MID, may_approve=True,
          escalates_to="regional"),
    Layer(rank=4, key="general", title="General & assistant management",
          decides="one change: implement it, test it, and report what it cost",
          max_tier=1, capability=MID, may_approve=False,
          escalates_to="district"),
    Layer(rank=5, key="line", title="Line",
          decides="routine work with a stated procedure: inventory, rename, "
                  "format, single-file edit, mechanical sweep",
          max_tier=1, capability=SMALL, may_approve=False,
          escalates_to="general"),
)

BY_KEY: dict[str, Layer] = {layer.key: layer for layer in LAYERS}
BY_RANK: dict[int, Layer] = {layer.rank: layer for layer in LAYERS}

#: The bottom layer, used wherever work could not be placed. Deliberately the
#: *cheapest* and least authorised: an unplaced job must not inherit the
#: authority of the layer that happened to ask for it, and the failure mode of
#: guessing too low is a needless escalation rather than an unreviewed change.
DEFAULT_LAYER = BY_KEY["line"]


# ── placing work ─────────────────────────────────────────────────────────────
#: Work kinds, and the layer each belongs to. The vocabulary is closed because
#: the kind decides the model that will be spent on it, and free text there
#: would make the hierarchy advisory in the worst way — every caller inventing a
#: kind that happens to route where they wanted to go.
WORK_KINDS: dict[str, str] = {
    # board
    "charter-change": "board",
    "tier3-authorisation": "board",
    "release-authority": "board",
    # executive
    "design-review": "executive",
    "referee": "executive",
    "adversarial-review": "executive",
    "promotion-decision": "executive",
    "cross-pillar-design": "executive",
    # regional
    "architecture": "regional",
    "specification": "regional",
    "research-synthesis": "regional",
    "threat-model": "regional",
    # district
    "subsystem-change": "district",
    "test-design": "district",
    "audit": "district",
    "migration-plan": "district",
    # general
    "implement": "general",
    "fix-defect": "general",
    "write-test": "general",
    "documentation": "general",
    # line
    "inventory": "line",
    "rename": "line",
    "format": "line",
    "mechanical-sweep": "line",
    "lookup": "line",
}

#: Phrases that place a *described* job when no kind was named. Ordered, first
#: match wins. This is a convenience for a caller holding prose, and it is
#: DERIVED-by-rule rather than understood — the limits say so.
_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), kind) for pattern, kind in (
        (r"\bauthoris|\bauthoriz|\bsign(ing)? key|\brelease\b", "tier3-authorisation"),
        (r"\brefere|\badversar|\bred[- ]team|\bcritiqu", "referee"),
        (r"\bdesign review|\breview the design|\bjudge\b", "design-review"),
        (r"\barchitect|\bspecif|\bthreat model", "architecture"),
        (r"\baudit\b|\bsubsystem|\bmigrat", "audit"),
        (r"\btest\b|\bfalsifi", "write-test"),
        (r"\bimplement|\bbuild\b|\bfix\b|\bdefect|\bbug\b", "implement"),
        (r"\brename|\bformat|\bsweep|\binventor|\blist all|\bfind all", "rename"),
    ))

LIMITS: tuple[str, ...] = (
    "A layer bounds AUTHORITY, not competence. Nothing here says a line-layer "
    "model cannot reason well; it says work of that kind does not get to "
    "approve anything, and that a change above its tier must escalate.",
    "Placing work from prose is a phrase rule, not understanding. A job "
    "described in words no pattern matches is placed at the bottom layer, "
    "which is the safe direction - it escalates rather than acts.",
    "This module dispatches nothing. The highest active model layer must read "
    "its placement before dispatch, choose the cheapest model that meets the "
    "capability and authority floors, and record why it overrides available "
    "standing evidence.",
    "The board layer cannot be occupied by a model. AGENTS.md section 3 "
    "requires explicit owner authority for Tier 3, so a model may prepare that "
    "decision and never make it.",
    "Cost weights are ordinal and unitless. Real prices differ by provider and "
    "change without notice; the ledger measures tokens, which is a fact.",
)


def layer(key: str) -> Layer | None:
    return BY_KEY.get(str(key or ""))


def layer_for_kind(kind: str) -> Layer:
    """The layer a named work kind belongs to, or the bottom layer."""
    return BY_KEY.get(WORK_KINDS.get(str(kind or ""), ""), DEFAULT_LAYER)


#: The highest layer a model may occupy. Everything above it is the owner's.
TOP_MODEL_LAYER = BY_KEY["executive"]


def _model_layer(target: Layer, evidence: str) -> tuple[Layer, str]:
    """Never hand a model the board's work.

    Found by running the placer over 1,006 real agent runs: briefs mentioning
    a release or an authorisation matched the board phrase rule, and the ledger
    then happily scored agents at a layer whose whole definition is that no
    model may occupy it. Placing work there is not a harmless label — it is the
    hierarchy quietly recording that a model did the owner's job.

    So board-level *subject matter* places at the executive layer, which is what
    actually happens: a model prepares the decision and the owner makes it.
    """
    if not target.human_only:
        return target, evidence
    return TOP_MODEL_LAYER, (
        f"{evidence}; that is {target.key}-level subject matter, which no model "
        f"may occupy, so it is placed at {TOP_MODEL_LAYER.key} — a model may "
        f"prepare an owner decision and may not make one")


def board_matter(text: str) -> bool:
    """Whether a job describes something only the owner may decide.

    Separate from `place()` on purpose. The layer answers "who does the work";
    this answers "whose signature does it need", and conflating them is how a
    model ends up recorded as having authorised something.
    """
    for pattern, kind in _PHRASES:
        if pattern.search(str(text or "")):
            return layer_for_kind(kind).human_only
    return False


def placement(text: str) -> tuple[Layer, str, str]:
    """(layer, work_kind, evidence) — `place()` without discarding the kind.

    The ledger's coordinate is `(layer, work_kind)`, so a caller holding only a
    layer names a row it cannot find. The same match that chooses the layer
    already knows the kind; `place()` simply threw it away, which is why nothing
    could look a brief up in the ledger.

    The kind is **""** when nothing matched. That is an absence for the caller to
    handle, not a default it may use: guessing a kind would fabricate a
    coordinate and read somebody else's standing out of it.

    One asymmetry is deliberate. When the kind is board-level, `_model_layer`
    moves the layer to `executive` while the kind stays what it matched — so the
    pair is **not** a coordinate the ledger holds. `board_matter()` is what
    detects that, and `fleet_dispatch` refuses to name a model from a standing
    there rather than reading the executive row for a decision that is the
    owner's.
    """
    body = str(text or "")
    known = WORK_KINDS.get(body.strip())
    if known:
        target, evidence = _model_layer(
            BY_KEY[known], f"the work kind {body.strip()!r} names this layer")
        return target, body.strip(), evidence
    for pattern, kind in _PHRASES:
        if pattern.search(body):
            target, evidence = _model_layer(
                layer_for_kind(kind),
                f"the phrase rule {pattern.pattern!r} placed it as {kind!r}")
            return target, kind, evidence
    return DEFAULT_LAYER, "", ("no work kind and no phrase rule matched, so it "
                               "is placed at the bottom layer and must escalate "
                               "rather than assume authority")


def place(text: str) -> tuple[Layer, str]:
    """(layer, evidence) for a described job. Never returns the board.

    A caller that knows its work kind should pass it to `layer_for_kind`. This
    is for the case that actually occurs — an agent brief written in prose —
    and it says which rule fired so a reader can disagree with the rule.
    """
    target, _kind, evidence = placement(text)
    return target, evidence


def escalation_path(key: str) -> tuple[str, ...]:
    """Every layer above this one, in the order it would escalate."""
    out: list[str] = []
    current = layer(key)
    seen: set[str] = set()
    while current is not None and current.escalates_to:
        if current.escalates_to in seen:
            break                                   # a cycle is a code defect
        seen.add(current.escalates_to)
        out.append(current.escalates_to)
        current = layer(current.escalates_to)
    return tuple(out)


def approves(reviewer: str, subject: str) -> tuple[bool, str]:
    """Whether `reviewer` may approve work done at `subject`, and why not.

    Two conditions, both structural: the reviewer must be a layer that approves
    at all, and it must sit strictly above the work. A layer approving its own
    output is the failure this exists to prevent — a fleet of twenty agents
    marking their own homework is exactly what a flat fleet does today.
    """
    above, below = layer(reviewer), layer(subject)
    if above is None or below is None:
        return False, "one of those is not a layer in this space"
    if not above.may_approve:
        return False, f"{above.key} does not approve; it escalates to " \
                      f"{above.escalates_to or 'nothing'}"
    if above.rank >= below.rank:
        return False, (f"{above.key} does not sit above {below.key}; a layer "
                       f"cannot approve its own rank or one above it")
    return True, f"{above.key} sits above {below.key} and may approve"


# ── where a model enters ─────────────────────────────────────────────────────
#: Starting capability class by model-name pattern. **A declaration, dated
#: 2026-08-03, not a measurement.** It exists so a model that has never run has
#: somewhere to begin; `fleet_ledger` decides where it ends up. Ordered, first
#: match wins.
#:
#: It encodes the owner's standing directive — the largest models for theory,
#: refereeing and design; the small ones for mechanical work — rather than
#: inventing a ranking of its own.
ENTRY_CLASS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), klass) for pattern, klass in (
        (r"opus|gpt-[45]|gemini.*(ultra|pro)|frontier", FRONTIER),
        (r"sonnet|fable", MID),
        (r"haiku|mini|flash|small", SMALL),
        (r"^ollama|:\d+b|localhost|127\.0\.0\.1|qwen|llama|mistral", LOCAL),
    ))


def entry_class(model: str) -> tuple[str, str]:
    """(class, evidence) for a model name that has no measured record yet."""
    name = str(model or "")
    if not name or name == "<synthetic>":
        return SMALL, ("no model name was recorded, so it enters at the "
                       "cheapest class rather than the most capable")
    for pattern, klass in ENTRY_CLASS:
        if pattern.search(name):
            return klass, (f"the name matched {pattern.pattern!r}; a naming "
                           f"convention is a declaration, and only the ledger "
                           f"can move it")
    return SMALL, (f"no entry rule matched {name!r}; an unknown model starts at "
                   f"the cheapest class and earns its way up")


def fits(model: str, layer_key: str, *, measured: str | None = None
         ) -> tuple[bool, str]:
    """Whether a model is good enough for a layer's work.

    `measured` is the class the ledger has established for this model. **It can
    only raise, never lower** — and that direction is load-bearing rather than
    generous. A model's measured class is the capability of the highest layer it
    holds, so a frontier model whose first standing happened to be won at a low
    layer would otherwise be *capped* by that success and locked out of the work
    it is actually for. On this repository's history that handed
    `executive/design-review` to a model scoring 0.716 over one scoring 0.943,
    because the better one had won `district` first.

    Demotion is the channel for performing badly. The capability class is not a
    punishment, so absence of evidence at a high layer never demotes a model
    below what it is.
    """
    target = layer(layer_key)
    if target is None:
        return False, f"{layer_key!r} is not a layer in this space"
    order = {FRONTIER: 3, MID: 2, SMALL: 1, LOCAL: 1}
    declared, evidence = entry_class(model)
    klass = declared
    if measured and order.get(measured, 0) > order.get(declared, 0):
        klass = measured
        evidence = ("measured by the ledger from completed runs, above what "
                    "its name declares")
    elif measured:
        evidence = f"{evidence}; the ledger has measured it no higher"
    ok = order.get(klass, 0) >= order.get(target.capability, 0)
    verdict = "meets" if ok else "is below"
    return ok, (f"{model or 'UNATTRIBUTED'} is {klass} ({evidence}) and "
                f"{verdict} {target.key}'s {target.capability} floor")


def charter() -> str:
    """The hierarchy as one printable table, for a reader or a report."""
    lines = [f"Fleet layer space v{LAYER_SPACE_VERSION}", ""]
    lines.append(f"  {'RANK':<5} {'LAYER':<11} {'TIER':<5} {'MODEL':<9} "
                 f"{'APPROVES':<9} DECIDES")
    for entry in LAYERS:
        lines.append(
            f"  {entry.rank:<5} {entry.key:<11} <={entry.max_tier:<3} "
            f"{entry.capability:<9} {'yes' if entry.may_approve else 'no':<9} "
            f"{entry.decides}")
    lines += ["", "  The board layer is the OWNER. AGENTS.md section 3 requires "
                  "explicit owner", "  authority for Tier 3, so no model may "
                  "occupy it.", ""]
    lines.append("WHAT THIS CANNOT TELL YOU")
    lines += [f"  - {limit}" for limit in LIMITS]
    return "\n".join(lines)


__all__ = [
    "BY_KEY", "BY_RANK", "CLASSES", "COST_WEIGHT", "DEFAULT_LAYER",
    "ENTRY_CLASS", "FRONTIER", "LAYERS", "LAYER_SPACE_VERSION", "LIMITS",
    "LOCAL", "MID", "SMALL", "UNPLACED", "Layer", "WORK_KINDS", "approves",
    "board_matter", "charter", "entry_class", "escalation_path", "fits",
    "layer", "layer_for_kind", "place", "placement",
]
