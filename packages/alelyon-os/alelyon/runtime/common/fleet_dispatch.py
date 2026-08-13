"""Read a standing before naming a model for an agent.

`docs/features/FLEET-HIERARCHY.md` §6 calls this "the step that turns this from
a record into a policy". This module reads the ledger and returns a
**recommendation with its provenance attached**; it never launches an agent,
selects a model, or mutates the ledger. The highest active model layer must read
the recommendation before dispatch and remains responsible for the route it
chooses: honour the measured standing, or record the concrete reason an override
has lower expected total cost while preserving capability, risk, and evidence
floors.

What it adds is the join that was missing. `fleet_hierarchy.place()` says which
layer a brief belongs to and `fleet_ledger.standing()` says which model holds a
coordinate, and until now no code put the two together, so a caller wanting to
honour the hierarchy had to do it by hand and in practice nobody did.

Three separations do the work, and each exists because collapsing it would make
the answer look stronger than it is:

1. **A named model and a named class are different answers.** With a standing,
   this returns a *model* somebody measured. Without one it returns the layer's
   *capability class* and no model at all — because the class is a property of
   the layer, which is known, while the model would be a guess.
2. **No evidence and no database are different absences.** A readable ledger
   holding no standing is `DERIVED`: we looked, and nothing has been measured
   here. A ledger that does not exist is `UNMEASURED`: we could not look. The
   second is not evidence about models and is never reported as if it were.
3. **Doing the work and authorising it are different questions.** Board-level
   subject matter is placed at `executive` for the doing, and this refuses to
   name a model as *deciding* it. `board_matter()` is asked separately, exactly
   as `fleet_hierarchy` asks it.

Read-only, and deliberately so twice over: it opens no database that does not
already exist — a store's constructor CREATES it, and a recommendation that
manufactured a fleet database in a fresh clone would report "nobody has been
measured" as though that were a finding — and it writes nothing to one that does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alelyon.runtime.common import fleet_hierarchy as H

DISPATCH_SCHEMA_VERSION = 1

# ── provenance, borrowed rather than invented ────────────────────────────────
#: The same three levels `worktree_graph` and `blueprint` use, plus the absence
#: `docs/cne/CLAIMS.md` requires be reported aloud rather than defaulted.
OBSERVED = "OBSERVED"        # a standing exists and was read
DERIVED = "DERIVED"          # computed from the layer space, no model measured
UNMEASURED = "UNMEASURED"    # the deciding record could not be read

# ── outcomes ─────────────────────────────────────────────────────────────────
#: A measured standing named a model, and that model still fits the layer.
STANDING_HELD = "standing-held"
#: Nothing holds this coordinate. The layer's class is the whole answer.
NO_STANDING = "no-standing"
#: The brief matched no work kind, so there is no coordinate to look up.
NO_COORDINATE = "no-coordinate"
#: The ledger could not be read at all.
NO_LEDGER = "no-ledger"
#: Board-level subject matter. A model may prepare the decision, never make it.
OWNER_MATTER = "owner-matter"


@dataclass(frozen=True)
class Recommendation:
    """What to run this brief on, and how strongly that is known.

    `model` is **""** whenever no model can be named from evidence. That is not
    a failure to be filled in with a default — the caller still has `capability`,
    which is what the layer space actually establishes.
    """

    #: The layer the brief was placed at. Never `board`.
    layer: str
    #: Why it was placed there, quoting the rule that fired.
    layer_evidence: str
    #: The ledger coordinate's second half. "" when nothing matched.
    work_kind: str
    #: The measured model, or "" when none can be named from evidence.
    model: str
    #: The capability class the layer's work is worth. Always present.
    capability: str
    #: OBSERVED / DERIVED / UNMEASURED.
    provenance: str
    #: One of the outcome constants above.
    reason: str
    #: A sentence a reader can disagree with.
    detail: str
    #: Whether the *authorisation* is the owner's, independent of who does it.
    needs_owner: bool = False
    #: The standing this was read from, when there was one.
    standing: object | None = None
    #: Every model measured at this coordinate, best first — so a reader sees
    #: who was considered rather than only who won.
    candidates: tuple = field(default_factory=tuple)

    @property
    def named(self) -> bool:
        """Whether a model was named from evidence."""
        return bool(self.model)

    def __str__(self) -> str:
        who = self.model or f"<no model; {self.capability} class>"
        return f"{self.layer}/{self.work_kind or '?'} -> {who} ({self.provenance})"


def open_ledger(database=None):
    """`(ledger, problem)` — never creating the database to find out.

    `FleetLedger(...)` creates its file on construction. A recommendation that
    did that in a fresh clone would go on to report "no model has been measured
    here", which reads as evidence about models and is really evidence that this
    function made an empty file a moment earlier.
    """
    from alelyon.runtime.common import fleet_ledger as L
    try:
        path = Path(database) if database else L.default_database()
        return L.FleetLedger.open_existing(path), ""
    except L.LedgerAbsent:
        return None, (f"no fleet ledger at {path}; nothing has been measured "
                      f"in this checkout, and reading a recommendation will "
                      f"not create one to find out")
    except Exception as exc:                                  # noqa: BLE001
        return None, (f"the fleet ledger could not be read "
                      f"({type(exc).__name__}: {exc})")


def recommend(brief: str, *, work_kind: str | None = None,
              database=None, ledger=None) -> Recommendation:
    """Which model the record supports for this brief, and how strongly.

    `work_kind` overrides what the brief's own wording matched, for a caller that
    already knows its coordinate. Passing one that names a different layer than
    the prose does is the caller's business: the layer comes from the placement,
    the coordinate from the kind, and a disagreement between them is reported in
    `detail` rather than silently resolved.
    """
    layer, matched_kind, layer_evidence = H.placement(brief)
    kind = (work_kind or matched_kind or "").strip()
    owner = H.board_matter(brief)

    def _out(model, provenance, reason, detail, *, standing=None, cands=()):
        return Recommendation(
            layer=layer.key, layer_evidence=layer_evidence, work_kind=kind,
            model=model, capability=layer.capability, provenance=provenance,
            reason=reason, detail=detail, needs_owner=owner, standing=standing,
            candidates=tuple(cands))

    if owner:
        # The layer was already moved to `executive` for the *doing*. Naming a
        # model here would record a model as having made the owner's decision,
        # which is the exact failure `_model_layer` exists to prevent — and it
        # would be worse for coming from a standing, which reads as evidence.
        return _out("", DERIVED, OWNER_MATTER,
                    f"this is board-level subject matter, so no standing may "
                    f"name who decides it. A model may prepare the decision at "
                    f"{layer.key}; the owner makes it. {layer_evidence}")

    if not kind:
        return _out("", DERIVED, NO_COORDINATE,
                    f"the brief matched no work kind, so there is no "
                    f"(layer, work_kind) coordinate to look up. It is placed at "
                    f"{layer.key}, whose work is worth the {layer.capability} "
                    f"class; naming a model would need a coordinate this brief "
                    f"does not supply")

    if kind not in H.WORK_KINDS:
        # The vocabulary is closed, and `FleetLedger.propose` refuses the same
        # pair as NOT_A_COORDINATE. Querying anyway would return an empty result
        # indistinguishable from "measured, and nobody qualified".
        return _out("", DERIVED, NO_COORDINATE,
                    f"{kind!r} is not a work kind in layer space "
                    f"v{H.LAYER_SPACE_VERSION}; the vocabulary is closed, so "
                    f"{layer.key}/{kind} names no row and an empty answer from "
                    f"it would not mean nobody qualified. The layer's work is "
                    f"worth the {layer.capability} class")

    store, problem = (ledger, "") if ledger is not None else open_ledger(database)
    if store is None:
        return _out("", UNMEASURED, NO_LEDGER,
                    f"{problem}. The layer space still places this at "
                    f"{layer.key} and its work at the {layer.capability} class, "
                    f"which is derived from the layer and needs no ledger")

    try:
        standing = store.standing(layer.key, kind)
        candidates = store.candidates(layer.key, kind)
    except Exception as exc:                                  # noqa: BLE001
        return _out("", UNMEASURED, NO_LEDGER,
                    f"the fleet ledger could not be queried for "
                    f"{layer.key}/{kind} ({type(exc).__name__}: {exc})")

    if standing is None:
        return _out("", DERIVED, NO_STANDING,
                    f"no model holds {layer.key}/{kind}; the ledger was read and "
                    f"nothing has been measured there. The work is worth the "
                    f"{layer.capability} class, which is a property of the layer "
                    f"rather than a measurement of any model",
                    cands=candidates)

    # The capability floor is NOT re-checked here, and that is a decision rather
    # than an omission. `eligible()` enforces it at promotion, which is the only
    # place it can be asked honestly:
    #
    #   - Asking it afterwards with `measured_class` is CIRCULAR. That value is
    #     the capability of the highest layer where the model holds a standing,
    #     so the holder of a standing always meets its own layer's floor by
    #     construction, and the check can only ever answer yes.
    #   - Asking it while excluding this coordinate breaks the ON-RAMP, which
    #     deliberately promotes a model that does *not* meet the floor on the
    #     evidence of holding the rank below. Every legitimately promoted model
    #     would read as unfit — the "career rather than a caste" route, undone by
    #     a check that thought it was auditing it.
    #
    # So a standing is reported as what it is: the gate's own verdict, already
    # taken, and re-litigating it here would produce a second opinion with less
    # information than the first.
    return _out(standing.model, OBSERVED, STANDING_HELD,
                f"{standing.model} holds {layer.key}/{kind} at "
                f"{standing.score:.3f} over {standing.runs} run(s) "
                f"({standing.state}), promoted because {standing.reason}. This "
                f"is the best MEASURED candidate, which is not the best model — "
                f"one nobody ran has no evidence here at all",
                standing=standing, cands=candidates)


def limits() -> str:
    """What a recommendation from this module cannot tell you.

    Printed beside the answer rather than living only in a docstring, because a
    recommendation read without them is read as a ranking of models.
    """
    return "\n".join((
        "- The score behind a standing measures COMPLETION AND COST, not "
        "quality. An agent that settled quickly and cheaply on a wrong answer "
        "scores well and nothing here notices.",
        "- The cost term rewards spending FEWER OUTPUT TOKENS, so acting on "
        "this score alone routes work to whichever model gives up soonest. The "
        "landing signal is the only thing pushing back on that, and it is one "
        "signal at branch level: it cannot separate two agents who worked on "
        "the same branch.",
        "- A model nobody ran has no evidence at all, so a standing is the best "
        "MEASURED candidate and never the best model.",
        "- Five runs is the eligibility bar and it is a working hypothesis: a "
        "coin clears it one time in sixteen.",
        "- Nothing here launches an agent or selects a model. The commanding "
        "model must read this recommendation before dispatch and record why an "
        "override better satisfies total-cost, capability, risk, and evidence "
        "constraints.",
        "- Only Claude Code writes the transcripts the ledger reads, so "
        "cross-vendor comparison is ABSENT rather than empty.",
    ))


__all__ = [
    "DERIVED", "DISPATCH_SCHEMA_VERSION", "NO_COORDINATE", "NO_LEDGER",
    "NO_STANDING", "OBSERVED", "OWNER_MATTER", "STANDING_HELD",
    "UNMEASURED", "Recommendation", "limits", "open_ledger", "recommend",
]
