"""The model, asked about itself — the general domain's own deterministic tool.

Every other pack here reads a subject: a book, a curve, an option chain. This
one reads the assistant. "How many parameters does the model I am talking to
have, and at what precision" is a question with an exact answer sitting in the
runtime's own inventory, and until now the only way to see it was to open the
Model Morphometry view and read it off the screen.

It exists because the general domain was otherwise nearly empty. Standalone
Lattice is sold as "an assistant that can ask a deterministic engine instead of
guessing", and after the domain split it held exactly one tool — a certified
calculator over market and FRED series. Asked how big it was, it answered from
the model's own weights, which is precisely the recall-instead-of-fetch that the
product exists to replace. A model reciting its own parameter count from
pre-training is quoting a number about *some* model, not necessarily this one,
and on a quantized local build it is reliably wrong about the storage.

What this establishes, and what it does not
-------------------------------------------
Every figure is **declared, not observed**. It comes from what the runtime says
about the model — its tensor inventory, or its architecture fields where it
publishes no inventory. No weight is read and no forward pass is run, so this
describes structure and storage and says nothing about what the model has
learned or how it behaves. The facts carry that sentence rather than implying
it, because a parameter census invites the other reading.

`UNMEASURED` is never rounded to zero. A mixture-of-experts model whose runtime
declares no routing, or routing that contradicts its tensor inventory, reports
its active parameter count as unavailable rather than as its total — a total
there would describe an idle expert bank as fully active.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from alelyon.runtime.oracle.assistant.tools import (
    Context, Fact, Param, Tool, ToolResult,
)

_TOOL = "model_anatomy"

#: The one sentence that has to travel with every figure here. A parameter
#: census read without it is a claim about capability; read with it, it is a
#: claim about storage, which is all it is.
_DECLARED = ("declared by the runtime, not observed — no weight was read and "
             "no forward pass was run")


def _resolve_model(ctx: Context, args: Dict[str, Any]) -> str:
    """The model asked about: the argument, the host's, or the selected one."""
    named = str(args.get("model", "") or "").strip()
    if named:
        return named
    from_host = str(ctx.extras.get("model", "") or "").strip()
    if from_host:
        return from_host
    from alelyon.runtime.oracle.assistant import local_model as LM
    return str(LM.selected_model() or "").strip()


def _anatomy(ctx: Context, args: Dict[str, Any]) -> ToolResult:
    from alelyon.runtime.oracle.assistant import local_model as LM
    from alelyon.runtime.vector.lattice import morphometry as MM

    model = _resolve_model(ctx, args)
    if not model:
        return ToolResult(_TOOL, args,
                          unavailable="no model is selected on this machine")

    payload = LM.show(model)
    if payload is None:
        # A stated reason, not an empty fact list. "The server is not running"
        # is a true and useful answer; a zero parameter count is not.
        return ToolResult(_TOOL, args,
                          unavailable=(f"the local model server did not "
                                       f"describe {model} — it may not be "
                                       f"running, or the model may not be "
                                       f"installed on this machine"))

    morph = MM.analyze(payload, model=model)
    if not morph.ok:
        # The engine refuses BY NAME rather than approximating — a
        # mixture-of-experts model with no tensor inventory is the live case.
        # Passing that refusal through is the whole point of a stated reason.
        return ToolResult(_TOOL, args,
                          unavailable=(morph.refusal
                                       or "the runtime did not describe this "
                                          "model well enough to measure"))

    counted = _measured_source(morph, MM)
    facts: List[Fact] = [
        Fact("model", morph.model or model, "", "", _DECLARED),
        Fact("measured from", counted, "", "",
             "a runtime that publishes a tensor list is counted cell by cell; "
             "one that does not is computed from its declared dimensions"),
    ]
    _add(facts, "architecture", morph.architecture)
    _add(facts, "family", morph.family)
    _add(facts, "storage precision", morph.quantization,
         "how the weights are stored on disk, not the precision they compute in")

    _add_count(facts, "declared parameters", morph.declared_parameters,
               "the runtime's own figure for the whole model")
    _add_count(facts, "counted parameters", morph.counted_parameters,
               _coverage_note(morph))
    _add_count(facts, "transformer blocks", morph.block_count)
    _add_count(facts, "context length (tokens)", morph.context_length)
    _add_count(facts, "embedding width", morph.embedding_length)

    gb = _nominal_gb(morph)
    if gb is not None:
        facts.append(Fact("nominal storage (GB)", gb, "", "",
                          "the weights at their declared precision; excludes "
                          "runtime overhead and the KV cache",
                          error_kind="exact"))
    bits = _bits_per_weight(morph)
    if bits is not None:
        facts.append(Fact("mean bits per weight", bits, "", "",
                          "nominal storage divided by the counted parameters",
                          error_kind="exact"))

    facts.extend(_routing_facts(morph))
    facts.extend(_gap_facts(morph))
    return ToolResult(_TOOL, args, facts=facts,
                      source=f"Model Morphometry (schema {morph.schema_version}) "
                             f"— declared structure, not learned behaviour")


# ── figure helpers ───────────────────────────────────────────────────────────
def _measured_source(morph, MM) -> str:
    return ("counted tensor inventory"
            if morph.source == MM.SOURCE_TENSOR_INVENTORY
            else "declared architecture fields")


def _add(facts: List[Fact], label: str, value, note: str = "") -> None:
    """A string fact, skipped when the runtime did not declare it.

    Skipped rather than emitted blank: a row reading `family:` with nothing
    after it is present, empty, and indistinguishable from a model whose family
    is genuinely unknown — the legibility defect the filings tool already
    shipped once.
    """
    text = str(value or "").strip()
    if text:
        facts.append(Fact(label, text, "", "", note or _DECLARED))


def _add_count(facts: List[Fact], label: str, value, note: str = "") -> None:
    """An integer the runtime declared. `exact` — it is a config value, not an
    estimate, so it carries no error bar and says so."""
    try:
        n = int(value or 0)
    except (TypeError, ValueError):
        return
    if n <= 0:
        return
    facts.append(Fact(label, float(n), "", "", note or _DECLARED,
                      error_kind="exact"))


def _coverage_note(morph) -> str:
    """Say what fraction of the declared model the count actually covered.

    A counted figure at 91% coverage is not the model's parameter count, and
    printing it beside the declared one without saying so invites the reader to
    treat the difference as a discrepancy rather than as a gap.
    """
    raw = morph.coverage
    if raw is None:
        # The engine returns None here when the runtime declared no total, and
        # is explicit that this is UNMEASURED rather than 1.0. Collapsing it
        # into the generic note would present a partial census as a complete
        # one, which is the whole failure this tool exists to avoid.
        return ("the runtime declared no total parameter count, so what "
                "fraction of the model this covers is UNMEASURED — it is not "
                "necessarily the whole")
    try:
        coverage = float(raw)
    except (TypeError, ValueError):
        return _DECLARED
    if coverage >= 0.9999:
        return "the inventory covers the runtime's declared parameter count"
    return (f"the inventory covers {coverage * 100:.1f}% of the declared "
            f"parameter count — the remainder is UNMEASURED, not zero")


def _nominal_gb(morph) -> Optional[float]:
    try:
        raw = float(morph.nominal_bytes or 0)
    except (TypeError, ValueError):
        return None
    return round(raw / 1e9, 2) if raw > 0 else None


def _bits_per_weight(morph) -> Optional[float]:
    try:
        params = float(morph.counted_parameters or 0)
        raw = float(morph.nominal_bytes or 0)
    except (TypeError, ValueError):
        return None
    if params <= 0 or raw <= 0:
        return None
    return round(raw * 8.0 / params, 2)


def _routing_facts(morph) -> List[Fact]:
    """What one token actually reaches, on a model that routes.

    Parameter share says where the weights are STORED. On a mixture-of-experts
    model that is a different question from what a token costs, and the second
    used to be readable off the first. A routed model whose runtime declares no
    routing therefore reports the active figure as unavailable rather than as
    its total — a total there describes an idle expert bank as fully active.
    Contradictory routing declarations carry the engine's more specific reason.
    """
    if not morph.is_mixture_of_experts:
        return []
    facts: List[Fact] = []
    _add_count(facts, "experts per block", morph.expert_count)
    _add_count(facts, "experts used per token", morph.expert_used_count)
    active = None
    try:
        active = float(morph.active_parameters or 0) or None
    except (TypeError, ValueError):
        active = None
    if active and morph.expert_used_count:
        facts.append(Fact("active parameters per token", active, "", "",
                          "what one token reaches, not what is stored",
                          error_kind="exact"))
    else:
        reason = str(morph.active_path_gap or "").strip()
        if not reason:
            reason = (
                "UNMEASURED — this model routes, but its runtime does not "
                "declare how many experts a token uses. Reporting the total "
                "here would describe an idle expert bank as fully active."
            )
        facts.append(Fact(
            "active parameters per token", None, "", "", reason))
    return facts


def _gap_facts(morph) -> List[Fact]:
    """The named gaps, verbatim. What the runtime did not declare is part of
    the answer, not an absence to be tidied away."""
    gaps = [str(g).strip() for g in (morph.gaps or ()) if str(g).strip()]
    if not gaps:
        return []
    return [Fact("named gaps", "; ".join(gaps[:6]), "", "",
                 "fields the runtime did not declare — each is UNMEASURED "
                 "rather than zero")]


def install(registry) -> None:
    registry.register(Tool(
        _TOOL,
        "how big the model you are talking to actually is: its parameter "
        "count, transformer blocks, context length, storage precision, mean "
        "bits per weight and — on a routed model — what one token reaches. "
        "Use this for any question about THIS assistant's own size, precision "
        "or architecture, rather than answering from memory",
        params=(Param("model", "str", False,
                      "model name, or omit for the one currently selected"),),
        fn=_anatomy, surface="Model Morphometry"))
