"""The certified calculation tool — the Verified Answer engine, as a tool.

Verified Answer and AI Analyst were the same product in two views: type a
question, get a number. They differed only in *how* the number was obtained —
one had the model author a safe DSL program that executed deterministically
with a calibrated error bar; the other queried the desks. Neither was a
superset, so the trader had to know which view could answer which question,
which is the app's job, not theirs.

They are one view now. The DSL path becomes a tool the router can pick, so a
question like *"what is the correlation of SPY and QQQ daily returns?"* takes
the certified path and comes back with its confidence interval, while *"where
is my biggest risk?"* takes the desk path — and the analyst asks neither.

Two properties of the underlying engine survive the move intact, because they
are the reason it exists:

**The model is structurally forbidden from emitting the number.** Its only
output channel is DSL source, which executes over the data layer. If the
question is outside that vocabulary the engine returns `CANNOT_EXPRESS` and
this tool reports the refusal — it does NOT fall back to a desk guess. A
refusal that quietly became an approximation would be worse than the overclaim
it was built to prevent.

**The interval is finite-sample, not decorative.** Fisher-z for a correlation,
standard error for a mean, and an explicit "no calibrated interval" for a point
indicator — never a fabricated tight band.
"""
from __future__ import annotations

from typing import Any, Dict, List

from alelyon.runtime.oracle.assistant.tools import (
    Context, Fact, Param, Tool, ToolResult,
)


def _calc(ctx: Context, args: Dict[str, Any]) -> ToolResult:
    question = str(args.get("question", "")).strip()
    if not question:
        return ToolResult("verified_calc", args, error="no question given")

    llm = ctx.extras.get("llm")
    if llm is None:
        return ToolResult("verified_calc", args,
                          unavailable="no language model is available to author "
                                      "the calculation")
    ds = ctx.data_service
    if ds is None:
        from alelyon.runtime.atlas.data.service import data_service
        ds = data_service()

    from alelyon.runtime.oracle.answer.engine import answer as verified_answer
    va = verified_answer(question, data_service=ds, llm_fn=llm)

    if va.refused or not va.ok:
        # Report the refusal as the answer. The engine declined because the
        # question needs data its safe vocabulary cannot express; substituting
        # an approximation here would defeat the whole design.
        return ToolResult("verified_calc", args,
                          unavailable=(va.error or "the calculation was refused"))

    facts: List[Fact] = []
    src = ", ".join(va.sources) if va.sources else ""
    cert = va.certificate
    note = ""
    if cert is not None and cert.lo is not None and cert.hi is not None:
        note = (f"{int(round(cert.conf * 100))}% interval "
                f"{cert.lo:.4g} to {cert.hi:.4g} — {cert.driver}")
    elif cert is not None:
        note = f"no calibrated interval — {cert.driver}"
    facts.append(Fact(question[:70], va.value, "", va.as_of or "", note))

    if cert is not None and cert.lo is not None:
        facts.append(Fact("lower bound", cert.lo, "", va.as_of or "", cert.note))
        facts.append(Fact("upper bound", cert.hi, "", va.as_of or ""))
    if src:
        facts.append(Fact("computed from", src, "", va.as_of or "",
                          "the series the program actually read"))
    # The program itself is the provenance: it is exactly what was executed,
    # and it is short enough to read.
    facts.append(Fact("program", va.dsl.replace("\n", " ; "), "", "",
                      "executed deterministically; the model wrote this, not "
                      "the number"))
    return ToolResult("verified_calc", args, facts=facts,
                      source="Certified Answer engine (Alelyon DSL, "
                             + ("repaired once" if va.repaired else "first try") + ")")


def install(registry) -> None:
    registry.register(Tool(
        "verified_calc",
        "compute a statistic over a price or FRED series with a CALIBRATED "
        "confidence interval — correlations, betas, z-scores, realised "
        "volatility, moving averages, RSI. Use this when the question asks for "
        "a figure that has to be COMPUTED, rather than one some view already "
        "shows",
        params=(Param("question", "str", True,
                      "the question, in plain English, exactly as asked"),),
        fn=_calc, surface="Verified Answer"))
