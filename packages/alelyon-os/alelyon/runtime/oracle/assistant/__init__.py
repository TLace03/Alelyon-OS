"""Lattice's answer path — domains, tools, grounding, persistence.

The path is: route → execute deterministic tools → narrate → check where every
figure came from. What it is *about* is not decided here: a `Domain` supplies
the tools, the words, the optional deterministic router and the privacy rule,
and the engine holds none of its own.

    from alelyon.runtime.oracle.assistant import ask, Context, GENERAL, MARKETS

    ans = ask("what is a convexity adjustment?", ctx=Context(), llm=my_llm)
    ans.prose        # the narration
    ans.facts        # the figures it was built from, each with source + as-of
    ans.grounded     # False when a figure in the prose is not in the facts
    ans.domain       # "general" — no book, no market intents

    ans = ask("what is my biggest risk?", ctx=Context(positions=...),
              llm=my_llm, domain=MARKETS)

`domain` defaults to `GENERAL`. That default is the point of the seam: the
markets desks used to be installed into one process-wide catalogue by
`install_tools()`, which meant every assistant in the application could reach a
book-risk tool whether or not it had a book. A product now asks for the subject
it is about, and gets a registry containing that and nothing else.
"""
from __future__ import annotations

from alelyon.runtime.oracle.assistant.domain import (
    GENERAL, MARKETS, Domain, Vocabulary, all_domains, get_domain,
    register_domain,
)
from alelyon.runtime.oracle.assistant.engine import (
    MODE_GROUNDED, MODE_OPEN, MODES, STAGE_CHECKING, STAGE_DESKS, STAGE_DONE,
    STAGE_ROUTING, STAGE_WRITING, STAGES, AnalystAnswer, ask,
)
from alelyon.runtime.oracle.assistant.grounding import GroundingReport, check
from alelyon.runtime.oracle.assistant.tools import (
    Context, Fact, Registry, Tool, ToolResult, all_tools, catalog_text,
)

__all__ = [
    "AnalystAnswer", "Context", "Domain", "Fact", "GENERAL", "GroundingReport",
    "MARKETS", "MODES", "MODE_GROUNDED", "MODE_OPEN", "Registry", "STAGES",
    "STAGE_CHECKING", "STAGE_DESKS", "STAGE_DONE", "STAGE_ROUTING",
    "STAGE_WRITING", "Tool", "ToolResult", "Vocabulary", "all_domains",
    "all_tools", "ask", "catalog_text", "check", "get_domain",
    "register_domain",
]
