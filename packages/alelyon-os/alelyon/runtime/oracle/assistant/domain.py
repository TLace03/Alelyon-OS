"""What an assistant is *about* — the seam that stops one product owning the core.

The answer path in this package used to be the Financial Markets AI Analyst with
a mode flag on it. Its router prompt introduced itself as "the router for a
financial analyst workstation", its narration prompt addressed "a portfolio
manager", its one deterministic router matched gamma and 2s10s, and
`install_tools()` put seven market desks into a single process-wide catalogue
that every assistant in the application then saw. Lattice ran on top of that and
inherited all of it, which is why a standalone general assistant offered to tell
you where your book risk was.

A **Domain** is the answer. It is the small, declarative bundle of everything
that is *about a subject* rather than about answering:

- the **vocabulary** — the words the prompts and the screen use for this
  subject's tools, and who the assistant is when it speaks;
- the **packs** — the modules that install this subject's tools, each into a
  registry of the domain's own;
- the **router** — an optional deterministic question→tool matcher for the
  questions whose shape is unmistakable in this subject;
- the **privacy predicate** — what makes a context private *here*, so the
  engine can hold the cloud boundary without knowing what a position is.

Everything else — routing, execution, narration, streaming, the grounding
check, provenance — is the host, and the host is now subject-free.

Declared as data, resolved late
-------------------------------
`packs` and `router` are dotted **strings**, exactly as `platform.catalog`
keeps a product's entry point as a string. Importing this module must not drag
in the markets data layer, the symbology index, or a desk engine; a domain is
a row in a table until something asks it to run. `toolset()` is what pays that
cost, once, and caches it.

What a domain is NOT
--------------------
It is not a permission boundary against a hostile caller. Nothing stops code
that already holds the markets registry from running a markets tool. What it
is: a structural guarantee that an assistant configured for one subject cannot
*discover* another's tools, because they were never put in its catalogue and
there is no path from one registry to another. That closes the failure this
exists for — a general assistant silently answering from a book.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from alelyon.runtime.oracle.assistant.tools import Context, Registry

log = logging.getLogger(__name__)


# ── the contracts ────────────────────────────────────────────────────────────
#
# These live here rather than in `engine` because a contract is a property of
# the subject, not of the answer path: what a figure costs to get wrong is a
# fact about the domain. `engine` re-exports them, so every existing
# `engine.MODE_OPEN` / `assistant.MODE_OPEN` import is unchanged.

#: Answer inside the tools' figures or refuse. Constrained decoding, prose
#: forbidden from containing a digit, a repair pass, and a refusal rather than
#: an answer when the tools hold nothing.
MODE_GROUNDED = "grounded"
#: Ordinary assistant behaviour with the tools available as a fact source. The
#: grounding check still runs; its report is provenance rather than a verdict.
MODE_OPEN = "open"
#: Let the DOMAIN choose, per answer, from the routed plan. A caller that says
#: this is saying "I do not know in advance whether this question is dangerous
#: here" — which is the honest position for an assistant that has to be both a
#: general one and a careful one. Resolves to one of the two above before any
#: narration happens; it is never the mode an answer is recorded under.
MODE_AUTO = "auto"
MODES = (MODE_GROUNDED, MODE_OPEN, MODE_AUTO)
#: The modes an answer can actually be produced under.
RESOLVED_MODES = (MODE_GROUNDED, MODE_OPEN)


# ── the words ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Vocabulary:
    """What this domain calls its own parts, everywhere a user or a model reads.

    Every field here was, until this refactor, a market noun hardcoded into a
    prompt or a panel. They are gathered rather than scattered so that adding a
    domain is filling in a form, and so that a reviewer can see the whole
    surface a new subject has to name.
    """

    #: Who the assistant is, in the first line of the open-mode narration prompt.
    persona: str = (
        "You are Lattice, Alelyon's assistant. You answer the way a "
        "knowledgeable colleague would.")
    #: What the ROUTER is told it is routing for. One clause, no trailing stop.
    router_role: str = "a general-purpose assistant"
    #: What one tool is called, in prose the user reads. "desk", "tool", "control".
    tool_noun: str = "tool"
    #: The plural of the above.
    tool_noun_plural: str = "tools"
    #: How the fact sheet is referred to: "the desk data", "the tool data".
    data_noun: str = "tool data"
    #: Who the GROUNDED narration is addressed to. Grounded mode only.
    audience: str = "a careful reader"
    #: One line naming what still works when no model can be reached. Domain
    #: knowledge: it names the questions this domain routes deterministically.
    offline_hint: str = ""
    #: One line for an empty conversation — what this assistant is good for.
    invitation: str = "Ask anything."
    #: Extra routing rules the model needs to pick this domain's tools well —
    #: how its identifiers are spelled, what to resolve before calling. Each
    #: becomes one bullet in the router prompt. Domain knowledge that had no
    #: home before and so lived hardcoded in the engine.
    router_hints: Tuple[str, ...] = ()

    def the_data(self) -> str:
        return f"the {self.data_noun}"


GENERAL_VOCABULARY = Vocabulary()

MARKETS_VOCABULARY = Vocabulary(
    persona=("You are Lattice, Alelyon's assistant, working inside a financial "
             "markets workstation. You can query this machine's quantitative "
             "desks for verified figures, and you answer everything else the "
             "way a knowledgeable colleague would."),
    router_role="a financial analyst workstation",
    tool_noun="desk",
    tool_noun_plural="desks",
    data_noun="desk data",
    audience="a portfolio manager",
    offline_hint=("Questions that name a desk directly — a quote, book risk, "
                  "technicals, news for a ticker — are answered without it."),
    invitation=("Ask anything. Questions that name a desk — a quote, book "
                "risk, the curve, gamma — are answered with that desk's own "
                "figures."),
    router_hints=("Resolve company names to tickers in the args (Apple -> AAPL).",),
)


# ── the domain ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Domain:
    """One subject an assistant can be about.

    Immutable and cheap. The expensive parts — importing packs, building the
    registry — happen on first `toolset()` and are cached per domain instance.
    """

    key: str
    label: str
    vocabulary: Vocabulary = field(default_factory=Vocabulary)
    #: Dotted module paths, each exposing `install(registry) -> None`.
    packs: Tuple[str, ...] = ()
    #: Dotted `module:attr` for `plan(question) -> Optional[Plan]`, or "" for a
    #: domain with no deterministic router. Empty is not a deficiency: it means
    #: every question goes to the model, which is correct where no question
    #: shape is unmistakable.
    router: str = ""
    #: Dotted `module:attr` for `(ctx, history) -> bool`. Decides whether this
    #: question's context may leave the machine. A domain that names none gets
    #: `context_or_history_is_private`, which is the STRICT default on purpose:
    #: the failure mode of an over-strict rule is a cloud model not being used,
    #: and the failure mode of a lax one is a book position leaving the box.
    privacy: str = ""
    #: Dotted `module:attr` for `(question, calls) -> MODE_GROUNDED|MODE_OPEN`.
    #: Consulted only under `MODE_AUTO`. A domain that names none answers
    #: grounded, for the same reason the privacy default is the strict one: an
    #: over-strict contract costs an answer its arithmetic, and a lax one puts
    #: an invented level on screen beside a live book.
    contract: str = ""

    # ── the toolset ─────────────────────────────────────────────────────────
    def toolset(self) -> Registry:
        """This domain's tools, built once.

        A pack that fails to import loses its own tools and is named by
        `failed_packs()`. It does NOT take the domain down: the rest of the
        catalogue still answers, and the loss has to be visible rather than
        looking like the assistant simply choosing not to use a tool.
        """
        built = _TOOLSETS.get(self.key)
        if built is not None:
            return built
        with _LOCK:
            built = _TOOLSETS.get(self.key)
            if built is not None:
                return built
            registry = Registry()
            failed: List[str] = []
            for path in self.packs:
                try:
                    import_module(path).install(registry)
                except Exception:  # noqa: BLE001 — one broken pack, not all
                    log.warning("assistant domain %r: pack %r failed to install",
                                self.key, path, exc_info=True)
                    failed.append(path.rsplit(".", 1)[-1])
            _TOOLSETS[self.key] = registry
            _FAILED[self.key] = failed
            return registry

    def failed_packs(self) -> List[str]:
        """Packs that could not be installed. Empty before `toolset()` has run,
        because nothing has been attempted yet."""
        self.toolset()
        return list(_FAILED.get(self.key, ()))

    def catalog_text(self) -> str:
        return self.toolset().catalog_text()

    # ── the deterministic router ────────────────────────────────────────────
    def plan(self, question: str) -> Optional[Any]:
        """A deterministic plan for `question`, or None.

        None means "ask the model", and it is what a domain with no router
        always returns. A router that raises is treated as no match: a broken
        matcher must cost the shortcut, never the answer.
        """
        fn = _resolve(self.router)
        if fn is None:
            return None
        try:
            return fn(question)
        except Exception:  # noqa: BLE001
            log.warning("assistant domain %r: router failed on a question",
                        self.key, exc_info=True)
            return None

    # ── the contract ────────────────────────────────────────────────────────
    def contract_for(self, question: str,
                     calls: Sequence = ()) -> str:
        """Which contract this answer runs under. `MODE_GROUNDED` on any doubt.

        Called only when the caller asked for `MODE_AUTO`. Three ways to end up
        grounded, and all of them are the same decision: nothing established
        that loosening the cage here is safe.

        * the domain declared no contract;
        * the predicate raised;
        * the predicate returned something that is not a resolved mode.

        A `MODE_AUTO` returned by a predicate is treated as no answer rather
        than recursing — a contract that cannot decide has not decided.
        """
        fn = _resolve(self.contract)
        if fn is None:
            return MODE_GROUNDED
        try:
            chosen = fn(question, tuple(calls or ()))
        except Exception:  # noqa: BLE001
            log.warning("assistant domain %r: contract predicate failed; "
                        "answering grounded", self.key, exc_info=True)
            return MODE_GROUNDED
        return chosen if chosen in RESOLVED_MODES else MODE_GROUNDED

    # ── the privacy boundary ────────────────────────────────────────────────
    def private_context(self, ctx: Context, history: Sequence) -> bool:
        """Does this question already carry state that must not leave the box?

        A predicate that raises is read as **private**. That is the safe
        direction and the only defensible one: a broken privacy check must not
        be the reason a book position reaches a cloud provider.
        """
        fn = _resolve(self.privacy) or context_or_history_is_private
        try:
            return bool(fn(ctx, history))
        except Exception:  # noqa: BLE001
            log.warning("assistant domain %r: privacy predicate failed; "
                        "treating the context as private", self.key,
                        exc_info=True)
            return True


# ── resolution + registry ────────────────────────────────────────────────────
_LOCK = threading.Lock()
_TOOLSETS: Dict[str, Registry] = {}
_FAILED: Dict[str, List[str]] = {}
_RESOLVED: Dict[str, Optional[Callable]] = {}


def _resolve(path: str) -> Optional[Callable]:
    """`"pkg.mod:attr"` → the callable, or None. Cached, including the misses."""
    path = str(path or "").strip()
    if not path:
        return None
    if path in _RESOLVED:
        return _RESOLVED[path]
    fn: Optional[Callable] = None
    try:
        module_path, _, attr = path.partition(":")
        candidate = getattr(import_module(module_path), attr, None)
        fn = candidate if callable(candidate) else None
        if fn is None:
            log.warning("assistant domain: %r did not resolve to a callable",
                        path)
    except Exception:  # noqa: BLE001
        log.warning("assistant domain: %r could not be imported", path,
                    exc_info=True)
    _RESOLVED[path] = fn
    return fn


_DESKS = "alelyon.runtime.oracle.assistant.desks."

#: The standalone Lattice product. Two subject-free deterministic capabilities:
#: the certified calculator, which has the model author a restricted program and
#: executes it with an error bar, and the model's own declared anatomy, which is
#: the one measurement an assistant can make about itself. No book, no
#: positions, no market intents — a question about gamma reaches the model,
#: exactly as it would in any other assistant.
GENERAL = Domain(
    key="general",
    label="General",
    vocabulary=GENERAL_VOCABULARY,
    packs=(_DESKS + "calc", _DESKS + "model_self"),
    router="",
    privacy="alelyon.runtime.oracle.assistant.domain:context_or_history_is_private",
    contract="alelyon.runtime.oracle.assistant.domain:always_open",
)

#: Financial Markets. Every desk, the market intent router, and the book/account
#: privacy rule. Reached by the markets product; never by a general assistant.
MARKETS = Domain(
    key="markets",
    label="Financial Markets",
    vocabulary=MARKETS_VOCABULARY,
    packs=tuple(_DESKS + name for name in
                ("book", "market", "derivatives", "macro", "company",
                 "engine_state", "calc")),
    router="alelyon.runtime.oracle.assistant.route:plan",
    privacy="alelyon.runtime.oracle.assistant.domain:book_context_is_private",
    contract="alelyon.runtime.oracle.assistant.domain:markets_contract",
)


# ── the shipped privacy predicates ───────────────────────────────────────────
def tool_history_is_private(ctx: Context, history: Sequence) -> bool:
    """Private once any tool in this conversation has returned something.

    The floor every domain builds on. Whatever a tool computed on this machine
    is this machine's; a follow-up question that carries it must not be the one
    that goes to a cloud provider.
    """
    return any(
        bool(getattr(turn, "facts", None) or getattr(turn, "tools", None))
        for turn in history or ()
    )


def context_or_history_is_private(ctx: Context, history: Sequence) -> bool:
    """The strict default: a populated context is private before the first call.

    This checks the markets-shaped fields even for domains that never fill
    them, and that asymmetry is deliberate. The fields exist on `Context`, so
    a host CAN put positions in one and hand it to a general assistant. If the
    general rule only looked at history, that host's book would reach a cloud
    provider on the first question — a privacy failure caused by a domain
    forgetting to opt in. A rule that costs an unused check is the cheaper
    error.
    """
    if ctx.positions or ctx.connected:
        return True
    if any(value not in (None, "") for value in (ctx.account or {}).values()):
        return True
    return tool_history_is_private(ctx, history)


# ── the shipped contract predicates ──────────────────────────────────────────
def always_open(question: str, calls: Sequence = ()) -> str:
    """A domain where no tool produces a number anyone trades on.

    `GENERAL`'s. Its two tools are the certified calculator, whose figures
    carry their own interval, and `model_anatomy`, whose figures are exact
    declarations about the model itself. Neither is a price. Open mode's
    bargain — the model may add figures of its own and provenance says which
    are which — is the right one when a wrong number costs an argument rather
    than a position.
    """
    return MODE_OPEN


def markets_contract(question: str, calls: Sequence = ()) -> str:
    """Grounded whenever the question engages the market at all.

    The rule this shipped with is NOT the one the embedding design proposed.
    That one read "grounded unless the plan is provably free of book, position,
    engine and order tools" — and measured against the live registry it left 21
    of 27 tools answering open, including `gamma`, `quote`, `vol_surface` and
    `analyst_report`. A gamma flip level is the founding example of this
    package's own docstring: asked for one, a model produced a level,
    confidently, from nothing. An allow-list keyed on *book* tools mistakes
    "whose money is it" for "is this a number somebody trades on".

    So the test is engagement, not tool class:

    1. **Any tool ran → grounded.** Every figure the markets desks return is a
       level, and a narration free to add its own beside them is exactly the
       failure the cage prevents.
    2. **No tool, but the question engages the market → grounded.** This is the
       dangerous case: a market question the router could not serve. Answering
       it open means the model supplies the level from memory.
    3. **Otherwise → open.** Arithmetic, code, prose, conceptual questions. The
       drawer is an assistant for these, which is the point of having a
       contract at all rather than one mode per product.

    Rule 2 makes this strictly stronger than a plan-only test, and it is the
    half a caller cannot supply: only the domain knows what its subject looks
    like in a sentence.
    """
    if calls:
        return MODE_GROUNDED
    try:
        from alelyon.runtime.oracle.assistant.route import engages_market
    except Exception:  # noqa: BLE001 - no router reachable is not a licence
        return MODE_GROUNDED
    return MODE_GROUNDED if engages_market(question) else MODE_OPEN


#: The markets rule is currently the strict default. Kept as a named alias so
#: the markets domain declares its own rule rather than inheriting one by
#: omission — if the default ever loosens, the book must not loosen with it.
book_context_is_private = context_or_history_is_private


# ── the domain table ─────────────────────────────────────────────────────────
_DOMAINS: Dict[str, Domain] = {}


def register_domain(domain: Domain) -> Domain:
    """Add a domain. A duplicate key is refused rather than overwriting: two
    domains answering to one name is how a product ends up with the other's
    tools, which is the whole failure this module exists to prevent."""
    key = str(domain.key or "").strip()
    if not key:
        raise ValueError("a domain must have a key")
    existing = _DOMAINS.get(key)
    if existing is not None and existing is not domain:
        raise ValueError(
            f"domain {key!r} is already registered as {existing.label!r}. "
            f"Pick another key rather than replacing it.")
    _DOMAINS[key] = domain
    return domain


def get_domain(key: str) -> Optional[Domain]:
    return _DOMAINS.get(str(key or "").strip())


def all_domains() -> Tuple[Domain, ...]:
    return tuple(_DOMAINS[k] for k in sorted(_DOMAINS))


register_domain(GENERAL)
register_domain(MARKETS)


def _reset_for_tests() -> None:
    """Drop every cached toolset and resolution. Test hook only."""
    with _LOCK:
        _TOOLSETS.clear()
        _FAILED.clear()
        _RESOLVED.clear()
