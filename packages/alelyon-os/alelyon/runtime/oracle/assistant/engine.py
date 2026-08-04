"""Route â†’ execute â†’ narrate â†’ check. Lattice's answer path.

The old panel put the question and a seven-line context header straight into a
local model and printed whatever came back. Asked "where is SPY's gamma flip?"
it produced a level â€” confidently, in the right format, from nothing. The
Gamma desk was running two tabs away.

This replaces that with four steps:

  1. **Route.** The model sees the tool catalogue and the question, and returns
     JSON naming the tools to run. That is its only structured output; it is
     never asked for a figure at this stage.
  2. **Execute.** `tools.run_plan` calls the desks deterministically. Same code
     paths the GUI draws from, so the chat cannot disagree with the screen.
  3. **Narrate.** The model writes prose from the fact sheet â€” the *rendered*
     figures, the same strings the panel prints.
  4. **Check.** `grounding.check` verifies every number in the prose came from
     the facts. One repair attempt, then whatever survives is flagged in the UI
     rather than quietly shipped.

Step 4 is what makes steps 1â€“3 worth anything. Handing a model facts makes it
*more likely* to quote them; nothing about it makes fabrication impossible, and
the failure looks identical to success. The check is cheap, deterministic, and
runs on every answer.

Two modes, and the difference is deliberate
-------------------------------------------
`MODE_GROUNDED` is everything above: constrained decoding, prose forbidden from
containing a digit, a repair pass, and a refusal rather than an answer when the
desks hold nothing. It exists because inside Financial Markets a figure beside a
live book is a number somebody may trade on, and there "I cannot say" is a
better answer than a confident guess.

`MODE_OPEN` keeps steps 1, 2 and 4 and drops step 3's cage. The desks are still
queried, the facts still arrive with their own as-of stamps, and the grounding
check still runs â€” but its report is **advisory**: it says which figures matched
desk data and which did not, and does not rewrite or suppress the answer. The
model may reason, explain, do arithmetic, write code and use what it knows. This
is the mode the standalone Lattice product runs in, where the desks are a source
of verified figures rather than a fence around what may be said.

What does NOT change between modes: tools are read-only, the private-context
rule that keeps book state off a cloud provider, and the provenance record. The
mode decides how much freedom the *narration* has, never what the assistant is
allowed to reach or where the context is allowed to go.

Watching it happen
------------------
`ask` takes two optional sinks. `on_stage` reports which of the four steps is
running; `on_text` receives the narration in fragments as the model writes it.
Both are advisory â€” the returned answer is byte-identical with or without them â€”
and `on_text` is honoured in `MODE_OPEN` only. Grounded prose can still be
repaired or replaced after step 4, and streaming a figure that step 4 then
withdraws is worse beside a live book than a blank panel for a minute.

Subject-free by construction
----------------------------
Nothing in this module knows what it is answering questions *about*. Which tools
exist, what they are called in prose, who the assistant is, which questions can
be routed without a model, and what makes a context private are all supplied by
a `Domain` (`domain.py`). This file used to introduce itself to the router as
"a financial analyst workstation" and address its narration to "a portfolio
manager"; a general assistant running on it inherited both. The engine now asks
the domain for those words and never holds one of its own.

`domain` defaults to `GENERAL` â€” an assistant with the certified calculator and
nothing else. A caller that wants the markets desks must say so, which is the
point: the market surface is opt-in rather than ambient.

Qt-free and offline-testable: `llm` is any `Callable[[str], str]`. The streaming
seam is discovered, not required â€” a plain callable simply does not have one.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from alelyon.runtime.oracle.assistant import constrain, grounding, tools as T
from alelyon.runtime.oracle.assistant.domain import (
    GENERAL, MODE_AUTO, MODE_GROUNDED, MODE_OPEN, MODES, RESOLVED_MODES,
    Domain, Vocabulary,
)

MAX_TOOLS = 4

# The mode vocabulary lives in `domain`, because which contract an answer needs
# is a property of the SUBJECT rather than of the answer path. Re-exported here
# so every existing `engine.MODE_OPEN` import is unchanged.
__all__ = ["MODE_AUTO", "MODE_GROUNDED", "MODE_OPEN", "MODES",
           "RESOLVED_MODES", "AnalystAnswer", "ask"]

# â”€â”€ progress stages â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# A closed vocabulary, because the panel maps these to captions and a typo would
# silently produce a blank status line. The detail string beside each one is
# free text for the user; the key is not.
STAGE_ROUTING = "routing"        # deciding which desks to ask
STAGE_DESKS = "desks"            # the desks are running
STAGE_WRITING = "writing"        # the model is composing the answer
STAGE_CHECKING = "checking"      # grounding/provenance check
STAGE_DONE = "done"
STAGES = (STAGE_ROUTING, STAGE_DESKS, STAGE_WRITING, STAGE_CHECKING, STAGE_DONE)


# â”€â”€ step 1: routing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _vocab(domain: Optional[Domain]) -> Vocabulary:
    return (domain or GENERAL).vocabulary


def route_prompt(question: str, history: Sequence = (), *,
                 domain: Optional[Domain] = None) -> str:
    domain = domain or GENERAL
    vocab = domain.vocabulary
    convo = ""
    if history:
        lines = []
        for h in history[-6:]:
            role = getattr(h, "role", "user")
            text = " ".join(str(getattr(h, "text", "")).split())[:220]
            lines.append(f"{role}: {text}")
        convo = "Recent conversation (for pronouns and follow-ups):\n" + "\n".join(lines) + "\n\n"
    return (
        f"You are the router for {vocab.router_role}. Decide which internal "
        f"{vocab.tool_noun} tools should be queried to answer the question.\n\n"
        "OUTPUT ONLY JSON. No prose, no markdown fences, no explanation.\n"
        '{"tools": [{"name": "<tool>", "args": {...}}], "note": "<one short line>"}\n\n'
        f"Tools available:\n{domain.catalog_text()}\n\n"
        f"Rules:\n"
        f"- Pick at most {MAX_TOOLS} tools. Fewer is better.\n"
        "- Use the EXACT tool names above. Do not invent tools or arguments.\n"
        + "".join(f"- {hint}\n" for hint in vocab.router_hints) +
        "- If the question is conceptual and needs no live data, return "
        '{"tools": [], "note": "conceptual"}.\n'
        "- If the question needs data no tool provides, still return an empty "
        'list and say what is missing in the note.\n\n'
        f"{convo}Question: {question}\n"
    )


def _json_block(text: str) -> Optional[dict]:
    """First JSON object in a model reply, fence- and <think>-tolerant."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return None
    return obj if isinstance(obj, dict) else None


def parse_plan(raw: str, *, domain: Optional[Domain] = None,
               registry: Optional[T.Registry] = None
               ) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    """(calls, note). Unknown tool names are DROPPED and named in the note â€”
    silently ignoring them would leave the reader wondering why their question
    went unanswered.

    A name is "unknown" relative to the DOMAIN'S registry, never a global one.
    That is what stops a model that has heard of `book_risk` from reaching it
    inside an assistant whose domain never installed it.
    """
    if registry is None:
        registry = (domain or GENERAL).toolset()
    obj = _json_block(raw)
    if obj is None:
        return [], "the router did not return a usable plan"
    note = str(obj.get("note", "") or "").strip()
    calls: List[Tuple[str, Dict[str, Any]]] = []
    unknown: List[str] = []
    for item in (obj.get("tools") or [])[:MAX_TOOLS * 2]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        args = item.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        if registry.get(name) is None:
            if name:
                unknown.append(name)
            continue
        calls.append((name, args))
    if unknown:
        extra = "no such tool: " + ", ".join(sorted(set(unknown)))
        note = f"{note} ({extra})" if note else extra
    return calls[:MAX_TOOLS], note


# â”€â”€ step 3: narration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _answer_rules(vocab: Vocabulary) -> str:
    return (
        "Rules:\n"
        f"- Use ONLY the figures in {vocab.the_data()} above. Every number you "
        f"write must appear there.\n"
        "- Do NOT estimate, interpolate, annualise, or convert figures into "
        f"other units. If a figure is not there, say which {vocab.tool_noun} "
        f"would have it.\n"
        "- Quote figures exactly as rendered above.\n"
        f"- If {vocab.the_data()} does not answer the question, say so plainly "
        f"in one sentence. That is a complete and acceptable answer.\n"
        f"- Be concise: 2-5 sentences. You are talking to {vocab.audience}.\n"
        "- No preamble, no restating the question, no disclaimers about being "
        "an AI."
    )


def answer_prompt(question: str, results: Sequence[T.ToolResult],
                  history: Sequence = (), *,
                  domain: Optional[Domain] = None) -> str:
    vocab = _vocab(domain)
    noun = vocab.data_noun
    if results:
        sheet = "\n".join(r.as_prompt_block() for r in results)
        data = f"The {noun} retrieved for this question:\n{sheet}\n\n"
    else:
        data = (f"No {noun} was retrieved for this question.\n"
                f"You therefore have NO figures. Do not state any number, "
                f"price, level, or percentage. Answer conceptually, or name "
                f"the {vocab.tool_noun} that would hold the answer.\n\n")
    convo = ""
    if history:
        lines = [f"{getattr(h, 'role', 'user')}: "
                 f"{' '.join(str(getattr(h, 'text', '')).split())[:220]}"
                 for h in history[-4:]]
        convo = "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"
    return (f"{vocab.persona}\n\n"
            f"{convo}{data}Question: {question}\n\n{_answer_rules(vocab)}\n")


def _open_rules(vocab: Vocabulary) -> str:
    plural = vocab.tool_noun_plural
    return (
        f"How to use {vocab.the_data()}:\n"
        f"- The figures above were computed deterministically by this "
        f"machine's own {plural}, from captured data, moments ago. Where they "
        f"answer the question they are better than your recollection â€” quote "
        f"them exactly as rendered and keep their as-of stamps.\n"
        f"- Never attribute a figure to a {vocab.tool_noun} that did not "
        f"return it. If you give a number of your own â€” an estimate, a worked "
        f"calculation, something you know â€” say so plainly in the sentence "
        f"that carries it.\n"
        f"- If the {plural} returned nothing relevant, answer anyway from what "
        f"you know, and say which {vocab.tool_noun} would hold the measured "
        f"version.\n\n"
        "Otherwise answer normally: reason it through, explain, work through "
        "the arithmetic, write code, ask a clarifying question if the request "
        "is genuinely ambiguous. Be direct and concrete; skip the preamble and "
        "the disclaimers about being an AI."
    )


def open_answer_prompt(question: str, results: Sequence[T.ToolResult],
                       history: Sequence = (), *, note: str = "",
                       persona: str = "",
                       domain: Optional[Domain] = None) -> str:
    """The narration prompt for `MODE_OPEN`.

    The fact block is framed as EVIDENCE rather than as the only permissible
    vocabulary. That is the whole difference from `answer_prompt`: the same
    facts, the same as-of stamps, the same provenance â€” but the model is being
    asked to think with them rather than to recite them.

    `persona` overrides the domain's, for a host that wants to introduce the
    assistant in its own words. Absent one, the domain's persona is used â€” the
    engine holds none of its own.
    """
    vocab = _vocab(domain)
    if results:
        sheet = "\n".join(r.as_prompt_block() for r in results)
        data = (f"The {vocab.data_noun} retrieved for this question "
                f"(deterministic, computed just now):\n{sheet}\n\n")
    else:
        data = f"No {vocab.tool_noun} was queried for this question.\n\n"
    convo = ""
    if history:
        lines = [f"{getattr(h, 'role', 'user')}: "
                 f"{' '.join(str(getattr(h, 'text', '')).split())[:400]}"
                 for h in history[-8:]]
        convo = "Conversation so far:\n" + "\n".join(lines) + "\n\n"
    who = persona or vocab.persona
    aside = f"Router note: {note}\n\n" if note else ""
    return (f"{who}\n\n{convo}{data}{aside}Question: {question}\n\n"
            f"{_open_rules(vocab)}\n")


def repair_prompt(question: str, prose: str, bad: Sequence[grounding.Mention],
                  results: Sequence[T.ToolResult], *,
                  domain: Optional[Domain] = None) -> str:
    vocab = _vocab(domain)
    listed = ", ".join(sorted({m.text for m in bad}))
    sheet = "\n".join(r.as_prompt_block() for r in results) or "(none)"
    return (
        f"Your answer contained figures that are not in {vocab.the_data()}.\n\n"
        f"Your answer:\n{prose}\n\n"
        f"Figures with no source: {listed}\n\n"
        f"The only figures you may use:\n{sheet}\n\n"
        "Rewrite the answer using only those figures. Where you had an "
        "unsupported number, either drop it or say the figure is not available. "
        "Do not add new numbers. Output only the rewritten answer.\n\n"
        f"Question: {question}\n"
    )


# â”€â”€ result â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@dataclass(frozen=True)
class AnalystAnswer:
    question: str
    prose: str
    results: List[T.ToolResult] = field(default_factory=list)
    report: Optional[grounding.GroundingReport] = None
    provider: str = ""
    plan_note: str = ""
    repaired: bool = False
    error: str = ""
    # True when the figures could not have been anything else: the decoder was
    # constrained to an enum of the desks' own rendered strings and prose was
    # forbidden from containing digits. Strictly stronger than `grounded`,
    # which is a check applied afterwards.
    constrained: bool = False
    #: Routed to its desk without asking a model. The facts are identical either
    #: way â€” this only records that no generation stood between the question and
    #: the desk, which is why the answer was immediate.
    deterministic: bool = False
    #: The prose is a rendering of the facts, not model output. Set when no model
    #: answered â€” the desks still did, and their figures are the answer.
    facts_only: bool = False
    #: Which contract produced this answer. See MODE_GROUNDED / MODE_OPEN.
    mode: str = MODE_GROUNDED
    #: Which domain answered â€” the key, not the object, so a saved transcript
    #: keeps it without pickling a callable. Worth recording: the same question
    #: gets a different answer in a domain with a book behind it, and a
    #: transcript that omits which one ran is missing the first thing you would
    #: want to know when it looks wrong.
    domain: str = ""
    #: True when the grounding report is INFORMATION, not a gate. In open mode
    #: the model may legitimately write a figure of its own, so an unsupported
    #: mention means "this number is the model's, not a desk's" â€” which is worth
    #: showing and is not a defect.
    advisory: bool = False
    #: The generation stopped before the model finished â€” a dropped connection
    #: or a size ceiling. The text is real as far as it goes, and a half answer
    #: presented as a whole one is the model's first thought published as its
    #: conclusion, so this must reach the screen.
    truncated: bool = False
    #: The user stopped it. Not a failure, and it must not be shown as one.
    cancelled: bool = False
    #: Fragments were delivered to a sink as they arrived. Recorded so a caller
    #: can tell a streamed answer from one that merely finished quickly.
    streamed: bool = False

    @property
    def facts(self) -> List[T.Fact]:
        return T.facts_of(self.results)

    @property
    def tools_run(self) -> List[str]:
        return [r.tool for r in self.results]

    @property
    def grounded(self) -> bool:
        return self.report.grounded if self.report is not None else True

    @property
    def unsupported_text(self) -> List[str]:
        if self.report is None:
            return []
        return [m.text for m in self.report.unsupported]

    @property
    def consulted(self) -> str:
        """The provenance line. Names only tools that actually returned data â€”
        a failed call did not inform the answer and must not appear to have."""
        good = [r.tool for r in self.results if r.ok]
        return ", ".join(dict.fromkeys(good))


# â”€â”€ the engine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def facts_of(results):
    return T.facts_of(results)


def _mark_private(llm: Callable[[str], str]) -> None:
    """Tell a policy-aware provider that subsequent prompts are private.

    Plain injected callables remain supported.  The desktop Auto chain exposes
    this seam; explicit Local/Cloud chains decide their own policy.
    """
    marker = getattr(llm, "mark_private", None)
    if callable(marker):
        marker()


def _try_constrained(question, results, history, llm):
    """One constrained attempt. Any failure returns a non-ok reply and the
    caller falls back â€” a backend that ignored the grammar must never produce
    an answer that gets badged as guaranteed."""
    facts = T.facts_of(results)
    try:
        raw = llm(constrain.schema_prompt(question, results, history),
                  schema=constrain.answer_schema(facts))
    except TypeError:
        return constrain.ConstrainedReply(False, error="provider has no grammar seam")
    except Exception as exc:  # noqa: BLE001
        return constrain.ConstrainedReply(False, error=f"{type(exc).__name__}: {exc}")
    if not (raw or "").strip():
        return constrain.ConstrainedReply(False, error="no constrained reply")
    return constrain.validate(raw, facts)


def _stage(on_stage, key: str, detail: str = "") -> None:
    """Report progress, never at the cost of the answer.

    A panel's status label is the least important thing in this function. A
    caller whose sink raises still gets their answer.
    """
    if on_stage is None:
        return
    try:
        on_stage(key, detail)
    except Exception:  # noqa: BLE001
        pass


def _streamer(llm) -> Optional[Callable[..., Any]]:
    """The provider chain's incremental seam, if this `llm` has one.

    `llm` is documented as any `Callable[[str], str]`, and most of the test
    suite passes a plain lambda. Streaming is therefore discovered rather than
    required: a callable without the seam takes the ordinary path and nothing
    about the answer changes.
    """
    fn = getattr(llm, "stream", None)
    return fn if callable(fn) else None


def ask(question: str, *, ctx: T.Context, llm: Callable[[str], str],
        history: Sequence = (), provider_name: str = "",
        max_tools: int = MAX_TOOLS, repair: bool = True,
        constrain_output: bool = True,
        deterministic_routing: bool = True,
        mode: str = MODE_GROUNDED,
        persona: str = "",
        domain: Optional[Domain] = None,
        on_text: Optional[Callable[[str], None]] = None,
        on_stage: Optional[Callable[[str, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> AnalystAnswer:
    """Answer `question`, optionally reporting progress as it goes.

    `domain` decides what this assistant is about: which tools it can reach,
    what they are called in the prompts, whether any question can be routed
    without a model, and what makes this context private. It defaults to
    `GENERAL`, so a caller that wants the markets desks has to name them.

    `on_text` receives the narration in fragments as the model produces it, and
    `on_stage` receives `(stage, detail)` from `STAGES`. Both are optional and
    both are advisory: the returned `AnalystAnswer` is identical either way.

    **`on_text` is honoured in open mode only, and that is a safety property
    rather than an omission.** The grounded contract may repair or replace prose
    after a grounding check; streaming it would put an unverified figure on
    screen beside a live book and retract it a second later, which is worse than
    a pause. Grounded callers still get stage events â€” those describe the work,
    not the answer.

    `cancel` is polled between fragments. A cancelled answer returns the text
    that had already arrived, flagged, rather than nothing.
    """
    question = " ".join(str(question or "").split())
    domain = domain or GENERAL
    registry = domain.toolset()
    vocab = domain.vocabulary
    if not question:
        return AnalystAnswer(question="", prose="", error="empty question",
                             mode=mode, domain=domain.key)
    # An unknown mode must not silently fall through to the permissive one.
    # Grounded is the safe default and the one a live book depends on.
    if mode not in MODES:
        mode = MODE_GROUNDED
    # The caller's originals, kept because MODE_AUTO can TIGHTEN after routing
    # and the grounded contract has to be restored intact when it does.
    want_constrain, want_repair = constrain_output, repair
    auto = (mode == MODE_AUTO)
    if auto:
        # First pass, from the question alone. The deterministic router has not
        # run yet, so this is the domain reading the sentence â€” which is the
        # half a caller cannot supply. It is re-decided once calls are known.
        mode = domain.contract_for(question, ())

    def _apply(chosen: str) -> bool:
        """Set the contract-derived flags. Returns whether it is open."""
        nonlocal constrain_output, repair
        if chosen == MODE_OPEN:
            # Constrained decoding IS the grounded contract: an enum of tool
            # strings and a digit-free prose pattern. There is no partial
            # version of it, so open mode turns it off rather than loosening it.
            constrain_output = False
            repair = False
            return True
        constrain_output, repair = want_constrain, want_repair
        return False

    is_open = _apply(mode)

    # Context the DOMAIN calls private is private before the first model call.
    # Auto mode may still use cloud for a public conceptual question, but never
    # for a question carrying that state. This is a PRIVACY boundary and holds
    # in both modes â€” open mode frees what the model may say, not where the
    # context may travel. The engine does not know what makes a context private
    # in any particular subject; it asks.
    if domain.private_context(ctx, history):
        _mark_private(llm)

    # DETERMINISTIC FIRST. In a domain with a router, most questions name their
    # tool unmistakably, and routing them through a generation costs a round
    # trip the answer does not need â€” on a 30B model that round trip is most of
    # the latency that made this look broken. The model still owns anything
    # open-ended: `plan()` returns None there, as it always does in a domain
    # with no router at all, and the LLM path below runs exactly as it did.
    _stage(on_stage, STAGE_ROUTING,
           f"deciding which {vocab.tool_noun_plural} to ask")
    det = domain.plan(question) if deterministic_routing else None
    router_note = ""
    if det is not None and not det.calls:
        # Understood, and answerable without any tool call: a horizon the tools
        # do not cover, or a subject that needs disambiguating.
        if not is_open:
            # Grounded mode: the refusal IS the answer, because substituting a
            # figure from another horizon would put a real number under a
            # question it does not answer.
            _stage(on_stage, STAGE_DONE, "")
            return AnalystAnswer(question=question,
                                 prose=det.refusal or det.ask_user,
                                 provider="deterministic router", deterministic=True,
                                 plan_note="answered without a model",
                                 mode=mode, domain=domain.key)
        # Open mode: the router's finding is real and worth telling the model,
        # but it is not a reason to stop. Pass it through as context and let the
        # assistant answer the question it was actually asked.
        router_note = det.refusal or det.ask_user
        det = None

    if det is not None:
        calls, note = det.calls, det.note
    else:
        try:
            raw_plan = llm(route_prompt(question, history, domain=domain))
        except Exception as exc:  # noqa: BLE001
            # A model that is down, slow, or missing is an ordinary condition
            # here â€” it must produce a stated reason, not an exception into the
            # panel, which is indistinguishable from the app hanging.
            _stage(on_stage, STAGE_DONE, "")
            hint = f" {vocab.offline_hint}" if vocab.offline_hint else ""
            return AnalystAnswer(
                question=question, prose=router_note, provider=provider_name,
                mode=mode, domain=domain.key,
                error=f"the local model could not be reached "
                      f"({type(exc).__name__}).{hint}")
        if not (raw_plan or "").strip():
            _stage(on_stage, STAGE_DONE, "")
            return AnalystAnswer(question=question, prose=router_note,
                                 provider=provider_name, mode=mode,
                                 domain=domain.key,
                                 error="no model answered the routing step")
        calls, note = parse_plan(raw_plan, registry=registry)

    # SECOND PASS, and it may only TIGHTEN. The first pass read the question;
    # this one has the plan, so it catches a market question the sentence test
    # missed and the model routed anyway â€” "what is my biggest risk" phrased in
    # a way no pattern matches, answered by `book_risk`.
    #
    # One direction only. A plan that turned out to name no tool must NOT buy
    # back open mode, because "the desks had nothing" is exactly when a model
    # supplies the level from memory. Grounded is sticky once chosen.
    if auto and is_open:
        tightened = domain.contract_for(question, calls)
        if tightened == MODE_GROUNDED:
            mode = MODE_GROUNDED
            is_open = _apply(mode)
            # Streaming is open-mode only, and this is the path where that
            # matters most: fragments may already have been promised to a sink.
            # `stream` is chosen below from `is_open`, which is now False, so
            # the answer is delivered whole â€” the correct behaviour beside a
            # live book, and the reason `on_text` is advisory rather than a
            # contract.
            on_text = None

    # Crossing the tool boundary makes the remainder private.  Mark it before
    # execution because a tool may itself use the injected model seam; this
    # also ensures narration and repair cannot spill returned facts to cloud.
    if calls:
        _mark_private(llm)
        _stage(on_stage, STAGE_DESKS,
               "consulting " + ", ".join(dict.fromkeys(name for name, _ in calls)))
    results = (T.run_plan(calls, ctx, limit=max_tools, registry=registry)
               if calls else [])

    # Prefer the CONSTRAINED path: a figure the desks did not return has no
    # representation in the grammar, so fabrication is prevented rather than
    # detected. Falls back rather than failing â€” a small model that cannot
    # phrase its answer inside the template should still get to answer, with
    # the after-the-fact check doing the work instead.
    constrained = False
    prose = ""
    # Constrain REGARDLESS of whether the desks returned anything. The gate used
    # to be `and facts_of(results)`, which skipped the grammar in exactly the
    # case it matters most: with no facts the allowed-figure enum is a single
    # NO_FIGURE, so a number becomes UNREPRESENTABLE. Leaving that case
    # unconstrained is what let "what can you tell me that the desks don't?"
    # come back with an RSI, a MACD histogram and a $121.01 fair value, all
    # invented, under a badge that correctly said no desk data was consulted.
    if constrain_output:
        reply = _try_constrained(question, results, history, llm)
        if reply.ok:
            prose, constrained = reply.prose, True
    truncated = False
    cancelled = False
    streamed = False
    if not prose:
        narration = (open_answer_prompt(question, results, history,
                                        note=router_note, persona=persona,
                                        domain=domain)
                     if is_open else
                     answer_prompt(question, results, history, domain=domain))
        _stage(on_stage, STAGE_WRITING, "writing the answer")
        # Streaming is open-mode only. See `ask`'s docstring: grounded prose can
        # still be rewritten after the check, and a figure shown then withdrawn
        # beside a live book is worse than a pause.
        stream = _streamer(llm) if (is_open and on_text is not None) else None
        if stream is not None:
            result = _stream_narration(stream, narration, on_text, cancel)
            prose = _strip_think((result.text or "").strip())
            streamed = True
            cancelled = bool(result.cancelled)
            # Truncation is only meaningful for text that arrived. An empty
            # failed stream is an ordinary "no answer" and falls through to the
            # facts below, exactly as a dead blocking provider does.
            truncated = bool(prose) and not result.complete and not cancelled
            if not prose and result.error:
                note = f"{note} Â· {result.error}".strip(" Â·")
        else:
            try:
                prose = _strip_think((llm(narration) or "").strip())
            except Exception as exc:  # noqa: BLE001 â€” a dead model must not eat the facts
                note = (f"{note} Â· narration failed: {type(exc).__name__}").strip(" Â·")
                prose = ""
    facts_only = False
    if not prose and results:
        # THE FACTS ARE THE ANSWER. A model that is missing, slow, or broken
        # costs the sentence around the numbers â€” it does not cost the numbers,
        # which the desks already returned. Rendering them is a complete reply,
        # and it cannot fabricate, because nothing generated it.
        prose = _facts_only_prose(results)
        facts_only = True
    if not prose and router_note:
        # Open mode with a dead model and no tool data: the router's own finding
        # is still true and is better than an empty box.
        prose, facts_only = router_note, True
    if not prose:
        _stage(on_stage, STAGE_DONE, "")
        return AnalystAnswer(question=question, prose="", results=results,
                             provider=provider_name, plan_note=note, mode=mode,
                             domain=domain.key,
                             cancelled=cancelled, streamed=streamed,
                             error=("stopped before the model had written "
                                    "anything" if cancelled else
                                    "the model returned an empty answer"))

    facts = T.facts_of(results)
    _stage(on_stage, STAGE_CHECKING, "checking where each figure came from")
    # The check runs in BOTH modes; what changes is what its result licenses.
    # Grounded mode treats an unsupported figure as a defect to repair or flag.
    # Open mode treats it as provenance: this number did not come from a desk.
    report = grounding.check(prose, facts, question=question)
    repaired = False
    # The check still runs on a constrained answer. If it ever fails there, the
    # constraint leaked â€” a bug in this module, not a model fabricating â€” and
    # the louder that shows up the better.
    if repair and not report.grounded and not constrained:
        second = _strip_think(
            (llm(repair_prompt(question, prose, report.unsupported, results,
                               domain=domain)) or "").strip())
        if second:
            second_report = grounding.check(second, facts, question=question)
            # Only accept the rewrite if it is actually better; a repair that
            # invents fresh numbers must not overwrite the original.
            if len(second_report.unsupported) < len(report.unsupported):
                prose, report, repaired = second, second_report, True

    answer_provider = "desk facts only" if facts_only else provider_name
    _stage(on_stage, STAGE_DONE, "")
    return AnalystAnswer(question=question, prose=prose, results=results,
                         report=report, provider=answer_provider, plan_note=note,
                         repaired=repaired, constrained=constrained,
                         deterministic=det is not None, facts_only=facts_only,
                         mode=mode, domain=domain.key, advisory=is_open,
                         truncated=truncated, cancelled=cancelled,
                         streamed=streamed)


def _stream_narration(stream, prompt: str, on_text, cancel):
    """One streamed narration, with the scratchpad filtered on the way past.

    The sink sees what the transcript will keep, not what the socket carried:
    a reasoning model's `<think>` block is removed here exactly as
    `_strip_think` removes it from the saved text. Showing it live and deleting
    it on save would make the panel look like it lost the answer.
    """
    from alelyon.runtime.oracle.answer.streaming import StreamResult, ThinkFilter

    scratchpad = ThinkFilter()

    def _sink(fragment: str) -> None:
        visible = scratchpad.feed(fragment)
        if visible:
            on_text(visible)

    try:
        result = stream(prompt, _sink, cancel)
    except Exception as exc:  # noqa: BLE001 â€” a dead model must not eat the facts
        return StreamResult("", False, error=f"narration failed: {type(exc).__name__}")
    tail = scratchpad.flush()
    if tail:
        try:
            on_text(tail)
        except Exception:  # noqa: BLE001
            pass
    return result


def _facts_only_prose(results: Sequence[T.ToolResult]) -> str:
    """Render the tools' own figures, with no model in the loop.

    Deliberately plain and deliberately not a sentence: this is the fact sheet,
    not an imitation of narration. Each figure carries its own as-of and its own
    uncertainty, so what is missing versus a narrated answer is the connective
    prose â€” not information.
    """
    lines = ["The tools answered; no model was available to write the summary, "
             "so here are the figures themselves."]
    for r in results:
        if r.unavailable:
            lines.append(f"\n{r.tool}: {r.unavailable}")
            continue
        if r.error:
            lines.append(f"\n{r.tool}: failed â€” {r.error}")
            continue
        if not r.facts:
            continue
        lines.append(f"\n{r.tool}" + (f" Â· {r.source}" if r.source else ""))
        for f in r.facts[:14]:
            bar = f.rendered_error()
            asof = f" (as of {f.as_of})" if f.as_of else ""
            lines.append(f"  {f.label}: {f.rendered()}"
                         + (f" {bar}" if bar else "") + asof)
    return "\n".join(lines)


def _strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    # A reasoning model that never closed the tag would otherwise leak its
    # scratchpad into the transcript.
    text = re.sub(r"<think>.*$", "", text, flags=re.S)
    return text.strip()
