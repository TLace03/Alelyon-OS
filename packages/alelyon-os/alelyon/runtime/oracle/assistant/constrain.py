"""Make fabrication unrepresentable, rather than detectable after the fact.

`grounding.py` catches a fabricated figure *after* the model has written it.
That is a real check and it earns its place, but it is a check: the model can
still emit the number, and the system's job is then to notice. The stronger
property — the one the owner actually asked for — is that a number the desks
did not return **cannot be produced at all**.

That property does not come from weights. No configuration of a network's
parameters makes fabrication impossible; weights move probability mass, and a
mass of 10⁻⁹ on a wrong token is still a token the sampler can draw. It comes
from the **decoder**. Constrain which tokens are legal at each position and the
wrong answer stops being unlikely and starts being unreachable.

So the answer is generated as structure, not prose:

    {"sentences": [{"before": "NVDA carries ",
                    "figure": "34.20%",
                    "after":  " of the book's risk."}, …]}

with two constraints compiled into the grammar:

  * `figure` is an **enum of the exact rendered strings the desks returned**.
    There is no other legal value, so a quoted figure is a desk figure by
    construction.
  * `before` and `after` match `^[^0-9]*$` — prose **cannot contain a digit**.
    Every number in the answer therefore comes through a `figure` slot.

**Two guarantees, and they are not the same one.** When the backend compiles the
schema into a grammar (llama.cpp/Ollama do), the constraint is structural and a
fabricated figure is literally unrepresentable. When a backend merely *suggests*
the schema, `validate()` still rejects any reply with a digit in prose or an
unknown figure. So the honest claim is: **structurally impossible where the
grammar is enforced, detected and refused everywhere else** — never "the model
was trained not to".

The cost is real and worth stating: constrained decoding narrows what a small
model can say, and a model that cannot phrase its answer inside the template
will fail to produce one. `ask()` therefore treats a constrained attempt as
preferred, not mandatory, and falls back to the checked path rather than
returning nothing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Prose segments may not contain a digit. Kept as one place so the grammar and
# the validator can never disagree about what "no numbers" means.
_PROSE_PATTERN = r"^[^0-9]*$"
_PROSE_RE = re.compile(_PROSE_PATTERN)
_MAX_PROSE = 240
_MAX_SENTENCES = 8

# The empty string is a legal `figure`: most sentences carry no number, and
# without it the model would be forced to quote one in every clause.
NO_FIGURE = ""


def allowed_figures(facts: Sequence) -> List[str]:
    """The exact strings a figure slot may contain.

    Deliberately the RENDERED form — the same string the panel prints and the
    same string the model was shown. Offering the raw float here would let the
    answer and the fact table disagree about precision on the same number.

    Order is preserved and is the order the desks returned, because the enum is
    also what the model reads: a set would make the prompt's figure list vary
    between runs on the same data, and a reader comparing two answers could not
    tell a reordering from a change in the facts.

    The membership test is a SET and not a scan of `out`. Measured 2026-08-05 on
    this module: `s not in out` made construction O(n·u) in the number of facts
    and the number of DISTINCT figures among them — 8,000 distinct figures took
    174 ms, and 8,000 facts carrying only 20 distinct figures took 1 ms, which
    is the same law seen from its cheap side. Nothing caps the fact count
    (`tools.facts_of` concatenates every desk result), so the cost is set by
    whichever desk returns the widest table. See `docs/papers/
    V-the-decoder-not-the-weights.md` §8.4, which recorded this as UNMEASURED.
    """
    out: List[str] = [NO_FIGURE]
    seen = {NO_FIGURE}
    for f in facts or []:
        try:
            s = f.rendered()
        except Exception:  # noqa: BLE001
            continue
        if s and s != "n/a" and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def answer_schema(facts: Sequence) -> Dict[str, Any]:
    """A JSON Schema in which a fabricated figure has no representation."""
    return {
        "type": "object",
        "properties": {
            "sentences": {
                "type": "array",
                "minItems": 1,
                "maxItems": _MAX_SENTENCES,
                "items": {
                    "type": "object",
                    "properties": {
                        "before": {"type": "string", "pattern": _PROSE_PATTERN,
                                   "maxLength": _MAX_PROSE},
                        "figure": {"type": "string",
                                   "enum": allowed_figures(facts)},
                        "after": {"type": "string", "pattern": _PROSE_PATTERN,
                                  "maxLength": _MAX_PROSE},
                    },
                    "required": ["before", "figure", "after"],
                },
            },
        },
        "required": ["sentences"],
    }


def schema_prompt(question: str, results: Sequence, history: Sequence = ()) -> str:
    """The instruction that accompanies the schema.

    The schema is the enforcement; this is what makes the output *good* rather
    than merely legal. A model told only "emit this shape" writes stilted
    fragments around the figures.
    """
    sheet = "\n".join(r.as_prompt_block() for r in results) or "(no desk data)"
    convo = ""
    if history:
        lines = [f"{getattr(h, 'role', 'user')}: "
                 f"{' '.join(str(getattr(h, 'text', '')).split())[:180]}"
                 for h in history[-4:]]
        convo = "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"
    return (
        "You are a quantitative analyst answering a portfolio manager.\n\n"
        f"{convo}Desk data retrieved for this question:\n{sheet}\n\n"
        f"Question: {question}\n\n"
        "Answer as a list of sentences. Each sentence is split into three "
        "parts:\n"
        "  before — the words leading up to a figure\n"
        "  figure — ONE figure, copied EXACTLY from the desk data above, or "
        "\"\" for a sentence with no figure\n"
        "  after  — the rest of the sentence\n\n"
        "Rules:\n"
        "- `before` and `after` must contain NO digits. Every number belongs in "
        "a `figure`.\n"
        # Asked for explicitly, because `grounded_by_construction` now refuses a
        # reply that spells one out. The grammar cannot forbid this — a digit-free
        # pattern admits "thirty-four point two percent" — so the only honest
        # options were to ask and then check, or to check something the model was
        # never told. Asking first also lowers how often the check has to fire.
        "- Do not spell numbers out in words either. \"thirty-four percent\" is "
        "a number and belongs in a `figure`, not in the prose.\n"
        "- A `figure` must be one of the exact strings shown above. Do not "
        "reformat, round, or convert them.\n"
        "- Write 2-4 sentences. Be direct; no preamble, no restating the "
        "question.\n"
        "- If the desk data does not answer the question, say so in one "
        "sentence with an empty figure. That is a complete answer."
    )


#: Number words, and the markers that turn one into a QUANTITY. The pairing is
#: what keeps the detector honest: "one of the desks" and "a second factor" are
#: ordinary prose, while "one hundred and forty basis points" is a figure that
#: never went through the enum. Requiring a marker is the difference.
_NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_SCALE_WORDS = (
    "percent|per ?cent|basis ?points?|bps|point|hundred|thousand|million|"
    "billion|trillion|times|fold|dollars?|cents?|euros?|pounds?"
)
#: A number word, optionally chained through hyphens/"and", then a marker within
#: two words. Case-insensitive; the model writes sentence case.
_SPELLED_RE = re.compile(
    rf"\b(?:{_NUMBER_WORDS})\b(?:[- ](?:and|{_NUMBER_WORDS})\b)*"
    rf"(?:\W+\w+){{0,2}}?\W+(?:{_SCALE_WORDS})\b",
    re.IGNORECASE)


def spelled_quantities(text: str) -> Tuple[str, ...]:
    """Spelled-out quantities in prose — the figures the enum never saw.

    `docs/papers/V-the-decoder-not-the-weights.md` §8.3 admitted that
    "Thirty-four percent" satisfies the digit-free prose pattern, and rested on
    the judgment *that a spelled-out numeral cannot carry the precision that
    makes a fabricated figure consequential in this setting*. It named that as
    the first item it would put to a reviewer.

    **Measured 2026-08-05, the judgment is false.** English carries arbitrary
    precision in words: "thirty-four point two percent" is three significant
    figures and "two hundred and fifteen million dollars" is exact. Every one of
    those passes `^[^0-9]*$` and reaches the reader inside a reply badged as
    grounded by construction.

    What this does NOT catch is stated rather than implied: a bare ratio
    ("nine out of ten names") carries a quantity and no marker, so it is
    admitted. Flagging it would mean flagging "one of the desks", and a
    detector that fires on ordinary prose gets switched off. The residual is
    real and is recorded in the paper rather than papered over.
    """
    return tuple(m.group(0) for m in _SPELLED_RE.finditer(text or ""))


@dataclass(frozen=True)
class ConstrainedReply:
    ok: bool
    prose: str = ""
    figures: Tuple[str, ...] = ()
    error: str = ""
    #: Spelled-out quantities found in the prose slots. Empty for a reply whose
    #: only numbers came through the enum.
    spelled: Tuple[str, ...] = ()

    @property
    def grounded_by_construction(self) -> bool:
        """True only for a reply whose every figure came through the enum.

        Named for what it means: every figure came from the allowed set because
        no other value was accepted — not because a checker looked afterwards
        and found nothing wrong.

        **A spelled-out quantity falsifies exactly that sentence**, so it is
        excluded here rather than noted somewhere a badge-renderer would not
        read. `ok` still reports grammar conformance, which is a different and
        narrower fact; this is the claim-bearing one, and until 2026-08-05 it
        said "every figure came from the allowed set" about replies that could
        contain "thirty-four point two percent" in the prose around the figure.
        """
        return self.ok and not self.spelled


def _strip_fences(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def validate(raw: str, facts: Sequence) -> ConstrainedReply:
    """Parse a constrained reply and enforce the constraints again.

    Re-checking is not redundant. A backend that ignores the schema, or honours
    `enum` but not `pattern`, would otherwise hand back an unconstrained answer
    wearing the shape of a constrained one — which is worse than no constraint,
    because the UI would badge it as guaranteed.
    """
    allowed = set(allowed_figures(facts))
    try:
        obj = json.loads(_strip_fences(raw))
    except Exception as exc:  # noqa: BLE001
        return ConstrainedReply(False, error=f"reply was not valid JSON: {exc}")
    if not isinstance(obj, dict) or not isinstance(obj.get("sentences"), list):
        return ConstrainedReply(False, error="reply had no `sentences` list")

    parts: List[str] = []
    prose_parts: List[str] = []
    used: List[str] = []
    for i, s in enumerate(obj["sentences"][:_MAX_SENTENCES]):
        if not isinstance(s, dict):
            return ConstrainedReply(False, error=f"sentence {i} was not an object")
        before = str(s.get("before", "") or "")
        figure = str(s.get("figure", "") or "")
        after = str(s.get("after", "") or "")
        if not _PROSE_RE.match(before) or not _PROSE_RE.match(after):
            return ConstrainedReply(
                False, error=f"sentence {i} put a digit in prose — the backend "
                             f"did not enforce the grammar")
        if figure and figure not in allowed:
            return ConstrainedReply(
                False, error=f"sentence {i} used a figure the desks did not "
                             f"return: {figure!r}")
        if figure:
            used.append(figure)
        parts.append(before + figure + after)
        prose_parts.extend((before, after))

    prose = " ".join(p.strip() for p in parts if p.strip()).strip()
    if not prose:
        return ConstrainedReply(False, error="the reply produced no text")
    # Scanned over the PROSE slots only. Running it across the assembled
    # sentence would read an enum figure and the words beside it as one span —
    # "12.4" followed by "percent" is a figure that DID come through the enum,
    # and flagging it would make the detector fire on the construction working.
    spelled = tuple(q for part in prose_parts for q in spelled_quantities(part))
    return ConstrainedReply(True, prose=prose, figures=tuple(used),
                            spelled=spelled)
