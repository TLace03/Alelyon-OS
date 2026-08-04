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
    """
    out: List[str] = [NO_FIGURE]
    for f in facts or []:
        try:
            s = f.rendered()
        except Exception:  # noqa: BLE001
            continue
        if s and s != "n/a" and s not in out:
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
        "- A `figure` must be one of the exact strings shown above. Do not "
        "reformat, round, or convert them.\n"
        "- Write 2-4 sentences. Be direct; no preamble, no restating the "
        "question.\n"
        "- If the desk data does not answer the question, say so in one "
        "sentence with an empty figure. That is a complete answer."
    )


@dataclass(frozen=True)
class ConstrainedReply:
    ok: bool
    prose: str = ""
    figures: Tuple[str, ...] = ()
    error: str = ""

    @property
    def grounded_by_construction(self) -> bool:
        """True only for a reply that passed validation.

        Named for what it means: every figure came from the allowed set because
        no other value was accepted — not because a checker looked afterwards
        and found nothing wrong.
        """
        return self.ok


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

    prose = " ".join(p.strip() for p in parts if p.strip()).strip()
    if not prose:
        return ConstrainedReply(False, error="the reply produced no text")
    return ConstrainedReply(True, prose=prose, figures=tuple(used))
