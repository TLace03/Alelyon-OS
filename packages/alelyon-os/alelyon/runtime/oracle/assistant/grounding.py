"""Does every figure in the answer trace back to a fact the desks produced?

The AI Analyst's one unforgivable failure is quoting a number it made up. A
language model asked "where is SPY's gamma flip?" will produce a confident level
whether or not it was ever shown one, and the level will look exactly like a
real one. The tool layer (`tools.py`) exists so the model is *given* the figures;
this module checks that the prose it wrote back only uses them.

**What this proves, and what it does not.** A GROUNDED answer is one where every
numeric token in the prose matches a figure the desks actually returned. That is
a check on *fabrication*, not on *reasoning*: the model can still draw a stupid
conclusion from correct numbers, and this will happily call it grounded. The
badge must be worded accordingly. Over-reading it would repeat the exact mistake
the Pipeline view's overclaims were caught making.

**Rounding is the model's, not ours.** If it writes "2.3%" for a fact of
2.34871, that is correct reporting, not fabrication. So a fact matches a mention
when the fact ROUNDED TO THE MENTION'S OWN PRECISION equals it. Comparing with a
fixed epsilon would flag every sensibly-rounded figure in the answer.

**What is deliberately not flagged**: years and dates, numbers the user typed in
the question (echoing the question is not inventing), and numbers that appear
verbatim inside a fact's own label or note ("20-day realised vol", "2s10s",
"25-delta skew" — the parameter is part of the name of the thing).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

# A numeric token, with the decorations a model writes around one:
#   1,234.5   $12.4bn   -3.2%   +45 bp   2.1x   0.87
#
# The trailing guard must reject a PREFIX of a longer number ("663.7" out of
# "663.705") without rejecting a figure that ends a sentence ("closed at
# 663.70."). A blanket `(?![\w.])` does the second by accident and silently
# matched nothing at all in ordinary prose.
_NUM_RE = re.compile(
    r"""(?<![\w.])            # not mid-identifier
        ([-+−]?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+−]?\$?\d+(?:\.\d+)?)
        \s*(%|bps?|bp|k|mm|m|bn|b|tn|x)?   # optional scale / unit suffix
        (?!\d)(?![A-Za-z_])(?!\.\d)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SCALE = {"k": 1e3, "m": 1e6, "mm": 1e6, "b": 1e9, "bn": 1e9, "tn": 1e12}

# Tokens that are never a market figure.
_YEAR_LO, _YEAR_HI = 1900, 2100


@dataclass(frozen=True)
class Mention:
    """One numeric token found in the prose."""
    text: str            # exactly as written, e.g. "$12.4bn"
    value: float         # normalised to the fact's units where a suffix implies it
    decimals: int        # how precisely the model wrote it
    start: int
    end: int
    unit: str = ""       # "%", "bp", "x", "" …
    scaled: bool = False  # written with a k/m/bn/tn suffix, so deliberately coarse


@dataclass(frozen=True)
class GroundingReport:
    mentions: List[Mention]
    unsupported: List[Mention]

    @property
    def total(self) -> int:
        return len(self.mentions)

    @property
    def grounded(self) -> bool:
        """An answer with no figures at all is grounded — it made no numeric
        claim. That is the honest reading, not a loophole: 'the curve is
        steepening' is a claim about direction, and this module has no opinion
        on it."""
        return not self.unsupported

    @property
    def summary(self) -> str:
        if not self.mentions:
            return "no figures quoted"
        if not self.unsupported:
            n = len(self.mentions)
            return f"{n} figure{'s' if n != 1 else ''}, all from desk data"
        n = len(self.unsupported)
        return f"{n} figure{'s' if n != 1 else ''} NOT found in the desk data"


def _decimals(tok: str) -> int:
    tok = tok.replace(",", "")
    return len(tok.split(".", 1)[1]) if "." in tok else 0


def _to_float(tok: str) -> float:
    return float(tok.replace(",", "").replace("$", "").replace("−", "-").lstrip("+"))


def find_mentions(prose: str) -> List[Mention]:
    out: List[Mention] = []
    for m in _NUM_RE.finditer(prose or ""):
        raw, suffix = m.group(1), (m.group(2) or "").lower()
        try:
            v = _to_float(raw)
        except ValueError:
            continue
        unit, scaled = "", False
        if suffix in ("%",):
            unit = "%"
        elif suffix in ("bp", "bps"):
            unit = "bp"
        elif suffix == "x":
            unit = "x"
        elif suffix in _SCALE:
            v *= _SCALE[suffix]
            scaled = True
        out.append(Mention(text=m.group(0).strip(), value=v, decimals=_decimals(raw),
                           start=m.start(), end=m.end(), unit=unit, scaled=scaled))
    return out


def _fact_forms(fact) -> Set[float]:
    """Every rendering of a fact that a careful analyst would call the same
    number. Kept deliberately narrow — each extra form is a hole in the check."""
    forms: Set[float] = set()
    v = getattr(fact, "value", None)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return forms
    v = float(v)
    unit = str(getattr(fact, "unit", "") or "")
    forms.add(v)
    forms.add(abs(v))
    if unit == "%":
        # A model handed 2.34 (percent) may write "0.023" and be right.
        forms.add(v / 100.0)
        forms.add(abs(v) / 100.0)
    elif unit in ("", "frac", "ratio") and abs(v) <= 1.0:
        # …and the reverse: a fraction rendered as a percentage.
        forms.add(v * 100.0)
        forms.add(abs(v) * 100.0)
    if unit == "bp":
        forms.add(v / 100.0)          # basis points quoted as a percentage
    if unit == "$":
        for s in (1e3, 1e6, 1e9):     # "$1.2 billion" against a raw 1_200_000_000
            forms.add(v / s)
            forms.add(abs(v) / s)

    # The fact's own ERROR BAR, and the interval it implies. A desk that reports
    # "VaR $12,340 ±$1,900" has supplied three defensible numbers, and an answer
    # written as "between $10,440 and $14,240" is quoting the desk, not inventing.
    # Without this the check would punish exactly the answers that carry their
    # uncertainty — the opposite of what it is for.
    err = getattr(fact, "error", None)
    if isinstance(err, (int, float)) and not isinstance(err, bool):
        e = float(err)
        if e == e and abs(e) != float("inf"):
            e = abs(e)
            for x in (e, v - e, v + e):
                forms.add(x)
                forms.add(abs(x))
            if unit == "%":
                for x in (e, v - e, v + e):
                    forms.add(x / 100.0)
    return forms


def _literal_tokens(text: str) -> Set[str]:
    """Numeric substrings appearing verbatim in a string — a parameter baked
    into a name ('20-day', '2s10s') is not a quoted figure."""
    return {t.replace(",", "") for t in re.findall(r"\d+(?:\.\d+)?", str(text or ""))}


def check(prose: str, facts: Sequence, *, question: str = "") -> GroundingReport:
    """Every figure in `prose` must match a figure in `facts`.

    `question` is passed so the model repeating the user's own numbers back
    ("compare AAPL over 30 days" → "over 30 days") is not called a fabrication.
    """
    mentions = find_mentions(prose)
    if not mentions:
        return GroundingReport(mentions=[], unsupported=[])

    forms: Set[float] = set()
    literals: Set[str] = set(_literal_tokens(question))
    for f in facts or []:
        forms |= _fact_forms(f)
        for attr in ("label", "note", "as_of", "source"):
            literals |= _literal_tokens(getattr(f, attr, "") or "")
        v = getattr(f, "value", None)
        if isinstance(v, str):
            literals |= _literal_tokens(v)

    unsupported: List[Mention] = []
    for men in mentions:
        if _is_supported(men, forms, literals):
            continue
        unsupported.append(men)
    return GroundingReport(mentions=mentions, unsupported=unsupported)


def _is_supported(men: Mention, forms: Iterable[float], literals: Set[str]) -> bool:
    # A year, or an integer date component, is not a market figure.
    if men.decimals == 0 and _YEAR_LO <= men.value <= _YEAR_HI and men.unit == "":
        return True
    # Verbatim in a fact's name or in the user's question.
    bare = men.text.lstrip("+-−$").split()[0].rstrip("%xkmbnt").rstrip(".")
    if bare.replace(",", "") in literals:
        return True
    d = min(max(men.decimals, 0), 6)
    for f in forms:
        # The model rounded; round the fact the same way before comparing.
        if round(f, d) == round(men.value, d):
            return True
        # A figure written with a scale suffix ("$1.2bn" for 1,234,000,000) is
        # deliberately coarse and cannot survive decimal rounding, so it gets a
        # relative tolerance. ONLY that case: applying the same slack to a bare
        # price would wave through "664.10" for a close of 663.70 — a wrong
        # number, of exactly the kind this check exists to catch.
        if men.scaled and f != 0 and abs(f - men.value) / abs(f) < 0.005:
            return True
    return False
