"""The tool layer — what an assistant is allowed to know, and from where.

This module is **domain-agnostic**. It defines what a tool is, what a fact is,
how a tool is validated and dispatched, and how a set of tools is held. It knows
nothing about markets, actuarial triangles, model anatomy, or any other subject:
a `Domain` (see `domain.py`) supplies the tools, and this layer runs them.

The model does not answer the question here. It *routes* it: it picks a tool and
its arguments, this module executes that tool deterministically against whatever
engines the domain wired up, and the resulting figures are rendered by the caller
verbatim. The model's prose is written afterwards, from those figures, and
checked against them (`grounding.py`).

Three rules hold the design together.

**A tool returns facts or a stated reason — never a number it had to guess.**
`ToolResult.unavailable` is a first-class outcome, distinct from an error and
from an empty list. "The engine is not connected" is a true and useful answer;
a zero is a lie with the same shape as a fact.

**Every fact carries its own as-of.** Tools update at different cadences — a
Treasury curve from this morning next to a position from a live socket — and an
answer that blends them without saying so is subtly wrong. The stamp travels
with the figure, not with the answer.

**Tools are read-only.** Nothing here places, cancels, or modifies anything. An
LLM-reachable side effect is not a feature this application will grow by
accident, and `Registry.register` refuses a tool whose name suggests one.

One registry per domain, and that is the isolation
--------------------------------------------------
`Registry` is a real object rather than a module-level dict, because the tools
one product exposes are not the tools another may reach. A single process-wide
catalogue meant that installing the markets desks put a book-risk tool in front
of every assistant in the application, including ones with no book. A domain
builds its own `Registry`; nothing merges them.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ── result types ─────────────────────────────────────────────────────────────
#: How well a figure is known, as a CLOSED vocabulary — the kind is the claim,
#: so a new one cannot be introduced by accident.
#:
#: The distinction that matters most is `exact` versus `unstated`. Both render
#: without a ± , and conflating them is the failure this exists to prevent: a
#: position size genuinely has no error, while a VaR whose bar nobody has
#: computed yet has an error that is simply unknown. A blank beside four figures
#: that carry bars reads as precision, so `unstated` says so out loud.
_ERROR_KINDS = frozenset({
    "exact",           # the number IS the quantity (share count, config value)
    "capture",         # a stored observation; the bar is the capture Δ
    "finite-sample",   # an estimate; the bar is a computed SE / bootstrap band
    "model",           # a model output; its uncertainty is not honestly bounded
    "unstated",        # not yet classified — never read as exact
})


@dataclass(frozen=True)
class Fact:
    """One figure, with everything needed to defend it."""
    label: str
    value: Any                 # float | str | None
    unit: str = ""             # "%", "$", "bp", "x", "" …
    as_of: str = ""
    note: str = ""
    # A CHANGE reads better with an explicit sign ("+3.79%"); a LEVEL does not
    # — "+99.17%" for a percentile rank is simply wrong-looking.
    signed: bool = False

    # ── how well this figure is known ───────────────────────────────────────
    #: Half-width in the SAME unit as `value`, or None when no bar exists. None
    #: is NOT a claim of exactness — `error_kind` carries that distinction, and
    #: the two must be read together.
    error: Optional[float] = None
    #: Which KIND of number this is, epistemically. Closed vocabulary, because
    #: the kind IS the claim. See `_ERROR_KINDS`.
    error_kind: str = "unstated"
    #: What the bar is, or why there is none. Free text, always shown.
    error_note: str = ""

    def __post_init__(self) -> None:
        # An unrecognised kind would quietly become an unauditable claim.
        if self.error_kind not in _ERROR_KINDS:
            object.__setattr__(self, "error_kind", "unstated")
        # A bar on a kind that cannot have one is a contradiction, and the
        # dangerous direction is keeping the number: it would render as a
        # measured band that nothing measured. Drop it and say so.
        if self.error is not None and self.error_kind in ("exact", "model"):
            object.__setattr__(self, "error", None)
            object.__setattr__(
                self, "error_note",
                (self.error_note + " ").lstrip()
                + f"[an error bar was supplied for a '{self.error_kind}' figure "
                  f"and dropped — that kind carries no measured band]")
        if self.error is not None:
            try:
                e = abs(float(self.error))
                object.__setattr__(self, "error",
                                   e if math.isfinite(e) else None)
            except (TypeError, ValueError):
                object.__setattr__(self, "error", None)

    @property
    def is_number(self) -> bool:
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    def rendered(self) -> str:
        """How the figure is displayed. The panel prints this; the model is only
        ever shown this string, so the two can never disagree."""
        v = self.value
        if v is None:
            return "n/a"
        if not self.is_number:
            return str(v)
        v = float(v)
        if not math.isfinite(v):
            return "n/a"
        if self.unit == "$":
            a = abs(v)
            if a >= 1e9:
                return f"${v/1e9:,.2f}bn"
            if a >= 1e6:
                return f"${v/1e6:,.2f}m"
            return f"${v:,.0f}" if a >= 1000 else f"${v:,.2f}"
        if self.unit == "%":
            return f"{v:+.2f}%" if self.signed else f"{v:.2f}%"
        if self.unit == "bp":
            return f"{v:+.0f} bp" if self.signed else f"{v:.0f} bp"
        if self.unit == "x":
            return f"{v:.2f}x"
        if self.unit == "shares":
            return f"{v:,.0f}"
        return f"{v:,.4g}"

    def rendered_error(self) -> str:
        """The ± half-width on its own, in the figure's own units. "" when
        there is no bar — read `uncertainty()` for WHY there is none."""
        if self.error is None or not self.is_number:
            return ""
        e = float(self.error)
        if not math.isfinite(e):
            return ""
        if self.unit == "$":
            a = abs(e)
            if a >= 1e9:
                return f"±${e/1e9:,.2f}bn"
            if a >= 1e6:
                return f"±${e/1e6:,.2f}m"
            return f"±${e:,.0f}" if a >= 1000 else f"±${e:,.2f}"
        if self.unit == "%":
            return f"±{e:.2f}%"
        if self.unit == "bp":
            return f"±{e:.0f} bp"
        if self.unit == "x":
            return f"±{e:.2f}x"
        if self.unit == "shares":
            return f"±{e:,.0f}"
        return f"±{e:,.4g}"

    def uncertainty(self) -> str:
        """One phrase saying how well this figure is known — ALWAYS non-empty.

        Silence is the thing being avoided. A figure printed bare beside figures
        that carry bars is read as the precise one, so every kind says something,
        including the kinds that have no bar to give.
        """
        bar = self.rendered_error()
        detail = f" · {self.error_note}" if self.error_note else ""
        if self.error_kind == "exact":
            return "exact" + detail
        if self.error_kind == "model":
            return ("model output — no measured error bar"
                    + (detail or " · uncertainty is model choice, not sampling"))
        if self.error_kind == "unstated":
            return "no error bar stated" + detail
        if not bar:
            return f"{self.error_kind} — bar not computed" + detail
        return f"{bar} ({self.error_kind}){detail}"

    def line(self) -> str:
        bits = [f"{self.label}: {self.rendered()}"]
        if self.error is not None:
            bits.append(self.rendered_error())
        if self.as_of:
            bits.append(f"(as of {self.as_of})")
        if self.note:
            bits.append(f"— {self.note}")
        return " ".join(bits)


@dataclass(frozen=True)
class ToolResult:
    tool: str
    args: Dict[str, Any]
    facts: List[Fact] = field(default_factory=list)
    source: str = ""             # which desk/module produced this
    unavailable: str = ""        # a STATED reason there is no answer
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.facts) and not self.error and not self.unavailable

    def as_prompt_block(self) -> str:
        """What the model is shown. Deliberately terse and figure-first."""
        head = f"[{self.tool}{_args_str(self.args)}]"
        if self.error:
            return f"{head} FAILED: {self.error}"
        if self.unavailable:
            return f"{head} NO DATA: {self.unavailable}"
        if not self.facts:
            return f"{head} NO DATA: the desk returned nothing for these arguments"
        lines = "\n".join("  " + f.line() for f in self.facts)
        src = f"\n  source: {self.source}" if self.source else ""
        return f"{head}\n{lines}{src}"


def stamp(value: Any) -> str:
    """Normalise whatever a desk calls its as-of into something readable.

    The desks disagree: `RatesDesk.asof` is an epoch float, `VolSurface.asof` is
    a date, others hand back a string. Passing them through `str()` put
    `1785254303.112` on screen next to a Treasury yield — technically the right
    instant, and useless to the person reading it."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        v = float(value)
        if v <= 0:
            return ""
        # Anything past 2001 is an epoch; a bare year is not a timestamp.
        if v > 1e9:
            return time.strftime("%Y-%m-%d %H:%M", time.localtime(v))
        return str(value)
    try:                                   # date / datetime / pandas Timestamp
        return value.strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return str(value)


def _args_str(args: Dict[str, Any]) -> str:
    if not args:
        return ""
    return "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"


# ── tool declaration ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Param:
    name: str
    kind: str = "str"            # "str" | "symbol" | "int" | "float"
    required: bool = True
    help: str = ""
    default: Any = None


@dataclass(frozen=True)
class Tool:
    name: str
    answers: str                 # the user's question, in plain English
    params: Tuple[Param, ...] = ()
    fn: Optional[Callable[["Context", Dict[str, Any]], ToolResult]] = None
    #: The view a user would open to see this with their own eyes. Free text
    #: owned by the domain — "Book Risk", "Model Morphometry", "" for a tool
    #: with no screen behind it.
    surface: str = ""

    def signature(self) -> str:
        if not self.params:
            return self.name + "()"
        inner = ", ".join(p.name if p.required else f"{p.name}?" for p in self.params)
        return f"{self.name}({inner})"


@dataclass
class Context:
    """The host's injected dependencies. Every one is optional; a tool whose
    dependency is absent says so rather than inventing a stand-in — which is why
    this is a plain container and not a service locator that would construct one.

    `extras` is the general seam and the one a new domain should reach for: it
    is an open dict the host fills and the domain's own tools read. The named
    fields below are the **markets** domain's, kept as fields because seven desk
    modules and a live panel already read them by name. A domain that has no
    book simply leaves them at their defaults, and no code in this package
    reads them — privacy is decided by `Domain.private_context`, so the engine
    never has to know what a position is.
    """
    data_service: Any = None
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    account: Dict[str, Any] = field(default_factory=dict)
    connected: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def net_liq(self) -> Optional[float]:
        try:
            v = float(self.account.get("net_liq") or 0.0)
        except Exception:  # noqa: BLE001
            return None
        return v if v > 0 else None


# ── the registry ─────────────────────────────────────────────────────────────
#: Substrings that would make a tool name read like an action. A read-only tool
#: layer is the property the whole design rests on, and the cheapest place to
#: hold it is at registration: a domain pack cannot introduce a side-effecting
#: tool by naming one, whoever wrote the pack.
_FORBIDDEN_NAME_PARTS = ("place", "submit", "cancel", "send_order", "execute",
                         "trade", "flatten", "liquidate")


class ToolNameRefused(ValueError):
    """A tool name that reads as an action was refused at registration."""


class Registry:
    """One domain's tool catalogue.

    Deliberately an object. The catalogue used to be a module-level dict, which
    made every installed tool reachable from every assistant in the process —
    so opening a general assistant on a machine with the markets desks
    installed put `book_risk` in its router menu. Isolation here is structural:
    a `Registry` holds what its domain installed and has no route to another's.
    """

    __slots__ = ("_tools",)

    def __init__(self, tools: Optional[Dict[str, Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = dict(tools or {})

    def register(self, tool: Tool) -> Tool:
        name = str(tool.name or "").strip()
        if not name:
            raise ToolNameRefused("a tool must have a name")
        lowered = name.lower()
        if any(part in lowered for part in _FORBIDDEN_NAME_PARTS):
            raise ToolNameRefused(
                f"tool {name!r} reads as an action. The tool layer is read-only "
                f"by construction; a tool that can change something does not "
                f"belong behind a language model.")
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(str(name or "").strip())

    def all_tools(self) -> List[Tool]:
        return [self._tools[k] for k in sorted(self._tools)]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def catalog_text(self) -> str:
        """The tool menu shown to the router model. One line each: signature,
        then the question it answers — models route far better on the question
        than on a type signature."""
        return "\n".join(f"- {t.signature()} — {t.answers}"
                         for t in self.all_tools())

    def run(self, name: str, args: Dict[str, Any], ctx: Context) -> ToolResult:
        return _run(self, name, args, ctx)

    def run_plan(self, calls: Sequence[Tuple[str, Dict[str, Any]]],
                 ctx: Context, *, limit: int = 4) -> List[ToolResult]:
        return _run_plan(self, calls, ctx, limit=limit)

    def clear(self) -> None:
        self._tools.clear()

    def snapshot(self) -> Dict[str, Tool]:
        """A copy, for a test that wants to restore the catalogue afterwards."""
        return dict(self._tools)

    def restore(self, snapshot: Dict[str, Tool]) -> None:
        self._tools = dict(snapshot)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return str(name) in self._tools

    def __repr__(self) -> str:                    # pragma: no cover - debugging
        return f"Registry({len(self._tools)} tools)"


#: The registry a caller gets when it names no domain. Ad-hoc registration in a
#: test, and nothing in the shipped answer path: every product reaches its tools
#: through its own `Domain`.
_REGISTRY = Registry()


def register(tool: Tool) -> Tool:
    return _REGISTRY.register(tool)


def get(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def all_tools() -> List[Tool]:
    return _REGISTRY.all_tools()


def catalog_text() -> str:
    return _REGISTRY.catalog_text()


# ── validation + dispatch ────────────────────────────────────────────────────
def _coerce(p: Param, raw: Any) -> Any:
    if p.kind == "symbol":
        s = str(raw or "").strip().upper()
        if not s:
            raise ValueError(f"{p.name} is empty")
        return s
    if p.kind == "int":
        return int(float(raw))
    if p.kind == "float":
        return float(raw)
    return str(raw).strip()


def validate(tool: Tool, args: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Returns (clean_args, error). Unknown keys are dropped rather than
    rejected — a model that adds a stray field should still get its answer."""
    clean: Dict[str, Any] = {}
    args = dict(args or {})
    for p in tool.params:
        if p.name not in args or args[p.name] in (None, ""):
            if p.required:
                return {}, f"missing required argument '{p.name}'"
            if p.default is not None:
                clean[p.name] = p.default
            continue
        try:
            clean[p.name] = _coerce(p, args[p.name])
        except Exception:  # noqa: BLE001
            return {}, f"argument '{p.name}' is not a valid {p.kind}"
    return clean, ""


def _run(registry: Registry, name: str, args: Dict[str, Any],
         ctx: Context) -> ToolResult:
    """Execute one tool. Never raises: a broken tool becomes a visible failed
    call, which the reader can see and reason about, rather than an exception
    that silently costs them the answer."""
    tool = registry.get(name)
    if tool is None:
        return ToolResult(tool=str(name), args=dict(args or {}),
                          error="no such tool")
    clean, err = validate(tool, args)
    if err:
        return ToolResult(tool=tool.name, args=dict(args or {}), error=err)
    if tool.fn is None:
        return ToolResult(tool=tool.name, args=clean,
                          error="tool is declared but not implemented")
    try:
        res = tool.fn(ctx, clean)
    except Exception as exc:  # noqa: BLE001
        return ToolResult(tool=tool.name, args=clean,
                          error=f"{type(exc).__name__}: {exc}")
    if not isinstance(res, ToolResult):
        return ToolResult(tool=tool.name, args=clean,
                          error="tool returned a malformed result")
    return res


def _run_plan(registry: Registry, calls: Sequence[Tuple[str, Dict[str, Any]]],
              ctx: Context, *, limit: int = 4) -> List[ToolResult]:
    """Run a routed plan in order, bounded. The cap is not arbitrary: each call
    costs the reader wall-clock time while they wait, and a model that asks for
    nine tools has usually misunderstood the question rather than decomposed it."""
    out: List[ToolResult] = []
    for name, args in list(calls)[:max(1, int(limit))]:
        out.append(_run(registry, name, args, ctx))
    return out


def run(name: str, args: Dict[str, Any], ctx: Context,
        *, registry: Optional[Registry] = None) -> ToolResult:
    """Execute one tool from `registry`, or from the default one."""
    return _run(registry if registry is not None else _REGISTRY, name, args, ctx)


def run_plan(calls: Sequence[Tuple[str, Dict[str, Any]]], ctx: Context,
             *, limit: int = 4,
             registry: Optional[Registry] = None) -> List[ToolResult]:
    """Run a routed plan against `registry`, or against the default one."""
    return _run_plan(registry if registry is not None else _REGISTRY,
                     calls, ctx, limit=limit)


def facts_of(results: Sequence[ToolResult]) -> List[Fact]:
    out: List[Fact] = []
    for r in results:
        out.extend(r.facts)
    return out


def _reset_registry_for_tests() -> None:
    _REGISTRY.clear()
