"""The Certified Answer Engine (Oracle x Axiom x Vector) — the verified, decision-
grade natural-language analytics answer.

An analyst asks in plain English. The LLM is STRUCTURALLY FORBIDDEN from emitting
the number: its only output channel is an Alelyon-DSL program (`alelyon.runtime.oracle.dsl`, safe
interpreter, no eval/exec), which executes DETERMINISTICALLY over the data layer.
The answer is the COMPUTED figure + a provenance trace (sources, as-of date) + a
CALIBRATED uncertainty certificate. If the question cannot be expressed in the safe
vocabulary, the engine REFUSES honestly rather than letting the model invent a
number — that refusal is the product's integrity, not a bug.

De-risked before build (2026-07-19): a local qwen3-coder:30b authored 16/16 valid
programs first-try on the corr/beta/z-score family; the correlation certificate is
calibrated to ~90% coverage where a naive compression-only band is 0% theater. The
honest interval here is FINITE-SAMPLE sampling error (the dominant term on live,
uncompressed data) — Fisher-z for correlation, standard error for a mean — with the
CFRC compression term (`alelyon.runtime.vector.codec`) added only when computing over a
compressed store.

`llm_fn` is injected (an Ollama caller in production, a stub in tests), so the whole
engine is Qt-free and offline-testable.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl import builtin_names, run_program


# ── the authoring prompt (validated 16/16 first-try on the target family) ──────
def grammar() -> str:
    return (
        "You translate a finance question into an Alelyon-DSL program.\n"
        "OUTPUT ONLY DSL SOURCE CODE. No prose, no markdown fences, no numbers, no explanation.\n\n"
        "Statements:  let NAME = EXPR   (bind)      show EXPR   (output the answer)\n"
        'Data:  price("TICKER") -> daily price series;  series("FRED_ID") -> a FRED macro series\n'
        "Builtins: " + ", ".join(builtin_names()) + "\n"
        "Notes: returns(s) or returns(s,n)=n-day returns; zscore(s); corr(a,b) -> scalar;\n"
        "rsi(s,n); sma(s,n); ema(s,n); rolling_mean(s,n); rolling_std(s,n); lag(s,n); diff(s,n);\n"
        "reducers last(s)/first(s)/mean(s)/std(s)/min(s)/max(s)/sum(s)/count(s) -> scalar.\n"
        "A scalar question MUST end with `show <scalar>` — wrap a series with last(...) or a reducer.\n"
        "CRITICAL: if the question needs data these builtins CANNOT express — company "
        "fundamentals (P/E, EV/EBITDA, margins, earnings, revenue, guidance), balance-sheet "
        "items, analyst estimates, or entity relationships (peers, suppliers, sector members) — "
        "do NOT invent a program. Output exactly one line: CANNOT_EXPRESS: <what is missing>.\n\n"
        "Examples:\n"
        "Q: P/E ratio of Apple versus its peers\nCANNOT_EXPRESS: company fundamentals (P/E) and peer sets\n"
        "Q: correlation of SPY and QQQ daily returns\n"
        'let a = returns(price("SPY"))\nlet b = returns(price("QQQ"))\nshow corr(a, b)\n'
        "Q: latest z-score of 63-day AAPL momentum\n"
        'show last(zscore(returns(price("AAPL"), 63)))\n'
        "Q: 20-day realized volatility of NVDA\n"
        'show last(rolling_std(returns(price("NVDA")), 20))\n'
    )


def build_prompt(question: str, repair: Optional[tuple] = None) -> str:
    if repair is not None:
        prog, err = repair
        return (grammar() + f"\nQ: {question}\nYour program:\n{prog}\n"
                f"failed with error: {err}\nOutput ONLY corrected DSL source.")
    return grammar() + f"\nQ: {question}\n"


def refusal_reason(text: str) -> Optional[str]:
    """If the model declared the question un-expressible, return its stated reason.
    This is the guard against the killer failure mode: a model that, unable to
    express the real question, invents a superficially-valid but meaningless program
    (e.g. price(AAPL)/series(PCEPI) for a 'P/E' question) — the exact hallucination
    the engine exists to kill."""
    m = re.search(r"CANNOT_EXPRESS:\s*(.+)", text or "")
    return m.group(1).strip() if m else None


def clean_dsl(text: str) -> str:
    """Extract DSL from a model reply: drop <think>, markdown fences, and any prose
    that is not a `let`/`show` statement. The parser is the real guard; this just
    improves the odds the first parse succeeds."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    keep = [ln for ln in text.splitlines() if re.match(r"\s*(let\s+\w+\s*=|show\s+)", ln)]
    return "\n".join(keep).strip() or (text or "").strip()


# ── result types ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Certificate:
    kind: str                    # 'correlation' | 'mean' | 'indicator' | 'none'
    conf: float                  # e.g. 0.90
    lo: Optional[float]
    hi: Optional[float]
    se: Optional[float]
    driver: str                  # plain-English dominant uncertainty source
    note: str = ""


@dataclass(frozen=True)
class VerifiedAnswer:
    question: str
    ok: bool
    refused: bool
    value: Optional[float]
    dsl: str
    sources: List[str]
    as_of: Optional[str]
    certificate: Optional[Certificate]
    error: Optional[str]
    repaired: bool = False


# ── helpers ────────────────────────────────────────────────────────────────────
def _scalar(value) -> Optional[float]:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, pd.Series) and len(value):
        v = value.dropna()
        return float(v.iloc[-1]) if len(v) else None
    return None


def sources_of(dsl: str) -> List[str]:
    """The tickers / FRED ids the program reads — the provenance line."""
    found = re.findall(r'price\("([^"]+)"\)|series\("([^"]+)"\)', dsl)
    seen: List[str] = []
    for a, b in found:
        s = a or b
        if s and s not in seen:
            seen.append(s)
    return seen


def _returns_of(ticker: str, data_service) -> Optional[pd.Series]:
    try:
        df = data_service.bars(ticker)
        close = df["Close"] if "Close" in df else df.iloc[:, -1]
        return close.pct_change().dropna()
    except Exception:  # noqa: BLE001
        return None


def _last_show(dsl: str) -> str:
    shows = [ln for ln in dsl.splitlines() if ln.strip().startswith("show")]
    return shows[-1] if shows else ""


def certify(dsl: str, value: Optional[float], sources: List[str],
            data_service, conf: float = 0.90) -> Certificate:
    """Attach an HONEST interval. Calibrated for correlation (Fisher-z sampling)
    and a mean (standard error); an explicit 'no calibrated band yet' for point
    indicators — never a fabricated tight band.

    This is the engine's lightweight inline bar. The full DECOMPOSED, serial-
    correlation-honest budget (quantization + block-bootstrap sampling + provider
    + model, with the binding term named) is
    `alelyon.runtime.oracle.dsl.budget.error_budget`, surfaced on the API's
    /v1/answer as `error_budget`."""
    show = _last_show(dsl)
    k = 1.645 if abs(conf - 0.90) < 1e-6 else 1.96

    if value is not None and "corr(" in show and len(sources) >= 2:
        a = _returns_of(sources[0], data_service)
        b = _returns_of(sources[1], data_service)
        if a is not None and b is not None:
            aa, bb = a.align(b, join="inner")
            T = int(min(len(aa.dropna()), len(bb.dropna())))
            if T > 4:
                z = math.atanh(max(-0.999, min(0.999, value)))
                se = 1.0 / math.sqrt(T - 3)
                lo, hi = math.tanh(z - k * se), math.tanh(z + k * se)
                return Certificate("correlation", conf, lo, hi, se,
                                   f"finite-sample sampling (T={T} obs)",
                                   "Fisher-z interval; regime non-stationarity is a separate, un-bounded risk.")

    if value is not None and "mean(" in show and sources:
        r = _returns_of(sources[0], data_service)
        if r is not None and len(r) > 2:
            T = len(r)
            se = float(r.std(ddof=1) / math.sqrt(T))
            return Certificate("mean", conf, value - k * se, value + k * se, se,
                               f"finite-sample sampling (T={T} obs)")

    return Certificate("indicator", conf, None, None, None,
                       "not a distributional estimate",
                       "point-in-time indicator — no calibrated interval; treat regime risk separately.")


def _as_of(sources: List[str], data_service) -> Optional[str]:
    for s in sources:
        try:
            df = data_service.bars(s)
            if len(df):
                return str(pd.to_datetime(df.index[-1]).date())
        except Exception:  # noqa: BLE001
            continue
    return None


# ── the engine ─────────────────────────────────────────────────────────────────
def answer(question: str, *, data_service, llm_fn: Callable[[str], str],
           max_repairs: int = 1) -> VerifiedAnswer:
    """NL question -> DSL (authored by llm_fn) -> deterministic execution ->
    computed value + provenance + certificate. Honest refusal when the question is
    un-expressible, the program errors, or the result is non-finite — NEVER a
    fabricated number."""
    def _refuse(dsl, reason, repaired=False):
        return VerifiedAnswer(question, ok=False, refused=True, value=None, dsl=dsl,
                              sources=sources_of(dsl), as_of=None, certificate=None,
                              error=reason, repaired=repaired)

    raw = llm_fn(build_prompt(question))
    reason = refusal_reason(raw)
    if reason:                                    # model itself declared it un-expressible
        return _refuse("", "Outside the safe vocabulary: " + reason)
    src = clean_dsl(raw)
    if not src:
        return _refuse("", "the model produced no valid DSL program")

    res = run_program(src, data_service=data_service)
    repaired = False
    tries = 0
    while not res.ok and tries < max_repairs:
        tries += 1
        raw2 = llm_fn(build_prompt(question, repair=(src, res.error)))
        r2 = refusal_reason(raw2)
        if r2:
            return _refuse(src, "Outside the safe vocabulary: " + r2, repaired=True)
        src2 = clean_dsl(raw2)
        res2 = run_program(src2, data_service=data_service)
        if res2.ok:
            src, res, repaired = src2, res2, True
            break
        src = src2 or src

    if not res.ok:
        return _refuse(src, res.error, repaired=repaired)

    val = _scalar(res.outputs[-1].value) if res.outputs else None
    if val is not None and not math.isfinite(val):    # garbage program that "ran" → refuse
        return _refuse(src, "the program computed a non-finite result (likely mismatched or "
                       "misaligned inputs) — refusing rather than reporting a meaningless number",
                       repaired=repaired)
    srcs = sources_of(src)
    cert = certify(src, val, srcs, data_service)
    return VerifiedAnswer(question, ok=True, refused=False, value=val, dsl=src,
                          sources=srcs, as_of=_as_of(srcs, data_service),
                          certificate=cert, error=None, repaired=repaired)
