"""The decomposed capture-to-answer ERROR BUDGET (data-verification foothold, W4).

The DRC certificate bounds STORAGE QUANTIZATION — and honestly says that at 24-bit
capture that term is ~1e4 smaller than the sampling error, i.e. the term that does
NOT bind the decision. The skeptic's rebuttal — "you certify the wrong thing" — is
answered not by a tighter quantization bound but by a DECOMPOSED budget that also
carries the term that DOES bind, each line labeled with its epistemic status and
NEVER summed into one laundered "certified" number:

  quantization  the DRC bound — a theorem (linear/branch-stable-exact) or a
                harness-validated first-order claim (smooth). Already tiny.
  sampling      finite-sample uncertainty of the statistic, SERIAL-CORRELATION-
                HONEST (circular block bootstrap, not an iid SE). This is the
                term that usually binds. Refusal-first: computed only for
                recognized distributional statistics over a stationary series;
                a point-in-time indicator gets an explicit "not a distributional
                estimate", never a fabricated band.
  provider      cross-source evidence about the INPUT — see the caveats below.
                This is an INPUT-SPACE diagnostic, not an output-space width.
  model         for a pure DSL program the program IS the computation, so there
                is no separate model-fit uncertainty (named N/A, never zero).

The composer reports the DOMINANT (binding) term and its ratio to quantization —
the honest headline — plus an addable composite (root-sum-square of the two
INDEPENDENT terms) that is explicitly a COMPOSITION, not a single certified scalar.

TWO CORRECTIONS TO THE PROVIDER TERM (2026-07-28), both of which shipped wrong:

1. UNITS. quantization and sampling are half-widths on the OUTPUT statistic (e.g.
   a mean daily return). The provider term is a half-range on the INPUT level (e.g.
   dollars of price). They were compared in the same `max()`, so a one-cent
   disagreement between two price sources would have been elected the binding term
   of a dimensionless return and printed as "provider binds the answer (half-width
   0.01)". Every `Term` now declares its `space`, and only output-space terms can
   be dominant or enter the composite. (Latent only because the seam it needed was
   never implemented — the shipped test asserted the wrong behaviour.)

2. FRAMING. The old text called cross-source dispersion "a LOWER-BOUND diagnostic"
   on the error. That is false for the number we serve. Writing x_i = μ + b_i + e_i,
   the observed half-range lower-bounds max_i |x_i − μ| by the pigeonhole principle
   — a statement about the SOURCES, not about the value we reported, which can be
   exactly right while the sources disagree wildly. And no statistic computed from
   k readings says anything about the COMMON bias mean(b): only the contrasts
   x_i − x_j are observable, so mean(b) is confounded with μ at every k. Sources
   that are wrong together agree perfectly. With k=2 (the most this deployment can
   reach) the disagreement cannot even be attributed to one source — splitting
   Var(x₁−x₂) = s₁² + s₂² needs k≥3 mutually independent datasets (triple
   collocation). The term is therefore reported as a CONSISTENCY CHECK between
   declared-distinct origins, never as a bound on data error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl import nodes as N
from alelyon.runtime.oracle.dsl.execcert import (FetchedSeries, StoreCertifiedFetcher,
                                                 certified_run, _prune_to_output)
from alelyon.runtime.oracle.dsl.interpreter import run_program
from alelyon.runtime.oracle.dsl.parser import parse

_STATIONARY = {"returns", "logret", "diff", "zscore"}   # (near-)stationary transforms


@dataclass(frozen=True)
class Term:
    name: str
    width: Optional[float]           # half-width at the requested confidence, or None
    status: str                      # see _STATUSES
    detail: str
    extra: dict = field(default_factory=dict)
    space: str = "output"            # "output" = a width on the ANSWER; "input" = a
                                     # diagnostic in the units of the raw data. Only
                                     # output-space terms are commensurable with each
                                     # other, so only they can be dominant or composed.


# Every status a term may carry. Kept explicit so a new one cannot be introduced
# silently — the statuses ARE the epistemic claim, and a reader who cannot tell
# `one-configured-origin` from `not-attempted` is being told nothing.
_STATUSES = frozenset({
    "theorem", "first-order",            # quantization
    "finite-sample",                     # sampling
    "not-applicable",                    # model, for a pure DSL program
    "refused", "unmeasured",             # nothing to report, reason given
    # The provider statuses all describe THIS DEPLOYMENT's configured providers,
    # never the datum. An earlier draft called the one-origin case "sole-source"
    # and glossed it as a permanent property of the data; it is not — it moves
    # with the provider list and with symbol-coverage heuristics.
    "no-configured-source",              # nothing here can serve the input at all
    "one-configured-origin",             # exactly one declared upstream, one client
    "single-origin-chain",               # >1 provider, all declaring the same origin
    "origins-unreachable",               # >=2 declared, <2 currently answering
    "incomparable-origins",              # >1 origin, but not the same measurand
    "not-attempted",                     # corroborable + comparable, none recorded
    "consistency-checked",               # observations exist and were compared
})


@dataclass
class ErrorBudget:
    value: Optional[float]
    refused: bool
    reason: Optional[str]
    conf: float
    terms: Dict[str, Term]
    dominant: Optional[str]
    dominant_note: str
    composite_width: Optional[float]   # RSS of the INDEPENDENT terms (quant ⊥ sampling)
    composite_note: str


# ── program recognition for the sampling estimator ───────────────────────────
def _recognize(prog: N.Program) -> Optional[dict]:
    """If the final output is a distributional statistic over a stationary series,
    return {kind, inners:[expr,...]} the sampling bootstrap can handle; else None.
    Recognized: mean/std/sum(count-rate) over a stationary transform; corr of two."""
    out = None
    for s in reversed(prog.statements):
        if isinstance(s, (N.Show, N.Signal)):
            out = s.expr
            break
    if not isinstance(out, N.Call):
        return None

    def _is_stationary(e) -> bool:
        return isinstance(e, N.Call) and e.func in _STATIONARY

    if out.func in ("mean", "std") and len(out.args) == 1 and _is_stationary(out.args[0]):
        return {"kind": out.func, "inners": [out.args[0]]}
    if out.func == "corr" and len(out.args) == 2 \
            and all(_is_stationary(a) for a in out.args):
        return {"kind": "corr", "inners": list(out.args)}
    return None


def _eval_series(expr, ctx) -> Optional[pd.Series]:
    res = run_program("", program=N.Program([N.Show(expr)]), ctx=ctx)
    if not getattr(res, "ok", False) or not res.outputs:
        return None
    v = res.outputs[-1].value
    return v if isinstance(v, pd.Series) else None


# ── circular block bootstrap (serial-correlation honest) ─────────────────────
def _block_len(T: int) -> int:
    return max(1, int(round(T ** (1.0 / 3.0))))          # standard rule of thumb


def _cbb_indices(T: int, L: int, rng) -> np.ndarray:
    n_blocks = -(-T // L)                                  # ceil(T/L)
    starts = rng.integers(0, T, size=n_blocks)
    return np.concatenate([(np.arange(L) + s) % T for s in starts])[:T]


def _sampling_term(prog: N.Program, fetched: Dict[Tuple[str, str], FetchedSeries],
                   conf: float, seed: int, B: int) -> Term:
    from alelyon.runtime.oracle.dsl.execcert import _FixedContext
    spec = _recognize(prog)
    if spec is None:
        return Term("sampling", None, "not-applicable",
                    "not a distributional statistic over a stationary series — a "
                    "point-in-time value has no finite-sample sampling band")
    ctx = _FixedContext({r: f.series for r, f in fetched.items()})
    inners = [_eval_series(e, ctx) for e in spec["inners"]]
    if any(s is None for s in inners):
        return Term("sampling", None, "refused",
                    "could not evaluate the stationary series to bootstrap")

    if spec["kind"] == "corr":
        a, b = inners[0].dropna(), inners[1].dropna()
        aa, bb = a.align(b, join="inner")
        va = aa.to_numpy(dtype=float); vb = bb.to_numpy(dtype=float)
        T = len(va)
        if T < 8:
            return Term("sampling", None, "refused", f"only {T} paired obs")
        stat = lambda ia: float(np.corrcoef(va[ia], vb[ia])[0, 1])
    else:
        v = inners[0].dropna().to_numpy(dtype=float)
        T = len(v)
        if T < 8:
            return Term("sampling", None, "refused", f"only {T} obs")
        if spec["kind"] == "mean":
            stat = lambda ia: float(np.mean(v[ia]))
        else:  # std (ddof=1)
            stat = lambda ia: float(np.std(v[ia], ddof=1))

    L = _block_len(T)
    rng = np.random.default_rng([seed, 0xB007])
    boots = []
    for _ in range(B):
        ia = _cbb_indices(T, L, rng)
        s = stat(ia)
        if s == s:                                        # skip NaN draws
            boots.append(s)
    if len(boots) < B // 2:
        return Term("sampling", None, "refused", "bootstrap degenerate")
    boots = np.asarray(boots)
    lo = float(np.quantile(boots, (1 - conf) / 2))
    hi = float(np.quantile(boots, (1 + conf) / 2))
    width = (hi - lo) / 2.0
    return Term("sampling", width, "finite-sample",
                f"circular block bootstrap ({spec['kind']}, T={T}, block≈{L}, "
                f"B={len(boots)}) — serial-correlation honest",
                {"T": T, "block_len": L, "se": float(boots.std(ddof=1)),
                 "lo": lo, "hi": hi})


#: Travels with every provider term, on the certificate itself rather than in a
#: design doc. Each clause is load-bearing; see the module docstring's correction 2.
SCOPE_SENTENCE = (
    "A corroboration record proves only that sources with DECLARED-DISTINCT upstream "
    "origins, sampled under a committed policy, disagreed by at most this much on the "
    "rows this answer consumed. It does NOT prove any of them is correct, that the "
    "origins are genuinely independent, or that the true value lies within it — "
    "sources that are wrong together agree perfectly, and no statistic computed from "
    "k readings constrains their common bias.")


def provider_status(refs: List[Tuple[str, str]], data_service) -> dict:
    """What cross-source evidence exists about these inputs — and when none exists,
    WHY, in terms a reader can check. Network-free (capability declarations only),
    so it is cheap enough to compute while issuing a certificate.

    The statuses are the point. `unmeasured` was one blank covering four completely
    different situations, and a blank next to four filled lines reads as agreement:

      no-configured-source  nothing configured here can serve the input at all.
                            NOT a source count of one.
      one-configured-origin exactly one declared upstream, reached by one client.
      single-origin-chain   several providers, all declaring ONE origin (the two
                            Yahoo client paths) — the redundancy present is not
                            the redundancy needed.
      incomparable-origins  >1 upstream, but they do not report the same measurand.
      not-attempted         corroborable and comparable, no observation recorded.
      consistency-checked   observations exist and were compared.

    EVERY status describes THIS DEPLOYMENT'S configured providers, never the datum.
    The count moves with the provider list and with symbol-coverage heuristics.

    `trust` is always "signer-attested": this describes the ISSUER'S DEPLOYMENT
    (which upstreams it has configured), and no verifier can confirm that from
    outside. Labeled the same way `width_trust` distinguishes authenticated from
    transparency-anchored — an attested claim is fine, an attested claim wearing
    the clothes of a verified one is not.
    """
    def _u(reason: str) -> dict:
        return {"status": "unmeasured", "reason": reason, "trust": "signer-attested",
                "scope_note": SCOPE_SENTENCE}

    if data_service is None:
        return _u("no data layer supplied — nothing was asked about the sources; "
                  "NOT a finding that they agree")
    cat_fn = getattr(data_service, "corroborability", None)
    if not callable(cat_fn):
        return _u("this data layer publishes no origin catalogue, so the number of "
                  "upstreams behind these inputs is unknown")
    per_ref = {}
    try:
        for kind, key in refs:
            per_ref[f"{kind}:{key}"] = cat_fn(kind, key)
    except Exception as exc:  # noqa: BLE001
        return _u(f"origin catalogue unreadable: {type(exc).__name__}")
    if not per_ref:
        return _u("program reads no inputs")

    # The weakest input governs: one un-corroborated series makes the whole
    # answer un-corroborated, however well the others are covered.
    # Worst first. EVERY status the classifier can emit is ranked here, including
    # the best one — an unranked status falls to -1 and wins as worst, so a label
    # added later surfaces instead of being silently outranked. (Ranking only the
    # bad statuses would have made `consistency-checked`, the BEST outcome, sort
    # to -1 and be reported as the weakest input.)
    order = {"no-configured-source": 0, "single-origin-chain": 1,
             "one-configured-origin": 2, "origins-unreachable": 3,
             "incomparable-origins": 4, "not-attempted": 5,
             "consistency-checked": 6}
    labelled = {}
    for ref, cat in per_ref.items():
        origins = list(cat.get("origin_ids") or [])
        n = int(cat.get("origins_possible", 0))
        # `origins_reachable` is absent on a catalogue that does not report health;
        # default to the declared count so a partial catalogue cannot be read as
        # a health FAILURE it never claimed.
        reach = int(cat.get("origins_reachable", n))
        if n <= 0:
            # NOT "one source". Nothing configured here serves this input, and
            # collapsing that into a one-origin status made a signed certificate
            # assert an upstream exists for a datum nothing can fetch.
            labelled[ref] = "no-configured-source"
        elif n == 1:
            # One origin. Distinguish "several clients, one upstream" from "one
            # upstream, one client" — different facts, and only the first tells
            # you the redundancy you have is not the redundancy you need.
            clients = (cat.get("providers") or {}).get(origins[0], [])
            labelled[ref] = ("single-origin-chain" if len(clients) > 1
                             else "one-configured-origin")
        elif reach < 2:
            # Declared >= 2 but fewer than 2 can currently answer. Reported ahead
            # of a convention mismatch because it is the BINDING constraint: you
            # cannot compare measurands you cannot fetch.
            labelled[ref] = "origins-unreachable"
        elif not cat.get("comparable"):
            labelled[ref] = "incomparable-origins"
        else:
            labelled[ref] = "not-attempted"
    worst = min(labelled.values(), key=lambda s: order.get(s, -1))
    weakest = [r for r, s in labelled.items() if s == worst]
    reasons = "; ".join(f"{r}: {per_ref[r].get('reason')}" for r in weakest)
    detail = {
        "no-configured-source": (
            f"NO CONFIGURED SOURCE — {reasons}. Nothing configured in this "
            f"deployment can serve these inputs, so no cross-source evidence was "
            f"or could be gathered here. This counts sources, not upstreams that "
            f"exist in the world."),
        "one-configured-origin": (
            f"ONE CONFIGURED ORIGIN — {reasons}. Exactly one declared upstream is "
            f"configured to serve these inputs HERE, so this deployment has no "
            f"cross-source evidence about them. That is a fact about this "
            f"configuration and its symbol coverage, NOT a claim that only one "
            f"source for the datum exists."),
        "single-origin-chain": (
            f"SINGLE-ORIGIN CHAIN — {reasons}. Several providers serve these inputs "
            f"but all declare the same upstream, so they are alternative client "
            f"paths to one source, not separate observations of the market."),
        "origins-unreachable": (
            f"ORIGINS UNREACHABLE — {reasons}. More than one upstream is declared "
            f"for these inputs, but fewer than two can currently answer, so there "
            f"was nothing to compare against. DECLARED is not REACHABLE: this is "
            f"an OBSERVED condition (circuit-breaker state), which can change, "
            f"unlike the configuration it qualifies."),
        "incomparable-origins": (
            f"INCOMPARABLE ORIGINS — {reasons}. More than one upstream is "
            f"configured, but they do not report the same measurand, so "
            f"differencing them would return the convention difference rather "
            f"than any disagreement about the data. Nothing was compared, "
            f"deliberately."),
        "not-attempted": (
            f"NOT ATTEMPTED — {reasons}. No cross-source observation was recorded "
            f"for these inputs. This is silence, not agreement."),
    }.get(worst, f"{worst.upper()} — {reasons}. No cross-source evidence is claimed "
                 f"for these inputs.")
    return {"status": worst, "reason": f"{detail} {SCOPE_SENTENCE}",
            "trust": "signer-attested", "scope_note": SCOPE_SENTENCE,
            "weakest": weakest, "catalogue": per_ref}


def _provider_term(refs: List[Tuple[str, str]], data_service) -> Term:
    """The budget's view of `provider_status`.

    An INPUT-SPACE term (`space="input"`): its units are the raw data's, not the
    answer's, so it is never ranked against quantization/sampling and never enters
    the composite. See the module docstring.

    No width, by construction — a numeric provider term is only meaningful once
    observations are recorded AND propagated into the answer's units. Until then the
    term states its status and refuses to put a number beside it.
    """
    st = provider_status(refs, data_service)
    detail = st["reason"]
    if SCOPE_SENTENCE not in detail:          # the `unmeasured` paths carry it apart
        detail = f"{detail} {SCOPE_SENTENCE}"
    return Term("provider", None, st["status"], detail,
                {"catalogue": st.get("catalogue", {}),
                 "weakest": st.get("weakest", [])}, space="input")


def error_budget(src: str, data_service=None, *, fetcher=None,
                 seed: Optional[int] = None, conf: float = 0.95,
                 B: int = 1000) -> ErrorBudget:
    """The decomposed capture-to-answer error budget for a DSL program's final
    scalar. Refusal-first per term; the total is a labeled composition, never a
    single 'certified' number."""
    if seed is None:
        seed = 0xA1E10
    try:
        prog = _prune_to_output(parse(src))
    except Exception as exc:  # noqa: BLE001
        return ErrorBudget(None, True, f"program does not parse: {exc}", conf,
                           {}, None, "", None, "")

    # quantization — the DRC certificate at the matching confidence
    qc = certified_run(src, data_service, fetcher=fetcher, seed=seed,
                       alpha=1.0 - conf, strict=True)
    if qc.refused:
        quant = Term("quantization", None, "refused", qc.reason or "refused")
    else:
        quant = Term("quantization", qc.width,
                     "theorem" if qc.level_exact else "first-order",
                     f"DRC {qc.program_class} at level {qc.level:.1%}",
                     {"level": qc.level, "tier": qc.program_class})
    value = qc.base_value

    # sampling — needs the same fetched inputs the DRC used
    fetcher = fetcher or StoreCertifiedFetcher(data_service)
    from alelyon.runtime.oracle.dsl.execcert import _walk_refs
    refs = _walk_refs(prog.statements)
    fetched: Dict[Tuple[str, str], FetchedSeries] = {}
    try:
        for r in refs:
            fetched[r] = fetcher.get(*r)
    except Exception as exc:  # noqa: BLE001
        fetched = {}
    sampling = (_sampling_term(prog, fetched, conf, seed, B) if fetched
                else Term("sampling", None, "refused", "input data unavailable"))

    provider = _provider_term(refs, data_service)
    model = Term("model", None, "not-applicable",
                 "the DSL program IS the computation — no separate model-fit "
                 "uncertainty (a fitted-model answer would carry one here)")

    terms = {"quantization": quant, "sampling": sampling,
             "provider": provider, "model": model}

    # Dominant (binding) term — OUTPUT-SPACE ONLY. quantization and sampling are
    # half-widths on the answer; the provider term is a half-range on the input
    # level. Ranking them together compared dollars of price against a dimensionless
    # return and would have printed the larger raw magnitude as "the binding term".
    finite = {k: t.width for k, t in terms.items()
              if t.width is not None and t.space == "output"}
    dominant = max(finite, key=finite.get) if finite else None
    dom_note = _dominant_note(dominant, finite)

    # composite = RSS of the INDEPENDENT terms (dither ⊥ sampling); provider is a
    # different-kind floor and is reported separately, not folded in.
    indep = [terms[k].width for k in ("quantization", "sampling")
             if terms[k].width is not None]
    composite = math.sqrt(sum(w * w for w in indep)) if indep else None
    comp_note = ("root-sum-square of the INDEPENDENT quantization + sampling terms "
                 "(a composition for convenience, NOT a single certified number); "
                 "provider/model are reported separately by kind")

    return ErrorBudget(value, False, None, conf, terms, dominant, dom_note,
                       composite, comp_note)


def _dominant_note(dominant: Optional[str], finite: Dict[str, float]) -> str:
    if not dominant:
        return "no numeric term dominates (all refused/unmeasured)"
    q = finite.get("quantization")
    d = finite[dominant]
    if dominant == "quantization":
        others = sorted(k for k in finite if k != "quantization")
        if others:
            # Saying "the only measured term" while sampling carries a number is a
            # false stated reason — quantization can dominate a point-in-time
            # indicator whose sampling term is also measured.
            return ("storage quantization binds the answer (half-width "
                    f"{d:.3g}), above the other measured term(s) {others}")
        return "storage quantization is the only measured term here"
    if q is None:
        tail = "; quantization not measured here"
    elif d <= 0.0:
        # A zero-width dominant term makes the ratio undefined; saying
        # "quantization not measured" there would be a false stated reason.
        tail = f"; quantization is {q:.3g} and the dominant term is zero-width"
    else:
        ratio = q / d
        tail = (f"; quantization is {ratio:.1e}× of it "
                f"({'negligible' if ratio < 1e-2 else 'comparable'})")
    return f"{dominant} binds the answer (half-width {d:.3g}){tail}"
