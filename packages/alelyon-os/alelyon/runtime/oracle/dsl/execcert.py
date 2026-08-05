"""Certified execution — the DRC (Dither-Resampling Conformal) certificate.

Extends the certified pipeline from the DATA boundary to ARBITRARY DSL
programs: given a program over capture-certified inputs, produce a bound on
how much storage quantization can have moved the final scalar — the missing
propagation layer of the deterministic quantitative computing OS.

Mechanism (design winner of the 2026-07-22 three-way referee panel; binding
amendments applied):
  The stored value is x̂ = x + e with e ~ iid U(-Δ/2, Δ/2) per element,
  independent of the signal (certkit Lemma 0, unbounded encoder). Re-execute
  the REAL interpreter K times on x̂ + e'_k with fresh iid dither e', and form
  increment pivots D_k = f(x̂ + e'_k) − f(x̂). Because e and e' are iid from
  the SAME known distribution, f(x̂) − f(x) is exchangeable with the D_k for
  linear programs — so the m-th order statistic of |D| is a conformal bound
  with EXACT level m/(K+1). Re-executing the real interpreter (not a model of
  it) eliminates the semantics-drift failure class an analytic calculus would
  carry forever.

Honesty tiers (static AST classification — the referees' binding scheme):
  linear-exact      ops ⊆ {price, series, +, −, num·, diff, lag, sma, ema,
                    rolling_mean, sum, mean, last, first, count}: the
                    conformal level is EXACT (exchangeability is a theorem).
  smooth-first-order adds {returns, logret, ×, ÷, sqrt, zscore, std,
                    rolling_std, corr}: exchangeability holds to first order
                    at capture deltas (~1e-7 relative); the level is labeled
                    approximate and the coverage harness is the falsifier.
  branch-sensitive  {rsi (continuous but not 1-Lipschitz — steep near
                    avg_loss→0), % (mod — discontinuous at every multiple),
                    unknown ops, and a where/and/or/not applied to a
                    NON-comparison (a truthiness branch with no traced
                    margin)}: a dither-sized perturbation can flip a discrete
                    branch and no sound first-order object exists. STRICT mode
                    REFUSES the quantization term; nothing is silently degraded.
  branch-stable     SALVAGE via margin-checked branch stability (W4b Part 2
                    generalized this from min/max to the discontinuous
                    element-wise ops). The salvageable set — min/max
                    (continuous 1-Lipschitz reducers, hazard = argmin/argmax
                    flip), sign (decision x vs 0), and numeric comparisons
                    (decision a vs b, per element) — is un-refused by two
                    INDEPENDENT guards:
                      · deterministic (branch-stable-exact): EVERY decision is
                        applied DIRECTLY to certified data and is Δ-separated —
                        for min/max every competitor gap_j > (Δ_win + Δ_j)/2,
                        for sign/comparison-vs-constant every element sits
                        > Δ_i/2 from its threshold. Then |stored−true| ≤ Δ/2
                        per element (the capture certificate) makes neither the
                        true data nor any dither resample flip the decision, the
                        program IS a fixed linear selection, and the conformal
                        level is EXACT.
                      · empirical (branch-stable-first-order): derived operands
                        — the WHOLE discrete-decision vector (min/max winner, or
                        the digest of the sign/comparison pattern) must be
                        IDENTICAL across the base + all K resamples AND the
                        min per-element margin must exceed BRANCH_MARGIN_SAFETY ×
                        the observed perturbation of that margin. First-order,
                        harness-validated. `where`/`and`/`or`/`not` are
                        transparent: their stability is inherited from the
                        comparison(s) that ground them, which carry the margin.
                    The two guards are INDEPENDENT: the deterministic
                    Δ-separation theorem certifies EXACT on its own. A flip, a
                    thin margin, or an uncertified row feeding a decision REFUSES
                    unconditionally (strict AND non-strict — the strict flag
                    governs uncertified-row tolerance, not a genuine branch
                    flip), with the cause named in the reason.

Per-row Δ attribution (binding): deltas come from exact CURRENT row-to-cert-log
membership for the exact column consumed (bars→close, fred→value). Neither a
batch's min/max span nor the scope watermark can turn an unmatched row into a
certified row. Missing exact membership is UNCERTIFIED — strict mode refuses and
reports the fraction.

Named assumptions printed with every certificate: the resampling PRNG is
independent of the capture PRNG (fresh os.urandom seed, persisted for
reproducibility); the claim covers STORAGE QUANTIZATION ONLY — provider
error, finite-sample sampling error, and model error remain the answer
engine's separately named terms (at 24-bit capture, sampling error typically
dominates by ~1e4; the comparison is the consumer's job, and the answer
engine prints both).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl import nodes as N
from alelyon.runtime.oracle.dsl.interpreter import run_program
from alelyon.runtime.oracle.dsl.parser import parse, program_shape

# ── static op classification (referee-binding tiers) ─────────────────────────
LINEAR_OPS = {"price", "series", "table", "diff", "lag", "sma", "ema", "rolling_mean",
              "sum", "mean", "last", "first", "count"}
# abs and clip are CONTINUOUS and 1-Lipschitz — a dither-sized nudge moves the
# VALUE by at most that nudge (only the DERIVATIVE is kinked), so the DRC
# first-order bound is valid; they belong with the smooth ops, NOT hard-refused
# (W4b). rsi stays branch-sensitive: it is continuous but NOT 1-Lipschitz (steep
# near avg_loss→0) and can go non-finite there, so refusal is the honest default.
SMOOTH_OPS = {"returns", "logret", "sqrt", "zscore", "std", "rolling_std", "corr",
              "abs", "clip"}
# Genuinely hard-refused builtins: continuous-but-steep rsi (see above). The
# discontinuous element-wise ops (sign/where/comparisons) moved to the
# salvageable set in W4b Part 2 — they are margin-checked, not blanket-refused.
BRANCHY_OPS = {"rsi"}
# The salvageable set (W4b Part 2): discrete-decision ops whose only quantization
# hazard is a branch FLIP, guarded by the margin-checked branch-stability tests.
#   min/max      — continuous, 1-Lipschitz reducers; hazard is an argmin/argmax flip.
#   sign         — decision x_i vs 0 per element; the OUTPUT is the decision.
#   where        — selection on a numeric-comparison-grounded boolean condition.
#   comparisons  — a OP b per element; the boolean is the decision.
# where the comparison(s) that ground a boolean carry the continuous margin;
# where / and / or / not are transparent — their stability is inherited from
# those comparisons, so they emit no trace of their own.
SALVAGEABLE_CALL_OPS = {"min", "max", "sign", "where"}
_CMP_OPS = {">", "<", ">=", "<=", "==", "!="}
# Empirical guard headroom: base margin must exceed this multiple of the
# max observed margin perturbation across the K resamples (engine-set).
BRANCH_MARGIN_SAFETY = 3.0

#: Mirrored from `certify` rather than imported: `execcert` is inside the open
#: verifier's import closure and `certify` (the capture quantizer) is not, so a
#: direct import would drag the producer into the wheel. The pair is pinned by
#: `tests/oracle/test_exact_cents_capture.py`, which fails if they drift apart.
_EXACT_CENTS_LAW = "exact-cents/v0"
#: Last f64-exact integer: beyond it, consecutive integers are not representable.
_SAFE_EXACT_INT = 2.0 ** 53
# Verification accepts untrusted envelopes. Bound replay therefore needs a hard
# work limit before an attacker-controlled K can drive an effectively unbounded
# loop. This is deliberately far above the production default (63).
MAX_REPLAY_RESAMPLES = 10_000
# Aggregate work includes the base execution plus every dither replay, every
# consumed row, and every reachable AST node.  A limit on K alone is not a work
# limit when an attacker also controls input length and program complexity.
MAX_REPLAY_WORK_UNITS = 50_000_000

#: The data-reading builtins. A ref-bearing call, so every walker that collects or
#: scopes inputs must agree on the set — listing them inline at three sites is how a
#: new kind gets certified at one site and missed at another.
_DATA_READS = ("price", "series", "table")

#: Tier ordering for the `require_tier` floor: a program may be certified at its own
#: tier or anything STRICTER, never weaker. The branch-stable tiers are salvages of
#: branch-sensitive programs, so they rank with it for admission purposes — a caller
#: demanding linear-exact does not want a margin-guarded branch either.
_TIER_FLOORS = {
    "linear-exact": 0,
    "branch-stable-exact": 1,
    "smooth-first-order": 2,
    "branch-stable-first-order": 3,
    "branch-sensitive": 4,
}

_LINEAR_BINOPS = {"+", "-"}
# '%' (mod) is DISCONTINUOUS — it jumps by the divisor at every multiple, so a
# dither-sized nudge can flip it: it is branch-sensitive, not smooth. Only the
# genuinely continuous arithmetic ops belong here.
_SMOOTH_BINOPS = {"*", "/", "^"}


# ── program pruning: scope certification to the FINAL output's dependencies ──
def _names_in(expr) -> List[str]:
    """The let-binding names an expression references (transitively resolved by
    the caller)."""
    out: List[str] = []

    def w(n) -> None:
        if isinstance(n, N.Name):
            out.append(n.id)
        elif isinstance(n, N.Call):
            for a in n.args:
                w(a)
        elif isinstance(n, N.BinOp):
            w(n.left); w(n.right)
        elif isinstance(n, N.BoolOp):
            w(n.left); w(n.right)
        elif isinstance(n, (N.Not, N.UnaryMinus)):
            w(n.operand)

    w(expr)
    return out


def _prune_to_output(prog: N.Program) -> N.Program:
    """Return a program containing only the statement that produces the FINAL
    output (the scalar certified_run bounds via outputs[-1]) plus the let-
    bindings it transitively needs. This is what makes an UNRELATED min/max in
    an unused `let` not force a refusal: a site that cannot affect the certified
    scalar is not classified, executed, or guarded."""
    out_stmt = None
    for s in reversed(prog.statements):
        if isinstance(s, (N.Show, N.Signal)):
            out_stmt = s
            break
    if out_stmt is None:
        return prog
    # both let AND signal bind a name the output may reference; a bound
    # dependency must be kept even if it is itself an earlier output.
    binds = {s.name: s for s in prog.statements
             if isinstance(s, (N.Let, N.Signal)) and s is not out_stmt}
    needed: set = set()
    stack = [out_stmt.expr]
    while stack:
        for nm in _names_in(stack.pop()):
            if nm in binds and nm not in needed:
                needed.add(nm)
                stack.append(binds[nm].expr)
    kept = [s for s in prog.statements
            if s is not out_stmt and getattr(s, "name", None) in needed]
    kept.append(out_stmt)
    return N.Program(statements=kept)


def _stmt_exprs(statements) -> List:
    return [s.expr for s in statements if getattr(s, "expr", None) is not None]


def _binds_of(statements) -> Dict[str, object]:
    """name → bound expression (let/signal), for resolving a boolean condition
    that was introduced via a `let`."""
    return {s.name: s.expr for s in statements
            if isinstance(s, (N.Let, N.Signal)) and getattr(s, "expr", None) is not None}


def _str_cmp(node) -> bool:
    """A comparison with a string-literal operand — data-INDEPENDENT (a dither
    perturbation cannot move it), so it is NOT a branch hazard, emits no margin
    trace, and must not be classified salvageable (else a site with no trace would
    make the empirical guard vacuously pass and print a false 'decisions observed'
    claim). The interpreter already special-cases string ==/!= as constant."""
    return (isinstance(node, N.BinOp) and node.op in _CMP_OPS
            and (isinstance(node.left, N.Str) or isinstance(node.right, N.Str)))


def _is_boolean_expr(node, binds: Dict[str, object]) -> bool:
    """True iff `node` is a boolean-valued expression grounded ENTIRELY in numeric
    comparisons — so every discrete branch it encodes emits a margin trace and is
    margin-guarded. A raw numeric series used as a truthiness test is NOT boolean
    here: `where(price("X"), a, b)` would branch on price≠0 with no traced margin,
    which is exactly the unsound case this gate rejects. A data-independent
    string-literal comparison is likewise not admitted (it carries no margin).
    `and/or/not` are boolean iff their operands are; a `let`-bound name is
    resolved through `binds`."""
    if isinstance(node, N.BinOp):
        return node.op in _CMP_OPS and not _str_cmp(node)
    if isinstance(node, N.BoolOp):
        return (_is_boolean_expr(node.left, binds)
                and _is_boolean_expr(node.right, binds))
    if isinstance(node, N.Not):
        return _is_boolean_expr(node.operand, binds)
    if isinstance(node, N.Name):
        b = binds.get(node.id)
        return _is_boolean_expr(b, binds) if b is not None else False
    return False


def _walk_ops(statements) -> Tuple[List[str], List[str], List[str]]:
    """(hard branch ops, salvageable branch ops, smooth ops) over the given
    statements. Salvageable = min/max/sign/where + numeric comparisons; each is
    still classified branch-sensitive by classify_program — certified_run may
    upgrade them via the margin-checked branch-stability guards."""
    smooth: List[str] = []
    hard: List[str] = []
    salv: List[str] = []
    binds = _binds_of(statements)

    def walk(node) -> None:
        if isinstance(node, N.Call):
            name = node.func
            if name in ("min", "max", "sign"):
                salv.append(name)
            elif name == "where":
                # salvageable only when the condition is comparison-grounded;
                # otherwise it is an untraceable truthiness branch → hard.
                if node.args and _is_boolean_expr(node.args[0], binds):
                    salv.append("where")
                else:
                    hard.append("where(non-boolean condition)")
            elif name in SMOOTH_OPS:
                smooth.append(name)
            elif name in BRANCHY_OPS:
                hard.append(name)
            elif name not in LINEAR_OPS:
                hard.append(name)                  # unknown op: fail closed
            for a in node.args:
                walk(a)
        elif isinstance(node, N.BinOp):
            if node.op in _SMOOTH_BINOPS:
                smooth.append(node.op)
            elif node.op in _CMP_OPS:
                if not _str_cmp(node):
                    salv.append(node.op)           # margin-guarded comparison
                # a string-literal comparison is data-independent → not a branch
            elif node.op not in _LINEAR_BINOPS:    # '%' etc.
                hard.append(node.op)
            walk(node.left)
            walk(node.right)
        elif isinstance(node, N.BoolOp):
            # transparent: stability is inherited from the comparison operands'
            # traces. A boolean combination of non-comparisons is a truthiness
            # branch with no traced margin → hard.
            if not _is_boolean_expr(node, binds):
                hard.append("bool(non-boolean operand)")
            walk(node.left)
            walk(node.right)
        elif isinstance(node, N.Not):
            if not _is_boolean_expr(node, binds):
                hard.append("not(non-boolean operand)")
            walk(node.operand)
        elif isinstance(node, N.UnaryMinus):
            walk(node.operand)

    for expr in _stmt_exprs(statements):
        walk(expr)
    return hard, salv, smooth


def _classify_detail(src: str) -> Tuple[List[str], List[str], List[str]]:
    """(hard branch ops, salvageable branch ops (min/max), smooth ops)."""
    return _walk_ops(parse(src).statements)


def classify_program(src: str) -> Tuple[str, List[str]]:
    """('linear-exact' | 'smooth-first-order' | 'branch-sensitive',
    [offending op names]). min/max/sign/where and numeric comparisons classify
    branch-sensitive here (they ARE); certified_run may still upgrade them via the
    margin-checked branch-stability guards."""
    hard, salv, smooth = _classify_detail(src)
    if hard or salv:
        return "branch-sensitive", sorted(set(hard + salv))
    if smooth:
        return "smooth-first-order", sorted(set(smooth))
    return "linear-exact", []


def _read_ref(node) -> Optional[Tuple[str, str]]:
    """The (kind, key) of a direct price()/series() read, else None."""
    if (isinstance(node, N.Call) and node.func in _DATA_READS
            and node.args and isinstance(node.args[0], N.Str)):
        return (node.func, str(node.args[0].value))
    return None


def _const_val(node) -> Optional[float]:
    """The numeric value of a literal (or a negated literal), else None."""
    if isinstance(node, N.Num):
        return float(node.value)
    if isinstance(node, N.UnaryMinus) and isinstance(node.operand, N.Num):
        return -float(node.operand.value)
    return None


def _walk_sites(statements) -> List[Dict]:
    """Every margin-guarded branch DECISION site, in a uniform shape
    {op, kind, det}. `det`, when not None, carries what the deterministic
    Δ-separation theorem needs to certify that site EXACT:
        ('extremum', ref)      min/max applied DIRECTLY to a read
        ('sign', ref)          sign applied DIRECTLY to a read
        ('cmp', ref, const)    a read compared to a numeric constant
    det=None ⇒ only the empirical guard can upgrade the site (derived operands,
    read-vs-read comparisons, etc.). `where` is transparent (no site of its own);
    its condition's comparison IS a site, so recursion still covers it."""
    sites: List[Dict] = []

    def walk(node) -> None:
        if isinstance(node, N.Call):
            if node.func in ("min", "max"):
                ref = _read_ref(node.args[0]) if node.args else None
                sites.append({"op": node.func, "kind": "extremum",
                              "det": ("extremum", ref) if ref else None})
            elif node.func == "sign":
                ref = _read_ref(node.args[0]) if node.args else None
                sites.append({"op": "sign", "kind": "sign",
                              "det": ("sign", ref) if ref else None})
            for a in node.args:
                walk(a)
        elif isinstance(node, N.BinOp):
            if node.op in _CMP_OPS and not _str_cmp(node):
                lr, rr = _read_ref(node.left), _read_ref(node.right)
                lc, rc = _const_val(node.left), _const_val(node.right)
                det = None
                if lr is not None and rc is not None:
                    det = ("cmp", lr, rc)
                elif rr is not None and lc is not None:
                    det = ("cmp", rr, lc)
                sites.append({"op": node.op, "kind": "cmp", "det": det})
            walk(node.left); walk(node.right)
        elif isinstance(node, N.BoolOp):
            walk(node.left); walk(node.right)
        elif isinstance(node, (N.Not, N.UnaryMinus)):
            walk(node.operand)

    for expr in _stmt_exprs(statements):
        walk(expr)
    return sites


def _extremum_sites(src: str) -> List[Dict]:
    return _walk_sites(parse(src).statements)


def _walk_refs(statements) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = []

    def walk(node) -> None:
        if isinstance(node, N.Call):
            if node.func in _DATA_READS and node.args \
                    and isinstance(node.args[0], N.Str):
                refs.append((node.func, str(node.args[0].value)))
            for a in node.args:
                walk(a)
        elif isinstance(node, N.BinOp):
            walk(node.left); walk(node.right)
        elif isinstance(node, N.BoolOp):
            walk(node.left); walk(node.right)
        elif isinstance(node, (N.Not, N.UnaryMinus)):
            walk(node.operand)

    for expr in _stmt_exprs(statements):
        walk(expr)
    seen = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def data_refs(src: str) -> List[Tuple[str, str]]:
    """[('price', 'AAPL'), ('series', 'DGS10'), ...] — literal data reads."""
    return _walk_refs(parse(src).statements)


# ── certified fetching (per-row delta attribution) ───────────────────────────
@dataclass
class FetchedSeries:
    series: pd.Series          # the stored (certified) values, as normally served
    deltas: np.ndarray         # per-row delta; NaN = uncertified row
    uncertified: int
    #: The CAPTURE LAW that produced `deltas`, when known (None = the relative-dither
    #: law, or simply unattributed). Carried because Δ=0 means different things under
    #: different laws, and because the exact-cents law brings an AGGREGATE
    #: representability obligation that per-row Δ cannot express — see the 2^53 guard
    #: in `certified_run`.
    law: Optional[str] = None


class StoreCertifiedFetcher:
    """Resolves price/series through the SAME paths the normal DataContext
    uses (value parity with an uncertified run is exact), then attributes a
    per-row delta only from exact current row-to-cert-log membership."""

    def __init__(self, data_service) -> None:
        self._ds = data_service

    def get(self, kind: str, key: str) -> FetchedSeries:
        if kind == "table":
            # Keyed-table DATA lives with the actuarial ingestion adapters (product
            # #1 Phase 3), not in the market-data store this fetcher reads. Refuse
            # with the reason rather than falling through to the series path, which
            # would look up an unrelated FRED scope.
            raise ValueError(
                f"no store-side reader for keyed-table input {key!r}: supply a "
                f"fetcher (the ingestion adapter) — the market-data store holds no "
                f"tables")
        if kind == "price":
            from alelyon.runtime.vector.risk.covariance import _close
            s = _close(self._ds.bars(str(key)))
            if s is None or len(s) == 0:
                raise ValueError(f"no price history for {key!r}")
            s = s.rename(str(key).upper())
            plan = self._ds.store.cert_batches("bars", str(key).upper(), "1d")
            col = "close"
        else:
            s = self._ds.fred_series(str(key))
            if s is None or len(s) == 0:
                raise ValueError(f"no series data for {key!r}")
            s = s.rename(str(key).upper())
            plan = self._ds.store.cert_batches("series", "fred", str(key).upper())
            col = "value"
        deltas = np.full(len(s), np.nan)
        ts = np.array([pd.Timestamp(i).timestamp() for i in s.index])
        attributed = {}
        for batch in plan["batches"]:
            if col not in batch["columns"]:
                continue
            delta = batch["columns"][col]
            for member_ts in batch.get("rows", ()):
                prior = attributed.get(member_ts)
                attributed[member_ts] = (delta if prior is None
                                         else max(prior, delta))
        for i, t in enumerate(ts):
            best = attributed.get(t)
            if best is not None:
                deltas[i] = best
        return FetchedSeries(series=s, deltas=deltas,
                             uncertified=int(np.isnan(deltas).sum()))


class _FixedContext:
    """DataContext stand-in serving pre-fetched (optionally perturbed) series.
    When `trace` is a list, min/max builtins append (op, winner label, margin)
    to it — the branch-stability observable."""

    def __init__(self, table: Dict[Tuple[str, str], pd.Series],
                 trace: Optional[list] = None) -> None:
        self._t = table
        self.branch_trace = trace

    def price(self, ticker: str) -> pd.Series:
        return self._t[("price", str(ticker))].copy()

    def series(self, series_id: str) -> pd.Series:
        return self._t[("series", str(series_id))].copy()

    def table(self, key: str) -> pd.Series:
        return self._t[("table", str(key))].copy()


def _delta_separated(op: str, fs: FetchedSeries) -> bool:
    """Deterministic no-flip guard for an extremum applied DIRECTLY to
    certified data: every competitor j must beat the capture bound,
    gap_j > (Δ_win + Δ_j)/2. Then |stored−true| ≤ Δ/2 per element (the capture
    certificate) makes a winner flip impossible for the true data AND for
    every dither resample. Requires all consumed rows certified (finite Δ)."""
    v = fs.series.to_numpy(dtype=np.float64)
    d = fs.deltas
    m = np.isfinite(v)
    if not np.all(np.isfinite(d[m])):
        return False                                # uncertified row in play
    v, d = v[m], d[m]
    if len(v) == 0:
        return False
    if len(v) == 1:
        return True
    i = int(np.argmin(v) if op == "min" else np.argmax(v))
    gap = (v - v[i]) if op == "min" else (v[i] - v)
    guard = gap - (d + d[i]) / 2.0
    guard[i] = np.inf
    return bool(np.min(guard) > 0.0)


def _elementwise_separated(fs: FetchedSeries, threshold) -> bool:
    """Deterministic no-flip guard for an element-wise discrete decision applied
    DIRECTLY to certified data: sign(x) is the decision x vs 0, a comparison x OP c
    is the decision x vs the constant c. Each element is stable iff its distance to
    the threshold exceeds the capture bound Δ_i/2 — then |stored−true| ≤ Δ_i/2 (the
    capture certificate) makes neither the true data nor any dither resample flip
    that decision, so the whole discrete-decision vector is EXACT. Requires all
    consumed rows certified (finite Δ)."""
    v = fs.series.to_numpy(dtype=np.float64)
    d = fs.deltas
    m = np.isfinite(v)
    if not np.all(np.isfinite(d[m])):
        return False                                # uncertified row in play
    v, d = v[m], d[m]
    if len(v) == 0:
        return False
    return bool(np.all(np.abs(v - float(threshold)) > d / 2.0))


def _site_delta_separated(site: Dict, fetched: Dict[Tuple[str, str], FetchedSeries]) -> bool:
    """Dispatch a site's deterministic Δ-separation check (EXACT tier eligibility)."""
    det = site.get("det")
    if det is None:
        return False
    if det[0] == "extremum":
        ref = det[1]
        return ref is not None and ref in fetched and _delta_separated(site["op"], fetched[ref])
    if det[0] == "sign":
        ref = det[1]
        return ref is not None and ref in fetched and _elementwise_separated(fetched[ref], 0.0)
    if det[0] == "cmp":
        _, ref, const = det
        return ref is not None and ref in fetched and _elementwise_separated(fetched[ref], const)
    return False


# ── the certificate ───────────────────────────────────────────────────────────
@dataclass
class ExecCertificate:
    ok: bool
    refused: bool
    reason: Optional[str]
    program_class: str
    level: Optional[float]         # m/(K+1); EXACT only for linear-exact
    level_exact: bool
    width: Optional[float]         # the conformal |D| bound
    base_value: Optional[float]
    K: int
    seed: Optional[int]
    uncertified_fraction: float
    assumptions: List[str] = field(default_factory=list)
    note: str = ""
    branch_sites: List[Dict] = field(default_factory=list)


def _all_integral(series: pd.Series) -> bool:
    """True iff every finite value is an exact integer inside f64's exact-integer
    range — the signature of a count-valued input such as whole cents. Derived from
    the data rather than from a declared law, so the producer and the verifier reach
    the same answer without either trusting the other."""
    v = np.asarray(series.to_numpy(dtype=np.float64))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return False
    return bool(np.all(np.abs(v) <= _SAFE_EXACT_INT) and np.all(v == np.rint(v)))


def _scalar_of(res) -> Optional[float]:
    outs = getattr(res, "outputs", None)
    if not outs:
        return None
    v = outs[-1].value
    if isinstance(v, pd.Series):
        v = v.dropna().iloc[-1] if len(v.dropna()) else None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def certified_run(src: str, data_service=None, *, alpha: float = 0.05,
                  K: int = 63, seed: Optional[int] = None, strict: bool = True,
                  fetcher=None,
                  require_tier: Optional[str] = None) -> ExecCertificate:
    """Execute `src` and produce the storage-quantization certificate for its
    final scalar. Refusal-first: any precondition failure yields refused=True
    with the reason named — never a fabricated bound."""
    def _refuse(reason: str, cls: str = "?", uf: float = 0.0) -> ExecCertificate:
        return ExecCertificate(ok=False, refused=True, reason=reason,
                               program_class=cls, level=None, level_exact=False,
                               width=None, base_value=None, K=K, seed=seed,
                               uncertified_fraction=uf)

    if not (0.0 < alpha < 1.0):
        return _refuse(f"alpha must be in (0,1), got {alpha}")
    if require_tier is not None and require_tier not in _TIER_FLOORS:
        return _refuse(f"require_tier {require_tier!r} is not one of "
                       f"{sorted(_TIER_FLOORS)}")

    try:
        prog = _prune_to_output(parse(src))       # scope to the final output only
        ast_nodes, _ = program_shape(prog)
        hard_ops, salv_ops, smooth_ops = _walk_ops(prog.statements)
    except Exception as exc:  # noqa: BLE001
        return _refuse(f"program does not parse: {exc}")
    cls = ("branch-sensitive" if (hard_ops or salv_ops)
           else "smooth-first-order" if smooth_ops else "linear-exact")

    # ── the opt-in TIER FLOOR ─────────────────────────────────────────────────
    # A caller can demand a tier and get a REFUSAL rather than a weaker
    # certificate. Without this, a program that strays out of the exact tier is
    # silently certified one tier down: `sum(a)/sum(b)` is smooth, so it earns a
    # first-order bound with an approximate level, which is correct in general and
    # wrong for a product whose whole claim is that the level is a theorem
    # (docs/cne/products/01-insurance-actuarial.md: "the MVP mandate is
    # linear-exact only"). Enforcement has to live here, not in the template
    # layer, because the case that matters is the HAND-FED program that bypassed
    # the templates.
    #
    # Carried in the envelope's `params` so replay reproduces the refusal; a
    # verifier that ignored it would fail to reproduce and fail closed.
    if require_tier is not None and _TIER_FLOORS[cls] > _TIER_FLOORS[require_tier]:
        offenders = sorted(set(hard_ops + salv_ops + smooth_ops))
        return _refuse(
            f"program classifies {cls!r} but this issuer requires "
            f"{require_tier!r}: {', '.join(offenders) or 'unknown op'}. Outside "
            f"{require_tier!r} the conformal level is no longer exact, so the "
            f"result would carry a weaker claim than the one asked for", cls)

    if hard_ops and strict:
        return _refuse("branch-sensitive ops — a dither-sized perturbation can "
                       f"flip a discrete branch, no sound bound exists: "
                       f"{', '.join(sorted(set(hard_ops)))}", cls)
    # min/max alone are salvageable: run with branch tracing and let the
    # margin guards decide, instead of refusing statically.
    trace_branches = bool(salv_ops) and not hard_ops

    refs = _walk_refs(prog.statements)
    if not refs:
        return _refuse("program reads no certified data", cls)
    fetcher = fetcher or StoreCertifiedFetcher(data_service)
    fetched: Dict[Tuple[str, str], FetchedSeries] = {}
    try:
        for r in refs:
            fetched[r] = fetcher.get(*r)
    except Exception as exc:  # noqa: BLE001
        return _refuse(f"data fetch failed: {exc}", cls)

    n_rows = sum(len(f.series) for f in fetched.values())
    n_unc = sum(f.uncertified for f in fetched.values())
    uf = n_unc / n_rows if n_rows else 1.0
    if K > MAX_REPLAY_RESAMPLES:
        return _refuse(
            f"K={K} exceeds the verifier limit of "
            f"{MAX_REPLAY_RESAMPLES} resamples", cls, uf)
    replay_work = (K + 1) * n_rows * ast_nodes
    if replay_work > MAX_REPLAY_WORK_UNITS:
        return _refuse(
            f"replay work {replay_work} exceeds the verifier limit of "
            f"{MAX_REPLAY_WORK_UNITS} units (rows={n_rows}, K={K}, "
            f"AST nodes={ast_nodes})", cls, uf)
    if strict and n_unc > 0:
        return _refuse(f"{n_unc}/{n_rows} consumed rows carry no capture "
                       f"certificate (no exact current cert-log membership) — a "
                       f"bound conditional on 'uncertified rows are exact' "
                       f"would be theater", cls, uf)

    # Non-strict uncertified handling: an uncertified row (NaN Δ) must NOT be
    # modeled as zero storage error — that would collapse the conformal width to
    # a vacuous (under-covering) bound. Instead perturb it at the COARSEST Δ this
    # input actually carries (a data-grounded conservative proxy from its
    # certified rows). If an input is FULLY uncertified there is no Δ to stand in, so
    # even non-strict must refuse — a width there would be fabricated.
    eff_deltas: Dict[Tuple[str, str], np.ndarray] = {}
    for r, f in fetched.items():
        d = np.asarray(f.deltas, dtype=np.float64).copy()
        nan = np.isnan(d)
        if nan.any():
            finite = d[~nan]
            if finite.size == 0:
                return _refuse(f"input {r} is fully uncertified — no capture Δ to "
                               f"bound its storage error against", cls, uf)
            d[nan] = float(np.max(finite))     # coarsest certified-row Δ proxy
        eff_deltas[r] = d

    base_table = {r: f.series for r, f in fetched.items()}
    base_trace: list = [] if trace_branches else None
    base_res = run_program(src, program=prog,
                           ctx=_FixedContext(base_table, trace=base_trace))
    base_val = _scalar_of(base_res)
    if not getattr(base_res, "ok", False) or base_val is None:
        return _refuse(f"base execution failed: {getattr(base_res, 'error', '?')}",
                       cls, uf)

    # ── the exact-storage AGGREGATE guard ─────────────────────────────────────
    # Per-element exactness does not imply the RESULT is exact. Capture refuses any
    # single value beyond 2^53 cents (~$90 trillion), but a control total over many
    # values can pass every per-element check and still land where f64 has stopped
    # counting by ones — at which point `sum` returns a rounded number while the
    # certificate claims Δ=0, i.e. "stored exactly", and the receipt reads as a
    # penny-exact tie-out that is nothing of the kind. This is the
    # element-vs-aggregate lesson in its arithmetic form: a guard that binds each
    # element is not a guard on their sum.
    #
    # Keyed on the DATA (Δ ≡ 0 over whole-integer values), not on a declared capture
    # law. The law is only recoverable from a proven capture leaf, so gating on it
    # made this guard — and the assumption line below — fire at issuance and not on
    # replay, which then disagreed about the signed `assumptions` and failed a
    # legitimate unanchored envelope on `budget-mismatch`. Δ and the values are
    # committed, so both sides see them identically.
    #
    # Refusal, not silent rounding: an actuarial tie-out that cannot be represented
    # is exactly the case where guessing is worst.
    exact_storage = bool(n_rows) and all(
        np.all(np.asarray(f.deltas, dtype=np.float64) == 0.0)
        for f in fetched.values())
    integral_inputs = exact_storage and all(
        _all_integral(f.series) for f in fetched.values())
    if integral_inputs and abs(base_val) > _SAFE_EXACT_INT:
        return _refuse(
            f"a result of {base_val!r} over exactly-stored integer inputs exceeds "
            f"2^53 (~$90 trillion in cents): beyond this f64 no longer represents "
            f"consecutive integers, so the value cannot be claimed exact even "
            f"though every input element was — the guard binds the aggregate, "
            f"not only the elements", cls, uf)

    # Defensive honesty guard: if the ops classified salvageable turned out to
    # emit NO branch decision at runtime (a data-independent op), never claim a
    # margin guard ran on zero observed decisions — drop the branch tier and fall
    # back to the honest non-branch class. The string-literal-comparison
    # classification makes this unreachable in practice; it is belt-and-suspenders.
    if trace_branches and not base_trace:
        trace_branches = False
        cls = "smooth-first-order" if smooth_ops else "linear-exact"

    m = math.ceil((1.0 - alpha) * (K + 1))
    if m > K:
        return _refuse(f"K={K} too small for alpha={alpha} "
                       f"(need K >= {math.ceil((1 - alpha) / alpha)})", cls, uf)

    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little") >> 1   # A1: independent PRNG

    # Branch-stability accounting is streamed across the K resamples so it stays
    # O(#sites × n) not O(K × #sites × n): sig0 fixes the decision signature;
    # base_mvec[i] holds a site's per-element margin vector (None for min/max,
    # which carry a single scalar gap); run_pert[i] accumulates the WORST
    # per-element perturbation of that site's margin across resamples — element-wise
    # for a vector site (so a benign near-threshold element cannot mask a flippable
    # high-Δ one), scalar for min/max.
    sig0 = [(op, str(key)) for op, key, _, _ in base_trace] if trace_branches else []
    base_mmin = [mm for _, _, mm, _ in base_trace] if trace_branches else []
    base_mvec = [mv for _, _, _, mv in base_trace] if trace_branches else []
    run_pert = [0.0] * len(base_trace) if trace_branches else []
    stable = True
    flip_source: Optional[str] = None   # 'dither' | 'systematic', for the reason text
    devs: List[float] = []

    def _probe(table: Dict[Tuple[str, str], pd.Series], source: str):
        """Run the program on one perturbed table and fold the result into the
        branch-stability accounting. Returns (scalar, error) — never raises.

        Both the decision signature and the per-site margin perturbation are
        updated here, so every caller's perturbation counts toward the guard
        regardless of which law produced it.
        """
        nonlocal stable, flip_source
        trace_k: list = [] if trace_branches else None
        res_k = run_program(src, program=prog,
                            ctx=_FixedContext(table, trace=trace_k))
        v_k = _scalar_of(res_k)
        if not getattr(res_k, "ok", False) or v_k is None:
            return None, str(getattr(res_k, "error", "non-finite output"))
        if trace_branches:
            if (len(trace_k) != len(base_trace)
                    or [(op, str(key)) for op, key, _, _ in trace_k] != sig0):
                stable = False                     # a decision flipped/changed shape
                if flip_source is None:
                    flip_source = source
            else:
                for i, (_, _, mmk, mvk) in enumerate(trace_k):
                    mv0 = base_mvec[i]
                    if mv0 is not None and mvk is not None:
                        fin = np.isfinite(mv0) & np.isfinite(mvk)
                        d_i = (float(np.max(np.abs(mvk[fin] - mv0[fin])))
                               if bool(fin.any()) else 0.0)
                    else:                          # min/max scalar gap
                        d_i = abs(float(mmk) - float(base_mmin[i]))
                    if d_i > run_pert[i]:
                        run_pert[i] = d_i
        return v_k, None

    def _shifted(offset_of) -> Dict[Tuple[str, str], pd.Series]:
        table: Dict[Tuple[str, str], pd.Series] = {}
        for r, f in fetched.items():
            vals = f.series.to_numpy(dtype=np.float64, copy=True)
            table[r] = pd.Series(vals + offset_of(r, len(vals)),
                                 index=f.series.index, name=f.series.name)
        return table

    for k in range(K):
        rng = np.random.default_rng([seed, k])
        table = _shifted(
            lambda r, n, _rng=rng: _rng.uniform(-0.5, 0.5, size=n) * eff_deltas[r])
        v_k, err = _probe(table, "dither")
        if err is not None:
            return _refuse(f"perturbed run {k + 1}/{K} failed ({err}) — "
                           f"the program is unstable at dither scale", cls, uf)
        devs.append(abs(v_k - base_val))

    # ── worst-case systematic probes (the branch guard only) ──────────────────
    # The K draws above are INDEPENDENT per element, which is a property of the
    # dither law and not of the declaration. On an aggregate of n rows independent
    # error cancels — the observed spread of a mean scales as Δ/√(12n) — while a
    # systematic rounding obeying the very same per-element promise |stored − true|
    # ≤ Δ/2 moves that mean by Δ/2. The gap grows like √(3n), so a fixed safety
    # factor is defeated by making the series longer, and `sign(mean(x) − c)` could
    # certify at width 0.0 with the decision inverted.
    #
    # What the declaration actually gives us is the per-element bound, so the guard
    # is measured against THAT rather than against the spread of an assumed law —
    # CLAIMS.md §2 rule 3 (validate against an independently-held invariant, never
    # against the shape of what the writer emitted) applied one level up, to the
    # resampler instead of to Δ itself.
    #
    # These probes feed the branch guard ONLY. `devs`, and therefore the certified
    # width, remain a conformal statistic over the dither law and are untouched:
    # a worst-case corner is not a draw from that law and must not be ordered with
    # its samples. Adding probes can only raise `run_pert` and can only clear
    # `stable`, so this is monotonically stricter — nothing previously refused
    # becomes admissible.
    if trace_branches:
        for sign, label in ((1.0, "+Δ/2"), (-1.0, "−Δ/2")):
            table = _shifted(lambda r, n, s=sign: s * 0.5 * eff_deltas[r])
            _v, err = _probe(table, "systematic")
            if err is not None:
                return _refuse(
                    f"the worst-case systematic probe ({label} on every element) "
                    f"failed ({err}) — the program is not evaluable across the "
                    f"range its own declaration permits", cls, uf)

    # ── margin-checked branch stability (min/max + sign/where/comparisons) ────
    # A failed branch guard ALWAYS refuses (both strict and non-strict): the
    # `strict` flag governs tolerance of uncertified ROWS, not of a genuine
    # branch flip, which is a distinct soundness failure. The two guards are
    # independent — the deterministic Δ-separation theorem certifies EXACT on
    # its own, without needing the empirical margin heuristic to also pass. Each
    # traced branch op (min/max, sign, comparison) carries a decision KEY (the
    # winner label, or a digest of the whole element-wise decision vector), a MIN
    # margin, and (for the element-wise ops) the full margin vector; the whole
    # decision must be identical across every resample and the min margin must
    # clear BRANCH_MARGIN_SAFETY × the WORST per-element perturbation.
    final_cls = cls
    branch_sites: List[Dict] = []
    branch_tier = None                # None | 'deterministic' | 'empirical'
    if trace_branches:
        margins0 = base_mmin
        pert_scale = run_pert
        # deterministic guard (EXACT): a standalone theorem — EVERY branch site is
        # applied directly to certified data with a Δ-separated decision (min/max
        # winner, sign vs 0, or a comparison vs a constant), so the true and every
        # resampled decision provably equal the stored one and the program reduces
        # to a fixed linear selection.
        sites = _walk_sites(prog.statements)
        det_ok = (n_unc == 0 and not smooth_ops and len(sites) > 0
                  and all(s["det"] is not None for s in sites)
                  and all(_site_delta_separated(s, fetched) for s in sites))
        # empirical guard (first-order): no uncertified row in play, every branch
        # decision identical across every resample, and every margin comfortably
        # above the observed perturbation of that margin.
        emp_ok = (n_unc == 0 and stable and all(
            mg == math.inf or (mg > 0.0 and mg > BRANCH_MARGIN_SAFETY * p)
            for mg, p in zip(margins0, pert_scale)))
        upgraded = det_ok or emp_ok
        for i, (op, lbl, mg, _mv) in enumerate(base_trace):
            branch_sites.append({
                "op": op, "winner": str(lbl),
                "margin": mg if math.isfinite(mg) else None,
                "perturb_scale": pert_scale[i], "stable": stable,
                "guard": ("deterministic" if det_ok
                          else "empirical" if emp_ok else "failed"),
            })
        if not upgraded:
            if n_unc > 0:
                reason = (f"branch decision rests on {n_unc}/{n_rows} uncertified "
                          f"rows — a discrete branch (min/max argmin, sign, or "
                          f"comparison) depends on values with no capture bound; "
                          f"freezing them would manufacture stability")
            elif not stable:
                reason = (
                    ("a branch decision flipped under a worst-case systematic "
                     "rounding (every element at ±Δ/2, which the declaration "
                     "permits) — a discrete branch (min/max argmin, sign, or "
                     "comparison) is not Δ-separated. Independent dither alone "
                     "would not have found this: on an aggregate it cancels")
                    if flip_source == "systematic" else
                    ("a branch decision flipped across dither resamples — a "
                     "discrete branch (min/max argmin, sign, or comparison) "
                     "is not Δ-separated"))
            else:
                worst = min(
                    ((mg, p) for mg, p in zip(margins0, pert_scale)
                     if not (mg == math.inf
                             or (mg > 0.0 and mg > BRANCH_MARGIN_SAFETY * p))),
                    key=lambda t: (t[0] - BRANCH_MARGIN_SAFETY * t[1]),
                    default=(0.0, 0.0))
                reason = (
                    "branch decision has no positive margin (empty, all-NaN, or "
                    "exact tie) — a flip cannot be ruled out"
                    if worst[0] <= 0.0 else
                    f"branch margin too thin: {worst[0]:.3g} vs perturbation "
                    f"scale {worst[1]:.3g} (need > {BRANCH_MARGIN_SAFETY:g}×) — "
                    f"a branch flip at dither scale cannot be ruled out")
            out = _refuse(reason, cls, uf)
            out.branch_sites = branch_sites
            return out
        branch_tier = "deterministic" if det_ok else "empirical"
        final_cls = "branch-stable-exact" if det_ok else "branch-stable-first-order"

    width = float(sorted(devs)[m - 1])
    level = m / (K + 1)
    exact = n_unc == 0 and (
        cls == "linear-exact" or final_cls == "branch-stable-exact")
    assumptions = [
        "resampling PRNG independent of capture PRNG (fresh seed, persisted)",
        "covers STORAGE QUANTIZATION only — sampling/provider/model error are "
        "separate terms",
    ]
    if branch_tier == "deterministic":
        assumptions.append(
            "every branch decision is Δ-separated: the min/max winner's gap beats "
            "(Δ_win+Δ_j)/2 and each sign/comparison sits > Δ/2 from its threshold, "
            "so the true decision provably equals the stored one and the program "
            "reduces to a fixed linear selection")
    elif branch_tier == "empirical":
        assumptions.append(
            "branch decisions (min/max argmin, sign, comparison) observed IDENTICAL "
            f"across base + all K dither resamples AND both worst-case systematic "
            f"probes (every element at ±Δ/2), with margin > "
            f"{BRANCH_MARGIN_SAFETY:g}× the largest perturbation any of them "
            "produced. The systematic probes are what make this survive an "
            "aggregate: independent dither cancels over n rows while a systematic "
            "rounding within the same per-element bound does not. Residual "
            "branch-flip risk is first-order and harness-validated, NOT a theorem — "
            "the probes are the two uniform-sign corners, and a mixed-sign "
            "perturbation that moves a non-monotone program further is UNMEASURED")
    if exact_storage:
        # Δ=0 here is genuine, and precisely because it is, the reader has to be
        # told what it does NOT cover. A zero storage-quantization term is not a
        # claim that the answer is exact: `mean` divides, and that division rounds.
        # Naming the boundary is the difference between "certified exact storage"
        # and a bare "certified" — the thing CLAIMS.md forbids.
        assumptions.append(
            "every consumed input was stored EXACTLY (Δ = 0 on every row — e.g. the "
            "exact-cents monetary capture law), so the storage-quantization term is "
            "genuinely zero rather than merely small. This bounds STORAGE only: "
            "floating-point rounding in the computation itself (the division in a "
            "mean, say) is a compute-side error this certificate does not cover"
            + (", and the aggregate was checked to lie within f64's exact-integer "
               "range" if integral_inputs else ""))
    if smooth_ops:
        assumptions.append("exchangeability holds to first order for smooth "
                           "ops at capture deltas; level is approximate and "
                           "harness-validated, not a theorem")
    if n_unc > 0:
        assumptions.append(
            f"{n_unc}/{n_rows} rows carry no capture certificate; non-strict mode "
            "modeled their storage error at the coarsest Δ the input carries (a "
            "certified-row proxy), NOT as zero — the width is an upper bound only "
            "under the assumption those rows are no coarser than that Δ")
    note = (f"|answer_stored − answer_true| ≤ {width:.3g} at conformal level "
            f"{level:.1%} ({'exact' if exact else 'first-order'}), over the "
            f"capture dither draw; K={K}, class={final_cls}"
            + (f", uncertified rows {n_unc}/{n_rows}" if n_unc else "")
            + (f", branches {'Δ-separated' if branch_tier == 'deterministic' else 'empirically stable'}"
               if branch_tier else ""))
    return ExecCertificate(ok=True, refused=False, reason=None,
                           program_class=final_cls, level=level,
                           level_exact=exact, width=width, base_value=base_val,
                           K=K, seed=seed, uncertified_fraction=uf,
                           assumptions=assumptions, note=note,
                           branch_sites=branch_sites)
