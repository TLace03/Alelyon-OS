"""Interpreter for the Alelyon modeling language.

Evaluates the AST against real data with a FIXED, whitelisted vocabulary — there
is no eval/exec, no attribute access, no import, no arbitrary call. A `Name`
resolves only to a bound variable; a `Call` resolves only to a builtin in the
BUILTINS table. That is what makes it safe to run researcher-authored text.

Values flow as Python floats, strings, booleans, and pandas Series (time
series). Arithmetic/comparison are Series-aware (elementwise, index-aligned);
comparisons yield boolean Series; `and/or/not` combine boolean Series
elementwise. Data enters through `price("SPY")` / `series("CPIAUCSL")`, which go
through the DataContext (DataService-backed by default, injectable for tests).
"""
from __future__ import annotations

import hashlib
import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl.nodes import (
    BinOp, BoolOp, Call, Let, Name, Not, Num, Program, Show, Signal, Str,
    UnaryMinus,
)
from alelyon.runtime.oracle.dsl.parser import parse
from alelyon.runtime.oracle.dsl.lexer import DSLSyntaxError
from alelyon.runtime.vector import native


class DSLError(Exception):
    """A runtime error in a DSL program (unknown name/function, bad argument)."""


# ── data context ──────────────────────────────────────────────────────────────

class DataContext:
    """Resolves price/series names to pandas Series via DataService. Injectable:
    any object with `price(ticker)->Series` and `series(id)->Series` works."""

    def __init__(self, data_service=None):
        self._ds = data_service
        if self._ds is None:
            try:
                from alelyon.runtime.atlas.data.service import data_service as _ds
                self._ds = _ds()
            except Exception:  # noqa: BLE001
                self._ds = None

    def price(self, ticker: str) -> pd.Series:
        if self._ds is None:
            raise DSLError("no data service available for price()")
        from alelyon.runtime.vector.risk.covariance import _close
        s = _close(self._ds.bars(str(ticker)))
        if s is None or len(s) == 0:
            raise DSLError(f"no price history for {ticker!r}")
        return s.rename(str(ticker).upper())

    def series(self, series_id: str) -> pd.Series:
        if self._ds is None:
            raise DSLError("no data service available for series()")
        s = self._ds.fred_series(str(series_id))
        if s is None or len(s) == 0:
            raise DSLError(f"no series data for {series_id!r}")
        return s.rename(str(series_id).upper())


# ── builtin helpers ───────────────────────────────────────────────────────────

def _num(v, fn: str, arg: str = "argument") -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise DSLError(f"{fn}(): {arg} must be a number, got {type(v).__name__}")
    return float(v)


def _int(v, fn: str, arg: str = "n") -> int:
    return int(round(_num(v, fn, arg)))


def _series(v, fn: str, arg: str = "argument") -> pd.Series:
    if not isinstance(v, pd.Series):
        raise DSLError(f"{fn}(): {arg} must be a series, got {type(v).__name__}")
    return v


def _arity(args, fn: str, lo: int, hi: Optional[int] = None) -> None:
    hi = lo if hi is None else hi
    if not (lo <= len(args) <= hi):
        want = f"{lo}" if lo == hi else f"{lo}-{hi}"
        raise DSLError(f"{fn}() takes {want} argument(s), got {len(args)}")


# ── branch-decision tracing (certified execution only) ────────────────────────
# The discontinuous element-wise ops — sign and the numeric comparisons — carry a
# discrete decision per element. When `ctx.branch_trace` is present, each records
# (op, decision-vector digest, MIN per-element margin): the digest lets the DRC
# layer detect a decision FLIP across dither resamples, and the min-margin (the
# smallest distance any element sits from its threshold) is what the margin guard
# checks against the observed perturbation. Values returned are bit-identical with
# or without the trace — this is pure observability, like the min/max trace.
def _decision_digest(decision_int8: np.ndarray) -> str:
    """Deterministic hash of a discrete-decision vector (int8; NaN positions carry
    a fixed sentinel). Stable across processes/machines so a branch certificate
    reproduces bit-for-bit."""
    arr = np.ascontiguousarray(decision_int8, dtype=np.int8)
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


def _trace_elementwise(tr: list, op: str, values: np.ndarray,
                       decision_int8: np.ndarray) -> None:
    """Append (op, decision digest, MIN per-element margin, per-element margin
    VECTOR). margin_i = |value_i| (values already carry the signed distance to the
    threshold); a NaN row gets a NaN margin and is excluded. The FULL vector is
    what lets the guard check EACH element's margin against its OWN perturbation —
    a lone series-wide min would let a benign near-threshold element with tiny Δ
    mask a genuinely flippable high-Δ element (a coverage hole the red team found)."""
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    margins = np.where(fin, np.abs(v), np.nan)
    mn = float(np.min(margins[fin])) if bool(fin.any()) else 0.0
    tr.append((op, _decision_digest(decision_int8), mn, margins))


# ── builtin vocabulary ────────────────────────────────────────────────────────
# Each builtin is fn(ctx, *args) — data builtins use ctx; pure ones ignore it.

def _b_price(ctx, *a):
    _arity(a, "price", 1); return ctx.price(a[0])

def _b_series(ctx, *a):
    _arity(a, "series", 1); return ctx.series(a[0])

def _b_table(ctx, *a):
    """A KEYED-TABLE read: rows identified by text keys rather than timestamps
    (a claims extract, a GL extract, a triangle's cells). Same shape as price/series
    — a pandas Series — but its index is text, so it commits under the keyed-table
    digest layout and anchors by digest identity rather than by time span."""
    _arity(a, "table", 1); return ctx.table(a[0])

def _b_returns(ctx, *a):
    _arity(a, "returns", 1, 2)
    n = _int(a[1], "returns") if len(a) > 1 else 1
    return _series(a[0], "returns").pct_change(n)

def _b_logret(ctx, *a):
    _arity(a, "logret", 1, 2)
    n = _int(a[1], "logret") if len(a) > 1 else 1
    s = _series(a[0], "logret")
    return np.log(s / s.shift(n))

def _b_diff(ctx, *a):
    _arity(a, "diff", 1, 2)
    n = _int(a[1], "diff") if len(a) > 1 else 1
    return _series(a[0], "diff").diff(n)

def _b_lag(ctx, *a):
    _arity(a, "lag", 1, 2)
    n = _int(a[1], "lag") if len(a) > 1 else 1
    return _series(a[0], "lag").shift(n)

def _b_sma(ctx, *a):
    _arity(a, "sma", 2)
    return _series(a[0], "sma").rolling(_int(a[1], "sma")).mean()

def _b_ema(ctx, *a):
    _arity(a, "ema", 2)
    return _series(a[0], "ema").ewm(span=max(_int(a[1], "ema"), 1), adjust=False).mean()

def _b_rolling_mean(ctx, *a):
    _arity(a, "rolling_mean", 2)
    return _series(a[0], "rolling_mean").rolling(_int(a[1], "rolling_mean")).mean()

def _b_rolling_std(ctx, *a):
    _arity(a, "rolling_std", 2)
    return _series(a[0], "rolling_std").rolling(_int(a[1], "rolling_std")).std()

def _b_zscore(ctx, *a):
    _arity(a, "zscore", 1, 2)
    s = _series(a[0], "zscore")
    if len(a) > 1:
        w = _int(a[1], "zscore")
        return (s - s.rolling(w).mean()) / s.rolling(w).std()
    sd = s.std()
    return (s - s.mean()) / (sd if sd else np.nan)

def _b_rsi(ctx, *a):
    _arity(a, "rsi", 1, 2)
    s = _series(a[0], "rsi")
    n = _int(a[1], "rsi") if len(a) > 1 else 14
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / n, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)

def _b_corr(ctx, *a):
    _arity(a, "corr", 2, 3)
    x, y = _series(a[0], "corr", "first"), _series(a[1], "corr", "second")
    if len(a) > 2:
        return x.rolling(_int(a[2], "corr", "window")).corr(y)
    # scalar Pearson: align on index and drop pairwise-NaN (pandas .corr
    # semantics), then compute on the deterministic native kernel so the result
    # does not ride BLAS's threaded, reassociating dot product.
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 2:
        return float("nan")
    v = native.det_corr(df.iloc[:, 0].to_numpy(dtype=np.float64),
                        df.iloc[:, 1].to_numpy(dtype=np.float64))
    return float(v) if v == v else float("nan")

def _b_clip(ctx, *a):
    _arity(a, "clip", 3)
    return _series(a[0], "clip").clip(_num(a[1], "clip", "lo"), _num(a[2], "clip", "hi"))

def _b_abs(ctx, *a):
    _arity(a, "abs", 1)
    v = a[0]
    return v.abs() if isinstance(v, pd.Series) else abs(_num(v, "abs"))

def _b_sign(ctx, *a):
    _arity(a, "sign", 1)
    v = a[0]
    tr = getattr(ctx, "branch_trace", None)
    if isinstance(v, pd.Series):
        out = np.sign(v)
        if tr is not None:
            vals = v.to_numpy(dtype=float)
            fin = np.isfinite(vals)
            dec = np.full(vals.shape, 2, dtype=np.int8)      # NaN sentinel
            dec[fin] = np.sign(vals[fin]).astype(np.int8)    # −1 / 0 / +1
            _trace_elementwise(tr, "sign", vals, dec)
        return out
    x = _num(v, "sign")
    if tr is not None:
        tr.append(("sign", "s%+d" % int(np.sign(x)), abs(x), None))
    return float(np.sign(x))

def _b_where(ctx, *a):
    _arity(a, "where", 3)
    cond, x, y = a
    if isinstance(cond, pd.Series):
        idx = cond.index
        xx = x.reindex(idx) if isinstance(x, pd.Series) else x
        yy = y.reindex(idx) if isinstance(y, pd.Series) else y
        return pd.Series(np.where(cond.astype(bool), xx, yy), index=idx)
    return x if bool(cond) else y

def _reducer(name: str, fn) -> Callable:
    def f(ctx, *a):
        _arity(a, name, 1)
        return float(fn(_series(a[0], name)))
    return f


def _det_reducer(name: str, kind: str) -> Callable:
    """Terminal scalar reducer routed through the deterministic native kernel
    (fixed-order Neumaier). On the native substrate the result is bit-identical
    across runs / thread counts / machines — the property the certified pipeline
    needs so a (program, data, certificate) triple reproduces the same number;
    `corr` in particular escapes BLAS's threaded, reassociating dot product.
    NaN handling matches pandas' skipna default (drop, then reduce): all-NaN sum
    → 0.0, all-NaN mean/std → NaN, n<2 std → NaN — so this is a substrate swap,
    not a semantic change. Agreement with pandas is bit-exact for sum/mean on
    well-conditioned data and within a ULP for std (the two-pass compensated
    variance here is the MORE accurate of the two — never the less)."""
    def f(ctx, *a):
        _arity(a, name, 1)
        d = _series(a[0], name).dropna().to_numpy(dtype=np.float64)
        if kind == "sum":
            return float(native.det_sum(d))
        mean, var = native.det_mean_var(d)
        if kind == "mean":
            return float(mean)
        return float(math.sqrt(var)) if var == var else float("nan")   # std
    return f


def _extremum(name: str, mode: str) -> Callable:
    """min/max reducer. Value semantics identical to the plain reducer; when the
    context exposes a `branch_trace` list (certified execution only), also
    records (op, winner index label, winner-vs-runner-up margin) so the DRC
    layer can margin-check branch stability instead of blanket-refusing."""
    def f(ctx, *a):
        _arity(a, name, 1)
        s = _series(a[0], name)
        out = float(s.min() if mode == "min" else s.max())
        tr = getattr(ctx, "branch_trace", None)
        if tr is not None:
            d = s.dropna()
            # min/max carry a single scalar margin (the winner-vs-runner-up gap);
            # mvec is None → the guard uses the scalar-margin path for them.
            if len(d) >= 2:
                v = d.to_numpy(dtype=float)
                order = np.argsort(v, kind="stable")
                w, r = (order[0], order[1]) if mode == "min" else (order[-1], order[-2])
                tr.append((name, d.index[w], float(abs(v[r] - v[w])), None))
            elif len(d) == 1:
                tr.append((name, d.index[0], float("inf"), None))   # no competitor
            else:
                tr.append((name, None, 0.0, None))                  # empty: degenerate
        return out
    return f

def _b_sqrt(ctx, *a):
    _arity(a, "sqrt", 1)
    v = a[0]
    return v.pow(0.5) if isinstance(v, pd.Series) else math.sqrt(_num(v, "sqrt"))


BUILTINS: Dict[str, Callable] = {
    "price": _b_price, "series": _b_series, "table": _b_table,
    "returns": _b_returns, "logret": _b_logret, "diff": _b_diff, "lag": _b_lag,
    "sma": _b_sma, "ema": _b_ema, "rolling_mean": _b_rolling_mean,
    "rolling_std": _b_rolling_std, "zscore": _b_zscore, "rsi": _b_rsi,
    "corr": _b_corr, "clip": _b_clip, "abs": _b_abs, "sign": _b_sign,
    "where": _b_where, "sqrt": _b_sqrt,
    "mean": _det_reducer("mean", "mean"),
    "std": _det_reducer("std", "std"),
    "last": _reducer("last", lambda s: s.dropna().iloc[-1] if len(s.dropna()) else float("nan")),
    "first": _reducer("first", lambda s: s.dropna().iloc[0] if len(s.dropna()) else float("nan")),
    "min": _extremum("min", "min"),
    "max": _extremum("max", "max"),
    "sum": _det_reducer("sum", "sum"),
    "count": _reducer("count", lambda s: float(s.notna().sum())),
}

_ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul,
          "/": operator.truediv, "%": operator.mod, "^": operator.pow}
_CMP = {">": operator.gt, "<": operator.lt, ">=": operator.ge,
        "<=": operator.le, "==": operator.eq, "!=": operator.ne}


# ── outputs / result ──────────────────────────────────────────────────────────

@dataclass
class Output:
    kind: str                  # 'show' | 'signal'
    name: str
    value: object
    summary: str


@dataclass
class Result:
    ok: bool
    outputs: List[Output] = field(default_factory=list)
    bindings: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None
    note: str = ""


def _summarize(v) -> str:
    if isinstance(v, pd.Series):
        d = v.dropna()
        last = f"{d.iloc[-1]:.4g}" if len(d) else "n/a"
        return f"series[{len(v)}] last={last}, NaN={int(v.isna().sum())}"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.6g}"
    return repr(v)


# ── the interpreter ───────────────────────────────────────────────────────────

class Interpreter:
    def __init__(self, ctx: DataContext):
        self.ctx = ctx
        self.env: Dict[str, object] = {}

    def eval(self, node):
        if isinstance(node, Num):
            return node.value
        if isinstance(node, Str):
            return node.value
        if isinstance(node, Name):
            if node.id not in self.env:
                raise DSLError(f"unknown name {node.id!r} (not a let-binding)")
            return self.env[node.id]
        if isinstance(node, Call):
            fn = BUILTINS.get(node.func)
            if fn is None:
                raise DSLError(f"unknown function {node.func!r}")
            args = [self.eval(a) for a in node.args]
            try:
                return fn(self.ctx, *args)
            except DSLError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise DSLError(f"{node.func}(): {exc}")
        if isinstance(node, UnaryMinus):
            v = self.eval(node.operand)
            return -v if isinstance(v, pd.Series) else -_num(v, "unary -")
        if isinstance(node, Not):
            v = self.eval(node.operand)
            return (~v.astype(bool)) if isinstance(v, pd.Series) else (not bool(v))
        if isinstance(node, BoolOp):
            l, r = self.eval(node.left), self.eval(node.right)
            if isinstance(l, pd.Series) or isinstance(r, pd.Series):
                lb = l.astype(bool) if isinstance(l, pd.Series) else bool(l)
                rb = r.astype(bool) if isinstance(r, pd.Series) else bool(r)
                return (lb & rb) if node.op == "and" else (lb | rb)
            return (bool(l) and bool(r)) if node.op == "and" else (bool(l) or bool(r))
        if isinstance(node, BinOp):
            return self._binop(node.op, self.eval(node.left), self.eval(node.right))
        raise DSLError(f"cannot evaluate node {type(node).__name__}")

    def _binop(self, op: str, a, b):
        if op in _CMP:
            if isinstance(a, str) or isinstance(b, str):
                if op in ("==", "!="):
                    return _CMP[op](a, b)   # data-independent — no branch hazard
                raise DSLError(f"cannot compare strings with {op!r}")
            result = _CMP[op](a, b)
            tr = getattr(self.ctx, "branch_trace", None)
            if tr is not None:
                self._trace_cmp(op, a, b, result, tr)
            return result
        # arithmetic — numbers and/or series only
        for v in (a, b):
            if isinstance(v, str):
                raise DSLError(f"cannot apply {op!r} to a string")
            if isinstance(v, bool):
                raise DSLError(f"cannot apply {op!r} to a boolean")
        try:
            r = _ARITH[op](a, b)
        except ArithmeticError as exc:      # scalar 1/0, 1%0, 2^2000 (Series → inf/nan)
            raise DSLError(f"arithmetic error in {op!r}: {exc}")
        if isinstance(r, complex):          # e.g. negative base ^ fractional exponent
            raise DSLError(f"{op!r} produced a complex value "
                           "(negative base with a fractional exponent?)")
        return r

    def _trace_cmp(self, op: str, a, b, result, tr: list) -> None:
        """Record a numeric comparison's branch decision: margin = |a − b| (the
        distance to the threshold), decision = the boolean result. Only fires for
        numeric operands (string ==/!= is data-independent and never traced)."""
        if isinstance(result, pd.Series):
            av = a if isinstance(a, pd.Series) else float(a)
            bv = b if isinstance(b, pd.Series) else float(b)
            diff = av - bv                                   # pandas index-aligns
            dv = (diff.to_numpy(dtype=float) if isinstance(diff, pd.Series)
                  else np.asarray(diff, dtype=float))
            fin = np.isfinite(dv)
            dec = np.where(fin, result.to_numpy().astype(np.int8), np.int8(2))
            _trace_elementwise(tr, op, dv, dec.astype(np.int8))
        else:
            dv = float(a) - float(b)
            tr.append((op, "c" + str(bool(result)), abs(dv), None))

    def run(self, program: Program) -> Result:
        outputs: List[Output] = []
        for stmt in program.statements:
            try:
                if isinstance(stmt, Let):
                    self.env[stmt.name] = self.eval(stmt.expr)
                elif isinstance(stmt, Signal):
                    val = self.eval(stmt.expr)
                    self.env[stmt.name] = val
                    outputs.append(Output("signal", stmt.name, val, _summarize(val)))
                elif isinstance(stmt, Show):
                    val = self.eval(stmt.expr)
                    outputs.append(Output("show", "", val, _summarize(val)))
            except DSLError as exc:
                ln = getattr(stmt, "line", 0)
                return Result(ok=False, outputs=outputs, bindings=dict(self.env),
                              error=f"line {ln}: {exc}",
                              note="evaluation stopped at the first error")
        return Result(ok=True, outputs=outputs, bindings=dict(self.env))


def run_program(src: str, *, data_service=None, ctx: Optional[DataContext] = None,
                program: Optional[Program] = None) -> Result:
    """Parse and evaluate a DSL program. Returns a Result (never raises); parse
    and runtime errors are reported in `.error` with a line number. Pass a
    pre-parsed `program` (e.g. a pruned AST) to skip parsing `src`."""
    if program is not None:
        try:
            interp = Interpreter(ctx or DataContext(data_service))
            return interp.run(program)
        except Exception as exc:  # noqa: BLE001 — never raises
            return Result(ok=False, error=f"internal error: {exc}",
                          note="evaluation aborted")
    try:
        program = parse(src)
    except DSLSyntaxError as exc:
        return Result(ok=False, error=f"syntax error: {exc}", note="parse failed")
    except RecursionError:
        return Result(ok=False, error="syntax error: expression nested too deeply",
                      note="parse failed")
    except Exception as exc:  # noqa: BLE001 — parsing must never raise out of here
        return Result(ok=False, error=f"internal parse error: {exc}", note="parse failed")
    try:
        interp = Interpreter(ctx or DataContext(data_service))
        return interp.run(program)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders: run_program NEVER raises
        # per-statement DSLErrors are already turned into a Result inside run();
        # this catches only exotic escapes (deep-nesting RecursionError in eval, etc.)
        return Result(ok=False, error=f"internal error: {exc}",
                      note="evaluation aborted")


def builtin_names() -> List[str]:
    """The whitelisted vocabulary (for help / autocomplete)."""
    return sorted(BUILTINS.keys())
