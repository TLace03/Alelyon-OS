"""The computational engine (roadmap Phase 3): a typed dependency DAG with
uncertainty propagation.

The vision names a "computational engine — graph construction → state estimation
→ inference → simulation → optimization → recommendation." Today's data flow is
linear (bars → regime → factors → picks) with no explicit dependency graph and,
crucially, **no uncertainty propagation**. This is the honest first framework:

  • `ComputationGraph` — declare source nodes (each an uncertain `Distribution`)
    and compute nodes (a pure elementwise `fn` over named dependencies). The
    graph validates that it is a DAG (cycle + missing-dependency detection) and
    evaluates in topological order.

  • `evaluate()` — the deterministic forward pass: every source at its mean,
    every compute node applied once. This is the point estimate.

  • `propagate()` — Monte-Carlo uncertainty propagation: draw every source from
    its distribution, push all draws through the DAG at once (vectorised over the
    sample axis), and summarise each node's resulting marginal (mean, std, 90%
    credible interval). Reuses `engines/stats_engine.MonteCarloSimulator` as the
    RNG kernel (roadmap mandate). Also attributes each sink node's variance back
    to its sources (first-order share) so a caller can say *which input drives
    the uncertainty*.

Node `fn`s MUST be elementwise / sample-axis-agnostic — the SAME callable runs
on scalars (evaluate) and on (n,) arrays (propagate). Sums, products, weighted
combinations, `np.maximum`, `np.where` all qualify. A node may also return a bare
scalar (broadcast to a constant across draws — a legitimate "derived constant"
node). The one shape the engine can reject is a foreign-length array (length ≠ 1
and ≠ n) — that always signals a bug. It CANNOT distinguish a legitimate constant
from a fn that wrongly reduces the sample axis to a scalar (both look identical),
so DO NOT write sample-axis reductions inside a node — there is no runtime guard
against it. Pure NumPy, framework-free — Phase-4/8 reuse it on any graph, not
just finance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from alelyon.runtime.vector.compute.types import (
    Constant, Distribution, GraphResult, NodeResult,
)


class ComputeGraphError(Exception):
    """Raised for a malformed graph: duplicate node, missing dependency, cycle,
    or an ill-shaped node output."""


@dataclass
class _Node:
    name: str
    deps: List[str]
    fn: Optional[Callable[[Dict[str, np.ndarray]], np.ndarray]]  # None → source
    dist: Optional[Distribution]                                 # set iff source

    @property
    def is_source(self) -> bool:
        return self.fn is None


class ComputationGraph:
    """A typed dependency DAG. Build with `add_input`/`add`, then `evaluate`
    (point) or `propagate` (Monte-Carlo). Construction order is free — node
    existence is validated lazily at evaluation/topo time."""

    def __init__(self) -> None:
        self._nodes: Dict[str, _Node] = {}
        self._order: List[str] = []        # insertion order, for stable output

    # ── construction ──────────────────────────────────────────────────────────
    def add_input(self, name: str, dist: Distribution) -> "ComputationGraph":
        """A source node carrying an uncertain value. Accepts any `Distribution`;
        a bare float is promoted to a `Constant`."""
        if isinstance(dist, (int, float)):
            dist = Constant(float(dist))
        if not (hasattr(dist, "mean") and hasattr(dist, "sample")):
            raise ComputeGraphError(
                f"input '{name}' needs a Distribution (got {type(dist).__name__})")
        self._register(_Node(name, [], None, dist))
        return self

    def add(self, name: str, fn: Callable[[Dict[str, np.ndarray]], np.ndarray],
            deps: Sequence[str]) -> "ComputationGraph":
        """A compute node: `fn(inputs)` where `inputs` maps each dependency name
        to its value (scalar in evaluate, (n,) array in propagate). `deps` may be
        empty (a derived constant) and may reference nodes added later."""
        if not callable(fn):
            raise ComputeGraphError(f"compute node '{name}' needs a callable fn")
        self._register(_Node(name, list(deps), fn, None))
        return self

    def _register(self, node: _Node) -> None:
        if node.name in self._nodes:
            raise ComputeGraphError(f"duplicate node '{node.name}'")
        if not node.name:
            raise ComputeGraphError("node name must be non-empty")
        self._nodes[node.name] = node
        self._order.append(node.name)

    # ── structure ─────────────────────────────────────────────────────────────
    def has(self, name: str) -> bool:
        return name in self._nodes

    def node_names(self) -> List[str]:
        """All node names in insertion order (stable for rendering)."""
        return list(self._order)

    def deps_of(self, name: str) -> List[str]:
        """The dependency names of a node (empty for a source or unknown node)."""
        nd = self._nodes.get(name)
        return list(nd.deps) if nd is not None else []

    def is_source(self, name: str) -> bool:
        nd = self._nodes.get(name)
        return bool(nd is not None and nd.is_source)

    def sources(self) -> List[str]:
        return [n for n in self._order if self._nodes[n].is_source]

    def sinks(self) -> List[str]:
        """Nodes that nothing depends on (the graph's outputs)."""
        depended: set = set()
        for nd in self._nodes.values():
            depended.update(nd.deps)
        return [n for n in self._order if n not in depended]

    def topological_order(self) -> List[str]:
        """A topological ordering of all nodes. Raises `ComputeGraphError` on a
        missing dependency or any cycle (including a self-loop)."""
        WHITE, GREY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._nodes}
        order: List[str] = []

        # iterative DFS so a deep chain cannot blow the Python recursion limit
        for root in self._order:
            if color[root] != WHITE:
                continue
            stack = [(root, iter(self._nodes[root].deps))]
            color[root] = GREY
            while stack:
                node, deps_it = stack[-1]
                advanced = False
                for dep in deps_it:
                    if dep not in self._nodes:
                        raise ComputeGraphError(
                            f"node '{node}' depends on undefined node '{dep}'")
                    c = color[dep]
                    if c == GREY:
                        raise ComputeGraphError(
                            f"cycle detected through '{dep}' (via '{node}')")
                    if c == WHITE:
                        color[dep] = GREY
                        stack.append((dep, iter(self._nodes[dep].deps)))
                        advanced = True
                        break
                if not advanced:
                    color[node] = BLACK
                    order.append(node)
                    stack.pop()
        return order

    # ── deterministic forward pass ─────────────────────────────────────────────
    def evaluate(self, overrides: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Point value of every node: sources at their `mean()` (or an override),
        compute nodes applied once. Returns {name: float}."""
        overrides = overrides or {}
        vals: Dict[str, float] = {}
        for name in self.topological_order():
            node = self._nodes[name]
            if node.is_source:
                vals[name] = float(overrides.get(name, node.dist.mean()))
            else:
                inp = {d: vals[d] for d in node.deps}
                out = node.fn(inp)
                if np.ndim(out):
                    arr = np.asarray(out, dtype=float).reshape(-1)
                    if arr.size == 0:
                        raise ComputeGraphError(
                            f"compute node '{name}' returned an empty array")
                    vals[name] = float(arr[0])
                else:
                    vals[name] = float(out)
        return vals

    # ── Monte-Carlo uncertainty propagation ────────────────────────────────────
    def propagate(self, n_samples: int = 4000, *, seed: Optional[int] = None,
                  simulator=None, keep: Optional[Sequence[str]] = None,
                  attribute: Optional[Sequence[str]] = None) -> GraphResult:
        """Push `n_samples` joint draws through the DAG and summarise every node.

        `simulator` — reuse an existing `MonteCarloSimulator` (its RNG is the
        kernel); otherwise one is built from `seed`. `keep` — node names whose
        raw (n,) sample arrays are returned (for post-processing, e.g. a
        categorical classifier); default keeps none. `attribute` — target nodes
        to decompose variance for; default is every sink.
        """
        n = int(n_samples)
        if n < 2:
            raise ComputeGraphError("propagate needs n_samples >= 2")
        order = self.topological_order()          # validates the DAG first

        if simulator is not None:
            rng = simulator.rng
        else:
            try:                       # reuse the engine's seeded MC kernel if present
                from alelyon.runtime.vector.stats_engine import MonteCarloSimulator
                rng = MonteCarloSimulator(seed=seed).rng
            except Exception:  # noqa: BLE001 — decoupled fallback (e.g. web deploy,
                rng = np.random.default_rng(seed)   # no engine): a plain seeded RNG

        draws: Dict[str, np.ndarray] = {}
        for name in order:
            node = self._nodes[name]
            if node.is_source:
                s = np.asarray(node.dist.sample(rng, n), dtype=float)
                if s.shape != (n,):
                    raise ComputeGraphError(
                        f"source '{name}' sampled shape {s.shape}, expected ({n},)")
                draws[name] = s
            else:
                inp = {d: draws[d] for d in node.deps}
                out = np.asarray(node.fn(inp), dtype=float)
                if out.size == 0:
                    raise ComputeGraphError(
                        f"compute node '{name}' returned an empty array")
                if out.ndim == 0 or out.size == 1:
                    out = np.full(n, float(out.reshape(-1)[0]))
                elif out.shape != (n,):
                    raise ComputeGraphError(
                        f"compute node '{name}' returned shape {out.shape}; a node "
                        f"fn must be elementwise over the sample axis (expected "
                        f"({n},) or a scalar)")
                draws[name] = out

        point = self.evaluate()
        results: Dict[str, NodeResult] = {}
        for name in self._order:
            x = draws[name]
            q05, q50, q95 = (float(v) for v in np.percentile(x, [5, 50, 95]))
            results[name] = NodeResult(
                name=name, point=float(point[name]),
                mean=float(np.mean(x)), std=float(np.std(x)),
                q05=q05, q50=q50, q95=q95)

        targets = list(attribute) if attribute is not None else self.sinks()
        sensitivities = {
            t: self._attribute_variance(t, draws) for t in targets if t in self._nodes
        }

        kept = {}
        if keep:
            for name in keep:
                if name in draws:
                    kept[name] = draws[name]

        return GraphResult(results=results, sensitivities=sensitivities,
                           samples=kept, n_samples=n)

    def _attribute_variance(self, target: str,
                            draws: Dict[str, np.ndarray]) -> Dict[str, float]:
        """First-order variance share of each SOURCE in `target`'s variance,
        normalised to sum to 1 and sorted largest-first.

        Honest scope: shares are squared source→target correlations. For
        INDEPENDENT sources (how `propagate` draws them) feeding a near-linear
        target this is the exact first-order variance decomposition; strong
        source interactions or nonlinearity make it an approximation, not an
        ANOVA. Returns {} when the target has no variance to attribute, and at
        n < 3 (two points are always perfectly collinear, so corr² ≡ 1 would
        yield uniform shares regardless of the true contributions — refuse
        rather than mislead; small n beyond that is merely noisy).
        """
        y = draws[target]
        if y.shape[0] < 3:
            return {}
        yv = float(np.var(y))
        if yv <= 1e-15:
            return {}
        raw: Dict[str, float] = {}
        for name in self.sources():
            x = draws[name]
            xv = float(np.var(x))
            if xv <= 1e-15:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if np.isfinite(r):
                raw[name] = r * r
        tot = sum(raw.values())
        if tot <= 0.0:
            return {}
        shares = {k: v / tot for k, v in raw.items()}
        return dict(sorted(shares.items(), key=lambda kv: kv[1], reverse=True))
