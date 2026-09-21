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
    credible interval). The RNG kernel is `np.random.default_rng(seed)`; pass
    `simulator=` to share an existing one. Also attributes each sink node's
    variance back to its sources (first-order share) so a caller can say *which
    input drives the uncertainty*.

  • `add_joint_inputs()` — sources that are NOT independent of each other.
    `add_input` sources are drawn INDEPENDENTLY, which is correct only when they
    are independent in fact. For a sum the true variance is `sum(var) +
    2*sum(cov)`; drawing independently omits the covariance term entirely, so
    the intervals come out too NARROW — anticonservative, the one direction an
    uncertainty interval must never fail in. The omitted term grows O(k²)
    against the retained O(k), so the error DEEPENS with graph depth rather than
    staying constant. Measured on real data in two independent research lanes:
    at nominal 90%, observed coverage fell to 0.7187 at depth 24 and 0.6080 at
    depth 28, with sources only MILDLY correlated (mean pairwise r ≈ +0.21) —
    ordinary correlation is enough, this is not a pathology of unusually coupled
    inputs. A decorrelation control that preserved every marginal exactly
    restored nominal coverage at every depth, isolating the cause. Registering
    correlated sources through `add_joint_inputs` draws them from PAIRED
    observations with one shared row index, which preserves their joint
    structure exactly and non-parametrically.

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
    Constant, Distribution, Empirical, GraphResult, NodeResult,
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
        # Sources that covary, registered via add_joint_inputs: (names, (m,k)).
        # Held on the graph rather than in a Distribution because the
        # `Distribution` protocol is per-source by construction — sample(rng, n)
        # cannot express a dependence between two of them.
        self._joint_blocks: List[tuple] = []
        self._order: List[str] = []        # insertion order, for stable output

    # ── construction ──────────────────────────────────────────────────────────
    def add_joint_inputs(self, names: Sequence[str],
                         observations) -> "ComputationGraph":
        """Register k sources that COVARY, from (m, k) paired observations.

        `add_input` draws each source independently, which silently omits the
        covariance term and produces intervals that are too narrow (module
        docstring). This registers a BLOCK: at propagate time one row index of
        length n is drawn for the whole block and every member is indexed with
        it, so each draw is a real observed row and every pairwise dependence —
        linear or not — survives exactly. No correlation matrix is estimated and
        no copula is assumed; the joint structure IS the data.

        Each member also gets an `Empirical` marginal over its own column, so
        `evaluate()` and the per-source summaries are unchanged. Rows containing
        a non-finite value are REFUSED rather than dropped: dropping them would
        silently change the joint distribution being sampled, which is the
        failure this method exists to prevent.
        """
        cols = list(names)
        if len(cols) < 2:
            raise ComputeGraphError(
                "add_joint_inputs needs at least 2 names; a single source has no "
                "joint structure to preserve — use add_input")
        obs = np.asarray(observations, dtype=float)
        if obs.ndim != 2 or obs.shape[1] != len(cols):
            raise ComputeGraphError(
                f"observations must be (m, {len(cols)}) for names {cols}; got "
                f"shape {obs.shape}")
        if obs.shape[0] < 2:
            raise ComputeGraphError(
                "add_joint_inputs needs at least 2 observed rows to resample")
        if not np.all(np.isfinite(obs)):
            bad = int(np.count_nonzero(~np.isfinite(obs)))
            raise ComputeGraphError(
                f"observations contain {bad} non-finite value(s). They are "
                f"refused rather than dropped: dropping rows would change the "
                f"joint distribution being sampled without saying so")
        for j, nm in enumerate(cols):
            self.add_input(nm, Empirical(tuple(obs[:, j])))
        self._joint_blocks.append((cols, obs))
        return self

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
        """Push `n_samples` draws through the DAG and summarise every node.

        Each draw is joint across the TOPOLOGY — one row of source values is
        propagated through the whole graph together, so a node sees a coherent
        set of inputs. It is NOT joint across the SOURCES: every `add_input`
        source is drawn independently of every other. That is correct when the
        sources are independent and WRONG when they are not — omitting the
        covariance term makes intervals too narrow, and the error grows with
        depth (module docstring has the measured numbers). Sources that covary
        must be registered with `add_joint_inputs`, which shares one draw index
        across the block; this method then uses that block's paired rows.

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

        # `MonteCarloSimulator(seed).rng` IS `np.random.default_rng(seed)` — one
        # line, stats_engine.py:180. Importing it bought a vector -> sentinel
        # dependency (stats_engine.py:10 pulls in sentinel.alert_engine at module
        # scope) for no numerical difference, and stats_engine does not ship in
        # the alelyon-os wheel — so every external caller took the except branch
        # while every in-repo caller took the try branch. Two populations running
        # different code, identical only by coincidence and pinned by nothing.
        # Calling numpy directly makes them identical by construction.
        # `simulator=` remains the seam for sharing an RNG across graphs.
        rng = simulator.rng if simulator is not None else np.random.default_rng(seed)

        draws: Dict[str, np.ndarray] = {}

        # JOINT BLOCKS FIRST. One row index for the whole block, shared by every
        # member — that shared index IS the mechanism: it makes each draw a real
        # observed row, so the covariance term the independent path omits is
        # carried exactly, without estimating a correlation matrix or assuming a
        # copula. Drawn before the topological loop so the loop can skip them.
        for _blk_names, _blk_obs in self._joint_blocks:
            _idx = rng.integers(0, _blk_obs.shape[0], n)
            for _j, _nm in enumerate(_blk_names):
                draws[_nm] = _blk_obs[_idx, _j]

        for name in order:
            node = self._nodes[name]
            if node.is_source:
                if name in draws:      # already drawn, jointly, above
                    continue
                s = np.asarray(node.dist.sample(rng, n), dtype=float)
                if s.shape != (n,):
                    raise ComputeGraphError(
                        f"source '{name}' sampled shape {s.shape}, expected ({n},)")
                # `Distribution` (types.py:31) says implementations MUST return
                # finite means and length-n arrays. Only the length half was
                # enforced, two lines up, and the finite half decided nothing —
                # so a `Normal(mu, inf)` sampled to +/-inf, every statistic
                # downstream became nan, and NOTHING raised. Worse than the nan:
                # `_attribute_variance` then dropped that source, so the input
                # with UNBOUNDED uncertainty vanished from "what drives the
                # uncertainty" while the ranking still looked healthy.
                if not np.all(np.isfinite(s)):
                    bad = int(np.count_nonzero(~np.isfinite(s)))
                    raise ComputeGraphError(
                        f"source '{name}' sampled {bad} non-finite value(s) of "
                        f"{n}; a Distribution must return finite draws "
                        f"(types.py:31). An infinite or undefined uncertainty is "
                        f"a refusal to state one, and it must not enter the DAG "
                        f"as though it were a number")
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
        # A non-finite target has no decomposition. Ranking the finite sources
        # against a quantity that is not a number reports a share of nothing.
        if not np.all(np.isfinite(y)):
            return {}
        yv = float(np.var(y))
        if yv <= 1e-15:
            return {}
        raw: Dict[str, float] = {}
        for name in self.sources():
            x = draws[name]
            # NOT `continue`, which is what a bare `isfinite(r)` filter amounted
            # to. A source whose draws are non-finite carries the LARGEST
            # uncertainty there is; skipping it renormalises the rest to 1.0 and
            # presents the remaining sources as the whole story, with the real
            # driver absent rather than first. Refuse the attribution instead —
            # naming nothing is recoverable, naming the wrong input is not.
            # Unreachable for a source now that `propagate` rejects non-finite
            # draws at the sampling step; kept because this is the place the
            # evidence was destroyed, and a future caller may reach it another way.
            if not np.all(np.isfinite(x)):
                return {}
            xv = float(np.var(x))
            if xv <= 1e-15:          # a genuine constant contributes no variance
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(r):
                return {}
            raw[name] = r * r
        tot = sum(raw.values())
        if tot <= 0.0:
            return {}
        shares = {k: v / tot for k, v in raw.items()}
        return dict(sorted(shares.items(), key=lambda kv: kv[1], reverse=True))
