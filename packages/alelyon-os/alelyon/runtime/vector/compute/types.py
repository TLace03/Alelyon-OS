"""Typed value model for the computational engine (roadmap Phase 3).

Two small, domain-agnostic vocabularies the DAG in `graph.py` is built on:

  • Distributions — a source node's uncertainty. Each exposes `mean()` (used by
    the deterministic forward pass) and `sample(rng, n)` (used by Monte-Carlo
    propagation). Deliberately tiny: Constant / Normal / TruncatedNormal /
    Empirical cover every current need; more can be added without touching the
    graph.

  • Results — `NodeResult` (one node's point value + its propagated
    distribution summary) and `GraphResult` (every node's result + the
    first-order sensitivity attribution + the raw kept samples a caller can
    post-process, e.g. a categorical classifier over joint draws).

Everything here is pure NumPy and framework-free so Phase-4/8 (the same engine
applied to a non-finance graph) reuses it unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


# ── distributions ─────────────────────────────────────────────────────────────
@runtime_checkable
class Distribution(Protocol):
    """A source of uncertainty. `mean()` is the deterministic point value;
    `sample(rng, n)` draws an (n,) array. Implementations MUST return finite
    means and length-n arrays."""

    def mean(self) -> float: ...

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray: ...


@dataclass(frozen=True)
class Constant:
    """A degenerate distribution — a known scalar with no uncertainty."""
    value: float

    def mean(self) -> float:
        return float(self.value)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return np.full(int(n), float(self.value), dtype=float)


@dataclass(frozen=True)
class Normal:
    mu: float
    sigma: float

    def mean(self) -> float:
        return float(self.mu)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        s = max(0.0, float(self.sigma))
        if s == 0.0:
            return np.full(int(n), float(self.mu), dtype=float)
        return rng.normal(float(self.mu), s, int(n))


@dataclass(frozen=True)
class TruncatedNormal:
    """Normal clipped to [lo, hi]. The natural choice for a bounded score (e.g. a
    desk bias in [-1, 1]): `mean()` returns the *clipped* central value so the
    deterministic pass and the sample mean agree at the boundaries."""
    mu: float
    sigma: float
    lo: float = -np.inf
    hi: float = np.inf

    def __post_init__(self):
        if self.hi < self.lo:
            raise ValueError(f"TruncatedNormal: hi ({self.hi}) < lo ({self.lo})")

    def mean(self) -> float:
        return float(np.clip(self.mu, self.lo, self.hi))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        s = max(0.0, float(self.sigma))
        if s == 0.0:
            base = np.full(int(n), float(self.mu), dtype=float)
        else:
            base = rng.normal(float(self.mu), s, int(n))
        return np.clip(base, self.lo, self.hi)


@dataclass(frozen=True)
class Empirical:
    """Bootstrap from an observed sample (resampled with replacement)."""
    samples: Sequence[float]

    def _arr(self) -> np.ndarray:
        a = np.asarray(self.samples, dtype=float)
        a = a[np.isfinite(a)]
        return a

    def mean(self) -> float:
        a = self._arr()
        return float(a.mean()) if a.size else 0.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = self._arr()
        if a.size == 0:
            return np.zeros(int(n), dtype=float)
        idx = rng.integers(0, a.size, int(n))
        return a[idx]


# ── results ───────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NodeResult:
    """One node's value after evaluation. `point` is the deterministic forward
    pass (sources at their mean); the rest summarize the Monte-Carlo marginal."""
    name: str
    point: float
    mean: float
    std: float
    q05: float
    q50: float
    q95: float


@dataclass(frozen=True)
class GraphResult:
    """The full propagation result: every node's marginal, the first-order
    variance attribution per target, the kept raw samples, and the shared
    sample count."""
    results: Dict[str, NodeResult]
    sensitivities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    samples: Dict[str, np.ndarray] = field(default_factory=dict)
    n_samples: int = 0

    def get(self, name: str) -> Optional[NodeResult]:
        return self.results.get(name)

    def sensitivity(self, target: str) -> Dict[str, float]:
        """{input_name: variance share in [0,1]} for `target`, largest first.
        Empty if the target has no variance or wasn't attributed."""
        return self.sensitivities.get(target, {})

    def top_driver(self, target: str) -> Optional[str]:
        """The input contributing the most variance to `target`, or None."""
        s = self.sensitivities.get(target, {})
        if not s:
            return None
        return max(s.items(), key=lambda kv: kv[1])[0]
