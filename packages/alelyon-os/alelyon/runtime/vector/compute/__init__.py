"""Vector compute — the domain-agnostic computational engine (roadmap Phase 3).

A typed dependency DAG (`graph.py`), its value/distribution types (`types.py`),
and the Qt-free render-dict layout helper (`layout.py`). Monte-Carlo
uncertainty propagation rides the DAG. No finance in this package: the
desk-synthesis instances built on this engine live in
`alelyon.runtime.oracle.synthesis`.
"""
from __future__ import annotations

from alelyon.runtime.vector.compute.graph import ComputationGraph, ComputeGraphError
from alelyon.runtime.vector.compute.layout import graph_layout
from alelyon.runtime.vector.compute.types import (
    Constant, Distribution, Empirical, GraphResult, Normal, NodeResult,
    TruncatedNormal,
)

__all__ = [
    "ComputationGraph", "ComputeGraphError",
    "Constant", "Normal", "TruncatedNormal", "Empirical", "Distribution",
    "NodeResult", "GraphResult",
    "graph_layout",
]
