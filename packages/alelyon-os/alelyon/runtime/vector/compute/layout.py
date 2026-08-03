"""Qt-free rendering layout for a `ComputationGraph` (roadmap Phase 3 GUI seam).

Turns a graph (+ an optional propagation `GraphResult`) into a plain dict a
renderer lays out without touching graph internals or importing Qt:

    {
      "nodes": [{name, label, level, kind, tone, point, mean, q05, q95, highlight,
                 driver_share}, ...],   # level = longest path from a source
      "edges": [{src, dst}, ...],
      "n_levels": int,
    }

`level` (longest path from a source) drives the column a node is drawn in, so
sources sit on the left and sinks on the right. Tone (green/red/amber/neutral) is
supplied by the caller since only the pipeline knows a value's semantics
(a +score is constructive for the desk graph but a +score is *stress* for the
macro graph). Pure — unit-testable, no Qt.
"""
from __future__ import annotations

from typing import Dict, Optional

from alelyon.runtime.vector.compute.graph import ComputationGraph
from alelyon.runtime.vector.compute.types import GraphResult


def _levels(graph: ComputationGraph) -> Dict[str, int]:
    """Longest-path level per node: a source is 0, a compute node is 1 + the max
    of its dependencies' levels (a depless compute node is also 0)."""
    level: Dict[str, int] = {}
    for name in graph.topological_order():        # deps precede dependents
        deps = graph.deps_of(name)
        level[name] = 0 if not deps else 1 + max(level[d] for d in deps)
    return level


def graph_layout(graph: ComputationGraph, result: Optional[GraphResult] = None, *,
                 labels: Optional[Dict[str, str]] = None,
                 tones: Optional[Dict[str, str]] = None,
                 highlight: Optional[str] = None,
                 target: Optional[str] = None) -> dict:
    """Render-ready layout for `graph`. `result` fills each node's point/credible
    interval; `labels`/`tones` override display text/colour per node; `highlight`
    marks one node (e.g. the top driver) for emphasis; `target` selects which
    node's variance shares annotate the sources (defaults to the first sink)."""
    labels = labels or {}
    tones = tones or {}
    level = _levels(graph)

    shares: Dict[str, float] = {}
    if result is not None:
        if target is not None:
            # an explicit target that wasn't attributed gets NO shares — never
            # silently borrow another target's attribution
            tgt = target if target in result.sensitivities else None
        else:
            tgt = next(iter(result.sensitivities)) if result.sensitivities else None
        if tgt is not None:
            shares = result.sensitivities.get(tgt, {})

    nodes = []
    for name in graph.node_names():
        nr = result.get(name) if result is not None else None
        nodes.append({
            "name": name,
            "label": labels.get(name, name),
            "level": level.get(name, 0),
            "kind": "source" if graph.is_source(name) else "compute",
            "tone": tones.get(name, "neutral"),
            "point": (None if nr is None else nr.point),
            "mean": (None if nr is None else nr.mean),
            "q05": (None if nr is None else nr.q05),
            "q95": (None if nr is None else nr.q95),
            "highlight": (name == highlight),
            "driver_share": float(shares.get(name, 0.0)),
        })

    edges = [{"src": d, "dst": name}
             for name in graph.node_names() for d in graph.deps_of(name)]

    return {"nodes": nodes, "edges": edges,
            "n_levels": (max(level.values()) + 1 if level else 0)}
