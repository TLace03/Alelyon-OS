"""Project the worktree mesh and its cache into a graph any renderer can draw.

The observer knows what is true, the cache knows what changed. Neither is a shape
a picture can be made from, and every frontend that wants one would otherwise
re-derive the same nodes and edges slightly differently. This is that projection,
once, as data.

Provenance is on every edge, and it has three levels
-----------------------------------------------------
Graphify's knowledge-graph output tags each edge `EXTRACTED`, `INFERRED` or
`AMBIGUOUS` so a reader "always knows what was found vs guessed". That discipline
is right and this adopts it, split the way this domain actually divides:

* `OBSERVED`  — git said so. A worktree exists at a path; its HEAD is this commit;
  this file is dirty. Nothing an agent writes can change it.
* `DERIVED`   — computed from observed facts by a rule stated in `EDGE_RULES`.
  Contention is a path-set intersection; tool family is a directory convention.
  True if the rule is sound, and the rule is named so it can be disputed.
* `DECLARED`  — a session said it about itself. The only route to model identity,
  and a claim. Never promoted, never merged into the other two.

A renderer that cannot show the difference should show only `OBSERVED`, and
`GraphProjection.filtered()` exists so that is one call rather than a judgement.

Colour is carried, not chosen
------------------------------
Node colour comes from the cache's persisted slot, so a node keeps its hue across
sessions and a new worktree never repaints an existing one. Past the measured
capacity of three, `colour` is None and `sigil` carries identity instead — see
`worktree_cache` for why the number is three and not eight.

Pure: takes records, returns records. No git, no filesystem, no clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json

from alelyon.runtime.common.worktree import UNATTRIBUTED, WorktreeMesh
from alelyon.runtime.common.worktree_cache import (
    UNSLOTTED, Operation, WorktreeCache, WorktreeIdentity,
)

GRAPH_SCHEMA = "alelyon.worktree-graph/0.1"

OBSERVED = "OBSERVED"
DERIVED = "DERIVED"
DECLARED = "DECLARED"
PROVENANCE = (OBSERVED, DERIVED, DECLARED)

NODE_WORKTREE = "worktree"
NODE_PATH = "path"
NODE_SESSION = "session"

EDGE_TOUCHES = "touches"
EDGE_CONTENDS = "contends-with"
EDGE_CLAIMS = "claims"

#: The rule behind every DERIVED edge, so a reader can disagree with the
#: inference rather than only with the picture.
EDGE_RULES: dict[str, str] = {
    EDGE_CONTENDS: "both worktrees hold outstanding work on at least one shared "
                   "path; path-level, not semantic",
}


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    label: str
    provenance: str
    #: Hex for the viewer's mode, or None past colour capacity.
    colour: str | None = None
    #: The stable text badge. Always present — it is what identity falls back to.
    sigil: str = ""
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str
    provenance: str
    label: str = ""
    #: Why this edge exists, in one line, for a DERIVED edge.
    rule: str = ""
    weight: int = 1


@dataclass(frozen=True)
class GraphProjection:
    schema: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    #: What the graph cannot say, carried so a renderer cannot omit it.
    limits: tuple[str, ...] = ()

    def filtered(self, *provenances: str) -> "GraphProjection":
        """Keep only edges at the given provenance levels, and the nodes they use.

        A renderer with no way to distinguish found from guessed should call
        `filtered(OBSERVED)` rather than draw all three identically.
        """
        unknown = [v for v in provenances if v not in PROVENANCE]
        if unknown:
            raise ValueError(f"unknown provenance: {unknown}")
        keep = set(provenances)
        edges = tuple(e for e in self.edges if e.provenance in keep)
        used = {e.source for e in edges} | {e.target for e in edges}
        nodes = tuple(n for n in self.nodes
                      if n.id in used or n.kind == NODE_WORKTREE)
        return GraphProjection(schema=self.schema, nodes=nodes, edges=edges,
                               limits=self.limits)

    @property
    def hubs(self) -> tuple[tuple[str, int], ...]:
        """Highest-degree nodes, busiest first.

        Graphify calls these god nodes. Here the useful one is a path several
        worktrees are all editing, which is where a collision is already
        happening rather than merely possible.
        """
        degree: dict[str, int] = {}
        for edge in self.edges:
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1
        return tuple(sorted(degree.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_json(self, *, indent: int = 2) -> str:
        """Canonical JSON: sorted keys, so two runs over one state diff cleanly."""
        payload = {
            "schema": self.schema,
            "provenance_levels": list(PROVENANCE),
            "edge_rules": EDGE_RULES,
            "limits": list(self.limits),
            "nodes": [
                {"id": n.id, "kind": n.kind, "label": n.label,
                 "provenance": n.provenance, "colour": n.colour,
                 "sigil": n.sigil, "detail": n.detail}
                for n in self.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "kind": e.kind,
                 "provenance": e.provenance, "label": e.label, "rule": e.rule,
                 "weight": e.weight}
                for e in self.edges
            ],
        }
        return json.dumps(payload, indent=indent, sort_keys=True) + "\n"


def _path_id(path: str) -> str:
    return f"path:{path}"


def project(mesh: WorktreeMesh, cache: WorktreeCache | None = None, *,
            dark: bool = True) -> GraphProjection:
    """Build the graph from one observation, coloured by the cache if given.

    Without a cache the graph is still correct and simply has no colour: colour
    is a persisted property, and inventing one per run is exactly the repainting
    the cache exists to prevent.
    """
    identities: dict[str, WorktreeIdentity] = {}
    if cache is not None:
        identities = {i.path.replace("\\", "/").rstrip("/"): i
                      for i in cache.identities()}

    nodes: list[Node] = []
    edges: list[Edge] = []
    seen_paths: set[str] = set()

    for tree in mesh.worktrees:
        key = tree.path.replace("\\", "/").rstrip("/")
        identity = identities.get(key)
        node_id = f"worktree:{identity.key}" if identity else f"worktree:{key}"
        nodes.append(Node(
            id=node_id,
            kind=NODE_WORKTREE,
            label=tree.label,
            provenance=OBSERVED,
            colour=(identity.colour(dark=dark)
                    if identity and identity.colour_slot != UNSLOTTED else None),
            sigil=identity.sigil if identity else "",
            detail={
                "path": tree.path,
                "branch": tree.branch,
                "detached": tree.detached,
                "primary": tree.is_primary,
                "present": tree.present,
                "tool_family": tree.tool_family,
                "tool_evidence": tree.tool_evidence,
                "on_mainline": tree.on_mainline,
                "touched": len(tree.touched_paths),
                # DERIVED, not observed: a path convention, and a path is chosen
                # by the tool. Carried with its rule so a renderer can show the
                # difference between this and a session's own claim about itself.
                "session": tree.session,
                "session_evidence": tree.session_evidence,
            },
        ))
        for path in sorted(tree.touched_paths):
            if path not in seen_paths:
                seen_paths.add(path)
                nodes.append(Node(id=_path_id(path), kind=NODE_PATH, label=path,
                                  provenance=OBSERVED))
            edges.append(Edge(source=node_id, target=_path_id(path),
                              kind=EDGE_TOUCHES, provenance=OBSERVED,
                              label="touches"))

    by_path = {t.path.replace("\\", "/").rstrip("/"): n.id
               for t, n in zip(mesh.worktrees, [n for n in nodes
                                                if n.kind == NODE_WORKTREE])}
    for contention in mesh.contentions:
        left = by_path.get(contention.left.replace("\\", "/").rstrip("/"))
        right = by_path.get(contention.right.replace("\\", "/").rstrip("/"))
        if left and right:
            edges.append(Edge(
                source=left, target=right, kind=EDGE_CONTENDS,
                provenance=DERIVED, rule=EDGE_RULES[EDGE_CONTENDS],
                label=f"{contention.count} shared path(s)",
                weight=contention.count))

    if cache is not None:
        for declaration in cache.declarations():
            worktree = next((i for i in cache.identities()
                             if i.key == declaration.key), None)
            session_id = f"session:{declaration.session_id}"
            if not any(n.id == session_id for n in nodes):
                nodes.append(Node(
                    id=session_id, kind=NODE_SESSION,
                    label=declaration.session_id, provenance=DECLARED,
                    detail={"model": declaration.model or UNATTRIBUTED,
                            "self_reported": True}))
            target = (f"worktree:{worktree.key}" if worktree
                      else f"worktree:{declaration.key}")
            edges.append(Edge(source=session_id, target=target,
                              kind=EDGE_CLAIMS, provenance=DECLARED,
                              label="claims (self-reported)"))

    return GraphProjection(
        schema=GRAPH_SCHEMA,
        nodes=tuple(nodes),
        edges=tuple(edges),
        limits=mesh.limits + (
            "Every DECLARED node and edge is self-reported by a session about "
            "itself. It is drawn because it is useful, not because it is checked.",
            "A DERIVED edge is only as good as its stated rule; see edge_rules.",
        ),
    )


def report(projection: GraphProjection, mesh: WorktreeMesh,
           operations: tuple[Operation, ...] = ()) -> str:
    """A plain-Markdown findings summary of one observation.

    Graphify ships a GRAPH_REPORT.md beside its graph because a picture answers
    "what is the shape" and a reader usually arrives with "what should I look
    at". This is that: the collisions, the stale, the unattributed, in that order.
    """
    lines = ["# Worktree mesh report", "",
             f"Repository: `{mesh.repo_root}`", ""]

    worktrees = [n for n in projection.nodes if n.kind == NODE_WORKTREE]
    lines += [f"- {len(worktrees)} worktree(s), "
              f"{len([n for n in worktrees if not n.detail.get('primary')])} "
              f"from agents",
              f"- {len(projection.edges)} edge(s) across "
              f"{len(set(e.provenance for e in projection.edges))} provenance "
              f"level(s)", ""]

    contentions = [e for e in projection.edges if e.kind == EDGE_CONTENDS]
    lines.append("## Collisions")
    lines.append("")
    if contentions:
        lines.append("Two worktrees holding work on one path. Path-level, not "
                     "semantic — see the rule below.")
        lines.append("")
        for edge in sorted(contentions, key=lambda e: -e.weight):
            left = next(n.label for n in projection.nodes if n.id == edge.source)
            right = next(n.label for n in projection.nodes if n.id == edge.target)
            lines.append(f"- **{left} ↔ {right}** — {edge.label}")
        lines.append("")
        lines.append(f"> rule: {EDGE_RULES[EDGE_CONTENDS]}")
    else:
        lines.append("None. No two worktrees hold work on the same path.")
    lines.append("")

    hubs = [(node_id, degree) for node_id, degree in projection.hubs
            if degree > 1 and node_id.startswith("path:")]
    lines += ["## Most contested paths", ""]
    if hubs:
        lines.append("Files several worktrees are changing at once.")
        lines.append("")
        for node_id, degree in hubs[:10]:
            lines.append(f"- `{node_id[len('path:'):]}` — {degree} worktrees")
    else:
        lines.append("None — no path is touched by more than one worktree.")
    lines.append("")

    stale = mesh.stale(older_than_days=7.0, now=mesh.observed_at)
    lines += ["## Stale", ""]
    lines.append("\n".join(
        f"- `{t.label}` — {t.age_days(mesh.observed_at):.0f} days"
        for t in stale) if stale else "None older than 7 days.")
    lines.append("")

    unattributed = [t for t in mesh.agent_worktrees
                    if t.tool_family == UNATTRIBUTED]
    lines += ["## Unattributed", ""]
    lines.append("\n".join(f"- `{t.label}` — {t.tool_evidence}"
                           for t in unattributed)
                 if unattributed else "None — every agent worktree matched a "
                                      "known directory convention.")
    lines.append("")

    if operations:
        lines += ["## Since the last observation", ""]
        for operation in operations[:20]:
            lines.append(f"- `{operation.kind}` — {operation.detail}")
        lines.append("")

    lines += ["## What this cannot tell you", ""]
    lines += [f"- {limit}" for limit in projection.limits]
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "DECLARED", "DERIVED", "EDGE_CLAIMS", "EDGE_CONTENDS", "EDGE_RULES",
    "EDGE_TOUCHES", "GRAPH_SCHEMA", "NODE_PATH", "NODE_SESSION", "NODE_WORKTREE",
    "OBSERVED", "PROVENANCE", "Edge", "GraphProjection", "Node", "project",
    "report",
]
