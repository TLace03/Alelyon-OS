"""What KIND of work a path is, as distinct from where it lives.

`worktree_areas` answers *where* — `Area(pillar, surface)`, the mesh's existing
coordinate. This answers the second question a team needs and the mesh has never
had: **what discipline does touching this require?**

Both questions are needed because neither implies the other. A cryptographer and
an API developer editing one gateway directory are one `Area` and two
disciplines; a certification change in one pillar and one in another are two
`Area`s and one discipline. With only the first axis, a `defect-found` about a
signed width routes to whoever happens to have that file open rather than to
whoever owns certification.

The axes belong to the repository, not to this module
-----------------------------------------------------
`worktree_areas` earned its coordinate space by *not* inventing the pillars: it
reads them from the repository under observation, which states its own. The same
discipline is available for this axis and is used the same way — and for the
same reason it had to be. This module ships in the public `alelyon-os` wheel,
and version 1 carried five disciplines hard-coded out of one private
repository's `AGENTS.md`, with its globs, its section numbers and its incident
history baked in. In that repository the list was owner-authored policy the
module did not get to edit, which is exactly what made it trustworthy. Anywhere
else it was a stranger's policy document asserting rules over directories its
author had never seen.

So the table moved to `.alelyon/fleet.toml` in the repository that owns it, and
this module became the mechanism. A repository that declares no disciplines gets
`UNSPECIALISED` for every path, which is an honest answer: no rule was stated,
so no rule matched. It is emphatically **not** a clearance, and `LIMITS` says so
at every read.

A `CODEOWNERS` file is the obvious alternative and is not equivalent: CODEOWNERS
names *people*, and a discipline is a kind of rule. A repository with a real
review-routing file should keep using it; this axis answers a different question.

Two trigger kinds, and only one of them is a path
--------------------------------------------------
This is the load-bearing distinction and it is the reason this module cannot be
a gate.

* `TRIGGER_PATHS` — the policy names the files it governs. A path match here is
  **DERIVED** and exact.
* `TRIGGER_REACHABILITY` — the policy names what the code must *reach* ("any
  code that can reach the order path"). That is a property of the import graph,
  not of a path string, and **this module cannot compute it**. It matches only
  the anchors the policy names by path.

So for reachability-triggered disciplines the answer **UNDER-approximates**: it
returns fewer disciplines than really apply, never more. That direction is safe
for routing a finding to an interested party and **unsafe for gating a change**,
so gating stays where it already is — the policy document itself, the CI
workflows, and a reviewer. `LIMITS` says so at every read.

Computing reachability properly needs the import graph. That is named as an
INFRA GAP rather than papered over with plausible globs, because a guessed glob
would report a discipline as *checked* on ground nobody checked.

`UNSPECIALISED` is first class
------------------------------
Most of a repository needs no specialist rule, and saying so is an answer. A
path that fires nothing returns an empty result and `UNSPECIALISED`, never a
nearest-neighbour discipline — the same reason `worktree_areas` keeps `UNMAPPED`
and `worktree` keeps `UNATTRIBUTED`. Rounding a path into the nearest discipline
would report specialist coverage over ground no specialist rule governs.

Pure and read-only once a space is held: takes path strings, returns records,
runs no git. Loading a space reads one declared file.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Mapping, Optional, Tuple
import os

#: Bump when the SHAPE of a discipline changes — a field added or removed, or a
#: trigger kind redefined. Stored discipline strings from before a change would
#: otherwise silently re-point, which DYNAMIC-CACHE.md §4 rule 1 makes a version
#: bump rather than a refactor.
#:
#: 1 — initial. Five disciplines hard-coded from this repository's AGENTS.md.
#: 2 — the built-in table REMOVED. Disciplines are declared by the repository
#:     under observation, in `.alelyon/fleet.toml`. Ids and triggers are
#:     unchanged where this repository declares them, so no stored record
#:     re-points here; every other repository moved from "five rules about
#:     somebody else's directories" to its own or to none.
DISCIPLINE_SPACE_VERSION = 2

#: A path that needs no specialist rule. A value, never a blank or a guess.
UNSPECIALISED = "UNSPECIALISED"

#: The policy states the files it governs. A match is exact.
TRIGGER_PATHS = "paths"
#: The policy states what the code must REACH. Not computable from a path, so
#: only the anchors the policy names by path are matched. Under-approximates.
TRIGGER_REACHABILITY = "reachability"

TRIGGERS = (TRIGGER_PATHS, TRIGGER_REACHABILITY)


@dataclass(frozen=True, order=True)
class Discipline:
    """One kind of specialist work, and the policy that defines it."""

    #: Short stable id. Used in stored records, so renaming one is a version bump.
    id: str
    #: The policy section number this comes from, or 0 where the repository's
    #: policy is unnumbered. Provenance, not a label.
    section: int
    #: The section's heading, verbatim, so a test in the owning repository can
    #: fail when the document is retitled.
    title: str
    #: TRIGGER_PATHS or TRIGGER_REACHABILITY — see the module docstring.
    trigger: str
    #: Globs the policy names literally. Never a glob invented here.
    paths: Tuple[str, ...]
    #: Why this discipline matches what it matches, quoting the policy where the
    #: policy is what decides. Carried so a reader can disagree with it.
    evidence: str
    #: The document the rule lives in, e.g. "AGENTS.md". Printed beside the
    #: section number so a reader in another repository is not shown a bare
    #: number pointing at a file this module has never named.
    policy: str = ""

    def __str__(self) -> str:
        return self.id

    @property
    def exact(self) -> bool:
        """Whether a path match is the whole answer for this discipline."""
        return self.trigger == TRIGGER_PATHS

    @property
    def citation(self) -> str:
        """Human reference to where the rule is written."""
        if self.policy and self.section:
            return f"{self.policy} section {self.section}"
        if self.policy:
            return self.policy
        if self.section:
            return f"section {self.section}"
        return "undeclared policy"


#: True of every discipline space, whatever a repository declares. A repository
#: may add its own; it may not remove these, because they describe the mechanism
#: rather than the contents.
BASE_LIMITS: Tuple[str, ...] = (
    "This is a routing hint, NOT a gate. A discipline triggered by what code "
    "can REACH rather than by where it lives is under-matched here, because "
    "reach is an import-graph property this module does not compute - so the "
    "answer names fewer disciplines than really apply. Gating stays with the "
    "policy document, the CI workflows and a reviewer.",
    "Under-approximation is the safe direction for routing and the unsafe one "
    "for enforcement. A path returning no discipline has NOT been cleared of "
    "one; it means no rule stated as a path matched it.",
    "Only a discipline whose policy lists its files literally is exact. Treat "
    "every reachability-triggered one as 'at least this'.",
    "A discipline is not a person and not a permission. It says which "
    "specialist rule applies to a path, never who may change it - risk tier "
    "decides authority, and that is a separate axis again.",
    "A repository that declares no disciplines gets UNSPECIALISED everywhere. "
    "That is the absence of a declaration, not a finding that the work needs "
    "no specialist.",
)


@dataclass(frozen=True)
class DisciplineSpace:
    """The specialist-rule vocabulary of ONE repository."""

    disciplines: Tuple[Discipline, ...] = ()
    #: Limits the repository states on top of `BASE_LIMITS` — a discipline it
    #: knows it cannot express as a path is worth naming rather than omitting.
    extra_limits: Tuple[str, ...] = ()
    evidence: str = "no disciplines declared; every path is UNSPECIALISED"

    @property
    def by_id(self) -> Mapping[str, Discipline]:
        """`{id: Discipline}`, for reading a stored id back."""
        return {d.id: d for d in self.disciplines}

    @property
    def limits(self) -> Tuple[str, ...]:
        return BASE_LIMITS + self.extra_limits

    @property
    def empty(self) -> bool:
        return not self.disciplines

    def of(self, path: str) -> Tuple[Discipline, ...]:
        """Every discipline whose stated trigger matches one path.

        A path can carry several — a test for a sealed store is security work
        and testing work at once — so this returns a tuple and never picks a
        winner. An empty result means no rule *stated as a path* matched, which
        is not the same as the path having been cleared.
        """
        posix = str(path or "").replace("\\", "/").lstrip("/")
        if not posix:
            return ()
        return tuple(sorted(
            discipline for discipline in self.disciplines
            if any(_matches(posix, pattern) for pattern in discipline.paths)))

    def among(self, paths) -> Tuple[Discipline, ...]:
        """Distinct disciplines a set of paths reaches, in sorted order."""
        found = set()
        for path in paths or ():
            found.update(self.of(path))
        return tuple(sorted(found))

    def describe(self, paths) -> str:
        """One line naming the disciplines a change touches, for a CLI."""
        found = self.among(paths)
        if not found:
            if self.empty:
                return (f"{UNSPECIALISED} - this repository declares no "
                        f"disciplines, so nothing could match. That is an "
                        f"absent declaration, not a clearance")
            return (f"{UNSPECIALISED} - no rule stated as a path matched. That "
                    f"is not a clearance: a reachability-triggered discipline "
                    f"is not computed here")
        inexact = [d.id for d in found if not d.exact]
        text = ", ".join(d.id for d in found)
        if not inexact:
            return f"{text} (exact)"
        return f"{text} (at least; {', '.join(inexact)} trigger on reach, not path)"


#: The space every module-level call falls back to when nothing was declared.
EMPTY_SPACE = DisciplineSpace()


def _matches(posix: str, pattern: str) -> bool:
    """Glob match with `**` meaning 'at any depth below'.

    `fnmatch` treats `*` as crossing separators, which would make
    `some/dir/*` match `some/dir/a/b.py` and make the distinction between the
    two forms meaningless. Anchor `**` explicitly instead.
    """
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return posix == prefix or posix.startswith(prefix + "/")
    return fnmatch(posix, pattern)


def from_config(data: Mapping, *, source: str = "configuration") -> DisciplineSpace:
    """Parse a declared discipline space.

    The shape, which `.alelyon/fleet.toml` writes as TOML:

        [[discipline]]
        id       = "cne-claims"
        section  = 8
        policy   = "AGENTS.md"
        title    = "CNE, verifier, reconciliation, and claim rules"
        trigger  = "paths"
        paths    = ["alelyon/verify/**"]
        evidence = "section 8 lists the paths it governs literally"

    A malformed entry raises rather than being skipped. A discipline silently
    dropped is a specialist rule that stops routing without anyone being told.
    """
    raw = data.get("discipline") or data.get("disciplines") or ()
    if isinstance(raw, Mapping):
        raw = [raw]
    out = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError(f"discipline entry must be a table, got {entry!r}")
        trigger = str(entry.get("trigger", TRIGGER_REACHABILITY))
        if trigger not in TRIGGERS:
            raise ValueError(
                f"discipline {entry.get('id')!r} declares trigger {trigger!r}; "
                f"expected one of {TRIGGERS}")
        identifier = str(entry.get("id", "")).strip()
        if not identifier:
            raise ValueError("every discipline needs a stable id")
        out.append(Discipline(
            id=identifier,
            section=int(entry.get("section", 0)),
            title=str(entry.get("title", "")),
            trigger=trigger,
            paths=tuple(str(p) for p in (entry.get("paths") or ())),
            evidence=str(entry.get("evidence", "")),
            policy=str(entry.get("policy", "")),
        ))
    # An explicit `[limits]` table rather than a bare root key. In TOML a bare
    # key written after an array-of-tables belongs to the LAST table in that
    # array, not to the root -- so `discipline_limits = [...]` placed at the end
    # of the file silently became a field of the final discipline, and the
    # limits it declared were never read. A table header is unambiguous wherever
    # it appears in the file.
    declared_limits = data.get("limits") or {}
    if not isinstance(declared_limits, Mapping):
        raise ValueError("[limits] must be a table")
    limits = tuple(str(x) for x in (declared_limits.get("disciplines") or ()))
    return DisciplineSpace(
        disciplines=tuple(out),
        extra_limits=limits,
        evidence=f"declared in {source}: {len(out)} disciplines",
    )


def load(repo_root: Optional[str] = None) -> DisciplineSpace:
    """The discipline space of one repository, or an empty one.

    Reads the same file `worktree_areas` reads, because the two axes describe
    one repository and splitting them across two files is how they come to
    disagree about which repository they are describing.
    """
    from alelyon.runtime.common import worktree_areas as areas

    root = repo_root if repo_root is not None else areas.repo_root_of()
    declared = areas.config_path(root)
    if declared is None:
        return EMPTY_SPACE
    import tomllib
    with Path(declared).open("rb") as handle:
        data = tomllib.load(handle)
    return from_config(data, source=str(declared))


# ── the process default ─────────────────────────────────────────────────────

_DEFAULT: Optional[DisciplineSpace] = None


def default_space() -> DisciplineSpace:
    """The space module-level helpers use, resolved on first need."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load()
    return _DEFAULT


def set_default_space(space: Optional[DisciplineSpace]) -> None:
    """Install a space (or `None` to force re-resolution on next use)."""
    global _DEFAULT
    _DEFAULT = space


def disciplines_of(path: str,
                   space: Optional[DisciplineSpace] = None) -> Tuple[Discipline, ...]:
    """Every discipline whose stated trigger matches one path."""
    return (space or default_space()).of(path)


def disciplines_in(paths,
                   space: Optional[DisciplineSpace] = None) -> Tuple[Discipline, ...]:
    """Distinct disciplines a set of paths reaches, in sorted order."""
    return (space or default_space()).among(paths)


def describe(paths, space: Optional[DisciplineSpace] = None) -> str:
    """One line naming the disciplines a change touches, for a CLI to print."""
    return (space or default_space()).describe(paths)


def limits(space: Optional[DisciplineSpace] = None) -> Tuple[str, ...]:
    """What this axis cannot tell you, for the space in force."""
    return (space or default_space()).limits


#: Names that used to be module constants and are now properties of whichever
#: space is in force. Resolved through PEP 562 so `D.DISCIPLINES` still reads as
#: a table to every existing caller, while the module itself carries none.
_LAZY = {
    "DISCIPLINES": lambda s: s.disciplines,
    "BY_ID": lambda s: dict(s.by_id),
    "LIMITS": lambda s: s.limits,
    #: `{section: title}` for the repository's own policy document, so a test in
    #: that repository can assert the two have not drifted.
    "POLICY_SECTIONS": lambda s: {d.section: d.title for d in s.disciplines
                                  if d.section},
}


def __getattr__(name: str):
    resolve = _LAZY.get(name)
    if resolve is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return resolve(default_space())


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
