"""Where in a codebase a piece of work is — the mesh's coordinate space.

[DYNAMIC-CACHE.md](../../../docs/features/DYNAMIC-CACHE.md) §9 open question 2
asks for a canonical coordinate vocabulary and warns that it is "the single
decision most likely to need a version bump later, so it is worth the most care
now". This module answers it, and the care taken is in *not inventing the axes*.

**The axes belong to the repository being observed, not to this module.**

That sentence used to be false. Version 3 of this file carried one hard-coded
table of path prefixes — `alelyon/runtime/common/`, `research/`, `tools/`,
`web/` — copied from the ownership pillars declared in this repository's own
`AGENTS.md`. Inside this repository that was exactly right, and it is why the
rule "validate attribution against an independently-held invariant rather than
against the shape of what the writer emitted" was satisfiable at all: the list
was owner-authored policy the module did not get to edit.

Outside this repository it was wrong in two directions at once. Every path in a
stranger's checkout fell through the table and read `UNMAPPED`, so the fleet
could see no coordinates at all; and the table itself published one particular
private tree's directory layout as though it were a general vocabulary. The
module now ships with **no directory table of its own**. What it has instead is
a resolver:

    declared   `.alelyon/fleet.toml` in the repository root, if present.
               The repository states its own pillars. This repository does
               exactly that, which is why nothing about its behaviour changed
               when the built-in table was removed.
    discovered otherwise, from the repository's OWN tracked paths — the
               directories that actually exist in the checkout the user opened.
    empty      when neither is available. Everything is `UNMAPPED`, which is an
               answer, and a truthful one: nothing was observed, so nothing is
               placed.

Nothing is ever placed into a directory that was not observed or declared in the
repository under observation. There is no ambient list.

Two axes, and the second is deliberately shallow
------------------------------------------------

    Area(pillar="runtime.oracle", surface="assistant")
    Area(pillar="frontend",       surface="desktop.lattice")

`pillar` is the owner. `surface` is the subsystem inside it, taken from the path
at a depth declared per pillar — because pillars are not shaped alike. One
segment under a source pillar may name a subsystem; one segment under a UI
pillar may name only a *toolkit*, and the thing a reader cares about is a level
further down. Depth is therefore a per-rule field rather than a constant.

`FLAT` (depth 0) is the third shape: a pillar that is a bag of independent
programs rather than a tree of subsystems, where the FILE is the unit. It exists
because a directory of unrelated scripts collapsed to one surface makes a single
session editing a single script read as occupying the whole directory, and
`open-areas` then hides every other program in it.

**`UNMAPPED` is first-class**, for the same reason `UNATTRIBUTED` is in
`worktree.py`: a path that falls outside every rule must say so. Rounding it
into a nearest-neighbour pillar would put work in an area nobody owns and report
the fleet as covering ground it is not on.

**What an area is not.** It is not a measure of size, difficulty, or importance,
and two areas being distinct says nothing about whether work in them can conflict
— that is what the mesh's touched-path contention is for. An area is a
coordinate, not a judgement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Optional, Sequence, Tuple
import os
import subprocess

from alelyon.runtime.common import toolpath

#: Extending the pillar table or changing a surface depth changes what an area
#: *means*, and stored areas from before the change would silently re-point.
#: DYNAMIC-CACHE.md §4 rule 1 makes that a version bump rather than a refactor.
#:
#: 2 — `alelyon/languages/` and `alelyon/studio/` added.
#: 3 — flat pillars split per file.
#: 4 — the built-in table REMOVED. Rules now come from the observed repository:
#:     declared in `.alelyon/fleet.toml`, or discovered from its tracked paths.
#:     This is a re-point for every repository that is not this one — they moved
#:     from "everything UNMAPPED" to real coordinates — which is exactly the case
#:     the rule above is for. Within this repository the declared config
#:     reproduces version 3 exactly, and `tests/runtime/test_worktree_areas.py`
#:     asserts that path-by-path.
#: 5 — this repository's `web/` rule REPOINTED to `www/`. The Streamlit surface
#:     it named moved to `alelyon/frontend/web/` in `33c650c6` and has resolved
#:     as `frontend/web` ever since, so the rule placed nothing while the
#:     corporate website — public, claim-bearing, Tier 2 — was UNMAPPED. A
#:     re-point rather than an addition: paths under `www/` that stored
#:     `UNMAPPED` now store `web/src`, which is exactly the case rule 1 above
#:     asks for a bump for. Only this repository's own declared space moves; a
#:     stranger's checkout is unaffected because it never held this rule.
AREA_SPACE_VERSION = 5

#: Surface depth for a pillar that is a bag of files rather than a tree of
#: subsystems. Named rather than written as a bare 0 at every call site.
FLAT = 0

#: A path no rule placed. Never an empty string and never a nearest guess.
UNMAPPED = "UNMAPPED"

#: Where a repository declares its own coordinate space, relative to its root.
CONFIG_PATH = ".alelyon/fleet.toml"

#: Overrides the config location entirely. A path to a TOML file.
CONFIG_ENV = "ALELYON_FLEET_CONFIG"

#: Discovery walks the repository's own tracked paths. This bounds that read so
#: an enormous checkout cannot turn a coordinate lookup into a minute of I/O.
_MAX_DISCOVERY_PATHS = 200_000

_GIT_TIMEOUT = 30


@dataclass(frozen=True, order=True)
class Area:
    """One coordinate in the mesh's space: who owns it, and which part."""

    pillar: str
    surface: str

    def __str__(self) -> str:
        if self.pillar == UNMAPPED:
            return UNMAPPED
        return f"{self.pillar}/{self.surface}" if self.surface else self.pillar

    @property
    def mapped(self) -> bool:
        return self.pillar != UNMAPPED

    # `known` and `tier3` are properties of an area WITHIN A SPACE, and the
    # space is what varies between repositories. They are kept on `Area` because
    # every caller reads them as adjectives of a coordinate, and they consult
    # the process default — correct for a tool standing in the repository it is
    # asking about. A caller holding paths from elsewhere must use
    # `AreaSpace.known` / `AreaSpace.tier3` with that repository's own space.

    @property
    def known(self) -> bool:
        """Whether a path in the DEFAULT space could resolve to this area."""
        return default_space().known(self)

    @property
    def tier3(self) -> bool:
        """Whether the DEFAULT space declares this area owner-authority-only."""
        return default_space().tier3(self)


#: The single instance every unplaced path resolves to, so callers can compare
#: identity rather than remembering to test two fields.
UNMAPPED_AREA = Area(UNMAPPED, "")


@dataclass(frozen=True)
class Rule:
    """One path prefix, the pillar that owns it, and how deep its surface is.

    `depth` is how many path segments AFTER the prefix name the surface:
      1 — the segment under the prefix is already a subsystem
      2 — the first segment is a container and the second is the real subject
      0 — FLAT: the pillar is a bag of independent files, so the FILE is the
          unit and its stem is the surface
    """

    prefix: str
    pillar: str
    depth: int = 1

    def __post_init__(self) -> None:
        if not self.prefix:
            raise ValueError("a rule prefix cannot be empty")
        if not self.pillar or self.pillar == UNMAPPED:
            raise ValueError(f"{self.pillar!r} is not a usable pillar name")
        if self.depth < 0:
            raise ValueError("surface depth cannot be negative")


@dataclass(frozen=True)
class AreaSpace:
    """The coordinate vocabulary of ONE repository.

    Explicit and threadable. A bus, a mesh or a report over repository X should
    hold X's space; the module-level convenience functions delegate to a default
    resolved from the current checkout, which is correct for a tool run inside
    the repository it is asking about and wrong for anything else.

    `evidence` records how the rules were obtained, in the same spirit as
    `worktree.tool_evidence`: a reader can disagree with a derivation only if it
    is told which one ran.
    """

    #: Ordered; first match wins, so a longer prefix must precede the shorter
    #: one it extends. `normalised()` enforces that rather than trusting it.
    rules: Tuple[Rule, ...] = ()
    #: Whole pillars that need explicit owner authority — capital, destructive,
    #: trust, or release. A surface that offers free work must withhold these.
    #: Empty by default: this module does not get to decide that a stranger's
    #: directory is dangerous, and inventing a guess would be worse than none.
    tier3_pillars: frozenset = frozenset()
    #: Individual `(pillar, surface)` pairs that are Tier 3 while their pillar
    #: is not. Needed because danger is not always pillar-shaped.
    tier3_areas: frozenset = frozenset()
    evidence: str = "no rules; every path is UNMAPPED"

    @property
    def known_pillars(self) -> frozenset:
        """Every pillar a path could actually resolve to.

        Derived from the rules rather than stored twice, so a pillar cannot be
        accepted by one and rejected by the other.
        """
        return frozenset(rule.pillar for rule in self.rules)

    @property
    def empty(self) -> bool:
        return not self.rules

    def area_of(self, path: str) -> Area:
        """The area one repository-relative path belongs to.

        Accepts either slash convention because git reports forward slashes on
        every platform while `str(Path(...))` does not, and a caller mixing the
        two is the likeliest way this gets a false `UNMAPPED`.
        """
        posix = _normalise(path)
        if not posix:
            return UNMAPPED_AREA
        for rule in self.rules:
            if not posix.startswith(rule.prefix):
                continue
            rest = PurePosixPath(posix[len(rule.prefix):])
            if rule.depth == FLAT:
                # The file is the unit. A path with directories under a flat
                # pillar takes the first directory instead, so `tools/pkg/mod.py`
                # groups as `tools/pkg` rather than splitting a package apart.
                parts = list(rest.parts)
                if not parts:
                    return Area(rule.pillar, "")
                if len(parts) > 1:
                    return Area(rule.pillar, parts[0])
                return Area(rule.pillar, PurePosixPath(parts[0]).stem)
            # Segments that name a directory are all parts except the final one
            # only when the path has a suffix; a caller may pass either, so take
            # parts and trim the filename when there is one.
            segments = list(rest.parts)
            if segments and "." in segments[-1]:
                segments = segments[:-1]
            # A dot-directory is configuration, not a subsystem. Left in, it
            # produces a phantom area with a doubled separator that no session
            # will ever work in, offered in the free-work list beside real ones.
            segments = [s for s in segments if not s.startswith(".")]
            return Area(rule.pillar, ".".join(segments[:rule.depth]))
        return UNMAPPED_AREA

    def areas_of(self, paths: Optional[Iterable[str]]) -> Tuple[Area, ...]:
        """Distinct areas covered by a set of paths, in sorted order.

        `UNMAPPED` is included when any path fell outside the rules. Dropping it
        here is how a fleet report ends up claiming full coverage of work it
        never placed.
        """
        return tuple(sorted({self.area_of(p) for p in (paths or ())}))

    def all_pillars(self) -> Tuple[str, ...]:
        """Every declared pillar, once, in rule order."""
        seen: list = []
        for rule in self.rules:
            if rule.pillar not in seen:
                seen.append(rule.pillar)
        return tuple(seen)

    def surfaces_in(self, pillar: str, paths) -> Tuple[Area, ...]:
        """The areas of `pillar` that `paths` actually reach."""
        return tuple(sorted({a for a in self.areas_of(paths)
                             if a.pillar == pillar}))

    def known(self, area: Area) -> bool:
        """Whether a path could ever actually resolve to `area`.

        `Area.mapped` only says "not the UNMAPPED sentinel", which any string
        satisfies — `parse_area` partitions on `/` and hands back whatever it
        was given. So a typo produces a well-formed Area on a pillar that
        appears in no path, and anything keyed on it is invisible for ever after.

        That is not hypothetical. A session claimed `platform.gateway` — the
        dotted form, because pillars really do contain dots — while routing
        derives `platform/gateway`. The claim was accepted and reported as
        success, the session was never reachable, and four findings published at
        the real area reported REACHED NOBODY. Creating a coordinate must
        therefore be checked against the space; reading one back must not, so a
        bad record can still be inspected and released.
        """
        return area.pillar in self.known_pillars

    def derivable(self, area: Area, surfaces) -> bool:
        """Whether a path in this repository actually derives `area`.

        `known` checks the PILLAR and stops there, which is why it accepted
        `runtime.common/session_activity`: `runtime.common` is a real pillar, so
        the surface was never examined. But that pillar's rule has depth 1 over a
        directory with no subdirectories, so `area_of` trims the filename and
        every file under it derives `runtime.common` with an empty surface. No
        path can produce that coordinate, and a claim on it is invisible to
        routing while reporting success — the `platform.gateway` failure one rung
        down, where the pillar is right and only the surface is imagined.

        Measured on this repository at `9ec3d59`: 8 of 39 active claims were in
        that state, all of the `runtime.common/<file>` form. The habit is
        imported from `tools/`, which is declared FLAT — "the file is the unit" —
        where `tools/relay` genuinely is derivable.

        `surfaces` is what `observed_surfaces` returns. **An empty set means the
        surfaces were never observed, not that none exist**, so this answers True
        rather than refusing everything on missing evidence — the same direction
        `tier3` takes for an undeclared space, and for the same reason.
        """
        if not surfaces:
            return True
        if not area.surface:
            return area.pillar in {pillar for pillar, _s in surfaces}
        return (area.pillar, area.surface) in surfaces

    def tier3(self, area: Area) -> bool:
        """Work here needs explicit owner authority.

        Covers capital, destructive, **trust** and release authority — not only
        order paths. A surface offering free work must withhold these. What
        counts is declared by the repository; an undeclared space returns False
        everywhere, and `open-areas` says so rather than implying a clearance.
        """
        return (area.pillar in self.tier3_pillars
                or (area.pillar, area.surface) in self.tier3_areas)

    def suggest(self, text: str) -> Optional[Area]:
        """A known area the caller plausibly meant, or None.

        Exists for one confusion specifically, because it is the one that
        actually happened and it is built into the vocabulary rather than being
        carelessness: a pillar may contain a dot (`runtime.common`), and a
        pillar and its surface are joined by a slash (`platform/gateway`). Both
        separators are legitimate, so `platform.gateway` looks entirely
        reasonable and resolves to nothing.

        Refusing without a suggestion leaves the caller re-reading a vocabulary
        list to spot a single character.
        """
        raw = (text or "").strip()
        if not raw or self.known(parse_area(raw)):
            return None
        candidates: list = []
        if "/" not in raw and "." in raw:
            head, _, tail = raw.rpartition(".")          # platform.gateway
            candidates.append(f"{head}/{tail}")          # -> platform/gateway
        if "/" in raw:
            candidates.append(raw.replace("/", ".", 1))  # runtime/common
            candidates.append(raw.partition("/")[0])     # bare pillar
        for candidate in candidates:
            area = parse_area(candidate)
            if self.known(area):
                return area
        return None

    def normalised(self) -> "AreaSpace":
        """Longest prefix first, so an extending rule cannot be shadowed.

        Order is load-bearing — `first match wins` — and a hand-written config
        that lists `alelyon/` before `alelyon/runtime/common/` would silently
        swallow the more specific rule. Sorting by descending prefix length
        makes the file order irrelevant instead of making it a trap. Ties keep
        their declared order, so a repository can still express a deliberate
        preference between two same-length prefixes.
        """
        ordered = sorted(enumerate(self.rules),
                         key=lambda pair: (-len(pair[1].prefix), pair[0]))
        return AreaSpace(
            rules=tuple(rule for _index, rule in ordered),
            tier3_pillars=self.tier3_pillars,
            tier3_areas=self.tier3_areas,
            evidence=self.evidence,
        )


#: The space every module-level call falls back to when none was threaded in.
EMPTY_SPACE = AreaSpace()


def _normalise(path: str) -> str:
    posix = str(path or "").replace("\\", "/").strip()
    while posix.startswith("./"):
        posix = posix[2:]
    return posix.lstrip("/")


def parse_area(text: str) -> Area:
    """Read back what `str(Area)` wrote. `UNMAPPED` round-trips.

    Deliberately space-free: a stored record must remain readable even when the
    space that produced it is gone, otherwise a coordinate written yesterday
    cannot be released today.
    """
    raw = (text or "").strip()
    if not raw or raw == UNMAPPED:
        return UNMAPPED_AREA
    pillar, _, surface = raw.partition("/")
    return Area(pillar, surface)


# ── discovery: the repository's own directories, and nothing else ───────────


def discover(paths: Sequence[str], *, source: str = "tracked paths") -> AreaSpace:
    """Build a coordinate space from the paths a repository actually contains.

    Pure: takes path strings, returns a space, reads nothing. `load()` supplies
    the strings from `git ls-files`, and a caller with its own listing — a
    directory the user selected, an export manifest — can pass that instead.

    The rules, stated plainly so a reader can disagree with them:

    * **A pillar is a top-level directory.** Files at the repository root are
      not placed at all rather than being swept into a `(root)` pillar that
      would then read as somewhere a session could work.
    * **Depth is 1 by default** — the segment under the pillar names a surface.
    * **A pillar whose immediate children are all files is FLAT**, because a
      directory of unrelated programs is a bag, not a tree, and collapsing it
      to one surface makes one session look like it occupies all of it.
    * **Dot-directories are skipped.** `.github/` and `.venv/` are
      configuration and tooling, not subsystems anyone claims.

    Nothing here consults a list of names. A directory is a pillar because it
    was observed in this checkout, and for no other reason.
    """
    immediate_dirs: dict = {}
    order: list = []
    seen = 0
    for raw in paths or ():
        seen += 1
        if seen > _MAX_DISCOVERY_PATHS:
            break
        posix = _normalise(raw)
        if not posix:
            continue
        parts = posix.split("/")
        if len(parts) < 2:
            continue                      # a root file names no pillar
        top = parts[0]
        if not top or top.startswith("."):
            continue
        if top not in immediate_dirs:
            immediate_dirs[top] = set()
            order.append(top)
        if len(parts) > 2:
            immediate_dirs[top].add(parts[1])

    rules = tuple(
        Rule(prefix=f"{top}/", pillar=top,
             depth=1 if immediate_dirs[top] else FLAT)
        for top in order
    )
    return AreaSpace(
        rules=rules,
        evidence=(f"discovered from {min(seen, _MAX_DISCOVERY_PATHS)} "
                  f"{source}: {len(rules)} top-level directories"),
    ).normalised()


# ── declaration: the repository states its own space ────────────────────────


def from_config(data: Mapping, *, source: str = "configuration") -> AreaSpace:
    """Parse a declared coordinate space.

    The shape, which `.alelyon/fleet.toml` writes as TOML:

        [[area]]
        prefix = "src/engine/"
        pillar = "engine"
        depth  = 1

        [tier3]
        pillars = ["release"]
        areas   = [["engine", "keys"]]

    A malformed entry raises rather than being skipped. A rule silently dropped
    is a directory that stops being tracked without anyone being told, which is
    the failure this whole module exists to avoid.
    """
    raw_rules = data.get("area") or data.get("areas") or ()
    if isinstance(raw_rules, Mapping):
        raw_rules = [raw_rules]
    rules = []
    for entry in raw_rules:
        if not isinstance(entry, Mapping):
            raise ValueError(f"area entry must be a table, got {entry!r}")
        prefix = str(entry.get("prefix", "")).replace("\\", "/")
        if prefix and not prefix.endswith("/") and "." not in Path(prefix).name:
            prefix += "/"
        rules.append(Rule(prefix=prefix,
                          pillar=str(entry.get("pillar", "")),
                          depth=int(entry.get("depth", 1))))
    tier3 = data.get("tier3") or {}
    if not isinstance(tier3, Mapping):
        raise ValueError("[tier3] must be a table")
    pillars = frozenset(str(p) for p in (tier3.get("pillars") or ()))
    areas = frozenset(
        (str(pair[0]), str(pair[1]))
        for pair in (tier3.get("areas") or ())
        if isinstance(pair, (list, tuple)) and len(pair) == 2
    )
    return AreaSpace(
        rules=tuple(rules),
        tier3_pillars=pillars,
        tier3_areas=areas,
        evidence=f"declared in {source}: {len(rules)} rules",
    ).normalised()


def config_path(repo_root: Optional[str] = None) -> Optional[Path]:
    """The configuration file this repository would use, if it has one."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    if repo_root is None:
        return None
    candidate = Path(repo_root) / CONFIG_PATH
    return candidate if candidate.is_file() else None


def _read_config(path: Path) -> AreaSpace:
    import tomllib                      # stdlib since 3.11; read-only here
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return from_config(data, source=str(path))


def _tracked_paths(repo_root: str) -> Sequence[str]:
    try:
        probe = subprocess.run(
            toolpath.argv("git", "ls-files"), cwd=repo_root, check=False,
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
            **toolpath.no_window())
    except (OSError, subprocess.SubprocessError):
        return ()
    if probe.returncode != 0:
        return ()
    return probe.stdout.splitlines()


def repo_root_of(start: Optional[str] = None) -> Optional[str]:
    """The git checkout containing `start`, or None.

    None rather than a guess: a directory that is not in a repository has no
    coordinate space, and inventing one from the filesystem would track
    directories the user never opened.
    """
    try:
        probe = subprocess.run(
            toolpath.argv("git", "rev-parse", "--show-toplevel"),
            cwd=str(start or Path.cwd()), check=False,
            capture_output=True, text=True, timeout=_GIT_TIMEOUT,
            **toolpath.no_window())
    except (OSError, subprocess.SubprocessError):
        return None
    root = probe.stdout.strip()
    return root if probe.returncode == 0 and root else None


def load(repo_root: Optional[str] = None) -> AreaSpace:
    """The coordinate space of one repository: declared, else discovered.

    `repo_root=None` means "the checkout this process is standing in", resolved
    with git. Where there is no checkout the answer is `EMPTY_SPACE` — every
    path `UNMAPPED` — because the alternative is to place work into directories
    nobody opened.
    """
    root = repo_root if repo_root is not None else repo_root_of()
    declared = config_path(root)
    if declared is not None:
        return _read_config(declared)
    if root is None:
        return EMPTY_SPACE
    listing = _tracked_paths(root)
    if not listing:
        return AreaSpace(evidence=f"{root} lists no tracked files; nothing placed")
    return discover(listing, source=f"tracked paths in {root}")


# ── the process default, for callers that hold no space ─────────────────────
#
# Module-level `area_of(path)` has to answer without being told which repository
# the path came from. It resolves the checkout this process is standing in, once,
# and caches it. That is correct for a tool run inside the repository it is
# asking about, which is every current caller, and it is wrong for anything
# holding paths from elsewhere — so a space is threadable everywhere it matters
# and `set_default_space` exists for a host that knows better.

_DEFAULT: Optional[AreaSpace] = None


def default_space() -> AreaSpace:
    """The space module-level helpers use, resolved on first need."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = load()
    return _DEFAULT


def set_default_space(space: Optional[AreaSpace]) -> None:
    """Install a space (or `None` to force re-resolution on next use)."""
    global _DEFAULT
    _DEFAULT = space
    _SURFACES.clear()


def surfaces_of(paths: Iterable[str],
                space: Optional[AreaSpace] = None) -> frozenset:
    """Every `(pillar, surface)` some path in `paths` actually derives.

    Pure, and the whole of the rule: a coordinate exists because a path produces
    it, never because it is well-formed. `UNMAPPED` is excluded — it is the
    answer for a path the rules do not place, not a surface anyone works in.
    """
    resolved = space or default_space()
    return frozenset((area.pillar, area.surface)
                     for area in resolved.areas_of(paths) if area.mapped)


#: `observed_surfaces` per repository root. Cached because the answer costs a
#: `git ls-files` and does not change within one command.
_SURFACES: dict = {}


def observed_surfaces(repo_root: Optional[str] = None, *,
                      space: Optional[AreaSpace] = None) -> frozenset:
    """The coordinates this repository's tracked paths actually derive.

    Not folded into `load()` on purpose. A **declared** space is read from
    `.alelyon/fleet.toml` and costs no git call at all, and making every process
    that resolves one path pay for `git ls-files` to satisfy a check only the
    claim path performs would be a poor trade. This is therefore asked for
    explicitly, by the one caller that is creating a coordinate rather than
    reading one back.

    Returns an empty set when there is no checkout or nothing is tracked, which
    every caller must read as **"not observed"** and never as "nothing exists".
    """
    root = repo_root if repo_root is not None else repo_root_of()
    if root is None:
        return frozenset()
    key = str(root)
    if key not in _SURFACES:
        _SURFACES[key] = surfaces_of(_tracked_paths(key), space)
    return _SURFACES[key]


def area_of(path: str, space: Optional[AreaSpace] = None) -> Area:
    """The area one repository-relative path belongs to."""
    return (space or default_space()).area_of(path)


def areas_of(paths, space: Optional[AreaSpace] = None) -> Tuple[Area, ...]:
    """Distinct areas covered by a set of paths, in sorted order."""
    return (space or default_space()).areas_of(paths)


def all_pillars(space: Optional[AreaSpace] = None) -> Tuple[str, ...]:
    """Every declared pillar, once, in rule order."""
    return (space or default_space()).all_pillars()


def surfaces_in(pillar: str, paths,
                space: Optional[AreaSpace] = None) -> Tuple[Area, ...]:
    """The areas of `pillar` that `paths` actually reach."""
    return (space or default_space()).surfaces_in(pillar, paths)


def suggest_area(text: str, space: Optional[AreaSpace] = None) -> Optional[Area]:
    """A known area the caller plausibly meant, or None."""
    return (space or default_space()).suggest(text)


def known_pillars(space: Optional[AreaSpace] = None) -> frozenset:
    """Every pillar a path in this repository could resolve to."""
    return (space or default_space()).known_pillars
