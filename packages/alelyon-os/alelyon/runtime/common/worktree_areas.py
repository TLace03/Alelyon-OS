"""Where in the codebase a piece of work is — the mesh's coordinate space.

[DYNAMIC-CACHE.md](../../../docs/features/DYNAMIC-CACHE.md) §9 open question 2
asks for a canonical coordinate vocabulary and warns that it is "the single
decision most likely to need a version bump later, so it is worth the most care
now". This module answers it, and the care taken is in *not inventing the axes*.

**The axes are already binding policy.** `AGENTS.md` §2 declares the ownership
pillars — `runtime.common`, `runtime.atlas`, `runtime.vector`, `runtime.nexus`,
`runtime.sentinel`, `runtime.oracle`, `products`, `frontend`, `platform`,
`research`. That matters more than it looks: §5's rule is that attribution must
be validated against an independently-held invariant rather than against the
shape of what the writer emitted, and this list is exactly such an invariant. It
was fixed before this feature existed, it is owner-authored policy this module
does not get to edit, and `test_pillars_come_from_the_policy_document` fails if
the table here and the list there ever disagree.

An invented taxonomy — "backend", "UI", "infra" — would have had none of those
properties. It would be this module asserting a structure rather than reading one.

**How strongly it is held, precisely.** `.importlinter` machine-enforces the
*outer* boundaries (runtime must not import products or the frontend, and so on),
and AGENTS.md says in the same breath that "intra-runtime direction remains
partly cyclic and is not yet machine-enforced". So `runtime.oracle` versus
`runtime.vector` is a policy boundary with a document behind it, **not** a
checked one. The distinction is worth keeping straight: this coordinate space is
as reliable as a written ownership rule, which is considerably better than a
taxonomy invented here and considerably worse than a compiler.

**Two axes, and the second is deliberately shallow.**

    Area(pillar="runtime.oracle", surface="assistant")
    Area(pillar="frontend",       surface="desktop.lattice")

`pillar` is the owner. `surface` is the subsystem inside it, taken from the path
at a depth declared per pillar — because the pillars are not shaped alike. One
segment under `runtime.oracle` names a subsystem (`assistant`, `dsl`, `desks`);
one segment under `frontend` names only a *toolkit* (`desktop`), and the thing a
reader cares about — which product — is one level further down. The depth is a
table entry rather than a constant for that reason.

**`UNMAPPED` is first-class**, and for the same reason `UNATTRIBUTED` is in
`worktree.py`: a path that falls outside every rule must say so. Rounding it into
a nearest-neighbour pillar would put work in an area nobody owns and report the
fleet as covering ground it is not on.

**What an area is not.** It is not a measure of size, difficulty, or importance,
and two areas being distinct says nothing about whether work in them can conflict
— that is what the mesh's touched-path contention is for. An area is a coordinate,
not a judgement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

#: Extending the pillar table or changing a surface depth changes what an area
#: *means*, and stored areas from before the change would silently re-point.
#: DYNAMIC-CACHE.md §4 rule 1 makes that a version bump rather than a refactor.
#:
#: 2 — `alelyon/languages/` and `alelyon/studio/` added. Both are real, tracked
#:     source trees that version 1 could not place: 25 files read `UNMAPPED`,
#:     including the Rust crates that build the CNE verifier kernel. A session
#:     working there was invisible to the fleet. This is a re-point, not merely
#:     an extension — records stored as `UNMAPPED` under version 1 now resolve
#:     to a real area — which is exactly the case the rule above is for.
#:
#: 3 — flat pillars split per file. Found by using the tool: one session had
#:     `tools/route_split.py` ahead of the mainline, and because every script in
#:     `tools/` collapsed to the single area `tools`, `open-areas` reported the
#:     whole directory occupied and hid roughly fifteen unrelated programs. The
#:     mesh's documented false positive is *path* overlap inside one file; this
#:     was a coarser one invented here, and it steers work away from ground that
#:     is genuinely free.
AREA_SPACE_VERSION = 3

#: Surface depth for a pillar that is a bag of files rather than a tree of
#: subsystems. Named rather than written as a bare 0 at eleven call sites.
FLAT = 0

#: A path no rule placed. Never an empty string and never a nearest guess.
UNMAPPED = "UNMAPPED"

#: (path prefix, pillar, surface depth). Ordered; first match wins, so a longer
#: prefix must precede the shorter one it extends.
#:
#: `depth` is how many path segments AFTER the prefix name the surface:
#:   1 — the segment under the prefix is already a subsystem
#:       (`alelyon/runtime/oracle/assistant/…` → `assistant`)
#:   2 — the first segment is a container and the second is the real subject
#:       (`alelyon/frontend/desktop/lattice/…` → `desktop.lattice`)
#:   0 — **flat**: the pillar is a bag of independent files, not a tree of
#:       subsystems, so the FILE is the unit and its stem is the surface
#:       (`tools/fleet.py` → `tools/fleet`).
#:
#: Flat exists because `tools/` and the root of `docs/` hold programs and
#: documents that share a directory and nothing else. Under depth 1 they all
#: collapsed to one surface, so a single session editing a single script made
#: the entire pillar read occupied and `open-areas` hid the rest. Two sessions
#: in two unrelated scripts do not contend, and a coordinate space that says
#: they do sends work somewhere else for no reason.
#:
#: The pillar names are copied from AGENTS.md §2 verbatim. If that list changes,
#: this table is wrong and `test_pillars_match_the_import_contracts` fails.
_PILLARS: tuple[tuple[str, str, int], ...] = (
    ("alelyon/runtime/common/", "runtime.common", 1),
    ("alelyon/runtime/atlas/", "runtime.atlas", 1),
    ("alelyon/runtime/vector/", "runtime.vector", 1),
    ("alelyon/runtime/nexus/", "runtime.nexus", 1),
    ("alelyon/runtime/sentinel/", "runtime.sentinel", 1),
    ("alelyon/runtime/oracle/", "runtime.oracle", 1),
    # `products/enterprise/` is the live engine composition and `products/api/`
    # is the read-only service; one segment separates them, which is the
    # distinction that matters for who is working where.
    ("alelyon/products/", "products", 1),
    # `frontend/desktop/` is a toolkit, not a subject. The product under it is.
    ("alelyon/frontend/", "frontend", 2),
    ("alelyon/platform/", "platform", 1),
    ("alelyon/verify/", "verify", 1),
    # The polyglot trees. Depth 1 because the segment under `languages/` names
    # the implementation — `cne_verify`, `vector_core`, `axiom` — and those are
    # not interchangeable: one of them is the verifier kernel and Section 8 of
    # AGENTS.md governs it.
    ("alelyon/languages/", "languages", 1),
    ("alelyon/studio/", "studio", 1),
    ("research/", "research", 1),
    # Flat: a bag of independent programs. `tools/route_split.py` and
    # `tools/fleet.py` share a directory and nothing else.
    ("tools/", "tools", FLAT),
    ("scripts/", "tools", FLAT),
    ("packaging/", "packaging", 1),
    ("docs/", "docs", 1),
    ("tests/", "tests", 1),
    ("web/", "web", 1),
)

#: Every pillar a path can actually resolve to. Derived from the table rather
#: than written twice, so a new pillar cannot be accepted by one and rejected by
#: the other. `Area.known` is the only thing that should consult this.
KNOWN_PILLARS: frozenset[str] = frozenset(p for _prefix, p, _depth in _PILLARS)

#: Whole pillars that are Tier 3 under AGENTS.md §3, withheld from any surface
#: that offers work to an agent. §1 forbids a probationary model from changing
#: them at all, so offering one as a free slot points a session that asked
#: "where am I needed?" straight at authority it does not have.
#:
#: This started as `CAPITAL_BEARING` and held only the two order-path pillars.
#: That name was narrower than the job: §3's Tier 3 is "capital, destructive,
#: **trust**, or release authority" and names signing keys, CNE verification
#: semantics and public package contents in the same breath. Under the old set
#: `open-areas` offered `verify` and `packaging` as free work — the verifier and
#: the release path — which is the same mistake as offering an order path, with
#: a claim boundary instead of a position at the end of it.
TIER3_PILLARS: frozenset[str] = frozenset({
    "products",          # live enterprise composition; order paths
    "runtime.sentinel",  # PnL, intent/fill ledgers, execution records
    "verify",            # the open verifier: §8 claim discipline
    "packaging",         # public package contents and release tooling
})

#: Individual areas that are Tier 3 while their pillar is not. Needed because
#: Tier 3 is not always pillar-shaped: `languages/axiom` is an ordinary DSL and
#: `languages/cne_verify` is the verifier kernel, and withholding the whole
#: pillar to catch one of them would hide real work.
TIER3_AREAS: frozenset[tuple[str, str]] = frozenset({
    ("languages", "cne_verify"),
    ("languages", "vector_core"),    # the native kernel behind certified widths
    ("languages", "vector_native"),
})


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

    @property
    def known(self) -> bool:
        """Whether a path could ever actually resolve to this area.

        `mapped` only says "not the UNMAPPED sentinel", which any string
        satisfies — `parse_area` partitions on `/` and hands back whatever it was
        given. So a typo produced a well-formed Area on a pillar that appears in
        no path, and anything keyed on it was invisible for ever after.

        That is not hypothetical. A session claimed `platform.gateway` — the
        dotted form, because pillars like `runtime.common` really do contain
        dots — while routing derives `platform/gateway`. The claim was accepted
        and reported as success, the session was never reachable, and four
        findings published at the real area reported REACHED NOBODY. Creating a
        coordinate must therefore be checked against the table; reading one back
        must not, so a bad record can still be inspected and released.
        """
        return self.pillar in KNOWN_PILLARS

    @property
    def tier3(self) -> bool:
        """Work here needs explicit owner authority. See AGENTS.md §3.

        Covers capital, destructive, **trust** and release authority — not only
        the order paths. A surface offering free work must withhold these.
        """
        return (self.pillar in TIER3_PILLARS
                or (self.pillar, self.surface) in TIER3_AREAS)


#: The single instance every unplaced path resolves to, so callers can compare
#: identity rather than remembering to test two fields.
UNMAPPED_AREA = Area(UNMAPPED, "")


def parse_area(text: str) -> Area:
    """Read back what `str(Area)` wrote. `UNMAPPED` round-trips."""
    raw = (text or "").strip()
    if not raw or raw == UNMAPPED:
        return UNMAPPED_AREA
    pillar, _, surface = raw.partition("/")
    return Area(pillar, surface)


def suggest_area(text: str) -> "Area | None":
    """A known area the caller plausibly meant, or None.

    Exists for one confusion specifically, because it is the one that actually
    happened and it is built into the vocabulary rather than being carelessness:
    a pillar may contain a dot (`runtime.common`), and a pillar and its surface
    are joined by a slash (`platform/gateway`). Both separators are legitimate,
    so `platform.gateway` looks entirely reasonable and resolves to nothing.

    Refusing without a suggestion would leave the caller re-reading a vocabulary
    list to spot a single character.
    """
    raw = (text or "").strip()
    if not raw or parse_area(raw).known:
        return None

    candidates: list[str] = []
    if "/" not in raw and "." in raw:
        head, _, tail = raw.rpartition(".")        # platform.gateway
        candidates.append(f"{head}/{tail}")        # -> platform/gateway
    if "/" in raw:
        candidates.append(raw.replace("/", ".", 1))  # runtime/common -> runtime.common
        candidates.append(raw.partition("/")[0])     # bare pillar

    for candidate in candidates:
        area = parse_area(candidate)
        if area.known:
            return area
    return None


def area_of(path: str) -> Area:
    """The area one repository-relative path belongs to.

    Accepts either slash convention because git reports forward slashes on every
    platform while `str(Path(...))` does not, and a caller mixing the two is the
    likeliest way this gets a false `UNMAPPED`.
    """
    posix = str(path or "").replace("\\", "/").lstrip("./")
    if not posix:
        return UNMAPPED_AREA
    for prefix, pillar, depth in _PILLARS:
        if not posix.startswith(prefix):
            continue
        rest = PurePosixPath(posix[len(prefix):])
        if depth == FLAT:
            # The file is the unit. A path with directories under a flat pillar
            # takes the first directory instead, so `tools/pkg/mod.py` groups as
            # `tools/pkg` rather than splitting a package across areas.
            parts = list(rest.parts)
            if not parts:
                return Area(pillar, "")
            if len(parts) > 1:
                return Area(pillar, parts[0])
            return Area(pillar, PurePosixPath(parts[0]).stem)
        # `rest.parts[:-1]` would drop the last directory for a deep path but the
        # FILE for a shallow one. Segments that name a directory are all parts
        # except the final one only when the path has a suffix; a caller may pass
        # either, so take parts and trim the filename when there is one.
        segments = list(rest.parts)
        if segments and "." in segments[-1]:
            segments = segments[:-1]
        # A dot-directory is configuration, not a subsystem. Left in, it produced
        # `frontend/web..streamlit` from `alelyon/frontend/web/.streamlit/` — a
        # phantom area with a doubled separator that no session will ever work
        # in, offered in the free-work list beside real ones. Dropping it puts
        # those files in the surface that owns them.
        segments = [s for s in segments if not s.startswith(".")]
        surface = ".".join(segments[:depth])
        return Area(pillar, surface)
    return UNMAPPED_AREA


def areas_of(paths) -> tuple[Area, ...]:
    """Distinct areas covered by a set of paths, in sorted order.

    `UNMAPPED` is included when any path fell outside the table. Dropping it here
    is how a fleet report ends up claiming full coverage of work it never placed.
    """
    return tuple(sorted({area_of(p) for p in (paths or ())}))


def all_pillars() -> tuple[str, ...]:
    """Every declared pillar, once, in table order."""
    seen: list[str] = []
    for _prefix, pillar, _depth in _PILLARS:
        if pillar not in seen:
            seen.append(pillar)
    return tuple(seen)


def surfaces_in(pillar: str, paths) -> tuple[Area, ...]:
    """The areas of `pillar` that `paths` actually reach."""
    return tuple(sorted({a for a in areas_of(paths) if a.pillar == pillar}))
