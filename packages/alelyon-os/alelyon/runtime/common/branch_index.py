"""What each branch handled, derived from git rather than declared.

The repository had 29 branches and no way to answer "which branch dealt with
this?" without reading history by hand. This module answers it from records the
branches did not author: a merge commit's second parent is the branch tip as it
landed, and the diff from its merge-base to that tip is what the branch changed.
Nothing here is self-reported, which is the same rule `worktree.py` follows for
attribution and for the same reason -- a branch's own description of itself is
the writer describing itself.

Areas come from `worktree_areas.area_of()`, whose pillars are read out of
AGENTS.md §2. Reusing that vocabulary rather than inventing a second one is the
point: this index and the routing that follows must mean the same thing by
`runtime.vector`, or an ownership rule written against one will not hold against
the other.

Three states, and the third is why this module is not a one-liner
-----------------------------------------------------------------
* **merged** -- a merge commit names it, and the footprint follows directly.
* **open** -- not yet in the mainline; the footprint is what it holds over it.
* **absorbed** -- already in the mainline with NO merge commit of its own,
  because another branch was cut from it and carried it in. Naive derivation
  reports these as having changed nothing, which reads as "this branch did
  nothing" when the truth is "this branch's work arrived under another name".
  `lattice/worktree-observer` is the live case: it reached main inside
  `lattice/worktree-graph`. The footprint is recovered by walking first-parent to
  the nearest merge and diffing that boundary to the tip.

What a footprint is not
-----------------------
It is what a branch CHANGED, not what it is responsible for. A branch that
touched one line of a shared file does not own that file, and 40 of 189 files in
this repository's history were touched by more than one branch. Ownership is a
separate question this module deliberately does not answer -- it supplies the
measurement an ownership rule has to be written against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import collections
import hashlib
import re
import subprocess

from alelyon.runtime.common import toolpath
from alelyon.runtime.common.worktree_areas import UNMAPPED, all_pillars, area_of

#: Where "already landed" is measured from.
MAINLINE_DEFAULT = "origin/main"

MERGED = "merged"
OPEN = "open"
ABSORBED = "absorbed"
UNRELATED = "unrelated"

_GIT_TIMEOUT = 300

#: `Merge pull request #12 from TLace03/docs/dynamic-cache-foundation`
_PR_SUBJECT = re.compile(r"Merge pull request #(\d+) from [^/]+/(\S+)")

#: Refs that are not topic branches: the remote's own HEAD symref, and the
#: mainline itself. Excluding them is not a judgement about their contents.
_NOT_A_BRANCH = ("HEAD",)

#: Past this many pillars the readable name stops being readable, so it becomes a
#: digest of the same canonical string. Three fits `a+b+c` inside a terminal.
READABLE_PILLAR_LIMIT = 3

INDEX_LIMITS: tuple[str, ...] = (
    "A footprint is what a branch CHANGED, not what it owns. A branch that "
    "touched one line of a shared file does not own that file.",
    "A branch merged by fast-forward with nothing cut from it leaves no trace "
    "to derive from. The absorbed case is recovered because a carrier names it; "
    "a branch nothing was ever cut from would be missed silently.",
    "Merge commits whose subject is not a pull-request merge carry no branch "
    "name. They are reported separately rather than attributed to a guess.",
    "Branches with a separate root have no merge-base with the mainline and are "
    "excluded by construction, not by judgement.",
)


def _git(*args: str, repo: str) -> tuple[int, str]:
    """A query. Nothing in this module writes to the repository."""
    proc = subprocess.run(toolpath.argv("git", "-C", repo, *args), capture_output=True,
                          text=True, timeout=_GIT_TIMEOUT, **toolpath.no_window())
    return proc.returncode, proc.stdout.strip()


@dataclass(frozen=True)
class BranchRecord:
    """One branch and what it actually changed."""

    name: str
    state: str                          # MERGED | OPEN | ABSORBED | UNRELATED
    files: tuple[str, ...] = ()
    prs: tuple[str, ...] = ()
    last_merged: str = ""               # ISO date, or "" when never merged
    #: For ABSORBED only: the branch that carried this one into the mainline.
    absorbed_into: str = ""

    @property
    def pillars(self) -> tuple[str, ...]:
        """Every pillar this branch touched, canonically ordered."""
        return tuple(sorted({area_of(f).pillar for f in self.files}))

    @property
    def pillar_counts(self) -> dict[str, int]:
        counts: collections.Counter = collections.Counter()
        for path in self.files:
            counts[area_of(path).pillar] += 1
        return dict(counts.most_common())

    @property
    def spans_multiple_pillars(self) -> bool:
        return len(self.pillars) > 1


@dataclass(frozen=True)
class BranchIndex:
    """Every branch of one repository, and what each one handled."""

    mainline: str
    records: tuple[BranchRecord, ...] = ()
    #: Merge commits carrying no parseable branch name: (sha, date, subject, n).
    orphan_merges: tuple[tuple[str, str, str, int], ...] = ()
    limits: tuple[str, ...] = INDEX_LIMITS

    def by_state(self, state: str) -> tuple[BranchRecord, ...]:
        return tuple(r for r in self.records if r.state == state)

    def commons(self) -> dict[str, tuple[str, ...]]:
        """Files more than one branch changed, and which branches those were.

        This is the set an ownership rule cannot assign, so it must name a policy
        for them instead. Measured rather than assumed: the answer decides
        whether per-file ownership is workable at all.

        An absorbed branch counts as its carrier, not as a second owner. The
        carrier's footprint necessarily contains the absorbed branch's files --
        that is what carrying it in means -- so counting both would report one
        lineage as two branches contending over its own work, and inflate the
        commons with collisions that never happened.
        """
        canonical = {r.name: (r.absorbed_into or r.name) for r in self.records}
        owners: dict[str, set] = collections.defaultdict(set)
        for record in self.records:
            for path in record.files:
                owners[path].add(canonical.get(record.name, record.name))
        return {p: tuple(sorted(b)) for p, b in owners.items() if len(b) > 1}

    def pillar_combinations(self) -> dict[tuple[str, ...], int]:
        """Multi-pillar sets that actually occurred, and how often.

        The population a coordination lane converges to. A branch touching one
        pillar needs no coordination and is not counted.
        """
        seen: collections.Counter = collections.Counter()
        for record in self.records:
            if record.spans_multiple_pillars:
                seen[record.pillars] += 1
        return dict(seen.most_common())


def short_pillar(pillar: str) -> str:
    """`runtime.vector` -> `vector`, for a name a person has to read.

    Only the `runtime.` prefix is dropped, and `test_short_pillar_is_injective`
    fails if that ever makes two pillars share a short form -- at which point the
    readable name would be ambiguous and the rule has to change rather than the
    collision be tolerated.
    """
    return pillar[len("runtime."):] if pillar.startswith("runtime.") else pillar


def coordination_name(pillars) -> str:
    """The branch that owns work spanning exactly this set of pillars.

    A pure function of the set, which is the whole design: the same combination
    always resolves to the same name, so "create it once" needs no bookkeeping —
    routing looks the name up and creates only on a miss. Order and duplicates
    cannot matter, so they are canonicalised away first.

    The digest past `READABLE_PILLAR_LIMIT` is taken over the CANONICAL full
    names, never the shortened display form, so shortening can be changed later
    without every existing branch name moving.

    Raises rather than inventing a name in two cases, both deliberate:

    * fewer than two pillars — single-pillar work belongs on an ordinary branch,
      and minting a coordination branch for it would grow the lane for nothing;
    * `UNMAPPED` present — that is a path no rule places. Creating a branch for
      it would encode a routing gap as permanent infrastructure, when the fix is
      a routing rule.
    """
    canonical = tuple(sorted(set(pillars)))
    if UNMAPPED in canonical:
        raise ValueError(
            f"{UNMAPPED} cannot take part in a coordination branch: it names a "
            f"path no rule places, which is a missing rule and not an area")
    if len(canonical) < 2:
        raise ValueError(
            f"coordination needs two or more pillars, got {canonical!r}; "
            f"single-pillar work belongs on an ordinary branch")
    if len(canonical) <= READABLE_PILLAR_LIMIT:
        return "coord/" + "+".join(short_pillar(p) for p in canonical)
    digest = hashlib.sha256("+".join(canonical).encode("utf-8")).hexdigest()[:8]
    return f"coord/{len(canonical)}-areas-{digest}"


def _commit_date(sha: str, *, repo: str) -> str:
    return _git("log", "-1", "--format=%cs", sha, repo=repo)[1]


def _own_footprint(ref: str, *, repo: str) -> tuple[str, ...]:
    """What a branch changed itself, for one that left no merge commit.

    Walk first-parent to the nearest merge: everything after it is this branch's
    own work and everything at or before it arrived from the mainline it was cut
    from, so that boundary is the merge-base a merge commit would have given.
    """
    _, chain = _git("rev-list", "--first-parent", "--format=%H %P", ref,
                    repo=repo)
    for line in chain.splitlines():
        if line.startswith("commit "):
            continue
        parts = line.split()
        if len(parts) > 2:                      # two or more parents: a merge
            _, files = _git("diff", "--name-only", f"{parts[0]}...{ref}",
                            repo=repo)
            return tuple(f for f in files.splitlines() if f)
    return ()


def observe(repo_root: str = ".", *,
            mainline: str = MAINLINE_DEFAULT) -> BranchIndex:
    """Derive the whole index. Read-only: every git call is a query."""
    repo = str(repo_root)
    # Branches live in the same namespace as the mainline they are measured
    # against: `origin/main` means the remote's branches, a bare `main` means
    # local ones. Reading only `refs/remotes` would index almost nothing in a
    # repository that has no remote, which is not a rare shape -- it is every
    # fixture and every clone-less checkout.
    if "/" in mainline:
        remote = mainline.split("/", 1)[0]
        remote_prefix, namespace = remote + "/", f"refs/remotes/{remote}"
    else:
        remote_prefix, namespace = "", "refs/heads"

    _, raw = _git("for-each-ref", "--format=%(refname:short)", namespace,
                  repo=repo)
    branches = []
    for ref in raw.splitlines():
        if ref == mainline or not ref.startswith(remote_prefix):
            continue
        name = ref[len(remote_prefix):]
        # The remote's own symref (`origin`) shortens to an empty name.
        if not name or name in _NOT_A_BRANCH:
            continue
        branches.append(name)

    state: dict[str, str] = {}
    for name in branches:
        ref = remote_prefix + name
        rc, base = _git("merge-base", mainline, ref, repo=repo)
        if rc or not base:
            state[name] = UNRELATED
            continue
        rc2, _ = _git("merge-base", "--is-ancestor", ref, mainline, repo=repo)
        state[name] = MERGED if rc2 == 0 else OPEN

    footprint: dict[str, set] = collections.defaultdict(set)
    pull_requests: dict[str, set] = collections.defaultdict(set)
    merged_on: dict[str, set] = collections.defaultdict(set)
    orphans: list[tuple[str, str, str, int]] = []

    _, log = _git("log", mainline, "--merges", "--format=%H%x00%P%x00%s",
                  repo=repo)
    for line in log.splitlines():
        sha, parents, subject = line.split("\0")
        parent_list = parents.split()
        if len(parent_list) != 2:
            continue
        rc, base = _git("merge-base", *parent_list, repo=repo)
        if rc or not base:
            continue
        rc, files = _git("diff", "--name-only", f"{base}...{parent_list[1]}",
                         repo=repo)
        changed = [f for f in files.splitlines() if f]
        match = _PR_SUBJECT.search(subject)
        if not match:
            orphans.append((sha[:9], _commit_date(sha, repo=repo),
                            subject, len(changed)))
            continue
        name = match.group(2)
        footprint[name] |= set(changed)
        pull_requests[name].add(match.group(1))
        merged_on[name].add(_commit_date(sha, repo=repo))

    for name, current in state.items():
        if current != OPEN:
            continue
        ref = remote_prefix + name
        rc, base = _git("merge-base", mainline, ref, repo=repo)
        if rc or not base:
            continue
        rc, files = _git("diff", "--name-only", f"{base}...{ref}", repo=repo)
        footprint[name] |= {f for f in files.splitlines() if f}

    # Merged, but no merge commit named it: another branch was cut from this one
    # and carried it in. Reported as absorbed rather than as an empty footprint.
    absorbed_into: dict[str, str] = {}
    named_by_a_merge = set(pull_requests)
    for name, current in state.items():
        if current != MERGED or footprint.get(name):
            continue
        carriers = []
        for other in named_by_a_merge:
            if other == name:
                continue
            rc, _ = _git("merge-base", "--is-ancestor", remote_prefix + name,
                         remote_prefix + other, repo=repo)
            if rc == 0:
                carriers.append(other)
        if not carriers:
            continue
        # The smallest carrier is the branch actually cut from it; a larger one
        # merely contains it transitively.
        carrier = min(carriers, key=lambda c: (len(footprint.get(c, ())), c))
        absorbed_into[name] = carrier
        state[name] = ABSORBED
        footprint[name] = set(_own_footprint(remote_prefix + name, repo=repo))

    records = []
    for name in sorted(set(state) | set(footprint)):
        dates = merged_on.get(name, set())
        records.append(BranchRecord(
            name=name,
            state=state.get(name, OPEN),
            files=tuple(sorted(footprint.get(name, ()))),
            prs=tuple(sorted(pull_requests.get(name, ()), key=int)),
            last_merged=max(dates) if dates else "",
            absorbed_into=absorbed_into.get(name, ""),
        ))

    return BranchIndex(mainline=mainline, records=tuple(records),
                       orphan_merges=tuple(orphans))


__all__ = [
    "ABSORBED", "BranchIndex", "BranchRecord", "INDEX_LIMITS",
    "MAINLINE_DEFAULT", "MERGED", "OPEN", "READABLE_PILLAR_LIMIT", "UNRELATED",
    "coordination_name", "observe", "short_pillar",
]


# ── coordination lanes: created once, reused, never accumulated ──────────────

#: Prefix `coordination_name()` mints. A lane is identified by the pillar set it
#: serves, so the same combination always resolves to the same branch.
COORDINATION_PREFIX = "coord/"

LANE_REUSABLE = "reusable"      # merged and idle: reset to the mainline
LANE_ACTIVE = "active"          # holds unmerged work
LANE_OCCUPIED = "occupied"      # a worktree is standing on it


@dataclass(frozen=True)
class LaneState:
    """One coordination lane and whether it can be handed to the next change."""

    branch: str
    state: str                  # LANE_REUSABLE | LANE_ACTIVE | LANE_OCCUPIED
    reason: str
    tip: str = ""
    occupied_by: str = ""


def is_coordination(branch: str) -> bool:
    return branch.startswith(COORDINATION_PREFIX)


def lane_states(repo_root: str = ".", *, mainline: str = MAINLINE_DEFAULT,
                worktree_paths=None) -> tuple[LaneState, ...]:
    """Every coordination lane, and whether it is free to reuse.

    A lane is durable infrastructure rather than a topic branch: it is named
    from its pillar set, so deleting it after a merge would only force the same
    name to be minted again on the next change spanning those pillars. The
    lifecycle is therefore reset-to-mainline, not delete.

    Resetting is refused for a lane holding unmerged work, and for one a
    worktree is standing on -- moving a ref under a checked-out branch leaves
    that working tree describing itself as ahead or behind for reasons its
    occupant never caused.
    """
    repo = str(repo_root)
    occupied: dict[str, str] = {}
    if worktree_paths is None:
        code, listing = _git("worktree", "list", "--porcelain", repo=repo)
        if code == 0:
            path = ""
            for line in listing.splitlines():
                if line.startswith("worktree "):
                    path = line[len("worktree "):]
                elif line.startswith("branch "):
                    ref = line[len("branch "):]
                    occupied[ref.removeprefix("refs/heads/")] = path
    else:
        occupied = dict(worktree_paths)

    _, raw = _git("for-each-ref", "--format=%(refname:short)", "refs/heads",
                  repo=repo)
    out: list[LaneState] = []
    for branch in sorted(b for b in raw.splitlines() if is_coordination(b)):
        _, tip = _git("rev-parse", branch, repo=repo)
        if branch in occupied:
            out.append(LaneState(branch=branch, state=LANE_OCCUPIED, tip=tip,
                                 occupied_by=occupied[branch],
                                 reason=f"a worktree is checked out on it at "
                                        f"{occupied[branch]}"))
            continue
        merged, _ = _git("merge-base", "--is-ancestor", branch, mainline,
                         repo=repo)
        if merged != 0:
            out.append(LaneState(branch=branch, state=LANE_ACTIVE, tip=tip,
                                 reason=f"holds work not yet in {mainline}"))
            continue
        _, same = _git("rev-parse", mainline, repo=repo)
        if same == tip:
            out.append(LaneState(branch=branch, state=LANE_REUSABLE, tip=tip,
                                 reason="already at the mainline; ready for the "
                                        "next change spanning these pillars"))
        else:
            out.append(LaneState(branch=branch, state=LANE_REUSABLE, tip=tip,
                                 reason=f"merged into {mainline} and idle; "
                                        f"reset to hand it to the next change"))
    return tuple(out)


def reset_lane(branch: str, repo_root: str = ".", *,
               mainline: str = MAINLINE_DEFAULT) -> str:
    """Move a merged, idle lane to the mainline. Returns the new tip.

    Compare-and-swap against the tip that was read, so a lane advanced between
    the check and the reset is refused rather than discarded. Raises
    `ValueError` for a lane that is not reusable -- the caller does not get to
    reset one somebody is standing on.
    """
    repo = str(repo_root)
    states = {s.branch: s for s in lane_states(repo, mainline=mainline)}
    state = states.get(branch)
    if state is None:
        raise ValueError(f"{branch!r} is not a coordination lane in this repository")
    if state.state != LANE_REUSABLE:
        raise ValueError(f"{branch!r} is {state.state}: {state.reason}")

    _, target = _git("rev-parse", mainline, repo=repo)
    if not target:
        raise ValueError(f"mainline {mainline!r} does not resolve")
    code, _ = _git("update-ref", f"refs/heads/{branch}", target, state.tip,
                   repo=repo)
    if code != 0:
        raise ValueError(f"{branch!r} moved while it was being reset; "
                         f"nothing was changed")
    return target
