"""Observe the git worktrees agent sessions leave behind.

This is the **observation-only** half of the Dynamic Worktree Cache described in
`docs/features/DYNAMIC-CACHE.md`. It is the half that works without cooperation:
nothing here asks an agent to register, so nothing here breaks when an agent does
not. What it can see, it sees for every worktree on disk regardless of which tool
made it.

Three questions the raw `git worktree list` output cannot answer, and this can:

* **Is this worktree still alive, or did its session end weeks ago?**
* **Who is touching which files right now?**
* **Which two worktrees are about to collide?**

The third is the one that earns the module. Two agents each behaving correctly in
isolation produce work that cannot be reconciled when neither could see the other,
and an intersection of touched paths is the earliest observable sign of it.

Attribution is derived, never believed
--------------------------------------
Git's `author` and `committer` are freely settable, a worktree's directory name is
chosen by whatever created it, and any session id written into a file is written by
the thing claiming that identity. All three are the writer describing itself, so
none is used here.

What *is* used is the record git keeps for its own purposes — the worktree's
administrative path, its `HEAD`, and reachability — plus the parent-directory
convention, which identifies a **tool family** and nothing finer. Every derivation
carries the rule that produced it in `tool_evidence`, so a reader can disagree with
it. Where no rule fires the answer is `UNATTRIBUTED`, which is a value, not a blank.

**Model and session identity are not derivable from git and are not guessed here.**
Tool family is the honest ceiling for a worktree found on disk. `MESH_LIMITS`
states that where a caller will read it.

Read-only and side-effect free: every git invocation is a query, nothing is
written, pruned, checked out or fetched, and a worktree whose directory has been
deleted is reported rather than repaired.
"""
from __future__ import annotations

from concurrent import futures
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import threading
import time

from alelyon.runtime.common import toolpath
from alelyon.runtime.common import throughput

#: Returned wherever a fact could not be derived. Never an empty string: a blank
#: beside a filled field reads as "checked, nothing there".
UNATTRIBUTED = "UNATTRIBUTED"

#: Parent-directory conventions that identify the tool family that created a
#: worktree. Ordered, first match wins, and each entry is (marker, family).
#: A convention identifies a TOOL, never a model and never a session.
_TOOL_CONVENTIONS: tuple[tuple[str, str], ...] = (
    (".claude/worktrees/", "claude-code"),
    (".codex/worktrees/", "codex"),
    (".cursor/worktrees/", "cursor"),
    (".copilot/worktrees/", "copilot"),
    (".antigravity/worktrees/", "antigravity"),
    (".git/worktrees/", "git-native"),
)

#: Conventions whose path additionally carries a SESSION identifier.
#:
#: Added because the mesh met a real worktree it could not place: a second agent
#: session working in this repository from
#: `…/Temp/claude/<project-slug>/<session-uuid>/scratchpad/wt-lattice`, which
#: matched no rule above and read UNATTRIBUTED. The shape was then confirmed
#: against a second, independent session's path before being encoded here — one
#: sighting is an anecdote.
#:
#: What this buys is more than a tool name. The session id is IN the path, so for
#: this convention a session is identifiable from a location git records, without
#: the agent volunteering anything. That is a genuine advance on
#: `worktree_cache.declare()`, which is self-reported.
#:
#: It is still DERIVED, not observed, and the difference matters: a worktree can
#: be created at any path, so anything able to choose a directory name can wear a
#: session-shaped one. Treat it as a strong hint with a stated rule, never as
#: authentication — `session_evidence` carries that caveat to every reader.
_SESSION_CONVENTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(
        r"/claude/[^/]+/"
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/",
        re.IGNORECASE), "claude-code"),
)

#: What this module cannot establish, stated once and carried on every mesh so a
#: consumer cannot render the output without the caveat being available to it.
MESH_LIMITS: tuple[str, ...] = (
    "Model identity is not derivable from git. A worktree records no author "
    "beyond freely-settable commit metadata, so no model is attributed here.",
    "Session identity is derivable ONLY where a tool puts it in the path, and "
    "then only as a derivation: a worktree can be created anywhere, so anything "
    "able to choose a directory name can wear a session-shaped one. Everywhere "
    "else, tool family remains the ceiling.",
    # ASCII only, like every other entry: these are printed by
    # tools/worktree_report.py, and a Windows console codepage cannot encode an
    # em dash. A limit that crashes the reader is not a limit that was stated.
    "A session id in the path names the session that CREATED the directory, "
    "which is not necessarily the one working in it now. A second session can "
    "enter an existing worktree and leave changes there, and every one of those "
    "changes is attributed to the creator. That is the one case where this mesh "
    "reports a confident answer that is wrong rather than an honest "
    "UNATTRIBUTED. Only a declaration can name the current occupant; see "
    "worktree_cache.occupancy_conflicts().",
    "Contention is computed from paths touched, not from semantics. Two "
    "worktrees editing different functions in one file are reported as "
    "contending, which is a deliberate false positive rather than a missed one.",
    "Contention is computed BETWEEN worktrees. Two sessions working inside the "
    "SAME worktree share one record and one set of touched paths, so they never "
    "contend with each other here no matter how directly they collide. The "
    "highest-risk arrangement is the one this pairing cannot see.",
    "A clean worktree that a session is still actively reading is "
    "indistinguishable from an abandoned one. Staleness measures its commit, "
    "not its session.",
)

_GIT_TIMEOUT = 30
_MAX_PATHS_PER_WORKTREE = 5_000

#: Aggregate wall-clock ceiling for one `observe()`, in seconds.
#:
#: `_GIT_TIMEOUT` bounds a single query and nothing bounded the sum of them. At
#: this repository's 287 worktrees that is 287 x 4 queries x 30 s of unbounded
#: exposure -- hours -- on a reading whose caller is a GUI worker thread. The
#: per-call timeout is not a budget; it is the point at which one hung query is
#: abandoned, and a reading can be entirely composed of slow-but-not-hung
#: queries.
#:
#: A reading that runs out of budget returns the SAME prefix a `should_stop`
#: caller gets: `stopped` set, a note saying how far it reached, and contention
#: that a consumer must not draw. That is deliberate -- there is exactly one
#: partial-observation contract and this reuses it rather than inventing a
#: second one that views would have to learn.
#:
#: The bound is the budget PLUS the tail of the worktree in flight when it
#: expires, because `subprocess.run` is not interruptible once entered. That is
#: at most four queries, so the honest ceiling is OBSERVE_BUDGET_SECONDS + 4 x
#: `_GIT_TIMEOUT`, not OBSERVE_BUDGET_SECONDS.
OBSERVE_BUDGET_SECONDS = 120.0

#: Wall seconds for one WARM `observe()` at N pool workers, measured through
#: this module's own code path on 2026-08-11 on this workstation, over this
#: repository's 163 worktrees. `_OBSERVE_POOL_SWEEP_EVIDENCE` records what the
#: rows are and what they are not.
#:
#: A warm reading is the right subject: it is `git status --untracked-files=all`
#: once per worktree and almost nothing else, because everything that is a
#: function of a commit hash is already memoised. It is also the reading a GUI
#: actually repeats -- a cold one happens once per process.
#:
#: Each figure is the MINIMUM of five round-robin rounds. Nineteen sessions
#: share this box, so a single timing measures whoever else was running; the
#: minimum is the least-contended observation and therefore a lower bound on
#: cost, and the ratio of two lower bounds is the cleanest speedup this machine
#: can report. Round-robin rather than five rounds of one width, so load drift
#: is spread across every width instead of landing on one of them.
_OBSERVE_POOL_SWEEP: tuple[tuple[int, float], ...] = (
    (1, 7.610), (2, 4.500), (4, 2.344), (8, 1.375),
    (16, 0.985), (24, 1.125), (32, 1.079),
)

_OBSERVE_POOL_SWEEP_EVIDENCE = (
    "THE ROWS ARE ONE WORKLOAD ON ONE MACHINE. Each is a warm `observe()` of "
    "this repository, whose per-worktree cost is dominated by a full untracked "
    "walk of a working tree that shares one physical disk with 162 others. A "
    "repository with fewer or larger worktrees, or one on a different storage "
    "device, is UNMEASURED here and `ALELYON_MESH_WORKERS` is the override.",
    "THE ROWS ARE NOT IN `throughput_knee`'s UNITS AND MUST BE CONVERTED. A "
    "row accepted by that helper is N INDEPENDENT copies and the wall "
    "clock for all N, so its throughput is `N/wall`. A row here is ONE FIXED "
    "workload -- 163 worktrees -- split N ways, so its throughput is `1/wall` "
    "and N/wall is a unit error. Handing these rows to `_throughput_knee` "
    "unconverted returns 32, the last row of the table, at a width whose wall "
    "clock is measurably WORSE than 16's; see `observe_pool_width` for the "
    "arithmetic showing that on raw rows the test can never fail at any width. "
    "That is a guard calibrated on the wrong law, and the conversion in "
    "`observe_pool_width` is what stops it.",
    "THE KNEE IS NOT COMPUTED HERE. `observe_pool_width()` hands the converted "
    "rows to `throughput.throughput_knee`, which is where this repository's rule "
    "for 'the last worker that paid for itself' lives. The private `local_ci` "
    "adapter and `ci_sweep` pass their rows to that same function. A second "
    "copy of the arithmetic could disagree with the first for reasons that had "
    "nothing to do with either measurement.",
    "WHAT THE POOL DOES NOT CHANGE. It runs the same git commands, per "
    "worktree, that the serial loop ran; it spawns no extra process and skips "
    "none. Only the wall clock moves. `test_mesh_read_cost.py` asserts that "
    "field-for-field against a serial reading of the same tree.",
)

#: Environment override for the pool width. `1` restores the serial reading
#: exactly, which is what the falsifier uses to show the speedup is real.
POOL_WIDTH_ENV = "ALELYON_MESH_WORKERS"


def observe_pool_width() -> int:
    """How many worktrees `observe()` reads at once.

    **Why a pool is admissible at all.** The per-worktree cost here is a
    `subprocess.run`, which releases the GIL for its whole duration. The threads
    are not computing anything; they are each holding one git process's hand.
    So this is concurrency over a wait, not parallelism over a computation, and
    the answers are unchanged because the commands are unchanged.

    **Why not a cheap invalidation token instead.** Because there is not one.
    The dominant cost is `git status --untracked-files=all`, and every candidate
    key for it is a lie: an index mtime does not move when an untracked file
    appears, and a directory mtime does not move when content changes in a
    subdirectory. A token built from either reports stale uncommitted work,
    which is the single thing this mesh exists to show. That is a design truth,
    not a missing optimisation, and it is why the answer here is to pay the
    walks faster rather than to skip them.

    **The number.** Derived, not chosen: `throughput.throughput_knee` over
    `_OBSERVE_POOL_SWEEP` -- the last worker that still returned at least half
    of what the first one returned. The dependency-free `throughput` module
    owns that rule and its `DEFAULT_MARGINAL_FLOOR`; this module supplies a
    sweep of its own workload and nothing else, because a second knee policy
    would be two constants each derived from the other's consequences.

    **The conversion, and why leaving it out is a wrong answer and not a
    rounding error.** `_throughput_knee` reads a row's second column as the wall
    clock for `workers` INDEPENDENT units, so it computes throughput as
    `workers/wall`. A row here is one FIXED workload split `workers` ways, whose
    throughput is `1/wall`. Multiplying the wall by `workers` restates the row
    in the function's units -- "the wall clock this concurrency would need for
    `workers` whole meshes" -- and `workers/(workers*wall)` is then `1/wall`,
    which is the rate this workload actually has.

    Unconverted, the rule cannot fail at any width, and that is arithmetic
    rather than bad luck. The wall clock has a floor `F` (0.985 s here, the
    point where adding workers stops helping), so the marginal rate per added
    worker approaches `1/F = 1.02`, while the bar it must clear is
    `_MARGINAL_FLOOR / wall(1) = 0.5/7.61 = 0.066`. A test whose threshold is
    fifteen times below the value it is testing is not a knee; it returns the
    last row of whatever table it is given. On this sweep that is 32 workers, at
    a wall clock (1.079 s) measurably WORSE than 16's.

    Converted, the same function and the same `_MARGINAL_FLOOR` return **8**:
    workers 9 through 16 each returned 0.036 mesh/s against a bar of 0.066, so
    they are where the pool starts paying more than it collects. That is the
    number, and it is a real trade rather than a free one -- 16 workers were
    measured at 0.985 s against 8 workers' 1.375 s, so eight more threads bought
    0.39 s. The rule says those eight did not earn it, and on a box that
    nineteen sessions share, threads that each hold a git process cost
    neighbours something this wall clock does not show.

    **Why `cpu_count()` is NOT the ceiling here, unlike `default_jobs()`.**
    There it bounds work that saturates a box; a two-core machine must not run a
    sixteen-core knee. Here each worker owns one short-lived git process that
    spends its life in the filesystem, so cores are not the resource in
    question -- capping at `cpu_count() - 2` would serialise a two-core laptop
    onto a wait it is not paying for. The sweep's own knee is the ceiling, and
    the population is a second one: a repository with four worktrees gets four
    workers, never the knee.

    `ALELYON_MESH_WORKERS` overrides both, and a value below 1 is ignored rather
    than obeyed, because a zero-width pool is not a slower reading, it is no
    reading at all.
    """
    override = os.environ.get(POOL_WIDTH_ENV, "").strip()
    if override:
        try:
            asked = int(override)
        except ValueError:
            asked = 0
        if asked >= 1:
            return asked
    return max(1, throughput.throughput_knee(
        observe_pool_sweep_in_jobs_units()))


def observe_pool_sweep_in_jobs_units(
        sweep=None) -> tuple[tuple[int, float], ...]:
    """`_OBSERVE_POOL_SWEEP` restated in `throughput_knee`'s units.

    A row becomes `(workers, workers * wall)`: the wall clock this concurrency
    would need to finish `workers` whole meshes rather than the one it actually
    finished. See `observe_pool_width` for why the raw rows are a unit error and
    what the unconverted answer is.

    Separate and public so a test can assert the two answers differ, rather than
    the conversion being an unremarked `*` inside a return statement.
    """
    rows = _OBSERVE_POOL_SWEEP if sweep is None else tuple(sweep)
    return tuple((workers, workers * wall) for workers, wall in rows)

#: SHAs per batched `git show`. Windows caps a command line near 32k characters
#: and a full hash costs 41 of them, so this stays an order of magnitude clear
#: of the limit rather than computing how close it can get.
_COMMIT_BATCH = 100

#: Answers memoised across observations, keyed on the git object hashes they are
#: a function of.
#:
#: `observe()` is re-run every `MESH_INTERVAL_MS` for as long as a Fleet view is
#: on screen, and measured over this repository's forty-eight worktrees it is
#: 164 git subprocesses and 3.3 seconds a pass. Two of the three big consumers
#: do not need re-running at all:
#:
#: * a commit's timestamp is a property of the commit. `show -s --format=%ct`
#:   over one SHA returns the same integer forever.
#: * ancestry and the ahead-diff are properties of a **pair** of commits, so
#:   they are fixed once `(head, mainline)` are both named by hash.
#:
#: So the key is the content, not the clock, and the cache cannot go stale in
#: the way a time-based one can: a moved HEAD or a fetched mainline is a
#: different key and misses. That is the whole reason to do it this way rather
#: than with an expiry. `git status` is deliberately NOT cached -- it is the
#: question "what has changed since the last commit", which is exactly the thing
#: with no immutable key, and it is what the view is for.
#:
#: Only successful answers are stored. Freezing a transient git failure for the
#: life of the process would trade a small cost for a wrong picture.
_OBJECT_CACHE: dict = {}

#: Entries kept before the cache is dropped whole. Keys are commit hashes, so
#: the set grows with the repository's history rather than with uptime; this is
#: a backstop against a long-lived process, not a working limit.
_OBJECT_CACHE_LIMIT = 8192

#: Guards `_OBJECT_CACHE`, because `observe()` reads worktrees on a thread pool.
#:
#: It covers the dict operations ONLY, never the `compute()` that fills them. A
#: lock held across the git subprocess would serialise the pool back down to one
#: worker, which is the whole thing the pool exists to stop.
#:
#: The consequence is explicit and accepted: two threads that miss the same key
#: at the same moment both compute it. That costs a duplicate git process and
#: cannot produce a wrong answer, because every memoised value here is a pure
#: function of the git hashes in its key -- the second writer stores what the
#: first one did. Duplicate compute is possible only where two worktrees share a
#: HEAD, and only on the pass that first sees it.
_CACHE_LOCK = threading.Lock()


def forget_git_objects() -> None:
    """Drop the memoised per-commit answers. For tests and for a cold read."""
    with _CACHE_LOCK:
        _OBJECT_CACHE.clear()


def _memoised(key: tuple, compute):
    """`compute()` once per key. Only truthy-resolved answers are kept.

    Safe to call from several threads; see `_CACHE_LOCK` for what that does and
    does not promise.
    """
    with _CACHE_LOCK:
        if key in _OBJECT_CACHE:
            return _OBJECT_CACHE[key]
    value, keep = compute()
    if keep:
        with _CACHE_LOCK:
            if len(_OBJECT_CACHE) >= _OBJECT_CACHE_LIMIT:
                _OBJECT_CACHE.clear()
            _OBJECT_CACHE[key] = value
    return value


def _prefetch_commit_times(root: str | Path, heads) -> None:
    """Fill the commit-time memo for many SHAs using ONE git process per batch.

    `git show -s` takes any number of revisions, so asking one commit time per
    worktree was a batching opportunity rather than a necessary spawn. Measured
    on this repository at 287 worktrees, a cold observation spent 273 of its 886
    subprocesses on exactly this question.

    It changes no answer. The key, the value and the "only successful answers
    are kept" rule are the ones `_memoised` already applies, so a batch that
    fails or comes back short simply leaves those SHAs to the per-worktree path
    that was always there. That is the property worth keeping: this is a
    prefetch, never a source of truth, and deleting it must slow the reading
    down without changing a single field on the mesh.
    """
    with _CACHE_LOCK:
        wanted = [sha for sha in dict.fromkeys(heads)
                  if sha and ("committed-at", sha) not in _OBJECT_CACHE]
    for start in range(0, len(wanted), _COMMIT_BATCH):
        batch = wanted[start:start + _COMMIT_BATCH]
        code, out = _git("show", "-s", "--format=%H %ct", *batch, cwd=root)
        if code != 0:
            return
        for line in out.splitlines():
            sha, _, when = line.strip().partition(" ")
            when = when.strip()
            if not when.isdigit():
                continue
            with _CACHE_LOCK:
                if ("committed-at", sha) in _OBJECT_CACHE:
                    continue
                if len(_OBJECT_CACHE) >= _OBJECT_CACHE_LIMIT:
                    _OBJECT_CACHE.clear()
                _OBJECT_CACHE[("committed-at", sha)] = int(when)


def _git(*args: str, cwd: str | Path | None = None) -> tuple[int, str]:
    """Run a read-only git query. Returns (returncode, stdout); never raises."""
    environment = dict(os.environ)
    # `git status` is observational here.  Without this flag Git is allowed to
    # refresh the index as an optional optimisation even though the command's
    # answer is read-only.  That turns opening a Fleet view into a write to
    # another session's worktree and can contend on index.lock.
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        probe = subprocess.run(
            toolpath.argv("git", *args),
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            env=environment,
            **toolpath.no_window(),
        )
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return probe.returncode, probe.stdout


@dataclass(frozen=True)
class Worktree:
    """One worktree, as git describes it plus what can be derived about it."""

    #: As git reports it, which means **forward slashes on every platform**
    #: including Windows. Comparing it to `str(Path(...))` will mismatch there;
    #: compare `Path(worktree.path)` against `Path(other)` instead.
    path: str
    head: str
    branch: str | None
    detached: bool
    is_primary: bool
    #: Directory basename — a label to show, never an identity to trust.
    label: str
    #: The tool family that made it, or UNATTRIBUTED.
    tool_family: str
    #: The rule that produced `tool_family`, so a reader can disagree with it.
    tool_evidence: str
    #: Whether the directory still exists. A worktree git still lists but whose
    #: directory is gone is reported, not repaired.
    present: bool
    #: Commit time of HEAD in unix seconds, or None when it could not be read.
    head_committed_at: int | None
    #: Whether HEAD is reachable from the mainline ref. None means undetermined,
    #: which is not the same as False.
    on_mainline: bool | None
    #: The session that MADE it, where the path convention carries one, else
    #: UNATTRIBUTED. Derived from a location git records rather than from
    #: anything the agent volunteered — but a directory name is chosen by the
    #: tool, so this is a strong hint and never authentication.
    #:
    #: It names the creator, not the occupant. A second session can enter this
    #: worktree and work in it, and nothing in the path changes when it does, so
    #: its edits land here under the creator's id. `worktree_cache` compares this
    #: against what a session declares about itself, which is the only place the
    #: difference can be seen.
    session: str = UNATTRIBUTED
    session_evidence: str = "no path convention carried a session id"
    #: Uncommitted paths, and paths changed in commits not on the mainline.
    dirty_paths: tuple[str, ...] = ()
    ahead_paths: tuple[str, ...] = ()

    @property
    def touched_paths(self) -> frozenset[str]:
        """Everything this worktree has changed and not yet landed."""
        return frozenset(self.dirty_paths) | frozenset(self.ahead_paths)

    @property
    def dirty(self) -> bool:
        return bool(self.dirty_paths)

    def age_days(self, now: float | None = None) -> float | None:
        """Days since HEAD was committed. None when the commit time is unknown —
        never 0, which would read as "committed just now"."""
        if self.head_committed_at is None:
            return None
        return max(0.0, ((now if now is not None else time.time())
                         - self.head_committed_at) / 86_400.0)


@dataclass(frozen=True)
class Contention:
    """Two worktrees with work outstanding on the same paths."""

    left: str
    right: str
    paths: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.paths)


@dataclass(frozen=True)
class WorktreeMesh:
    """Every worktree of one repository, and where they collide."""

    repo_root: str
    observed_at: int
    worktrees: tuple[Worktree, ...] = ()
    contentions: tuple[Contention, ...] = ()
    #: Conditions encountered while observing — a missing directory, a query that
    #: failed. Named rather than swallowed.
    notes: tuple[str, ...] = ()
    limits: tuple[str, ...] = MESH_LIMITS
    #: True when the caller asked `observe` to stop before it had read every
    #: worktree. The mesh is then a prefix of one, not a small one, and the
    #: difference is not visible from the contents: three worktrees observed of
    #: forty-eight looks exactly like a repository with three. A reader that
    #: cannot tell them apart would report "no contention" about a repository it
    #: never finished looking at, so the flag is carried rather than inferred.
    stopped: bool = False

    @property
    def agent_worktrees(self) -> tuple[Worktree, ...]:
        return tuple(w for w in self.worktrees if not w.is_primary)

    @property
    def unattributed(self) -> tuple[Worktree, ...]:
        return tuple(w for w in self.agent_worktrees
                     if w.tool_family == UNATTRIBUTED)

    def stale(self, *, older_than_days: float = 7.0,
              now: float | None = None) -> tuple[Worktree, ...]:
        """Agent worktrees whose HEAD is older than the threshold.

        A worktree whose commit time is unknown is NOT returned: "unknown age"
        and "old" are different answers and only one of them is this one.
        """
        out = []
        for worktree in self.agent_worktrees:
            age = worktree.age_days(now)
            if age is not None and age > older_than_days:
                out.append(worktree)
        return tuple(out)


def _session_hint(path: str) -> tuple[str, str]:
    """Derive (session, evidence) from a path that carries a session id.

    UNATTRIBUTED for every convention that does not. This is the only route to a
    session identity that does not depend on an agent volunteering one, and it is
    a path rule rather than a fact git asserts — see `_SESSION_CONVENTIONS`.
    """
    posix = path.replace("\\", "/")
    for pattern, family in _SESSION_CONVENTIONS:
        match = pattern.search(posix)
        if match:
            return match.group(1).lower(), (
                f"session id in the {family} scratchpad path; a directory name "
                f"is chosen by the tool, so this is a derivation and not proof")
    return UNATTRIBUTED, "no path convention carried a session id"


def session_for_path(path: str) -> str:
    """The session a worktree path attributes itself to, or UNATTRIBUTED.

    The same derivation `observe()` records on each `Worktree`, exposed so a
    caller holding only a stored path can ask the question without re-observing
    the repository. It answers who CREATED the directory — see `Worktree.session`
    for why that is not the same as who is in it.
    """
    return _session_hint(path)[0]


def _tool_family(path: str) -> tuple[str, str]:
    """Derive (family, evidence) from the path convention alone."""
    posix = path.replace("\\", "/")
    for marker, family in _TOOL_CONVENTIONS:
        if marker in posix:
            return family, f"parent directory {marker!r}"
    # A session-carrying path identifies its tool too, and is checked second so
    # an explicit worktrees/ directory still wins where both would match.
    for pattern, family in _SESSION_CONVENTIONS:
        if pattern.search(posix):
            return family, f"{family} session scratchpad path"
    # The sibling convention (`<repo>.worktrees/…`) is a real convention but is
    # not owned by any one tool, so it identifies placement and stops there.
    if ".worktrees/" in posix:
        return UNATTRIBUTED, "sibling '.worktrees/' directory, tool not identified"
    return UNATTRIBUTED, "no known directory convention matched"


def _parse_worktree_list(stdout: str) -> list[dict]:
    """Parse `git worktree list --porcelain` into records."""
    records: list[dict] = []
    current: dict = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            if current:
                records.append(current)
            current = {"worktree": value}
        elif key == "HEAD":
            current["HEAD"] = value
        elif key == "branch":
            current["branch"] = value
        elif key == "detached":
            current["detached"] = True
        elif key == "prunable":
            current["prunable"] = value or True
    if current:
        records.append(current)
    return records


def _short_branch(ref: str | None) -> str | None:
    if not ref:
        return None
    return ref[len("refs/heads/"):] if ref.startswith("refs/heads/") else ref


def _changed_paths(cwd: str, *args: str) -> tuple[tuple[str, ...], str | None]:
    code, out = _git(*args, cwd=cwd)
    if code != 0:
        return (), f"{' '.join(args)} failed in {cwd}"
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    if len(paths) > _MAX_PATHS_PER_WORKTREE:
        kept = tuple(paths[:_MAX_PATHS_PER_WORKTREE])
        return kept, (f"{cwd} reports {len(paths)} changed paths; only the first "
                      f"{_MAX_PATHS_PER_WORKTREE} were compared")
    return tuple(paths), None


def _status_paths(cwd: str) -> tuple[tuple[str, ...], str | None]:
    code, out = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=cwd)
    if code != 0:
        return (), f"status failed in {cwd}"
    paths = []
    for line in out.splitlines():
        if len(line) < 4:
            continue
        entry = line[3:].strip()
        # Renames read "old -> new"; the new path is the one being contended.
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1]
        paths.append(entry.strip('"'))
    if len(paths) > _MAX_PATHS_PER_WORKTREE:
        return tuple(paths[:_MAX_PATHS_PER_WORKTREE]), (
            f"{cwd} reports {len(paths)} uncommitted paths; only the first "
            f"{_MAX_PATHS_PER_WORKTREE} were compared")
    return tuple(paths), None


def _main_worktree(root: Path, worktree_list_output: str) -> str:
    """The repository's MAIN worktree, whichever worktree we were run from.

    `git rev-parse --show-toplevel` answers "the root of the checkout I am
    standing in", which is a different question and gives a different answer
    inside every linked worktree. Using it here marked the caller's own worktree
    as the primary checkout, and primary is exempt from session derivation -- so
    a session running `tools/fleet.py` from its own worktree could not identify
    itself, which is precisely how AGENTS.md §18 says to run it.

    `git worktree list --porcelain` lists the main worktree first, and that is
    documented behaviour rather than an observed accident. `--git-common-dir` is
    tried first because it states the answer outright: it resolves to the main
    worktree's `.git` from anywhere in the repository.
    """
    code, common = _git("rev-parse", "--path-format=absolute",
                        "--git-common-dir", cwd=root)
    if code == 0 and common.strip():
        git_dir = Path(common.strip())
        # `<main>/.git` for an ordinary repository; a bare repo has no worktree
        # above it, in which case fall through rather than invent a parent.
        if git_dir.name == ".git" and git_dir.parent.is_dir():
            return str(git_dir.parent).replace("\\", "/")
    for record in _parse_worktree_list(worktree_list_output):
        if record.get("worktree"):
            return record["worktree"]
    return str(root)


#: A worktree the reading never reached, because the aggregate budget expired
#: before its turn came up. Distinct from `_FAILED`: this one TRUNCATES.
_NOT_OBSERVED_BUDGET = object()

#: The same, for a caller's `should_stop` predicate.
_NOT_OBSERVED_STOP = object()

#: A worktree that was reached and raised. Noted and skipped; it does NOT
#: truncate, because everything after it was still read.
_FAILED = object()


def _observe_one(record: dict, *, root: Path, canonical_root: str,
                 mainline: str, mainline_ok: bool, mainline_sha: str,
                 deadline: float | None, should_stop,
                 include_session_hints: bool = True):
    """Read ONE worktree. Returns `(outcome, notes)`.

    `outcome` is a `Worktree`, `None` for a record with no path, or one of the
    `_NOT_OBSERVED_*` sentinels. Notes are RETURNED rather than appended to a
    shared list, because several of these run at once and the order a shared
    list ended up in would be the order the pool finished in.

    Everything this touches is either its own argument, an immutable module
    constant, or `_OBJECT_CACHE` behind `_CACHE_LOCK`. It writes nothing else,
    which is the property that makes running it concurrently a scheduling change
    and not a semantic one.

    The budget is checked HERE, at the start of the task, rather than by the
    submitting loop. A task whose turn comes after the deadline therefore
    returns without spawning anything, so a pool with a hundred tasks queued
    drains in microseconds instead of running them all out.
    """
    if deadline is not None and time.monotonic() >= deadline:
        return _NOT_OBSERVED_BUDGET, ()
    # Called from a pool thread now, and possibly from several at once. A
    # predicate that reads a flag -- which is what every caller in this
    # repository passes -- is fine; one that mutates unguarded state is not.
    if should_stop is not None and should_stop():
        return _NOT_OBSERVED_STOP, ()

    notes: list[str] = []
    path = record.get("worktree", "")
    if not path:
        return None, ()
    present = Path(path).is_dir()
    is_primary = _same_path(path, canonical_root)
    family, evidence = ((("primary", "the repository's own checkout")
                         if is_primary else _tool_family(path)))
    # The primary checkout belongs to whoever is at the keyboard, so reading
    # a session out of its path would attribute the owner's own tree to
    # whichever agent happened to observe it.
    if is_primary:
        session, session_evidence = (
            UNATTRIBUTED,
            "the repository's own checkout belongs to no session",
        )
    elif include_session_hints:
        session, session_evidence = _session_hint(path)
    else:
        # The content-free observation boundary. It has to be honoured HERE:
        # the pool replaced the loop this branch used to live in, and a flag
        # that still type-checks in `observe` while reaching no reader is a
        # privacy seam that reports itself as enforced.
        session, session_evidence = (
            UNATTRIBUTED,
            "session hints disabled; structural worktree observation only",
        )
    head = record.get("HEAD", "")

    # A commit's timestamp is a property of the commit, so this is asked once
    # per SHA for the life of the process rather than once per observation.
    committed_at = None
    if head:
        def _committed_at(sha=head):
            code, when = _git("show", "-s", "--format=%ct", sha, cwd=root)
            resolved = (code == 0 and when.strip().isdigit())
            return (int(when.strip()) if resolved else None), resolved
        committed_at = _memoised(("committed-at", head), _committed_at)

    # Ancestry and the ahead-diff are properties of the PAIR, so the key
    # names both hashes. A commit, a fetch or a rebase moves one of them and
    # the next pass simply misses.
    on_mainline = None
    ahead: tuple[str, ...] = ()
    if head and mainline_ok:
        def _ancestry(sha=head):
            landed = _git("merge-base", "--is-ancestor", sha, mainline,
                          cwd=root)[0] == 0
            if landed:
                return (True, (), None), True
            paths, note = _changed_paths(
                str(root), "diff", "--name-only", f"{mainline}...{sha}")
            # A truncated or failed diff is not cached: it would freeze a
            # partial answer against a key that can never miss again.
            return (False, paths, note), note is None
        on_mainline, ahead, note = _memoised(
            ("ancestry", head, mainline_sha), _ancestry)
        if note:
            notes.append(note)

    dirty: tuple[str, ...] = ()
    if present:
        dirty, note = _status_paths(path)
        if note:
            notes.append(note)
    else:
        notes.append(f"{path} is listed by git but its directory is missing; "
                     f"its uncommitted work is UNMEASURED")

    return Worktree(
        path=path,
        head=head,
        branch=_short_branch(record.get("branch")),
        detached=bool(record.get("detached")),
        is_primary=is_primary,
        label=PurePosixPath(path.replace("\\", "/")).name,
        tool_family=family,
        tool_evidence=evidence,
        session=session,
        session_evidence=session_evidence,
        present=present,
        head_committed_at=committed_at,
        on_mainline=on_mainline,
        dirty_paths=dirty,
        ahead_paths=ahead,
    ), tuple(notes)


def observe(repo_root: str | Path | None = None, *,
            mainline: str = "origin/main",
            now: float | None = None,
            should_stop=None,
            include_session_hints: bool = True) -> WorktreeMesh:
    """Read every worktree of a repository and compute where they contend.

    ``mainline`` is the ref that "already landed" means. It is a parameter rather
    than a constant because a repository whose default branch is named otherwise
    would otherwise have every worktree reported as ahead of nothing.

    The worktrees are read on a bounded thread pool -- see ``observe_pool_width``
    for the width and where it comes from. The commands, their arguments and
    their number are exactly those the serial reading issued; only the wall
    clock moves, because ``subprocess.run`` releases the GIL for the whole life
    of the git process it is waiting on. Tasks are submitted in the order git
    listed the worktrees and collected in that same order, so **the result does
    not depend on which worker finished first** -- asserted, not assumed, in
    ``tests/runtime/test_mesh_read_cost.py``.

    ``should_stop`` is an optional predicate that decides whether each worktree
    is read. It exists because this function is slow in a way callers cannot
    bound: it is a few git subprocesses per worktree, this repository has 163 of
    them, and each query may sit for ``_GIT_TIMEOUT`` seconds. A GUI running it
    on a worker thread had no way to abandon it, so a window close had to wait
    the whole reading out -- which Windows records as an application hang and
    ends the process for.

    **It is now called from the pool's threads**, and from several of them at
    once. Every caller in this repository passes a predicate that reads a flag,
    which is safe; one that mutates unguarded state is the caller's problem and
    this sentence is the notice.

    One worktree is still the honest granularity and the docstring says so
    rather than promising better: a ``subprocess.run`` already in flight is not
    interruptible, so the tail of the worktrees in flight is still owed. There
    are up to ``observe_pool_width()`` of those rather than one -- but they are
    owed CONCURRENTLY, so the wall-clock tail is unchanged.

    An exception reading one worktree costs that worktree and no other: it is
    reported as a note and the reading continues. It does not set ``stopped``,
    because everything after it was read.

    A stopped reading returns what it had, with ``stopped`` set and a note. It is
    a **prefix** of an observation and not a small one, so contention computed
    from it can only under-report; do not draw it.

    The pool does not weaken that word. Truncation happens at the FIRST worktree
    that was not read, and any later one a worker had already finished is
    discarded rather than reported. That costs at most a pool's width of
    completed readings at the boundary, and it buys the two properties the
    contract is made of: it is still literally a prefix, and it is still the
    same prefix whichever order the pool ran in.

    ``OBSERVE_BUDGET_SECONDS`` is the aggregate ceiling and it ends a reading the
    same way, with the same prefix contract. ``_GIT_TIMEOUT`` bounds one query;
    it never bounded their sum, and a caller with no cancel had no ceiling at all.

    Per-commit answers are memoised in ``_OBJECT_CACHE`` across calls, keyed on
    the git hashes they are a function of rather than on a clock. Repeated
    observations of an unchanged repository therefore cost roughly the
    ``git status`` walks alone. ``forget_git_objects()`` empties it; correctness
    never depends on doing so, because a changed commit is a changed key.

    ``include_session_hints=False`` is the content-free observation boundary.
    It never calls the session hint adapter and reports every non-primary
    worktree session as ``UNATTRIBUTED`` with an explicit reason.  The default
    remains True for compatibility with the Fleet surfaces that intentionally
    use path-derived hints.  The flag is exact-bool gated so ``1`` cannot enable
    a privacy-sensitive seam by accident.
    """
    if type(include_session_hints) is not bool:
        raise TypeError("include_session_hints must be an exact bool")
    root = Path(repo_root or Path.cwd())
    observed_at = int(now if now is not None else time.time())
    notes: list[str] = []

    code, out = _git("worktree", "list", "--porcelain", cwd=root)
    if code != 0:
        return WorktreeMesh(
            repo_root=str(root), observed_at=observed_at,
            notes=(f"{root} is not a readable git repository; no worktree could "
                   f"be observed",))

    canonical_root = _main_worktree(root, out)

    # A mainline that does not resolve makes "ahead of the mainline" unanswerable.
    # Say so once rather than reporting every worktree as ahead of nothing.
    #
    # The SHA this prints is kept rather than thrown away: it is what makes the
    # ancestry answers below content-addressable. A cache keyed on the ref NAME
    # would go stale the moment somebody fetched; keyed on what the name
    # resolved to, a fetch is simply a different key.
    code, resolved = _git("rev-parse", "--verify", "--quiet", mainline, cwd=root)
    mainline_ok = code == 0
    mainline_sha = resolved.strip() if mainline_ok else ""
    if not mainline_ok:
        notes.append(f"mainline ref {mainline!r} does not resolve; "
                     f"on_mainline and ahead-of-mainline paths are UNMEASURED")

    worktrees: list[Worktree] = []
    records = _parse_worktree_list(out)
    stopped = False
    # One process for every commit time this reading still needs, before the
    # loop asks for them one at a time. Purely a prefetch into the same memo.
    _prefetch_commit_times(root, [record.get("HEAD", "") for record in records])
    budget = OBSERVE_BUDGET_SECONDS
    deadline = (time.monotonic() + budget) if budget and budget > 0 else None
    width = max(1, min(observe_pool_width(), len(records) or 1))

    # Submitted in record order and COLLECTED in record order, so nothing below
    # can observe the order the pool happened to finish in.
    outcomes: list[tuple[object, tuple[str, ...]]] = []
    with futures.ThreadPoolExecutor(
            max_workers=width, thread_name_prefix="mesh-observe") as pool:
        pending = [
            pool.submit(
                _observe_one, record,
                root=root, canonical_root=canonical_root, mainline=mainline,
                mainline_ok=mainline_ok, mainline_sha=mainline_sha,
                deadline=deadline, should_stop=should_stop,
                include_session_hints=include_session_hints)
            for record in records
        ]
        for index, pledge in enumerate(pending):
            try:
                outcomes.append(pledge.result())
            except BaseException as exc:  # noqa: BLE001 - see below
                # One worktree must not cost the other 162. A worker already
                # swallows every failure git can produce -- `_git` never raises
                # -- so reaching here means something genuinely unexpected, and
                # the honest report is to lose that ONE record loudly rather
                # than to emit a record whose empty `dirty_paths` would read as
                # a clean worktree.
                outcomes.append((
                    _FAILED,
                    (f"reading {records[index].get('worktree', '')!r} raised "
                     f"{type(exc).__name__}: {exc}; it is absent from this "
                     f"mesh and its uncommitted work is UNMEASURED",),
                ))

    for index, (produced, produced_notes) in enumerate(outcomes):
        if produced is _NOT_OBSERVED_BUDGET or produced is _NOT_OBSERVED_STOP:
            # Truncate at the FIRST worktree that was not read, discarding any
            # later one the pool had already finished. That is what keeps this a
            # prefix rather than a completion-order-dependent subset, and it is
            # the reason the result does not depend on how the pool scheduled.
            stopped = True
            if produced is _NOT_OBSERVED_BUDGET:
                notes.append(
                    f"the observation ran out of its {budget:.0f}s budget after "
                    f"{len(worktrees)} of {len(records)} worktree(s); what is "
                    f"here is a prefix of an observation, not a complete small "
                    f"one, and its contention is an under-count")
            else:
                notes.append(
                    f"the observation was stopped after {len(worktrees)} of "
                    f"{len(records)} worktree(s); what is here is a prefix of "
                    f"an observation, not a complete small one, and its "
                    f"contention is an under-count")
            break
        notes.extend(produced_notes)
        if produced is not None and produced is not _FAILED:
            worktrees.append(produced)

    return WorktreeMesh(
        repo_root=canonical_root,
        observed_at=observed_at,
        worktrees=tuple(worktrees),
        contentions=find_contentions(worktrees),
        notes=tuple(notes),
        stopped=stopped,
    )


def _same_path(left: str, right: str) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return left.replace("\\", "/") == right.replace("\\", "/")


def find_contentions(worktrees) -> tuple[Contention, ...]:
    """Every pair of worktrees with outstanding work on a shared path.

    Pure: takes records, returns records, reads nothing. The primary checkout is
    included — work in progress there contends with an agent's worktree exactly
    as two agents' worktrees contend with each other, and excluding it would hide
    the most likely collision of all.

    The pairing is BETWEEN worktrees, and that bounds what it can find. Two
    sessions working inside one worktree produce a single record whose
    `touched_paths` already merges both, so they cannot appear as a pair here and
    this function returns nothing however hard they collide. That case is real —
    it is how a session's uncommitted work gets swept into a peer's next
    `git add -A` — and it is answered by `worktree_cache.occupancy_conflicts()`,
    which compares declared occupants instead of paths. Stated here because a
    caller reading "no contention" deserves to know which question was asked.
    """
    items = [w for w in worktrees if w.touched_paths]
    out: list[Contention] = []
    for i, left in enumerate(items):
        for right in items[i + 1:]:
            shared = left.touched_paths & right.touched_paths
            if shared:
                out.append(Contention(left=left.path, right=right.path,
                                      paths=tuple(sorted(shared))))
    # Most-contended first: a reader wants the worst collision, not the first one
    # git happened to list.
    out.sort(key=lambda c: (-c.count, c.left, c.right))
    return tuple(out)
