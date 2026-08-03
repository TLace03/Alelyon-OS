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

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import subprocess
import time

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


def _git(*args: str, cwd: str | Path | None = None) -> tuple[int, str]:
    """Run a read-only git query. Returns (returncode, stdout); never raises."""
    try:
        probe = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
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


def observe(repo_root: str | Path | None = None, *,
            mainline: str = "origin/main",
            now: float | None = None) -> WorktreeMesh:
    """Read every worktree of a repository and compute where they contend.

    ``mainline`` is the ref that "already landed" means. It is a parameter rather
    than a constant because a repository whose default branch is named otherwise
    would otherwise have every worktree reported as ahead of nothing.
    """
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
    mainline_ok = _git("rev-parse", "--verify", "--quiet", mainline, cwd=root)[0] == 0
    if not mainline_ok:
        notes.append(f"mainline ref {mainline!r} does not resolve; "
                     f"on_mainline and ahead-of-mainline paths are UNMEASURED")

    worktrees: list[Worktree] = []
    for record in _parse_worktree_list(out):
        path = record.get("worktree", "")
        if not path:
            continue
        present = Path(path).is_dir()
        is_primary = _same_path(path, canonical_root)
        family, evidence = ((("primary", "the repository's own checkout")
                             if is_primary else _tool_family(path)))
        # The primary checkout belongs to whoever is at the keyboard, so reading
        # a session out of its path would attribute the owner's own tree to
        # whichever agent happened to observe it.
        session, session_evidence = (
            (UNATTRIBUTED, "the repository's own checkout belongs to no session")
            if is_primary else _session_hint(path))
        head = record.get("HEAD", "")

        committed_at = None
        if head:
            code, when = _git("show", "-s", "--format=%ct", head, cwd=root)
            if code == 0 and when.strip().isdigit():
                committed_at = int(when.strip())

        on_mainline = None
        ahead: tuple[str, ...] = ()
        if head and mainline_ok:
            on_mainline = _git("merge-base", "--is-ancestor", head, mainline,
                               cwd=root)[0] == 0
            if not on_mainline:
                ahead, note = _changed_paths(
                    str(root), "diff", "--name-only", f"{mainline}...{head}")
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

        worktrees.append(Worktree(
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
        ))

    return WorktreeMesh(
        repo_root=canonical_root,
        observed_at=observed_at,
        worktrees=tuple(worktrees),
        contentions=find_contentions(worktrees),
        notes=tuple(notes),
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
