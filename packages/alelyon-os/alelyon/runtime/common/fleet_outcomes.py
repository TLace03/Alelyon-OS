"""Did the work land? The outcome label the ledger's score could not supply.

`fleet_ledger` scores completion and cost, and says so in its own limits: *an
agent that settled quickly and cheaply on a wrong answer scores well, and
nothing here would notice.* `docs/features/FLEET-HIERARCHY.md` §5 names that as
the single largest gap in the hierarchy and says it is not closable without an
outcome label somebody has to supply.

This supplies one, from git.

Why landing, and why it is not self-reported
--------------------------------------------
An agent worked on a branch. Either that branch reached the mainline or it did
not, and the answer is written by whoever merged it — not by the agent, not by
the harness, and not by anything in this repository that an agent can reach.
That is the same rule every other input to the ledger already follows, and the
reason `CLAIMS.md` §2.3 gives for it: a guard keyed on what the writer emitted
is no guard.

The branch comes from `gitBranch`, which the harness stamps on each transcript
record as it is written, so it is the branch that was checked out AT THE TIME
rather than whatever is checked out when something reads the transcript back.

Four outcomes, and only one of them moves a score
--------------------------------------------------
``LANDED``      the branch reached the mainline.
``IN_FLIGHT``   it has not, and not enough time has passed to call that a
                result. **No evidence yet**, so it must not penalise.
``ABANDONED``   it has not, and the settling window has passed.
``UNKNOWN``     nothing here can say — no branch was recorded, the work was
                done on the mainline itself, the branch belongs to another
                repository, or its history is unrelated to this one.

``ABANDONED`` is the only label the ledger scores, and it scores it **downward**.
Landing is recorded and does not raise anything. That asymmetry is deliberate
and is argued where it applies, in `fleet_ledger.Run.score`.

Why PR evidence is read before ref state
-----------------------------------------
`branch_index.observe()` builds its records from the union of the refs that
exist now and the branch names it parsed out of merge subjects, so a branch that
was merged and then deleted falls through to `OPEN` — it has no ref to be an
ancestor with. **Measured against this repository on 2026-08-04: 4 of 74
branches had been merged by a pull request and still read `OPEN`.** Taking
`state == MERGED` as the question would have scored all four ABANDONED, which is
a false penalty in the one direction that matters. So a merge commit naming the
branch is checked first, and the ref state only decides the cases it cannot.

Nothing here writes to the repository. Every git call is a query.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import time

from alelyon.runtime.common import toolpath

#: The label vocabulary. Closed, for the same reason the ledger's refusal
#: reasons are closed: "why was my run scored that way" must have an answer that
#: is the same every time it is asked.
LANDED = "LANDED"
IN_FLIGHT = "IN-FLIGHT"
ABANDONED = "ABANDONED"
UNKNOWN = "UNKNOWN"
OUTCOMES = (LANDED, IN_FLIGHT, ABANDONED, UNKNOWN)

#: Outcomes that can still change as the repository moves. A run recorded while
#: its branch was in flight is re-read later; a terminal one never is.
PROVISIONAL = (IN_FLIGHT, UNKNOWN)
TERMINAL = (LANDED, ABANDONED)

#: How long an unmerged branch is given before its silence is read as a result.
#: **A declaration with a date on it, not a measurement** — the same standing
#: this module's neighbours give `MIN_RUNS`. Fourteen days is longer than any
#: branch in this repository's history took to land and short enough that the
#: label eventually says something; it is not derived from that distribution,
#: because fitting it against the runs it scores would be circular.
SETTLING_DAYS = 14
SETTLING_SECONDS = SETTLING_DAYS * 24 * 60 * 60

_GIT_TIMEOUT = 60

#: Local-ref states, for branches no remote namespace knows about.
_NO_REF = "no-ref"
_ANCESTOR = "ancestor"
_NOT_ANCESTOR = "not-ancestor"


def _git(*args: str, repo) -> tuple[int, str]:
    """A query. Nothing in this module writes to the repository."""
    try:
        proc = subprocess.run(toolpath.argv("git", "-C", str(repo), *args),
                              capture_output=True, text=True,
                              timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return 1, ""
    return proc.returncode, proc.stdout.strip()

LANDING_LIMITS: tuple[str, ...] = (
    "Landing is not correctness. Work can land and be wrong, and work can be "
    "right and be superseded by a better approach the same week. This measures "
    "whether a branch was taken up, which is the best externally-written "
    "signal available and is not a quality judgement.",
    "The label is a property of the BRANCH, so every agent that ran on it "
    "carries the same one. An agent that contributed nothing to a branch that "
    "landed reads LANDED, and one whose good work sat on a branch the owner "
    "closed for unrelated reasons reads ABANDONED.",
    "A branch that was squash-merged leaves neither an ancestor nor, in "
    "general, a parseable merge subject, so it reads as never landed. This "
    "repository merges rather than squashes, which is what makes the signal "
    "readable here; it would be near-silent on a squash-only repository.",
    "A branch that both was deleted AND arrived without a pull-request merge "
    "subject leaves nothing to derive from at all. It reads UNKNOWN rather "
    "than being guessed at in either direction.",
    "A branch that was never pushed is read from this checkout's local refs, "
    "which every linked worktree shares. One that only ever existed in a clone "
    "somewhere else leaves no trace here and reads UNKNOWN.",
    "The settling window is a declaration, not a finding. A branch merged the "
    "day after the window closes was ABANDONED when it was read and LANDED "
    "afterwards, and only the second reading is kept.",
    "IN-FLIGHT and UNKNOWN are the absence of evidence, not evidence of "
    "absence. Neither moves a score, and reporting either as a result would be "
    "reporting a null check beside a filled one as though both had passed.",
    "A branch carrying no commits of its own is trivially an ancestor of the "
    "mainline and reads LANDED, even though nothing landed. It costs nothing "
    "because LANDED moves no score - which is the asymmetry earning its keep - "
    "but it is not evidence that any work arrived.",
)


@dataclass(frozen=True)
class Landing:
    """What became of the branch a run was done on, and how that was decided."""

    outcome: str
    branch: str
    evidence: str

    @property
    def penalises(self) -> bool:
        """Whether this outcome is allowed to move a score. Only one is."""
        return self.outcome == ABANDONED

    @property
    def is_terminal(self) -> bool:
        """Whether re-reading the repository could still change this."""
        return self.outcome in TERMINAL

    def __str__(self) -> str:
        return f"{self.outcome} — {self.evidence}"


def _unknown(branch: str, evidence: str) -> Landing:
    return Landing(outcome=UNKNOWN, branch=branch, evidence=evidence)


class LandingIndex:
    """Answers the landing question for many runs off one reading of git.

    `branch_index.observe()` costs a few hundred subprocesses on a repository
    this size, so it is read once and held. That makes the index a **snapshot**:
    every answer it gives is as of its construction, which is the property
    `reconcile` on the ledger depends on being able to state.
    """

    def __init__(self, repo_root: str | Path = ".", *,
                 mainline: str | None = None,
                 settling_seconds: int = SETTLING_SECONDS,
                 index=None) -> None:
        # `branch_index` is imported HERE and not at module scope so that the
        # vocabulary above — which is all `fleet_ledger` needs to store and
        # score a label — costs nothing to import. The ledger is the record;
        # this class is the deriver, and a record that cannot be read without
        # git installed is a record with a dependency it never uses. It also
        # keeps the alelyon-os fleet subsystem installable without the branch
        # machinery: the ledger degrades to UNKNOWN, which is what it means.
        # (Written as a subsystem and not as an install extra on purpose —
        # `fleet` is a subsystem; the declared extras are `sdk` and `stream`,
        # and naming a non-existent extra in shipped code sends a reader to an
        # install that resolves to nothing. A test scans for exactly that.)
        from alelyon.runtime.common import branch_index as BI

        self._BI = BI
        self.repo_root = Path(repo_root)
        self.mainline = mainline or BI.MAINLINE_DEFAULT
        self.settling_seconds = int(settling_seconds)
        self.index = index if index is not None else BI.observe(
            str(repo_root), mainline=self.mainline)
        self._records = {r.name: r for r in self.index.records}
        #: `origin/main` and `main` are the same branch wearing two names, and
        #: the harness stamps the local one.
        self._mainline_names = {self.mainline, self.mainline.split("/")[-1]}
        self._repo_key = _git_common_dir(self.repo_root)
        self._cwd_cache: dict[str, bool | None] = {}
        self._local_cache: dict[str, str] = {}

    # ── the question ─────────────────────────────────────────────────────────
    def of(self, branch: str, *, settled_at: int, now: int | None = None,
           cwd: str = "") -> Landing:
        """What became of `branch`, for a run that settled at `settled_at`.

        The order of these tests is load-bearing, and the reason is in the
        module docstring: positive evidence of landing is read before ref state,
        because a merged-and-deleted branch has no ref to be an ancestor with.
        """
        name = str(branch or "").strip()
        moment = int(now if now is not None else time.time())

        if not name:
            return _unknown(name, "the harness stamped no branch on this "
                                  "transcript, so there is nothing to look up")
        if name in self._mainline_names:
            return _unknown(name, f"this ran on {name}, the mainline itself, "
                                  f"where there is no branch whose landing "
                                  f"could be observed")
        if self._elsewhere(cwd):
            return _unknown(name, f"this ran in {cwd!r}, which is a different "
                                  f"repository from {self.repo_root}; a branch "
                                  f"name means nothing across repositories")

        record = self._records.get(name)
        if record is None:
            return self._from_local_ref(name, settled_at=settled_at,
                                        now=moment)

        if record.prs:
            pulls = ", ".join(f"#{p}" for p in record.prs)
            return Landing(LANDED, name, (
                f"a merge commit on {self.mainline} names {name} as the source "
                f"of pull request {pulls}"))
        if record.state == self._BI.MERGED:
            return Landing(LANDED, name,
                           f"{name} is an ancestor of {self.mainline}")
        if record.state == self._BI.ABSORBED:
            return Landing(LANDED, name, (
                f"{name} reached {self.mainline} inside "
                f"{record.absorbed_into!r}, which was cut from it — it landed "
                f"under another branch's name"))
        if record.state == self._BI.UNRELATED:
            return _unknown(name, (
                f"{name} shares no merge-base with {self.mainline}; it is a "
                f"separate history and landing does not apply to it"))

        return self._by_window(name, settled_at=settled_at, now=moment,
                               how=f"{name} has not reached {self.mainline}")

    def _from_local_ref(self, name: str, *, settled_at: int,
                        now: int) -> Landing:
        """The branch is in no remote namespace. Ask the local refs.

        Without this, every branch that was never pushed reads UNKNOWN — and
        those are exactly the branches most likely to have been abandoned, so
        the one label that carries a penalty would be the one systematically
        withheld. An agent worktree's branch is local until something publishes
        it, and most never are.
        """
        state = self._local_cache.get(name)
        if state is None:
            state = self._local_cache[name] = self._read_local_ref(name)
        if state == _NO_REF:
            return _unknown(name, (
                f"no branch named {name!r} appears in {self.mainline}'s "
                f"history, its refs, or this checkout's local refs, so neither "
                f"landing nor abandonment can be shown"))
        if state == _ANCESTOR:
            return Landing(LANDED, name, (
                f"the local branch {name} is an ancestor of {self.mainline}; "
                f"it landed without a remote ref being kept for it"))
        return self._by_window(
            name, settled_at=settled_at, now=now,
            how=(f"the local branch {name} has not reached {self.mainline} and "
                 f"was never pushed"))

    def _read_local_ref(self, name: str) -> str:
        rc, _ = _git("rev-parse", "--verify", "--quiet", f"refs/heads/{name}",
                     repo=self.repo_root)
        if rc != 0:
            return _NO_REF
        rc, _ = _git("merge-base", "--is-ancestor", f"refs/heads/{name}",
                     self.mainline, repo=self.repo_root)
        return _ANCESTOR if rc == 0 else _NOT_ANCESTOR

    def _by_window(self, name: str, *, settled_at: int, now: int,
                   how: str) -> Landing:
        """IN-FLIGHT or ABANDONED, decided only by how long the silence lasted.

        A run with no settle time is UNKNOWN, not ABANDONED. `runs_from_activity`
        records `at=agent.last_at or 0`, so a transcript whose timestamps could
        not be parsed arrives here as zero — and measuring a window from the
        epoch makes every such run look decades stale. That would fire the one
        label carrying a penalty on the ABSENCE of evidence, which is the exact
        failure this vocabulary exists to prevent.
        """
        if int(settled_at or 0) <= 0:
            return _unknown(name, (
                f"{name} has not reached {self.mainline}, but this run carries "
                f"no settle time, so there is no interval to measure the "
                f"{self.settling_seconds // 86400}d window against"))
        waited = max(0, int(now) - int(settled_at or 0))
        days, window = waited // 86400, self.settling_seconds // 86400
        if waited < self.settling_seconds:
            return Landing(IN_FLIGHT, name, (
                f"{how}, and only {days}d of the {window}d settling window has "
                f"passed — too early to call that a result"))
        return Landing(ABANDONED, name, (
            f"{how} in the {days}d since this run settled, past the {window}d "
            f"settling window"))

    # ── whose repository was this ────────────────────────────────────────────
    def _elsewhere(self, cwd: str) -> bool:
        """Whether `cwd` is positively known to be a DIFFERENT repository.

        Deliberately asymmetric. A working directory that cannot be read tells
        us nothing and must not refuse the lookup: agent worktrees are created
        under the system temp directory and are routinely deleted once the work
        is done, so treating "the path is gone" as "another repository" would
        discard the outcome of most agent runs in this fleet. Only a readable
        checkout whose git directory is genuinely a different one refuses.
        """
        path = str(cwd or "").strip()
        if not path or not self._repo_key:
            return False
        if path not in self._cwd_cache:
            other = _git_common_dir(Path(path))
            self._cwd_cache[path] = (
                False if other is None else other != self._repo_key)
        return bool(self._cwd_cache[path])

    # ── reporting ────────────────────────────────────────────────────────────
    def report(self) -> str:
        states: dict[str, int] = {}
        for record in self.index.records:
            states[record.state] = states.get(record.state, 0) + 1
        lines = [f"Landing index — {self.repo_root} against {self.mainline}", ""]
        # `a + b or c` binds as `(a + b) or c`, and the prefix is never empty,
        # so writing the fallback inline would have made it unreachable.
        breakdown = ", ".join(f"{n} {s}" for s, n in sorted(states.items()))
        lines.append(f"  {len(self.index.records)} branch(es)"
                     + (f": {breakdown}" if breakdown
                        else " — this repository has no topic branches at all"))
        deleted = sum(1 for r in self.index.records
                      if r.prs and r.state == self._BI.OPEN)
        lines.append(f"  {deleted} merged by a pull request whose ref is gone — "
                     f"these read LANDED on the merge subject, not on ancestry")
        lines.append(f"  {len(self.index.orphan_merges)} merge commit(s) name no "
                     f"branch; work that arrived through one cannot be "
                     f"attributed")
        lines += ["", "WHAT THIS CANNOT TELL YOU"]
        lines += [f"  - {limit}" for limit in LANDING_LIMITS]
        return "\n".join(lines)


def _git_common_dir(path: Path) -> str | None:
    """The one directory every worktree of a repository shares, or None.

    `--git-common-dir` rather than `--git-dir`: a linked worktree has a git dir
    of its own, so comparing those would report every agent worktree as a
    different repository, which is the opposite of true.
    """
    try:
        proc = subprocess.run(
            toolpath.argv("git", "-C", str(path), "rev-parse",
                          "--path-format=absolute", "--git-common-dir"),
            capture_output=True, text=True, timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return str(Path(out).resolve()).lower()
    except OSError:
        return out.lower()


__all__ = [
    "ABANDONED", "IN_FLIGHT", "LANDED", "LANDING_LIMITS", "OUTCOMES",
    "PROVISIONAL", "SETTLING_DAYS", "SETTLING_SECONDS", "TERMINAL", "Landing",
    "LandingIndex",
]
