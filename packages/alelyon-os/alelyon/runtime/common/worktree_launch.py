"""Wake a dormant worktree: the half `worktree_resume` deliberately refused.

`worktree_resume` computes a reviewable PLAN and stops, because "launching an
agent is a separate, explicitly authorised step that belongs to whatever
actually owns agent processes". Nothing owned them. This module is that thing.

It exists under an explicit owner authorisation recorded on 2026-08-04, and it
is built so that the authorisation is the only thing that can make it act:
`admit()` refuses every candidate when `authority` is None, and `launch()`
cannot be called without a spawner passed by name.

THE SPLIT, AND WHY IT IS THE WHOLE DESIGN
-----------------------------------------
`admit()` is PURE. No process, no disk, no clock it did not receive. It answers
"may this worktree be woken, and if not, by what name" and nothing else. Every
refusal in this module is decided there, which is why the refusals can be tested
without a display, a repository, or a cent of spend.

`launch()` is the only function with an effect. It re-runs `admit()` immediately
before each spawn -- a plan is a reading taken at `observed_at`, and a session
can walk into a directory in the seconds since -- then claims a reservation and
calls the spawner. It contains no policy of its own.

WHAT IT DOES NOT DO
-------------------
It does not recompute dormancy; `worktree_resume._dormancy` is the single source
and a second threshold would give two answers to "is this old". It does not
write into the target worktree -- no checkout, no reset, no clean -- because the
uncommitted work there is owner data. It does not claim areas or publish
findings as the woken session, because those are self-reports a session makes
about itself and forging them corrupts the only routing the fleet has. It does
not record a run in `fleet_ledger`; runs arrive from harness transcripts the
agent did not author, and a row written here would be the self-report that
ledger exists to exclude.

A STARTED PROCESS IS NOT A WORKING AGENT
----------------------------------------
`LaunchHandle` says a process was created. That is all it says. Whether an agent
is doing anything comes from `session_activity`, from transcripts the harness
wrote, and even that cannot tell a crashed agent from a finished one. Nothing
here should ever be rendered as "running".
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Callable, Mapping, Optional, Sequence, Tuple

from alelyon.runtime.common import fleet_hierarchy as HIER
from alelyon.runtime.common import paths as PATHS
from alelyon.runtime.common import worktree_resume as R

#: Bumped when the SHAPE of a launch record changes. A record written under
#: another schema is not comparable to one written under this one.
LAUNCH_SCHEMA = "alelyon.worktree-launch/0.1"

#: Where the repository declares the command that starts an agent. The same
#: file `worktree_areas` and `worktree_disciplines` read their vocabulary from,
#: for the same reason: one private repository's tooling must not be a constant
#: in a module every checkout imports.
CONFIG_PATH = ".alelyon/fleet.toml"
CONFIG_ENV = "ALELYON_FLEET_CONFIG"

#: How many launches may be in flight at once.
#:
#: **A declaration with a date on it, not a measurement** -- the same standing
#: `worktree_resume.SETTLING_DAYS` and `DORMANT_AFTER_DAYS` carry. The owner set
#: it to 1 on 2026-08-04, on the reasoning that waking one worktree per press is
#: the cheapest way to find out whether these refusals hold before anything runs
#: in parallel. It is not derived from any observed concurrency, and raising it
#: is an owner decision rather than a tuning exercise.
DEFAULT_MAX_IN_FLIGHT = 1

#: How stale a plan may be before it must be recomputed. Dormancy was read at
#: `plan.observed_at`; a session can enter a directory in the interval, and that
#: is the one error this module is most expensive to get wrong. Also a
#: declaration, not a measurement.
PLAN_TTL_SECONDS = 300


# ── the refusal vocabulary ──────────────────────────────────────────────────
#: Closed, and for the reason `fleet_ledger` states about its own: "why was my
#: worktree not woken" must have an answer that is the same every time it is
#: asked. Values are kebab-case; the constant is the name.
ACCEPTED = "accepted"

# authority -- evaluated first, so a refusal costs nothing to compute
NO_AUTHORITY = "no-owner-authority"
NOT_AUTHORISED_HERE = "authority-does-not-cover-this-worktree"
TIER3_AREA = "tier3-area"
BOARD_MATTER = "board-matter"

# eligibility -- read off worktree_resume, never recomputed
NOT_DORMANT = "not-dormant"
DORMANCY_UNKNOWN = "dormancy-unknown"
MISSING_WORKTREE = "worktree-directory-missing"
NOT_IN_PLAN = "not-in-plan"
PLAN_STALE = "plan-stale"

# occupancy -- the expensive mistake
LIVE_SESSION = "live-session-in-directory"
CONTENDED = "contends-with-selection"
AREA_CLAIMED = "area-claimed-by-another-session"

# the model and the money
MODEL_NOT_NAMED = "model-not-named"
BELOW_FLOOR = "below-capability-floor"
FLEET_FULL = "fleet-at-ceiling"
ALREADY_LAUNCHED = "already-launched"
LAUNCHER_NOT_CONFIGURED = "launcher-not-configured"

REASONS: Tuple[str, ...] = (
    ACCEPTED,
    NO_AUTHORITY, NOT_AUTHORISED_HERE, TIER3_AREA, BOARD_MATTER,
    NOT_DORMANT, DORMANCY_UNKNOWN, MISSING_WORKTREE, NOT_IN_PLAN, PLAN_STALE,
    LIVE_SESSION, CONTENDED, AREA_CLAIMED,
    MODEL_NOT_NAMED, BELOW_FLOOR, FLEET_FULL, ALREADY_LAUNCHED,
    LAUNCHER_NOT_CONFIGURED,
)

#: ASCII only. A non-ASCII arrow in a neighbouring module raised
#: UnicodeEncodeError on a cp1252 console and truncated the report it was
#: explaining, which is the worst possible line to lose.
LAUNCH_LIMITS: Tuple[str, ...] = (
    "A started process is not a working agent. This records that a process was "
    "created; whether anything is being done comes from the harness "
    "transcripts `session_activity` reads, and even those cannot tell a "
    "crashed agent from a finished one.",
    "DORMANT measures a commit, not a session. A clean worktree somebody is "
    "silently reading is indistinguishable from an abandoned one, and only a "
    "live harness record can contradict commit age.",
    "The brief is DERIVED from paths the worktree touched, not from anything "
    "it declared about its intent. A worktree that read widely and changed one "
    "file gets a brief about that one file.",
    "Nothing here records what a launch cost. Tokens appear later, from the "
    "harness, and are scored by `fleet_ledger` from records the agent did not "
    "author.",
    "A refusal that the owner overrode is recorded as overridden. The override "
    "does not delete the reason, because 'nobody may wake this' and 'somebody "
    "decided to anyway' are different facts.",
    "The concurrency ceiling is the only spend guard here. It bounds how many "
    "agents start, never what any one of them spends.",
)


# ── authority ───────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Authority:
    """Who authorised waking what, and when.

    A typed object rather than `force=True`, so a verdict can say WHICH
    authorisation admitted a launch. `AGENTS.md` reserves this to the owner; a
    session cannot mint one for itself, and nothing here checks that it did not
    -- this records the claim, it does not verify it.
    """

    granted_by: str
    granted_at: float
    #: Worktree paths this covers. Empty means every path in the plan, which is
    #: a broader grant and is recorded as such.
    covers: Tuple[str, ...] = ()
    note: str = ""

    def covers_path(self, path: str) -> bool:
        if not self.covers:
            return True
        target = _normal(path)
        return any(_normal(one) == target for one in self.covers)


def _normal(path: str) -> str:
    return str(path or "").replace("\\", "/").rstrip("/").lower()


# ── the records ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LaunchRequest:
    """What WOULD be run. Constructing one has no effect."""

    candidate: R.Candidate
    cwd: str
    model: str
    layer: str
    brief: str
    argv: Tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LaunchHandle:
    """What a spawner returns. A process was created; nothing more is claimed."""

    pid: int
    started_at_ts: float
    argv: Tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class Verdict:
    """`accepted` is one value of `reason`, never a bare bool."""

    accepted: bool
    reason: str
    detail: str = ""
    candidate: Optional[R.Candidate] = None
    handover: Optional[R.Handover] = None
    overridden: bool = False

    def __post_init__(self) -> None:
        if self.reason not in REASONS:
            raise ValueError(
                f"{self.reason!r} is not in the closed vocabulary; a refusal "
                f"nobody can look up is not a refusal")


@dataclass(frozen=True)
class LaunchReport:
    at: int
    requested: Tuple[LaunchRequest, ...] = ()
    started: Tuple[Tuple[LaunchRequest, LaunchHandle], ...] = ()
    refused: Tuple[Verdict, ...] = ()
    limits: Tuple[str, ...] = LAUNCH_LIMITS

    @property
    def any_started(self) -> bool:
        return bool(self.started)


# ── the brief ───────────────────────────────────────────────────────────────
def brief_for(candidate: R.Candidate) -> str:
    """A brief composed from what the worktree TOUCHED, labelled as derived.

    `Candidate` carries no statement of intent, and `worktree_resume` says why:
    what a worktree "was working on" is read from paths, not from anything it
    declared. So this cannot say "you were doing X" and does not pretend to.
    """
    lines = [
        f"This worktree ({candidate.label}) has been dormant for "
        f"{_age(candidate)}. What follows is DERIVED from the paths it "
        f"touched and the branch it sits on. It is not a statement the "
        f"previous session made about its own intent -- nothing recorded one.",
        "",
        f"branch: {candidate.branch or 'UNKNOWN'}",
        f"head:   {candidate.head or 'UNKNOWN'}",
    ]
    if candidate.areas:
        lines.append("areas:  " + ", ".join(str(a) for a in candidate.areas))
    touched = tuple(candidate.touched_paths or ())
    if touched:
        shown = ", ".join(touched[:12])
        more = f" (+{len(touched) - 12} more)" if len(touched) > 12 else ""
        lines.append(f"files:  {shown}{more}")
    if candidate.document:
        lines.append(f"queued: {candidate.document} -- {candidate.rank_reason}")
    if candidate.overlaps:
        lines.append("")
        lines.append(
            "CONTENDED. Another worktree touches the same paths; read what "
            "changed there before editing:")
        for overlap in candidate.overlaps[:6]:
            lines.append(
                f"  - {overlap.label} ({overlap.provenance}): "
                f"{', '.join(overlap.shared_paths[:4])}")
    lines.append("")
    lines.append(
        "Read AGENTS.md before editing. Run `python tools/fleet.py status` and "
        "claim your area before you start.")
    return "\n".join(lines)


def _age(candidate: R.Candidate) -> str:
    if candidate.age_days is None:
        return "an unmeasured time"
    return f"{candidate.age_days:.1f} days"


# ── the policy, pure ────────────────────────────────────────────────────────
def admit(candidate: R.Candidate, *,
          plan: Optional[R.ResumePlan] = None,
          authority: Optional[Authority] = None,
          live_sessions: Sequence[str] = (),
          space=None,
          model: str = "",
          in_flight: Sequence[str] = (),
          max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
          configured: bool = True,
          exists: Optional[Callable[[str], bool]] = None,
          now: Optional[float] = None) -> Verdict:
    """Whether this worktree may be woken, and if not, by what name.

    PURE. It starts nothing, writes nothing, and reads no clock or filesystem it
    was not handed. Every refusal this module can produce is decided here.

    Ordering is authority, then eligibility, then occupancy, then cost -- the
    same "authority before affinity" rule `worktree_resume.handover` states. A
    candidate failing several conditions reports the first by that order, so the
    answer is stable rather than depending on which check ran first.
    """
    moment = time.time() if now is None else float(now)

    # ── authority ───────────────────────────────────────────────────────────
    if authority is None:
        return Verdict(False, NO_AUTHORITY, candidate=candidate, detail=(
            "starting a coding agent is Tier 3 process control and spends the "
            "owner's API budget; no authorisation was supplied"))
    if not authority.covers_path(candidate.path):
        return Verdict(False, NOT_AUTHORISED_HERE, candidate=candidate, detail=(
            f"the authorisation from {authority.granted_by} names "
            f"{len(authority.covers)} worktree(s) and this is not one of them"))
    if space is not None:
        for area in candidate.areas or ():
            if _is_tier3(space, area):
                return Verdict(False, TIER3_AREA, candidate=candidate, detail=(
                    f"{area} is a capital-bearing area; it is not "
                    f"self-directed work whoever authorised the wake"))
    if HIER.board_matter(candidate.label or ""):
        return Verdict(False, BOARD_MATTER, candidate=candidate, detail=(
            "the work reads as board matter, which needs the owner's signature "
            "rather than an agent's attention"))

    # ── eligibility, read off worktree_resume ───────────────────────────────
    if candidate.dormancy == R.ACTIVE:
        return Verdict(False, NOT_DORMANT, candidate=candidate,
                       detail=candidate.dormancy_evidence)
    if candidate.dormancy != R.DORMANT:
        return Verdict(False, DORMANCY_UNKNOWN, candidate=candidate, detail=(
            f"dormancy is {candidate.dormancy}; a worktree whose age could not "
            f"be read is the one case nothing was established about"))
    if plan is not None:
        known = {_normal(one.path) for one in plan.selected}
        if _normal(candidate.path) not in known:
            return Verdict(False, NOT_IN_PLAN, candidate=candidate, detail=(
                "the plan is the reviewed artifact; this was not in it"))
        age = moment - float(plan.observed_at or 0)
        if plan.observed_at and age > PLAN_TTL_SECONDS:
            return Verdict(False, PLAN_STALE, candidate=candidate, detail=(
                f"the plan was read {age:.0f}s ago and dormancy is a reading "
                f"taken then; recompute it"))
    if exists is not None and not exists(candidate.path):
        return Verdict(False, MISSING_WORKTREE, candidate=candidate, detail=(
            "git lists the worktree but the directory is not there"))

    # ── occupancy ───────────────────────────────────────────────────────────
    if candidate.session and candidate.session in tuple(live_sessions):
        return Verdict(False, LIVE_SESSION, candidate=candidate, detail=(
            f"the harness records {candidate.session} as live in this "
            f"directory; waking one that is still held is the expensive "
            f"mistake here"))
    if plan is not None and _in_conflicts(candidate, plan):
        return Verdict(False, CONTENDED, candidate=candidate, detail=(
            "another worktree in this same selection touches these paths; "
            "waking both reproduces the collision that stopped them"))
    for overlap in candidate.overlaps or ():
        if overlap.provenance == R.DECLARED:
            return Verdict(False, AREA_CLAIMED, candidate=candidate, detail=(
                f"{overlap.label} claimed these paths; a claim is advisory, so "
                f"this is a refusal the owner may override and not one to skip"))

    # ── the model and the money ─────────────────────────────────────────────
    if not str(model or "").strip():
        return Verdict(False, MODEL_NOT_NAMED, candidate=candidate, detail=(
            "no model was named; the layer space answers what RANK work is, "
            "never which model must run it"))
    fits, why = HIER.fits(model, candidate.layer or HIER.DEFAULT_LAYER.key)
    if not fits:
        return Verdict(False, BELOW_FLOOR, candidate=candidate, detail=why)
    if len(tuple(in_flight)) >= int(max_in_flight):
        return Verdict(False, FLEET_FULL, candidate=candidate, detail=(
            f"{len(tuple(in_flight))} launch(es) in flight and the ceiling is "
            f"{max_in_flight}"))
    if _normal(candidate.path) in {_normal(one) for one in in_flight}:
        return Verdict(False, ALREADY_LAUNCHED, candidate=candidate, detail=(
            "this worktree already holds a launch reservation"))
    if not configured:
        return Verdict(False, LAUNCHER_NOT_CONFIGURED, candidate=candidate,
                       detail=(
                           f"no agent command is declared in {CONFIG_PATH}; "
                           f"this repository refuses to guess one"))

    return Verdict(True, ACCEPTED, candidate=candidate,
                   detail=f"authorised by {authority.granted_by}")


def _is_tier3(space, area) -> bool:
    """Ask the coordinate space, tolerating a space that cannot answer."""
    for name in ("tier3", "is_tier3"):
        probe = getattr(space, name, None)
        if callable(probe):
            try:
                return bool(probe(area))
            except Exception:            # noqa: BLE001 - a space that cannot
                return False             # answer must not admit by raising
    return False


def _in_conflicts(candidate: R.Candidate, plan: R.ResumePlan) -> bool:
    target = _normal(candidate.path)
    for overlap in plan.internal_conflicts or ():
        if _normal(overlap.path) == target:
            return True
    return False


def requests(plan: R.ResumePlan, *,
             model_for: Optional[Callable[[R.Candidate], str]] = None,
             argv_for: Optional[Callable[[LaunchRequest], Tuple[str, ...]]] = None,
             env_for: Optional[Callable[[R.Candidate], Mapping[str, str]]] = None,
             ) -> Tuple[LaunchRequest, ...]:
    """What would be run for each selected candidate. PURE; starts nothing."""
    out = []
    for candidate in plan.selected:
        model = model_for(candidate) if model_for else ""
        request = LaunchRequest(
            candidate=candidate,
            cwd=candidate.path,
            model=model,
            layer=candidate.layer or HIER.DEFAULT_LAYER.key,
            brief=brief_for(candidate),
            env=dict(env_for(candidate)) if env_for else {},
        )
        if argv_for:
            request = LaunchRequest(
                candidate=request.candidate, cwd=request.cwd,
                model=request.model, layer=request.layer, brief=request.brief,
                argv=tuple(argv_for(request)), env=request.env)
        out.append(request)
    return tuple(out)


# ── the repository declares its own agent command ───────────────────────────
#: Environment variables a child is given when the repository names none. PATH
#: and the platform's own directories, and nothing else.
#:
#: `os.environ.copy()` is what the engine spawners do, and it is correct for an
#: engine that needs the owner's whole environment. It is wrong here: this child
#: is a coding agent, and the owner's environment is where the API keys, broker
#: credentials and account paths live. Anything beyond this list has to be named
#: in `[launch] env_passthrough`, so a secret reaching an agent is always
#: something somebody wrote down.
DEFAULT_ENV_PASSTHROUGH: Tuple[str, ...] = (
    "PATH", "SystemRoot", "windir", "COMSPEC", "PATHEXT",
    "USERPROFILE", "HOME", "TEMP", "TMP", "TMPDIR",
    "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL",
)

#: Substituted into each element of the declared command.
PLACEHOLDERS: Tuple[str, ...] = (
    "brief", "brief_file", "cwd", "model", "layer", "branch", "label",
)


@dataclass(frozen=True)
class LaunchConfig:
    """The command this repository says starts an agent, and where it said so."""

    command: Tuple[str, ...]
    env_passthrough: Tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH
    source: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.command)


def config_path(repo_root: Optional[str] = None) -> Optional[Path]:
    """The configuration this repository would use, or None."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None
    if not repo_root:
        return None
    candidate = Path(repo_root) / CONFIG_PATH
    return candidate if candidate.is_file() else None


def from_config(data: Mapping, *, source: str = "configuration") -> LaunchConfig:
    """Parse the `[launch]` table.

    The shape, which `.alelyon/fleet.toml` writes as TOML::

        [launch]
        command = ["claude", "--permission-mode", "plan", "-p", "{brief_file}"]
        env_passthrough = ["PATH", "USERPROFILE"]

    A malformed entry RAISES rather than being skipped, for the reason
    `worktree_areas.from_config` gives about its own table: a rule silently
    dropped is a behaviour that stops happening without anyone being told. Here
    the behaviour is "refuse to start an agent", so a dropped rule would be a
    launcher that quietly starts the wrong thing.
    """
    table = data.get("launch") or {}
    if not isinstance(table, Mapping):
        raise ValueError("[launch] must be a table")
    raw = table.get("command") or ()
    if isinstance(raw, str):
        raise ValueError(
            "[launch] command must be a LIST of arguments, not a string; a "
            "string would have to be split by a shell, and this does not use "
            "one")
    command = tuple(str(part) for part in raw)
    for part in command:
        _check_placeholders(part, source)
    passthrough = table.get("env_passthrough")
    if passthrough is None:
        names: Tuple[str, ...] = DEFAULT_ENV_PASSTHROUGH
    elif isinstance(passthrough, (list, tuple)):
        names = tuple(str(one) for one in passthrough)
    else:
        raise ValueError("[launch] env_passthrough must be a list of names")
    return LaunchConfig(command=command, env_passthrough=names, source=source)


def _check_placeholders(part: str, source: str) -> None:
    """A misspelled placeholder must not reach the command line verbatim."""
    depth = 0
    name = []
    for character in part:
        if character == "{":
            depth += 1
            name = []
            continue
        if character == "}" and depth:
            depth -= 1
            found = "".join(name)
            if found and found not in PLACEHOLDERS:
                raise ValueError(
                    f"{source}: {{{found}}} is not a placeholder; known ones "
                    f"are {', '.join(PLACEHOLDERS)}")
            continue
        if depth:
            name.append(character)


def read_config(repo_root: Optional[str] = None) -> Optional[LaunchConfig]:
    """What this repository declared, or None when it declared nothing.

    None rather than a default: a repository that has not said how to start an
    agent has not consented to one being started, and inventing a command would
    run whatever happened to be on PATH under that name.
    """
    path = config_path(repo_root)
    if path is None:
        return None
    import tomllib                      # stdlib since 3.11; read-only here
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    config = from_config(data, source=str(path))
    return config if config.configured else None


def brief_file(repo_root: str, worktree: str, brief: str) -> Path:
    """Write the brief where the child can read it, OUTSIDE the worktree.

    Outside deliberately. The target worktree holds uncommitted owner data and
    nothing here may write into it. It also solves a real limit: Windows caps a
    command line at about 32k and a brief naming a hundred touched paths can
    approach it, so `{brief_file}` is the shape that always works and `{brief}`
    is the convenience that sometimes does not.
    """
    directory = _reservation_dir() / "briefs"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.blake2b(
        f"{_normal(repo_root)}\x00{_normal(worktree)}".encode("utf-8"),
        digest_size=10).hexdigest()
    path = directory / f"{digest}.md"
    path.write_text(brief, encoding="utf-8")
    return path


def argv_from(config: LaunchConfig, request: LaunchRequest, *,
              brief_path: str = "") -> Tuple[str, ...]:
    """The declared command with this request's values substituted."""
    values = {
        "brief": request.brief,
        "brief_file": brief_path,
        "cwd": request.cwd,
        "model": request.model,
        "layer": request.layer,
        "branch": request.candidate.branch or "",
        "label": request.candidate.label or "",
    }
    out = []
    for part in config.command:
        for name, value in values.items():
            part = part.replace("{" + name + "}", str(value))
        out.append(part)
    return tuple(out)


def child_env(config: LaunchConfig, extra: Optional[Mapping[str, str]] = None
              ) -> dict:
    """The child's environment: an allowlist, never a copy."""
    out = {name: os.environ[name]
           for name in config.env_passthrough if name in os.environ}
    out.update(dict(extra or {}))
    return out


# ── reservations ────────────────────────────────────────────────────────────
class ReservationError(RuntimeError):
    """A launch slot for this worktree could not be claimed."""


def _reservation_dir() -> Path:
    return Path(PATHS.GLOBALS_DIR) / "worktree-launch"


def _reservation_path(repo_root: str, worktree: str) -> Path:
    digest = hashlib.blake2b(
        f"{_normal(repo_root)}\x00{_normal(worktree)}".encode("utf-8"),
        digest_size=10).hexdigest()
    return _reservation_dir() / f"{digest}.json"


def claim(repo_root: str, worktree: str, *, model: str = "") -> str:
    """Claim the launch slot for one worktree, atomically.

    `O_CREAT | O_EXCL`, the same discipline `engine_lifecycle` uses for an
    engine slot, keyed per worktree instead of per engine kind. It is a second
    implementation of that mechanism rather than a reuse of it, because the
    engine version is private and keyed on engine identity; that duplication is
    real and is recorded here rather than left for someone to discover.
    """
    path = _reservation_path(repo_root, worktree)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = hashlib.blake2b(os.urandom(16), digest_size=16).hexdigest()
    record = {
        "schema": LAUNCH_SCHEMA,
        "worktree": worktree,
        "token": token,
        "owner_pid": os.getpid(),
        "created_at_ts": time.time(),
        "model": model,
        "child_pid": None,
    }
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ReservationError(
            f"a launch is already reserved for {worktree}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(record, handle)
    return token


def release(repo_root: str, worktree: str) -> None:
    """Drop the slot. Safe to call when it was never claimed."""
    try:
        _reservation_path(repo_root, worktree).unlink()
    except FileNotFoundError:
        return


def mark_child(repo_root: str, worktree: str, pid: int) -> None:
    """Bind a claimed slot to the process it started."""
    path = _reservation_path(repo_root, worktree)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    record["child_pid"] = int(pid)
    record["child_marked_at_ts"] = time.time()
    try:
        path.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        return


def reserved(repo_root: str) -> Tuple[str, ...]:
    """Worktree paths currently holding a launch slot.

    These are RESERVATIONS, not running agents. A reservation says a launch was
    started from here and never released; the process may have finished, and
    nothing in this file can tell the difference.
    """
    out = []
    directory = _reservation_dir()
    if not directory.is_dir():
        return ()
    for entry in sorted(directory.glob("*.json")):
        try:
            record = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("schema") != LAUNCH_SCHEMA:
            continue
        worktree = str(record.get("worktree") or "")
        if worktree:
            out.append(worktree)
    return tuple(out)


# ── the real spawner ────────────────────────────────────────────────────────
Spawner = Callable[["LaunchRequest"], "LaunchHandle"]

# Windows process-creation flags, named here rather than imported so this module
# still imports on POSIX. Same three `engine_lifecycle` uses; the duplication is
# recorded rather than hidden, and the extraction that would remove it touches a
# Tier 3 module and belongs in its own change.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


def detached_spawner(config: LaunchConfig, *, repo_root: str,
                     popen=None, resolve=None) -> Spawner:
    """A spawner that starts the declared command, detached.

    Detached is the owner's decision of 2026-08-04: closing the window that
    started an agent must not kill work in progress. The consequence is stated
    rather than hidden -- these processes outlive this one, nothing here can
    stop them, and `reserved()` is the only trace they leave.

    `popen` and `resolve` are injected so the whole path can be tested without
    starting anything. They default to the real ones only when the caller does
    not say otherwise; `launch()` itself still refuses to default the spawner,
    which is the seam that matters.
    """
    import subprocess

    spawn_with = popen or subprocess.Popen
    resolver = resolve or _resolve_executable

    def _spawn(request: LaunchRequest) -> LaunchHandle:
        path = brief_file(repo_root, request.cwd, request.brief)
        argv = list(request.argv or argv_from(config, request,
                                              brief_path=str(path)))
        if not argv:
            raise ValueError(
                "the declared command produced no arguments; refusing to run "
                "an empty command line")
        argv[0] = resolver(argv[0])
        environment = child_env(config)
        keywords = dict(
            cwd=request.cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        if os.name == "nt":
            keywords["creationflags"] = (
                _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP
                | _CREATE_NO_WINDOW)
        else:
            keywords["start_new_session"] = True
        child = spawn_with(argv, **keywords)
        return LaunchHandle(
            pid=int(getattr(child, "pid", 0) or 0),
            started_at_ts=time.time(),
            argv=tuple(argv),
            detail=f"detached; brief at {path}")

    return _spawn


def _resolve_executable(name: str) -> str:
    """The command's absolute path where it can be found, else the bare name.

    `shutil.which` answers "is this on the PATH of the process asking", which
    for a GUI launched from a shortcut is not the machine. `toolpath` searches
    the documented install locations too, and falls back to the bare name so the
    operating system reports the failure rather than a different one invented
    here.
    """
    try:
        from alelyon.runtime.common import toolpath
    except ImportError:                  # pragma: no cover - toolpath is a peer
        return name
    return toolpath.argv(name)[0]


# ── the one effectful function ──────────────────────────────────────────────


def launch(plan: R.ResumePlan, *,
           spawn: Spawner,
           authority: Optional[Authority],
           model_for: Optional[Callable[[R.Candidate], str]] = None,
           argv_for: Optional[Callable[[LaunchRequest], Tuple[str, ...]]] = None,
           env_for: Optional[Callable[[R.Candidate], Mapping[str, str]]] = None,
           live_sessions: Sequence[str] = (),
           space=None,
           max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
           configured: bool = True,
           exists: Optional[Callable[[str], bool]] = None,
           now: Optional[float] = None) -> LaunchReport:
    """Admit each selected candidate, and spawn the ones that pass.

    `spawn` is keyword-only and has NO default, deliberately. Everywhere else in
    this repository an injected dependency falls back to a module-private real
    one; here that would mean `launch(plan)` spends money from a mistyped test.
    A missing spawner is a `TypeError` before anything is evaluated.

    Agents started here are DETACHED and outlive the caller. That is the owner's
    decision of 2026-08-04: closing a window must not kill work in progress. The
    reservation records what was started so a later session can see it.
    """
    moment = time.time() if now is None else float(now)
    prepared = requests(plan, model_for=model_for, argv_for=argv_for,
                        env_for=env_for)
    started: list = []
    refused: list = []
    in_flight = list(reserved(plan.repo_root))

    for request in prepared:
        verdict = admit(
            request.candidate, plan=plan, authority=authority,
            live_sessions=live_sessions, space=space, model=request.model,
            in_flight=in_flight, max_in_flight=max_in_flight,
            configured=configured, exists=exists, now=moment)
        if not verdict.accepted:
            refused.append(verdict)
            continue
        try:
            token = claim(plan.repo_root, request.cwd, model=request.model)
        except ReservationError as exc:
            refused.append(Verdict(False, ALREADY_LAUNCHED,
                                   candidate=request.candidate,
                                   detail=str(exc)))
            continue
        try:
            handle = spawn(request)
        except Exception:
            # The slot must not outlive the failure that stopped it being used.
            release(plan.repo_root, request.cwd)
            raise
        mark_child(plan.repo_root, request.cwd, handle.pid)
        in_flight.append(request.cwd)
        started.append((request, handle))
        del token

    return LaunchReport(at=int(moment), requested=prepared,
                        started=tuple(started), refused=tuple(refused))


__all__ = [
    "ACCEPTED", "ALREADY_LAUNCHED", "AREA_CLAIMED", "BELOW_FLOOR",
    "BOARD_MATTER", "CONFIG_ENV", "CONFIG_PATH", "CONTENDED",
    "DEFAULT_MAX_IN_FLIGHT", "DORMANCY_UNKNOWN", "FLEET_FULL",
    "LAUNCHER_NOT_CONFIGURED", "LAUNCH_LIMITS", "LAUNCH_SCHEMA", "LIVE_SESSION",
    "MISSING_WORKTREE", "MODEL_NOT_NAMED", "NOT_AUTHORISED_HERE",
    "NOT_DORMANT", "NOT_IN_PLAN", "NO_AUTHORITY", "PLAN_STALE",
    "PLAN_TTL_SECONDS", "REASONS", "TIER3_AREA",
    "Authority", "LaunchHandle", "LaunchReport", "LaunchRequest",
    "ReservationError", "Spawner", "Verdict",
    "DEFAULT_ENV_PASSTHROUGH", "PLACEHOLDERS", "LaunchConfig",
    "admit", "argv_from", "brief_file", "brief_for", "child_env", "claim",
    "config_path", "detached_spawner", "from_config", "launch", "mark_child",
    "read_config", "release", "requests", "reserved",
]
