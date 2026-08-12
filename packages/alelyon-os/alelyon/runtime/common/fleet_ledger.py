"""The career: what each model actually did at each layer, and the ratchet.

`fleet_hierarchy` says what a layer is worth. This says what a *model* is worth
at that layer, from runs that happened — and it is the half that makes the
hierarchy improve rather than merely exist.

The ratchet, stated exactly
---------------------------
One standing per `(layer, work_kind)`: the model currently held to be best for
that job. A challenger replaces it only through `propose()`, which refuses
unless **all** of these hold:

1. the challenger has at least `MIN_RUNS` completed runs at that coordinate —
   *completed*, which until 2026-08-06 this compared against the total instead,
   so runs that established nothing counted towards the bar;
2. those runs are **later than the incumbent's standing was set** — held out by
   time, so a challenger cannot win on the evidence that promoted the incumbent;
3. it beats the incumbent's score by at least `MARGIN_FRACTION` of the
   headroom still available above that incumbent (see `MARGIN_FRACTION`
   for why this is a fraction and not an absolute step);
4. it meets the layer's capability floor.

Everything else is refused **by name and with the number that failed**. Nothing
is ever deleted: a demotion appends, and the previous standing stays readable.

So the guarantee is precise, and it is smaller than "the fleet gets better":

    The standing at a coordinate never moves to a candidate that scored worse
    on the evidence available when it moved.

That is a real monotonicity and it is worth having. What it is **not** is a
claim that the score measures quality — see `SCORE_LIMITS`, which is printed
with every scorecard rather than kept in a docstring.

Why these signals and not better ones
-------------------------------------
Every input is derived from a record the agent did not author:

* **settled** — the workflow journal recorded a result for it. Harness-written.
* **cost** — output tokens, from the harness's own usage figures.
* **effort** — turns taken.
* **contested-after** — another session published a `defect-found` or
  `interface-changed` finding about a file this agent touched, *after* it
  settled. Ordering by timestamp is what makes this a signal rather than a
  coincidence, and it is the only negative-quality evidence available without a
  human in the loop.

There is deliberately no self-report. An agent's own account of how it did is
the one input that would be free to collect and worthless to trust, and
`CLAIMS.md` §2.3 forbids validating against the shape of what the writer
emitted.

The lifecycle is `docs/MODELS.md` §0, reused
--------------------------------------------
`OBSERVATION` → `PAPER` → `LIVE`, and demotion when a model misses its own bar.
That ladder is owner-authored doctrine for market models; applying it to agent
models is reuse rather than a second invented lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sqlite3
import time

from alelyon.runtime.common import sqlite_wal as _DB
from alelyon.runtime.common import fleet_hierarchy as H
from alelyon.runtime.common import fleet_outcomes as O
from alelyon.runtime.common import session_records as SR

#: How long a connection waits for a lock before giving up. Ten seconds was
#: already the connect timeout here; it is now also the explicit busy_timeout,
#: so the two halves of "how long may this block" are one number.
_CONNECT_TIMEOUT = 10.0

#: 2 added `branch`, `cwd` and `landed` to `runs`. Migrated in place rather than
#: versioned into a new layer space: the columns are additive and `landed`
#: defaults to the one value that scores exactly as v1 did, so a v1 ledger reads
#: back at v2 with every historical score unchanged. A test pins that.
#:
#: 3 adds the producer-owned, append-only ``run_changes`` feed.  The owner
#: writer backfills one ``recorded`` change for every older row before exposing
#: the version.  A read-only observer refuses pre-v3 databases and never runs
#: this migration itself.
LEDGER_SCHEMA_VERSION = 3

#: What an ABANDONED landing does to a run's score.
#:
#: It is a **penalty and never a bonus**, and that asymmetry is the load-bearing
#: part rather than the number. `standings.score` is a snapshot frozen under the
#: scoring rule in force when it was written, so a rule that could RAISE a score
#: would let a challenger unseat an incumbent by the rule change alone — beating
#: a number computed under the old rule with one computed under the new one,
#: which is not a comparison. Every landing outcome except ABANDONED multiplies
#: by exactly 1.0, so this release can only ever make it harder to move a
#: standing, never easier. A test asserts the direction.
#:
#: 0.5 sits between neutral and `contested`'s 0.4 because the two signals are
#: about equally weak and equally noisy, and both must bite hard enough to
#: matter against a MARGIN of 0.05.
ABANDONED_PENALTY = 0.5

#: Runs required at a coordinate before a model may be proposed for it. Small,
#: because the population is small — and stated as the weak number it is: five
#: runs is evidence a coin would clear one time in sixteen, so a promotion is a
#: working hypothesis, not a finding.
MIN_RUNS = 5

#: The most `Run.score` can express. The score is `0.75 + 0.25 * cost` with
#: `cost` in (0, 1], so the attainable band is [0.75, 1.0] — a fact the margin
#: below has to respect, because a margin that can exceed the ceiling is not a
#: bar, it is a lock.
MAX_SCORE = 1.0

#: How much better a challenger must score, as a FRACTION OF THE HEADROOM still
#: available above the incumbent — not as an absolute quantity.
#:
#: It was absolute (0.05) and that made the ratchet unable to ratchet. The score
#: is bounded at 1.0, so an incumbent above 0.95 required a challenger to clear
#: more than 1.0, which no run can reach; and since any run under ~12,877 output
#: tokens already scores above 0.95, the FIRST model to win a coordinate with an
#: ordinary-sized run held it permanently. `propose()` could never accept anyone
#: again, while the scorecard went on printing numbers that made it look like a
#: contest. Reported by session da438960 with a reproduction, confirmed by
#: `tests/runtime/test_fleet_ledger.py`.
#:
#: 0.2 is not a new tuning: at the bottom of the band the headroom is 0.25, so
#: 0.2 of it is exactly the 0.05 the absolute margin asked for. The behaviour at
#: mid-scale is unchanged and only the unsatisfiable region is removed. Near the
#: ceiling the bar shrinks in absolute terms and stays meaningful in relative
#: ones: beating an incumbent at 1,000 output tokens still requires about 800,
#: which is a real ~20% improvement rather than noise.
MARGIN_FRACTION = 0.2

#: Lifecycle states, from docs/MODELS.md section 0.
OBSERVATION = "OBSERVATION"   # running and being scored; holds nothing
PAPER = "PAPER"               # met the bar once; recorded, not yet standing
LIVE = "LIVE"                 # the standing at this coordinate
DEMOTED = "DEMOTED"           # held the standing and fell below its own bar
STATES = (OBSERVATION, PAPER, LIVE, DEMOTED)

#: Refusal reasons. A closed vocabulary because "why was my model not promoted"
#: must have an answer that is the same every time it is asked.
TOO_FEW_RUNS = "too-few-runs"
NOT_HELD_OUT = "not-held-out"
NO_MARGIN = "no-margin"
#: The incumbent sits at MAX_SCORE, so the signal has no resolution left to
#: distinguish a better challenger. Named rather than silently refused: "the
#: standing is unbeatable because the metric ran out" is a different fact from
#: "the challenger was worse", and a caller acting on the second would be wrong.
SCORE_CEILING = "score-ceiling"
BELOW_FLOOR = "below-capability-floor"
NOT_A_COORDINATE = "not-a-coordinate"
ACCEPTED = "accepted"

SCORE_LIMITS: tuple[str, ...] = (
    "The score measures COMPLETION and COST, not quality. An agent that "
    "settled quickly and cheaply on a wrong answer scores well, and nothing "
    "here would notice.",
    "contested-after is weak: it fires when somebody else published a finding "
    "about a file this agent touched afterwards, which catches a real defect "
    "and also catches two sessions working near each other.",
    "Whether the branch LANDED is the second negative signal, and it is a "
    "signal about the branch rather than about the agent - every agent that "
    "ran on one carries the same outcome. It is also silent by default: a run "
    "recorded before anyone merged anything is IN-FLIGHT, and only reconcile() "
    "reading the repository later can turn that into a result.",
    "A run is attributed to the model the harness recorded for its turns. An "
    "agent re-driven on a second model is attributed to whichever ran most of "
    "its turns, so a mixed run is scored as one model's work.",
    "MIN_RUNS is five. Five runs is a working hypothesis, not a finding - a "
    "coin clears that bar one time in sixteen.",
    "A run whose outcome no journal recorded is NOT SCORED, and is not counted "
    "towards MIN_RUNS either. It is still recorded and still shown, because "
    "how much of a model's work left no outcome is worth seeing - measured "
    "2026-08-06, 272 of 1439 runs, and every one of them QUIET rather than "
    "STOPPED. What it is not is a bad result: session_activity says a crashed "
    "agent and an agent whose workflow wrote no journal look identical, and a "
    "directly-spawned agent has no journal to be settled by at all.",
    "The ratchet guarantees the standing never moves to a candidate that "
    "scored worse on the evidence available. It cannot guarantee the standing "
    "is the best model, because a model nobody ran has no evidence at all.",
    "A career's pooled score mixes coordinates. Work kinds are not one scale, "
    "so a model that mostly ran mechanical jobs and one that mostly refereed "
    "designs are not comparable on it - only their per-coordinate cards are.",
    "Nothing here dispatches. A standing is a recommendation a caller may read "
    "before naming a model, and a session is free to ignore it.",
)

_DDL = (
    """CREATE TABLE IF NOT EXISTS meta (
           name TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    # An f-string for one reason: the `landed` default is the outcome
    # vocabulary's own UNKNOWN rather than a second copy of the word that could
    # drift from it and silently start scoring historical rows.
    f"""CREATE TABLE IF NOT EXISTS runs (
           run_id      TEXT PRIMARY KEY,
           at          INTEGER NOT NULL,
           layer       TEXT NOT NULL,
           work_kind   TEXT NOT NULL,
           model       TEXT NOT NULL,
           agent_id    TEXT NOT NULL,
           fleet_id    TEXT NOT NULL,
           session_id  TEXT NOT NULL,
           settled     INTEGER NOT NULL,
           turns       INTEGER NOT NULL,
           out_tokens  INTEGER NOT NULL,
           seconds     INTEGER,
           contested   INTEGER NOT NULL DEFAULT 0,
           space       INTEGER NOT NULL,
           branch      TEXT NOT NULL DEFAULT '',
           cwd         TEXT NOT NULL DEFAULT '',
           landed      TEXT NOT NULL DEFAULT '{O.UNKNOWN}')""",
    """CREATE INDEX IF NOT EXISTS runs_coord
           ON runs(layer, work_kind, model, at)""",
    # `seq` rather than `set_at` decides which standing is current, and the
    # difference is not academic: several standings can be written inside one
    # second, and ordering by a second-resolution clock then resolves "latest"
    # by whatever order SQLite happens to return. That put a 0.716 model over a
    # 0.943 one on this repository's own history. `set_at` still exists, and is
    # still what holds evidence out by time.
    """CREATE TABLE IF NOT EXISTS standings (
           seq         INTEGER PRIMARY KEY AUTOINCREMENT,
           layer       TEXT NOT NULL,
           work_kind   TEXT NOT NULL,
           model       TEXT NOT NULL,
           state       TEXT NOT NULL,
           score       REAL NOT NULL,
           runs        INTEGER NOT NULL,
           set_at      INTEGER NOT NULL,
           reason      TEXT NOT NULL,
           space       INTEGER NOT NULL)""",
    """CREATE INDEX IF NOT EXISTS standings_coord
           ON standings(layer, work_kind, seq)""",
)

RECORDED = "recorded"
RECONCILED = "reconciled"
CHANGE_KINDS = (RECORDED, RECONCILED)
CHANGE_FEED_SCHEMA = "alelyon.fleet-run-changes/v1"
MAX_CHANGE_PAGE = 1_000
MAX_CHANGE_TEXT_BYTES = 4_096

_CHANGE_COLUMNS = (
    "run_id, at, layer, work_kind, model, agent_id, fleet_id, session_id, "
    "settled, turns, out_tokens, seconds, contested, space, branch, cwd, landed"
)

_CHANGE_DDL = (
    f"""CREATE TABLE IF NOT EXISTS run_changes (
           seq         INTEGER PRIMARY KEY AUTOINCREMENT,
           change_kind TEXT NOT NULL CHECK(change_kind IN ('{RECORDED}','{RECONCILED}')),
           run_id      TEXT NOT NULL,
           at          INTEGER NOT NULL,
           layer       TEXT NOT NULL,
           work_kind   TEXT NOT NULL,
           model       TEXT NOT NULL,
           agent_id    TEXT NOT NULL,
           fleet_id    TEXT NOT NULL,
           session_id  TEXT NOT NULL,
           settled     INTEGER NOT NULL,
           turns       INTEGER NOT NULL,
           out_tokens  INTEGER NOT NULL,
           seconds     INTEGER,
           contested   INTEGER NOT NULL,
           space       INTEGER NOT NULL,
           branch      TEXT NOT NULL,
           cwd         TEXT NOT NULL,
           landed      TEXT NOT NULL)""",
    f"""CREATE UNIQUE INDEX IF NOT EXISTS run_changes_recorded_once
           ON run_changes(run_id) WHERE change_kind='{RECORDED}'""",
    f"""CREATE TRIGGER IF NOT EXISTS runs_change_after_insert
           AFTER INSERT ON runs
           BEGIN
             INSERT INTO run_changes(change_kind, {_CHANGE_COLUMNS})
             VALUES('{RECORDED}', NEW.run_id, NEW.at, NEW.layer, NEW.work_kind,
                    NEW.model, NEW.agent_id, NEW.fleet_id, NEW.session_id,
                    NEW.settled, NEW.turns, NEW.out_tokens, NEW.seconds,
                    NEW.contested, NEW.space, NEW.branch, NEW.cwd, NEW.landed);
           END""",
    f"""CREATE TRIGGER IF NOT EXISTS runs_change_after_landing
           AFTER UPDATE OF landed ON runs
           WHEN OLD.landed IS NOT NEW.landed
           BEGIN
             INSERT INTO run_changes(change_kind, {_CHANGE_COLUMNS})
             VALUES('{RECONCILED}', NEW.run_id, NEW.at, NEW.layer, NEW.work_kind,
                    NEW.model, NEW.agent_id, NEW.fleet_id, NEW.session_id,
                    NEW.settled, NEW.turns, NEW.out_tokens, NEW.seconds,
                    NEW.contested, NEW.space, NEW.branch, NEW.cwd, NEW.landed);
           END""",
)


@dataclass(frozen=True)
class Run:
    """One completed agent run, scored."""

    run_id: str
    at: int
    layer: str
    work_kind: str
    model: str
    agent_id: str
    fleet_id: str
    session_id: str
    settled: bool
    turns: int
    out_tokens: int
    seconds: int | None
    contested: bool = False
    #: Where the work was done, as the harness stamped it on the transcript.
    #: Stored so the row can be re-read against the repository later without
    #: going back to the transcripts — a record that carries its own inputs is
    #: one somebody else can check.
    branch: str = ""
    cwd: str = ""
    #: What became of that branch: `fleet_outcomes.LANDED` / `IN_FLIGHT` /
    #: `ABANDONED` / `UNKNOWN`. Provisional until it is one of the terminal two.
    landed: str = O.UNKNOWN

    @property
    def measured(self) -> bool:
        """Whether this run established anything at all.

        `settled` means the workflow journal recorded a result. Its negation is
        NOT "the work failed": `session_activity` has four statuses, and the two
        that reach here unsettled are STOPPED — the journal recorded `stopped`,
        `error` or `failed`, which is real negative evidence — and QUIET, which
        that module defines as *"no write for N minutes and no outcome in a
        journal; a crashed agent looks exactly like this"* and whose own LIMITS
        say is "not a report from the agent".

        Measured 2026-08-06 over 1439 recorded runs: 1167 SETTLED, 272 QUIET,
        and **STOPPED zero times**. So in practice the unsettled rows are
        entirely the absence of a journal entry rather than evidence of a
        failure. By fleet kind the split is workflow 1167/197 and direct 0/75 —
        `_read_journal` reads a WORKFLOW journal, so a directly-spawned agent
        has none, can never be SETTLED, and would otherwise be scored on which
        mechanism spawned it rather than on anything it did.
        """
        return self.settled

    @property
    def score(self) -> float | None:
        """Completion, penalised by cost, by a later defect finding, and by
        the branch having been abandoned. **`None` where nothing was measured.**

        In [0, 1] when it is a number at all. Deliberately simple and
        deliberately printed beside its limits: a compound score with tuned
        weights would look like a measurement of quality, and this is a
        measurement of whether the job finished, what it cost to finish, and
        whether anyone took it up.

        **Landing only ever subtracts.** LANDED, IN-FLIGHT and UNKNOWN all
        multiply by exactly 1.0, so a run recorded before this term existed
        scores identically under it — and, more importantly, no challenger can
        beat an incumbent whose stored score was frozen under the older rule
        merely because the rule got more generous. See `ABANDONED_PENALTY`.

        **Completion now obeys that same rule, and did not.** An unmeasured run
        returned `0.0` — the worst attainable value — and that number was then
        averaged into `Scorecard.mean_score`, which is what the gate compares.
        The test covering it said the quiet part out loud: *"an unsettled run
        measured nothing"*, and then asserted the worst possible measurement of
        it. Absence of evidence was the only thing the penalty ever fired on
        (see `measured`), and it decided coordinates: at `executive/design-
        review`, the busiest at 733 runs, excluding the unmeasured rows from the
        mean puts a different model on top.

        `None` rather than a neutral 1.0, and rather than a documented 0.0,
        because this is the producer. A neutral number would still be a number
        an aggregator could average; `None` cannot be summed by accident, so a
        caller has to decide what an absence means instead of inheriting an
        answer. There are three callers and each one now says.
        """
        if not self.measured:
            return None
        # Cost term: 1.0 at zero tokens, decaying with a 40k half-life. A
        # constant rather than a fitted parameter, because fitting it against
        # the same runs it scores would be circular.
        cost = 0.5 ** (max(0, self.out_tokens) / 40_000.0)
        value = 0.75 + 0.25 * cost
        landing = ABANDONED_PENALTY if self.landed == O.ABANDONED else 1.0
        return round(value * (0.4 if self.contested else 1.0) * landing, 6)


class ChangeFeedState(str, Enum):
    AVAILABLE = "available"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class FleetRunChange:
    """One immutable run snapshot at the producer change sequence."""

    seq: int
    change_kind: str
    space: int
    run: Run = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.seq) is not int or self.seq <= 0:
            raise ValueError("seq must be a positive exact int")
        if type(self.change_kind) is not str or self.change_kind not in CHANGE_KINDS:
            raise ValueError("change_kind is not supported")
        if type(self.space) is not int or self.space < 0:
            raise ValueError("space must be a non-negative exact int")
        if type(self.run) is not Run:
            raise TypeError("run must be an exact Run")


@dataclass(frozen=True, slots=True)
class FleetChangePage:
    """A bounded page on one stable feed high-water mark.

    ``next_after_seq`` is exclusive.  Reusing ``as_of_seq`` while following
    pages gives a finite snapshot; starting again after the returned cursor
    observes changes committed later without repeating an earlier sequence.
    """

    schema: str
    after_seq: int
    as_of_seq: int
    changes: tuple[FleetRunChange, ...] = field(repr=False)
    next_after_seq: int
    has_more: bool

    def __post_init__(self) -> None:
        if self.schema != CHANGE_FEED_SCHEMA:
            raise ValueError("schema is not supported")
        for name in ("after_seq", "as_of_seq", "next_after_seq"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative exact int")
        if type(self.changes) is not tuple or any(
            type(change) is not FleetRunChange for change in self.changes
        ):
            raise TypeError("changes must be exact FleetRunChange values")
        if type(self.has_more) is not bool:
            raise TypeError("has_more must be an exact bool")
        if self.changes:
            sequences = tuple(change.seq for change in self.changes)
            if sequences != tuple(sorted(sequences)) or len(set(sequences)) != len(sequences):
                raise ValueError("change sequences must be unique and ordered")
            if sequences[0] <= self.after_seq or sequences[-1] > self.as_of_seq:
                raise ValueError("change sequences are outside the page bounds")
            if self.next_after_seq != sequences[-1]:
                raise ValueError("next_after_seq must name the last returned change")
        elif self.next_after_seq != self.after_seq:
            raise ValueError("an empty page must preserve its input cursor")


@dataclass(frozen=True, slots=True)
class FleetChangeReport:
    state: ChangeFeedState
    page: FleetChangePage | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.state) is not ChangeFeedState:
            raise TypeError("state must be an exact ChangeFeedState")
        if (self.state is ChangeFeedState.AVAILABLE) != (self.page is not None):
            raise ValueError("only an available report may carry a page")


@dataclass(frozen=True)
class Scorecard:
    """What one model has done at one coordinate."""

    layer: str
    work_kind: str
    model: str
    runs: int
    settled: int
    contested: int
    mean_score: float
    mean_tokens: float
    last_at: int
    #: How many of those runs were done on a branch that reached the mainline,
    #: and how many on one that did not. They do not sum to `runs`: the rest are
    #: IN-FLIGHT or UNKNOWN, which are the absence of evidence and are reported
    #: as their own number rather than folded into either side.
    landed: int = 0
    abandoned: int = 0

    @property
    def measured(self) -> bool:
        """Whether `mean_score` is a score at all.

        False means every run here was unmeasured (`Run.measured`), so the mean
        is over nothing and reads 0.0 for want of a number rather than because
        anything scored badly. A surface that prints `mean_score` without asking
        this prints the worst value on the scale as a finding.
        """
        return self.settled > 0

    @property
    def undecided(self) -> int:
        """Runs whose branch has told us nothing yet. Named, never implied."""
        return max(0, self.runs - self.landed - self.abandoned)

    @property
    def completion(self) -> float:
        return self.settled / self.runs if self.runs else 0.0


@dataclass(frozen=True)
class Career:
    """Everything one model has done across the whole layer space.

    A `Scorecard` answers "how did this model do at this coordinate"; a career
    answers "what has this model done at all", which is the question a reader
    asks first and which no reader could ask before. It is an aggregate over
    coordinates and therefore weaker than its parts: `mean_score` pools runs
    from work of different kinds, so a model that ran mostly cheap mechanical
    jobs is not comparable on it to one that ran mostly design reviews. The
    per-coordinate cards are carried alongside precisely so the pooled number
    never has to be the only thing on screen.
    """

    model: str
    runs: int
    settled: int
    contested: int
    mean_score: float
    mean_tokens: float
    last_at: int
    landed: int
    abandoned: int
    #: Every coordinate this model has a run at, best-scoring first.
    cards: tuple[Scorecard, ...] = ()
    #: Coordinates it currently holds, as `(layer, work_kind)`.
    holds: tuple[tuple[str, str], ...] = ()
    #: What its NAME declares it is, and what the ledger has MEASURED it into.
    #: Kept apart because the first is a dated convention and the second is a
    #: record of runs — collapsing them would hide which one is speaking.
    declared_class: str = ""
    measured_class: str | None = None

    @property
    def measured(self) -> bool:
        """Whether `mean_score` is a score at all. See `Scorecard.measured`.

        `<synthetic>` is why this is not hypothetical: on the live ledger it is
        94 runs of which 0 measured anything, so its pooled mean is over nothing
        and drew as 0.000 — the worst value on the scale, for work whose only
        established property is that no journal recorded an outcome for it.
        """
        return self.settled > 0

    @property
    def undecided(self) -> int:
        """Runs whose branch has told us nothing yet. Named, never implied."""
        return max(0, self.runs - self.landed - self.abandoned)

    @property
    def completion(self) -> float:
        return self.settled / self.runs if self.runs else 0.0

    @property
    def layers(self) -> tuple[str, ...]:
        """Layers it currently holds a standing at, highest rank first."""
        ranked = sorted({layer for layer, _kind in self.holds},
                        key=lambda key: getattr(H.layer(key), "rank", 99))
        return tuple(ranked)


def _run_from_row(row: sqlite3.Row) -> Run:
    """Materialise one stored run without duplicating the schema mapping."""
    return Run(
        run_id=row["run_id"], at=int(row["at"]), layer=row["layer"],
        work_kind=row["work_kind"], model=row["model"],
        agent_id=row["agent_id"], fleet_id=row["fleet_id"],
        session_id=row["session_id"], settled=bool(row["settled"]),
        turns=int(row["turns"]), out_tokens=int(row["out_tokens"]),
        seconds=None if row["seconds"] is None else int(row["seconds"]),
        contested=bool(row["contested"]), branch=row["branch"] or "",
        cwd=row["cwd"] or "", landed=row["landed"] or O.UNKNOWN)


def _scorecard_from_runs(layer: str, work_kind: str, model: str,
                         runs: tuple[Run, ...]) -> Scorecard | None:
    """Aggregate an already-bounded run population at one coordinate."""
    selected = tuple(run for run in runs
                     if run.layer == layer and run.work_kind == work_kind
                     and run.model == model)
    if not selected:
        return None
    scores = tuple(score for score in (run.score for run in selected)
                   if score is not None)
    return Scorecard(
        layer=layer, work_kind=work_kind, model=model,
        runs=len(selected), settled=sum(run.settled for run in selected),
        contested=sum(run.contested for run in selected),
        mean_score=round(sum(scores) / len(scores), 6) if scores else 0.0,
        mean_tokens=round(
            sum(run.out_tokens for run in selected) / len(selected), 1),
        last_at=max(run.at for run in selected),
        landed=sum(run.landed == O.LANDED for run in selected),
        abandoned=sum(run.landed == O.ABANDONED for run in selected))


@dataclass(frozen=True)
class Standing:
    """The model currently held to be best for one coordinate."""

    layer: str
    work_kind: str
    model: str
    state: str
    score: float
    runs: int
    set_at: int
    reason: str


@dataclass(frozen=True)
class Verdict:
    """The gate's answer. `accepted` is one value of `reason`, never a bare bool."""

    accepted: bool
    reason: str
    detail: str
    challenger: Scorecard | None = None
    incumbent: Standing | None = None

    def __str__(self) -> str:
        return f"{'ACCEPTED' if self.accepted else 'REFUSED'} — {self.detail}"


def pooled_text(record) -> str:
    """A `Scorecard` or `Career` mean, or UNMEASURED where it is over nothing.

    Shared rather than written per surface for the reason the number exists at
    all: `mean_score` reads 0.0 when nothing under it measured anything, and 0.0
    is the BOTTOM of the scale. Four surfaces print it — two panels, the ledger
    report and the careers command — and a fifth copy of `if measured` is how
    one of them quietly starts printing the worst value on the scale as a
    finding again.
    """
    return f"{record.mean_score:.3f}" if record.measured else "UNMEASURED"


class LedgerAbsent(FileNotFoundError):
    """No ledger at this path, and a reader will not make one to find out.

    A refusal rather than an empty result, deliberately. `FleetLedger(...)`
    creates its file, so a reader that opened one manufactures an empty ledger
    and then truthfully reports "nothing has been measured" — which reads as a
    finding about models and is a description of the file it made a moment
    earlier. Three call sites had each grown their own `if not path.exists()`
    guard around that; this is the one refusal they now share.
    """


class LedgerSchemaUnsupported(RuntimeError):
    """The ledger was written by a NEWER build than this one.

    Named, because the alternative is what this replaces: an unconditional
    `UPDATE meta SET value=?` walked the marker BACKWARDS to whatever version
    the opening process happened to be, while leaving the newer build's tables
    in place. The newer reader then saw a marker it could not reconcile with the
    schema in front of it and refused for the rest of that ledger's life. A
    downgrade is not a migration and must not happen silently.
    """


class LedgerUnreadable(RuntimeError):
    """The file exists and SQLite will not read it as this ledger."""


def default_database() -> Path:
    """The ledger, anchored on the REPOSITORY rather than on this worktree.

    `GLOBALS_DIR` is "the directory holding `pyproject.toml`", which a linked
    worktree satisfies too — 287 of them on this workstation, each of which would
    keep a private career record of a fleet whose whole point is to be shared.
    `paths.fleet_state_dir()` keys on `git rev-parse --git-common-dir`, the one
    directory every worktree of a repository reports identically, and degrades to
    `GLOBALS_DIR` where there is no git (an installed wheel, which was never the
    broken case).

    In the primary checkout, under `$ALELYON_HOME`, and in a packaged install the
    answer is byte-identical to what it was. Only a linked worktree moves, and it
    moves onto the file the primary checkout was already using.
    """
    from alelyon.runtime.common import paths
    return Path(paths.fleet_state_dir()) / "fleet_ledger.db"


_CHANGE_TABLE_CONTRACT = (
    ("seq", "INTEGER", 0, 1),
    ("change_kind", "TEXT", 1, 0),
    ("run_id", "TEXT", 1, 0),
    ("at", "INTEGER", 1, 0),
    ("layer", "TEXT", 1, 0),
    ("work_kind", "TEXT", 1, 0),
    ("model", "TEXT", 1, 0),
    ("agent_id", "TEXT", 1, 0),
    ("fleet_id", "TEXT", 1, 0),
    ("session_id", "TEXT", 1, 0),
    ("settled", "INTEGER", 1, 0),
    ("turns", "INTEGER", 1, 0),
    ("out_tokens", "INTEGER", 1, 0),
    ("seconds", "INTEGER", 0, 0),
    ("contested", "INTEGER", 1, 0),
    ("space", "INTEGER", 1, 0),
    ("branch", "TEXT", 1, 0),
    ("cwd", "TEXT", 1, 0),
    ("landed", "TEXT", 1, 0),
)


def _feed_text(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError("invalid structural text")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError:
        raise ValueError("invalid structural text") from None
    if len(encoded) > MAX_CHANGE_TEXT_BYTES:
        raise ValueError("invalid structural text")
    return value


def _feed_optional_text(value: object) -> str:
    if value == "" and type(value) is str:
        return ""
    return _feed_text(value)


def _feed_integer(value: object, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if type(value) is not int or value < 0 or value > (1 << 63) - 1:
        raise ValueError("invalid structural integer")
    return value


def _change_from_row(row: sqlite3.Row) -> FleetRunChange:
    settled = row["settled"]
    contested = row["contested"]
    if type(settled) is not int or settled not in (0, 1):
        raise ValueError("invalid structural state")
    if type(contested) is not int or contested not in (0, 1):
        raise ValueError("invalid structural state")
    landed = _feed_text(row["landed"])
    if landed not in O.OUTCOMES:
        raise ValueError("invalid structural outcome")
    run = Run(
        run_id=_feed_text(row["run_id"]),
        at=_feed_integer(row["at"]),
        layer=_feed_text(row["layer"]),
        work_kind=_feed_text(row["work_kind"]),
        model=_feed_text(row["model"]),
        agent_id=_feed_text(row["agent_id"]),
        fleet_id=_feed_text(row["fleet_id"]),
        session_id=_feed_text(row["session_id"]),
        settled=bool(settled),
        turns=_feed_integer(row["turns"]),
        out_tokens=_feed_integer(row["out_tokens"]),
        seconds=_feed_integer(row["seconds"], optional=True),
        contested=bool(contested),
        branch=_feed_optional_text(row["branch"]),
        cwd=_feed_optional_text(row["cwd"]),
        landed=landed,
    )
    return FleetRunChange(
        seq=_feed_integer(row["seq"]),
        change_kind=_feed_text(row["change_kind"]),
        space=_feed_integer(row["space"]),
        run=run,
    )


def read_change_feed(
    database: str | Path,
    *,
    after_seq: int = 0,
    as_of_seq: int | None = None,
    limit: int = 256,
) -> FleetChangeReport:
    """Read a stable page without creating, migrating, or repairing a ledger.

    A pre-v3 database is ``UNSUPPORTED`` even when its legacy ``runs`` table is
    readable.  Only constructing :class:`FleetLedger` owns the migration.  The
    sequence, rather than ``Run.at``, is the cursor so a late insert with an
    older run timestamp cannot fall behind the reader permanently.
    """
    if type(after_seq) is not int or after_seq < 0:
        raise ValueError("after_seq must be a non-negative exact int")
    if as_of_seq is not None and (
        type(as_of_seq) is not int or as_of_seq < after_seq
    ):
        raise ValueError("as_of_seq must be an exact int at or after the cursor")
    if type(limit) is not int or not 1 <= limit <= MAX_CHANGE_PAGE:
        raise ValueError("limit is outside the supported range")
    path = Path(database)
    try:
        if not path.is_file():
            return FleetChangeReport(ChangeFeedState.MISSING)
        resolved = path.resolve(strict=True)
        uri = f"{resolved.as_uri()}?mode=ro&cache=private"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
    except (OSError, sqlite3.Error, ValueError):
        return FleetChangeReport(ChangeFeedState.INVALID)
    try:
        conn.execute("PRAGMA query_only=ON")
        meta = conn.execute(
            "SELECT value FROM meta WHERE name='schema'"
        ).fetchone()
        if meta is None or type(meta["value"]) is not str:
            return FleetChangeReport(ChangeFeedState.INVALID)
        if meta["value"] != str(LEDGER_SCHEMA_VERSION):
            return FleetChangeReport(ChangeFeedState.UNSUPPORTED)
        contract = tuple(
            (row["name"], str(row["type"]).upper(), row["notnull"], row["pk"])
            for row in conn.execute("PRAGMA table_info(run_changes)")
        )
        if contract != _CHANGE_TABLE_CONTRACT:
            return FleetChangeReport(ChangeFeedState.INVALID)
        maximum_row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS maximum FROM run_changes"
        ).fetchone()
        maximum = _feed_integer(maximum_row["maximum"])
        boundary = maximum if as_of_seq is None else as_of_seq
        if boundary > maximum:
            return FleetChangeReport(ChangeFeedState.INVALID)
        rows = conn.execute(
            "SELECT seq, change_kind, " + _CHANGE_COLUMNS
            + " FROM run_changes WHERE seq>? AND seq<=? ORDER BY seq LIMIT ?",
            (after_seq, boundary, limit + 1),
        ).fetchall()
        has_more = len(rows) > limit
        changes = tuple(_change_from_row(row) for row in rows[:limit])
        next_after = changes[-1].seq if changes else after_seq
        page = FleetChangePage(
            schema=CHANGE_FEED_SCHEMA,
            after_seq=after_seq,
            as_of_seq=boundary,
            changes=changes,
            next_after_seq=next_after,
            has_more=has_more,
        )
    except (sqlite3.Error, TypeError, ValueError, OverflowError):
        return FleetChangeReport(ChangeFeedState.INVALID)
    finally:
        conn.close()
    return FleetChangeReport(ChangeFeedState.AVAILABLE, page)


class FleetLedger:
    """Durable owner writer for what every model did at every layer.

    Construction creates and migrates.  Read-only observation must use
    :func:`read_change_feed`, which never constructs this class.
    """

    def __init__(self, database: str | Path | None = None) -> None:
        """Open, CREATING the ledger when it is not there.

        This is the create path and is spelled `create()` when the intent
        matters. A caller that is only reading must use `open_existing()`, which
        refuses instead — see `LedgerAbsent`.
        """
        self.database = Path(database or default_database())
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            self._guard_schema(conn)
            for statement in _DDL:
                conn.execute(statement)
            self._migrate(conn)
            self._install_change_feed(conn)
            conn.execute(
                "INSERT OR IGNORE INTO meta(name, value) VALUES('schema', ?)",
                (str(LEDGER_SCHEMA_VERSION),))
            # A RATCHET, not an assignment. `CAST(value AS INTEGER)` is how
            # SQLite compares a marker stored as TEXT; the row is only ever
            # raised. `_guard_schema` has already refused anything higher, so
            # this clause is the second of two guards rather than the only one:
            # the WHERE holds even under a concurrent writer that raised the
            # version between the read and this statement.
            conn.execute(
                "UPDATE meta SET value=? "
                "WHERE name='schema' AND CAST(value AS INTEGER) < ?",
                (str(LEDGER_SCHEMA_VERSION), LEDGER_SCHEMA_VERSION))

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def create(cls, database: str | Path | None = None) -> "FleetLedger":
        """Open the ledger, creating it when absent. The explicit spelling."""
        return cls(database)

    @classmethod
    def open_existing(cls, database: str | Path | None = None) -> "FleetLedger":
        """Open an EXISTING ledger. Refuses rather than creating one.

        Raises `LedgerAbsent` when there is no file, `LedgerSchemaUnsupported`
        when a newer build wrote it, and `LedgerUnreadable` when SQLite will not
        read it. Each is a named refusal and none of them is an empty ledger:
        "nothing has been measured here" and "this could not be read" are
        different facts and a reader must not print the first for the second.
        """
        path = Path(database or default_database())
        if not path.exists():
            raise LedgerAbsent(
                f"no fleet ledger at {path}; nothing has been recorded here, "
                f"which is not the same as nothing having happened, and reading "
                f"will not create one to find out")
        try:
            return cls(path)
        except (LedgerAbsent, LedgerSchemaUnsupported):
            raise
        except sqlite3.DatabaseError as exc:
            raise LedgerUnreadable(
                f"the fleet ledger at {path} could not be read "
                f"({type(exc).__name__}: {exc})") from exc

    def _guard_schema(self, conn: sqlite3.Connection) -> int | None:
        """Refuse BY NAME on a ledger a newer build wrote. Returns its version.

        Read before any DDL runs, so a refusal leaves the file untouched.
        `None` means there is no marker yet — a fresh file, or one written
        before `meta` existed — and that is not an error.
        """
        try:
            row = conn.execute(
                "SELECT value FROM meta WHERE name='schema'").fetchone()
        except sqlite3.OperationalError:
            return None                      # no `meta` table yet
        if row is None:
            return None
        raw = row[0] if not isinstance(row, sqlite3.Row) else row["value"]
        try:
            found = int(str(raw).strip())
        except (TypeError, ValueError) as exc:
            raise LedgerSchemaUnsupported(
                f"the fleet ledger at {self.database} carries a schema marker "
                f"{raw!r} that is not a version number; refusing to overwrite "
                f"it") from exc
        if found > LEDGER_SCHEMA_VERSION:
            raise LedgerSchemaUnsupported(
                f"the fleet ledger at {self.database} is schema v{found}; this "
                f"build understands v{LEDGER_SCHEMA_VERSION}. Refusing: writing "
                f"v{LEDGER_SCHEMA_VERSION} over it would leave the newer "
                f"build's tables in place under an older marker, and the newer "
                f"reader would then refuse the ledger permanently")
        return found

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> tuple[str, ...]:
        """Bring an older `runs` table up to the current columns.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a ledger written at schema v1 keeps v1's columns forever
        unless something adds them — and every read here would then fail on a
        missing column rather than degrade. Adding them is safe in a way a
        rewrite would not be: each carries a DEFAULT that reproduces v1's
        behaviour exactly, so migrating rescores nothing.

        Idempotent, and returns what it actually did so a caller can report it.
        """
        have = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        added = []
        for column, ddl in (
            ("branch", "ALTER TABLE runs ADD COLUMN branch TEXT NOT NULL "
                       "DEFAULT ''"),
            ("cwd", "ALTER TABLE runs ADD COLUMN cwd TEXT NOT NULL DEFAULT ''"),
            ("landed", f"ALTER TABLE runs ADD COLUMN landed TEXT NOT NULL "
                       f"DEFAULT '{O.UNKNOWN}'"),
        ):
            if column not in have:
                conn.execute(ddl)
                added.append(column)
        return tuple(added)

    @staticmethod
    def _install_change_feed(conn: sqlite3.Connection) -> None:
        """Install v3 only inside the owner writer and backfill older runs.

        The insert trigger is installed before this method returns, in the same
        transaction as the schema version update.  Existing runs receive one
        deterministic ``recorded`` change in their SQLite insertion order.
        Later inserts and landing reconciliations are then captured by triggers
        in the transaction that owns the source mutation.
        """
        for statement in _CHANGE_DDL:
            conn.execute(statement)
        conn.execute(
            f"""INSERT OR IGNORE INTO run_changes(change_kind, {_CHANGE_COLUMNS})
                SELECT ?, {_CHANGE_COLUMNS} FROM runs ORDER BY rowid""",
            (RECORDED,),
        )

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        """One connection. `readonly=True` for anything that only reads.

        The read connection is opened `mode=ro` with `query_only=ON`, so "a read
        does not write to the ledger" is enforced by SQLite rather than asserted
        by this module. It is not decoration: under WAL an ordinary read-write
        connection CHECKPOINTS on close, which moves pages out of the `-wal` and
        physically rewrites the main file. Nothing logical changes, but the
        bytes do, and `test_it_writes_nothing_to_a_ledger_it_reads` measures the
        bytes. Rather than loosen that test to accommodate WAL, the reads were
        made incapable of the write it was watching for.

        A read-only connection also cannot create the file, which is the same
        guarantee `open_existing()` makes at the other end.
        """
        if readonly:
            uri = Path(self.database).resolve(strict=False).as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=_CONNECT_TIMEOUT)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(_CONNECT_TIMEOUT * 1000)}")
            self._register_functions(conn)
            return conn
        conn = sqlite3.connect(str(self.database), timeout=_CONNECT_TIMEOUT)
        conn.row_factory = sqlite3.Row
        # WAL, because under the rollback journal this used to run on, a writer
        # takes an EXCLUSIVE whole-database lock and every reader blocks behind
        # it — including the panels that poll this file. `db.try_set_wal` owns
        # the one subtlety (`PRAGMA journal_mode=WAL` does not honour
        # busy_timeout, so a cold concurrent open needs a short retry) and
        # returns False rather than refusing where the filesystem cannot do WAL.
        _DB.try_set_wal(conn)
        # busy_timeout as well as the connect timeout: they are not the same
        # budget. `sqlite3.connect(timeout=...)` sets the busy handler for this
        # connection, but a PRAGMA makes it explicit and survives anything that
        # resets the handler.
        conn.execute(f"PRAGMA busy_timeout = {int(_CONNECT_TIMEOUT * 1000)}")
        self._register_functions(conn)
        return conn

    @staticmethod
    def _register_functions(conn: sqlite3.Connection) -> None:
        # Repository placement is evaluated inside SQLite so an unrelated
        # machine-ledger row never crosses the query boundary and reaches
        # ``_run_from_row``.  A Python function is used instead of spelling
        # path rules in SQL: ``same_directory`` is the one canonical rule for
        # drive-letter case, separators, extended Windows paths and resolution.
        conn.create_function(
            "alelyon_same_directory", 2,
            lambda left, right: int(SR.same_directory(
                str(left or ""), str(right or ""))),
            deterministic=True)

    @staticmethod
    def _scope_predicate(
            cwds: tuple[str, ...],
            cwd_inceptions: tuple[tuple[str, int], ...]
            ) -> tuple[str, tuple[object, ...]]:
        """A parameterised exact-root/incarnation predicate for ``runs``.

        The predicate deliberately calls the canonical path comparator from
        SQLite.  Fetching every row and applying the same comparator later
        gets the final aggregate right but still materialises another local
        repository's model, branch and session fields in this process.
        """
        predicates: list[str] = []
        values: list[object] = []
        for accepted in cwds:
            root = str(accepted or "")
            if not root:
                continue
            floors = tuple(
                int(inception) for candidate, inception in cwd_inceptions
                if SR.same_directory(root, candidate))
            clause = "alelyon_same_directory(cwd, ?) = 1"
            values.append(root)
            if floors:
                clause += " AND at >= ?"
                values.append(max(floors))
            predicates.append(f"({clause})")
        if not predicates:
            return "0", ()
        return f"({' OR '.join(predicates)})", tuple(values)

    # ── recording ────────────────────────────────────────────────────────────
    def record(self, run: Run) -> bool:
        """Append one run. Returns False when it was already recorded.

        Idempotent on `run_id`, because the activity reader is polled and will
        offer the same finished agent on every pass. A ledger that counted a run
        once per poll would make a long-lived agent look like a hundred.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO runs
                   (run_id, at, layer, work_kind, model, agent_id, fleet_id,
                    session_id, settled, turns, out_tokens, seconds, contested,
                    space, branch, cwd, landed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run.run_id, run.at, run.layer, run.work_kind, run.model,
                 run.agent_id, run.fleet_id, run.session_id, int(run.settled),
                 run.turns, run.out_tokens, run.seconds, int(run.contested),
                 H.LAYER_SPACE_VERSION, run.branch, run.cwd,
                 run.landed or O.UNKNOWN))
            return cursor.rowcount > 0

    def record_all(self, runs) -> int:
        return sum(1 for run in runs if self.record(run))

    def reconcile(self, index, *, now: int | None = None) -> dict[str, int]:
        """Re-read the runs whose landing was not yet decided. Returns a tally.

        A run recorded the hour it finished has a branch nobody has merged yet,
        so its outcome is IN-FLIGHT — and if that were the last word, the only
        label that carries a penalty would never once be applied. The outcome
        has to be looked at again later, which is the whole reason `branch` and
        `cwd` are stored on the row: this needs no transcript to re-derive them.

        **Provisional to terminal only.** LANDED and ABANDONED are never
        overwritten, and the guard is in the UPDATE's own WHERE clause rather
        than in a Python branch above it, so a second caller reconciling
        concurrently cannot walk a decided row backwards either. What this is
        not is an append-only history of the outcome: the row carries the
        current answer, and the previous provisional non-answer is not kept,
        because "we did not know yet" is not a finding worth a row of its own.
        """
        moment = int(now if now is not None else time.time())
        tally: dict[str, int] = {"read": 0, "unchanged": 0}
        with self._connect() as conn:
            pending = conn.execute(
                "SELECT run_id, at, branch, cwd, landed FROM runs "
                f"WHERE landed IN ({','.join('?' * len(O.PROVISIONAL))})",
                O.PROVISIONAL).fetchall()
            for row in pending:
                tally["read"] += 1
                landing = index.of(row["branch"] or "", settled_at=row["at"],
                                   now=moment, cwd=row["cwd"] or "")
                if landing.outcome == (row["landed"] or O.UNKNOWN):
                    tally["unchanged"] += 1
                    continue
                cursor = conn.execute(
                    "UPDATE runs SET landed=? WHERE run_id=? AND landed IN "
                    f"({','.join('?' * len(O.PROVISIONAL))})",
                    (landing.outcome, row["run_id"], *O.PROVISIONAL))
                if cursor.rowcount:
                    tally[landing.outcome] = tally.get(landing.outcome, 0) + 1
        return tally

    # ── reading ──────────────────────────────────────────────────────────────
    def scorecard(self, layer: str, work_kind: str, model: str, *,
                  since: int = 0) -> Scorecard | None:
        with self._connect(readonly=True) as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS runs, SUM(settled) AS settled,
                          SUM(contested) AS contested, AVG(out_tokens) AS tokens,
                          MAX(at) AS last_at
                     FROM runs
                    WHERE layer=? AND work_kind=? AND model=? AND at>? AND space=?""",
                (layer, work_kind, model, since, H.LAYER_SPACE_VERSION)).fetchone()
            if not row or not row["runs"]:
                return None
            scored = conn.execute(
                """SELECT settled, out_tokens, contested, landed FROM runs
                    WHERE layer=? AND work_kind=? AND model=? AND at>? AND space=?""",
                (layer, work_kind, model, since, H.LAYER_SPACE_VERSION)).fetchall()
        # The mean is over the runs that MEASURED something, never over all of
        # them. Dividing by `runs` while the unmeasured rows contributed 0.0 was
        # a double count of the same absence: it entered the numerator as the
        # worst possible score and the denominator as though it were evidence.
        total = 0.0
        measured = 0
        landed = abandoned = 0
        for entry in scored:
            outcome = entry["landed"] or O.UNKNOWN
            landed += outcome == O.LANDED
            abandoned += outcome == O.ABANDONED
            value = Run(
                run_id="", at=0, layer=layer, work_kind=work_kind, model=model,
                agent_id="", fleet_id="", session_id="",
                settled=bool(entry["settled"]), turns=0,
                out_tokens=int(entry["out_tokens"] or 0), seconds=None,
                contested=bool(entry["contested"]), landed=outcome).score
            if value is not None:
                total += value
                measured += 1
        return Scorecard(
            layer=layer, work_kind=work_kind, model=model,
            runs=int(row["runs"]), settled=int(row["settled"] or 0),
            contested=int(row["contested"] or 0),
            # 0.0 where nothing was measured, and `measured` is False there so a
            # reader is never handed that zero as though it were a score. The
            # gate cannot reach it either: `MIN_RUNS` is now checked against the
            # measured count, which is zero in exactly that case.
            mean_score=round(total / measured, 6) if measured else 0.0,
            mean_tokens=round(float(row["tokens"] or 0.0), 1),
            last_at=int(row["last_at"] or 0),
            landed=landed, abandoned=abandoned)

    def candidates(self, layer: str, work_kind: str, *, since: int = 0,
                   cwds: tuple[str, ...] | None = None,
                   cwd_inceptions: tuple[tuple[str, int], ...] = ()
                   ) -> tuple[Scorecard, ...]:
        """Every model with a run at this coordinate, best first."""
        if cwds is not None:
            scoped = tuple(run for run in self._scoped_runs(
                               cwds, cwd_inceptions=cwd_inceptions)
                           if run.layer == layer
                           and run.work_kind == work_kind and run.at > since)
            scoped_models = {run.model for run in scoped}
            cards = [card for card in (
                _scorecard_from_runs(layer, work_kind, model, scoped)
                for model in scoped_models) if card is not None]
            return tuple(sorted(cards,
                                key=lambda card: (-card.mean_score, card.model)))
        with self._connect(readonly=True) as conn:
            global_models = [r["model"] for r in conn.execute(
                """SELECT DISTINCT model FROM runs
                    WHERE layer=? AND work_kind=? AND at>? AND space=?""",
                (layer, work_kind, since, H.LAYER_SPACE_VERSION)).fetchall()]
        cards = [c for c in (self.scorecard(layer, work_kind, m, since=since)
                             for m in global_models) if c is not None]
        return tuple(sorted(cards, key=lambda c: (-c.mean_score, c.model)))

    def runs(self, *, layer: str | None = None, work_kind: str | None = None,
             model: str | None = None, limit: int = 50,
             cwds: tuple[str, ...] | None = None,
             cwd_inceptions: tuple[tuple[str, int], ...] = ()
             ) -> tuple[Run, ...]:
        """The individual runs behind the scores, newest first.

        Every other reader here aggregates, and an aggregate is where a wrong
        number is hardest to disbelieve: a coordinate reading 0.448 over 12 runs
        gives a reader nothing to check it against. These are the rows those
        figures were computed from, so the evidence can be looked at rather than
        taken.

        The filters are three fixed columns and the values are bound, never
        interpolated — the vocabulary a caller may filter on is closed by this
        signature rather than by what a caller happens to pass.
        """
        clauses = ["space=?"]
        values: list[object] = [H.LAYER_SPACE_VERSION]
        for column, value in (("layer", layer), ("work_kind", work_kind),
                              ("model", model)):
            if value:
                clauses.append(f"{column}=?")
                values.append(value)
        capped = max(0, int(limit))
        if cwds is not None:
            scope_clause, scope_values = self._scope_predicate(
                cwds, cwd_inceptions)
            clauses.append(scope_clause)
            values.extend(scope_values)
        values.append(capped)
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                f"""SELECT * FROM runs WHERE {' AND '.join(clauses)}
                     ORDER BY at DESC, run_id LIMIT ?""", values).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def _scoped_runs(
            self, cwds: tuple[str, ...], *,
            cwd_inceptions: tuple[tuple[str, int], ...] = ()) -> tuple[Run, ...]:
        """All current-space runs placed at one of the exact checkout roots.

        `None` is deliberately not accepted here: callers use it to mean the
        historical machine-wide ledger. An empty tuple therefore means an
        explicit project scope with no admissible checkout, and returns empty.
        Prefix containment is never placement; `repo-private` is not `repo`.
        Per-root inception floors distinguish unrelated repositories that used
        the same path at different times; a retired occupant's rows stay retired.
        """
        if not cwds:
            return ()
        scope_clause, scope_values = self._scope_predicate(
            cwds, cwd_inceptions)
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                f"SELECT * FROM runs WHERE space=? AND {scope_clause} "
                "ORDER BY at, run_id",
                (H.LAYER_SPACE_VERSION, *scope_values)).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def all_runs(
            self, *, cwds: tuple[str, ...] | None = None,
            cwd_inceptions: tuple[tuple[str, int], ...] = ()) -> tuple[Run, ...]:
        """Every run in the current layer space, oldest first.

        Separate from `runs()` rather than `runs(limit=huge)`, because the two
        are different jobs. `runs()` is an EVIDENCE reader: newest first, capped,
        so a panel can show the rows a figure was computed from. This is an
        AGGREGATION reader, and an aggregate computed over a capped query is
        wrong in the way that is hardest to see — it looks like a total.

        Ordered by `at` ascending with `run_id` breaking ties, so a caller that
        buckets these produces the same buckets on every reading.

        ``cwds=None`` preserves the historical machine-global aggregation API.
        An explicit tuple is an exact repository-placement query and is
        bounded in SQLite before a ``Run`` is constructed.
        """
        if cwds is not None:
            return self._scoped_runs(
                cwds, cwd_inceptions=cwd_inceptions)
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE space=? ORDER BY at, run_id",
                (H.LAYER_SPACE_VERSION,)).fetchall()
        return tuple(_run_from_row(row) for row in rows)

    def run_counts_by_space(self) -> tuple[tuple[int, int], ...]:
        """`(space, runs)` for every layer space in the table, ascending.

        The one reader here that deliberately does NOT filter on
        `H.LAYER_SPACE_VERSION`. Every other query does, which is correct — a
        standing earned under one coordinate space says nothing under another —
        but it means an aggregate built on those queries shrinks silently the
        first time the space is versioned, and still presents itself as
        complete. This is how a caller states that remainder instead of
        inheriting it.

        Today it excludes nothing: all 1,439 rows are in space 1, measured
        2026-08-09. That is the argument for having it now rather than against.
        """
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                "SELECT space, COUNT(*) AS runs FROM runs "
                "GROUP BY space ORDER BY space").fetchall()
        return tuple((int(r["space"]), int(r["runs"])) for r in rows)

    def careers(self, *, cwds: tuple[str, ...] | None = None,
                cwd_inceptions: tuple[tuple[str, int], ...] = ()
                ) -> tuple[Career, ...]:
        """What every model has done across the space, most-run first.

        The ledger is keyed on coordinates, so the model is the one axis it
        could not be read along: answering "what has this model done, and where
        does it hold anything" meant walking every coordinate by hand, and a
        caller doing that in a panel would be computing a fact rather than
        drawing one.

        Ordered by runs and not by score, deliberately. Sorting models by a
        pooled mean would present a ranking the number cannot support — the
        runs behind it are of different kinds — whereas how much a model has
        actually done here is a fact about the record itself.
        """
        if cwds is not None:
            return self._scoped_careers(
                cwds, cwd_inceptions=cwd_inceptions)

        by_model: dict[str, list[Scorecard]] = {}
        current: dict[str, list[tuple[str, str]]] = {}
        for layer_key, kind in self.coordinates():
            # `candidates` rather than a scorecard per (model, coordinate) pair:
            # one query per coordinate for the models that actually ran there,
            # instead of the full product against models that never did.
            for card in self.candidates(layer_key, kind):
                by_model.setdefault(card.model, []).append(card)
            standing = self.standing(layer_key, kind)
            if standing is not None:
                current.setdefault(standing.model, []).append((layer_key, kind))

        out: list[Career] = []
        for model, found in by_model.items():
            cards = tuple(sorted(
                found, key=lambda c: (-c.mean_score, c.layer, c.work_kind)))
            runs = sum(c.runs for c in cards)
            declared, _evidence = H.entry_class(model)
            out.append(Career(
                model=model,
                runs=runs,
                settled=sum(c.settled for c in cards),
                contested=sum(c.contested for c in cards),
                mean_score=round(
                    sum(c.mean_score * c.runs for c in cards) / runs, 6)
                if runs else 0.0,
                mean_tokens=round(
                    sum(c.mean_tokens * c.runs for c in cards) / runs, 1)
                if runs else 0.0,
                last_at=max(c.last_at for c in cards),
                landed=sum(c.landed for c in cards),
                abandoned=sum(c.abandoned for c in cards),
                cards=cards,
                holds=tuple(current.get(model, ())),
                declared_class=declared,
                measured_class=self.measured_class(model)))
        return tuple(sorted(out, key=lambda c: (-c.runs, c.model)))

    def _scoped_careers(
            self, cwds: tuple[str, ...], *,
            cwd_inceptions: tuple[tuple[str, int], ...] = ()
            ) -> tuple[Career, ...]:
        """Careers from exact checkout runs, with global standings withheld."""
        scoped = self._scoped_runs(
            cwds, cwd_inceptions=cwd_inceptions)
        by_coordinate: dict[tuple[str, str, str], list[Run]] = {}
        for run in scoped:
            by_coordinate.setdefault(
                (run.model, run.layer, run.work_kind), []).append(run)

        by_model: dict[str, list[Scorecard]] = {}
        for (model, layer, work_kind), coordinate_runs in by_coordinate.items():
            card = _scorecard_from_runs(
                layer, work_kind, model, tuple(coordinate_runs))
            if card is not None:
                by_model.setdefault(model, []).append(card)

        out: list[Career] = []
        for model, model_cards in by_model.items():
            cards = tuple(sorted(
                model_cards, key=lambda card: (
                    -card.mean_score, card.layer, card.work_kind)))
            runs = sum(card.runs for card in cards)
            declared, _evidence = H.entry_class(model)
            out.append(Career(
                model=model, runs=runs,
                settled=sum(card.settled for card in cards),
                contested=sum(card.contested for card in cards),
                mean_score=round(sum(
                    card.mean_score * card.runs for card in cards) / runs, 6)
                if runs else 0.0,
                mean_tokens=round(sum(
                    card.mean_tokens * card.runs for card in cards) / runs, 1)
                if runs else 0.0,
                last_at=max(card.last_at for card in cards),
                landed=sum(card.landed for card in cards),
                abandoned=sum(card.abandoned for card in cards),
                cards=cards,
                # Standings have no checkout key. Attaching a machine-global
                # standing to a project-scoped career would claim evidence the
                # schema cannot provide, so scoped readers receive no holders
                # or measured class rather than a relabelled global answer.
                holds=(), declared_class=declared, measured_class=None))
        return tuple(sorted(out, key=lambda career: (
            -career.runs, career.model)))

    def standing(self, layer: str, work_kind: str) -> Standing | None:
        """The current standing, or None where nothing has been promoted."""
        with self._connect(readonly=True) as conn:
            row = conn.execute(
                """SELECT * FROM standings
                    WHERE layer=? AND work_kind=? AND space=?
                    ORDER BY seq DESC LIMIT 1""",
                (layer, work_kind, H.LAYER_SPACE_VERSION)).fetchone()
        if row is None or row["state"] == DEMOTED:
            return None
        return Standing(layer=row["layer"], work_kind=row["work_kind"],
                        model=row["model"], state=row["state"],
                        score=float(row["score"]), runs=int(row["runs"]),
                        set_at=int(row["set_at"]), reason=row["reason"])

    def history(self, layer: str, work_kind: str) -> tuple[Standing, ...]:
        """Every standing this coordinate has ever had, newest first.

        The history is the point. A demotion appends; nothing is deleted, so
        "which model held this and when did it stop" is always answerable.
        """
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                """SELECT * FROM standings WHERE layer=? AND work_kind=?
                    ORDER BY seq DESC""", (layer, work_kind)).fetchall()
        return tuple(Standing(
            layer=r["layer"], work_kind=r["work_kind"], model=r["model"],
            state=r["state"], score=float(r["score"]), runs=int(r["runs"]),
            set_at=int(r["set_at"]), reason=r["reason"]) for r in rows)

    def coordinates(self) -> tuple[tuple[str, str], ...]:
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                """SELECT DISTINCT layer, work_kind FROM runs WHERE space=?
                    ORDER BY layer, work_kind""",
                (H.LAYER_SPACE_VERSION,)).fetchall()
        return tuple((r["layer"], r["work_kind"]) for r in rows)

    # ── the ratchet ──────────────────────────────────────────────────────────
    def assess(self, layer: str, work_kind: str, model: str) -> Verdict:
        """What `propose()` would answer, **without writing anything**.

        Every rule the gate applies lives here and `propose()` calls it, so
        there is one statement of "may this model take this coordinate" rather
        than two that can drift. That direction matters: the copy a reader
        consults must be the copy the ratchet obeys, or a surface showing the
        distance to the bar would be describing a gate nobody passes through.

        It exists because asking the question used to require answering it. A
        panel wanting to show *why* a candidate does not hold a coordinate had
        only `propose()`, which appends a standing when the answer is yes — so
        reading the gate could promote a model, and the only safe surface was
        one that did not ask. An accepted verdict here is a statement about the
        record as it stands; nothing is appended, and the standing does not
        move until somebody calls `propose()`.

        It takes no clock, and that is a property of the gate rather than an
        omission: every condition is decided against the incumbent's own
        `set_at`, so the answer does not depend on when it is asked. Only the
        append `propose()` performs needs a moment.
        """
        return self._assess(layer, work_kind, model)

    def propose(self, layer: str, work_kind: str, model: str, *,
                now: int | None = None) -> Verdict:
        """Offer `model` as the standing for a coordinate. Usually refuses.

        Every refusal names the condition and the number that failed it, so a
        caller can act on the answer rather than guess at it. The decision is
        `assess()`'s; what this adds is the append that makes it durable.

        **Read and write are one transaction.** This is a ratchet: it reads the
        incumbent, decides against it, and appends. Between those two steps
        another session's `propose` could commit, and then both would read the
        same incumbent, both find their margin against it, and both append LIVE
        — two live standings at one coordinate, or an incumbent demoted after it
        had already been replaced. The margin gate would have been evaluated
        against a standing that no longer existed.

        `BEGIN IMMEDIATE` takes the writer lock before the read, so proposers
        serialise and the second one assesses against what the first actually
        wrote. `pr_relay.register` does the same thing for the same reason, and
        this module was the one that did not; `alelyon/runtime/atlas/data/
        history.py` and `runtime/common/db.py` carry the pattern too.

        Readers are unaffected — a RESERVED lock does not exclude them — so
        `report`, `standing` and the panels keep working while a proposal is in
        flight.
        """
        moment = int(now if now is not None else time.time())
        with self._connect() as guard:
            guard.execute("BEGIN IMMEDIATE")
            verdict = self._assess(layer, work_kind, model)
            if not verdict.accepted:
                return verdict

            card, incumbent = verdict.challenger, verdict.incumbent
            assert card is not None   # an accepted verdict always carries one
            if incumbent is not None and incumbent.model == model:
                # Re-affirming an incumbent is not a promotion. It is recorded
                # so the standing carries a current score rather than a stale
                # one, and it can still be DEMOTED later.
                self._append(layer, work_kind, model, LIVE, card.mean_score,
                             card.runs, moment,
                             "incumbent re-affirmed on later runs", conn=guard)
                return verdict
            if incumbent is not None:
                self._append(layer, work_kind, incumbent.model, DEMOTED,
                             incumbent.score, incumbent.runs, moment,
                             f"beaten by {model} at {card.mean_score:.3f}",
                             conn=guard)
            self._append(layer, work_kind, model, LIVE, card.mean_score,
                         card.runs, moment,
                         (f"beat {incumbent.model} by "
                          f"{card.mean_score - incumbent.score:.3f}"
                          if incumbent
                          else f"first standing at this coordinate over "
                               f"{card.runs} run(s)"), conn=guard)
            return verdict

    def _assess(self, layer: str, work_kind: str, model: str) -> Verdict:
        target = H.layer(layer)
        if target is None or work_kind not in H.WORK_KINDS:
            return Verdict(False, NOT_A_COORDINATE,
                           f"{layer}/{work_kind} is not a coordinate in layer "
                           f"space v{H.LAYER_SPACE_VERSION}")
        if target.human_only:
            # The one refusal that must never be relaxed. A ratchet able to
            # promote a model into owner authority would dissolve the gate
            # AGENTS.md section 3 is built on, so it is refused here as well as
            # in the placer — two independent places, because this one matters.
            return Verdict(False, NOT_A_COORDINATE,
                           f"{layer} may only be occupied by the owner; "
                           f"AGENTS.md section 3 requires explicit owner "
                           f"authority for Tier 3 and no standing can grant it")

        incumbent = self.standing(layer, work_kind)
        # Held out BY TIME: a challenger is judged only on runs that happened
        # after the incumbent was promoted. Without this a model could win on
        # the very evidence that promoted the other one, which is not a
        # comparison but a re-reading.
        since = incumbent.set_at if incumbent else 0
        card = self.scorecard(layer, work_kind, model, since=since)

        # MEASURED runs, not runs. The module docstring has always said "at
        # least MIN_RUNS **completed** runs" and this compared the total, so a
        # model could clear the bar on rows that established nothing — the same
        # absence counted as evidence here that was counted as failure in the
        # mean. Both halves of the gate now read the same column.
        if card is None or card.settled < MIN_RUNS:
            have = card.settled if card else 0
            unmeasured = (card.runs - card.settled) if card else 0
            reason = NOT_HELD_OUT if (incumbent and card is None) else TOO_FEW_RUNS
            return Verdict(False, reason,
                           f"{model} has {have} measured run(s) at "
                           f"{layer}/{work_kind}"
                           + (f" since the incumbent was set" if incumbent else "")
                           + f"; {MIN_RUNS} are required"
                           + (f" ({unmeasured} more left no outcome in a "
                              f"journal, which is not evidence either way)"
                              if unmeasured else ""),
                           challenger=card, incumbent=incumbent)

        ok, evidence = self.eligible(layer, model)
        if not ok:
            return Verdict(False, BELOW_FLOOR, evidence, challenger=card,
                           incumbent=incumbent)

        if incumbent is not None:
            if incumbent.model == model:
                return Verdict(True, ACCEPTED,
                               f"{model} remains the standing at "
                               f"{card.mean_score:.3f} over {card.runs} later "
                               f"run(s)", challenger=card, incumbent=incumbent)
            # The bar is a fraction of what is still ATTAINABLE above the
            # incumbent, never an absolute step. An absolute step on a bounded
            # score is unsatisfiable near the ceiling, which is how this gate
            # came to refuse every challenger forever.
            headroom = MAX_SCORE - incumbent.score
            if headroom <= 0.0:
                return Verdict(
                    False, SCORE_CEILING,
                    f"{incumbent.model} holds {layer}/{work_kind} at "
                    f"{incumbent.score:.3f}, which is the most this score can "
                    f"express; no challenger can be measurably better on it. "
                    f"The standing is held by exhausted resolution, not by "
                    f"demonstrated merit",
                    challenger=card, incumbent=incumbent)
            required = incumbent.score + MARGIN_FRACTION * headroom
            if card.mean_score < required:
                return Verdict(
                    False, NO_MARGIN,
                    f"{model} scored {card.mean_score:.3f} against "
                    f"{incumbent.model}'s {incumbent.score:.3f}; a challenger "
                    f"must clear {required:.3f}, which is "
                    f"{MARGIN_FRACTION:.0%} of the {headroom:.3f} still "
                    f"available above the incumbent",
                    challenger=card, incumbent=incumbent)

        return Verdict(True, ACCEPTED,
                       f"{model} is now the standing at {layer}/{work_kind} "
                       f"with {card.mean_score:.3f} over {card.runs} run(s)",
                       challenger=card, incumbent=incumbent)

    def demote(self, layer: str, work_kind: str, *, floor: float,
               now: int | None = None) -> Verdict:
        """Drop the standing when it has fallen below its own bar.

        The other half of `docs/MODELS.md` §0 — "models that miss their own bars
        get demoted". Without it the ratchet only ever tightens against
        challengers and never against an incumbent that got worse.
        """
        moment = int(now if now is not None else time.time())
        incumbent = self.standing(layer, work_kind)
        if incumbent is None:
            return Verdict(False, NOT_A_COORDINATE,
                           f"nothing holds {layer}/{work_kind}")
        card = self.scorecard(layer, work_kind, incumbent.model,
                              since=incumbent.set_at)
        if card is None or card.settled < MIN_RUNS:
            return Verdict(False, TOO_FEW_RUNS,
                           f"{incumbent.model} has "
                           f"{card.settled if card else 0} measured run(s) "
                           f"since it was promoted; {MIN_RUNS} are required "
                           f"before it can be demoted on them",
                           incumbent=incumbent)
        if card.mean_score >= floor:
            return Verdict(False, NO_MARGIN,
                           f"{incumbent.model} is at {card.mean_score:.3f}, "
                           f"which is not below its {floor:.3f} bar",
                           challenger=card, incumbent=incumbent)
        self._append(layer, work_kind, incumbent.model, DEMOTED,
                     card.mean_score, card.runs, moment,
                     f"fell to {card.mean_score:.3f}, below its "
                     f"{floor:.3f} bar")
        return Verdict(True, ACCEPTED,
                       f"{incumbent.model} demoted at {card.mean_score:.3f}, "
                       f"below its {floor:.3f} bar. The coordinate now has no "
                       f"standing.", challenger=card, incumbent=incumbent)

    def eligible(self, layer: str, model: str) -> tuple[bool, str]:
        """Whether `model` may hold `layer` at all, and why not.

        Two routes, and the second is the one that makes this a career rather
        than a caste:

        1. **Its class meets the floor.** The ordinary case.
        2. **It currently holds the standing one rank below.** A model promotes
           out of the rung beneath it, on the evidence of having held that rung.

        Without route 2 the ratchet deadlocks, and the deadlock is not obvious
        until you look for it: an unrecognised model enters at the cheapest
        class, the floor then refuses it everywhere above the bottom layer, and
        it can therefore never accumulate the record that would prove it belongs
        higher. Every model the naming table has not heard of would be stuck at
        the bottom forever — which is exactly the failure mode a hierarchy
        built on measurement exists to avoid. It was found by a test rather than
        by reasoning.
        """
        target = H.layer(layer)
        if target is None:
            return False, f"{layer!r} is not a layer in this space"
        ok, evidence = H.fits(model, layer, measured=self.measured_class(model))
        if ok:
            return True, evidence
        below = H.BY_RANK.get(target.rank + 1)
        if below is not None:
            held = self.standings_held(model)
            if below.key in held:
                return True, (f"{model} holds the standing at {below.key}, the "
                              f"rank immediately below {target.key}, so it is "
                              f"promotable into it on merit rather than on its "
                              f"name")
            return False, (f"{evidence}; and it does not hold the standing at "
                           f"{below.key}, which is the other route into "
                           f"{target.key}")
        return False, evidence

    def standings_held(self, model: str) -> tuple[str, ...]:
        """Layers where this model is currently the standing."""
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                """SELECT DISTINCT layer, work_kind FROM standings
                    WHERE model=? AND state=? AND space=?""",
                (model, LIVE, H.LAYER_SPACE_VERSION)).fetchall()
        held = set()
        for row in rows:
            current = self.standing(row["layer"], row["work_kind"])
            if current is not None and current.model == model:
                held.add(row["layer"])
        return tuple(sorted(held))

    def measured_class(self, model: str) -> str | None:
        """The capability class the ledger has established, or None.

        A model that holds a LIVE standing at a layer has been *measured* into
        that layer's class, which is what lets `fits()` override the name-based
        declaration. A model with no standing anywhere returns None and falls
        back to its entry class.
        """
        with self._connect(readonly=True) as conn:
            rows = conn.execute(
                """SELECT layer FROM standings
                    WHERE model=? AND state=? AND space=?""",
                (model, LIVE, H.LAYER_SPACE_VERSION)).fetchall()
        best: str | None = None
        rank = 99
        for row in rows:
            entry = H.layer(row["layer"])
            if entry is not None and entry.rank < rank:
                rank, best = entry.rank, entry.capability
        return best

    def _append(self, layer: str, work_kind: str, model: str, state: str,
                score: float, runs: int, at: int, reason: str,
                conn: sqlite3.Connection | None = None) -> None:
        """Write one standing row, on `conn` when a caller is holding one.

        `propose` holds the writer lock across its whole decision and passes
        that connection in. Opening a second connection here would block on the
        lock the caller already holds — the same process deadlocking against
        itself — so the parameter is not a convenience.
        """
        statement = """INSERT INTO standings
                       (layer, work_kind, model, state, score, runs, set_at,
                        reason, space)
                       VALUES (?,?,?,?,?,?,?,?,?)"""
        row = (layer, work_kind, model, state, float(score), int(runs),
               int(at), reason, H.LAYER_SPACE_VERSION)
        if conn is not None:
            conn.execute(statement, row)
            return
        with self._connect() as owned:
            owned.execute(statement, row)

    # ── reporting ────────────────────────────────────────────────────────────
    def report(self) -> str:
        lines = [f"Fleet ledger — {self.database}", ""]
        coordinates = self.coordinates()
        if not coordinates:
            lines.append("  no runs recorded yet")
        for layer_key, work_kind in coordinates:
            standing = self.standing(layer_key, work_kind)
            lines.append(f"  {layer_key}/{work_kind}")
            if standing is None:
                lines.append("      no standing — nothing has cleared the gate")
            else:
                lines.append(f"      STANDING {standing.model}  "
                             f"{standing.score:.3f} over {standing.runs} run(s)"
                             f"  ({standing.reason})")
            for card in self.candidates(layer_key, work_kind)[:5]:
                # The undecided count is printed beside the decided ones rather
                # than left out. A landed/abandoned pair on its own reads as a
                # complete tally of the runs, and it usually is not one.
                lines.append(f"        {card.model:<28} {pooled_text(card):<11}  "
                             f"{card.settled}/{card.runs} settled  "
                             f"{card.mean_tokens:>8.0f} tok  "
                             f"{card.landed} landed / {card.abandoned} "
                             f"abandoned / {card.undecided} undecided")
        lines += ["", "WHAT THIS CANNOT TELL YOU"]
        lines += [f"  - {limit}" for limit in SCORE_LIMITS]
        return "\n".join(lines)


# ── deriving runs from what the fleet already recorded ───────────────────────
def runs_from_activity(activity, *, contested_paths=(), landing=None,
                       now: int | None = None) -> tuple[Run, ...]:
    """Score every settled agent in an `Activity` reading.

    `contested_paths` is the files some session **later** published a defect or
    interface finding about; an agent that touched one is marked contested. The
    caller supplies it because ordering the finding against the run is a
    question about the bus, not about the transcripts.

    Each item is either a bare path or a `(path, at)` pair carrying the moment
    the finding was published. **Pass the pair wherever you have the time.** A
    bare path cannot express "later" at all, so it marks every run that touched
    the file including the ones that finished after the defect was already
    known — and `contested` multiplies a score by 0.4, so that is a heavy
    penalty applied on no evidence. The pair form is honoured against `Run.at`:
    a finding published before a run settled does not contest it. A run whose
    own time is unknown (`at == 0`) is contested by any finding about a file it
    touched, because nothing here can order the two and the bare-path reading
    is the one that was already in force.

    `landing` is a `fleet_outcomes.LandingIndex`. Optional, and its absence is
    not neutrality dressed up as a default — without it every run records
    UNKNOWN, which is the honest reading of "nobody asked the repository", and
    `reconcile()` can supply the answer later.

    The branch is taken from the AGENT and falls back to the session's. An
    agent given `isolation: worktree` runs on a branch of its own, so reading
    the session's would attribute its work to whatever the parent happened to
    have checked out.
    """
    from alelyon.runtime.common import session_activity as SA

    always, latest = _contested_index(contested_paths)
    out: list[Run] = []
    for session in activity.sessions:
        for fleet in session.fleets:
            for agent in fleet.agents:
                if agent.status == SA.RUNNING:
                    continue        # score it when it has stopped moving
                placed, _evidence = H.place(agent.brief or agent.agent_type)
                kind = next((k for k, v in H.WORK_KINDS.items()
                             if v == placed.key), placed.key)
                # `repo_files`, not `files`. The bus records a finding's subject
                # as a REPOSITORY-RELATIVE posix path, and the harness stamps an
                # agent's files as absolute native ones — so intersecting the two
                # could never match, and `contested` had never once fired on this
                # repository however many defects the fleet published. Measured
                # 2026-08-05: 58 runs, 109 subject paths, 0 contested before this
                # line and non-zero after. `repo_files` also drops files outside
                # the checkout, which is right: a scratchpad file is not
                # something another session can publish a finding about.
                touched = set(getattr(agent, "repo_files", None)
                              or agent.files)
                branch = getattr(agent, "branch", "") or session.branch
                cwd = getattr(agent, "cwd", "") or session.cwd
                at = agent.last_at or 0
                outcome = O.UNKNOWN
                if landing is not None:
                    outcome = landing.of(branch, settled_at=at, now=now,
                                         cwd=cwd).outcome
                out.append(Run(
                    run_id=f"{agent.session_id}/{agent.fleet_id}/{agent.agent_id}",
                    at=at,
                    layer=placed.key, work_kind=kind, model=agent.model,
                    agent_id=agent.agent_id, fleet_id=agent.fleet_id,
                    session_id=agent.session_id,
                    settled=agent.status == SA.SETTLED,
                    turns=agent.turns, out_tokens=agent.output_tokens,
                    seconds=agent.elapsed_seconds,
                    contested=_is_contested(touched, at, always, latest),
                    branch=branch, cwd=cwd, landed=outcome))
    return tuple(out)


def _contested_index(contested_paths) -> tuple[set[str], dict[str, int]]:
    """`(always, latest)` — the bare paths, and the timed ones' last publication.

    The question is *does ANY finding about this file postdate the run*, so the
    LATEST publication per path is the one that answers it: if the last one does
    not postdate the run, none of them do. Keeping the earliest instead would
    lose every contest a later finding establishes, which is the same false
    negative in reverse and was caught by a test rather than by reading this.
    """
    always: set[str] = set()
    latest: dict[str, int] = {}
    for item in contested_paths:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            path, at = item
            key = str(path).replace("\\", "/")
            when = int(at)
            latest[key] = max(latest.get(key, when), when)
        else:
            always.add(str(item).replace("\\", "/"))
    return always, latest


def _is_contested(touched, at: int, always: set[str],
                  latest: dict[str, int]) -> bool:
    """Whether a defect finding lands on this run, respecting publication order.

    `>=` and not `>`: transcript and bus times are both whole seconds, so a
    finding published in the same second a run settled cannot be ordered
    against it at all. The tie keeps the conservative reading already in force
    rather than choosing the lenient one on evidence that does not exist.
    """
    if touched & always:
        return True
    return any(when >= at
               for when in (latest.get(p) for p in touched)
               if when is not None)


__all__ = [
    "ABANDONED_PENALTY",
    "ACCEPTED", "BELOW_FLOOR", "CHANGE_FEED_SCHEMA", "CHANGE_KINDS",
    "Career", "ChangeFeedState", "DEMOTED", "FleetChangePage",
    "FleetChangeReport", "FleetLedger", "FleetRunChange",
    "LEDGER_SCHEMA_VERSION", "LIVE", "MARGIN_FRACTION", "MAX_SCORE",
    "MIN_RUNS", "NOT_A_COORDINATE", "SCORE_CEILING",
    "NOT_HELD_OUT", "NO_MARGIN", "OBSERVATION", "PAPER", "RECORDED",
    "RECONCILED", "Run", "SCORE_LIMITS", "STATES", "Scorecard", "Standing",
    "TOO_FEW_RUNS", "Verdict", "default_database", "pooled_text",
    "read_change_feed", "runs_from_activity",
]
