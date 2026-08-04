"""The career: what each model actually did at each layer, and the ratchet.

`fleet_hierarchy` says what a layer is worth. This says what a *model* is worth
at that layer, from runs that happened — and it is the half that makes the
hierarchy improve rather than merely exist.

The ratchet, stated exactly
---------------------------
One standing per `(layer, work_kind)`: the model currently held to be best for
that job. A challenger replaces it only through `propose()`, which refuses
unless **all** of these hold:

1. the challenger has at least `MIN_RUNS` completed runs at that coordinate;
2. those runs are **later than the incumbent's standing was set** — held out by
   time, so a challenger cannot win on the evidence that promoted the incumbent;
3. it beats the incumbent's score by at least `MARGIN`;
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

from dataclasses import dataclass
from pathlib import Path
import sqlite3
import time

from alelyon.runtime.common import fleet_hierarchy as H

LEDGER_SCHEMA_VERSION = 1

#: Runs required at a coordinate before a model may be proposed for it. Small,
#: because the population is small — and stated as the weak number it is: five
#: runs is evidence a coin would clear one time in sixteen, so a promotion is a
#: working hypothesis, not a finding.
MIN_RUNS = 5

#: How much better a challenger must score. A margin rather than "greater than",
#: because with tens of runs a hair's difference is noise and a standing that
#: flipped on noise would make the ratchet a random walk with extra steps.
MARGIN = 0.05

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
BELOW_FLOOR = "below-capability-floor"
NOT_A_COORDINATE = "not-a-coordinate"
ACCEPTED = "accepted"

SCORE_LIMITS: tuple[str, ...] = (
    "The score measures COMPLETION and COST, not quality. An agent that "
    "settled quickly and cheaply on a wrong answer scores well, and nothing "
    "here would notice.",
    "contested-after is the only negative-quality signal, and it is weak: it "
    "fires when somebody else published a finding about a file this agent "
    "touched afterwards, which catches a real defect and also catches two "
    "sessions working near each other.",
    "A run is attributed to the model the harness recorded for its turns. An "
    "agent re-driven on a second model is attributed to whichever ran most of "
    "its turns, so a mixed run is scored as one model's work.",
    "MIN_RUNS is five. Five runs is a working hypothesis, not a finding - a "
    "coin clears that bar one time in sixteen.",
    "The ratchet guarantees the standing never moves to a candidate that "
    "scored worse on the evidence available. It cannot guarantee the standing "
    "is the best model, because a model nobody ran has no evidence at all.",
    "Nothing here dispatches. A standing is a recommendation a caller may read "
    "before naming a model, and a session is free to ignore it.",
)

_DDL = (
    """CREATE TABLE IF NOT EXISTS meta (
           name TEXT PRIMARY KEY, value TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS runs (
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
           space       INTEGER NOT NULL)""",
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

    @property
    def score(self) -> float:
        """Completion, penalised by cost and by a later defect finding.

        In [0, 1]. Deliberately simple and deliberately printed beside its
        limits: a compound score with tuned weights would look like a
        measurement of quality, and this is a measurement of whether the job
        finished and what it cost to finish.
        """
        if not self.settled:
            return 0.0
        # Cost term: 1.0 at zero tokens, decaying with a 40k half-life. A
        # constant rather than a fitted parameter, because fitting it against
        # the same runs it scores would be circular.
        cost = 0.5 ** (max(0, self.out_tokens) / 40_000.0)
        value = 0.75 + 0.25 * cost
        return round(value * (0.4 if self.contested else 1.0), 6)


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

    @property
    def completion(self) -> float:
        return self.settled / self.runs if self.runs else 0.0


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


def default_database() -> Path:
    from alelyon.runtime.common.paths import GLOBALS_DIR
    return Path(GLOBALS_DIR) / "fleet_ledger.db"


class FleetLedger:
    """Durable record of what every model did at every layer. Append-only."""

    def __init__(self, database: str | Path | None = None) -> None:
        self.database = Path(database or default_database())
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            for statement in _DDL:
                conn.execute(statement)
            conn.execute(
                "INSERT OR IGNORE INTO meta(name, value) VALUES('schema', ?)",
                (str(LEDGER_SCHEMA_VERSION),))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.database), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

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
                    space)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run.run_id, run.at, run.layer, run.work_kind, run.model,
                 run.agent_id, run.fleet_id, run.session_id, int(run.settled),
                 run.turns, run.out_tokens, run.seconds, int(run.contested),
                 H.LAYER_SPACE_VERSION))
            return cursor.rowcount > 0

    def record_all(self, runs) -> int:
        return sum(1 for run in runs if self.record(run))

    # ── reading ──────────────────────────────────────────────────────────────
    def scorecard(self, layer: str, work_kind: str, model: str, *,
                  since: int = 0) -> Scorecard | None:
        with self._connect() as conn:
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
                """SELECT settled, out_tokens, contested FROM runs
                    WHERE layer=? AND work_kind=? AND model=? AND at>? AND space=?""",
                (layer, work_kind, model, since, H.LAYER_SPACE_VERSION)).fetchall()
        total = 0.0
        for entry in scored:
            total += Run(
                run_id="", at=0, layer=layer, work_kind=work_kind, model=model,
                agent_id="", fleet_id="", session_id="",
                settled=bool(entry["settled"]), turns=0,
                out_tokens=int(entry["out_tokens"] or 0), seconds=None,
                contested=bool(entry["contested"])).score
        return Scorecard(
            layer=layer, work_kind=work_kind, model=model,
            runs=int(row["runs"]), settled=int(row["settled"] or 0),
            contested=int(row["contested"] or 0),
            mean_score=round(total / int(row["runs"]), 6),
            mean_tokens=round(float(row["tokens"] or 0.0), 1),
            last_at=int(row["last_at"] or 0))

    def candidates(self, layer: str, work_kind: str, *,
                   since: int = 0) -> tuple[Scorecard, ...]:
        """Every model with a run at this coordinate, best first."""
        with self._connect() as conn:
            models = [r["model"] for r in conn.execute(
                """SELECT DISTINCT model FROM runs
                    WHERE layer=? AND work_kind=? AND at>? AND space=?""",
                (layer, work_kind, since, H.LAYER_SPACE_VERSION)).fetchall()]
        cards = [c for c in (self.scorecard(layer, work_kind, m, since=since)
                             for m in models) if c is not None]
        return tuple(sorted(cards, key=lambda c: (-c.mean_score, c.model)))

    def standing(self, layer: str, work_kind: str) -> Standing | None:
        """The current standing, or None where nothing has been promoted."""
        with self._connect() as conn:
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
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM standings WHERE layer=? AND work_kind=?
                    ORDER BY seq DESC""", (layer, work_kind)).fetchall()
        return tuple(Standing(
            layer=r["layer"], work_kind=r["work_kind"], model=r["model"],
            state=r["state"], score=float(r["score"]), runs=int(r["runs"]),
            set_at=int(r["set_at"]), reason=r["reason"]) for r in rows)

    def coordinates(self) -> tuple[tuple[str, str], ...]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT layer, work_kind FROM runs WHERE space=?
                    ORDER BY layer, work_kind""",
                (H.LAYER_SPACE_VERSION,)).fetchall()
        return tuple((r["layer"], r["work_kind"]) for r in rows)

    # ── the ratchet ──────────────────────────────────────────────────────────
    def propose(self, layer: str, work_kind: str, model: str, *,
                now: int | None = None) -> Verdict:
        """Offer `model` as the standing for a coordinate. Usually refuses.

        Every refusal names the condition and the number that failed it, so a
        caller can act on the answer rather than guess at it.
        """
        moment = int(now if now is not None else time.time())
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

        if card is None or card.runs < MIN_RUNS:
            have = card.runs if card else 0
            reason = NOT_HELD_OUT if (incumbent and card is None) else TOO_FEW_RUNS
            return Verdict(False, reason,
                           f"{model} has {have} run(s) at {layer}/{work_kind}"
                           + (f" since the incumbent was set" if incumbent else "")
                           + f"; {MIN_RUNS} are required",
                           challenger=card, incumbent=incumbent)

        ok, evidence = self.eligible(layer, model)
        if not ok:
            return Verdict(False, BELOW_FLOOR, evidence, challenger=card,
                           incumbent=incumbent)

        if incumbent is not None:
            if incumbent.model == model:
                # Re-affirming an incumbent is not a promotion. It is recorded
                # so the standing carries a current score rather than a stale
                # one, and it can still be DEMOTED below.
                self._append(layer, work_kind, model, LIVE, card.mean_score,
                             card.runs, moment,
                             "incumbent re-affirmed on later runs")
                return Verdict(True, ACCEPTED,
                               f"{model} remains the standing at "
                               f"{card.mean_score:.3f} over {card.runs} later "
                               f"run(s)", challenger=card, incumbent=incumbent)
            if card.mean_score < incumbent.score + MARGIN:
                return Verdict(
                    False, NO_MARGIN,
                    f"{model} scored {card.mean_score:.3f} against "
                    f"{incumbent.model}'s {incumbent.score:.3f}; a challenger "
                    f"must clear {incumbent.score + MARGIN:.3f}",
                    challenger=card, incumbent=incumbent)
            self._append(layer, work_kind, incumbent.model, DEMOTED,
                         incumbent.score, incumbent.runs, moment,
                         f"beaten by {model} at {card.mean_score:.3f}")

        state = LIVE
        self._append(layer, work_kind, model, state, card.mean_score, card.runs,
                     moment,
                     (f"beat {incumbent.model} by "
                      f"{card.mean_score - incumbent.score:.3f}" if incumbent
                      else f"first standing at this coordinate over "
                           f"{card.runs} run(s)"))
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
        if card is None or card.runs < MIN_RUNS:
            return Verdict(False, TOO_FEW_RUNS,
                           f"{incumbent.model} has {card.runs if card else 0} "
                           f"run(s) since it was promoted; {MIN_RUNS} are "
                           f"required before it can be demoted on them",
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
        with self._connect() as conn:
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
        with self._connect() as conn:
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
                score: float, runs: int, at: int, reason: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO standings
                   (layer, work_kind, model, state, score, runs, set_at,
                    reason, space)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (layer, work_kind, model, state, float(score), int(runs),
                 int(at), reason, H.LAYER_SPACE_VERSION))

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
                lines.append(f"        {card.model:<28} {card.mean_score:.3f}  "
                             f"{card.settled}/{card.runs} settled  "
                             f"{card.mean_tokens:>8.0f} tok")
        lines += ["", "WHAT THIS CANNOT TELL YOU"]
        lines += [f"  - {limit}" for limit in SCORE_LIMITS]
        return "\n".join(lines)


# ── deriving runs from what the fleet already recorded ───────────────────────
def runs_from_activity(activity, *, contested_paths=()) -> tuple[Run, ...]:
    """Score every settled agent in an `Activity` reading.

    `contested_paths` is the set of files some session later published a defect
    or interface finding about; an agent that touched one is marked contested.
    The caller supplies it because ordering the finding against the run is a
    question about the bus, not about the transcripts.
    """
    from alelyon.runtime.common import session_activity as SA

    contested = {str(p).replace("\\", "/") for p in contested_paths}
    out: list[Run] = []
    for session in activity.sessions:
        for fleet in session.fleets:
            for agent in fleet.agents:
                if agent.status == SA.RUNNING:
                    continue        # score it when it has stopped moving
                placed, _evidence = H.place(agent.brief or agent.agent_type)
                kind = next((k for k, v in H.WORK_KINDS.items()
                             if v == placed.key), placed.key)
                touched = {p.replace("\\", "/") for p in agent.files}
                out.append(Run(
                    run_id=f"{agent.session_id}/{agent.fleet_id}/{agent.agent_id}",
                    at=agent.last_at or 0,
                    layer=placed.key, work_kind=kind, model=agent.model,
                    agent_id=agent.agent_id, fleet_id=agent.fleet_id,
                    session_id=agent.session_id,
                    settled=agent.status == SA.SETTLED,
                    turns=agent.turns, out_tokens=agent.output_tokens,
                    seconds=agent.elapsed_seconds,
                    contested=bool(touched & contested)))
    return tuple(out)


__all__ = [
    "ACCEPTED", "BELOW_FLOOR", "DEMOTED", "FleetLedger",
    "LEDGER_SCHEMA_VERSION", "LIVE", "MARGIN", "MIN_RUNS", "NOT_A_COORDINATE",
    "NOT_HELD_OUT", "NO_MARGIN", "OBSERVATION", "PAPER", "Run", "SCORE_LIMITS",
    "STATES", "Scorecard", "Standing", "TOO_FEW_RUNS", "Verdict",
    "default_database", "runs_from_activity",
]
