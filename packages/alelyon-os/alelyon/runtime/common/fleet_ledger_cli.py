"""Fill the fleet ledger, and re-read later what became of the work.

    alelyon-ledger status                 # what has been measured, and by whom
    alelyon-ledger record                 # copy settled agent runs in
    alelyon-ledger reconcile              # re-read the outcomes that were open
    alelyon-ledger promote                # offer the best candidate to the gate
    alelyon-ledger landings               # the branch picture, writing nothing

Why this exists
---------------
`docs/features/FLEET-HIERARCHY.md` §6 listed **"something that calls
`reconcile()`"** as not built, and said §3.4 does nothing without it: a run is
recorded the hour it finishes, when nobody has merged anything yet, so its
landing is `IN-FLIGHT` — and if that were the last word the one label carrying a
penalty would never once be applied.

Looking for that caller found a larger absence. `record_all()`, `reconcile()` and
`propose()` had **no caller anywhere in the repository**. Nothing wrote the
ledger at all, so `fleet_dispatch.recommend()` — shipped, tested, and reading
`standing()` — was guaranteed to answer "no standing" forever, and the Fleet
hub's hierarchy view was guaranteed to draw an empty column. The three commands
below are that chain, in the order it has to run: **record** what happened,
**reconcile** what became of it, **promote** on the result.

Nothing here runs unattended, and that is the design
----------------------------------------------------
§6 raised the question rather than assuming the answer: *whether anything should
reconcile unattended is a genuine question, because the argument for making a
durable write a deliberate act applies unchanged to re-reading the branch graph.*
This answers it by staying a command. Every write is something a person or an
agent decided to make, at a moment, with the output in front of them. There is no
timer, no daemon, and no hook — a promotion made on a schedule is a standing
nobody chose, and the ledger's whole claim is that a standing is evidence
somebody looked at.

The division of labour between `record` and `reconcile`
-------------------------------------------------------
`record` asks git **nothing**. It reads the harness transcripts, places each
settled agent on the layer space, and appends. Every run it writes starts
`UNKNOWN`, which is the honest reading of "nobody asked the repository".

`reconcile` is the expensive half: it builds a `LandingIndex`, which costs a few
hundred git queries on a repository this size, and re-reads only the rows whose
outcome is still provisional. Splitting them that way means the cheap pass can be
run often and the git pass when the branch graph has actually moved.

Read-only with respect to git — every git call is a query. It writes to one
SQLite file, and `status` will not even create that.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from alelyon.runtime.common import fleet_hierarchy as H
from alelyon.runtime.common import fleet_ledger as L
from alelyon.runtime.common import fleet_outcomes as O
from alelyon.runtime.common import session_activity as SA

#: How many transcripts `record` reads by default. `session_activity` defaults to
#: 12, which is right for a live view of who is working now and wrong here: this
#: is building a career record, and the runs worth measuring are mostly older
#: than the twelve most recently written files. Whatever is left unread is
#: reported by `Activity.notes` rather than silently dropped.
DEFAULT_MAX_SESSIONS = 400

LIMITS: tuple[str, ...] = (
    "Nothing here runs unattended. Every command is a deliberate act, because a "
    "durable write made on a timer is a write nobody decided to make - and once "
    "the answer is stored, re-reading the branch graph is a durable write.",
    "`record` marks no run CONTESTED. The deriver takes one flat set of paths, "
    "which cannot express 'a finding published AFTER this run', and passing "
    "every finding's paths would penalise the agent that FIXED a defect exactly "
    "as hard as the one that caused it. Contestation is UNMEASURED here rather "
    "than guessed at, and the score is completion and cost only.",
    "`record` asks git nothing, so every run it appends starts UNKNOWN. That is "
    "the absence of an answer, not a finding that the work went nowhere, and it "
    "is what `reconcile` exists to settle later.",
    "`reconcile` has no dry run. Previewing it would need a second copy of the "
    "clause deciding which rows are still provisional, and two copies of that "
    "clause is how a guard drifts from the thing it guards. It only ever moves "
    "a row from provisional to terminal, and never the other way.",
    "Reading every session reads other repositories' sessions too. A model's "
    "coordinate is repository-agnostic and a branch name is not, so work done "
    "in another checkout reads UNKNOWN against this one - which moves no score. "
    "Narrow with --cwd if that is not wanted.",
    "Nothing here demotes. `demote()` needs a floor, and a floor is a policy "
    "number the owner sets rather than one a command line invents.",
    "A standing is a recommendation a caller may read before naming a model. "
    "This writes the record the recommendation is computed from; it dispatches "
    "nothing and a session is free to ignore all of it.",
)


def _limits() -> None:
    print()
    print("WHAT THIS COMMAND CANNOT DO")
    for limit in LIMITS:
        print(f"  - {limit}")


def _database(args) -> Path:
    return Path(args.database) if args.database else L.default_database()


def _open_existing(args) -> tuple[Optional[L.FleetLedger], Path]:
    """The ledger, or None where there is none.

    `FleetLedger(...)` CREATES its file, so a reader that opened one to draw
    itself would manufacture an empty ledger in any checkout that has never run
    these commands and then report "nothing has been measured" — which reads as
    a finding about models and is really a description of the file it just made.
    The hierarchy panel learned this first; the same rule applies here.
    """
    path = _database(args)
    if not path.exists():
        return None, path
    return L.FleetLedger(path), path


def _no_ledger(path: Path) -> None:
    print(f"No fleet ledger at {path}.")
    print("  Nothing has been recorded here, which is not the same as nothing")
    print("  having happened: the runs are in the harness transcripts until")
    print("  `record` copies them in. This command creates no file.")


def _landing_index(args) -> O.LandingIndex:
    return O.LandingIndex(args.repo, mainline=args.mainline,
                          settling_seconds=int(args.settling_days) * 86_400)


def _by_coordinate(runs) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for run in runs:
        key = (run.layer, run.work_kind)
        counts[key] = counts.get(key, 0) + 1
    return counts


# ── commands ─────────────────────────────────────────────────────────────────
def _cmd_status(args) -> int:
    ledger, path = _open_existing(args)
    if ledger is None:
        _no_ledger(path)
        _limits()
        return 0
    print(ledger.report())
    return 0


def contested_paths(database: str = "", *, repo: str = ".") -> tuple:
    """`(path, published_at)` pairs for every defect or interface finding.

    TIMED PAIRS, not bare paths. `contested` multiplies a run's score by 0.4 —
    the largest single penalty in the rule — and what earns it, per
    `docs/features/FLEET-HIERARCHY.md` §3.1, is a finding published about a
    touched file **afterwards**. A bare path cannot express "afterwards", so it
    marks every run that ever touched the file, including the runs that
    finished after the defect was already known and the ones sent to fix it.
    `runs_from_activity` honours the pair form against the run's own settle
    time.

    The bus is NOT created when it is absent: an empty result then means nobody
    published anything, which is what an unread bus means. Creating one to
    discover it is empty would report "no defects" about a file made a moment
    earlier.
    """
    from alelyon.runtime.common import worktree_areas as A
    from alelyon.runtime.common import worktree_bus as B

    path = Path(database) if database else B.default_database()
    if not path.exists():
        return ()
    # The space belongs to the repository under observation, for the reason
    # `fleet_cli.main` gives: the process default places another repository's
    # paths with this one's rules the moment the two differ.
    bus = B.FleetBus(path, space=A.load(Path(repo)))
    wanted = (B.KIND_DEFECT, B.KIND_INTERFACE)
    return tuple((subject, finding.at)
                 for finding in bus.findings(limit=10_000)
                 if finding.kind in wanted
                 for subject in finding.subject_paths)


def _cmd_record(args) -> int:
    """Copy every settled agent run the harness recorded into the ledger."""
    activity = SA.read_activity(cwd=args.cwd or None,
                                records_root=args.records_root or None,
                                max_sessions=args.max_sessions)
    contested = () if args.no_bus else contested_paths(args.bus, repo=args.repo)
    runs = L.runs_from_activity(activity, contested_paths=contested)

    print(f"{len(activity.sessions)} session(s) read from "
          f"{activity.records_root}")
    for note in activity.notes:
        print(f"  note: {note}")
    if args.no_bus:
        print("  --no-bus: no run is marked contested, whatever the fleet found")
    elif contested:
        print(f"  {len(set(p for p, _ in contested))} path(s) carry a defect or "
              f"interface finding, ordered against each run's settle time")
    else:
        print("  no defect or interface finding was read from the bus")
    print(f"{len(runs)} settled agent run(s) derived, "
          f"{sum(1 for r in runs if r.contested)} contested")
    for (layer_key, kind), count in sorted(_by_coordinate(runs).items()):
        print(f"    {layer_key:<11} {kind:<18} {count:>5}")

    if args.dry_run:
        print("\n--dry-run: nothing was written, and no ledger was created.")
        _limits()
        return 0

    ledger = L.FleetLedger(_database(args))
    added = ledger.record_all(runs)
    print(f"\n{added} new run(s) recorded in {ledger.database}")
    print(f"{len(runs) - added} were already there — `record` is idempotent on "
          f"the run id, because the same finished agent is offered again on "
          f"every pass.")
    _limits()
    return 0


def _cmd_reconcile(args) -> int:
    """The caller §6 named. Re-read the runs whose landing was still open."""
    ledger, path = _open_existing(args)
    if ledger is None:
        _no_ledger(path)
        _limits()
        return 0

    index = _landing_index(args)
    if args.explain:
        print(index.report())
        print()

    tally = ledger.reconcile(index)
    read = tally.pop("read", 0)
    unchanged = tally.pop("unchanged", 0)
    print(f"{read} provisional run(s) re-read against {args.mainline} "
          f"as of {index.repo_root}")
    print(f"  {unchanged} unchanged")
    for outcome in O.OUTCOMES:
        if tally.get(outcome):
            print(f"  {tally[outcome]:>5} -> {outcome}")
    if read and not tally:
        print("  nothing moved. IN-FLIGHT is not a result and does not become "
              "one until the settling window closes.")
    if not read:
        print("  every recorded run already carries a terminal outcome, or "
              "there are none to read.")
    _limits()
    return 0


def _cmd_promote(args) -> int:
    """Offer each coordinate's best candidate to the gate. It usually refuses."""
    ledger, path = _open_existing(args)
    if ledger is None:
        _no_ledger(path)
        _limits()
        return 0

    coordinates = [c for c in ledger.coordinates()
                   if (not args.layer or c[0] == args.layer)
                   and (not args.work_kind or c[1] == args.work_kind)]
    if not coordinates:
        print("No coordinate has a run recorded against it"
              + (" that matches that filter." if args.layer or args.work_kind
                 else ". Run `record` first."))
        _limits()
        return 0

    accepted = 0
    for layer_key, kind in coordinates:
        cards = ledger.candidates(layer_key, kind)
        model = args.model or (cards[0].model if cards else "")
        if not model:
            continue
        verdict = ledger.propose(layer_key, kind, model)
        accepted += bool(verdict.accepted)
        mark = "ACCEPTED" if verdict.accepted else "refused "
        print(f"  {mark} {layer_key}/{kind}  {model}")
        print(f"           {verdict.reason}: {verdict.detail}")
    print(f"\n{accepted} standing(s) moved out of {len(coordinates)} "
          f"coordinate(s) offered. Refusal is the ordinary outcome: the gate "
          f"holds evidence out by time and requires {L.MIN_RUNS} runs.")
    _limits()
    return 0


def _cmd_landings(args) -> int:
    """The branch picture the outcome label is derived from. Writes nothing."""
    print(_landing_index(args).report())
    return 0


_COMMANDS = {
    "status": _cmd_status, "record": _cmd_record, "reconcile": _cmd_reconcile,
    "promote": _cmd_promote, "landings": _cmd_landings,
}


def _add_global_options(parser: argparse.ArgumentParser, *,
                        suppress: bool) -> None:
    """The options that apply to every command, on the parser AND each command.

    Declared twice on purpose. Argparse only accepts a top-level option BEFORE
    the subcommand, so `alelyon-ledger reconcile --repo X` — which is how anyone
    actually types it — is an error on a parser that declares them once. Adding
    them to each subparser as well makes both orders work.

    The copies default to `SUPPRESS` rather than to the same values. With real
    defaults the subparser writes its default over whatever the top level
    parsed, so `--database X status` would silently fall back to the default
    database: a flag accepted, ignored, and never reported.
    """
    kwargs = {"default": argparse.SUPPRESS} if suppress else {}
    parser.add_argument("--repo", help="the repository whose branches decide a "
                                       "landing (default: here)",
                        **({"default": "."} | kwargs))
    parser.add_argument("--mainline", **({"default": "origin/main"} | kwargs))
    parser.add_argument("--database",
                        help="override the ledger file; the default is "
                             "resolved by fleet_ledger.default_database()",
                        **({"default": ""} | kwargs))
    parser.add_argument("--settling-days", type=int,
                        help="how long an unmerged branch is given before its "
                             "silence is read as a result (a declaration, not "
                             "a measurement)",
                        **({"default": O.SETTLING_DAYS} | kwargs))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alelyon-ledger",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_global_options(parser, suppress=False)
    shared = argparse.ArgumentParser(add_help=False)
    _add_global_options(shared, suppress=True)

    sub = parser.add_subparsers(dest="command", required=True)

    def sub_parser(name: str, **kwargs) -> argparse.ArgumentParser:
        return sub.add_parser(name, parents=[shared], **kwargs)

    sub_parser("status", help="what has been measured, and by whom")

    record = sub_parser("record", help="copy settled agent runs in")
    record.add_argument("--dry-run", action="store_true",
                        help="derive and count the runs, write nothing, and "
                             "create no ledger")
    record.add_argument("--cwd", default="",
                        help="only sessions whose working directory is this "
                             "one; the default reads every session the harness "
                             "recorded")
    record.add_argument("--records-root", default="",
                        help="where the harness transcripts live")
    record.add_argument("--max-sessions", type=int,
                        default=DEFAULT_MAX_SESSIONS)
    record.add_argument("--bus", default="",
                        help="fleet bus file to read defect and interface "
                             "findings from (default: the shared one). An "
                             "absent bus is not created to find out it is empty")
    record.add_argument("--no-bus", action="store_true",
                        help="do not read the bus; no run is marked contested. "
                             "The score then measures completion and cost only")

    reconcile = sub_parser(
        "reconcile", help="re-read the outcomes that were still open")
    reconcile.add_argument("--explain", action="store_true",
                           help="print the branch picture first")

    promote = sub_parser("promote",
                         help="offer the best candidate to the gate")
    promote.add_argument("--layer", default="", choices=[""] + [
        entry.key for entry in H.LAYERS])
    promote.add_argument("--work-kind", default="")
    promote.add_argument("--model", default="",
                         help="offer this model rather than the best-scoring "
                              "candidate at each coordinate")

    sub_parser("landings", help="the branch picture, writing nothing")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
