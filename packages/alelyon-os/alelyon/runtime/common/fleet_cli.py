"""Talk to the other agent sessions working in a repository, and find work.

    alelyon-fleet status                       # who is where
    alelyon-fleet inbox                        # what others told me
    alelyon-fleet publish --kind refactor-in-flight \\
        --about src/engine/tools.py \\
        --body "splitting the tool layer; transient NameErrors expected"
    alelyon-fleet open-areas                   # where nobody is working
    alelyon-fleet claim engine/tools --note "adding a unit checker"
    alelyon-fleet ack <finding-id>
    alelyon-fleet areas                        # the coordinate space in force

Read-only with respect to git — every git call is a query. It writes to one
SQLite file.

**Which repository?** `--repo` (default: the current directory). Everything is
resolved from there: the worktrees, the coordinate space, the tracked paths.
Nothing is read from anywhere else and no directory is tracked because a table
somewhere named it. A checkout that declares `.alelyon/fleet.toml` gets the
vocabulary it declares; one that declares nothing gets one discovered from its
own tracked directories; a directory that is not a git checkout gets no
coordinates at all rather than invented ones. `areas` prints which of those
happened, and on what evidence.

**Who am I?** Derived from the worktree this command runs in, using the path
conventions in `alelyon.runtime.common.worktree`. Where no convention carries a
session id the answer is UNATTRIBUTED and `--session` is required, and what you
pass is recorded as self-reported. That asymmetry is the point: a derived
identity and a declared one are not the same evidence and are never merged.

**What this cannot do** prints at the end of every command rather than living
only in a docstring. Read it once. The short version: a claim is not a lock, a
finding's body is nobody's fact but the publisher's, and a session working
outside every known convention is invisible here and is not being told anything.

This module holds the whole CLI. `tools/fleet.py` in the source repository is a
shim onto it, and the `alelyon-fleet` console script is the same entry point, so
there is one implementation rather than one that ships and one that works.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from alelyon.runtime.common import session_records as S
from alelyon.runtime.common import worktree as W
from alelyon.runtime.common import worktree_areas as A
from alelyon.runtime.common import worktree_bus as B
from alelyon.runtime.common import worktree_cache as C
from alelyon.runtime.common import worktree_disciplines as D


def _identify(mesh, override: str) -> tuple[str, str]:
    """(session, evidence) for the caller. Derived first, declared as fallback.

    Two derivations are tried, in order of how specific they are.

    The **worktree path** comes first because it names the tree's creator, which
    is the sharper answer where it exists. It cannot answer for the primary
    checkout: `worktree.py` exempts it deliberately, since the owner's own tree
    carries no session id and reading one out of it would attribute that tree to
    whichever agent looked.

    The **harness session records** answer where the path cannot. They are less
    specific — they name every session that *started* in the directory, not the
    one editing it — but they come from files the harness named and wrote, so
    they are a derivation rather than a self-report, and they are the only thing
    that gives the primary checkout an identity at all.
    """
    if override:
        # A declaration stays a declaration. But where the harness recorded a
        # candidate set, it can be checked against one, and "corroborated by a
        # record the agent did not write" is materially better evidence than
        # "typed on a command line". Non-membership is reported, never refused:
        # sessions already publish under names of their own choosing and
        # rejecting those would lose findings rather than improve them.
        known = S.candidates(Path.cwd())
        if known and override in known:
            return override, ("declared on the command line, corroborated by a "
                              "harness session record for this directory")
        if known:
            return override, ("declared on the command line; NOT among the "
                              f"{len(known)} session(s) the harness records for "
                              f"this directory, so nothing corroborates it")
        return override, "self-reported on the command line"
    here = Path.cwd().resolve()
    best: Optional[tuple[int, str, str]] = None
    for tree in mesh.worktrees:
        try:
            root = Path(tree.path).resolve()
        except OSError:
            continue
        if here == root or root in here.parents:
            # Deepest match wins: the primary checkout contains every nested
            # worktree, so a shallower match is not the caller's worktree.
            depth = len(root.parts)
            if tree.session != W.UNATTRIBUTED and (best is None or depth > best[0]):
                best = (depth, tree.session, tree.session_evidence)
    if best is not None:
        return best[1], best[2]
    return S.identify(here)


def _require_session(session: str) -> Optional[int]:
    if session != W.UNATTRIBUTED:
        return None
    known = S.candidates(Path.cwd())
    print("Cannot tell which session this is.\n"
          "  No path convention around this directory carries a session id, and\n"
          "  the harness records did not resolve to exactly one session either.",
          file=sys.stderr)
    if known:
        # The whole point of printing these: --session stops being free text and
        # becomes a selection from a list the agent did not author.
        print(f"\n  {len(known)} session(s) are live in this directory, "
              f"according to\n"
              f"  records the harness wrote rather than anything they claim:",
              file=sys.stderr)
        for candidate in known:
            print(f"    --session {candidate}", file=sys.stderr)
        print("\n  Selecting one of those is still a declaration, and is still\n"
              "  labelled one - but it is corroborated, which invented text is "
              "not.", file=sys.stderr)
    else:
        print("\n  Pass --session <id> to declare one; it will be recorded as\n"
              "  self-reported, which is weaker evidence and is labelled as such\n"
              "  wherever it is read.", file=sys.stderr)
    return 2


def _limits(bus) -> None:
    print()
    print("WHAT THIS CANNOT TELL YOU")
    for limit in bus.limits:
        print(f"  - {limit}")


def _warn_if_unmapped(space) -> None:
    """Say so when the repository has no coordinates, rather than showing none.

    An empty space and a quiet fleet render identically — every area missing,
    nothing contested — and only one of them means "nobody is working here".
    """
    if not space.empty:
        return
    print()
    print("NO COORDINATE SPACE. Every path in this repository is UNMAPPED, so")
    print("nothing can be placed, claimed, or routed. That is the absence of a")
    print("vocabulary, not an idle fleet.")
    print(f"  evidence: {space.evidence}")
    print(f"  fix: declare one in {A.CONFIG_PATH}, or run this inside a git")
    print("       checkout whose tracked directories can be discovered.")


def _cmd_areas(args, mesh, bus, session, evidence, space) -> int:
    """Print the coordinate space in force, and where it came from.

    Exists because every other command's output is only as meaningful as this,
    and until now there was no way to ask. A repository whose paths all read
    UNMAPPED looks exactly like a repository nobody is working in.
    """
    print(f"Coordinate space for {mesh.repo_root}")
    print(f"  {space.evidence}")
    print(f"  area space version {A.AREA_SPACE_VERSION}, "
          f"discipline space version {D.DISCIPLINE_SPACE_VERSION}")
    declared = A.config_path(mesh.repo_root)
    print(f"  config  {declared if declared else '(none; discovered)'}")
    print()
    if space.empty:
        _warn_if_unmapped(space)
        return 0
    print(f"  {'PILLAR':<24} {'PREFIX':<34} SURFACE DEPTH")
    for rule in space.rules:
        depth = "flat (the file is the unit)" if rule.depth == A.FLAT \
            else str(rule.depth)
        print(f"  {rule.pillar:<24} {rule.prefix:<34} {depth}")
    tier3 = sorted(space.tier3_pillars)
    if tier3:
        print()
        print(f"  Tier 3 pillars (withheld from open-areas): {', '.join(tier3)}")
    if space.tier3_areas:
        print(f"  Tier 3 areas: "
              f"{', '.join(f'{p}/{s}' for p, s in sorted(space.tier3_areas))}")

    disciplines = D.load(mesh.repo_root)
    print()
    print(f"  {disciplines.evidence}")
    for discipline in disciplines.disciplines:
        mark = "exact" if discipline.exact else "at least"
        print(f"    {discipline.id:<18} {discipline.citation}  [{mark}]")
    return 0


def _cmd_status(args, mesh, bus, session, evidence, space) -> int:
    print(f"Fleet over {mesh.repo_root}")
    print(f"  you are {session}   ({evidence})")
    print(f"  {len(mesh.worktrees)} worktrees, "
          f"{len(mesh.agent_worktrees)} from agents")
    print()
    states = bus.survey(mesh)
    if not states:
        print("  nothing is in flight anywhere the mesh can see")
    else:
        print(f"  {'AREA':<34} {'WORKING':<10} {'CLAIMED':<10} FINDINGS")
        for state in states:
            working = str(len(state.working)) if state.working else (
                f"{len(state.worktrees)}?" if state.worktrees else "-")
            claimed = str(len(state.claimed_by)) if state.claimed_by else "-"
            flag = "  << CONTESTED" if state.contested else ""
            print(f"  {str(state.area):<34} {working:<10} {claimed:<10} "
                  f"{state.open_findings}{flag}")
        print()
        print("  WORKING counts sessions with outstanding edits (derived).")
        print("  A count with '?' is worktrees whose session could not be "
              "derived.")
        print("  CLAIMED counts sessions that said so (self-reported).")

    contested = bus.contested()
    if contested:
        print()
        print("CONTESTED - more than one session holds these. A claim is not a lock.")
        for area in contested:
            holders = [c.session_id for c in bus.active_claims() if c.area == area]
            print(f"  {area}: {', '.join(sorted(set(holders)))}")

    orphans = bus.undelivered(limit=10)
    if orphans:
        print()
        print("PUBLISHED BUT REACHED NOBODY")
        for finding in orphans:
            print(f"  {finding.id}  {finding.kind:<20} {finding.body[:60]}")
    _warn_if_unmapped(space)
    _limits(bus)
    return 0


def _cmd_inbox(args, mesh, bus, session, evidence, space) -> int:
    failure = _require_session(session)
    if failure:
        return failure
    deliveries = bus.inbox(session, include_acknowledged=args.all,
                           max_age_days=args.max_age_days)
    print(f"Inbox for {session}  ({evidence})")
    if not deliveries:
        print("  nothing. That means nobody addressed you, not that nobody "
              "is working near you - run `status` for that.")
        _warn_if_unmapped(space)
        _limits(bus)
        return 0
    for delivery in deliveries:
        finding = delivery.finding
        mark = "  " if not delivery.acknowledged else "* "
        print()
        print(f"{mark}{finding.id}  [{finding.kind}]  {finding.severity}")
        print(f"    from    {finding.from_session}  ({finding.from_evidence})")
        print(f"    reached {delivery.reason}  [{delivery.provenance}]")
        if finding.subject_paths:
            shown = ", ".join(finding.subject_paths[:4])
            more = (f" (+{len(finding.subject_paths) - 4} more)"
                    if len(finding.subject_paths) > 4 else "")
            print(f"    about   {shown}{more}")
        print(f"    says    {finding.body}")
    print()
    print("The 'says' line is the publisher's own words. Nothing checked it.")
    _limits(bus)
    return 0


def _cmd_publish(args, mesh, bus, session, evidence, space) -> int:
    failure = _require_session(session)
    if failure:
        return failure
    # An unresolvable --to-area is the same trap as an unresolvable claim, and
    # worse here: publish already prints REACHED NOBODY for a legitimate empty
    # audience, so a typo is indistinguishable from "nobody is working there".
    if args.to_area and _require_known_area(args.to_area, space) is None:
        return 2
    try:
        finding, deliveries = bus.publish(
            kind=args.kind, body=args.body, from_session=session,
            from_evidence=evidence, mesh=mesh, subject_paths=args.about or (),
            to_session=args.to_session or "", to_area=args.to_area or "",
            broadcast=args.broadcast, severity=args.severity,
            cache=C.WorktreeCache(args.database or C.default_database()))
    except ValueError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    print(f"Published {finding.id}  [{finding.kind}]")
    areas = finding.areas_in(space)
    if areas:
        print(f"  areas   {', '.join(str(a) for a in areas)}")
    if args.about:
        # WHERE the finding is (`areas`) and WHAT KIND of rule governs it are
        # different questions, and only the first has ever been answered here.
        print(f"  needs   {D.describe(args.about, D.load(mesh.repo_root))}")
    if not deliveries:
        print("  REACHED NOBODY. The finding is recorded, but no session the "
              "mesh can see is working on this.")
        print("  That is a real outcome, not a failure - do not assume the "
              "fleet has been warned.")
    else:
        print(f"  reached {len(deliveries)} session(s):")
        for delivery in deliveries:
            print(f"    {delivery.to_session}  [{delivery.provenance}]")
            print(f"      because {delivery.reason}")
    _warn_if_unmapped(space)
    _limits(bus)
    return 0


def _cmd_disciplines(args, mesh, bus, session, evidence, space) -> int:
    """Which specialist rules govern what this session is touching.

    The paths come from the mesh's own `touched_paths` rather than from a fresh
    `git status` parse, so this reuses the reader that already handles renames,
    quoting and the porcelain column layout instead of growing a second one.
    """
    paths = list(args.paths or ())
    source = "the paths you named"
    if not paths:
        here = Path.cwd().resolve()
        for tree in mesh.worktrees:
            try:
                root = Path(tree.path).resolve()
            except OSError:
                continue
            if here == root or root in here.parents:
                paths.extend(tree.touched_paths)
                source = f"outstanding work in {tree.label}"
    if not paths:
        print("Nothing to place: no paths given and this worktree has no "
              "outstanding work.")
        return 0

    disciplines = D.load(mesh.repo_root)
    found = disciplines.among(paths)
    print(f"{len(paths)} path(s) from {source}")
    print(f"  {disciplines.evidence}")
    print()
    if not found:
        print(f"  {D.UNSPECIALISED}")
        print(f"  {disciplines.describe(paths)}")
    for discipline in found:
        mark = "exact" if discipline.exact else "at least"
        print(f"  {discipline.id:<18} {discipline.citation}  [{mark}]")
        if discipline.title:
            print(f"      {discipline.title}")
        hits = [p for p in sorted(set(paths))
                if discipline in disciplines.of(p)]
        for path in hits[:6]:
            print(f"        {path}")
        if len(hits) > 6:
            print(f"        (+{len(hits) - 6} more)")
    print()
    print("WHAT THIS CANNOT TELL YOU")
    for limit in disciplines.limits:
        print(f"  - {limit}")
    return 0


def _cmd_open_areas(args, mesh, bus, session, evidence, space) -> int:
    """Areas nobody is working in — derived from paths that really exist.

    The candidate set comes from walking tracked files, NOT from listing every
    pillar in a table. An area is only offered if the repository actually has
    code in it.
    """
    code, out = W._git("ls-files", cwd=mesh.repo_root)
    if code != 0:
        print("Could not list tracked files; cannot enumerate areas.",
              file=sys.stderr)
        return 1
    candidates = space.areas_of(out.splitlines())
    free = bus.open_areas(mesh, candidates=candidates,
                          include_tier3=args.include_tier3)
    scope = "areas"
    if args.pillar:
        # Narrow the denominator too. "22 of 166" after a filter reads as a
        # share of the whole repository when it is a share of one pillar.
        free = tuple(a for a in free if a.pillar == args.pillar)
        candidates = tuple(a for a in candidates if a.pillar == args.pillar)
        scope = f"{args.pillar} areas"
    print(f"{len(free)} of {len(candidates)} {scope} have no session on them")
    print()
    # Grouped by pillar: an ungrouped list of a hundred-odd lines is a dump, and
    # the question this answers — "where could I work?" — is asked one pillar at
    # a time.
    by_pillar: dict = {}
    for area in free:
        by_pillar.setdefault(area.pillar, []).append(area)
    for pillar in sorted(by_pillar):
        surfaces = [a.surface or "(root)" for a in sorted(by_pillar[pillar])]
        print(f"  {pillar}  ({len(surfaces)})")
        print(f"      {', '.join(surfaces)}")
    skipped = [a for a in candidates if space.tier3(a)]
    if skipped and not args.include_tier3:
        print()
        print(f"  {len(skipped)} area(s) withheld as needing owner authority: "
              f"{', '.join(sorted({str(a) for a in skipped}))}")
        print("  These were declared capital, destructive, trust or release "
              "authority by")
        print(f"  this repository in {A.CONFIG_PATH}. They need explicit owner "
              f"authority,")
        print("  not a free slot. --include-tier3 lists them.")
    elif not space.tier3_pillars and not space.tier3_areas:
        print()
        print("  This repository declares NO areas as needing owner authority,")
        print("  so none were withheld. That is an absent declaration, not a")
        print("  finding that every area is safe to start in.")
    print()
    print("An area being free means no session the mesh can SEE is on it.")
    print("It does not mean the work there is wanted, safe, or ready to start.")
    _warn_if_unmapped(space)
    _limits(bus)
    return 0


def _require_known_area(text: str, space):
    """Parse `text` into an area a path can actually resolve to, or explain.

    `parse_area` accepts anything — it partitions on `/` and returns what it was
    handed — so an unchecked claim could land on a pillar no file reaches. The
    session then holds a coordinate nobody can route to, and the tool says
    "Claimed". That happened: `platform.gateway` instead of `platform/gateway`,
    and four findings at the real area reached nobody while their author
    believed the territory was announced.

    Returns None after printing the reason, so callers `return 2`.
    """
    area = A.parse_area(text)
    if space.known(area):
        return area

    print(f"Refused: {text!r} is not an area of this repository.",
          file=sys.stderr)
    if space.empty:
        print("  This repository has NO coordinate space, so no area resolves.",
              file=sys.stderr)
        print(f"  {space.evidence}", file=sys.stderr)
        print(f"  Declare one in {A.CONFIG_PATH}, or run inside a git checkout.",
              file=sys.stderr)
        return None
    suggestion = space.suggest(text)
    if suggestion is not None:
        print(f"  Did you mean {suggestion}?", file=sys.stderr)
        print("  A pillar may contain a dot (runtime.common); a pillar and its "
              "surface are joined by a slash (platform/gateway).",
              file=sys.stderr)
    else:
        print("  Run `open-areas` for the vocabulary, or `areas` for the rules.",
              file=sys.stderr)
    print("  Refused rather than recorded: a claim on an area nothing resolves "
          "to is invisible to routing, and reads as success.", file=sys.stderr)
    return None


def _cmd_claim(args, mesh, bus, session, evidence, space) -> int:
    failure = _require_session(session)
    if failure:
        return failure
    area = _require_known_area(args.area, space)
    if area is None:
        return 2
    bus.claim(area, session, note=args.note or "")
    holders = sorted({c.session_id for c in bus.active_claims()
                      if c.area == str(area)})
    print(f"Claimed {area} for {session}")
    others = [h for h in holders if h != session]
    if others:
        print(f"  ALSO HELD BY {', '.join(others)}. A claim is not a lock and "
              f"this was not refused.")
        print("  Both holds are recorded. Reconcile it with them, not with this "
              "tool.")
    if space.tier3(area):
        print(f"  NOTE: this repository declares {area} as needing explicit "
              f"owner authority.")
        print("  The claim was recorded; the authority is not something this "
              "tool can grant.")
    _limits(bus)
    return 0


def _cmd_release(args, mesh, bus, session, evidence, space) -> int:
    failure = _require_session(session)
    if failure:
        return failure
    released = bus.release(A.parse_area(args.area), session)
    print(f"{'Released' if released else 'No active claim to release on'} "
          f"{args.area}")
    return 0


def _cmd_ack(args, mesh, bus, session, evidence, space) -> int:
    failure = _require_session(session)
    if failure:
        return failure
    done = bus.acknowledge(args.finding_id, session)
    print(f"{'Acknowledged' if done else 'Nothing to acknowledge for'} "
          f"{args.finding_id}")
    return 0


_COMMANDS = {
    "status": _cmd_status, "inbox": _cmd_inbox, "publish": _cmd_publish,
    "open-areas": _cmd_open_areas, "claim": _cmd_claim,
    "release": _cmd_release, "ack": _cmd_ack,
    "disciplines": _cmd_disciplines, "areas": _cmd_areas,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alelyon-fleet",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=".",
                        help="the repository to observe (default: here). Its "
                             "worktrees, its coordinate space, its tracked "
                             "paths -- nothing is read from anywhere else")
    parser.add_argument("--mainline", default="origin/main")
    parser.add_argument("--session", default="",
                        help="declare a session id when none can be derived; "
                             "recorded as self-reported")
    parser.add_argument("--database", default="", help="override the store")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="who is working where")
    sub.add_parser("areas", help="the coordinate space in force, and its source")

    inbox = sub.add_parser("inbox", help="findings addressed to this session")
    inbox.add_argument("--all", action="store_true",
                       help="include ones already acknowledged")
    inbox.add_argument("--max-age-days", type=float,
                       default=B.DEFAULT_INBOX_AGE_DAYS)

    publish = sub.add_parser("publish", help="tell the fleet something")
    publish.add_argument("--kind", required=True, choices=list(B.KINDS))
    publish.add_argument("--body", required=True)
    publish.add_argument("--about", action="append", metavar="PATH",
                         help="repository-relative path this is about; repeatable. "
                              "Routing is derived from these, so they decide who "
                              "hears it")
    publish.add_argument("--to-area", default="",
                         help="address an area instead of paths")
    publish.add_argument("--to-session", default="",
                         help="address one session explicitly (recorded DECLARED)")
    publish.add_argument("--broadcast", action="store_true",
                         help="every live session; use sparingly")
    publish.add_argument("--severity", default="info",
                         choices=["info", "warn", "urgent"])

    free = sub.add_parser("open-areas", help="areas with no session on them")
    free.add_argument("--include-tier3", action="store_true",
                      help="also list areas the repository declared as needing "
                           "owner authority")
    free.add_argument("--pillar", default="",
                      help="narrow to one pillar")

    claim = sub.add_parser("claim", help="take an advisory hold on an area")
    claim.add_argument("area")
    claim.add_argument("--note", default="")

    release = sub.add_parser("release", help="drop a hold")
    release.add_argument("area")

    ack = sub.add_parser("ack", help="mark a finding read")
    ack.add_argument("finding_id")

    disciplines = sub.add_parser(
        "disciplines", help="which specialist rules govern what you touched")
    disciplines.add_argument("--paths", nargs="+", default=None,
                             help="place these paths instead of this "
                                  "worktree's outstanding work")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    mesh = W.observe(args.repo, mainline=args.mainline)
    # The space belongs to the repository under observation, resolved from
    # `--repo`. Falling back to the process default here would place another
    # repository's paths with this one's rules the moment the two differ.
    space = A.load(mesh.repo_root)
    bus = B.FleetBus(args.database or B.default_database(), space=space)
    session, evidence = _identify(mesh, args.session)
    return _COMMANDS[args.command](args, mesh, bus, session, evidence, space)


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
