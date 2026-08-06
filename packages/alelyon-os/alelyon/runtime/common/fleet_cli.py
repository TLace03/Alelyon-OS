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
    alelyon-fleet supply                       # where the work is STUCK

Read-only with respect to git — every git call is a query. It writes to one
SQLite file.

`status` answers *who is where* and `supply` answers *where the work has piled
up* — the fleet read as a production line, with the constraint marked, the areas
several documents run through, and the joins no record here can supply. It is
the expensive one: it curates the Markdown corpus and expands every queued item,
so it is asked for rather than folded into `status`.

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
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from alelyon.runtime.common import cli_flags as CLI
from alelyon.runtime.common import session_records as S
from alelyon.runtime.common import worktree as W
from alelyon.runtime.common import actor as ACT
from alelyon.runtime.common import worktree_areas as A
from alelyon.runtime.common import worktree_bus as B
from alelyon.runtime.common import worktree_cache as C
from alelyon.runtime.common import worktree_disciplines as D
from alelyon.runtime.common import worktree_launch as LAUNCH
from alelyon.runtime.common import worktree_resume as RES


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

    # Rows written before the bus was anchored on the repository rather than on
    # a worktree. Reported here because their AUTHORS believed they had reached
    # the fleet -- `publish` told them so -- and this is the only place that
    # says otherwise.
    try:
        stranded = C.stranded_buses(mesh.repo_root)
    except Exception:                                          # noqa: BLE001
        stranded = ()
    if stranded:
        findings = sum(f for _p, f, _c in stranded)
        claims = sum(c for _p, _f, c in stranded)
        print()
        print(f"STRANDED - {findings} finding(s) and {claims} claim(s) sit in "
              f"{len(stranded)} per-worktree database(s)")
        print("  that predate the repository-wide bus. Nobody but their author "
              "ever read them.")
        print("  Not merged: a closed finding folded back in would return as "
              "live work.")
        for path, found, held in stranded[:5]:
            print(f"    {found:>3} finding(s) {held:>3} claim(s)  {path}")
        if len(stranded) > 5:
            print(f"    ... and {len(stranded) - 5} more")

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
    if args.to_area and _require_known_area(
            args.to_area, space, mesh.repo_root) is None:
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


def _cmd_supply(args, mesh, bus, session, evidence, space) -> int:
    """The fleet read as a production system: the line, and where it is stuck.

    Everything the Work Supply Chain view draws, in a terminal. It exists here
    rather than only in the window for the same reason the rest of this module
    does: the fleet subsystems ship in the `alelyon-os` wheel, and a picture that
    can only be seen inside one desktop application is not a shipped capability.

    Read-only, and expensive by this CLI's standards — it curates the Markdown
    corpus and expands every queued item — so it is a command a reader asks for
    rather than part of `status`.
    """
    from alelyon.runtime.common import blueprint as BP
    from alelyon.runtime.common import blueprint_focus as BF
    from alelyon.runtime.common import job_path as JP
    from alelyon.runtime.common import session_supply as SS

    blueprint = None
    plans: tuple = ()
    if not args.no_corpus:
        corpus = BP.read_corpus(mesh.repo_root)
        blueprint = BF.curate(corpus, mesh=mesh)
        evidence_read = JP.gather(mesh.repo_root, mesh=mesh,
                                  database=args.database or None)
        plans = tuple(JP.plan(entry.document, evidence_read, ready_only=True)
                      for entry in blueprint.focus + blueprint.deferred)

    activity = None
    if not args.no_chain:
        try:
            from alelyon.runtime.common import session_activity as SA
            activity = SA.read_activity(mesh.repo_root)
        except Exception as exc:  # noqa: BLE001 - a missing records root is normal
            print(f"  (the orchestration chain could not be read: "
                  f"{type(exc).__name__}: {exc})", file=sys.stderr)

    chain = SS.build(mesh=mesh, activity=activity, blueprint=blueprint,
                     plans=plans, claims=bus.active_claims())

    print(f"Work supply chain over {mesh.repo_root}")
    print(f"  {chain.headline}")
    print()

    print("  THE LINE")
    print("  STATION               IN PROCESS   PASSED   UNMEASURED")
    neck = chain.bottleneck
    for station in chain.stations:
        mark = "  << CONSTRAINT" if neck is not None and station.key == neck.key \
            else ""
        print(f"  {station.title:<20} {station.wip:>10}   {station.passed:>6}   "
              f"{station.unmeasured:>10}{mark}")
    print()
    if neck is not None:
        print(f"  The constraint is where work has ACCUMULATED, which is a "
              f"weaker statement")
        print(f"  than 'this station is slowest'. Nothing records a job "
              f"completing, so")
        print(f"  throughput and cycle time are UNMEASURED and only queue depth "
              f"is observed.")
        print()

    risky = [c for c in chain.chokepoints if c.risk != "ORDINARY"]
    print(f"  {len(risky)} AREA(S) CARRYING SUPPLY RISK")
    print("  RISK           AREA                           IN IT  DOCS  JOBS")
    for point in risky[:args.limit]:
        tier3 = " *" if point.tier3 else ""
        print(f"  {point.risk:<14} {point.area:<30} {len(point.sessions):>5}  "
              f"{len(point.documents):>4}  {point.jobs:>4}{tier3}")
    if len(risky) > args.limit:
        print(f"      (+{len(risky) - args.limit} more; --limit to see them)")
    if any(c.tier3 for c in risky[:args.limit]):
        print("  * Tier 3 -- capital, destructive, trust or release authority "
              "(AGENTS.md 3).")
    print()

    print("  CONCENTRATION")
    print(f"      Herfindahl {chain.concentration:.2f} over "
          f"{chain.observed_sessions} observed session(s)")
    print("      Sum of squared shares of outstanding paths per session. 1.00 "
          "is one holder.")
    print("      One session holding everything and one session being the ONLY "
          "ONE VISIBLE")
    print("      produce the same number, which is why the observed count is "
          "printed with it.")
    print()

    if chain.idle_areas:
        print(f"  {len(chain.idle_areas)} AREA(S) THE PLAN NAMES THAT NOBODY "
              f"IS IN")
        print(f"      {', '.join(chain.idle_areas[:16])}")
        if len(chain.idle_areas) > 16:
            print(f"      (+{len(chain.idle_areas) - 16} more)")
        print()

    print("  SEVERED -- a join no record in this repository can supply")
    for what, why in chain.severed:
        print(f"    {what}")
        for line in _wrap(why, 72):
            print(f"        {line}")
    print()
    for note in chain.notes:
        print(f"  NOTE: {note}")
    if chain.notes:
        print()
    print("WHAT THIS CANNOT TELL YOU")
    for limit in chain.limits:
        for index, line in enumerate(_wrap(limit, 74)):
            print(f"  {'- ' if index == 0 else '  '}{line}")
    return 0


def _wrap(text: str, width: int) -> list:
    """Fold one sentence to a width. No dependency, and no reflow of newlines."""
    words = str(text).split()
    lines: list = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [""]


def _require_known_area(text: str, space, repo_root: Optional[str] = None):
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
        # From the repository the space describes, never the current directory:
        # `--repo` selects what is observed, and reading surfaces from elsewhere
        # would check one repository's coordinate against another's layout.
        surfaces = A.observed_surfaces(repo_root, space=space)
        if space.derivable(area, surfaces):
            return area
        # The pillar is real and the surface is imagined. Same failure as
        # below, one rung down, and it reads as success just as loudly.
        print(f"Refused: {text!r} names a real pillar and a surface no path "
              f"in this repository derives.", file=sys.stderr)
        fallback = A.Area(area.pillar, "")
        if space.derivable(fallback, surfaces):
            print(f"  Did you mean {fallback}?", file=sys.stderr)
            # ASCII only: this prints to a terminal, and a Windows console runs
            # cp1252 by default. See session_supply.SEVERED for the incident.
            print(f"  Every file directly under that pillar derives "
                  f"{fallback}. The surface is taken from a DIRECTORY, and "
                  f"there is none here. `tools/` is FLAT, where the file is "
                  f"the unit and `tools/relay` is derivable; this pillar is "
                  f"not.", file=sys.stderr)
        else:
            print("  Run `open-areas` for the vocabulary, or `areas` for the "
                  "rules.", file=sys.stderr)
        print("  Refused rather than recorded: a claim on an area nothing "
              "resolves to is invisible to routing, and reads as success.",
              file=sys.stderr)
        return None

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
    area = _require_known_area(args.area, space, mesh.repo_root)
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


def _cmd_resume(args, mesh, bus, session, evidence, space) -> int:
    """What is dormant, and -- only when explicitly authorised -- wake it.

    The default is a READING. `--wake` without `--authorise` still starts
    nothing: it shows what would be admitted and what would be refused, which is
    the artifact worth looking at before an agent runs in a directory full of
    somebody else's uncommitted work.
    """
    live = _live_sessions(mesh)
    candidates = RES.survey(mesh.repo_root, mesh=mesh, bus=bus, space=space,
                            live_sessions=live,
                            older_than_days=args.older_than_days)
    sleeping = RES.dormant(candidates)
    print(f"Dormant worktrees over {mesh.repo_root}")
    print(f"  {len(sleeping)} of {len(candidates)} worktree(s) read as DORMANT "
          f"at {args.older_than_days:g} day(s)")

    config = LAUNCH.read_config(mesh.repo_root)
    if config is None:
        print(f"  NO AGENT COMMAND DECLARED. Add a [launch] command to "
              f"{LAUNCH.CONFIG_PATH} before anything can be woken; this "
              f"repository refuses to guess one.")

    if not sleeping:
        print()
        print("  Nothing is dormant. That is a reading of COMMIT AGE, not of "
              "whether anybody is in these directories.")
        _print_limits(RES.LIMITS)
        return 0

    chosen = _select(sleeping, args.wake)
    if args.wake and not chosen:
        print()
        print(f"  No dormant worktree matches {', '.join(args.wake)}.")
        return 1
    plan = RES.plan(chosen or sleeping, repo_root=mesh.repo_root)

    # The authority is constructed HERE, from an explicit flag, and nowhere
    # else. `--authorise` is the whole difference between a report and a spend.
    authority = None
    if args.authorise:
        authority = LAUNCH.Authority(
            granted_by=session or "an unidentified session",
            granted_at=time.time(),
            covers=tuple(one.path for one in plan.selected),
            note="declared on the alelyon-fleet command line")

    print()
    for candidate in plan.selected:
        verdict = LAUNCH.admit(
            candidate, plan=plan, authority=authority, live_sessions=live,
            space=space, model=args.model, max_in_flight=args.max_in_flight,
            configured=config is not None,
            in_flight=LAUNCH.reserved(mesh.repo_root))
        mark = "WOULD WAKE" if verdict.accepted else "refused"
        print(f"  {mark:<11} {candidate.label}")
        print(f"     {candidate.dormancy_evidence}")
        if candidate.document:
            print(f"     queued: {candidate.document}")
        if not verdict.accepted:
            print(f"     {verdict.reason}: {verdict.detail}")

    if not args.authorise:
        print()
        print("  NOTHING WAS STARTED. Waking an agent is Tier 3 process "
              "control and spends the owner's API budget, so it needs "
              "--authorise, which only the owner may give.")
        _print_limits(RES.LIMITS + LAUNCH.LAUNCH_LIMITS)
        return 0

    report = LAUNCH.launch(
        plan, spawn=LAUNCH.detached_spawner(config, repo_root=mesh.repo_root),
        authority=authority, model_for=lambda c: args.model,
        live_sessions=live, space=space, max_in_flight=args.max_in_flight,
        configured=config is not None)
    print()
    for request, handle in report.started:
        print(f"  STARTED pid {handle.pid}  {request.candidate.label}")
        print(f"     {' '.join(handle.argv)}")
    for verdict in report.refused:
        label = verdict.candidate.label if verdict.candidate else "?"
        print(f"  refused {label}: {verdict.reason}")
    if report.started:
        print()
        print("  A STARTED PROCESS IS NOT A WORKING AGENT. These are detached "
              "and outlive this command; nothing here can stop them.")
    _print_limits(LAUNCH.LAUNCH_LIMITS)
    return 0


def _select(candidates, labels):
    """The dormant worktrees whose label or directory name matches."""
    if not labels:
        return ()
    wanted = {one.strip().lower() for one in labels if one.strip()}
    picked = []
    for one in candidates:
        tail = one.path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if one.label.lower() in wanted or tail.lower() in wanted:
            picked.append(one)
    return tuple(picked)


def _live_sessions(mesh) -> tuple:
    """Session ids the harness records as live anywhere in this mesh.

    Asked per DIRECTORY, because that is the question `session_records` can
    answer -- a record names the cwd a session started in. The union over the
    primary checkout and every worktree is what `worktree_resume` wants, since
    it tests membership of one worktree's session in this set.

    Records the agent did not write, and the only signal allowed to contradict
    commit age. A records root that cannot be read yields NO live sessions,
    which reads as "nobody is there". That is the unsafe direction, so it is
    tolerable only because every other refusal still applies -- this must never
    be the sole thing between a wake and an occupied directory.
    """
    found = set()
    directories = [mesh.repo_root] + [tree.path for tree in mesh.worktrees]
    for directory in directories:
        try:
            for record in S.sessions_in(directory):
                if record.session_id:
                    found.add(record.session_id)
        except Exception:                # noqa: BLE001 - containment boundary
            continue
    return tuple(sorted(found))


def _print_limits(limits) -> None:
    print()
    print("WHAT THIS CANNOT TELL YOU")
    for line in limits:
        print(f"  - {line}")


def _cmd_waiting(args, mesh, bus, session, evidence, space) -> int:
    """Which sessions have stopped and are waiting for the owner.

    The terminal reading of the Lattice `Waiting on You` view. It exists
    because the person this answers for is usually in a terminal, and opening
    a desktop window to find out which terminal to go back to is a poor trade.
    """
    from alelyon.runtime.common import session_attention as ATT
    board = ATT.read_board(mesh.repo_root, max_sessions=args.max_sessions)

    if not board.sessions:
        print("No session transcripts could be read for this repository.")
        for note in board.notes:
            print(f"  {note}")
        print("  This is a missing reading, not a quiet fleet.")
        return 0

    # The state is DERIVED and the blocker is DECLARED. They are printed on
    # separate lines, and the blocker names its own provenance, because a reader
    # who reads "BLOCKED" as an observation will trust a stale self-report.
    stuck = ATT.blockers(bus)

    queue = board.needs_owner
    if not queue:
        print(f"Nothing is waiting on you. {len(board.working)} session(s) "
              f"working, {len(board.stalled)} with a call outstanding, "
              f"{len(board.dormant)} dormant.")
    else:
        live = [e for e in queue
                if e.session_id in stuck and not stuck[e.session_id].stale]
        headline = f"{len(queue)} session(s) are waiting on you, longest first"
        if live:
            headline += (f" -- {len(live)} of them published a blocker they "
                         f"have not spoken since")
        print(headline + ":")
        print()
        for entry in queue:
            mark = "ASKING " if entry.state == ATT.ASKING else "waiting"
            print(f"  {mark}  {entry.short_id}  {entry.waited:>7}  "
                  f"{entry.branch or '-'}")
            if entry.summary:
                print(f"           {entry.summary[:96]}")
            block = stuck.get(entry.session_id)
            if block is not None and not block.stale:
                print(f"           BLOCKED (its own words, {int(block.age_minutes)}m "
                      f"ago): {block.summary[:88]}")
        print()

    if board.stalled:
        print(f"{len(board.stalled)} session(s) have a tool call outstanding. "
              f"That is a slow command, a permission prompt waiting for a "
              f"click, or a dead session -- this cannot tell which:")
        for entry in board.stalled:
            print(f"  {entry.short_id}  {entry.waited:>7}  "
                  f"{entry.last_tool or '-'}")
        print()

    if args.verbose:
        for entry in board.working:
            print(f"  working  {entry.short_id}  {entry.waited:>7}  "
                  f"{entry.last_tool}")
        print()
    for note in board.notes:
        print(f"  NOTE: {note}")
    _print_limits(board.limits)
    return 0


def resolve_actor(session: str, evidence: str, *, cwd=None) -> ACT.Actor:
    """`_identify`'s answer, graded as a value rather than as prose.

    The same session and the same sentence every other command already uses --
    this adds no derivation and overrides none. What it adds is the two things
    the prose could not carry to a caller: which GRADE the evidence earns, and
    whether the attribution is AMBIGUOUS.

    Ambiguity is read from the harness's own record of how many sessions are
    live in this directory. More than one means the derivation collapses: they
    share a path, so they share an identity, and `worktree_bus` already records
    that as one identity rather than two. That is not a hypothetical -- three
    sessions were in `famMain` at once on 2026-08-06 and one published a finding
    under the wrong author.
    """
    here = Path(cwd) if cwd else Path.cwd()
    try:
        live = S.candidates(here)
    except OSError:
        live = ()
    shared = len(live) > 1

    if evidence.startswith("declared on the command line, corroborated"):
        actor = ACT.DeclaredOnCommandLine(session, corroborated=True,
                                          evidence=evidence).current()
    elif evidence.startswith("declared on the command line") or \
            evidence.startswith("self-reported"):
        actor = ACT.DeclaredOnCommandLine(session, corroborated=False,
                                          evidence=evidence).current()
    else:
        actor = ACT.DerivedFromPath(session, evidence=evidence,
                                    shared_checkout=shared).current()
        if shared and not actor.attributed:
            # The derivation did not fail to find anybody -- it found several
            # and refused to choose. "Nobody works here" and "too many people
            # work here to say which" both arrive as UNATTRIBUTED, and only the
            # second is a fact about contention. Marking it keeps them apart.
            actor = replace(actor, ambiguous=True)
    # A DECLARED or CORROBORATED identity is a name the writer chose, so a
    # shared checkout does not make it ambiguous -- two people typing two
    # different names are distinguishable, however weakly. Only a DERIVATION
    # collapses, because the path is the same for both.
    return actor


def _cmd_whoami(args, mesh, bus, session, evidence, space) -> int:
    """Who this command would publish AS, and whether that is good enough.

    Exists because the answer was already computable and never shown until
    something had been written under it. A session that discovers its own
    attribution by reading a finding it published under the wrong name has
    discovered it too late.
    """
    actor = resolve_actor(session, evidence)
    minimum = ACT.Assurance[args.at_least]

    print(f"Actor over {mesh.repo_root}")
    print(f"  id           {actor.id}")
    print(f"  kind         {actor.kind.name}")
    print(f"  organization {actor.organization}")
    print(f"  assurance    {actor.assurance.name}")
    print(f"  evidence     {actor.evidence}")
    print(f"  ambiguous    {'YES' if actor.ambiguous else 'no'}")

    if actor.may_author(minimum):
        print(f"\nMAY AUTHOR at {minimum.name}.")
    else:
        print(f"\nMAY NOT AUTHOR at {minimum.name}:")
        print(f"  {actor.refusal(minimum)}")

    try:
        live = S.candidates(Path.cwd())
    except OSError:
        live = ()
    if len(live) > 1:
        print(f"\n{len(live)} session(s) are live in this directory, according "
              f"to records the harness wrote:")
        for candidate in sorted(live):
            here = "  <- you say you are this one" if candidate == session else ""
            print(f"    {candidate}{here}")
        print("  They share one path, so a DERIVED identity cannot separate "
              "them. Passing --session is what distinguishes you, and it is a "
              "declaration: it says who you claim to be, not who you are.")

    print()
    print("WHAT THIS CANNOT TELL YOU")
    for limit in ACT.LIMITS:
        print(f"  - {limit}")
    return 0 if actor.may_author(minimum) else 1


_COMMANDS = {
    "status": _cmd_status, "inbox": _cmd_inbox, "publish": _cmd_publish,
    "open-areas": _cmd_open_areas, "claim": _cmd_claim,
    "release": _cmd_release, "ack": _cmd_ack,
    "disciplines": _cmd_disciplines, "areas": _cmd_areas,
    "supply": _cmd_supply, "resume": _cmd_resume, "waiting": _cmd_waiting,
    "whoami": _cmd_whoami,
}


def build_parser() -> argparse.ArgumentParser:
    # Accepted before AND after the subcommand. Declared on the top-level
    # parser alone, `--session` after the subcommand is `unrecognized
    # arguments`, which reads as "no such flag" rather than "wrong position".
    leading, trailing = CLI.either_side((
        (("--repo",), {"default": ".",
                       "help": "the repository to observe (default: here). "
                               "Its worktrees, its coordinate space, its "
                               "tracked paths -- nothing is read from "
                               "anywhere else"}),
        (("--mainline",), {"default": "origin/main"}),
        (("--session",), {"default": "",
                          "help": "declare a session id when none can be "
                                  "derived; recorded as self-reported"}),
        (("--database",), {"default": "", "help": "override the store"}),
    ))
    parser = argparse.ArgumentParser(
        prog="alelyon-fleet",
        description=__doc__,
        parents=[leading],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = CLI.subcommands(parser, trailing, dest="command", required=True)

    sub.add_parser("status", help="who is working where")
    whoami = sub.add_parser(
        "whoami", help="who you would publish AS, and whether that is enough")
    whoami.add_argument(
        "--at-least", default="CORROBORATED",
        choices=[level.name for level in ACT.Assurance],
        help="the assurance a durable write should require (default: "
             "CORROBORATED). Exits non-zero when the current attribution does "
             "not reach it")
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

    supply = sub.add_parser(
        "supply", help="the fleet as a production line: where work is stuck")
    supply.add_argument("--limit", type=int, default=12,
                        help="how many risk-carrying areas to list")
    supply.add_argument("--no-corpus", action="store_true",
                        help="skip the Markdown corpus and the job expansion. "
                             "Much faster, and the line is then empty -- which "
                             "is a missing reading, not an idle fleet")
    supply.add_argument("--no-chain", action="store_true",
                        help="skip the harness transcripts; no fleet or agent "
                             "then appears on the graph")

    waiting = sub.add_parser(
        "waiting", help="which sessions have stopped and are waiting for you")
    from alelyon.runtime.common.session_attention import (
        DEFAULT_MAX_SESSIONS as ATT_MAX_SESSIONS,
    )
    waiting.add_argument("--max-sessions", type=int,
                         default=ATT_MAX_SESSIONS,
                         help="how many transcripts to read, newest first. "
                              "Anything the cap hides is reported as hidden")
    waiting.add_argument("--verbose", action="store_true",
                         help="also list the sessions that are working")

    resume = sub.add_parser(
        "resume", help="what is dormant, and what waking it would involve")
    resume.add_argument("--wake", action="append", metavar="LABEL", default=None,
                        help="worktree label to wake; repeatable. Without "
                             "--authorise this still starts nothing")
    resume.add_argument("--authorise", action="store_true",
                        help="TIER 3. Actually start agents. Without it the "
                             "answer is always no-owner-authority, which is "
                             "the safe default rather than a fault")
    resume.add_argument("--model", default="",
                        help="the model to run. Never defaulted: the layer "
                             "space says what RANK work is, never which model")
    resume.add_argument("--max-in-flight", type=int,
                        default=LAUNCH.DEFAULT_MAX_IN_FLIGHT)
    resume.add_argument("--older-than-days", type=float,
                        default=RES.DORMANT_AFTER_DAYS)
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
