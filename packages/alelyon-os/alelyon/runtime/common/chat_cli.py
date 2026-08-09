"""Channels, threads and mentions over the fleet bus -- chat for agent sessions.

    alelyon-chat channels                      # the rooms, and what is unread
    alelyon-chat join runtime-common
    alelyon-chat post runtime-common "starting on the bus schema" \\
        --about alelyon/runtime/common/worktree_bus.py
    alelyon-chat read runtime-common           # the room, oldest last
    alelyon-chat reply <id> "ack, watching that file"
    alelyon-chat thread <id>                   # one conversation
    alelyon-chat dm <session> "your branch broke my suite"
    alelyon-chat unread                        # per-room counts, exit 1 if any
    alelyon-chat search "worktree_bus"

`alelyon-fleet` is the operational surface: publish a finding, claim an area,
find work. This is the conversational one. They are the same store and the same
router, and that is deliberate -- a message about `worktree_bus.py` reaches
whoever is editing `worktree_bus.py` whether it was sent as a finding or said in
a room.

**How a message finds you.** Three ways, and the difference is printed on every
delivery rather than left for you to assume:

* you JOINED the channel -- your own declaration, worth what a self-report is
  worth;
* you were @mentioned -- the author's choice, worth less;
* the repository shows you EDITING the files the message is about -- derived,
  and the only one the author could not fake.

The third is why this is not Slack. You cannot be talked about in a room you are
not in without being told, and joining a channel does not exempt you from
messages about your own files.

**Who am I?** Derived from the worktree, exactly as `alelyon-fleet` does it.
Where no convention carries a session id, `--session` is required and what you
pass is recorded as self-reported.

**What this cannot do** prints after every command. The short version: a channel
is not private, nobody is obliged to be in one, and an unread count of zero means
nobody addressed you -- not that nothing is happening near your files.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

from alelyon.runtime.common import actor as ACT
from alelyon.runtime.common import cli_flags as CLI
from alelyon.runtime.common import fleet_chat as CHAT
from alelyon.runtime.common import worktree as W
from alelyon.runtime.common import worktree_areas as A
from alelyon.runtime.common import worktree_bus as B
# Imported rather than reimplemented, underscore and all. Who a session IS
# decides what its messages are worth here -- derived identity and declared
# identity are different evidence and the CLIs must not disagree about which one
# a caller has. A second copy of this logic would drift, and the drift would show
# up as two tools attributing the same session differently.
from alelyon.runtime.common.fleet_cli import _identify

#: Every string this module prints is ASCII. A Windows console runs cp437 by
#: default and this repository is developed on Windows, so an em dash raises
#: `UnicodeEncodeError` on the way out -- see
#: `test_everything_the_cli_prints_survives_a_windows_console`.
_RULE = "-" * 72

class _LazyMesh:
    """A mesh that is observed on first use, not on startup.

    Only `post`, `reply` and `dm` need it, because only they route. Everything
    else is a read over rows the bus already wrote and must not pay for a mesh
    scan -- measured at 63s over 80 worktrees in this repository, which is not a
    price a `read` may charge. Chat is read-dominated in a way the findings bus
    never was: a findings bus is written far more often than it is read.

    Stands in for `WorktreeMesh` for the two attributes `_identify` and routing
    touch. Deliberately NOT a general proxy: anything else raises, so a future
    command that quietly needs the mesh fails loudly here rather than silently
    reintroducing the startup cost this exists to remove.
    """

    def __init__(self, repo, mainline: str) -> None:
        self._repo, self._mainline = repo, mainline
        self._mesh = None

    def resolve(self):
        if self._mesh is None:
            self._mesh = W.observe(self._repo, mainline=self._mainline)
        return self._mesh

    @property
    def observed(self) -> bool:
        return self._mesh is not None

    @property
    def worktrees(self):
        return self.resolve().worktrees

    @property
    def repo_root(self):
        return self.resolve().repo_root


def _repo_root(start) -> Path:
    """The checkout containing `start`, without observing anything.

    Walks up for a `.git` entry, accepting a FILE as readily as a directory: a
    linked worktree's `.git` is a file, and treating only directories as roots
    would resolve every agent worktree to the primary checkout and load the
    wrong coordinate space.
    """
    here = Path(start).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return here


def _identify_cheaply(override: str) -> Optional[tuple[str, str]]:
    """Identity from the path alone, or `None` when only the mesh can answer.

    `_identify` finds the deepest observed worktree containing the caller and
    reads the session id off ITS path. That is a path rule, and `session_for_path`
    applies the same one to a single path -- so walking up from here reproduces
    the answer for the ordinary case, where an agent stands in its own worktree,
    without observing eighty others to find the one it is standing in.

    Returns `None` rather than guessing when no ancestor carries a session id.
    The mesh is then observed and `_identify` decides, so the slow path still
    exists and still wins; this only skips it when it would agree.
    """
    if override:
        return None            # the override branch has corroboration logic of
                               # its own in `_identify`, and one copy is enough
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        session = W.session_for_path(str(candidate))
        if session != W.UNATTRIBUTED:
            return session, (
                "session id in this directory's path, the same derivation "
                "`alelyon-fleet` makes from the observed worktree; a directory "
                "name is chosen by the tool, so this is a derivation not proof")
    return None


def _ago(at: int, now: int) -> str:
    """Human elapsed time, ASCII only."""
    if not at:
        return "never"
    seconds = max(0, now - at)
    for size, unit in ((86_400, "d"), (3_600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def _short(session: str, width: int = 12) -> str:
    """Session ids are UUIDs. A room listing needs a handle, not a paragraph."""
    return session[:width] if len(session) > width else session


def _limits(bus) -> None:
    print("\nWHAT THIS CANNOT TELL YOU")
    print("  - A '~' after a speaker means NOTHING corroborated that id: they "
          "typed it and no record they did not author agrees. A session may "
          "post under any name, including another live session's.")
    for limit in bus.limits:
        print(f"  - {limit}")


def _message_line(finding, now: int, *, indent: str = "") -> None:
    marker = "" if finding.kind == B.KIND_MESSAGE else f"[{finding.kind}] "
    # A trailing `~` where nothing corroborated the speaker's id. A room listing
    # has no width for the evidence sentence `inbox` prints, and a bare handle
    # reads as established -- which is how a session was credited for findings
    # somebody else published under its name.
    speaker = _short(finding.from_session, 11)
    if ACT.is_uncorroborated(finding.from_evidence):
        speaker += "~"
    print(f"{indent}{speaker:>12}  {marker}{finding.body}")
    trail = f"{indent}              {finding.id}  {_ago(finding.at, now)}"
    if finding.subject_paths:
        shown = ", ".join(finding.subject_paths[:2])
        extra = (f" (+{len(finding.subject_paths) - 2} more)"
                 if len(finding.subject_paths) > 2 else "")
        trail += f"  about {shown}{extra}"
    print(trail)


# ── commands ────────────────────────────────────────────────────────────────
def _cmd_channels(args, mesh, bus, session, evidence, space) -> int:
    now = int(time.time())
    views = CHAT.channel_summaries(bus, session)
    print(f"Channels on {mesh.repo_root}")
    print(f"  you are {_short(session, 40)}  ({evidence})")
    if not views:
        print("\n  no channels yet. `post <name> \"...\"` creates one by "
              "speaking in it.")
        _limits(bus)
        return 0
    print(f"\n  {'':2} {'CHANNEL':28} {'MSGS':>5} {'UNREAD':>7} "
          f"{'MEMBERS':>8}  LAST")
    for view in views:
        mark = "*" if view.joined else " "
        badge = str(view.unread) if view.unread else "-"
        print(f"  {mark:2} {view.display:28} {view.messages:>5} {badge:>7} "
              f"{len(view.members):>8}  {_ago(view.last_at, now)}")
    print("\n  * = you have joined. Rooms you have not joined are still listed "
          "and still readable;\n    a channel is not private.")
    _limits(bus)
    return 0


def _cmd_join(args, mesh, bus, session, evidence, space) -> int:
    record = bus.join(args.channel, session)
    members = bus.members(record.channel)
    print(f"joined #{record.channel}  ({len(members)} member(s))")
    print(f"  history is readable from before you joined: "
          f"`read {record.channel}`")
    print("  joining is a self-report. It routes messages to you and asserts "
          "nothing about\n  whether you work here.")
    _limits(bus)
    return 0


def _cmd_leave(args, mesh, bus, session, evidence, space) -> int:
    left = bus.leave(args.channel, session)
    print(f"left #{B.normalise_channel(args.channel)}" if left
          else f"you were not in #{B.normalise_channel(args.channel)}")
    _limits(bus)
    return 0


def _cmd_topic(args, mesh, bus, session, evidence, space) -> int:
    bus.ensure_channel(args.channel, created_by=session)
    bus.set_topic(args.channel, args.topic)
    print(f"#{B.normalise_channel(args.channel)} topic: {args.topic}")
    _limits(bus)
    return 0


def _post(args, mesh, bus, session, evidence, space, *, channel: str,
          reply_to: str, body: str) -> int:
    """Shared by `post`, `reply` and `dm` -- one publish, one report."""
    cache = None
    try:
        from alelyon.runtime.common import worktree_cache as C
        cache = C.WorktreeCache(args.database or B.default_database())
    except Exception:  # noqa: BLE001 - a cache is an optimisation, not a gate
        cache = None
    if not mesh.observed:
        # Sending is the one thing here that must observe the repository, and it
        # takes a minute on a large mesh. Said out loud so the wait reads as work
        # rather than a hang -- and so the cost is attributed to the thing that
        # incurs it, which is derived routing, not chat.
        print("observing the repository to resolve who this reaches "
              "(derived routing; this is the slow part)...", flush=True)
    finding, deliveries = bus.publish(
        kind=args.kind, body=body, from_session=session, from_evidence=evidence,
        mesh=mesh, cache=cache, subject_paths=tuple(args.about or ()),
        channel=channel, reply_to=reply_to,
        to_session=getattr(args, "to_session", ""))
    where = f"#{finding.channel}" if finding.channel else "no channel"
    print(f"posted to {where}  id {finding.id}")

    if not deliveries:
        # The single most important line this tool prints. `fleet publish`
        # already says REACHED NOBODY out loud and this must not be quieter,
        # because chat makes the outcome ordinary: an empty room looks exactly
        # like a full one until somebody replies.
        print("\n  REACHED NOBODY.")
        print("  It is stored and readable in the channel, and no session was "
              "notified.")
        if not finding.subject_paths:
            print("  This message names no paths, so derived routing had "
                  "nothing to match on.\n  `--about <path>` reaches whoever is "
                  "editing that file, joined or not.")
    else:
        print(f"\n  reached {len(deliveries)} session(s):")
        for delivery in sorted(deliveries, key=lambda d: d.to_session):
            print(f"    {_short(delivery.to_session, 40)}  [{delivery.provenance}]")
            print(f"        {delivery.reason}")

    report = CHAT.mention_report(finding.body, deliveries)
    if not report.clean:
        print("\n  MENTIONED NOBODY: " +
              ", ".join(f"@{h}" for h in report.unmatched))
        print("  A handle that matches no session, or matches two at once, "
              "delivers to neither\n  rather than guessing. Check the id with "
              "`alelyon-fleet status`.")
    _limits(bus)
    return 0


def _cmd_post(args, mesh, bus, session, evidence, space) -> int:
    return _post(args, mesh, bus, session, evidence, space,
                 channel=args.channel, reply_to="", body=args.body)


def _cmd_reply(args, mesh, bus, session, evidence, space) -> int:
    parent = CHAT.render_thread(bus, args.finding_id)
    if parent is None:
        print(f"no message with id {args.finding_id}")
        print("  ids are printed by `read`, `thread` and `post`.")
        return 2
    return _post(args, mesh, bus, session, evidence, space,
                 channel=parent.root.channel, reply_to=parent.root.id,
                 body=args.body)


def _cmd_dm(args, mesh, bus, session, evidence, space) -> int:
    args.to_session = args.session_id
    return _post(args, mesh, bus, session, evidence, space,
                 channel="", reply_to="", body=args.body)


def _cmd_read(args, mesh, bus, session, evidence, space) -> int:
    now = int(time.time())
    name = B.normalise_channel(args.channel)
    threads = CHAT.threads_in(bus, name, limit=args.limit)
    members = bus.members(name)
    print(f"#{name}   {len(members)} member(s)")
    channel = next((c for c in bus.channels() if c.name == name), None)
    if channel is not None and channel.topic:
        print(f"  topic: {channel.topic}")
    if not threads:
        print("\n  nothing here yet.")
        _limits(bus)
        return 0
    print(_RULE)
    for view in threads:
        _message_line(view.root, now)
        for reply in view.replies:
            _message_line(reply, now, indent="    ")
        if view.replies:
            print(f"      {view.count} repl{'y' if view.count == 1 else 'ies'}")
        print()
    print(_RULE)
    print(f"  `reply <id> \"...\"` to answer. Reading a channel does NOT "
          "acknowledge anything;\n  `alelyon-fleet ack <id>` is what a "
          "publisher can see.")
    _limits(bus)
    return 0


def _cmd_thread(args, mesh, bus, session, evidence, space) -> int:
    now = int(time.time())
    view = CHAT.render_thread(bus, args.finding_id)
    if view is None:
        print(f"no message with id {args.finding_id}")
        return 2
    where = f"#{view.root.channel}" if view.root.channel else "no channel"
    print(f"thread in {where}")
    print(_RULE)
    _message_line(view.root, now)
    print()
    for reply in view.replies:
        _message_line(reply, now, indent="    ")
    if not view.replies:
        print("    no replies yet.")
    _limits(bus)
    return 0


def _cmd_unread(args, mesh, bus, session, evidence, space) -> int:
    """Cheap and exit-coded, so it can run before every task or from a hook.

    Exit 1 means "you have unread messages", not "something broke" -- the same
    contract `tools/mail.py check` established, kept identical here so a hook
    can call either without learning a second convention.
    """
    counts = CHAT.unread(bus, session)
    total = sum(counts.values())
    if not total:
        print(f"no unread for {_short(session, 40)}")
        print("  That means nobody addressed you. It does NOT mean nothing is "
              "happening near your\n  files -- `alelyon-fleet status` answers "
              "that.")
        _limits(bus)
        return 0
    print(f"{total} unread for {_short(session, 40)}  ({evidence})")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        where = f"#{name}" if name else "(no channel; findings and DMs)"
        how = f"`read {name}`" if name else "`alelyon-fleet inbox`"
        print(f"  {count:>4}  {where:34} {how}")
    _limits(bus)
    return 1


def _cmd_search(args, mesh, bus, session, evidence, space) -> int:
    now = int(time.time())
    hits = bus.search(args.text, channel=args.channel, limit=args.limit)
    scope = f" in #{B.normalise_channel(args.channel)}" if args.channel else ""
    if not hits:
        print(f"no message matching {args.text!r}{scope}")
        _limits(bus)
        return 0
    print(f"{len(hits)} match(es) for {args.text!r}{scope}")
    print(_RULE)
    for finding in hits:
        where = f"#{finding.channel}" if finding.channel else "(no channel)"
        print(f"  {where}")
        _message_line(finding, now, indent="  ")
        print()
    _limits(bus)
    return 0


_COMMANDS = {
    "channels": _cmd_channels, "join": _cmd_join, "leave": _cmd_leave,
    "topic": _cmd_topic, "post": _cmd_post, "reply": _cmd_reply,
    "dm": _cmd_dm, "read": _cmd_read, "thread": _cmd_thread,
    "unread": _cmd_unread, "search": _cmd_search,
}


def build_parser() -> argparse.ArgumentParser:
    leading, trailing = CLI.either_side((
        (("--repo",), {"default": ".",
                       "help": "the repository to observe (default: here)"}),
        (("--mainline",), {"default": "origin/main"}),
        (("--session",), {"default": "",
                          "help": "declare a session id when none can be "
                                  "derived; recorded as self-reported"}),
        (("--database",), {"default": "", "help": "override the store"}),
    ))
    parser = argparse.ArgumentParser(
        prog="alelyon-chat", description=__doc__, parents=[leading],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = CLI.subcommands(parser, trailing, dest="command", required=True)

    sub.add_parser("channels", help="the rooms, and what is unread")
    sub.add_parser("unread", help="per-room unread counts; exit 1 if any")

    join = sub.add_parser("join", help="subscribe to a channel")
    join.add_argument("channel")
    leave = sub.add_parser("leave", help="unsubscribe")
    leave.add_argument("channel")

    topic = sub.add_parser("topic", help="set what a channel is for")
    topic.add_argument("channel")
    topic.add_argument("topic")

    post = sub.add_parser("post", help="say something in a channel")
    post.add_argument("channel")
    post.add_argument("body")
    post.add_argument("--about", action="append", default=[],
                      help="a path this message is about. Repeatable. This is "
                           "what reaches sessions who never joined the channel")
    post.add_argument("--kind", default=B.KIND_MESSAGE, choices=B.KINDS,
                      help="default: message. An operational kind outranks chat "
                           "in every inbox")

    reply = sub.add_parser("reply", help="answer a message in its thread")
    reply.add_argument("finding_id")
    reply.add_argument("body")
    reply.add_argument("--about", action="append", default=[])
    reply.add_argument("--kind", default=B.KIND_MESSAGE, choices=B.KINDS)

    dm = sub.add_parser("dm", help="address one session directly")
    dm.add_argument("session_id")
    dm.add_argument("body")
    dm.add_argument("--about", action="append", default=[])
    dm.add_argument("--kind", default=B.KIND_MESSAGE, choices=B.KINDS)

    read = sub.add_parser("read", help="a channel, oldest last")
    read.add_argument("channel")
    read.add_argument("--limit", type=int, default=50)

    thread = sub.add_parser("thread", help="one conversation")
    thread.add_argument("finding_id")

    search = sub.add_parser("search", help="substring search over bodies")
    search.add_argument("text")
    search.add_argument("--channel", default="")
    search.add_argument("--limit", type=int, default=50)
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    mesh = _LazyMesh(args.repo, args.mainline)
    # Resolved from the checkout the caller points at, not from the mesh, so a
    # read never observes. `A.load` costs ~0.01s; `W.observe` costs a minute.
    space = A.load(_repo_root(args.repo))
    bus = B.FleetBus(args.database or B.default_database(), space=space)
    identity = _identify_cheaply(args.session)
    session, evidence = identity if identity else _identify(mesh, args.session)
    if session == W.UNATTRIBUTED:
        print("cannot tell which session you are, and a message needs an "
              "author.", file=sys.stderr)
        print("  pass --session <id>; it is recorded as self-reported.",
              file=sys.stderr)
        return 2
    try:
        return _COMMANDS[args.command](args, mesh, bus, session, evidence, space)
    except ValueError as exc:
        # `normalise_channel` and `publish` refuse rather than mangle. A refusal
        # is a result, not a crash, and it must read as one.
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
