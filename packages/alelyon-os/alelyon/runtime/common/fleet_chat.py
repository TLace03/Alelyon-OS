"""A Slack-shaped reading of the fleet bus: rooms, threads, unread counts.

`worktree_bus` owns the substrate — the tables, and the routing rules that decide
who a message reaches. This module owns none of that. It is the presentation
half: it assembles what a session needs to *read* a conversation, and it is kept
separate for one reason worth stating, because it is the rule that decides what
may be added here and what may not.

**Routing is not presentation.** The honesty property of the bus lives in
`FleetBus._route`, and it survives only while there is exactly one place that
decides an audience. A function here that quietly widened a delivery — "also show
this to everyone in the area" — would be a second router, indistinguishable from
the first in its output and answerable to nothing. So this module reads. Every
function is a query over rows the bus already wrote, and the only writes it
performs are the ones it delegates straight back to the bus.

What it adds
------------
* `channel_summaries` — the left-hand rail: rooms, membership, unread per room.
* `unread` — what a session has not acknowledged, per channel, so a badge can be
  a number rather than a boolean.
* `render_thread` — a root and its replies with the reply-count Slack shows.
* `mention_report` — which `@handles` in a body actually reached somebody. The
  bus deliberately delivers nothing for a handle that matches zero or several
  sessions; without this, an author would never learn that.
* `area_channel` — the channel name for an `Area`, so rooms and derived routing
  share one coordinate vocabulary instead of drifting into two.

What it deliberately does not add
---------------------------------
Presence and typing indicators. The mesh knows which worktrees have outstanding
edits, which is *activity*, and rendering that as a green dot would state
something stronger than the evidence: an agent session that is thinking, or
waiting on a tool, has no outstanding edits and is not idle. `sessions_live` is
offered instead and named for what it measures.
"""
from __future__ import annotations

from dataclasses import dataclass

from alelyon.runtime.common.worktree_areas import Area
from alelyon.runtime.common.worktree_bus import (
    Channel, Finding, FleetBus, KIND_MESSAGE, normalise_channel, parse_mentions,
)

#: A room whose last message is older than this is folded away by default. Read
#: -time only and never deletes: the bus keeps history because history is the
#: point, and a quiet channel is quiet, not gone.
DEFAULT_QUIET_DAYS = 21.0


def area_channel(area: Area) -> str:
    """The channel name for an area: `runtime.common/fleet` -> `runtime-common-fleet`.

    Rooms and routing share one vocabulary on purpose. A session that claims
    `runtime.common` and joins `#runtime-common` is in the room about the work it
    declared, and the two names cannot drift apart because one is computed from
    the other.
    """
    return normalise_channel(str(area).replace("/", "-").replace(".", "-"))


@dataclass(frozen=True)
class ChannelView:
    """One room as a reader needs it: who is in it, and what is waiting."""

    channel: Channel
    members: tuple[str, ...]
    messages: int
    #: Deliveries to this reader from this channel that they have not acked.
    unread: int
    #: Epoch seconds of the newest message, or 0 for a room with none.
    last_at: int
    joined: bool

    @property
    def display(self) -> str:
        return f"#{self.channel.name}"

    def quiet(self, *, now: int, days: float = DEFAULT_QUIET_DAYS) -> bool:
        if not self.last_at:
            return True
        return (now - self.last_at) > days * 86_400


@dataclass(frozen=True)
class MentionReport:
    """Which `@handles` in a body reached somebody, and which reached nobody.

    The unmatched half is the load-bearing one. `FleetBus._route` refuses to
    guess when a handle matches zero sessions or two, and refusing silently would
    reproduce the exact failure the bus already names for explicit addresses: a
    publisher believing they reached someone they did not. An author gets told.
    """

    matched: tuple[str, ...] = ()
    unmatched: tuple[str, ...] = ()
    everyone: bool = False

    @property
    def clean(self) -> bool:
        return not self.unmatched


def mention_report(body: str, deliveries) -> MentionReport:
    """Compare the handles an author wrote against the sessions actually reached.

    A handle counts as matched when some delivery went to a session it prefixes.
    That is the same prefix rule the router applied, evaluated against the
    router's own output rather than re-derived from the mesh — so this cannot
    disagree with what really happened, which is the whole point of asking.
    """
    handles, everyone = parse_mentions(body)
    reached = {d.to_session.lower() for d in deliveries}
    matched, unmatched = [], []
    for handle in handles:
        if any(session.startswith(handle) for session in reached):
            matched.append(handle)
        else:
            unmatched.append(handle)
    return MentionReport(tuple(matched), tuple(unmatched), everyone)


def unread(bus: FleetBus, session_id: str, *,
           max_age_days: float = 14.0, now: int | None = None) -> dict[str, int]:
    """Unacknowledged deliveries for one session, counted per channel.

    Keyed by channel name; the empty string holds everything that arrived
    outside any room, which is every finding published the original way. That
    key is not a bug to tidy away — a `refactor-in-flight` addressed by subject
    belongs to no channel and is the most urgent thing a session can receive.
    """
    counts: dict[str, int] = {}
    for delivery in bus.inbox(session_id, max_age_days=max_age_days, now=now):
        key = delivery.finding.channel
        counts[key] = counts.get(key, 0) + 1
    return counts


def channel_summaries(bus: FleetBus, session_id: str, *,
                      max_age_days: float = 14.0,
                      now: int | None = None) -> tuple[ChannelView, ...]:
    """Every room, with this reader's membership and unread count.

    Rooms the reader has joined sort first, then by most recent activity. Rooms
    they have not joined are still listed, because a channel is not private and
    hiding one would suggest it was.
    """
    pending = unread(bus, session_id, max_age_days=max_age_days, now=now)
    joined = {m.channel for m in bus.memberships_of(session_id)}
    stats = bus.channel_stats()
    views = []
    for channel in bus.channels():
        messages, last_at = stats.get(channel.name, (0, 0))
        views.append(ChannelView(
            channel=channel,
            members=tuple(m.session_id for m in bus.members(channel.name)),
            messages=messages,
            unread=pending.get(channel.name, 0),
            last_at=last_at,
            joined=channel.name in joined))
    views.sort(key=lambda v: (not v.joined, -v.last_at, v.channel.name))
    return tuple(views)


@dataclass(frozen=True)
class ThreadView:
    """A root message and its replies, with the count Slack puts under a root."""

    root: Finding
    replies: tuple[Finding, ...] = ()

    @property
    def count(self) -> int:
        return len(self.replies)

    @property
    def last_at(self) -> int:
        return self.replies[-1].at if self.replies else self.root.at


def render_thread(bus: FleetBus, finding_id: str) -> ThreadView | None:
    """Assemble one thread, or `None` when the id names nothing.

    Accepts the id of a reply as readily as a root: a session reading an inbox
    holds whichever message reached it, and making it find the root first would
    be asking it for the one thing it does not have.
    """
    messages = bus.thread(finding_id)
    if not messages:
        return None
    return ThreadView(root=messages[0], replies=tuple(messages[1:]))


def threads_in(bus: FleetBus, channel: str, *,
               limit: int = 50) -> tuple[ThreadView, ...]:
    """A channel as a list of conversations rather than a flat log.

    Replies are folded under their root, so a channel where two sessions
    exchanged twenty messages reads as two conversations and not as twenty
    interruptions.
    """
    history = bus.history(channel, limit=limit)
    roots = [m for m in history if not m.reply_to]
    known = {m.id for m in roots}
    replies: dict[str, list[Finding]] = {}
    for message in history:
        if message.reply_to:
            replies.setdefault(message.reply_to, []).append(message)
    # `history` already returns insertion order within a second, so the grouping
    # above preserves it. Re-sorting by id here would undo that and put an answer
    # before its question whenever two messages share a timestamp.
    out = [ThreadView(root, tuple(replies.get(root.id, ()))) for root in roots]
    # A reply whose root fell outside this page still has to be readable, or a
    # busy thread would vanish from the channel the moment its opening message
    # aged past `limit`. Fetch the root rather than dropping the conversation.
    for parent_id, children in replies.items():
        if parent_id in known:
            continue
        thread = bus.thread(parent_id)
        if thread:
            out.append(ThreadView(thread[0], tuple(thread[1:])))
    out.sort(key=lambda t: t.last_at)
    return tuple(out)


def sessions_live(mesh) -> tuple[str, ...]:
    """Sessions with outstanding edits the mesh can see.

    Named for what it measures, and not called `online`. A session with no
    uncommitted changes is invisible here whether it is thinking, waiting on a
    tool, or gone — this reports observed work, and there is no evidence in the
    repository that distinguishes those three.
    """
    from alelyon.runtime.common.worktree import UNATTRIBUTED
    return tuple(sorted({
        w.session for w in mesh.worktrees
        if w.session != UNATTRIBUTED and w.touched_paths}))


def post(bus: FleetBus, *, channel: str, body: str, from_session: str,
         mesh=None, cache=None, subject_paths=(), reply_to: str = "",
         kind: str = KIND_MESSAGE, from_evidence: str = "self-reported",
         at: int | None = None):
    """Say something in a room. Thin by design.

    Delegates straight to `FleetBus.publish`, adding no audience of its own. The
    one-line signature is the point: a chat helper that assembled its own
    recipient list would be the second router this module's docstring forbids.
    """
    return bus.publish(
        kind=kind, body=body, from_session=from_session,
        from_evidence=from_evidence, mesh=mesh, cache=cache,
        subject_paths=subject_paths, channel=channel, reply_to=reply_to, at=at)
