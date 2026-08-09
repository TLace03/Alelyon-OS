"""Who did this — as a value a caller can require, not a sentence it can read.

This repository already grades its own identity evidence, and grades it well.
`fleet_cli._identify` distinguishes a session derived from a worktree path, one
declared on a command line, and one declared *and* corroborated by a harness
record the writer did not author — and it says which in every command's output.
The problem is that the grade exists **only as English**. `_identify` returns
`(session, evidence)` where `evidence` is prose, so a caller can print how good
an attribution is and cannot **branch** on it.

That is fine while every surface reading it is read-only, and it stops being
fine at the first write path. The reason was put most sharply by the frontend
reviewer on 2026-08-06, and it is the argument this module exists to serve:

    The Lattice fleet hub is read-only, and the stated reason is "a window is
    not a session". The read-only property is currently what makes the
    path-derived identity SAFE. A window cannot post, so a window cannot post
    AS THE WRONG PERSON. The moment any of those panels gains a write path --
    reply, ack, claim -- the self-reported identity becomes a write capability,
    and "two sessions in one checkout are one identity" becomes "one person can
    act as another".

So the identity seam is a **prerequisite** for any write path, not follow-up
work. What follows is the smallest thing that makes that expressible.

Not a future problem: measured today
------------------------------------
Three sessions worked in `C:/Users/tommy/famMain` at the same time on
2026-08-06 — one checkout, one path, one derived identity. They were
distinguishable only because each passed `--session`, which the CLI correctly
labels self-reported, and one of them published a finding under the **wrong
author** and caught it only because derived routing disagreed. The enterprise
failure is already reproducible in a single-owner repository.

What this module does and does not do
-------------------------------------
It **adds a vocabulary without changing fleet behaviour**. Nothing here is
wired into `worktree`, `worktree_bus`, `fleet_chat` or fleet authorship; those
keep deriving exactly as they do, and
`test_the_seam_agrees_with_todays_derivation` asserts that this module's answer
matches `fleet_cli`'s for the same input.  The local Teams contact repository is
the first write consumer: it explicitly requires an authenticated human, while
leaving every existing fleet surface unchanged.

What it adds is that a future write path can say `require(actor,
Assurance.CORROBORATED)` instead of hoping its caller passed something real.

Ambiguity is not weak assurance -- it is a different failure
------------------------------------------------------------
A derivation can be perfectly sound and still name a *set*. Three agents in one
checkout derive to one identity: the derivation did not guess, it collapsed.
`worktree_bus` already records this ("two sessions sharing one checkout are one
path and therefore one derived identity") and `worktree` already returns
`UNATTRIBUTED` rather than a nearest guess.

So `ambiguous` is carried beside `assurance` and **defeats it**: an ambiguous
actor may not author anything at any assurance level, because the failure mode
is not "we are unsure who this is" but "this name denotes more than one person,
and a write under it is one person acting as another". A single flag would have
had to rank that somewhere on the strength scale, and there is no correct place
for it.

Organizations, before there are any
-----------------------------------
`Organization.LOCAL` is the single implicit organization every existing record
belongs to. It exists so new state can be keyed on `(organization, actor)`
today, while the only implementation is local and single-tenant — see
[ENTERPRISE-TEAMS.md](../../../docs/features/ENTERPRISE-TEAMS.md) §4. A record
written now without an organization is a migration later; one written with it
costs a column and nothing else. Nothing here makes the product multi-tenant,
and nothing here authorizes hosting.

Read-only and pure: takes strings and records, returns records. No git, no
database, no clock.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Optional, Protocol

#: The value every record belongs to until organizations are real. Named rather
#: than left implicit so a key can be written now and mean something later.
LOCAL_ORGANIZATION = "local"

#: What `worktree` returns where no rule fired. Repeated here rather than
#: imported so this module stays free of the derivation it describes.
UNATTRIBUTED = "UNATTRIBUTED"


class Assurance(IntEnum):
    """How an attribution was established, ordered by what it is worth.

    Ordered so a caller can write `>=`. The order is the repository's existing
    evidence vocabulary, not a new one:

    * `DECLARED` — the writer typed it. Nothing checks it, and a stale or wrong
      declaration looks exactly like a true one.
    * `CORROBORATED` — declared, and an independent record the writer did not
      author is consistent with it. Still a declaration; what changed is that it
      is now **refutable**, which is the whole of its value.
    * `DERIVED` — computed from something the writer did not write, such as a
      path a tool chose. Stronger than a declaration and still not proof: it
      answers who *created* a directory, never who is in it.
    * `AUTHENTICATED` — possession of a credential was proved. The only level
      that survives a second person sharing the machine.

    `DERIVED` outranks `CORROBORATED` only where the derivation is
    unambiguous; see `Actor.ambiguous`, which is why the two are separate
    fields rather than one scale.
    """

    DECLARED = 1
    CORROBORATED = 2
    DERIVED = 3
    AUTHENTICATED = 4


class Kind(IntEnum):
    """What sort of thing acted. Both appear on one board, deliberately.

    An enterprise wants to see that a colleague *and their agents* are in a
    file; collapsing agents into the human who launched them would throw away
    the thing the fleet views are for.
    """

    UNKNOWN = 0
    AGENT = 1
    HUMAN = 2


class InsufficientAssurance(PermissionError):
    """A write was attempted under an attribution too weak to carry it."""


@dataclass(frozen=True)
class Actor:
    """Who did something, and how firmly that is known.

    `evidence` is the sentence today's tools already print. It is carried rather
    than replaced, because a grade without its reason is how a reader stops
    being able to disagree with the machine.
    """

    id: str
    kind: Kind = Kind.UNKNOWN
    display: str = ""
    organization: str = LOCAL_ORGANIZATION
    assurance: Assurance = Assurance.DECLARED
    #: True where the attribution names more than one possible actor. Sound
    #: derivation, collapsed answer -- see the module docstring.
    ambiguous: bool = False
    evidence: str = ""

    @property
    def attributed(self) -> bool:
        """Whether this names anybody at all."""
        return bool(self.id) and self.id != UNATTRIBUTED

    @property
    def key(self) -> tuple[str, str]:
        """The `(organization, actor)` pair new state should be keyed on."""
        return (self.organization, self.id)

    def may_author(self, minimum: Assurance = Assurance.CORROBORATED) -> bool:
        """Whether a durable write may be recorded under this attribution.

        Ambiguity defeats assurance outright rather than lowering it: a name
        denoting two people is not a weaker author, it is the wrong question.
        """
        return (self.attributed
                and not self.ambiguous
                and self.assurance >= minimum)

    def refusal(self, minimum: Assurance = Assurance.CORROBORATED) -> str:
        """Why `may_author` said no, in the words a reader needs. '' if it said yes.

        Ambiguity is reported BEFORE absence, because the two are different
        facts and the ambiguous one is the more informative: "nobody works
        here" and "too many people work here to say which" both leave the id
        empty, and only the second tells a reader what to do about it.
        """
        if self.ambiguous and not self.attributed:
            return (f"more than one actor is working here ({self.evidence}), so "
                    f"nothing can say which of them is asking")
        if not self.attributed:
            return ("nothing attributed this action, so there is no author to "
                    "record it under")
        if self.ambiguous:
            return (f"{self.id} names more than one actor here ({self.evidence}), "
                    f"so a write under it would record one actor's work against "
                    f"another's name")
        if self.assurance < minimum:
            return (f"{self.id} is {self.assurance.name} ({self.evidence}) and "
                    f"this needs at least {minimum.name}")
        return ""


def require(actor: Actor, minimum: Assurance = Assurance.CORROBORATED) -> Actor:
    """Return `actor`, or raise if it may not author at `minimum`.

    The function a write path calls. It exists so the check is one expression
    rather than a convention each caller re-implements — the same reason
    `verification_receipt` makes an unscoped query impossible to write.
    """
    if not actor.may_author(minimum):
        raise InsufficientAssurance(actor.refusal(minimum))
    return actor


#: Nobody. Returned rather than `None` so a caller cannot forget to check, in
#: the same spirit as `worktree`'s `UNATTRIBUTED` being a value and not a blank.
NOBODY = Actor(id=UNATTRIBUTED, kind=Kind.UNKNOWN,
               evidence="no attribution was supplied")


class ActorResolver(Protocol):
    """Answers who is acting. The seam a hosted deployment replaces."""

    def current(self) -> Actor:                       # pragma: no cover - shape
        ...


@dataclass(frozen=True)
class DerivedFromPath:
    """Today's answer, as a value: the session a worktree path attributes to.

    Wraps the existing derivation without changing it. `ambiguous` is set when
    the path cannot separate co-occupants — which for the primary checkout is
    always, because every session in it shares one path.
    """

    session: str
    evidence: str = ""
    shared_checkout: bool = False
    organization: str = LOCAL_ORGANIZATION

    def current(self) -> Actor:
        if not self.session or self.session == UNATTRIBUTED:
            return replace(NOBODY, organization=self.organization,
                           evidence=self.evidence or NOBODY.evidence)
        return Actor(
            id=self.session, kind=Kind.AGENT, display=self.session[:12],
            organization=self.organization, assurance=Assurance.DERIVED,
            ambiguous=self.shared_checkout,
            evidence=self.evidence or "derived from the worktree path")


@dataclass(frozen=True)
class DeclaredOnCommandLine:
    """A `--session` value, graded by whether anything corroborates it.

    `corroborated` is the caller's answer to "is this among the sessions the
    harness recorded for this directory", which is exactly what
    `fleet_cli._identify` already computes. Non-membership is reported and never
    refused, because sessions publish under names of their own choosing and
    rejecting those loses findings rather than improving them.
    """

    session: str
    corroborated: bool = False
    evidence: str = ""
    organization: str = LOCAL_ORGANIZATION

    def current(self) -> Actor:
        if not self.session:
            return replace(NOBODY, organization=self.organization)
        assurance = (Assurance.CORROBORATED if self.corroborated
                     else Assurance.DECLARED)
        default = ("declared on the command line, corroborated by a harness "
                   "session record for this directory" if self.corroborated
                   else "self-reported on the command line")
        return Actor(id=self.session, kind=Kind.AGENT, display=self.session[:12],
                     organization=self.organization, assurance=assurance,
                     evidence=self.evidence or default)


#: Evidence strings meaning NOTHING corroborated the name -- the writer typed it
#: and no record the writer did not author agrees. Both forms in the tree are
#: listed: `fleet_cli` and `relay` write the long one, `fleet_chat` defaults to
#: the short one.
UNCORROBORATED_EVIDENCE = frozenset({
    "self-reported on the command line",
    "self-reported",
})


def is_uncorroborated(evidence: str) -> bool:
    """Whether `evidence` says a session id rests on nothing but the claim.

    Driven off a set of KNOWN self-report strings rather than off "anything that
    is not the corroborated phrase", so an evidence form this function has not
    seen defaults to UNMARKED. The asymmetry is deliberate: failing to mark a
    declaration understates it, while marking a derived or corroborated author
    impugns somebody the records do vouch for, and that is the worse error.

    Why this exists. A session may publish under any name it likes -- rejecting
    the name loses findings rather than improving them, which
    `DeclaredOnCommandLine` states as settled design. The cost lands on READERS:
    on 2026-08-08 findings were published under another live session's id, that
    session was credited in a merged commit message for work it had not done,
    and it had to correct the record by hand. Only `fleet inbox` printed the
    evidence; the mail and chat listings showed a bare id, which reads as
    established.
    """
    return (evidence or "").strip() in UNCORROBORATED_EVIDENCE


@dataclass(frozen=True)
class AuthenticatedAccount:
    """A signed-in human. The only resolver that survives a shared machine.

    Takes the fields rather than an `Account`, so this module does not import
    `platform.identity` — `runtime.common` sits below `platform` and an import
    the other way would be a layering inversion the `.importlinter` contracts
    exist to prevent.
    """

    account_id: str
    display: str = ""
    organization: str = LOCAL_ORGANIZATION

    def current(self) -> Actor:
        if not self.account_id:
            return replace(NOBODY, organization=self.organization)
        return Actor(id=str(self.account_id), kind=Kind.HUMAN,
                     display=self.display or str(self.account_id),
                     organization=self.organization,
                     assurance=Assurance.AUTHENTICATED,
                     evidence="signed in; possession of a credential was proved")


LIMITS: tuple[str, ...] = (
    "This module grades an attribution; it does not establish one. Every "
    "resolver here is handed its answer by a caller that derived or declared "
    "it elsewhere, so a wrong input produces a confidently graded wrong actor.",
    "AUTHENTICATED means a credential was proved at sign-in, not that the "
    "person at the keyboard is still the one who signed in. Nothing here "
    "observes the machine after the fact.",
    "DERIVED from a worktree path answers who CREATED the directory, never who "
    "is working in it now. That is why a shared checkout is marked ambiguous "
    "rather than trusted.",
    "Ambiguity is detected only where a caller reports it. Nothing in this "
    "module can notice two sessions sharing a path on its own, so an "
    "unambiguous-looking actor is not evidence that the path is exclusive.",
    "There is exactly one organization today and it is implicit. Keying on "
    "`(organization, actor)` makes a future tenant expressible; it does not "
    "isolate anything now, and no store enforces it yet.",
    "Two bounded consumers exist. `alelyon-fleet whoami` only REPORTS an "
    "attribution; the local Teams ContactStore requires an AUTHENTICATED human "
    "before a contact write. The bus, chat and every fleet authoring command "
    "still identify exactly as they did and may publish under a DECLARED "
    "session. Enforcement in one consumer does not promote the others.",
)
