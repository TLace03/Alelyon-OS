# Observed versus Declared

### Coordinating concurrent agent sessions from records they did not write

**Status:** systems paper. Ships in `alelyon-os` as a library and the
`alelyon-fleet` CLI. Developed against a live workload of up to 26 concurrent
worktrees in one repository.

---

## Abstract

When several autonomous coding agents work in one repository at once, each
behaving correctly in isolation, they produce work that cannot be reconciled —
because none of them could see the others. The obvious fix is a registry every
agent writes to. The obvious fix does not work: an agent that does not
participate is invisible, and an agent that participates dishonestly is worse
than invisible.

We describe a coordination layer built on a strict separation between two kinds
of evidence. **Observed** facts are derived from records the agent did not
author — git's own worktree administration, commit reachability, the file paths
a session has touched, and (for one vendor) a transcript index keyed to a
directory the agent did not name. **Declared** facts are what an agent says about
itself. The two are stored apart, labelled at every read, and **never merged**.

The design's value is almost entirely in what it refuses to conclude.

## 1. The failure this addresses

Two sessions edit the same file in different worktrees. Neither is wrong. The
merge is a mess, and it is discovered hours later.

An intersection of touched paths is the earliest observable sign, and it is
available without anyone cooperating: `git worktree list --porcelain` plus a
`status` in each tree is enough to compute it. That observation is the floor the
rest of the system is built on.

## 2. Attribution is derived, never believed

Git's `author` and `committer` are freely settable. A worktree's directory name
is chosen by whatever created it. Any session id written into a file is written
by the thing claiming that identity. All three are *the writer describing
itself*, so none is used.

What is used instead:

- **The worktree's administrative path**, its `HEAD`, and reachability from the
  mainline — records git keeps for its own purposes.
- **The parent-directory convention**, which identifies a **tool family** and
  nothing finer. `.claude/worktrees/`, `.codex/worktrees/`, `.cursor/worktrees/`
  and so on.

Every derivation carries the rule that produced it in an `evidence` field, so a
reader can disagree with it. Where no rule fires, the answer is `UNATTRIBUTED` —
a value, never a blank. A blank beside a filled field reads as "checked, nothing
there".

**Model identity is not derivable from git and is not guessed.** Tool family is
the honest ceiling for a worktree found on disk.

## 3. One genuine advance, and its exact limit

Some tools place a session identifier in the worktree path. Where that
convention holds, a session is identifiable from a location git records, without
the agent volunteering anything. That is a real improvement over self-registration.

It is still a **derivation**, not authentication: a worktree can be created at
any path, so anything able to choose a directory name can wear a session-shaped
one.

And it names the session that **created** the directory, not the one working in
it now. A second session can enter an existing worktree and leave changes there,
and every one of those changes is attributed to the creator. **This is the one
case where the system reports a confident answer that is wrong** rather than an
honest `UNATTRIBUTED`. It is stated in the limits that print at every read,
because a limit that is not stated reads as a limit that does not exist.

A second, independent derivation covers the case the path cannot: a per-session
transcript file, named by the harness with a `cwd` field the harness wrote. It
over-approximates occupancy — it names every session that *started* in a
directory, not the one editing it — and that asymmetry is deliberate. A mailbox
that reaches you along with somebody else is a working mailbox; no mailbox
reaches nobody.

## 4. The coordinate space belongs to the repository

Work is placed on an `(pillar, surface)` coordinate. Routing is derived from
those coordinates, so what places a path decides who hears a finding about it.

The first version of this module hard-coded one repository's directory table.
Inside that repository it was exactly right, and it was *why* attribution could
be validated against an independently-held invariant: the table was owner-authored
policy the module did not get to edit.

Published as a general library, the same table was wrong in two directions at
once. Every path in a stranger's checkout fell through it and read `UNMAPPED`, so
the coordination layer saw an empty repository; and the table asserted one private
tree's layout as though it were a general vocabulary.

The resolver now has three outcomes, and reports which one ran:

| outcome | source |
|---|---|
| **declared** | `.alelyon/fleet.toml` in the repository root. The repository states its own pillars. |
| **discovered** | the repository's own tracked paths — the directories that actually exist in the checkout the user opened. |
| **empty** | neither is available. Everything is `UNMAPPED`, and the CLI says so rather than showing an empty table. |

Discovery is deliberately simple and explainable: a pillar is a top-level
directory; depth is 1 by default; a directory whose children are all files is
*flat*, so the file is the unit. That last rule exists because a directory of
unrelated scripts collapsed to one surface makes a single session editing a
single script read as occupying all of it — which steers other work away from
ground that is genuinely free.

**Nothing is ever placed into a directory that was not observed or declared in
the repository under observation. There is no ambient list.**

## 5. A claim is not a lock

Territory can be claimed, and a claim is explicitly advisory. Two sessions can
hold one area; the tool records that, flags it as contested, and refuses nothing.
Reconciliation is between the sessions, not with the tool.

This is not a limitation to be fixed. A lock in this setting is either unenforceable
(nothing stops an agent editing a file) or a deadlock generator (an agent that
crashes holding a lock). Recording the contention and making it visible is the
achievable goal.

## 6. Publishing reaches somebody, or it does not

When a finding is published, the tool prints **who it reached**. `REACHED NOBODY`
is a real outcome and is worded so it cannot be read as success.

This came from an observed failure. A session claimed `platform.gateway` — the
dotted form, because pillars legitimately contain dots — while routing derives
`platform/gateway`. The claim was accepted, reported as success, and the session
was never reachable. Four findings published at the real area reported REACHED
NOBODY while their author believed the territory was announced.

The fix has two halves, and the second is the interesting one. Creating a
coordinate is now checked against the vocabulary and refused with a suggestion.
Reading one back is **not** checked — so a bad record can still be inspected and
released. A validator applied symmetrically would have made the broken claims
unreadable and therefore unreleasable.

## 7. A second axis: what kind of work a path is

Where a path *lives* and what *discipline* touching it requires are different
questions, and neither implies the other. A cryptographer and an API developer
editing one directory are one area and two disciplines.

Disciplines are declared by the repository, in the same file as the pillars — one
repository, one description of it. Each names its trigger kind:

- **paths** — the policy lists the files it governs. The match is exact.
- **reachability** — the policy names what code must *reach*. That is a property
  of the import graph, which this module does not compute, so it matches only the
  anchors named by path.

Reachability-triggered disciplines therefore **under-approximate**: fewer
disciplines than really apply, never more. That direction is safe for routing a
finding to an interested party and **unsafe for gating a change**, so gating stays
with the policy document, CI and a reviewer. The gap is named as an infrastructure
gap rather than papered over with plausible globs, because a guessed glob reports
a discipline as *checked* on ground nobody checked.

## 8. What this cannot tell you

Printed at the end of every command, not buried in a docstring:

- A finding's body is self-reported. Nothing checks it, and a stale finding looks
  exactly like a fresh one except for its timestamp.
- Routing reaches only sessions the mesh can see. A session working outside every
  known convention receives nothing and is not counted as having been told.
- **Two sessions sharing one checkout are one path and therefore one derived
  identity.** They contend most directly and the pairing cannot see them at all.
- Contention is computed from paths touched, not semantics. Two sessions editing
  different functions in one file are reported as contending — a deliberate false
  positive rather than a missed one.
- A clean worktree someone is actively reading is indistinguishable from an
  abandoned one. Staleness measures a commit, not a session.
- **Silence is not consent.** An unacknowledged finding means nobody pressed the
  button, not that nobody was affected.

## 9. Reproduction

```bash
pip install "alelyon-os[fleet]"
cd /any/git/repo
alelyon-fleet areas        # the coordinate space in force, and where it came from
alelyon-fleet status       # who is working where
alelyon-fleet open-areas   # areas no session is on
```

On a repository that declares nothing, `areas` reports discovery from that
repository's own tracked directories and lists them. No configuration is
required, and no directory outside the checkout you point it at is read.

## 10. What we would want reviewed first

1. Whether the discovery heuristic (top-level directory as pillar, all-files
   means flat) is right for repository layouts we have not seen — notably
   monorepos with a single `packages/` directory containing everything.
2. The over-approximation in the transcript-derived identity. It is deliberate,
   but we have not characterised how wrong it gets with long-running sessions.
3. Whether the observed/declared separation survives contact with a tool that
   writes its own worktree metadata — which would blur the two categories the
   whole design rests on.
