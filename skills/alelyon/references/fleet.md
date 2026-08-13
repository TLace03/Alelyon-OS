# Fleet — several agent sessions in one repository

Load this when more than one agent session is working in the same checkout, or when you
need to know whether an area is already taken before you start editing.

```bash
alelyon-fleet <command>      # who is where, what they found
alelyon-chat  <command>      # channels, threads, mentions
alelyon-ledger <command>     # the ledger's own store
```

Three CLIs rather than one because they answer different questions and write different
stores. Folding `post` in beside `claim` would make a routine message read as an
operational act; overloading one `--database` flag would point it at two different files
depending on the subcommand.

## Orient before you edit

```bash
alelyon-fleet status         # who is working where
alelyon-fleet areas          # the coordinate space in force, AND ITS SOURCE
alelyon-fleet open-areas     # areas with no session on them
alelyon-fleet whoami         # who you would publish as, and whether that is enough
alelyon-fleet inbox          # findings addressed to this session
```

`areas` reporting its source matters: the coordinate space is declared by the repository
under observation, in its own `.alelyon/fleet.toml`, or discovered from tracked
directories, or **empty**. An empty space means everything is `UNMAPPED`, and that is
said out loud rather than papered over with a default.

## Claims are advisory

```bash
alelyon-fleet claim <area> --note "..."
alelyon-fleet release <area>
```

A claim is an **advisory hold**. It does not lock anything. Its value is that another
session can see it before starting work in the same place, which is worth more than a
lock nobody can override.

**Read the claim command's output in full.** A claim can be refused or contested, and a
session that assumes success because the command returned is exactly the failure this
tool exists to prevent.

## Publishing findings

```bash
alelyon-fleet publish --kind <kind> --body "..." \
    --about path/to/file.py            # repeatable; ROUTING DERIVES FROM THESE
    [--to-area A | --to-session S | --broadcast]
    [--severity info|warn|urgent]
```

`--about` paths decide who hears it. A finding published with no paths and no address
reaches nobody in particular. `--broadcast` reaches every live session — use it sparingly;
it is the option that trains people to ignore the channel.

`--to-session` is recorded as **DECLARED**, not established. You are asserting who you
addressed, and the record says so.

Acknowledge what you read: `alelyon-fleet ack <finding_id>`.

## Assurance levels

`alelyon-fleet whoami --at-least CORROBORATED` exits non-zero when the current
attribution does not reach the named level. Use it as a gate before a durable write: a
session that cannot establish who it is should not be publishing as anyone.

## Where work is stuck

```bash
alelyon-fleet waiting           # sessions that stopped and are waiting for you
alelyon-fleet resume            # what is dormant, and what waking it would involve
alelyon-fleet disciplines       # which specialist rules govern what you touched
```

In a full source checkout, where the complete work-supply planning graph is present,
`alelyon-fleet supply` adds the fleet-as-production-line view. The public wheel omits
that source-only graph, so its CLI does not advertise the command. Do not infer a broken
installation from its absence.

Where `supply` is available, `supply --no-corpus` is much faster and the line is then
**empty** — which is a missing reading, not an idle fleet. The tool says so rather than
showing you a clean board. Same for `waiting --max-sessions`: anything the cap hides is
reported as hidden.

That pattern is the whole design. A tool that cannot see something reports that it cannot
see it, because a blank panel and a clean panel look identical and mean opposite things.

## Starting agents is Tier 3

`alelyon-fleet resume --wake <label>` starts nothing without `--authorise`. Without it
the answer is always `no-owner-authority`, which is the safe default rather than a fault.
`--model` is never defaulted: the layer space says what *rank* work is, never which model
should do it.

## State locations

`$ALELYON_HOME` → repository root → per-user platform directory, with `paths.INSTALLED`
recording which resolution was used. An installed wheel has no `pyproject.toml` above it,
so the repo-root branch does not apply and state goes to the platform directory rather
than into `site-packages`.

Findings do not cross checkouts. A separate worktree has a separate bus, so a finding
published in one is not visible in another.
