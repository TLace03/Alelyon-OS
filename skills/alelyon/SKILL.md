---
name: alelyon
description: Apply Alelyon's evidence-reporting discipline and use its receipt verifier, producer, coordinate registration, uncertainty propagation and fleet tools. Required before substantive work in the Alelyon source repository.
---

# Alelyon

Use this skill to keep a result connected to the evidence that supports it.
Inside the source repository, read AGENTS.md as well. Load specialist references
only for the components your task reaches.

## Evidence before claims

An observation comes from a command or inspection performed against the stated
artifact. A declaration comes from an author, a prior document, a configured
assumption or an unexecuted example. Keep that distinction visible.

Report an absent required measurement as **UNMEASURED**, with the particular
property and reason. Do not copy that marker from an old report as though the
gap were re-established now. A skipped check, unavailable dependency, caught
exception or successful HTTP response does not prove the underlying result.

Record the command, tree or artifact, relevant inputs and result. Include counts,
skips, warnings and timing when they affect interpretation. Describe inherited
test results as inherited. Do not report a benchmark you did not run.

If an acceptance criterion cannot be met, distinguish a limitation inherent in
the design from missing infrastructure. Explain or fix that condition; do not
relax the criterion to produce a passing result.

## Receipt claims

Say what was certified: arithmetic, a quantization term, replay, or a particular
coordinate correspondence. Avoid bare “certified,” an unspecified “tight error
bound,” or an unqualified claim of trustlessness.

A receipt binds committed inputs to computation and signing evidence. It does
not establish that the inputs were true when captured. Authentication requires
a public key obtained and pinned independently of the receipt. A second signing
key is not evidence of an independent operator.

Keep quantization, sampling, provider and model terms separate. A missing term
is not zero. Inapplicability needs a reason. Per-check null values mean that a
property was not established; preserve that state instead of coercing it to pass.

A nonzero-width result can depend on its specified deterministic substrate.
Fallback replay must not invent an authenticated width. Exact-zero width is
substrate-independent under the applicable capture law; report the actual
verdict rather than assuming which case a receipt belongs to.

## Start with the installed interface

The supported public distribution is `alelyon-os`. It installs the
`alelyon-verify` command. Do not turn the command name into a legacy package
installation instruction.

```bash
python -m pip install alelyon-os
alelyon-verify version
alelyon-verify selftest
alelyon-verify verify --envelope receipt.json --data extract.json --key <pinned-public-key-hex>
```

Use `--help` for the installed version's options. The verification command's
successful exit corresponds to its `ok` verdict. A self-test is a local check of
that installation, not evidence that a third party verified your receipt.

## Task references

| Task | Reference |
|---|---|
| Interpret verdicts, keys and refusal classes | `references/verify.md` |
| Declare input storage laws and issue a receipt | `references/produce.md` |
| Register coordinate systems and model axes | `references/lattice.md` |
| Propagate uncertainty through a computation graph | `references/compute.md` |
| Coordinate repository sessions | `references/fleet.md` |

## Source and generated copies

In the private monorepo, edit `plugins/alelyon/skills/` and run
`python tools/sync_agent_skills.py --write`. The provider discovery directories
are generated copies; `--check` detects drift.

The public Alelyon-OS mirror is generated through the packaging/export allowlist:
`packages/alelyon-os/**`, `spec/**` and `skills/alelyon/**` there are produced
from this private source and are never edited in the mirror.
Make implementation changes in the private source, then use the authorized
export procedure. A manifest containing commit ids and hashes is declared
traceability; it is not by itself authenticated provenance.

## Handoff

Lead with the outcome. Identify the supporting observations and commands, each
remaining measurement gap, any relevant design limitation or infrastructure gap,
and external effects. A reader must be able to distinguish completed work from
proposed or unexecuted work without reconstructing the tool log.
