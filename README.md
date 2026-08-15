<div align="center">

<img src="assets/banner.svg" alt="Alelyon — numbers that carry their own evidence" width="820">

<p>
  <a href="https://pypi.org/project/alelyon-os/"><img alt="PyPI" src="https://img.shields.io/pypi/v/alelyon-os?style=flat-square&label=pypi&color=e6c46a&labelColor=0f0f0f"></a>
  <a href="https://pypi.org/project/alelyon-os/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/alelyon-os?style=flat-square&color=57c7b0&labelColor=0f0f0f"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-57c7b0?style=flat-square&labelColor=0f0f0f"></a>
  <a href="spec/cne-v0.md"><img alt="Spec" src="https://img.shields.io/badge/spec-cne--v0-e6c46a?style=flat-square&labelColor=0f0f0f"></a>
  <a href="skills/alelyon/"><img alt="Agent Skill" src="https://img.shields.io/badge/agent%20skill-%2Falelyon-e6c46a?style=flat-square&labelColor=0f0f0f"></a>
</p>
<b>The Open/Closed Beta experience is live! To download the UI go to: https://www.alelyon.com/
Please continue reading for further details.</b>

To unlock the full capabilities of the Alelyon Ecosystem on your machine, follow these steps:

1. Clone the repository and set up the skills

In your terminal, run: 

```bash
git clone https://github.com/TLace03/Alelyon-OS
mkdir -p .agents/skills && cp -r Alelyon-OS/skills/alelyon .agents/skills/
```

2. Install the package

Make sure you’re in a Python 3.10–3.13 environment, then run: 

```bash
pip install alelyon-os
```

3. Create an account and verify

Register an account and log in at https://www.alelyon.com/account/ then verify with LinkedIn. 

Once verified, an access key (generated and encrypted with leading cryptography algorithms) will appear on the account page: https://www.alelyon.com/account/. Copy it and paste it into the dedicated space which will appear on the sign-in screen when you launch the application the first time.

For those who prefer not to go through LinkedIn verification but still want to try the product, a limited/demo experience is available after registration. Full OS functionality, however, require verification. 

Why LinkedIn verification? There's a few reasons, before I list them it's important to note we are not scraping any data from LinkedIn, it's a QoL decision.

1) People are less likely to have a fake/botted LinkedIn and it not be obvious. As such, there is less chance a person will have impersonators with their profile information and it not be obvious who the fake is. You will have a chance to collaborate with other users, this should influence teams to work together using the application. Preventing impersonators from getting invited is incredibly important.

2) It allows for us to make sure no one is putting on the wrong company tag which is a social networking feature. I guess it can still happen if people put a fabricated employer on their LinkedIn profile.

3) It allows us to have a database where we will keep only the necessary information of verified users, enabling us to provide better help in the case a user needs to contact us for any reason at all.

We’re excited to have early users testing the system and helping shape what’s next. Feedback is welcome — jump in and let us know what you think!

<b>The open part of the Alelyon Deterministic Quantitative Computational Operating System.</b>

A number you are handed is a claim. This gives it a receipt that a stranger can check by
replaying it against their own copy of the inputs — and that fails, loudly and
specifically, the moment the inputs are revised after the fact.

```bash
pip install alelyon-os
```

<img src="https://raw.githubusercontent.com/TLace03/Alelyon-OS/main/assets/demo-verify.svg" alt="A terminal recording: alelyon-verify accepts a receipt whose data matches, then rejects the same receipt after one cent is changed." width="900">

</div>

> Every line in that recording is copied from an actual run — the reason classes, the
> `width_trust` values and both exit codes. See [Verify a receipt](#verify-a-receipt).

## The `/alelyon` agent skill

Coding agents are confident about numbers they never measured. This repository ships an
[Agent Skill](https://agentskills.io/) that teaches one the discipline — observed versus
declared, say **UNMEASURED** out loud, never weaken an acceptance criterion — alongside
the toolkit that makes a number checkable.

```bash
git clone https://github.com/TLace03/Alelyon-OS
mkdir -p .agents/skills && cp -r Alelyon-OS/skills/alelyon .agents/skills/
```

`.agents/skills/` is the shared convention, so one copy works across
**Claude Code, Codex, Cursor, Antigravity, Gemini CLI, GitHub Copilot, VS Code, Amp,
Goose, OpenCode, Kiro** and others. Invoke it with `/alelyon` (`$alelyon` in Codex), or
let the agent load it on its own when a task matches.

The skill is [`skills/alelyon/`](skills/alelyon/) — a `SKILL.md` plus five reference
files loaded only when the task needs them.

## What is here

| Import | What it does |
|---|---|
| `alelyon.verify` | Verify a Certified Number Envelope by replay against your own copy of the inputs, under a key you pin out of band. Ships the `alelyon-verify` CLI, the normative spec, and the conformance vectors. |
| `alelyon.runtime.vector.lattice` | Exact coordinate registration: immutable coordinate contracts, exact target-to-source transforms with a declared loss/invertibility surface, canonical byte encoding with content commitments, a replay checker, and a signed Registration Certificate. |
| `alelyon.runtime.vector.lattice.morphometry` | Model Morphometry — a canonical `(block, module)` template for transformer models and an exact registration of a model's native axis order onto it. |
| `alelyon.runtime.common.worktree*` | Fleet coordination: what several agent sessions in one repository can each observe and declare, stored apart and never merged. |
| `alelyon.runtime.vector.compute` | A typed dependency DAG with Monte-Carlo uncertainty propagation and variance attribution. |
| `alelyon.platform.sdk` | Python client for the Alelyon read-only HTTP API. Requires the `sdk` extra. |

Full documentation, the migration notes, and the caveats that matter before you rely on
any of it are in [packages/alelyon-os/README.md](packages/alelyon-os/README.md).

## Verify a receipt

```bash
alelyon-verify selftest          # the bundled conformance suite; needs no network
alelyon-verify verify --envelope receipt.json --data your_extract.json \
    --key <the issuer's public key, obtained OUT OF BAND>
```

The key must reach you by some path the receipt did not travel. Verifying an envelope
against a key embedded in that same envelope authenticates nothing.

A passing verification means the committed inputs were not revised after the fact and
the number replays from them under the pinned key. It does **not** establish that the
inputs were true when captured: a producer who fabricates an extract at capture signs a
receipt that verifies perfectly. The normative specification is
[spec/cne-v0.md](spec/cne-v0.md).

## This tree is generated

`packages/alelyon-os/**` and `spec/**` are produced from Alelyon's private monorepo by an
exporter that builds them from the same allowlist that builds the published wheel, so the
published source and the published artifact cannot disagree about what is open. A pull
request editing those paths cannot be merged as-is; the next export would overwrite it.

`packages/alelyon-os/UPSTREAM.json` names the exact source commit and records a SHA-256
for every generated file. It is self-declared traceability, not authenticated provenance.

Issues, discussions, and reports of anything wrong here are welcome and wanted — that is
what this repository is for.

## Superseded

This replaces `TLace03/Alelyon-Dev-Tools` and its three separate distributions.

> **The old PyPI names are gone and are not ours.** `alelyon`, `alelyon-sdk`,
> `alelyon-verify` and `alelyon-mock` were withdrawn from PyPI. Deleting a project
> releases its name, so those four are **unregistered and may be claimed by anyone**.
> Do not install them, and do not follow an older instruction that names them.

| Was | Now |
|---|---|
| `pip install alelyon-verify` | `pip install alelyon-os` |
| `from alelyon_sdk import AlelyonClient` | `from alelyon.platform.sdk import AlelyonClient` |
| `alelyon-verify verify …` | unchanged — same console script |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option. Both texts ship inside the wheel.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you shall be dual licensed as above, without any additional
terms or conditions.
