# Alelyon-OS

The open part of the Alelyon Deterministic Quantitative Computational Operating System.

```bash
pip install alelyon-os
```

> **Not on PyPI yet.** The distribution is packaged and this repository is its generated
> source, but the first release has not been published. Until it is, install from a
> clone: `pip install ./packages/alelyon-os`.

---

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

## This tree is generated — do not edit it here

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
