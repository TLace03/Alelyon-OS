# alelyon-os

The open part of the Alelyon Deterministic Quantitative Computational Operating System,
as one installable distribution.

```bash
pip install alelyon-os
pip install "alelyon-os[sdk]"     # additionally installs httpx, for the API client
```

Source: <https://github.com/TLace03/Alelyon-OS>

---

## What is in here

| Import | What it does |
|---|---|
| `alelyon.verify` | Verify a Certified Number Envelope by replay against your own copy of the inputs, under a key you pin out of band. Ships the `alelyon-verify` CLI, the normative spec, and the conformance vectors. |
| `alelyon.runtime.vector.lattice` | Exact coordinate registration: immutable coordinate contracts, exact target-to-source transforms with a declared loss/invertibility surface, canonical byte encoding with content commitments, a replay checker, and a signed Registration Certificate. |
| `alelyon.runtime.vector.lattice.morphometry` | Model Morphometry — a canonical `(block, module)` template for transformer models and an exact registration of a model's native axis order onto it. |
| `alelyon.runtime.common.worktree*` | Fleet coordination: what several agent sessions in one repository can each observe and declare, stored apart and never merged. |
| `alelyon.runtime.vector.compute` | A typed dependency DAG with Monte-Carlo uncertainty propagation and variance attribution. |
| `alelyon.platform.sdk` | Python client for the Alelyon read-only HTTP API. Requires the `sdk` extra. |

## Verify a receipt

```bash
alelyon-verify selftest          # the bundled conformance suite; needs no network
alelyon-verify verify --envelope receipt.json --data your_extract.json \
    --key <the issuer's public key, obtained OUT OF BAND>
```

The key must reach you by some path the receipt did not travel. Verifying an envelope
against a key embedded in that same envelope authenticates nothing.

**What a passing verification means.** The committed inputs were not revised after the
fact, and the number replays from them under the pinned key. It does **not** establish
that the inputs were true when captured: a producer who fabricates an extract at capture
signs a receipt that verifies perfectly. See `SPEC-cne-v0.md`, shipped inside the wheel.

## Things worth knowing before you rely on this

**Extras gate dependencies, not files.** Every install receives every module listed
above. `[sdk]` adds `httpx`; it does not change what code is on disk. There is no way to
install a subset.

**`import alelyon.platform.sdk` fails without the `sdk` extra.** `client.py` imports
`httpx` at module scope. This is the one deliberate sharp edge: someone installing this
to check a receipt should not also acquire an HTTP client.

**This wheel is not minimal, and does not claim to be.** Its predecessor `alelyon-verify`
was a wheel containing only the verifier, and said so. That is not true of `alelyon-os`.
What remains true, and is checked against the built artifact on every release, is the
*boundary*: the wheel contains exactly a reviewed allowlist of files and nothing else.
The capture engine, the history store, the GUI, the HTTP service, identity and auth, and
every signing key are outside it.

**Model Morphometry reads no weights.** It takes a deterministic inventory from a model
runtime's *declared* metadata and runs no forward pass, so it measures declared
architecture and storage precision — and nothing about learned behaviour.

**The fleet modules are observational.** A claim is not a lock, and a finding's body is
self-reported. Nothing in them verifies that another session's declaration is true.

## Migrating from the old packages

`alelyon-sdk`, `alelyon-verify` and `alelyon-mock` were separate distributions and have
been withdrawn from PyPI. Those project names are **no longer registered to Alelyon** and
may be claimed by anyone; do not install them.

| Was | Now |
|---|---|
| `pip install alelyon-verify` | `pip install alelyon-os` |
| `from alelyon_sdk import AlelyonClient` | `from alelyon.platform.sdk import AlelyonClient` |
| `alelyon-verify verify …` | unchanged — same console script |

The SDK's import path changed because the public copy had drifted from the source it was
generated from. It is now generated, so the two cannot diverge again.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option. Both texts ship inside the wheel.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in this work by you shall be dual licensed as above, without any additional
terms or conditions.
