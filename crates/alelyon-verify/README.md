# Rust CNE verifier

This Rust crate implements canonical encoding, signature/hash checks, input
commitments, key lifecycle, transparency proofs and CNE replay. Its library is
`alelyon_verify`. `Cargo.toml` sets `publish = false`; do not publish it as part
of an unrelated change.

It links no deterministic kernel. Replay's numeric substrate (a compensated sum,
a mean and the seeded dither stream) is the `ReplayKernel` trait in
`src/kernel.rs`, which the caller hands to `verify_envelope`/`verify_case`.
Handed none, the verifier replays nothing: `scalar`, `tier`, `budget` and
`width` stay null (not performed), no reason class is added for their absence,
and `ok` is false. Every other check runs as usual. There is no fallback
arithmetic.

The specified substrate is a private deterministic kernel. A build that links
it hands it to the verifier through that trait; such a build, and its run of
every published vector on the kernel, are not part of this crate.

## Build and test

From this directory:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```

This crate's own tests replay on a labelled test-only kernel
(`kernel::TestKernel`, id `test-fixture/0`), which no envelope names, so it can
never check a nonzero width exactly. `tests/vectors/` is a byte-identical copy
of the published conformance vectors (the specification's suite); the tests read
it, and a check in the source repository holds the copy to the original.

Use the toolchain requirements in the manifest and retain the lockfile. A
successful build is not a Python/Rust parity or public-package release result.
The CNE conformance and differential parity gates remain separate.

## Trust and verification

The caller supplies the receipt, corresponding inputs and independently pinned
issuer public key. A receipt's own key does not authenticate its issuer. Replay
checks arithmetic and commitments, not the truth of the original capture.
Preserve the specified encoding, stable reason classes, width semantics and
resource refusals when changing either implementation.

Matching Python and Rust answers establishes agreement on exercised cases; the
implementations can still share a defect. Keep known forgery vectors and
adversarial falsifiers alongside differential tests. Role-separated keys do not
prove independent operators, and code alone does not establish external receipt
verification.

## Source map

`canonical.rs` handles strict JSON and Python-compatible canonical float text;
`crypto.rs` handles hashes and Ed25519; `data.rs` handles committed input
representations; `transparency.rs` and `keylife.rs` handle their trust records;
`replay.rs` evaluates supported programs on a `kernel.rs` substrate;
`verifier.rs` composes verdicts.
The crate forbids unsafe Rust in its own source.
