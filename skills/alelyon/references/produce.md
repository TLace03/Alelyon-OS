# Producing a certified receipt

Load this when issuing receipts over your own data.

## The one thing to understand: Δ is a declaration

`delta` says *how the data was stored* — the quantization step. It is an assertion you
are making, not a property the library discovers.

```python
from alelyon.runtime.oracle.dsl.fetch import from_frame, from_csv, CENTS
```

- `delta=0.0` asserts the values are **exact as stored**.
- `CENTS` is `0.01` — a step constant, not a law name. Passing it as `law=` raises.
- A per-column mapping that omits a column is an **error**, never a defaulted zero.
  Defaulting would silently assert exact storage for the one column nobody considered.

`delta=0` over invented data certifies invented data, and nothing downstream detects
that. The receipt binds a number to inputs; it says nothing about where the inputs came
from. Never describe issuance as making data trustworthy.

## Capture laws

`law=` names the capture law the deltas were produced under. This build understands two:

| Law | Meaning |
|---|---|
| `"exact-cents/v0"` | Values are exact in whole cents. Every Δ is zero. |
| `"dither-relative/v0"` | Subtractively dithered storage, relative step. |

An unrecognised law is refused, not treated leniently — its delta semantics are
undefined, so the column is unusable rather than accepted.

`exact-cents/v0` carries an aggregate representability obligation that a per-row Δ cannot
express, which is why `certified_run` guards it against the 2^53 boundary separately.

## Why exact-cents receipts are the commercially useful ones

Δ=0 everywhere ⇒ the perturbation is identically zero ⇒ the order statistic is `0.0` on
any kernel. A zero width is **substrate-independent**, so the receipt verifies fully on a
plain `pip install alelyon-os` with no deterministic kernel present.

A nonzero width only reproduces bit-for-bit on the substrate that produced it. That is
why money, counts and ledger rows are the natural first product, and why market-data
receipts honestly degrade on a bare install.

## Building an envelope

```python
from alelyon.runtime.oracle.dsl.envelope import build_envelope

env = build_envelope(
    'show sum(price("CLAIMS"))',   # the program, in the DSL
    fetcher=fetcher,               # your declared inputs
    keystore=keystore,             # signs the envelope
    seed=7,                        # persisted; resampling PRNG
    now=0.0,                       # passed in, never wall-clock read
)
```

`now` is a parameter so envelopes are reproducible. The caller stamps real time if it
wants it — the library never reads the clock, because an envelope that changes when you
rebuild it cannot be a golden.

Inputs are fetched **once** and both certified and committed, so the bound and the
commitments cannot disagree.

## The program is a DSL, not Python

`build_envelope` takes source text. `data_refs(src)` returns the literal reads, e.g.
`[('price', 'CLAIMS')]`. Your fetcher must declare exactly those `(kind, key)` pairs.

A mismatch does not raise — it produces a **refusal envelope**: `refused=True` with a
`reason` such as `data fetch failed: ('price', 'CLAIMS')`, and `scalar=None`.

**A refusal is a valid certified object.** It verifies `ok=true`, with
`width_trust="refusal"`. The system certified that it declined to answer, and that is a
real, checkable result. Do not treat a refusal envelope as an error to be swallowed —
handing back "the engine refused, here is the signed refusal" is the honest outcome.

## Transparency anchoring and witnesses

With a keystore **and** a readable capture store, each input is transparency-anchored:
per-row Δ is attributed from the signed cert log, and covering leaves, inclusion proofs
and an STH are attached. A verifier can then confirm the width rests on
capture-time-committed deltas rather than the signer's later word.

Supplying `witness=` adds a co-signature over the STH under a cryptographically distinct
key, letting a client pinning that key detect a fork or rewind.

**Organizational independence exists only when another party operates the witness.**
A witness you run yourself is a second key, not a second opinion. Never call it an
"independent witness" in that configuration.

## Checklist before you ship a receipt

- [ ] Every Δ declaration traces to a real property of the storage, not a guess.
- [ ] The law is named and is one the build understands.
- [ ] `now` and `seed` are passed in, not read from the environment.
- [ ] The signing key is backed up; losing it strands every receipt under it.
- [ ] Recipients are told the key must reach them out of band.
- [ ] Your write-up says what a pass does *not* establish.
