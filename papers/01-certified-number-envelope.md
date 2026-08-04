# The Certified Number Envelope

### Replay verification of a computed scalar under a key pinned out of band

**Status:** systems paper. The artifact ships as `alelyon-os`. Two independent
referee passes (prior-art, buyer) returned **overclaimed** on an earlier
twelve-domain framing; this paper is written to the claim discipline that
verdict produced.

---

## Abstract

A spreadsheet cell says `4,182,905.33`. Six weeks later an auditor asks where it
came from. The usual answers are a screenshot, a re-run that no longer
reproduces, or a person who has left.

We describe a signed envelope binding four things together: the *program* that
produced a scalar, the *scalar*, a *decomposed error budget*, and a
*per-input commitment* — a digest, a quantization step, and a seed for each
input the program read. A third party holding their own copy of the inputs and a
public key pinned out of band can re-execute the program and check that they get
the same number, or learn precisely which component failed.

The contribution is not new mathematics. It is a competent assembly, in the
SCITT / Sigstore-Rekor / Sigsum / RFC-6962 family, of transparency-log machinery
around a *computation* rather than around an artifact — plus a refusal
discipline that makes the resulting object hard to misread.

## 1. What the object is for

Verifiable computation has a large literature and mature machinery. What it
mostly lacks, at the point where a number reaches a report, is anything that
survives the gap between the person who computed it and the person who must
defend it.

The concrete case that shaped this design is an actuarial control total. A
reserve figure is reconciled against a number held by a *different function* —
the general ledger, the administration system, a reinsurer. That is a genuinely
independently-held invariant, which is what makes verification meaningful rather
than circular: the tie-out target was not produced by the party being checked.

## 2. What the envelope binds

An `alelyon.cne/v0` envelope carries:

**The program.** Not free-form code — a restricted DSL with no `eval`, no
imports, no attribute access, and no filesystem or network reach. The program is
hashed into the signed bytes. A verifier parses and executes the same source.

**The scalar**, and the *refusal* if the computation refused. A refusal is a
signed, first-class outcome carrying its real reason. This matters more than it
sounds: a system that silently degrades produces a number under conditions where
it should have produced nothing, and nothing downstream can tell.

**A decomposed error budget**, never summed into one number called "certified":

| term | how it is obtained | typical magnitude |
|---|---|---|
| quantization | theorem (linear-exact tier) or first-order (smooth tier) | at 24-bit capture, ~1e-4 × the sampling term |
| sampling | circular block bootstrap, serial-correlation-honest | **usually binds** |
| provider | cross-source evidence about the *input* — an input-space diagnostic, not an output-space width | often `unmeasured` |
| model | `not-applicable` for a pure DSL program — the program *is* the computation | N/A |

The composer names the **dominant** term and reports its ratio to quantization.
The composite is a labelled root-sum-square of the independent terms, and is
described as a composition rather than as a single certified scalar.

This table is the paper's most important content. A certificate over storage
quantization *alone* would be certifying the term that does not bind the
decision. The honest headline is "sampling binds ±X; quantization is negligible
at 2.9e-7×", and the budget is what lets that sentence be written.

**Per-input commitments.** For each input: a content digest, the quantization
step Δ, and the dither seed. The Δ values are additionally committed in signed
transparency-log leaves, which defeats a specific attack: a producer who
declares Δ = 0.0 to shrink the certified width. A width whose Δ is anchored in a
leaf signed before the fact is `width_trust: "transparency-anchored"`; one whose
Δ is merely asserted in the envelope is `signer-attested`, and the two are never
conflated.

**A kernel id.** Reductions are order-sensitive in floating point. A
deterministic Rust substrate gives bit-identical reductions across runs, threads
and machines; a NumPy fallback is tested and always available. A certificate
never depends on the extension being installed — but a *width* re-derived on a
substrate that does not match degrades honestly: the scalar is verified to
tolerance and the width is left unverified with a stated reason, rather than
being silently accepted.

## 3. The trust model, stated as a limit

This section exists because the honest version of it is the one people skip.

**What verification establishes.** The inputs named in the envelope were not
revised after they were committed, the program is the program that was signed,
the arithmetic reproduces, and the signature is valid under a key the verifier
pinned out of band.

**What it does not establish.** That the inputs were *right*. A producer who
fabricates data at capture signs a receipt that verifies perfectly. We detect
**revision, not invention.**

This is not a gap to be closed by better cryptography; it is where the guarantee
lives. Any claim of the form "you don't have to trust the issuer" is false for
this object and is on the program's do-not-say list. The accurate phrasing is
*verifiable by replay against your own copy of the inputs, under a key you pin
out of band.*

Three further limits:

- The hash chain protects the **ledger**, not the rows the ledger describes. A
  "tamper-detectable" claim requires re-derivation against current rows, not
  chain verification alone.
- The shipped co-signing witness runs co-located with the signer. It is
  independent only when a party other than the signer operates it. Calling it an
  "independent witness" is prohibited.
- No external party has yet verified an envelope produced by this system. The
  verifier is published; that is a distribution fact, not evidence of external
  verification.

## 4. Refusal as protocol

The tier scheme classifies a program statically by the operations it uses:

- **linear-exact** — the conformal level is a theorem. Exchangeability of the
  dither increments is exact for linear programs, so the *m*-th order statistic
  of the resampled increments is a bound at exactly level *m*/(K+1).
- **smooth-first-order** — exchangeability holds to first order at capture
  deltas; the level is labelled approximate and a coverage harness is the
  falsifier.
- **branch-sensitive** — a dither-sized perturbation can flip a discrete branch
  and no sound first-order object exists. Strict mode **refuses**.
- **branch-stable** — salvaged from the above by per-element margin guards. Every
  decision must be Δ-separated *per element*. An aggregate margin is not
  acceptable here: one benign element can pin an aggregate and defeat the entire
  safety factor. That is a defect this program shipped once and now tests for.

An issuer can additionally demand a tier and receive a refusal rather than a
weaker certificate. The `require_tier` parameter travels in the signed params, so
a verifier reproduces the refusal; one that ignored it would fail to reproduce
and fail closed.

## 5. Conformance, and why the vectors are adversarial

The package ships 49 conformance vectors, the majority of them **forgeries**. Each
forgery must fail *for its own stated reason class* — not merely fail. The
distinction is load-bearing: a verifier that rejects everything passes a
"forgeries are rejected" test and is useless.

The forgery set encodes attacks the system was actually vulnerable to at some
point, including:

- `forgery-delta-fake-zero` — declaring Δ = 0 to shrink the width
- `forgery-shrunk-delta-resigned` — shrinking Δ and re-signing
- `forgery-undercounted-uncertified` — understating how many rows carry no certificate
- `forgery-key-manifest-rollback`, `-checkpoint-equivocation` — key-lifecycle attacks
- `forgery-provider-deleted-silence` — deleting the record of an origin that was asked and did not answer
- `forgery-refusal-reason-resigned` — changing why a refusal happened

A width falsifier deserves specific mention. **A mutation that only widens a
bound passes every coverage test.** Coverage tests are therefore insufficient on
their own, and the suite asserts that width *tracks* Δ rather than merely
covering.

## 6. The producer path

The published package can now produce envelopes, not only verify them. The
fetcher seam takes data a caller holds and a declaration about how it was stored:

```python
from alelyon.runtime.oracle.dsl.fetch import DeclaredFetcher, Declared, CENTS
from alelyon.runtime.oracle.dsl.execcert import certified_run

fetcher = DeclaredFetcher({
    ("price", "ACME"): Declared([100.00, 101.25, 99.50, 102.75, 101.00],
                                delta=CENTS, law="exact-cents/v0"),
})
cert = certified_run('show mean(price("ACME"))', fetcher=fetcher, seed=7)
# value 100.9  ·  width 0.00257  ·  level 0.953125 (exact)  ·  linear-exact
```

**Δ is a declaration and the envelope records it as one.** Declaring `delta=0` on
invented data yields a certificate of width zero over invented data. Nothing in
the producer, the certificate, or the verifier can detect that, and none of them
claim to. The honest reading is: *given that these values were stored under this
law with this step, the answer is x ± w, and anyone can re-derive that.*
Everything before "given" is the producer's to stand behind.

Capture laws are explicit because Δ means different things under different rules.
Under relative dither, Δ = 0 means the column was all zero; under exact-cents it
means the values *are* whole cents, which is a representability claim about every
value and brings an aggregate 2⁵³ guard. An unrecognised law makes a column
unusable rather than permissive — the Δ semantics of a rule nobody implemented
are unknown, not lenient.

## 7. Prior art and positioning

The attestation layer is a competent re-derivation in an established family:
RFC-6962 Merkle transparency logs with inclusion and consistency proofs,
ed25519-signed tree heads, and the SCITT / Sigstore-Rekor / Sigsum design
lineage. We claim assembly and application, not new cryptography.

Against **validated numerics** — Arb, INTLAB, IntervalArithmetic.jl, CAPD, Taylor
models — which give *guaranteed* enclosures over rounding, truncation and
discretization, our width is a probabilistic conformal order statistic over
storage quantization alone. Describing it as "a tight error bound" loses that
comparison twice, and the phrase is prohibited.

Vendor-dispersion-as-uncertainty is Markit Totem, EBA prudent-valuation AVA and
Bloomberg BVAL territory. Truth-discovery is Dong et al. Over-dispersion is the
Birge ratio and the "dark uncertainty" literature. Error-variance-without-truth
is triple collocation. None of those are claimed here.

Where we believe the assembly is genuinely useful is narrow and worth stating
plainly: **control totals and tie-outs in actuarial reconciliation.** They are
sums, counts and means — the exact conformal tier, where the level is a theorem.
`model = not-applicable` is genuinely true because the program *is* the
computation. The tie-out target is held by a different function. And the adversary
is a funded, institutionalised one: three lines of defence plus external audit.

Adjacent sectors we examined and retreated from — metrology (PTB's Digital
Calibration Certificate is free, LGPLv3 and NMI-backed), scientific
reproducibility (beneficiary and payer are different people), crypto oracles (a
contract cannot re-execute pandas), legal forensics (Daubert attacks the model,
not the arithmetic) — are documented as retreats rather than quietly dropped.

## 8. Reproduction

```bash
pip install alelyon-os
alelyon-verify selftest        # runs the bundled conformance suite
alelyon-verify vectors         # lists what is in it
alelyon-verify version         # prints the substrate, and warns on the fallback
```

The specification ships inside the wheel, so an implementer holding the artifact
does not have to find the repository to learn the format.

## 9. What we would want reviewed first

1. Whether the per-element margin guard is genuinely per-element everywhere, not
   just where we tested. This class of defect has bitten twice.
2. Whether the tier classifier's `branch-stable` salvage is sound, or whether
   there is a program it admits that it should refuse.
3. Whether the decomposed budget's independence assumption — for the root-sum-
   square composite — holds for the sampling and quantization terms in cases we
   have not enumerated.
