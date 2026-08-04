# Replayable Certificates over Computation: Binding a Program, Its Inputs, and What the Certificate Does Not Establish

---

## Standing honesty contract

1. **Name the certified object.** An unqualified "certified" launders a small
   guarantee over large ones. The certificates here bound *storage quantization*
   and *replay of a committed computation*. At 24-bit capture the quantization
   term is roughly four orders of magnitude below the sampling term, and the
   decomposed budget states that rather than concealing it.
2. **Revision is detected; invention is not.** A producer who fabricates data at
   capture signs a receipt that verifies perfectly.
3. **An absent measurement is reported as UNMEASURED.** A blank beside a filled
   field reads as "checked, and fine".
4. **Refusal is a first-class outcome.** Refusals in this system are signed and
   carry their real reason.

---

## Abstract

A figure in a report is defensible only if the party who must defend it can
re-derive it. This paper describes a class of signed object that binds four things
together — the program that produced a result, the result, a decomposed error
budget, and a per-input commitment consisting of a content digest, a quantization
step and a dither seed — such that a third party holding independently obtained
inputs and a public key pinned out of band can re-execute the program and either
reproduce the result or learn precisely which component failed. The same
construction is exhibited in two domains: a computed scalar produced by a
restricted domain-specific language, and an exactly invertible chain of coordinate
transforms. The domains share a canonical byte encoding, a transparency-log
lineage in the RFC 6962 family, an adversarial conformance suite in which the
majority of vectors are forgeries that must each fail for their own stated reason,
and a discipline under which the certificate's *absences* are signed content
rather than blank fields. Four audits are reported. Element-wise margin guards are
per-element at every site reachable in the certified-execution path. The
first-order branch-stability salvage is shown to be **unsound** for decisions taken
on aggregates when the storage error is not dithered, certifying a flipped
decision at width zero; the exact tier is immune and 14,833 firings produced no
violation. The root-sum-square composite's independence assumption is measurably
false but fails conservatively, and the composite's coverage shortfall is
attributable to the bootstrap rather than to the composition. The canonical
encoding is shown to be canonical in fact and **not in specification**: a second
encoder written from the governing architecture decision record derives a
different content reference for the same object.

**Keywords:** transparency logs, verifiable computation, replay verification,
canonical encoding, conformal prediction, error budgets, refusal semantics,
adversarial conformance testing.

---

## 1. Introduction

A spreadsheet cell reads 4,182,905.33. Six weeks later an auditor asks where it
came from. The available answers are ordinarily a screenshot, a re-run that no
longer reproduces, or a person who has left.

Verifiable computation has a large literature and mature machinery. What it tends
to lack at the point where a number enters a report is anything that survives the
gap between the party who computed the number and the party who must defend it.
The gap is organizational rather than cryptographic: the computation ran, in an
environment that has since changed, against data that has since been revised, by a
process nobody recorded.

The concrete setting that shaped this design is an actuarial control total. A
reserve figure is reconciled against a figure held by a *different function* — the
general ledger, an administration system, a reinsurer. That is a genuinely
independently held invariant, and it is what makes the verification meaningful
rather than circular: the tie-out target was not produced by the party being
checked.

This paper describes the signed object that supports such a reconciliation,
exhibits it in two unrelated domains, and — at greater length than is customary —
enumerates what it does not establish.

### 1.1 Contributions

1. A signed envelope binding a restricted program, its scalar result, a decomposed
   error budget and per-input commitments, replayable under an out-of-band key
   (§4).
2. The same construction applied to exactly invertible coordinate transform
   chains, where the absence of a similarity score forces the question of what is
   worth signing (§5).
3. A discipline in which a certificate's absences are *inside the signed bytes*,
   with a completeness test against the governing specification's field list, so
   that a partial implementation cannot present as a complete one (§6).
4. A static tier scheme classifying programs by the soundness of the available
   quantization bound, in which the unsound case refuses rather than degrades, and
   the demanded tier travels in the signed parameters (§7).
5. An adversarial conformance suite of 49 vectors, a majority of them forgeries,
   each required to fail for its own stated reason class; and two red-team defects
   that generalize to verifier construction at large (§8).
6. Four new audit results, one of which is a confirmed unsoundness in a shipped
   salvage path, reported with its exact premise boundary (§9).

### 1.2 Position in the series

Paper IV records why a certificate rather than a compression ratio became the
surviving artifact of this program. Paper II develops the certificate for
statistics computed from dithered storage. This paper concerns the *object* — how
a result is bound to its inputs and to a key — and is largely independent of what
the result is.

---

## 2. Notation and preliminaries

| Symbol | Meaning |
|---|---|
| Δ, Δ_i | Quantization step; per-row where subscripted |
| K | Number of dither resamples in the conformal procedure (K = 63) |
| m | Order statistic index; the certified level is m/(K+1) |
| α | Miscoverage target; level = 1 − α |
| w | Certified half-width on the result |
| q, s | Quantization and sampling half-widths in the decomposed budget |
| n | Number of rows consumed by a program |
| σ_min(M) | Smallest singular value of M |

**Definition 2.1 (subtractive dither).** A value x is stored as
y = Q(x + u) − u with u a dither signal independent of x, uniform on
(−Δ/2, Δ/2), and Q uniform quantization at step Δ. Under the Schuchman conditions
and absent overload, the error y − x is uniform on (−Δ/2, Δ/2), independent of x,
and independent across elements [11, 12, 13]. This is the storage model Paper II
develops in full; it is stated here because §9.2's finding turns on the difference
between it and a merely interval-bounded error.

**Definition 2.2 (interval-bounded storage).** A value x is stored as y with the
sole guarantee |y − x| ≤ Δ/2. Definition 2.1 implies Definition 2.2. The converse
is false, and §9.2 is the consequence.

**Definition 2.3 (out-of-band key pinning).** A verifier holds a public key
obtained through a channel that does not pass through the artifact under
verification. A key transported inside the object it authenticates authenticates
nothing.

**Definition 2.4 (replay verification).** Given a certificate, independently
obtained inputs, and a pinned public key, the verifier re-executes the committed
program over the inputs and compares against the committed result, reporting
either agreement or the specific component that disagreed.

---

## 3. Threat model and trust boundary

This section is consolidated from both domains because the boundary is identical
in each, and because the honest statement of it is the one readers skip.

### 3.1 What verification establishes

- The inputs named in the certificate were not revised after they were committed.
- The program is the program that was signed.
- The arithmetic reproduces, to the stated tolerance on the stated substrate.
- The signature is valid under a key the verifier pinned out of band
  (Definition 2.3).
- For the registration domain additionally: the committed bytes are the canonical
  encoding of the object they decode to; every reconstructed record satisfies its
  own contract invariants, because decoding rebuilds them through their ordinary
  constructors; and declared space references, loss class and invertibility are
  cross-checked against what the replay independently re-derived rather than read
  off the certificate and believed.

### 3.2 What verification does not establish

**That the inputs were right.** A producer who fabricates data at capture signs a
receipt that verifies perfectly. This system detects **revision, not invention**.
This is not a deficiency to be closed by better cryptography; it is where the
guarantee lives. Any claim of the form "the issuer need not be trusted" is false
for this object. The accurate phrasing is *verifiable by replay against
independently held inputs, under a key pinned out of band*.

**That the hash chain protects the data.** The chain protects the *ledger*, not
the rows the ledger describes. A tamper-detection claim requires re-derivation
against current rows, not chain verification alone.

**That the witness is independent.** The distributed co-signing witness runs
co-located with the signer. It is independent only when operated by a party other
than the signer. The term "independent witness" is prohibited in this program's
documentation for that reason.

**That the verifier is an independent implementation.** Verification shares the
contract, transform and canonical-encoding modules with the issuer, so a defect in
those modules is reproduced rather than caught. A second implementation written
from the specification is the only thing that would establish otherwise. §9.4
reports the first attempt at one, and its result is negative.

**That any external party has verified anything.** The verifier is distributed.
That is a distribution fact and not evidence of external verification. No external
verification is recorded.

**That the registration is correct for any dataset.** A semantically wrong but
internally consistent transform chain verifies cleanly, exactly as it replays
cleanly.

**That the signer is anyone in particular.** A signature binds bytes to a key.
Who holds the key is a question no certificate can answer.

---

## 4. Instance A: the certified number envelope

An `alelyon.cne/v0` envelope carries the following.

### 4.1 The program

Not free-form code: a restricted language with no evaluation of dynamic source, no
imports, no attribute access, and no filesystem or network reach. The program text
is hashed into the signed bytes, and a verifier parses and executes the same
source. Re-executing the actual interpreter rather than a model of it eliminates
the semantics-drift failure class that an analytic error calculus would carry
permanently.

### 4.2 The result, or the refusal

A refusal is a signed, first-class outcome carrying its real reason. This matters
more than it appears to. A system that degrades silently produces a number under
conditions in which it should have produced nothing, and no downstream consumer
can distinguish the two.

### 4.3 The decomposed error budget

**Table 1.** Budget terms, how each is obtained, and typical magnitude. The
composite is never presented as a single certified scalar.

| Term | How obtained | Typical magnitude |
|---|---|---|
| quantization | theorem (linear-exact tier) or first-order (smooth tier) | ≈ 1.0e-4 × the sampling term at 24-bit capture |
| sampling | circular block bootstrap [14], serial-correlation honest | usually binds |
| provider | cross-source evidence about the *input*; an input-space diagnostic, not an output-space width | frequently `unmeasured` |
| model | `not-applicable` for a pure program — the program *is* the computation | N/A |

The composer names the **dominant** term and reports its ratio to quantization.
The composite is a labeled root-sum-square of the independent terms and is
described as a composition.

This table is the most consequential content in the envelope. A certificate over
storage quantization alone would certify the term that does not bind the decision.
The honest headline is of the form "sampling binds ±X; quantization is negligible
at 2.9e-7×", and the decomposed budget is what permits that sentence to be
written. §9.3 reports an audit of the composite's independence assumption.

### 4.4 Per-input commitments

For each input: a content digest, the quantization step Δ, and the dither seed.
The Δ values are additionally committed in signed transparency-log leaves, which
defeats a specific attack — a producer declaring Δ = 0 to shrink the certified
width. A width whose Δ is anchored in a leaf signed before the fact carries
`width_trust: "transparency-anchored"`; one whose Δ is merely asserted in the
envelope carries `signer-attested`. The two are never conflated.

**Δ is a declaration, and the envelope records it as one.** Declaring Δ = 0 on
invented data yields a certificate of width zero over invented data. Nothing in the
producer, the certificate or the verifier detects this, and none of them claim to.
The honest reading of an envelope is: *given that these values were stored under
this law with this step, the result is x ± w, and the derivation is reproducible.*
Everything before "given" belongs to the producer.

Capture laws are explicit because Δ carries different meaning under different
rules. Under relative dither, Δ = 0 means the column was identically zero; under
an exact-cents law it means the values *are* whole cents, which is a
representability claim about every value and carries an aggregate 2⁵³ guard. An
unrecognized law renders a column unusable rather than permissive: the Δ semantics
of a rule nobody implemented are unknown, not lenient.

### 4.5 The kernel identifier

Floating-point reductions are order-sensitive. A deterministic substrate yields
bit-identical reductions across runs, threads and machines; a portable fallback is
tested and always available. A certificate never depends on the accelerated
extension being installed, but a width re-derived on a non-matching substrate
degrades explicitly: the scalar is verified to tolerance and the width is left
unverified with a stated reason, rather than being accepted silently. A width of
exactly zero is substrate-independent, because with every Δ = 0 the perturbation
is identically zero on any kernel.

---

## 5. Instance B: exact coordinate registration

Medical image registration supplies a mature vocabulary — fixed space, moving
space, transform chain, resampling, and a certificate of what was performed —
developed for data in which every step is lossy and the question is *how much* was
lost. This section applies that vocabulary where the answer can be *nothing*:
registration between coordinate spaces whose transforms are exactly invertible
over the rationals.

### 5.1 Why exactness changes the problem

In the lossy setting a registration is judged by a similarity metric and the
certificate reports a score. In the exact setting there is no score, because the
composed map either is or is not invertible over the field it operates in. That
removes the customary content of a certificate and substitutes a harder question:
what is worth signing?

The answer adopted here: the two coordinate spaces, the transform chain relating
them, the conditions under which the chain was produced, the loss class, the
invertibility, and a complete enumeration of what the certificate does not contain.

### 5.2 Structural rather than numerical losslessness

Losslessness is established by contract invariants on the composed chain rather
than by numerical comparison of round-tripped values. Every transform declares its
loss class and invertibility as class constants rather than constructor fields, so
a caller cannot construct a transform that misreports what it does. A chain
declares the weakest class any member declares. The capability surface is
committed in the encoding and **recomputed from the reconstructed transform on
decode**, so bytes claiming a stronger class than the type possesses are refused
rather than adopted.

Whether the contract invariants are sufficient to make *structurally invertible*
equivalent to *invertible* — whether some chain satisfies every invariant and is
not invertible — remains open and is listed in §10.

---

## 6. Absences as signed content

The governing specification for the registration certificate declares 34 fields.
The implementation populates 13. The remaining 21 are carried in the certificate
as named absences, each with a reason and a kind:

- **NOT_APPLICABLE** — the mechanism cannot apply to exact registration. There is
  no similarity objective to optimize, so no field claims an optimum.
- **UNMEASURED** — the mechanism could apply and nothing measured it. There is no
  artifact manifest, template registry, search plan, metric registry, payload
  remapping, uncertainty propagation, or execution trace.

The absences are **inside the signed bytes**. A certificate therefore cannot
understate what it omits: shrinking the absence list changes the signature. A test
asserts the declared absences are complete against the specification's field list,
so a newly specified field cannot be silently ignored — it must be populated or
named absent.

This is the design decision this paper would most recommend for adoption
elsewhere. The failure mode it prevents is the characteristic failure of every
partial implementation: a certificate with 13 populated fields and 21 blanks
reads, to anyone who has not memorized the specification, as a certificate that
checked 34 things.

### 6.1 Tripwire tests

Several tests in this subsystem fail deliberately when a boundary is crossed, and
their docstrings name the debt to be paid rather than the assertion to be updated.
The pattern: a test asserts that exactly 21 fields are absent. When an implementer
populates one, the test fails, and the correct response is to update the declared
absence list *and* the specification cross-reference — not to change the number. A
test whose failure mode is "increment the constant" will be incremented without
thought; a test whose docstring explains why the number is what it is at least
compels a reading.

---

## 7. Refusal as protocol

Programs are classified statically by the operations they use.

**Table 2.** Tier scheme. The demanded tier travels in the signed parameters, so a
verifier reproduces a refusal; a verifier ignoring it fails to reproduce and fails
closed.

| Tier | Basis | Level |
|---|---|---|
| `linear-exact` | Exchangeability of the dither increments is exact for linear programs, so the m-th order statistic of the resampled increments bounds at level exactly m/(K+1) | Theorem |
| `smooth-first-order` | Exchangeability holds to first order at capture deltas | Approximate; coverage harness is the falsifier |
| `branch-sensitive` | A dither-sized perturbation can flip a discrete branch and no sound first-order object exists | **Refuses** |
| `branch-stable-exact` | Every decision applied directly to certified data and Δ-separated per element | Theorem |
| `branch-stable-first-order` | Decisions identical across all K resamples and margins clearing a safety multiple of the observed perturbation | Approximate — **and see §9.2** |

The two salvage guards are independent. The deterministic guard certifies the
exact tier on its own: for an extremum, every competitor gap must exceed
(Δ_win + Δ_j)/2; for a sign or comparison against a constant, every element must sit
further than Δ_i/2 from its threshold. Then |stored − true| ≤ Δ_i/2 per element
makes neither the true data nor any dither resample flip the decision, the program
reduces to a fixed linear selection, and the conformal level is exact.

An aggregate margin is not acceptable here. One benign element can pin an
aggregate and defeat the entire safety factor. That is a defect this program
shipped once; §9.1 reports the audit that confirms it is now closed everywhere,
and §9.2 reports a different defect in the same neighborhood that is not.

---

## 8. Adversarial conformance

The package distributes 49 conformance vectors, a majority of them forgeries. Each
forgery must fail **for its own stated reason class**, not merely fail. The
distinction is load-bearing: a verifier that rejects everything passes a
"forgeries are rejected" test and is useless.

The forgery set encodes attacks the system was vulnerable to at some point,
including: declaring Δ = 0 to shrink the width; shrinking Δ and re-signing;
understating how many rows carry no certificate; key-manifest rollback and
checkpoint equivocation; deleting the record of an origin that was asked and did
not answer; and altering the reason a refusal was issued.

**A width falsifier deserves specific mention.** A mutation that only *widens* a
bound passes every coverage test. Coverage tests are therefore insufficient alone,
and the suite asserts that width *tracks* Δ rather than merely covering.

A generated vector suite is circular unless each case declares by hand the reason
class it exists to exercise, with generation failing if the verifier does not
produce it.

### 8.1 Two red-team findings that generalize

**A read-only view is not an immutable mapping.** A read-only proxy over a mutable
mapping leaves the underlying mapping writable through any other reference to it,
so a nominally frozen contract object could be edited after construction and
before signing. The distinction between *this reference cannot mutate it* and
*nothing can mutate it* is the entirety of the defect. In the language runtime
used here, a proxy additionally surrenders the underlying object to garbage-
collector reflection, so what actually holds is that the proxy is the mapping's
only holder — a property a test must assert rather than assume.

**Per-field limits are not a resource budget.** Bounding each field independently
does not bound the document. A certificate composed of many individually legal
fields is a resource-exhaustion vector against any verifier that accepts it.
Anything parsing untrusted structured input requires an aggregate budget in
addition to per-field limits, and per-field limits alone read as protection while
providing none.

Both findings concern verifier construction generally rather than either engine,
which is why they appear here.

---

## 9. Four audits

The audits below closed questions this work previously listed as open. Two returned
negative for the implementation. Both are reported, because a suppressed negative
is worse evidence than a reported one. Full protocols and figures are in the
research log accompanying this series.

### 9.1 Element-wise guards are per-element everywhere

**Result: confirmed.** Both guards are per-element at every site reachable in the
certified-execution path. The deterministic guard is literally per-element, pairing
each element's own Δ with the winner's. The empirical guard compares the *minimum*
per-element margin against the *maximum* per-element perturbation, which is
strictly stronger than a per-element test: min_i margin_i > c · max_i pert_i
implies margin_j > c · pert_j for every j.

Evidence: 12 isolated cases against the deterministic guards, including the
aggregate-masking shape originally found by the red team — a benign element with a
tiny Δ pinning the series minimum while a different element sits inside its own
Δ/2 — all correct; 5 end-to-end programs, adversarial cases refusing and positive
controls certifying at the exact tier; and a repository-wide search for a margin
quantity combined with an aggregation, returning no site that aggregates a guard.

### 9.2 The first-order branch-stability salvage is unsound on aggregates

**Result: confirmed defect.** This is the most consequential finding in the paper.

```
certified_run('show sign(mean(series("X")) - c)')
  →  ok = True, tier = branch-stable-first-order, width = 0.0
```

while the true decision carries the opposite sign. A certificate of width zero over
a flipped answer.

**Mechanism.** The empirical guard compares the base margin against the
perturbation observed across K *independent* dither resamples. Independent
resampling displaces an n-row aggregate by Θ(Δ/√n). A systematic rounding obeying
the same per-element bound |stored − true| ≤ Δ/2 displaces it by Θ(Δ). The fixed
safety multiple of 3 is therefore defeated once √n exceeds it.

**Table 3.** Measured behavior against n, at Δ = 0.1 with a systematic rounding of
0.0499 per element. The crossover falls where the mechanism predicts.

| n | √n | certificate | decision flipped? |
|---|---|---|---|
| 25 | 5.0 | refuses | yes |
| 100 | 10.0 | **certifies** | yes |
| 400 | 20.0 | **certifies** | yes |
| 1600 | 40.0 | **certifies** | yes |

**Premise boundary.** Under genuine subtractive dither (Definition 2.1) the guard
is **sound**: 0 of 400 draws flipped the decision. The exposure is that the
producer-side declaration defaults to a relative-dither law while the module's
prose teaches the *interval* reading of Definition 2.2 — "the width of the interval
a stored value could have come from". The guard requires the independence reading.
Nothing checks which reading the caller's data satisfies.

This is the program's own structural rule — validate against an independently held
invariant, never against the shape of what the writer emitted — recurring one level
up. The resampler validates against the writer's declared Δ *under an assumed law*.

**Not affected.** The exact tier is immune by construction, because its
deterministic guard uses the worst-case per-element Δ/2, which is systematic-safe.
Randomized search with adversarially thin margins produced **14,833 exact-tier
firings and 0 soundness violations** across sign, minimum, maximum and
comparison-against-constant.

**Status.** Audit only; not remediated, because an audit in this program does not
authorize remediation. Two fix directions are available: the empirical guard's
perturbation scale for a decision on an aggregate of n certified rows must use the
worst-case Δ/2 rather than the resample spread; or the producer surface must require
an explicit capture law and refuse aggregate-grounded branch decisions under any
law that does not guarantee independence.

### 9.3 The composite's independence assumption is false and conservative

**Result: the assumption does not hold, it fails in the safe direction, and the
concern belongs elsewhere.**

The correlation between the quantization and sampling error components was measured
across replications on AR(1) log-returns. It is significantly negative and grows
with the quantization-to-sampling ratio: −0.031 at a ratio of 1.6e-6, −0.343 at
4.8e-2, and −0.388 with 95% confidence interval [−0.457, −0.314] at a ratio of
3.4e-1 (n = 600). Negative covariance implies Var(q+s) < Var(q) + Var(s), so the
root-sum-square **overstates** the combined width. The composite is never narrower
than the truth on this evidence. At the ratio at which the platform operates —
approximately 1.0e-4 at 24-bit capture — the correlation is indistinguishable from
zero.

**The more useful result.** The composite's coverage shortfall is not attributable
to the composition. At Δ = 0, with no quantization term present at all, coverage is
**0.923 ± 0.011** against a 0.95 target (n = 600). The deficit belongs to the
circular block bootstrap's finite-sample behavior on serially correlated data.
Adding the quantization term moves coverage *up*, to 0.973 at Δ = 8. The open
question named the composition; the measurement names the sampling estimator.

### 9.4 The canonical encoding is canonical in fact, not in specification

**Result: confirmed fork.** The governing architecture decision record states that
field order is the declared dataclass order. For the coordinate-space record, the
schema-version field is declared eleventh and encoded **first**. A second encoder
written from that record's rules — deriving field order mechanically, as an
independent implementer would — produces a different content reference for the same
object:

| Encoder | Content reference for the same space |
|---|---|
| Distributed implementation | `sha256:7b4aad345c23bfa824e8f82313f08ec4…` |
| Second encoder, specification read literally | `sha256:9b39c66dc51b634e6fe06f13f05b9f68…` |

A second confirmed ambiguity: the space record carries a field whose name lacks the
leading underscore that marks the two excluded fields on the axis record, and which
nothing in the specification excludes, yet which is not encoded. An implementer
including it forks a third way.

One arm closed clean. The specification's requirement that strings be
NFC-normalized does **not** fork: the encoder never normalizes, but the contract
refuses non-NFC text at construction, so the encoder never receives a non-normal
string. Refusing rather than silently normalizing is the correct choice.

A positive control confirms the method: the **axis** encoding reproduced
bit-for-bit from the specification's rules alone, so the divergence above is signal
rather than an artifact of a careless second encoder.

**Classification.** This is a *specification* defect, not an implementation defect.
The distributed bytes are pinned by golden vectors and are the de facto standard.
What is wrong is the written rule that would permit a second party to reproduce
them — which is precisely the property a canonical encoding exists to provide. The
architecture decision record records this gap itself, stating that a
cross-implementation decoder remains unmeasured and that the encoding has one
implementation. This is the first measurement of it.

---

## 10. Limitations and open problems

1. **§9.2 is an open defect.** The first-order salvage tier is unsound under
   interval-bounded storage. It is documented, not fixed.
2. **§9.4 is an open specification defect.** A second implementation cannot
   reproduce the commitment from the written rules.
3. **Verification is not an independent implementation.** Per §3.2, the verifier
   shares modules with the issuer. §9.4's second encoder covers the encoding layer
   only, and found a divergence.
4. **Whether structural invertibility implies invertibility is open.** No chain
   satisfying every contract invariant and failing to be invertible is known; none
   is proven not to exist.
5. **The aggregate resource budget is incomplete.** Per §8.1, per-field limits are
   known to be insufficient; the aggregate budget is stated as a requirement and
   has not been audited to the standard of §9.
6. **No external verification is recorded.** Per §3.2.
7. **The sampling term's coverage is below its nominal level.** Per §9.3, measured
   at 0.923 ± 0.011 against 0.95 on AR(1) data. This is a property of the
   estimator; a block-length selection rule with better finite-sample behavior is
   the obvious direction and is UNMEASURED here.
8. **The provider term is unmeasured in this deployment.** No cross-source
   dispersion figure has been measured from this host, and cross-source
   independence is not a measurable property in any case: declared lineage
   undercounts copying, and identical-fraction over history can disprove
   independence but never establish it.

---

## 11. Related work

**Transparency logs and attestation.** The attestation layer is a competent
re-derivation within an established family: RFC 6962 Merkle transparency logs with
inclusion and consistency proofs [1], Merkle trees [2], Ed25519-signed tree
heads [3], and the design lineage running through Sigstore/Rekor [5], Sigsum [6]
and the SCITT architecture [4]. Assembly and application are claimed; cryptography
is not.

**Canonical encoding.** The requirement that one object have exactly one byte
encoding is the same requirement addressed by JSON canonicalization [7] and by the
distinguished encoding rules of ASN.1. §9.4 is a demonstration that meeting it in
implementation is not the same as meeting it in specification.

**Validated numerics.** Against interval and Taylor-model arithmetic — Arb [8],
INTLAB [9], and the interval-analysis tradition from Moore [10] — which give
*guaranteed* enclosures over rounding, truncation and discretization, the width
here is a probabilistic conformal order statistic over storage quantization alone.
Describing it as a tight error bound loses that comparison twice, and the phrase is
prohibited in this program.

**Conformal prediction.** The level guarantee is the standard exchangeability
argument [15, 16]; the contribution is identifying the class of programs for which
exchangeability of dither increments is exact rather than assumed (§7).

**Dither theory.** Definition 2.1 is Schuchman's condition [11] as developed by
Lipshitz, Wannamaker and Vanderkooy [12] and Gray and Stockham [13]. Paper II
treats it in full.

**Uncertainty from source dispersion.** Vendor-dispersion-as-uncertainty is the
territory of consensus pricing services and of prudent-valuation additional
valuation adjustments in bank regulation. Truth discovery from conflicting sources
is Dong et al. [17]; over-dispersion of stated uncertainties is the Birge
ratio [18]; error variance without a reference truth is triple collocation [19].
None of these is claimed here.

**Where the assembly is believed useful** is narrow and worth stating plainly:
control totals and tie-outs in actuarial reconciliation. They are sums, counts and
means — the exact conformal tier, where the level is a theorem. The model term is
genuinely not applicable because the program *is* the computation. The tie-out
target is held by a different function. And the adversary is funded and
institutionalized: three lines of defense plus external audit.

Adjacent domains examined and retreated from — metrology, where a national
metrology institute's digital calibration certificate is free and better supported;
scientific reproducibility, where beneficiary and payer differ; crypto oracles,
where a contract cannot re-execute a general-purpose data pipeline; legal
forensics, where admissibility challenges attack the model rather than the
arithmetic — are recorded as retreats rather than quietly dropped.

---

## 12. Conclusion

A certificate over a computation is useful in proportion to the precision with
which it states its own boundary. The object described here binds a program, a
result, a decomposed error budget and per-input commitments, and is replayable by a
party who holds the inputs independently and pins the key out of band. It detects
revision and not invention, and it says so. Its absences are signed content, so a
partial implementation cannot present as a complete one. Its tier scheme refuses
where no sound bound exists rather than degrading silently.

Two of the four audits reported here returned against the implementation: a
first-order salvage tier that certifies a flipped decision at width zero when the
storage error is bounded but not dithered, and a canonical encoding whose written
specification does not reproduce its own bytes. Both are consequences of the same
underlying pattern — a guarantee resting on a premise that nothing in the system
checks. That pattern, rather than either defect, is what this paper would put to a
reviewer first.

---

## Appendix A — Reproduction

```bash
pip install alelyon-os
alelyon-verify selftest        # runs the distributed conformance suite
alelyon-verify vectors         # lists the vectors and their reason classes
alelyon-verify version         # prints the substrate and warns on the fallback
```

Producing an envelope from data the caller holds:

```python
from alelyon.runtime.oracle.dsl.fetch import DeclaredFetcher, Declared, CENTS
from alelyon.runtime.oracle.dsl.execcert import certified_run

fetcher = DeclaredFetcher({
    ("price", "ACME"): Declared([100.00, 101.25, 99.50, 102.75, 101.00],
                                delta=CENTS, law="exact-cents/v0"),
})
cert = certified_run('show mean(price("ACME"))', fetcher=fetcher, seed=7)
# value 100.9 · width 0.00257 · level 0.953125 (exact) · linear-exact
```

The registration engine's declared absences and the specification field list:

```bash
pip install "alelyon-os[lattice]"
python -c "import alelyon.runtime.vector.lattice as L; print(sorted(L.DECLARED_ABSENCES)); print(L.SPEC_CERTIFICATE_FIELDS)"
```

**Environment.** Python 3.12; canonical interpreter `.venv312/Scripts/python.exe`.
Expect 3 of 9 golden vectors verified on the portable fallback and 9 of 9 with the
deterministic kernel; a claim of 9 of 9 on a fallback build indicates a fault. The
specification ships inside the distribution, so an implementer holding the artifact
need not locate the repository to learn the format.

**Audit reproduction (§9).** The four audits are recorded with their protocols,
seeds and full figures in `docs/papers/00-research-log.md`. All are offline and
require no network access.

**Data availability.** §9.3's replications are synthetic and fully specified by the
stated AR(1) parameters and seeds. §9.1, §9.2 and §9.4 operate on constructed
inputs and on the distributed package's own fixtures; none requires proprietary
data.

---

## Appendix B — Notation table

| Symbol | Definition | First use |
|---|---|---|
| Δ, Δ_i | Quantization step; per-row where subscripted | §2 |
| K | Dither resamples, K = 63 | §2 |
| m | Order statistic index | §2 |
| m/(K+1) | Certified conformal level | §7 |
| w | Certified half-width | §2 |
| q, s | Quantization, sampling half-widths | Table 1 |
| n | Rows consumed by a program | §9.2 |
| c | Branch-margin safety multiple (c = 3) | §9.2 |
| gap_j | Competitor gap in the extremum guard | §7 |

---

## References

The identifiers below were recorded from the authors' working bibliography and
should be checked against the primary sources before external publication; this
series has not yet had a bibliographic review.

[1] B. Laurie, A. Langley, E. Kasper. *Certificate Transparency.* RFC 6962, IETF,
2013.

[2] R. C. Merkle. *A Digital Signature Based on a Conventional Encryption
Function.* CRYPTO 1987.

[3] D. J. Bernstein, N. Duif, T. Lange, P. Schwabe, B.-Y. Yang. *High-Speed
High-Security Signatures.* Journal of Cryptographic Engineering 2(2), 2012.

[4] IETF SCITT Working Group. *An Architecture for Trustworthy and Transparent
Digital Supply Chains.* Internet-Draft.

[5] Sigstore project. *Rekor: a transparency log for software supply-chain
artifacts.* Software.

[6] Sigsum project. *Sigsum: minimal signed-checksum transparency logging.*
Software and specification.

[7] A. Rundgren, B. Jordan, S. Erdtman. *JSON Canonicalization Scheme (JCS).*
RFC 8785, IETF, 2020.

[8] F. Johansson. *Arb: Efficient Arbitrary-Precision Midpoint-Radius Interval
Arithmetic.* IEEE Transactions on Computers 66(8), 2017.

[9] S. M. Rump. *INTLAB — INTerval LABoratory.* In *Developments in Reliable
Computing*, Kluwer, 1999.

[10] R. E. Moore. *Interval Analysis.* Prentice-Hall, 1966.

[11] L. Schuchman. *Dither Signals and Their Effect on Quantization Noise.* IEEE
Transactions on Communication Technology 12(4), 1964.

[12] S. P. Lipshitz, R. A. Wannamaker, J. Vanderkooy. *Quantization and Dither: A
Theoretical Survey.* Journal of the Audio Engineering Society 40(5), 1992.

[13] R. M. Gray, T. G. Stockham. *Dithered Quantizers.* IEEE Transactions on
Information Theory 39(3), 1993.

[14] D. N. Politis, J. P. Romano. *A Circular Block-Resampling Procedure for
Stationary Data.* In *Exploring the Limits of Bootstrap*, Wiley, 1992.

[15] V. Vovk, A. Gammerman, G. Shafer. *Algorithmic Learning in a Random World.*
Springer, 2005.

[16] J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman.
*Distribution-Free Predictive Inference for Regression.* Journal of the American
Statistical Association 113(523), 2018.

[17] X. L. Dong, L. Berti-Equille, D. Srivastava. *Integrating Conflicting Data:
The Role of Source Dependence.* VLDB 2009.

[18] R. T. Birge. *The Calculation of Errors by the Method of Least Squares.*
Physical Review 40(2), 1932.

[19] A. Stoffelen. *Toward the True Near-Surface Wind Speed: Error Modeling and
Calibration Using Triple Collocation.* Journal of Geophysical Research 103(C4),
1998.
