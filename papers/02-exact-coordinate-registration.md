# Exact Coordinate Registration

### Lossless transform chains with canonical byte commitments and a signed certificate

**Status:** systems paper, early-stage engine. Survived an adversarial red team
(2026-08-02) that found two real defects, both described below. Decisions are
recorded in ADR-0001 through ADR-0014.

---

## Abstract

Medical image registration has a mature vocabulary — fixed space, moving space,
transform chain, resampling, a certificate of what was done — built for data
where every step is lossy and the question is *how much* was lost. We take that
vocabulary to a setting where the answer can be *nothing*: registration between
coordinate spaces whose transforms are exactly invertible over the rationals.

The engine composes transforms, proves losslessness structurally rather than
numerically, commits the chain to canonical bytes, and issues a signed
certificate that a party who was not present can replay. The interesting
engineering is almost entirely in what the certificate *refuses* to say.

## 1. Why exactness changes the problem

In the lossy setting, a registration is judged by a similarity metric and the
certificate reports a score. In the exact setting there is no score to report,
because the composed map either is or is not invertible over the field it
operates in. That removes the usual content of a certificate and replaces it with
a harder question: what *is* worth signing?

Our answer: the two coordinate spaces, the transform chain that maps one to the
other, the conditions under which the chain was produced, the loss class, the
invertibility, and — critically — **a complete enumeration of what the
certificate does not contain**.

## 2. Absences are signed content

The governing specification declares 34 certificate fields. This implementation
populates 14. The other 20 are carried in the certificate as named absences,
each with a reason and a kind:

- **NOT_APPLICABLE** — the mechanism cannot apply to exact registration. There is
  no similarity objective to optimise, so no field claims an optimum.
- **UNMEASURED** — the mechanism could apply and nothing measured it. There is no
  artifact manifest, no template registry, no search plan, no metric registry, no
  payload remapping, no uncertainty propagation, no execution trace.

The absences are **inside the signed bytes**. A certificate therefore cannot
understate what it leaves out: shrinking the absence list changes the signature.
A test asserts the declared absences are complete against the specification's
field list, so a new spec field cannot be silently ignored — it must be populated
or named absent.

This is the design decision we would most want other people to copy. The failure
mode it prevents is the one every partial implementation has: a certificate with
13 filled fields and 21 blanks reads, to anyone who has not memorised the spec,
as a certificate that checked 34 things.

## 3. What a CERTIFICATE_VERIFIED report establishes

- The bytes are canonical, hash to the reference they are addressed by, and carry
  a valid signature under the public key **the caller pinned**.
- The committed chain has not been revised, is the canonical encoding of the chain
  it decodes to, satisfies every contract invariant, and reproduces each claimed
  source coordinate.
- The chain the caller holds is the one the certificate is about — declared space
  references, loss class and invertibility are cross-checked against what the
  replay independently re-derived, rather than read off the certificate and
  believed.

## 4. What it does not establish

Stated at the same volume, because this list is longer than the one above:

- **Not that the registration is correct for any dataset.** A semantically wrong
  but internally consistent chain verifies cleanly, exactly as it replays
  cleanly.
- **Not optimality.** Nothing searches an objective, so no field claims a bound.
- **Nothing about payload values.** No artifact is read and no value is remapped.
- **Not an independent implementation.** Verification shares the contract,
  transform and canonical modules with the issuer, so a defect in those is
  reproduced rather than caught. A second implementation from the specification
  is the only thing that would establish otherwise, and it does not exist.
- **Not that the signer is anyone in particular.** A signature binds bytes to a
  key. Who holds the key is a question the certificate cannot answer.

## 5. Two red-team findings worth publishing

**`MappingProxyType` does not freeze.** A read-only view over a mutable mapping
is not an immutable mapping. The underlying dict remains mutable through any
other reference to it, so a "frozen" contract object could be edited after
construction and before signing. The distinction between *this reference cannot
mutate it* and *nothing can mutate it* is the whole of the defect.

**Per-field limits are not a resource budget.** Bounding the size of each field
independently does not bound the size of the document. A certificate with many
individually-legal fields is a resource-exhaustion vector against any verifier
that accepts it. Anything parsing untrusted structured input needs an aggregate
budget in addition to per-field limits, and per-field limits alone read as
protection while providing none.

Both generalise well beyond this engine, which is why they are here rather than
only in the audit log.

## 6. Tripwire tests

Several tests in this subsystem **fail on purpose** when a boundary is crossed.
Their docstrings name the debt to be paid, not an assertion to be bumped.

The pattern: a test asserts that exactly 21 fields are absent. When a future
implementer populates one, the test fails. The correct response is to update the
declared-absence list *and* the specification cross-reference — not to change the
number. A test whose failure mode is "increment the constant" is a test that will
be incremented without thought; one whose docstring says *why* the number is what
it is at least forces a reading.

## 7. Naming

*Lattice* is the product name for the AI assistant surface, not for this engine.
The engine keeps the `alelyon.runtime.vector.lattice` namespace that ADR-0001
froze; ADR-0003 records the collision and the decision to live with it. The word
also does **not** refer to lattice-based cryptography, which is a different field
entirely — this is registration on an integer/rational coordinate grid.

We flag the ambiguity because it is a genuine source of confusion and because we
walked into it ourselves.

## 8. Reproduction

```bash
pip install "alelyon-os[lattice]"
python -c "
import alelyon.runtime.vector.lattice as L
print(sorted(L.DECLARED_ABSENCES))     # the 21, with reasons
print(L.SPEC_CERTIFICATE_FIELDS)       # the 34 the spec declares
"
```

## 9. What we would want reviewed first

1. Whether the canonical encoding is genuinely canonical — a second encoder that
   disagrees on any input is a fork of the commitment.
2. Whether the contract invariants are sufficient to make "structurally
   invertible" equivalent to "invertible", or whether there is a chain that
   satisfies every invariant and is not.
3. The aggregate resource budget, now that per-field limits are known to be
   insufficient.
