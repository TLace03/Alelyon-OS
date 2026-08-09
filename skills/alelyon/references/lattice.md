# Lattice — exact coordinate registration

Load this when registering one coordinate system onto another, or when auditing a
transform chain someone else declared.

```python
from alelyon.runtime.vector import lattice
```

## What it is

Immutable coordinate contracts, exact target-to-source transforms carrying a declared
loss/invertibility surface, compatibility refusals, a canonical byte encoding with
content commitments, a replay checker that recovers a committed chain from those bytes
and re-executes it, and a signed Registration Certificate for an exact correspondence.

## What it deliberately is not

It does not read payloads, remap values, propagate uncertainty, or claim optimality.
A signature binds bytes to a key; it does not establish who holds the key.

Do not describe a registration as "validated data" — the payload was never opened.

## Absences are signed content

The certificate populates **15** of the specification's **34** fields and carries the
other **19 as named absences inside its signed bytes**.

This is the design's central move, and the thing to get right when reporting it: an
unfilled field is not missing from the certificate. It is present, named, and signed as
absent. A consumer can tell "this was not established" apart from "nobody thought about
this", because the certificate commits to the difference.

When summarising a certificate, never render the 19 as blanks or drop them. `FieldStatus`
and `FieldAbsence` exist to keep the distinction; carry it into your output.

## The two measured bounds

Everything else in the certificate is structural. Exactly two things are *measured*, and
both are weaker than the words for them usually suggest:

- **`inverse_consistency`** — a **count over a derived probe sample**, not a proof that
  the chain inverts everywhere. `derive_probe_coordinates` picks the probes;
  `measure_inverse_consistency` tallies them, capped at `MAX_PROBES`.
- **`execution_trace_commitment`** — binds that count to the probe executions it tallies,
  so the number cannot be swapped for a friendlier one.

Report `inverse_consistency` as "N of M probes inverted", never as "the transform is
invertible".

## Issuing and verifying

```python
cert   = lattice.issue_registration_certificate(...)
report = lattice.verify_registration_certificate(cert, ...)   # -> CertificateReport
```

Verification **re-runs the registration ladder**, so a chain that registration would
never have emitted is refused even when it replays cleanly. Replaying is necessary and
not sufficient — a well-formed chain that could not have been produced legitimately is
still rejected.

Related: `verify_transform_chain`, `read_certificate`, `read_transform_chain`,
`read_coordinate_space`, `chain_commitment`.

## Declared transforms and audits

The `Declared*` classes record what someone asserts a transform does:
`DeclaredUnitConversion`, `DeclaredTimezoneConversion`, `DeclaredOrientationFlip`,
`DeclaredReferenceShift`, `DeclaredLabelReindex`, `DeclaredCalendarAlias`.

The `audit_*` functions check an assertion against the transform actually present:

```python
lattice.audit_declared_conversion(...)
lattice.audit_timezone_transform(...)   # -> TimezoneAuditReport / TimezoneAuditVerdict
```

This is observed-versus-declared applied to geometry. The audit exists because a declared
unit conversion that does not match the transform is exactly the class of error nobody
catches by reading code.

## Model Morphometry

A canonical `(block, module)` template for transformer models, and an exact registration
of a model's native axis order onto it.

```python
lattice.morphometry_canonical_space()
lattice.register_model_morphometry(...)
lattice.analyze_model_morphometry(...)
lattice.certify_model_morphometry(...)
```

It registers **axis order and structure**. It does not inspect weights, and it makes no
claim about model behaviour or quality. A morphometry certificate says two models' axes
correspond exactly — nothing about what they compute.
