# Coordinate registration and Model Morphometry

The Vector lattice package represents coordinate contracts, exact transforms,
their declared loss/invertibility properties, canonical encodings and signed
registration certificates.

```python
from alelyon.runtime.vector import lattice
```

Use the exported contracts and registration functions for the installed version.
A transform maps coordinates; it does not validate the underlying payload,
establish model quality, or prove an optimization is best.

## Registration and replay

The entry points include `issue_registration_certificate`,
`verify_registration_certificate`, `verify_transform_chain`,
`read_certificate`, `read_transform_chain`, and `read_coordinate_space`.

Certificate verification checks more than whether a supplied chain executes:
the registration procedure must also be able to produce the claimed
correspondence. Preserve refusal results from contract and compatibility checks.

Canonical bytes and commitments detect changed records under their stated
assumptions. A signature authenticates bytes to a pinned key; ownership of that
key needs an independent trust arrangement.

## Preserve absent fields

Read field coverage from `SPEC_CERTIFICATE_FIELDS`, `POPULATED_FIELDS` and
`DECLARED_ABSENCES` in the certificate module. Do not repeat a field total
from another schema version.

An absence has a field name, status and reason and is included in the signed
record. Display those absences when summarizing a certificate. A blank cell
would hide which properties were never established.

`inverse_consistency` reports agreement on derived probes, bounded by
`MAX_PROBES`; it is not a universal invertibility proof.
`execution_trace_commitment` binds the record to those probe executions.

## Declared conversions

Declared unit, timezone, orientation, reference, label and calendar mappings
state what a transform is supposed to do. Use the relevant `audit_*`
function to compare the declaration with the actual transform. A declaration
alone is not an observed conversion result.

## Model Morphometry

`morphometry_canonical_space`, `register_model_morphometry`,
`analyze_model_morphometry` and `certify_model_morphometry` apply coordinate
registration to model structure and axis order. They do not inspect weights or
establish capability, accuracy, reasoning quality or numerical equivalence.

Source-checkout reference: `alelyon/runtime/vector/lattice/`.
Focused certificate checks: `tests/vector/test_lattice_certificate.py`.
