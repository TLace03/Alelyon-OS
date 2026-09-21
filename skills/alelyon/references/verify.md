# Verify a number receipt

Inspect the installed verifier and its input requirements before interpreting
a receipt:

```bash
alelyon-verify version
alelyon-verify verify --help
alelyon-verify verify --envelope receipt.json --data extract.json --key <pinned-public-key-hex>
```

The public distribution is `alelyon-os`; the verifier name is its console
command. Obtain the issuer's public key independently of the receipt and retain
your own copy of the inputs. A receipt that supplies its own trusted key does
not establish issuer authenticity.

## Read the complete verdict

`verify_envelope` returns `ok`, individual `checks`, human-readable
`reasons`, machine-readable `reason_classes`, and trust qualifiers such as
`width_trust` and `provider_trust`.

A check can be true, false or null. Null means the check did not establish its
property. Preserve it in interfaces and reports. An overall failure may be the
correct result for a malformed receipt or an unavailable replay substrate.

Use reason classes for branching and human-readable reasons for explanation.
`REASON_CLASSES` and `ADVISORY_REASON_CLASSES` are defined in the verifier
and mirrored by the specification's parsed vocabulary. Advisory classes can
appear in a successful verdict. Do not freeze their number in application prose.

## Width and provenance

Nonzero-width replay can require the specified deterministic kernel. A fallback
must leave a width unverified when it cannot reproduce it. Exact-zero width
under the applicable capture law avoids that substrate sensitivity.

Input digest agreement detects revision of committed inputs. It cannot discover
fabrication at capture. Provider-attempt evidence establishes recorded requests
and outcomes, not independence of upstream providers or truth of the observations.

## Key lifecycle

Use the manifest command when checking lifecycle records:

```bash
alelyon-verify manifest --help
alelyon-verify manifest --manifest keys.json --root <pinned-root-hex> --checkpoint checkpoint.json --checkpoint-key <pinned-checkpoint-key-hex> --trusted-checkpoint retained.json
```

Root, checkpoint key and retained checkpoint are separate trust inputs.
Do not remove one to make an incomplete chain pass. Receipt verification can
also receive `--key-manifest` and `--manifest-root`; inspect the command's
requirements for the installed version.

## Local acceptance

Run `alelyon-verify selftest` and inspect the golden and forgery outcomes.
Every bundled forgery must reject for its specified reason. A fallback can
honestly leave substrate-sensitive goldens incompletely verified; a specified
deterministic substrate must fully verify its goldens.

This is evidence about the installation and inputs used in that run. External
verification requires an actual external recipient and a recorded exercise.
