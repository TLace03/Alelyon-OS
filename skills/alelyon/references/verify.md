# Verifying a Certified Number Envelope

Load this when checking someone else's receipt, or when a verdict needs explaining.

## The verdict object

`verify_envelope(...)` returns a dict. Read it in this order:

```python
{
  "ok": bool,                  # every applicable check passed
  "checks": {                  # per-check tri-state
      "authenticity": True,    #   True  = established
      "width": None,           #   None  = NOT ESTABLISHED (not "passed", not "failed")
      "witness": None,         #   False = failed
      ...
  },
  "reasons": [...],            # human-readable
  "reason_classes": [...],     # machine-stable vocabulary -- key on THESE
  "width_trust": "authenticated" | "refusal" | ...,
  "provider_trust": "signer-attested" | ...,
}
```

**`None` is a third state.** Do not collapse it into pass or fail. A `null` width means
the width was not re-derived on this build — which on a fallback install is the honest
outcome, not a defect.

Key on `reason_classes`, never on `reasons`. The prose can be reworded; the classes are
frozen in two places (the `REASON_CLASSES` / `ADVISORY_REASON_CLASSES` frozensets and
machine-parsed blocks in the spec) and CI-diffed, so adding one means editing both.

## Advisory versus disqualifying

There are 67 reason classes. Six are **advisory** — they appear without forcing
`ok=false`:

```
scalar-tolerance-window     transparency-partial       unspecified-substrate
width-substrate-independent witness-partial            witness-unpinned
```

The other 61 are disqualifying. When you report a verdict, quote the classes; an
advisory class in the list is not a failure, and reporting it as one is its own error.

The classes cluster by what they indict:

| Prefix | What went wrong |
|---|---|
| `input-`, `scalar-`, `width-`, `budget-`, `program-` | The replay disagrees with the receipt. |
| `key-`, `unsigned`, `bad-signature`, `no-pinned-key` | Signature or key-lifecycle problem. |
| `anchor-` | Transparency anchoring: the per-row Δ is not backed by capture-time commitments. |
| `provider-` | Provider-attempt evidence is missing, partial, or inconsistent. |
| `witness-` | Co-signature over the STH is absent, unpinned, or invalid. |
| `substrate-`, `unspecified-substrate` | The width cannot be re-derived on this build. |

## Keys must arrive out of band

```bash
alelyon-verify verify --envelope receipt.json --data extract.json --key <hex>
```

`--key` is 64 hex characters obtained by a path the receipt did not travel. This is not
ceremony: an envelope carrying its own key authenticates the envelope to itself.

Never write "anyone can verify" about this. Anyone holding the key, out of band, can.

## Key lifecycle

An issuer who rotates keys publishes a manifest. Checking a signature without checking
whether the key was in service at signing time leaves a revoked key working forever.

```bash
alelyon-verify manifest \
  --manifest keys.json \
  --root <ROOT key, out of band> \
  --checkpoint ckpt.json --checkpoint-key <hex, out of band> \
  --trusted-checkpoint previously-retained.json \
  [--at <epoch seconds>]
```

Every one of `--root`, `--checkpoint`, `--checkpoint-key` and `--trusted-checkpoint` is
**required**. A chain checked against nothing vouches for nothing, and a checkpoint with
no retained predecessor cannot detect a rewind.

To fold the key check into a receipt check, pass `--key-manifest` plus `--manifest-root`
to `alelyon-verify verify`. A revoked signing key is then refused rather than reported.

## Conformance

```bash
alelyon-verify version    # verifier version, spec version, substrate, envelope types
alelyon-verify selftest   # installed suite totals and results, no network
alelyon-verify vectors    # list the bundled vectors
```

The suite grows as adversarial cases are added, so its exact totals belong to the
installed `selftest` output rather than this reference. Every bundled forgery must
reject for its declared reason. A fallback build must leave substrate-sensitive
nonzero-width goldens not fully verified, while exact-zero goldens may verify fully. A
specified deterministic substrate must fully verify every bundled golden.

## What a pass is worth

It establishes: the committed inputs were not revised after the fact, and the scalar
replays from them under the pinned key.

It does not establish: that the inputs were true at capture. A producer who fabricates at
capture time signs a receipt that verifies perfectly. This is a design truth, not a gap
to be closed by the verifier — the verifier sits downstream of capture and cannot see
behind it.

Report both halves. Reporting only the first is the misreading the claim discipline
exists to prevent.
