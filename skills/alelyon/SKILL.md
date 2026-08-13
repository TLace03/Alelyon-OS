---
name: alelyon
description: Produce and check numbers that carry their own evidence, using the alelyon-os toolkit (alelyon-verify, the CNE producer path, lattice registration, the compute DAG, fleet coordination) and the claim discipline that governs how results are reported. Use when verifying or issuing a certified receipt, replaying someone else's number, registering coordinates or model axes, propagating uncertainty through a computation, coordinating several agent sessions in one repository, or whenever you are about to state a numeric result, a benchmark, a "verified"/"certified"/"tested" claim, or fill in a metric you did not measure.
---

# Alelyon

Two things live here, and the second governs the first:

1. **A toolkit** for numbers that carry their own evidence — `pip install alelyon-os`.
2. **A discipline** for reporting results, which applies whether or not you install anything.

Read §1 before you write a number down. Read §2 before you reach for a command.

---

## 1. The discipline (governs everything below)

### Observed versus declared

Every value you hand a user is one or the other, and conflating them is the failure this
whole project exists to prevent.

- **Observed** — you ran it and read the result.
- **Declared** — someone (a doc, a comment, a previous agent, you a moment ago) asserted it.

Never present a declared value in the grammar of an observed one. If you did not run the
benchmark, do not report a number for it. If a test file exists but you did not execute
it, the code is untested *by you*.

### Say UNMEASURED out loud

An absent measurement is reported as **UNMEASURED**, never as a blank, a dash, an
"N/A", or — worst — a passing slot. A null cell beside a filled one reads as
"checked, fine."

Use the uppercase marker only for gaps that are open **right now**. Quoting a
previously-flagged gap in a summary re-asserts it; paraphrase instead.

### Never weaken an acceptance criterion

When a criterion cannot be met, say which kind of failure it is, then stop or fix it:

- **Design truth** — it cannot be met, by construction. State it plainly and stop.
  Do not soften the criterion so it passes.
- **Infrastructure gap** — it could be met with work that does not exist yet. Find an
  existing implementation or build one. Do not lower the bar to clear it.

Reporting "criterion adjusted" without naming which of the two you hit is the move this
rule exists to forbid.

### Claim vocabulary

Do not write, in code, comments, commit messages, docs, or anything user-facing:

| Banned | Why |
|---|---|
| "certified" with no noun | Certified *what*, by *whom*, against *what*? |
| "tight error bound" | Unquantified superlative. |
| "trustless" (unqualified) | Something is always trusted; name it. |
| "independent witness" | Only if another party operates it. Co-location is not independence. |
| "anyone can verify" | Verification needs a key obtained out of band. Say so. |

### Absences are content

A field that is missing is a fact about the object, not a gap in it. When a checker
reports `null` for a check, that is "not established", which is different from both
"passed" and "failed". Carry the distinction through — do not collapse `null` into
either neighbour.

---

## 2. The toolkit

```bash
pip install alelyon-os
```

### Verify a receipt someone gave you

```bash
alelyon-verify selftest                       # bundled conformance suite, no network
alelyon-verify verify --envelope receipt.json \
                      --data your_extract.json \
                      --key <issuer public key, obtained OUT OF BAND>
```

Exit code is `0` only when the verdict is `ok`. The key **must** reach you by a path the
receipt did not travel — checking an envelope against a key carried inside that same
envelope authenticates nothing.

### What a pass does and does not establish

A pass means: the committed inputs were not revised after the fact, and the number
replays from them under the key you pinned.

A pass does **not** mean the inputs were true when captured. A producer who fabricates an
extract at capture time signs a receipt that verifies perfectly. Say this whenever you
report a verification — it is the single most misread property of the system.

### `ok=false` is often correct behaviour, not a bug

`ok` requires the certified **width** to be re-derived, and a *nonzero* width only
reproduces bit-for-bit on the substrate that produced it. On a plain `pip install`
(the numpy fallback), a receipt carrying a nonzero width honestly degrades: the width is
left **unverified** rather than guessed.

The bundled suite grows as adversarial cases are added, so do not copy a denominator
from this document. Run `alelyon-verify selftest` to measure the installed version.
Every bundled forgery must reject for its declared reason. On the numpy fallback,
nonzero-width goldens honestly remain not fully verified; exact-zero goldens may verify
fully. A specified deterministic substrate must fully verify every bundled golden.

**The exception that matters commercially:** a width of exactly **zero** is
substrate-independent. Under the exact-cents law every Δ is zero, the perturbation is
identically zero, and the order statistic is `0.0` on any kernel. That is why
exact-storage receipts (money, counts, ledger rows) verify fully on a bare install.

### Issue a receipt for your own data

Measured end to end on a fallback build — this exact script produced `ok=true`:

```python
import pandas as pd
from alelyon.runtime.oracle.dsl.fetch import from_frame, CENTS
from alelyon.runtime.oracle.dsl.envelope import build_envelope
from alelyon.runtime.oracle.dsl.verify import verify_envelope
from alelyon.runtime.atlas.data.attest import KeyStore

frame = pd.DataFrame({"CLAIMS": [100.00, 250.25, 99.75, 1000.10]})

# `delta` DECLARES how the data was stored. law="exact-cents/v0" asserts the
# values are exact as stored -- which is what makes the width zero, and the
# receipt verifiable anywhere.
fetcher = from_frame(frame, delta=CENTS, law="exact-cents/v0", kind="price")
keystore = KeyStore("signing.pem")

env = build_envelope('show sum(price("CLAIMS"))',
                     fetcher=fetcher, keystore=keystore, seed=7, now=0.0)
# env["scalar"] == 1450.1

verdict = verify_envelope(env, {("price", "CLAIMS"): frame["CLAIMS"]},
                          public_key_hex=keystore.public_key_hex())
# verdict["ok"] is True; verdict["width_trust"] == "authenticated"
```

Move one cent in the recipient's copy and the verdict becomes
`ok=False` with `reason_classes == ['input-digest-mismatch', 'scalar-mismatch',
'width-mismatch']`.

**Δ is a declaration, not a discovery.** `delta=0` on invented data certifies invented
data, and nothing in the system detects that. A missing per-column declaration is an
error rather than a defaulted zero, deliberately — see `references/produce.md`.

### The rest of the surface

Load the reference file only when the task calls for it.

| Task | Read |
|---|---|
| Reason classes, key manifests, revocation, out-of-band key hygiene | `references/verify.md` |
| Declaring capture laws, Δ semantics, the DSL, what refusals mean | `references/produce.md` |
| Coordinate registration, exact transforms, Model Morphometry | `references/lattice.md` |
| Typed compute DAG, uncertainty propagation, variance attribution | `references/compute.md` |
| Several agent sessions in one repo: claims, findings, channels | `references/fleet.md` |

---

## 3. Working on the toolkit itself

This tree is **generated**. `packages/alelyon-os/**`, `spec/**`, and
`skills/alelyon/**` in the public mirror are produced from Alelyon's private monorepo;
the package export is driven by the same allowlist that builds the published wheel. A
pull request editing those paths cannot be merged as written — the next export
overwrites it. Open an issue instead, or edit upstream.

`UPSTREAM.json` names the source commit and a SHA-256 per generated file. That is
self-declared traceability, not authenticated provenance. Do not describe it as the
latter.

## 4. Reporting your results

When you finish a task governed by this skill, report in this shape:

- What you **observed**, with the command that produced it.
- What remains **UNMEASURED**, named individually.
- Any criterion you could not meet, labelled *design truth* or *infrastructure gap*.

If a step was skipped, say it was skipped. If a test failed, show the output. A summary
that reads as complete when it is not is the specific failure mode this skill exists to
prevent.
