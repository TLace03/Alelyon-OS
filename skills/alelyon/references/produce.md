# Issue a number receipt

Issuance records a computation over declared inputs. It does not make invented
or incorrect inputs true. Use a production signing key only through the approved
key lifecycle; examples belong in temporary storage.

## Declare storage semantics

`from_frame` and `from_csv` accept a quantization declaration. The
`delta` value describes storage, not an uncertainty inferred by the library.
A per-column mapping must cover every required column.

`CENTS` is a step constant, not a capture-law name. The named laws include
`exact-cents/v0` and `dither-relative/v0`. Unknown laws must refuse.
Exact-cent arithmetic has aggregate representability checks beyond the
per-element declaration; preserve those checks.

## Temporary local demonstration

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd

from alelyon.runtime.atlas.data.attest import KeyStore
from alelyon.runtime.oracle.dsl.fetch import CENTS, from_frame
from alelyon.runtime.oracle.dsl.envelope import build_envelope
from alelyon.runtime.oracle.dsl.verify import verify_envelope

frame = pd.DataFrame({"AMOUNTS": [10.00, 20.25, 5.75]})
fetcher = from_frame(frame, delta=CENTS, law="exact-cents/v0", kind="price")
with TemporaryDirectory() as temporary:
    key = KeyStore(str(Path(temporary) / "demo-signing.pem"))
    receipt = build_envelope(
        'show sum(price("AMOUNTS"))',
        fetcher=fetcher, keystore=key, seed=17, now=0.0,
    )
    verdict = verify_envelope(
        receipt, {("price", "AMOUNTS"): frame["AMOUNTS"]},
        public_key_hex=key.public_key_hex(),
    )
    print(receipt["scalar"], verdict["ok"], verdict["reason_classes"])
```

This self-contained example obtains its key directly from its own producer.
A real recipient must instead pin the issuer's public key out of band.
The numbers are illustrative input declarations; inspect the run's output.

## Computation and refusal

`build_envelope` accepts restricted DSL source and a fetcher for its literal
data references. Supply a seed and explicit timestamp for a reproducible run.
Arbitrary Python, imports, filesystem operations and network execution do not
belong in the DSL.

Fetch or computation refusal can produce a signed refusal envelope rather than
a scalar. Preserve its reason and inspect its verification verdict. A refusal
receipt is evidence of declining a computation, not a successful numeric answer.

## Capture evidence

The producer can attach capture transparency leaves, inclusion proofs and
signed tree heads when the required store and keys are available. Inspect the
actual anchors and trust status; a signing key alone does not establish
capture coverage.

A witness parameter adds a co-signature. Organizational independence requires
another party to operate that witness; another key on the producer's machine
does not satisfy that property.

Before handing a receipt to another party, retain its input declarations,
program, key lifecycle references and trust limitations. Test a changed-input
case and report the refusal classes observed on the installed verifier.
