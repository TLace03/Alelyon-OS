"""alelyon.verify — the OPEN verification surface for the certified pipeline.

This is the public face of what ships as the `alelyon-verify` open-source library:
everything a third party needs to check a Certified Number Envelope (CNE) or a
signed Merkle transparency-log proof, WITHOUT the proprietary capture engine, data
store, GUI, or broker. Its import closure is only the deterministic DSL kernel +
`cryptography` + numpy/pandas (contract-tested — no engine/store/frontend imports).

    from alelyon.verify import verify_envelope, verify_tree_head, verify_inclusion, KeyStore

    result = verify_envelope(cne, {("price", "AAPL"): my_aapl_series},
                             public_key_hex=trusted_key)   # pinned key REQUIRED
    assert result["ok"]     # authentic + re-derived on the deterministic kernel

The boundary that keeps monetization open: the verifier and the envelope FORMAT
are the shared standard (open); the engine that PRODUCES certificates — capture +
quantization, the analytics, the DSL-authoring model, the signing authority — is
the product. `build_envelope`/`signed_tree_head` are the REFERENCE producer used
for tests and interop; a commercial issuer runs them behind its own key.

Verifier-only (no store, no engine): verify_envelope, verify_tree_head,
verify_inclusion, verify_merkle_path, verify_consistency, verify_cosignature,
verify_witnessed_head, input_digest, canonical_series, KeyStore.verify, canonical.
Reference producer (needs data/a store): build_envelope, signed_tree_head,
inclusion_proof.

The normative format description is `docs/cne/SPEC-cne-v0.md`, versioned as
`SPEC_VERSION`. `SUPPORTED_ENVELOPE_TYPES` is this build's declaration of which
`alelyon.cne/*` types it verifies — pinned to the type string the verifier actually
accepts, and checked against it in CI, so the declaration and the behaviour cannot
drift (Track 0, W1(d) / W7).
"""
from alelyon.runtime.oracle.dsl.verify import verify_envelope
from alelyon.runtime.oracle.dsl.envelope import (
    build_envelope, input_digest, canonical_series, ENVELOPE_TYPE,
    MalformedEnvelope)
from alelyon.runtime.atlas.data.attest import (
    KeyStore, canonical, merkle_root, merkle_proof, verify_merkle_path,
    signed_tree_head, verify_tree_head, inclusion_proof, verify_inclusion,
    verify_consistency, verify_cosignature, verify_witnessed_head)
from alelyon.runtime.atlas.data.keylife import (
    build_manifest_checkpoint, verify_key_manifest, verify_manifest_checkpoint)
from alelyon.verify.conformance import (SPEC_VERSION, load_vectors,
                                        run_conformance, run_case)

__all__ = [
    # verifier
    "verify_envelope", "verify_tree_head", "verify_inclusion",
    "verify_merkle_path", "verify_consistency", "verify_cosignature",
    "verify_witnessed_head", "input_digest", "canonical_series", "canonical",
    "verify_key_manifest", "verify_manifest_checkpoint",
    "merkle_root", "merkle_proof", "KeyStore", "ENVELOPE_TYPE",
    "MalformedEnvelope",
    # conformance suite (W5)
    "run_conformance", "run_case", "load_vectors",
    # reference producer
    "build_envelope", "signed_tree_head", "inclusion_proof",
    "build_manifest_checkpoint",
    # declarations
    "SPEC_VERSION", "SUPPORTED_ENVELOPE_TYPES", "CNE_SPEC", "__version__",
]

__version__ = "0.3.0"

#: Every `alelyon.cne/*` type string this build verifies. Derived from
#: ENVELOPE_TYPE rather than restated, so the declaration cannot claim support the
#: code does not have. A v1 envelope handed to this build is refused with a stated
#: reason, never mis-verified under v0 rules (SPEC §10.3).
SUPPORTED_ENVELOPE_TYPES = (ENVELOPE_TYPE,)

#: Back-compat alias for the single supported type.
CNE_SPEC = ENVELOPE_TYPE
