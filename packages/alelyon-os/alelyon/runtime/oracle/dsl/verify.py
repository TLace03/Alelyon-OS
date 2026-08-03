"""The replay verifier — the open, dependency-light checker that makes a Certified
Number Envelope VERIFIABLE BY A THIRD PARTY (data-verification foothold, W2b).

Given a CNE, the signer's TRUSTED (pinned) public key, and its OWN copy of the
input data, this re-derives the number on the deterministic Rust kernel and
confirms every field. It trusts only the pinned key and the data — never the
engine that produced the envelope, never a running store.

What is independently verified vs attested (be precise — the red team was right to
insist):
  authenticity  the envelope is signed by the PINNED key (supplied out-of-band).
                Verifying against a key embedded in the same untrusted envelope
                authenticates NOTHING (self-signing), so a pinned key is REQUIRED
                for ok=True.
  inputs        the caller's data matches the committed digest (the number is over
                THIS data) — independently checked.
  scalar        the value is RE-DERIVED from the data on the kernel and must match
                — genuine verifiable-by-replay.
  width         the DRC bound is re-derived from the capture deltas the envelope
                commits. This confirms the signer's arithmetic.
  transparency  whether those committed deltas are ANCHORED: each is re-derived
                from the signed cert_log leaves whose Merkle inclusion is proven
                under the pinned key (recomputing each leaf_hash binds the
                payload Δ to the signed tree). When every input is anchored the
                width is TRANSPARENCY-ANCHORED: the deltas are bound to the signer's
                earlier capture-time log commitments. This detects revision of those
                commitments; it does not prove the signer captured truthful values or
                deltas. Absent an anchor the width is merely AUTHENTICATED (a trusted
                signer vouches for the deltas); a PRESENT-but-invalid anchor is a
                forgery signal and fails ok. `width_trust` names which case holds.
  witness       whether a co-signing witness seam signed the anchored STH (the
                equivocation guard): with a witness key pinned out of band the
                co-signature is verified and bound to the complete STH. A
                present-but-invalid co-signature fails ok; absent or unpinned is
                reported, not judged. Independence exists only when a party other
                than the signer operates the witness and preserves its state.
  tier          the honesty tier is re-derived and must match.

Bit-for-bit replay requires the SAME native kernel that produced the envelope
(the Neumaier reductions are bit-reproducible only on the Rust wheel). When the
local substrate differs, the verifier falls back to a tight relative tolerance and
SAYS SO — never a silent false-reject, never a silent weakening.

Every reason carries a stable REASON CLASS alongside its prose (`reason_classes`
in the result). The prose is for humans and may be reworded; the class is part of
the wire contract — SPEC-cne-v0.md §9 freezes the vocabulary, the conformance
vectors assert on it, and a second-language verifier is required to agree on it.
Two implementations that both say ok=False for different reasons have not agreed.

Dependency surface (narrow by design — no GUI, broker, engine, or store): the
deterministic DSL subgraph (`execcert` → interpreter/parser + native kernel),
`envelope` (canonical form), `attest` (signature), `cryptography`, numpy, pandas.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl.execcert import certified_run
from alelyon.runtime.oracle.dsl.envelope import (ENVELOPE_TYPE, TABLE_KIND,
                                                 commitment_digest, table_digest,
                                                 rebuild_fetcher, canonical_series,
                                                 _decompress_deltas, _unsigned,
                                                 _kernel_id, _scope_of)
from alelyon.runtime.oracle.dsl.execcert import (_EXACT_CENTS_LAW,
                                                 _SAFE_EXACT_INT)

_REL_TOL = 1e-9   # substrate-mismatch tolerance (a meaningful forgery is far larger)

#: The only substrate whose numeric semantics SPEC-cne-v0.md §8 freezes. The
#: numpy fallback is a convenience path: `np.sum` is pairwise summation whose
#: association order can differ across numpy builds and SIMD widths, so two
#: machines both reporting "numpy-fallback" are NOT guaranteed to agree on a
#: near-cancellation width. Measured on this repo, the Neumaier kernel and
#: np.sum disagree on 146/200 random 500-element reductions — the substrate is
#: load-bearing, not cosmetic.
_SPECIFIED_SUBSTRATE_PREFIX = "alelyon-vector/"


def _is_specified_substrate() -> bool:
    return _kernel_id().startswith(_SPECIFIED_SUBSTRATE_PREFIX)


#: The FROZEN reason-class vocabulary (SPEC-cne-v0.md §9.4). This is a wire
#: contract, not an implementation detail: the conformance vectors assert on it and
#: a second-language verifier is required to agree on it, so it may not grow by
#: accident. `_Reasons.add` rejects anything not listed here, and a test asserts
#: this set equals the set the spec documents — so code and spec cannot drift.
REASON_CLASSES = frozenset({
    # authenticity and structure
    "not-a-cne-v0", "no-pinned-key", "malformed-pinned-key", "unsigned",
    "key-id-mismatch", "bad-signature", "program-hash-mismatch",
    "malformed-envelope",
    # inputs
    "no-input-data", "input-missing", "input-digest-mismatch",
    "delta-count-mismatch", "uncertified-count-mismatch",
    # replay
    "replay-refusal-mismatch", "scalar-mismatch", "tier-mismatch",
    "budget-mismatch", "no-seed", "width-mismatch", "substrate-mismatch",
    "unspecified-substrate", "width-substrate-independent",
    "scalar-tolerance-window",
    # transparency anchor
    "transparency-no-pinned-key", "transparency-partial", "anchor-sth-invalid",
    "anchor-malformed-scope", "anchor-scope-mismatch",
    "anchor-sth-scope-mismatch", "anchor-malformed-leaf",
    "anchor-leaf-hash-mismatch", "anchor-proof-tree-size-mismatch",
    "anchor-proof-index-out-of-range", "anchor-inclusion-failed",
    "anchor-delta-unusable", "anchor-no-data", "anchor-length-mismatch",
    "anchor-row-uncovered", "anchor-delta-mismatch",
    "anchor-delta-zero-implausible",
    # witness
    "witness-unpinned", "witness-cosignature-invalid", "witness-partial",
    "witness-malformed",
    # key lifecycle (W6)
    "key-manifest-unrooted", "key-manifest-invalid", "key-not-in-manifest",
    "key-revoked", "key-outside-validity",
    "key-manifest-checkpoint-required", "key-manifest-checkpoint-invalid",
    "key-manifest-checkpoint-not-monotonic",
    # provider corroboration anchor (W3). Mirrors the capture-anchor classes above
    # because the failure modes are the same shape — a forged head, a substituted
    # scope, a leaf that does not hash to what was proven — plus three that are
    # specific to attempt records: a deleted silence, an edited outcome, and a
    # summary that disagrees with the leaves proving it.
    "provider-no-pinned-key", "provider-partial", "provider-sth-invalid",
    "provider-malformed-scope", "provider-scope-mismatch",
    "provider-sth-scope-mismatch", "provider-malformed-leaf",
    "provider-leaf-hash-mismatch", "provider-proof-tree-size-mismatch",
    "provider-proof-index-out-of-range", "provider-inclusion-failed",
    "provider-attempt-count-mismatch", "provider-attempts-digest-mismatch",
    "provider-outcome-unknown", "provider-summary-mismatch",
})

#: Classes reported WITHOUT changing `ok` — they inform, they do not judge.
ADVISORY_REASON_CLASSES = frozenset({
    "unspecified-substrate", "width-substrate-independent",
    "scalar-tolerance-window",
    "transparency-partial", "witness-partial", "witness-unpinned",
})


class _Reasons:
    """Collects (class, prose) pairs. `prose` stays the human-facing list the
    result has always carried; `classes` is the stable, spec-frozen vocabulary a
    second implementation must reproduce exactly."""

    def __init__(self) -> None:
        self._items: List[Tuple[str, str]] = []

    def add(self, cls: str, msg: str) -> None:
        if cls not in REASON_CLASSES:
            # Loud, not lenient. An undeclared class would flow into a vector's
            # expectation and become a de-facto contract nobody wrote down.
            raise AssertionError(
                f"reason class {cls!r} is not in the frozen vocabulary; add it to "
                f"REASON_CLASSES and to SPEC-cne-v0.md §9.4 in the same commit")
        self._items.append((str(cls), str(msg)))

    @property
    def prose(self) -> List[str]:
        return [m for _, m in self._items]

    @property
    def classes(self) -> List[str]:
        """Sorted and de-duplicated: two verifiers must agree on the SET of
        failure classes, not on the order they happened to be discovered in
        (which depends on input iteration order)."""
        return sorted({c for c, _ in self._items})


def _is_ed25519_hex(s) -> bool:
    """A pinned key must be 32 bytes of hex. Checked up front so a typo in the
    pin returns ok=False with a reason instead of raising out of the verifier."""
    return (isinstance(s, str) and len(s) == 64
            and all(char in "0123456789abcdef" for char in s))


def verify_envelope(cne: dict, input_data: Optional[Dict] = None, *,
                    public_key_hex: Optional[str] = None,
                    witness_key_hex: Optional[str] = None,
                    key_manifest: Optional[dict] = None,
                    manifest_root_hex: Optional[str] = None,
                    manifest_checkpoint: Optional[dict] = None,
                    checkpoint_public_key_hex: Optional[str] = None,
                    trusted_manifest_checkpoint: Optional[dict] = None) -> dict:
    """Verify a CNE against a TRUSTED public key and the caller's own input data.
    Returns {ok, checks:{...}, reasons:[...]}. `ok` is True only when the envelope
    is authentic under the pinned key AND the number was actually re-derived from
    the supplied data and matched.

    `public_key_hex` (REQUIRED for ok=True): the out-of-band trusted key. Without
    it there is no authentication. `input_data` maps (kind,key) or the bare key to
    the caller's pandas Series; without it the number cannot be re-derived.
    `witness_key_hex` (OPTIONAL): a witness key pinned out-of-band. When
    supplied and the envelope carries witness co-signatures, each is verified against
    it — a present-but-invalid co-signature is a forgery signal that fails ok; absent
    a pinned witness key the co-signatures are reported but not judged. This is the
    equivocation guard: independence exists only when the witness is run by a party
    other than the signer.

    `key_manifest` (OPTIONAL) opts into lifecycle checking. When it is supplied,
    `manifest_root_hex`, `manifest_checkpoint`, `checkpoint_public_key_hex`, and
    `trusted_manifest_checkpoint` are all required. The retained signed checkpoint
    is the verifier's rollback state; on success the checkpoint API returns the exact
    next object the caller may persist atomically.
    """
    checks: Dict[str, Optional[bool]] = {
        "authenticity": None, "inputs": None, "scalar": None,
        "width": None, "budget": None, "program": None, "tier": None,
        "transparency": None, "witness": None, "key_status": None,
        "provider": None}
    reasons = _Reasons()

    if not isinstance(cne, dict) or cne.get("type") != ENVELOPE_TYPE:
        # Fail CLOSED on an unknown or future envelope type: a v0 verifier must
        # refuse a v1 object with a stated reason, never mis-verify it under v0
        # rules (SPEC-cne-v0.md §10, the versioning policy).
        reasons.add("not-a-cne-v0", "not a CNE v0 object")
        return {"ok": False, "checks": checks, "reasons": reasons.prose,
                "reason_classes": reasons.classes, "width_trust": "unverified"}

    # 1. AUTHENTICITY — requires a pinned key; the embedded key cannot self-authenticate
    from alelyon.runtime.atlas.data.attest import KeyStore, canonical
    if public_key_hex is not None and not _is_ed25519_hex(public_key_hex):
        # A malformed pin used to raise ValueError out of key_id_of(); a verifier
        # must answer "not verified", never crash at its caller.
        checks["authenticity"] = False
        reasons.add("malformed-pinned-key",
                    "pinned public key is not 32 bytes of hex")
        public_key_hex = None
    elif public_key_hex is None:
        checks["authenticity"] = False
        reasons.add("no-pinned-key",
                    "no trusted public key pinned — the envelope-embedded key "
                    "cannot authenticate itself; supply the signer's key")
    elif "signature" not in cne:
        checks["authenticity"] = False
        reasons.add("unsigned", "envelope is unsigned")
    elif cne.get("key_id") and cne["key_id"] != KeyStore.key_id_of(public_key_hex):
        checks["authenticity"] = False
        reasons.add("key-id-mismatch",
                    "envelope key_id does not match the pinned public key")
    else:
        try:
            checks["authenticity"] = KeyStore.verify(
                public_key_hex, canonical(_unsigned(cne)), cne["signature"])
        except Exception:  # noqa: BLE001 — malformed signature material
            checks["authenticity"] = False
        if not checks["authenticity"]:
            reasons.add("bad-signature", "invalid signature under the pinned key")

    # 1b. PROGRAM — the signed program_hash must be the hash of the program the
    # envelope actually carries, so the human-readable source and the thing that
    # was certified cannot diverge.
    if cne.get("program_hash") is not None:
        from alelyon.runtime.oracle.dsl.envelope import _program_hash
        checks["program"] = (isinstance(cne.get("program"), str)
                             and _program_hash(cne["program"]) == cne["program_hash"])
        if not checks["program"]:
            reasons.add("program-hash-mismatch",
                        "program_hash does not match the carried program")

    # 2+3. INPUTS + REPLAY — require the caller's data; fully guarded (never crash)
    if input_data is not None:
        try:
            checks["inputs"] = _check_inputs(cne, input_data, reasons)
            _replay_checks(cne, input_data, checks, reasons)
            _check_transparency(cne, input_data, checks, reasons, public_key_hex)
        except Exception as exc:  # noqa: BLE001 — untrusted envelope must not crash us
            reasons.add("malformed-envelope",
                        f"malformed CNE or replay failure: "
                        f"{type(exc).__name__}: {exc}")
            return {"ok": False, "checks": checks, "reasons": reasons.prose,
                    "reason_classes": reasons.classes, "width_trust": "unverified"}
    else:
        reasons.add("no-input-data",
                    "no input data supplied — the number was NOT re-derived; "
                    "provide the data to verify the scalar and bound")

    # WITNESS co-signature (equivocation guard) — independent of the caller's data:
    # the co-signed STH travels in the envelope, so it is checked either way.
    try:
        _check_witness(cne, checks, reasons, witness_key_hex)
    except Exception as exc:  # noqa: BLE001 — untrusted co-signature must not crash us
        reasons.add("witness-malformed",
                    f"malformed witness co-signature: {type(exc).__name__}: {exc}")
        if witness_key_hex is not None:
            checks["witness"] = False

    # PROVIDER corroboration anchors (W3) — also independent of the caller's data:
    # the probe leaves and their proofs travel in the envelope. Same failure rule as
    # the capture anchor: present-but-invalid fails ok, absent abstains.
    try:
        _check_provider(cne, checks, reasons, public_key_hex)
    except Exception as exc:  # noqa: BLE001 — an untrusted anchor must not crash us
        reasons.add("provider-malformed-leaf",
                    f"malformed corroboration anchor: {type(exc).__name__}: {exc}")
        checks["provider"] = False

    # KEY LIFECYCLE (W6): authenticate the succession chain under its root, require
    # a separately signed monotonic checkpoint against retained verifier state, then
    # place the signing key in its validity window. Independent of caller data.
    try:
        _check_key_status(
            cne, checks, reasons, key_manifest, manifest_root_hex,
            manifest_checkpoint, checkpoint_public_key_hex,
            trusted_manifest_checkpoint)
    except Exception as exc:  # noqa: BLE001 - an untrusted manifest must not crash us
        reasons.add("key-manifest-invalid",
                    f"malformed key manifest: {type(exc).__name__}: {exc}")
        checks["key_status"] = False

    # ok: authentic under the pinned key AND the number re-derived AND (for a
    # non-refusal) the BOUND re-derived. The width is a tiny cancellation-derived
    # quantity that only reproduces on the matching kernel, so a substrate
    # mismatch leaves it unverified (checks['width']=None) — an honest PARTIAL
    # result, never a false ok. A transparency anchor that is PRESENT but invalid
    # (checks['transparency']=False) is a forgery signal and fails ok; absent
    # anchoring (None) leaves the width signer-authenticated, still ok. The witness
    # co-signature follows the same rule: present-but-invalid under a pinned witness
    # key (checks['witness']=False) fails ok; absent or unpinned (None) does not.
    performed = [v for v in checks.values() if v is not None]
    replayed = checks["scalar"] is not None
    bound_ok = cne.get("refused") is True or checks["width"] is True
    ok = bool(checks["authenticity"]) and replayed and all(performed) and bound_ok
    return {"ok": ok, "checks": checks, "reasons": reasons.prose,
            "reason_classes": reasons.classes,
            "width_trust": _width_trust(cne, checks),
            "provider_trust": _provider_trust(checks)}


def _provider_trust(checks: dict) -> str:
    """How far the provider block's claim about upstreams is checkable.

    `transparency-anchored` ONLY when carried leaves were proven to be in a signed
    tree and the stated summary re-derived from them. Everything else, including an
    envelope carrying no anchors at all, is `signer-attested` — the label
    `width_trust` established, applied to the slot that would otherwise be the
    easiest place in the envelope to assert something unfalsifiable.

    Scope, so the label is not read for more than it says: with one answering origin
    this anchors the RECORD OF HAVING ASKED, never a dispersion or agreement number.
    """
    return "transparency-anchored" if checks.get("provider") is True \
        else "signer-attested"


def _width_trust(cne: dict, checks: dict) -> str:
    """A crisp label for how far the width is verifiable. transparency-anchored =
    the deltas are proven against the signer's signed capture-log commitments;
    authenticated = the signer's arithmetic + a trusted signature vouch for the
    deltas, but they are not log-anchored; unverified = not reproduced here.
    Neither label establishes that the producer captured truthful inputs."""
    if cne.get("refused") is True:
        return "refusal"
    if checks.get("width") is None:
        return "unverified"
    if checks.get("transparency") is True:
        return "transparency-anchored"
    if checks.get("width") is True:
        return "authenticated"
    return "unverified"


def _check_inputs(cne: dict, input_data: Dict, reasons: _Reasons) -> bool:
    ok = True
    for c in cne.get("inputs", []):
        if not isinstance(c, dict) or "kind" not in c or "key" not in c \
                or "digest" not in c:
            raise ValueError("input commitment is malformed")
        ref, key = (c["kind"], c["key"]), c["key"]
        series = input_data.get(ref, input_data.get(key))
        if series is None:
            ok = False
            reasons.add("input-missing", f"no data supplied for input {ref}")
            continue
        if commitment_digest(str(c["kind"]), series) != c["digest"]:
            ok = False
            reasons.add("input-digest-mismatch",
                        f"input {ref} digest mismatch — data differs from what "
                        f"was certified")
            continue
        # The commitment's own bookkeeping must agree with the deltas it commits.
        # `uncertified` drives strict-mode refusal and the branch-stability
        # precondition, so a declared count that undercounts the NaNs would
        # switch both guards off.
        import numpy as _np
        deltas = _decompress_deltas(c["deltas"])
        n_rows = len(canonical_series(series))
        if len(deltas) != n_rows:
            ok = False
            reasons.add("delta-count-mismatch",
                        f"input {ref} commits {len(deltas)} per-row deltas for "
                        f"{n_rows} rows of data")
            continue
        declared = c.get("uncertified")
        actual = int(_np.isnan(deltas).sum())
        if declared is not None and int(declared) != actual:
            ok = False
            reasons.add("uncertified-count-mismatch",
                        f"input {ref} declares {int(declared)} uncertified rows "
                        f"but its committed deltas contain {actual}")
    return ok


def _replay_checks(cne: dict, input_data: Dict, checks: dict,
                   reasons: _Reasons) -> None:
    fetcher = rebuild_fetcher(cne, input_data)
    params = cne.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    raw_k = params.get("K", 63)
    if isinstance(raw_k, bool) or not isinstance(raw_k, int) \
            or not 0 <= raw_k <= (1 << 64) - 1:
        raise ValueError("params.K must be an unsigned 64-bit JSON integer")
    raw_alpha = params.get("alpha", 0.05)
    if isinstance(raw_alpha, bool) or not isinstance(raw_alpha, (int, float)) \
            or not math.isfinite(float(raw_alpha)):
        raise ValueError("params.alpha must be a finite JSON number")
    raw_strict = params.get("strict", True)
    if not isinstance(raw_strict, bool):
        raise ValueError("params.strict must be a JSON boolean")
    require_tier = params.get("require_tier")
    if "require_tier" in params and not isinstance(require_tier, str):
        raise ValueError("params.require_tier must be a string when present")
    seed = cne.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)
                             or not 0 <= seed <= (1 << 64) - 1):
        raise ValueError("seed must be an unsigned 64-bit JSON integer")
    rep = certified_run(
        cne["program"], fetcher=fetcher, seed=seed,
        K=raw_k, alpha=float(raw_alpha), strict=raw_strict,
        # Read back from the SIGNED params. An issuer that demanded a tier floor
        # produced a refusal under it; a verifier that ignored the floor would
        # replay to a certificate instead and report a replay mismatch.
        require_tier=require_tier)

    # bit-for-bit only when the local substrate matches the envelope's kernel
    exact = (cne.get("kernel") == _kernel_id())

    refused = cne.get("refused")
    if not isinstance(refused, bool):
        checks["scalar"] = False
        reasons.add("malformed-envelope",
                    "refused must be the JSON boolean true or false")
        return
    if refused:
        _replay_refusal_checks(cne, rep, checks, reasons)
        return

    qb = (cne.get("error_budget") or {}).get("quantization") or {}
    # the SCALAR is well-conditioned — verifiable across substrates to tolerance
    checks["scalar"] = (rep.ok and rep.base_value is not None
                        and _scalar_eq(rep.base_value, cne.get("scalar"), exact,
                                       reasons))
    checks["tier"] = (rep.program_class == cne.get("program_class"))

    # THE BOUND'S MEANING, not just its magnitude. A width without its confidence
    # level and exactness flag is unreadable, and both are signed — so both must
    # be re-derived. These are discrete/rational quantities computed identically
    # on any substrate, so they compare exactly even when the kernel differs.
    budget_ok = True
    budget_why = []
    if not _num_eq(rep.level, qb.get("level"), True):
        budget_ok = False
        budget_why.append(f"confidence level (replayed {rep.level!r}, "
                          f"envelope {qb.get('level')!r})")
    if bool(rep.level_exact) != bool(qb.get("exact")):
        budget_ok = False
        budget_why.append(f"exactness flag (replayed {bool(rep.level_exact)}, "
                          f"envelope {bool(qb.get('exact'))})")
    if qb.get("tier") is not None and rep.program_class != qb.get("tier"):
        budget_ok = False
        budget_why.append(f"budget tier (replayed {rep.program_class!r}, "
                          f"envelope {qb.get('tier')!r})")
    if sorted(cne.get("assumptions", [])) != sorted(rep.assumptions):
        budget_ok = False
        budget_why.append("assumptions do not match the replay")
    if (cne.get("branch_sites") or []) != (rep.branch_sites or []):
        budget_ok = False
        budget_why.append("branch sites do not match the replay")
    checks["budget"] = budget_ok
    if not budget_ok:
        reasons.add("budget-mismatch",
                    "the envelope's stated bound semantics were not "
                    "reproduced: " + "; ".join(budget_why))

    # A dither seed is what makes the bound reproducible. Without one,
    # certified_run reseeds from os.urandom and the width mismatch below would
    # be blamed on arithmetic rather than on the missing field.
    if cne.get("seed") is None:
        checks["width"] = False
        reasons.add("no-seed", "envelope carries no dither seed — its bound is not "
                    "reproducible")
        return
    # A width of EXACTLY ZERO is substrate-independent, and saying otherwise costs
    # the whole point of the exact-cents capture law.
    #
    # When every committed Δ is 0 the resample perturbation is `uniform(...) * 0`,
    # i.e. identically zero, so every pivot |f(x̂+0) − f(x̂)| is exactly 0 and the
    # order statistic is exactly 0.0 — on any kernel, with no near-cancellation
    # anywhere. Refusing to check it off-substrate would mean an actuarial receipt
    # under the exact-cents law could not reach ok=true on a bare `pip install`,
    # which is precisely the deployment product #1's pilot depends on.
    #
    # This is a triviality check, not a relaxation: all THREE of the committed Δ, the
    # stated width, and the replayed width must be exactly 0.0. The SCALAR is still
    # only compared to tolerance off-substrate — Neumaier and pairwise summation
    # genuinely differ there, and nothing here changes that.
    zero_width = (_all_committed_deltas_zero(cne)
                  and _num_eq(qb.get("width"), 0.0, True)
                  and rep.width is not None and _num_eq(rep.width, 0.0, True))
    if exact or zero_width:
        checks["width"] = (rep.width is not None
                           and _num_eq(rep.width, qb.get("width"), True))
        if not checks["width"]:
            reasons.add("width-mismatch",
                        "replayed bound does not match the envelope width")
        elif not exact:
            reasons.add("width-substrate-independent",
                        f"substrate '{_kernel_id()}' differs from the envelope's "
                        f"'{cne.get('kernel')}', but every committed Δ is 0 (stored "
                        f"exactly), so the bound is identically zero on any kernel "
                        f"and was verified exactly; the SCALAR is still compared to "
                        f"relative tolerance {_REL_TOL:g}")
        elif not _is_specified_substrate():
            # Honesty, not a verdict: `numpy-fallback` is a substrate NAME, not a
            # specified numeric semantics. Two machines both reporting it can run
            # different numpy builds whose pairwise summation differs in the last
            # ULPs, so a width that matched here matched by agreement of two
            # unspecified implementations. Only `alelyon-vector/<ver>` is frozen in
            # SPEC-cne-v0.md §8 as bit-reproducible. Does not change `ok`.
            reasons.add("unspecified-substrate",
                        f"substrate '{_kernel_id()}' matched the envelope's, but it "
                        f"is not a SPECIFIED substrate: bit-identical width replay "
                        f"is frozen only for 'alelyon-vector/<version>'. This width "
                        f"agreement is not portable — install the deterministic "
                        f"kernel for a specified verification")
    else:
        # the width is a near-cancellation quantity; it only reproduces on the
        # matching kernel — leave it UNVERIFIED here rather than false-accept
        # or false-reject. (The reference OSS verifier ships the kernel.)
        checks["width"] = None
        reasons.add("substrate-mismatch",
                    f"substrate '{_kernel_id()}' differs from the envelope's "
                    f"'{cne.get('kernel')}': the SCALAR is verified to relative "
                    f"tolerance {_REL_TOL:g}, but the BOUND cannot be reproduced "
                    f"without the matching kernel — install it for full verification")
    if not checks["scalar"]:
        reasons.add("scalar-mismatch", "replayed scalar does not match the envelope")
    if not checks["tier"]:
        reasons.add("tier-mismatch", "replayed honesty tier does not match")


def _replay_refusal_checks(cne: dict, rep, checks: dict,
                           reasons: _Reasons) -> None:
    """Validate a signed refusal's complete v0 shape and replay semantics.

    A valid signature proves who wrote the refusal, not that its stated cause is
    true. The verifier therefore requires the replay to refuse for the same reason
    and program class, and rejects success-only result/bound fields. Unknown
    extension members remain ignored under SPEC section 10.2.
    """
    replay_refused = bool(rep.refused)
    scalar_ok = "scalar" in cne and cne.get("scalar") is None
    stated_reason = cne.get("reason")
    reason_well_formed = (isinstance(stated_reason, str)
                          and bool(stated_reason.strip()))
    reason_matches = (replay_refused and reason_well_formed
                      and stated_reason == rep.reason)
    checks["scalar"] = bool(replay_refused and scalar_ok and reason_matches)

    if not replay_refused:
        reasons.add("replay-refusal-mismatch",
                    "envelope claims refusal but replay produced a bound")
    elif reason_well_formed and not reason_matches:
        reasons.add("replay-refusal-mismatch",
                    "signed refusal reason does not match the independently "
                    "replayed cause")
    elif not reason_well_formed:
        reasons.add("malformed-envelope",
                    "a refusal must carry a non-empty textual reason")
    if not scalar_ok:
        reasons.add("malformed-envelope",
                    "a refusal must carry scalar=null")

    checks["tier"] = (replay_refused
                      and isinstance(cne.get("program_class"), str)
                      and cne.get("program_class") == rep.program_class)
    if not checks["tier"]:
        reasons.add("tier-mismatch",
                    "signed refusal program_class does not match independent replay")

    budget = cne.get("error_budget")
    quantization_ok = (isinstance(budget, dict)
                       and "quantization" in budget
                       and budget.get("quantization") is None)
    known_success_budget = (
        [name for name in ("sampling", "provider", "model") if name in budget]
        if isinstance(budget, dict) else [])
    known_success_top = [name for name in ("seed", "assumptions", "branch_sites")
                         if name in cne]
    checks["budget"] = bool(
        quantization_ok and not known_success_budget and not known_success_top)
    if not checks["budget"]:
        details = []
        if not quantization_ok:
            details.append("error_budget.quantization must be present and null")
        if known_success_budget:
            details.append("success-only budget fields present: "
                           + ", ".join(known_success_budget))
        if known_success_top:
            details.append("success-only envelope fields present: "
                           + ", ".join(known_success_top))
        reasons.add("budget-mismatch",
                    "signed refusal has success-only bound semantics: "
                    + "; ".join(details))



def _all_committed_deltas_zero(cne: dict) -> bool:
    """True iff every input commits a per-row Δ of exactly 0.0 on every row, over at
    least one row. Read from the COMMITTED delta blocks, which the signature covers
    and the transparency anchor proves — not from any field asserting exactness."""
    inputs = [c for c in cne.get("inputs", []) if isinstance(c, dict)]
    if not inputs:
        return False
    total = 0
    for c in inputs:
        try:
            d = _decompress_deltas(c.get("deltas"))
        except Exception:      # noqa: BLE001 - malformed blocks are handled upstream
            return False
        if d.size == 0 or not bool(np.all(d == 0.0)):
            return False
        total += int(d.size)
    return total > 0


def _check_transparency(cne: dict, input_data: Dict, checks: dict,
                        reasons: _Reasons, public_key_hex: Optional[str]) -> None:
    """Prove the committed per-row Δ are the SIGNED capture deltas — not the
    signer's word. For each input carrying a transparency block: verify the STH
    under the pinned key, verify every covering leaf's inclusion (recomputing its
    leaf_hash so the payload Δ is bound to the signed tree), then re-derive the
    per-row Δ from the proven leaves and require it to equal the committed Δ. When
    every input is anchored this makes the WIDTH transparency-anchored; a
    present-but-invalid anchor is a forgery signal (transparency=False → fails
    ok). The anchor detects revision of signed capture commitments, not invention
    at capture."""
    inputs = [c for c in cne.get("inputs", []) if isinstance(c, dict)]
    present = [c for c in inputs if c.get("transparency")]
    if not present:
        return                                       # None: no anchoring attempted
    if public_key_hex is None:
        checks["transparency"] = False
        reasons.add("transparency-no-pinned-key",
                    "transparency anchors present but no pinned key to verify "
                    "their signed tree head")
        return
    any_invalid = False
    verified = 0
    for c in present:
        ok, cls, why = _verify_one_anchor(c, input_data, public_key_hex)
        if ok:
            verified += 1
        else:
            any_invalid = True
            reasons.add(cls,
                        f"transparency anchor for {(c.get('kind'), c.get('key'))} "
                        f"failed: {why}")
    if any_invalid:
        checks["transparency"] = False
    elif verified == len(inputs):
        checks["transparency"] = True                # width is transparency-anchored
    else:
        checks["transparency"] = None                # partial: honest, not a failure
        reasons.add("transparency-partial",
                    f"{verified}/{len(inputs)} inputs transparency-anchored; the "
                    f"rest are signer-authenticated (not log-anchored)")


def _check_provider(cne: dict, checks: dict, reasons: _Reasons,
                    public_key_hex: Optional[str]) -> None:
    """Verify the provider block's CORROBORATION anchors (W3).

    What this can and cannot establish, stated before the code because the
    temptation to overclaim here is specific and strong. It anchors **the record of
    having asked**: which upstreams were probed, and what each one did. With a
    single answering origin that is emphatically NOT a dispersion measurement or a
    cross-source agreement claim, and nothing here upgrades it into one — that limit
    is the program's cycle 5j/5k finding and it survives this change.

    What it does close is the issuer-side hole. `verify_corroboration` detects a
    deleted silence or an edited outcome, but the ISSUER runs it, on its own books;
    a reader had only the issuer's word that a probe was ever made. Carrying the
    probe scope's STH, each leaf's inclusion proof, and the attempt records lets a
    third party re-derive the digest and check the stated summary against leaves
    proven to be in the signed tree.

    A present-but-invalid anchor is a forgery signal and fails `ok`, exactly as the
    capture anchor does. An ABSENT anchor is not: the slot stays `None` and the
    provider trust label stays `signer-attested`.
    """
    # The provider block is a TERM OF THE ERROR BUDGET, not a top-level field — it
    # states what cross-source evidence exists about the inputs, alongside the
    # quantization and sampling terms.
    prov = (cne.get("error_budget") or {}).get("provider")
    if not isinstance(prov, dict):
        return
    anchors = prov.get("anchors")
    if not isinstance(anchors, dict) or not anchors:
        return                                       # None: no anchoring attempted
    if public_key_hex is None:
        checks["provider"] = False
        reasons.add("provider-no-pinned-key",
                    "corroboration anchors present but no pinned key to verify "
                    "their signed tree head")
        return

    stated = prov.get("inputs") or {}
    any_invalid = False
    verified = 0
    for ref, blk in sorted(anchors.items()):
        ok, cls, why = _verify_one_provider_anchor(
            str(ref), blk if isinstance(blk, dict) else {},
            stated.get(ref) if isinstance(stated, dict) else None,
            public_key_hex)
        if ok:
            verified += 1
        else:
            any_invalid = True
            reasons.add(cls, f"corroboration anchor for {ref!r} failed: {why}")
    if any_invalid:
        checks["provider"] = False
    elif verified == len(anchors):
        checks["provider"] = True
    else:  # pragma: no cover - unreachable while every anchor is judged
        checks["provider"] = None
        reasons.add("provider-partial",
                    f"{verified}/{len(anchors)} corroboration anchors verified")


def _verify_one_provider_anchor(ref: str, blk: dict, stated: Optional[dict],
                                public_key_hex: str):
    """(ok, reason_class, reason) for one input's corroboration anchor.

    The checks mirror `_verify_one_anchor` up to the leaf, then diverge: instead of
    re-deriving a Δ they re-derive the ATTEMPT DIGEST from the carried records, and
    then re-derive the summary from those same records rather than reading the one
    the envelope states. That ordering is the point — a summary checked against
    numbers the issuer also wrote would be checked against itself.
    """
    from alelyon.runtime.atlas.data.attest import (verify_tree_head,
                                                   verify_merkle_path, cert_leaf_hash,
                                                   corroboration_digest,
                                                   corroboration_tally,
                                                   CORROBORATION_OUTCOMES)
    sth = blk.get("sth")
    v = verify_tree_head(sth, public_key_hex=public_key_hex)
    if not v["ok"]:
        return False, "provider-sth-invalid", f"STH invalid ({v['reason']})"
    root = sth.get("root")

    scope = blk.get("scope")
    if not (isinstance(scope, (list, tuple)) and len(scope) == 2):
        return False, "provider-malformed-scope", "malformed scope"
    # The scope is DERIVED from the input reference, never read from the untrusted
    # block — the same rule the capture anchor learned the hard way. Otherwise a
    # signer anchors a claim about one input to some other input's probe log, where
    # the sources happened to answer.
    exp = _corroboration_scope_of(ref)
    if exp is None:
        return False, "provider-scope-mismatch", (
            f"cannot derive a canonical corroboration scope for input {ref!r}")
    if [str(scope[0]), str(scope[1])] != list(exp):
        return False, "provider-scope-mismatch", (
            f"anchor scope {[str(scope[0]), str(scope[1])]} is not this input's "
            f"canonical corroboration scope {list(exp)}")
    if str(sth.get("table")) != "corroboration" or \
            [str(x) for x in (sth.get("scope") or [])] != list(exp):
        return False, "provider-sth-scope-mismatch", (
            "signed tree head is for a different scope than the anchor block")

    leaves = blk.get("leaves") or []
    attempts_by_seq = blk.get("attempts") or {}
    if not leaves:
        return False, "provider-malformed-leaf", "anchor carries no leaves"

    all_attempts = []
    for lf in leaves:
        try:
            recomputed = cert_leaf_hash(
                "corroboration", str(scope[0]), str(scope[1]), int(lf["seq"]),
                lf["value_digest"], int(lf["n"]), float(lf["lo_ts"]),
                float(lf["hi_ts"]), int(lf["bits"]), lf["payload"], lf["prev_hash"])
        except (KeyError, TypeError, ValueError):
            return False, "provider-malformed-leaf", "malformed leaf record"
        proof = lf.get("inclusion_proof") or {}
        if recomputed != proof.get("leaf_hash"):
            return False, "provider-leaf-hash-mismatch", (
                "leaf record does not match its committed leaf_hash")
        if int(proof.get("tree_size", -1)) != int(sth.get("tree_size", -2)):
            return False, "provider-proof-tree-size-mismatch", (
                "inclusion proof's tree_size does not match the signed tree head")
        if not 0 <= int(proof.get("index", -1)) < int(sth["tree_size"]):
            return False, "provider-proof-index-out-of-range", (
                "inclusion proof index is outside the signed tree")
        if not verify_merkle_path(proof["leaf_hash"], int(proof["index"]),
                                  int(proof["tree_size"]), list(proof["proof"]), root):
            return False, "provider-inclusion-failed", (
                "leaf inclusion proof does not recompute the STH root")

        carried = attempts_by_seq.get(str(lf["seq"]))
        if carried is None:
            carried = attempts_by_seq.get(int(lf["seq"]))
        if not isinstance(carried, list):
            return False, "provider-malformed-leaf", (
                f"leaf {lf.get('seq')} carries no attempt records to re-derive")
        try:
            tuples = [(str(a[0]), str(a[1]), str(a[2]),
                       None if a[3] is None else float(a[3])) for a in carried]
        except (IndexError, TypeError, ValueError):
            return False, "provider-malformed-leaf", (
                f"leaf {lf.get('seq')}: malformed attempt record")
        # An outcome outside the closed vocabulary is a malformed record, not a new
        # kind of answer. Accepting it would let a novel string sit in the digest
        # and be counted as neither answered nor silent by a future reader.
        for t in tuples:
            if t[2] not in CORROBORATION_OUTCOMES:
                return False, "provider-outcome-unknown", (
                    f"leaf {lf.get('seq')}: outcome {t[2]!r} is outside the closed "
                    f"vocabulary {sorted(CORROBORATION_OUTCOMES)}")
        # A DELETED silence is the headline forgery, and it is caught here rather
        # than by the digest: the count was committed into the leaf separately, so
        # removing a row changes n before it changes anything else.
        if len(tuples) != int(lf["n"]):
            return False, "provider-attempt-count-mismatch", (
                f"leaf {lf.get('seq')} committed {int(lf['n'])} attempts, "
                f"{len(tuples)} carried — {int(lf['n']) - len(tuples)} deleted")
        if corroboration_digest(tuples) != str(lf["value_digest"]):
            return False, "provider-attempts-digest-mismatch", (
                f"leaf {lf.get('seq')}: carried attempts do not re-derive the digest "
                f"the signed leaf committed — an outcome or value was edited")
        all_attempts.extend(tuples)

    # Finally the summary, re-derived from the PROVEN records rather than read.
    if isinstance(stated, dict):
        claimed = {k: stated.get(k) for k in ("asked", "answered", "silent")
                   if stated.get(k) is not None}
        if claimed:
            actual = corroboration_tally(all_attempts)
            for k, v_ in claimed.items():
                if int(v_) != actual[k]:
                    return False, "provider-summary-mismatch", (
                        f"envelope states {k}={v_} but the proven leaves say "
                        f"{k}={actual[k]}")
    return True, None, None


def _corroboration_scope_of(ref: str):
    """The canonical `(scope1, scope2)` of an input's corroboration log.

    Derived from the input reference the same way `_scope_of` derives a capture
    scope, and returns None rather than guessing when the reference is not one this
    verifier knows how to scope — an unknown reference must fail closed, not be
    anchored against whatever scope the envelope proposed.
    """
    kind, _, key = str(ref).partition("|")
    if not key:
        kind, key = "price", str(ref)
    if kind not in ("price", "bars"):
        return None
    tick, _, interval = key.partition("@")
    return (tick.upper(), interval or "1d")


def _check_witness(cne: dict, checks: dict, reasons: _Reasons,
                   witness_key_hex: Optional[str]) -> None:
    """Verify co-signatures from the witness seam (the equivocation guard).

    Each anchored input may carry a `transparency.cosignature` over that input's
    complete STH. With a witness key pinned out of band every co-signature is
    checked against it and bound to that complete STH; a present-but-invalid one is
    a forgery signal (witness=False → fails ok). Without a pinned witness key the
    co-signatures are reported but not judged (witness=None). Independence is a
    deployment fact and exists only when another party operates the witness."""
    inputs = [c for c in cne.get("inputs", []) if isinstance(c, dict)]
    cosigned = []
    for c in inputs:
        blk = c.get("transparency") or {}
        cosig = blk.get("cosignature")
        if cosig:
            cosigned.append((c, cosig, blk.get("sth") or {}))
    if not cosigned:
        return                                       # None: no co-signatures present
    if witness_key_hex is None:
        reasons.add("witness-unpinned",
                    "witness co-signatures present but no witness key pinned — "
                    "not equivocation-checked (pin the witness key out of band; "
                    "operator independence is a separate deployment property)")
        return                                       # None: present but unjudged
    from alelyon.runtime.atlas.data.attest import verify_cosignature
    any_invalid = False
    verified = 0
    for c, cosig, sth in cosigned:
        v = verify_cosignature(cosig, witness_key_hex=witness_key_hex,
                               expected_root=sth.get("root"), expected_sth=sth)
        if v["ok"]:
            verified += 1
        else:
            any_invalid = True
            reasons.add("witness-cosignature-invalid",
                        f"witness co-signature for {(c.get('kind'), c.get('key'))} "
                        f"failed: {v['reason']}")
    if any_invalid:
        checks["witness"] = False
    elif verified == len(inputs):
        checks["witness"] = True                     # every input co-signed
    else:
        checks["witness"] = None                     # partial: honest, not a failure
        reasons.add("witness-partial",
                    f"{verified}/{len(inputs)} inputs witness-co-signed; the rest "
                    f"carry no equivocation guard")


def _check_key_status(cne: dict, checks: dict, reasons: _Reasons,
                      key_manifest: Optional[dict],
                      manifest_root_hex: Optional[str],
                      manifest_checkpoint: Optional[dict],
                      checkpoint_public_key_hex: Optional[str],
                      trusted_manifest_checkpoint: Optional[dict]) -> None:
    """Place the envelope's signing key in the issuer's published key history.

    Not supplied -> `key_status=None`: honestly not judged, exactly like an absent
    witness key. Supplying a manifest is opting IN to a stricter check, and opting in
    without the root it chains to is refused rather than half-performed — a manifest
    verified against nothing is a list of keys vouching for each other, which an
    attacker generates wholesale.

    A `revoked` key fails whenever the envelope was issued (see
    `keylife.key_status_at`): if the cause was compromise, "issued before the
    revocation date" establishes nothing, and waving those through would make
    revocation advisory.
    """
    checkpoint_args = (manifest_checkpoint, checkpoint_public_key_hex,
                       trusted_manifest_checkpoint)
    if key_manifest is None:
        if any(value is not None for value in checkpoint_args):
            checks["key_status"] = False
            reasons.add("key-manifest-checkpoint-required",
                        "manifest checkpoint material was supplied without the key "
                        "manifest it is meant to commit")
        return
    from alelyon.runtime.atlas.data.keylife import (key_status_at,
                                                    verify_manifest_checkpoint)
    if manifest_root_hex is None:
        checks["key_status"] = False
        reasons.add("key-manifest-unrooted",
                    "a key manifest was supplied with no pinned root key — the "
                    "succession chain would then vouch only for itself; supply the "
                    "root obtained out-of-band")
        return
    if any(value is None for value in checkpoint_args):
        checks["key_status"] = False
        reasons.add("key-manifest-checkpoint-required",
                    "a key manifest is accepted only with its signed checkpoint, "
                    "the checkpoint key pinned out of band, and the verifier's "
                    "previously trusted signed checkpoint state")
        return
    checkpointed = verify_manifest_checkpoint(
        key_manifest, manifest_checkpoint,
        root_public_key_hex=manifest_root_hex,
        checkpoint_public_key_hex=checkpoint_public_key_hex,
        trusted_checkpoint=trusted_manifest_checkpoint)
    if not checkpointed["ok"]:
        checks["key_status"] = False
        failure = checkpointed.get("failure")
        if failure == "not-monotonic":
            reasons.add("key-manifest-checkpoint-not-monotonic",
                        f"key manifest checkpoint is stale or equivocal relative "
                        f"to retained verifier state: {checkpointed['reason']}")
        elif failure == "required":
            reasons.add("key-manifest-checkpoint-required",
                        checkpointed["reason"])
        elif failure == "manifest-invalid":
            reasons.add("key-manifest-invalid",
                        checkpointed["reason"])
        else:
            reasons.add("key-manifest-checkpoint-invalid",
                        f"key manifest or checkpoint is invalid: "
                        f"{checkpointed['reason']}")
        return
    verified = checkpointed["manifest"]

    status, entry = key_status_at(verified, cne.get("key_id"), cne.get("created"))
    if status == "valid":
        checks["key_status"] = True
        return
    checks["key_status"] = False
    if status == "revoked":
        rev = (entry or {}).get("revocation") or {}
        reasons.add("key-revoked",
                    f"the signature is valid, but the signing key "
                    f"{cne.get('key_id')} was REVOKED as of "
                    f"{rev.get('revoked_at')} (reason: {rev.get('reason')!r}, "
                    f"attested by {rev.get('signer_key_id')}). A revocation applies "
                    f"to everything the key signed: if the cause was compromise, the "
                    f"key was held by someone else for an unknown period before "
                    f"anyone noticed")
    elif status == "unknown":
        reasons.add("key-not-in-manifest",
                    f"the signing key {cne.get('key_id')} does not appear in the "
                    f"issuer's published key history")
    else:
        reasons.add("key-outside-validity",
                    f"the signing key {cne.get('key_id')} was not in service at the "
                    f"envelope's stated creation time {cne.get('created')!r} "
                    f"(in service {(entry or {}).get('not_before')} to "
                    f"{(entry or {}).get('not_after')})")


def _verify_one_anchor(c: dict, input_data: Dict, public_key_hex: str):
    """(ok, reason_class, reason) for one input's transparency block. The class is
    part of the wire contract (SPEC-cne-v0.md §9); the prose is not."""
    from alelyon.runtime.atlas.data.attest import (verify_tree_head,
                                                   verify_merkle_path, cert_leaf_hash,
                                                   payload_deltas, payload_laws,
                                                   validated_payload_membership)
    blk = c.get("transparency") or {}
    sth = blk.get("sth")
    v = verify_tree_head(sth, public_key_hex=public_key_hex)
    if not v["ok"]:
        return False, "anchor-sth-invalid", f"STH invalid ({v['reason']})"
    root = sth.get("root")
    table, scope, col = blk.get("table"), blk.get("scope"), blk.get("column")
    if not (isinstance(scope, (list, tuple)) and len(scope) == 2):
        return False, "anchor-malformed-scope", "malformed scope"

    # BIND THE ANCHOR TO THE INPUT IT CLAIMS TO CERTIFY. Without this the log
    # identity is attacker-chosen: a signer could anchor a claim about one series
    # to an unrelated capture scope whose deltas happen to be smaller, and forge
    # an arbitrarily tighter transparency-anchored width that still verifies. The scope is
    # DERIVED from the input commitment, never read from the untrusted block.
    exp_table, exp_s1, exp_s2, exp_col = _scope_of((c.get("kind"), c.get("key")))
    if (str(table), [str(scope[0]), str(scope[1])], str(col)) != \
            (exp_table, [exp_s1, exp_s2], exp_col):
        return False, "anchor-scope-mismatch", (
            f"anchor scope {(table, list(scope), col)} is not this "
            f"input's canonical scope {(exp_table, [exp_s1, exp_s2], exp_col)}")
    # ...and the SIGNED head must be a head of that same scope, so a valid STH
    # for a different scope cannot be substituted.
    if str(sth.get("table")) != exp_table or \
            [str(x) for x in (sth.get("scope") or [])] != [exp_s1, exp_s2]:
        return False, "anchor-sth-scope-mismatch", (
            "signed tree head is for a different scope than the "
            "transparency block")

    s1, s2 = scope
    leaf_claims = []
    for lf in blk.get("leaves", []):
        try:
            recomputed = cert_leaf_hash(
                table, s1, s2, int(lf["seq"]), lf["value_digest"], int(lf["n"]),
                float(lf["lo_ts"]), float(lf["hi_ts"]), int(lf["bits"]),
                lf["payload"], lf["prev_hash"])
        except (KeyError, TypeError, ValueError):
            return False, "anchor-malformed-leaf", "malformed leaf record"
        proof = lf.get("inclusion_proof") or {}
        if recomputed != proof.get("leaf_hash"):
            return False, "anchor-leaf-hash-mismatch", (
                "leaf record does not match its committed leaf_hash")
        # Pin the proof's self-declared geometry to the SIGNED head. Otherwise a
        # proof carries its own tree_size and is checked against a root it was
        # never claimed to belong to.
        if int(proof.get("tree_size", -1)) != int(sth.get("tree_size", -2)):
            return False, "anchor-proof-tree-size-mismatch", (
                "inclusion proof's tree_size does not match the signed tree head")
        if not 0 <= int(proof.get("index", -1)) < int(sth["tree_size"]):
            return False, "anchor-proof-index-out-of-range", (
                "inclusion proof index is outside the signed tree")
        if not verify_merkle_path(proof["leaf_hash"], int(proof["index"]),
                                  int(proof["tree_size"]), list(proof["proof"]), root):
            return False, "anchor-inclusion-failed", (
                "leaf inclusion proof does not recompute the STH root")
        # A Δ the signer never wrote is UNKNOWN, not zero. Reading an absent field
        # as 0.0 let a key-holding signer omit it at capture and have the same
        # fabricated zero appear on BOTH sides of this comparison — an exactly
        # zero-width bound that verified as transparency-anchored.
        cols, unusable = payload_deltas(lf["payload"])
        if str(col).lower() in unusable:
            return False, "anchor-delta-unusable", (
                f"proven capture leaf {lf.get('seq')} commits no usable Δ "
                f"for column '{col}' — a Δ that was never written, or was written "
                f"under a capture law this verifier does not implement, cannot be "
                f"read as exact")
        laws = payload_laws(lf["payload"])
        membership = None
        if c.get("kind") != TABLE_KIND:
            membership = validated_payload_membership(
                lf["payload"], str(table), int(lf["n"]),
                float(lf["lo_ts"]), float(lf["hi_ts"]))
            if membership is None:
                return False, "anchor-row-uncovered", (
                    f"proven capture leaf {lf.get('seq')} does not commit a valid "
                    "exact timestamp membership; its min/max span is not coverage")
            membership = frozenset(membership)
        leaf_claims.append((membership, cols.get(str(col).lower()),
                            laws.get(str(col).lower()),
                            str(lf.get("value_digest"))))
    # re-derive per-row Δ from the PROVEN leaves; must equal the committed Δ
    series = input_data.get((c.get("kind"), c.get("key")), input_data.get(c.get("key")))
    if series is None:
        return False, "anchor-no-data", "no data supplied to re-derive the anchored Δ"
    s = canonical_series(series)
    committed = _decompress_deltas(c["deltas"])
    vals = s.to_numpy(dtype=float, copy=False)

    if c.get("kind") == TABLE_KIND:
        # A keyed table has no time axis, so coverage is DIGEST IDENTITY against a
        # proven capture leaf rather than interval containment — strictly stronger,
        # and decided on the verifier's own copy of the data (see
        # `_anchor_table_input`). `lo_ts`/`hi_ts` carry no meaning for a table scope
        # and MUST NOT be interval-tested.
        digest = table_digest(s)
        matched = [(d, law) for _membership, d, law, vd in leaf_claims
                   if d is not None and vd == digest]
        if not matched:
            return False, "anchor-row-uncovered", (
                "no proven capture leaf commits this table: its (key, value) digest "
                "matches none of the signed batches, so nothing vouches for the Δ")
        best, best_law = max(matched, key=lambda t: t[0])
        if len(committed) != len(s):
            return False, "anchor-length-mismatch", (
                "committed Δ length differs from the supplied table")
        for i in range(len(s)):
            if float(best) != float(committed[i]):
                return False, "anchor-delta-mismatch", (
                    "committed Δ does not match the signed capture leaf")
            bad = _zero_delta_implausible(best, best_law, vals[i])
            if bad is not None:
                return False, "anchor-delta-zero-implausible", bad
        return True, None, None

    ts = [float(pd.Timestamp(i).timestamp()) for i in s.index]
    if len(committed) != len(ts):
        return False, "anchor-length-mismatch", (
            "committed Δ length differs from the supplied data")
    for i, t in enumerate(ts):
        best, best_law = None, None
        for membership, d, law, _vd in leaf_claims:
            if d is not None and membership is not None and t in membership:
                if best is None or d > best:
                    best, best_law = d, law
        if best is None:
            return False, "anchor-row-uncovered", (
                "a consumed row is not covered by any proven capture leaf")
        if float(best) != float(committed[i]):
            return False, "anchor-delta-mismatch", (
                "committed Δ does not match the signed capture leaves")
        bad = _zero_delta_implausible(best, best_law, vals[i], column=col)
        if bad is not None:
            return False, "anchor-delta-zero-implausible", bad
    return True, None, None


def _zero_delta_implausible(delta, law, value, *, column: str = "?"):
    """A reason string when a Δ=0 claim is impossible for this value under this
    capture law, else None. Evaluated PER ELEMENT.

    Δ=0 means "stored exactly", and it is checkable without trusting anyone because
    the verifier holds its OWN copy of the data — but what makes it checkable is
    LAW-SPECIFIC, so the law has to be dispatched on rather than assumed:

      relative-dither (the default, and every leaf written before laws existed):
        Δ = scale·2^(1-bits) > 0 for any column with a nonzero scale, so Δ=0 means
        every finite value in the batch was 0. A nonzero value refutes it.

      exact-cents: the stored value IS an integer count of cents, so Δ=0 is the
        NORMAL case — but only for values that really are whole cents inside f64's
        exact-integer range. A fractional or oversized value refutes it.

    Per element, not per column, because of the 5j lesson: one benign element must
    never license the rest. A single genuine zero in a column would otherwise let a
    writer claim exactness across every other row.

    An unrecognised law never reaches here — `payload_deltas` has already marked its
    column unusable, which fails the anchor closed one step earlier.
    """
    if float(delta) != 0.0 or not np.isfinite(value):
        return None
    v = float(value)
    if law == _EXACT_CENTS_LAW:
        if abs(v) > _SAFE_EXACT_INT:
            return (f"a proven capture leaf claims the exact-cents law stored "
                    f"column '{column}' exactly, over a value {v!r} beyond 2^53 "
                    f"cents — past that point f64 does not represent consecutive "
                    f"integers, so 'exact' is not available for this value")
        if v != round(v):
            return (f"a proven capture leaf claims the exact-cents law stored "
                    f"column '{column}' exactly, over {v!r}, which is not a whole "
                    f"number of cents — that law stores an INTEGER cent count, so "
                    f"this leaf could not have been produced by it")
        return None
    if v != 0.0:
        return (f"a proven capture leaf claims Δ=0 (stored exactly) for column "
                f"'{column}' over a row whose value is {v!r} — the capture "
                f"quantizer emits Δ=0 only for an all-zero column, so this bound "
                f"could not have been produced by the pipeline that signed it")
    return None


def _scalar_eq(replayed, stated, exact: bool, reasons: "_Reasons") -> bool:
    """Compare the replayed figure to the stated one, tightening to EXACT wherever the
    substrate cannot be the reason they differ — and disclosing the window when it can.

    Off-substrate the scalar is compared to a relative tolerance, because a mismatched
    kernel genuinely moves the last bits. But a *relative* window on a large figure is a
    large *absolute* window, and that was a real hole: the Phase-4 tamper reel edits a
    $1,565,884,000.00 control total by one dollar, and at 1e-9 relative that is an
    absolute window of ±$1.57 — so the forgery VERIFIED on a fallback verifier. A
    penny-tie-out product cannot ship a verifier that tolerates a dollar.

    Two changes, neither of which can cause a false reject:

    1. **Integer figures compare EXACTLY, on any substrate.** A sum of integers is exact
       in f64 up to 2^53 on every implementation — Neumaier and pairwise summation agree
       because neither rounds — so if both the replayed and stated values are integral and
       in range, any difference is a real difference and no tolerance is warranted. This
       covers the exact-cents product entirely: its figures are integer cent counts.
    2. **When the tolerance path IS taken, say how wide it is.** A reader cannot weigh a
       verdict whose sensitivity is undisclosed, so the absolute window is reported in
       the reasons (advisory: it does not change `ok`). Hiding it would be the same
       defect class as a blank budget slot.

    The general non-integral case keeps its tolerance rather than being tightened by
    guesswork: an ill-conditioned smooth-tier program legitimately differs by more than a
    few ULPs across kernels, and a false reject on a genuine receipt is its own failure.
    Closing that properly needs a conditioning-aware bound; it is a recorded open item,
    not something to approximate here.
    """
    if replayed is None or stated is None:
        return replayed is None and stated is None
    if exact:
        return _num_eq(replayed, stated, True)

    a, b = float(replayed), float(stated)
    if (a == a and b == b                                   # neither NaN
            and abs(a) <= _SAFE_EXACT_INT and abs(b) <= _SAFE_EXACT_INT
            and a == round(a) and b == round(b)):
        return a == b                                       # integers: no slack needed

    ok = _num_eq(a, b, False)
    if ok:
        window = _REL_TOL * max(abs(a), abs(b), 1e-300)
        reasons.add("scalar-tolerance-window",
                    f"the figure was compared to a RELATIVE tolerance of "
                    f"{_REL_TOL:g} because the substrate differs, which at this "
                    f"magnitude is an absolute window of ±{window:.6g}. A discrepancy "
                    f"smaller than that would not be detected here; install the "
                    f"deterministic kernel for an exact comparison")
    return ok


def _num_eq(a, b, exact: bool) -> bool:
    """Bit-for-bit when the substrate matched; else within a tight relative
    tolerance. Both-None and both-NaN compare equal."""
    if a is None or b is None:
        return a is None and b is None
    fa, fb = float(a), float(b)
    if fa != fa or fb != fb:
        return fa != fa and fb != fb
    if exact:
        return fa == fb
    scale = max(abs(fa), abs(fb), 1e-300)
    return abs(fa - fb) <= _REL_TOL * scale
