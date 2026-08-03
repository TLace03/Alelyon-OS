"""Key lifecycle for a small signing authority — succession, revocation, manifest
(Track 0, W6).

The problem this closes. A CNE is authenticated by a key the verifier pinned out of
band, and until now there was exactly one key, generated on first use into a
gitignored PEM, with no ceremony record, no rotation mechanism and no revocation
story. That is survivable for a dev machine and unacceptable for an external party:
they cannot pin a key that has no out-of-band publication story, and they have no way
to learn that a key they pinned last year has since been retired or compromised.

What is cryptographic here, and what is merely transport. The **succession chain** is
the cryptographic content: key #1 is anchored out of band, and every later key is
introduced by a statement signed by its PREDECESSOR. A client pins the root once and
can then follow the chain forward without trusting us again. The **manifest** is only
the envelope that carries the chain plus status; it has no authority of its own, which
is why `verify_key_manifest` demands a pinned root and refuses without one. A manifest
checked against nothing vouches for nothing — the same rule as every other pin in this
system.

Three statuses, and the distinction between two of them is the whole point:

  active      currently signing.
  superseded  rotated out in the ordinary course. Envelopes it issued inside its
              validity window STILL VERIFY. Routine rotation must not invalidate
              history, or nobody will ever rotate.
  revoked     withdrawn for cause. Envelopes it issued do NOT get a bare ok, whenever
              they were issued — because if the cause was compromise, an attacker held
              the key for an unknown period before anyone noticed, so "issued before
              the revocation date" is not evidence of anything.

Revocation is normally attested by a SUCCESSOR, not by the revoked key itself. A
compromised key's own signature proves nothing about its compromise: whoever stole it
can sign that too. Self-signed revocation is accepted for a voluntary retirement and
is labeled as such, so a reader can tell the two apart.

Pure layer: stdlib + `cryptography`, no store and no engine, so it ships inside the
open verifier.

The signed checkpoint layer commits the exact manifest and its complete retained key
entries under a separately pinned checkpoint key. A verifier that retains its last
accepted signed checkpoint rejects sequence rollback, same-sequence equivocation,
successor truncation, status rollback, and revocation removal or rewriting. This is
relative freshness, not an oracle for the globally newest state: a first-time client
still needs an initial checkpoint out of band, and checkpoint-key compromise remains
outside this mechanism's protection. The checkpoint key identifier and material must
not overlap any manifest signing key. That cryptographic role separation is enforced
here; organizational custody separation remains a deployment property.
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Dict, List, Optional, Tuple

from alelyon.runtime.atlas.data.attest import KeyStore, canonical

SUCCESSION_TYPE = "alelyon.keysuccession/v0"
REVOCATION_TYPE = "alelyon.keyrevocation/v0"
MANIFEST_TYPE = "alelyon.keymanifest/v0"
CHECKPOINT_TYPE = "alelyon.keymanifest-checkpoint/v0"

#: Closed vocabulary. A free-text status could not be audited, and a status is a
#: claim about whether to trust signatures — the same reason the corroboration
#: ledger's outcomes are closed.
STATUSES = ("active", "superseded", "revoked")

#: Why a key left service. `compromise` is the case that makes past envelopes
#: untrustworthy; `retired` is voluntary. Recorded rather than inferred, because the
#: two demand different responses from a reader.
REVOCATION_REASONS = ("compromise", "retired", "superseded-early", "lost")


# ── statement builders (issuer side) ─────────────────────────────────────────
def key_succession_statement(predecessor: KeyStore, successor_public_key_hex: str,
                             *, not_before: float) -> dict:
    """The OLD key attests the NEW one. This is what lets a client who pinned the
    root follow the chain forward without a second out-of-band exchange.

    Signed by the predecessor, over the successor's identity — so a key can only
    enter the chain if the key already in it says so.
    """
    body = {
        "type": SUCCESSION_TYPE,
        "predecessor_key_id": predecessor.key_id(),
        "key_id": KeyStore.key_id_of(successor_public_key_hex),
        "public_key": str(successor_public_key_hex),
        "not_before": float(not_before),
    }
    body["signature"] = predecessor.sign(canonical(body)).hex()
    return body


def key_revocation_statement(signer: KeyStore, revoked_key_id: str, *,
                             revoked_at: float, reason: str) -> dict:
    """Withdraw a key for cause, signed by `signer` — normally the SUCCESSOR.

    `reason` must be from `REVOCATION_REASONS`: a revocation whose cause is unstated
    cannot be acted on, and `compromise` versus `retired` is the difference between
    "distrust everything this key ever signed" and "it simply stopped signing".
    """
    if reason not in REVOCATION_REASONS:
        raise ValueError(f"revocation reason {reason!r} is not one of "
                         f"{list(REVOCATION_REASONS)}")
    body = {
        "type": REVOCATION_TYPE,
        "key_id": str(revoked_key_id),
        "revoked_at": float(revoked_at),
        "reason": str(reason),
        "signer_key_id": signer.key_id(),
    }
    body["signature"] = signer.sign(canonical(body)).hex()
    return body


def build_key_manifest(issuer: str, entries: List[dict], *,
                       published_at: float) -> dict:
    """Package the chain for publication at a stable URL.

    Deliberately UNSIGNED as a whole. Signing the manifest with the current key would
    invite a reader to treat that signature as the authority, and it is not: the
    authority is the succession chain from the pinned root, which every entry carries
    its own proof of. One aggregate signature would also mean a compromised current
    key could rewrite the history of its own predecessors.

    This object alone does NOT provide freshness or completeness. Verification of a
    supplied manifest therefore requires its signed monotonic checkpoint, a separately
    pinned checkpoint key, and client-held prior signed checkpoint state.
    """
    return {
        "type": MANIFEST_TYPE,
        "issuer": str(issuer),
        "root_key_id": str(entries[0]["key_id"]) if entries else None,
        "keys": list(entries),
        "published_at": float(published_at),
    }


def _manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(canonical(manifest)).hexdigest()


def _status_ids(manifest: dict, status: str) -> List[str]:
    return [str(entry["key_id"]) for entry in manifest["keys"]
            if entry.get("status") == status]


def build_manifest_checkpoint(manifest: dict, signer: KeyStore, *,
                              sequence: int, issued_at: float) -> dict:
    """Sign a monotonic commitment to one exact key manifest.

    The checkpoint key is a separate trust role and must be pinned out of band by
    verifiers. Its signature does not replace succession-chain verification. The
    ordered key/status summaries let a verifier retaining a prior checkpoint reject
    successor truncation, revocation stripping, and status rollback even if a later
    checkpoint accidentally commits a regressed manifest.
    """
    if type(sequence) is not int or sequence < 1:
        raise ValueError("checkpoint sequence must be a positive integer")
    if not isinstance(manifest, dict) or manifest.get("type") != MANIFEST_TYPE:
        raise ValueError("checkpoint target is not a key manifest")
    keys = manifest.get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("checkpoint target lists no keys")
    checkpoint_key_id = signer.key_id()
    checkpoint_public_key = signer.public_key_hex()
    for entry in keys:
        if not isinstance(entry, dict):
            raise ValueError("checkpoint target contains a non-object key entry")
        if entry.get("key_id") == checkpoint_key_id:
            raise ValueError("checkpoint key must have a distinct identifier from "
                             "every manifest signing key")
        entry_public_key = entry.get("public_key")
        if _hex_bytes(entry_public_key, 32) and \
                bytes.fromhex(entry_public_key) == bytes.fromhex(checkpoint_public_key):
            raise ValueError("checkpoint key must use distinct key material from "
                             "every manifest signing key")
    if not _finite_number(manifest.get("published_at")):
        raise ValueError("checkpoint target published_at must be finite")
    if not _finite_number(issued_at) or float(issued_at) < float(manifest["published_at"]):
        raise ValueError("checkpoint issued_at must be finite and no earlier than "
                         "the manifest published_at")
    body = {
        "type": CHECKPOINT_TYPE,
        "issuer": manifest.get("issuer"),
        "root_key_id": manifest.get("root_key_id"),
        "sequence": sequence,
        "manifest_digest": _manifest_digest(manifest),
        "manifest_published_at": manifest.get("published_at"),
        "entries": copy.deepcopy(keys),
        "key_ids": [str(entry.get("key_id")) for entry in keys],
        "superseded_key_ids": _status_ids(manifest, "superseded"),
        "revoked_key_ids": _status_ids(manifest, "revoked"),
        "checkpoint_key_id": checkpoint_key_id,
        "issued_at": float(issued_at),
    }
    body["signature"] = signer.sign(canonical(body)).hex()
    return body


def manifest_entry(key_id: str, public_key_hex: str, *, not_before: float,
                   status: str, not_after: Optional[float] = None,
                   succession: Optional[dict] = None,
                   revocation: Optional[dict] = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {list(STATUSES)}")
    return {
        "key_id": str(key_id), "public_key": str(public_key_hex),
        "not_before": float(not_before),
        "not_after": None if not_after is None else float(not_after),
        "status": str(status),
        "succession": succession, "revocation": revocation,
    }


# ── verification (client side) ───────────────────────────────────────────────
def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _hex_bytes(value, length: int) -> bool:
    return (isinstance(value, str) and len(value) == 2 * length
            and all(char in "0123456789abcdef" for char in value))


def verify_key_manifest(manifest: dict, *,
                        root_public_key_hex: Optional[str] = None) -> dict:
    """Verify a key manifest's succession chain against a PINNED ROOT.

    Returns `{ok, reason, keys: {key_id: entry}, order: [key_id...]}`. `ok=True`
    means: the first entry is the pinned root, every later key is attested by a
    statement signed by its immediate predecessor, every entry's `key_id` really is
    the fingerprint of its `public_key`, and every revocation present is signed by a
    key entitled to say so.

    A pinned root is REQUIRED. Without it the chain is a list of keys vouching for
    each other, which an attacker can generate wholesale — exactly the self-signing
    problem `verify_tree_head` and `verify_cosignature` already refuse.
    """
    out: Dict[str, dict] = {}
    order: List[str] = []

    def bad(reason: str) -> dict:
        return {"ok": False, "reason": reason, "keys": out, "order": order}

    if not isinstance(manifest, dict) or manifest.get("type") != MANIFEST_TYPE:
        return bad("not a key manifest")
    if not isinstance(manifest.get("issuer"), str) or not manifest["issuer"]:
        return bad("manifest issuer must be a non-empty string")
    if not _finite_number(manifest.get("published_at")):
        return bad("manifest published_at must be a finite number")
    if root_public_key_hex is None:
        return bad("no pinned root key — a manifest verified against nothing "
                   "vouches for nothing; supply the root obtained out-of-band")
    try:
        if not _hex_bytes(root_public_key_hex, 32):
            raise ValueError
        root_id = KeyStore.key_id_of(root_public_key_hex)
    except (ValueError, TypeError):
        return bad("pinned root key is not valid hex")

    keys = manifest.get("keys")
    if not isinstance(keys, list) or not keys:
        return bad("manifest lists no keys")
    if manifest.get("root_key_id") != root_id:
        return bad("manifest root_key_id does not match the pinned root")

    prev: Optional[dict] = None
    for i, e in enumerate(keys):
        if not isinstance(e, dict):
            return bad(f"entry {i} is not an object")
        kid, pub = e.get("key_id"), e.get("public_key")
        if not isinstance(kid, str) or not isinstance(pub, str):
            return bad(f"entry {i} has no key_id/public_key")
        if kid in out:
            return bad(f"entry {i} duplicates key_id {kid!r}")
        # The fingerprint must actually be OF this public key, or the chain could
        # attest one identity while carrying another's key material.
        try:
            if not _hex_bytes(pub, 32):
                raise ValueError
            if KeyStore.key_id_of(pub) != kid:
                return bad(f"entry {i} key_id does not match its public_key")
        except (ValueError, TypeError):
            return bad(f"entry {i} public_key is not valid hex")
        if e.get("status") not in STATUSES:
            return bad(f"entry {i} has status {e.get('status')!r}, "
                       f"not one of {list(STATUSES)}")
        if not _finite_number(e.get("not_before")):
            return bad(f"entry {i} has no numeric not_before")
        nb = float(e["not_before"])
        not_after = e.get("not_after")
        if not_after is not None:
            if not _finite_number(not_after):
                return bad(f"entry {i} has a non-finite not_after")
            if float(not_after) < nb:
                return bad(f"entry {i} not_after precedes not_before")

        if i == 0:
            if kid != root_id or bytes.fromhex(pub) != bytes.fromhex(root_public_key_hex):
                return bad("the manifest's first key is not the pinned root")
            if e.get("succession") is not None:
                return bad("the root key must not carry a succession statement — "
                           "it is anchored out-of-band, not by a predecessor")
        else:
            why = _check_succession(e.get("succession"), prev, kid, pub, nb)
            if why:
                return bad(f"entry {i}: {why}")
            if nb < float(prev["not_before"]):
                return bad(f"entry {i} not_before precedes its predecessor's")

        # status/revocation consistency, both directions
        rev = e.get("revocation")
        if (e["status"] == "revoked") != (rev is not None):
            return bad(f"entry {i}: status {e['status']!r} disagrees with the "
                       f"presence of a revocation statement")
        if e["status"] == "superseded" and e.get("not_after") is None:
            return bad(f"entry {i}: a superseded key must carry not_after, or its "
                       f"validity window is unbounded and 'superseded' means nothing")
        if rev is not None:
            why = _check_revocation(rev, kid, keys, i)
            if why:
                return bad(f"entry {i}: {why}")

        out[kid] = e
        order.append(kid)
        prev = e

    active = [kid for kid in order if out[kid].get("status") == "active"]
    if len(active) > 1:
        return bad("manifest has more than one active key")
    if active and active[0] != order[-1]:
        return bad("only the final key in the succession chain may be active")
    return {"ok": True, "reason": None, "keys": out, "order": order}


def _checkpoint_error(checkpoint: dict, *, checkpoint_public_key_hex: str,
                      manifest: Optional[dict] = None) -> Optional[str]:
    if not isinstance(checkpoint, dict) or checkpoint.get("type") != CHECKPOINT_TYPE:
        return "not a key-manifest checkpoint"
    if not isinstance(checkpoint.get("issuer"), str) or not checkpoint["issuer"]:
        return "checkpoint issuer must be a non-empty string"
    if not isinstance(checkpoint.get("root_key_id"), str):
        return "checkpoint root_key_id is required"
    if type(checkpoint.get("sequence")) is not int or checkpoint["sequence"] < 1:
        return "checkpoint sequence must be a positive integer"
    if not _hex_bytes(checkpoint.get("manifest_digest"), 32):
        return "checkpoint manifest_digest must be 32 bytes of hex"
    if not _finite_number(checkpoint.get("manifest_published_at")):
        return "checkpoint manifest_published_at must be a finite number"
    if not _finite_number(checkpoint.get("issued_at")):
        return "checkpoint issued_at must be a finite number"
    if float(checkpoint["issued_at"]) < float(checkpoint["manifest_published_at"]):
        return "checkpoint issued_at precedes the manifest published_at"
    entries = checkpoint.get("entries")
    if not (isinstance(entries, list) and entries
            and all(isinstance(entry, dict) for entry in entries)):
        return "checkpoint entries must be a non-empty array of key entries"
    for field in ("key_ids", "superseded_key_ids", "revoked_key_ids"):
        values = checkpoint.get(field)
        if not (isinstance(values, list)
                and all(isinstance(value, str) for value in values)
                and len(values) == len(set(values))):
            return f"checkpoint {field} must be an array of unique key ids"
    if not checkpoint["key_ids"]:
        return "checkpoint key_ids must not be empty"
    entry_ids = [entry.get("key_id") for entry in entries]
    if entry_ids != checkpoint["key_ids"]:
        return "checkpoint entries do not match its ordered key_ids"
    if checkpoint["root_key_id"] != checkpoint["key_ids"][0]:
        return "checkpoint root_key_id does not identify its first entry"
    known = set(checkpoint["key_ids"])
    if not set(checkpoint["superseded_key_ids"]).issubset(known) or \
            not set(checkpoint["revoked_key_ids"]).issubset(known):
        return "checkpoint status lists name a key outside key_ids"
    if set(checkpoint["superseded_key_ids"]) & set(checkpoint["revoked_key_ids"]):
        return "checkpoint marks one key both superseded and revoked"
    if checkpoint["superseded_key_ids"] != [
            entry["key_id"] for entry in entries
            if entry.get("status") == "superseded"]:
        return "checkpoint superseded summary disagrees with its entries"
    if checkpoint["revoked_key_ids"] != [
            entry["key_id"] for entry in entries
            if entry.get("status") == "revoked"]:
        return "checkpoint revoked summary disagrees with its entries"
    if not _hex_bytes(checkpoint_public_key_hex, 32):
        return "pinned checkpoint key is not 32 bytes of hex"
    checkpoint_key_id = KeyStore.key_id_of(checkpoint_public_key_hex)
    if checkpoint.get("checkpoint_key_id") != checkpoint_key_id:
        return "checkpoint key_id does not match the pinned checkpoint key"
    if checkpoint_key_id in known:
        return ("checkpoint key identifier overlaps a manifest signing key; "
                "the trust roles require distinct keys")
    checkpoint_key_bytes = bytes.fromhex(checkpoint_public_key_hex)
    for entry in entries:
        entry_public_key = entry.get("public_key")
        if _hex_bytes(entry_public_key, 32) and \
                bytes.fromhex(entry_public_key) == checkpoint_key_bytes:
            return ("checkpoint key material overlaps a manifest signing key; "
                    "the trust roles require distinct keys")
    if not _hex_bytes(checkpoint.get("signature"), 64):
        return "checkpoint signature must be 64 bytes of hex"
    body = {k: v for k, v in checkpoint.items() if k != "signature"}
    try:
        encoded = canonical(body)
    except (TypeError, ValueError):
        return "checkpoint is not canonically encodable"
    if not KeyStore.verify(checkpoint_public_key_hex, encoded,
                           checkpoint["signature"]):
        return "checkpoint signature is not valid under the pinned checkpoint key"

    if manifest is not None:
        try:
            digest = _manifest_digest(manifest)
            entries_match = canonical(checkpoint["entries"]) == \
                canonical(manifest["keys"])
        except (TypeError, ValueError, KeyError):
            return "manifest is not canonically encodable for checkpointing"
        if checkpoint["manifest_digest"] != digest:
            return "checkpoint manifest_digest does not commit this manifest"
        if checkpoint["issuer"] != manifest.get("issuer"):
            return "checkpoint issuer does not match the manifest"
        if checkpoint["root_key_id"] != manifest.get("root_key_id"):
            return "checkpoint root_key_id does not match the manifest"
        if float(checkpoint["manifest_published_at"]) != \
                float(manifest.get("published_at")):
            return "checkpoint published_at does not match the manifest"
        manifest_ids = [str(entry["key_id"]) for entry in manifest["keys"]]
        if checkpoint["key_ids"] != manifest_ids:
            return "checkpoint key_ids do not match the manifest chain"
        if not entries_match:
            return "checkpoint entries do not match the manifest entries"
        if checkpoint["superseded_key_ids"] != _status_ids(manifest, "superseded"):
            return "checkpoint superseded keys do not match the manifest"
        if checkpoint["revoked_key_ids"] != _status_ids(manifest, "revoked"):
            return "checkpoint revoked keys do not match the manifest"
    return None


def _entry_transition_error(previous: dict, current: dict) -> Optional[str]:
    kid = previous.get("key_id")
    for field in ("key_id", "public_key", "not_before", "succession"):
        try:
            unchanged = canonical(previous.get(field)) == canonical(current.get(field))
        except (TypeError, ValueError):
            unchanged = False
        if not unchanged:
            return f"checkpoint rewrote prior key {kid}'s {field}"

    old_status, new_status = previous.get("status"), current.get("status")
    permitted = {
        "active": {"active", "superseded", "revoked"},
        "superseded": {"superseded", "revoked"},
        "revoked": {"revoked"},
    }
    if new_status not in permitted.get(old_status, set()):
        return (f"checkpoint moved prior key {kid} from {old_status!r} back to "
                f"{new_status!r}")

    old_end, new_end = previous.get("not_after"), current.get("not_after")
    if old_end is not None and old_end != new_end:
        return f"checkpoint rewrote prior key {kid}'s not_after"
    if old_end is None and new_end is not None and new_status == "active":
        return f"checkpoint gave active key {kid} a closed validity window"

    old_rev, new_rev = previous.get("revocation"), current.get("revocation")
    if old_rev is not None:
        try:
            same_revocation = canonical(old_rev) == canonical(new_rev)
        except (TypeError, ValueError):
            same_revocation = False
        if not same_revocation:
            return f"checkpoint removed or rewrote prior key {kid}'s revocation"
    elif new_rev is not None and new_status != "revoked":
        return f"checkpoint added revocation evidence without revoking key {kid}"

    known = {"key_id", "public_key", "not_before", "not_after", "status",
             "succession", "revocation"}
    for field, value in previous.items():
        if field in known:
            continue
        if field not in current:
            return f"checkpoint stripped prior key {kid}'s extension {field!r}"
        try:
            same_extension = canonical(value) == canonical(current[field])
        except (TypeError, ValueError):
            same_extension = False
        if not same_extension:
            return f"checkpoint rewrote prior key {kid}'s extension {field!r}"
    return None


def verify_manifest_checkpoint(manifest: dict, checkpoint: dict, *,
                               root_public_key_hex: Optional[str] = None,
                               checkpoint_public_key_hex: Optional[str] = None,
                               trusted_checkpoint: Optional[dict] = None) -> dict:
    """Verify a manifest and its monotonic checkpoint against retained state.

    ``trusted_checkpoint`` is the last signed checkpoint the verifier retained, or
    an initial checkpoint obtained with the checkpoint key out of band. The call
    fails closed without it. On success ``next_checkpoint`` is the exact signed
    object the caller may persist atomically as its new rollback state.

    This proves non-regression relative to retained state. It does not prove that a
    newly bootstrapping client received the globally latest checkpoint, and it does
    not protect a compromised checkpoint signing key.
    """
    def bad(reason: str, failure: str) -> dict:
        return {"ok": False, "reason": reason, "failure": failure,
                "manifest": None, "next_checkpoint": None}

    if root_public_key_hex is None or checkpoint_public_key_hex is None or \
            trusted_checkpoint is None:
        return bad("manifest freshness requires a pinned root, a separately pinned "
                   "checkpoint key, and a previously trusted signed checkpoint",
                   "required")
    verified = verify_key_manifest(
        manifest, root_public_key_hex=root_public_key_hex)
    if not verified["ok"]:
        return bad(f"key manifest is invalid: {verified['reason']}",
                   "manifest-invalid")
    current_error = _checkpoint_error(
        checkpoint, checkpoint_public_key_hex=checkpoint_public_key_hex,
        manifest=manifest)
    if current_error:
        return bad(current_error, "invalid")
    trusted_error = _checkpoint_error(
        trusted_checkpoint,
        checkpoint_public_key_hex=checkpoint_public_key_hex)
    if trusted_error:
        return bad(f"trusted checkpoint state is invalid: {trusted_error}", "invalid")

    if checkpoint["issuer"] != trusted_checkpoint["issuer"] or \
            checkpoint["root_key_id"] != trusted_checkpoint["root_key_id"]:
        return bad("checkpoint changed issuer or root identity", "not-monotonic")
    current_seq, trusted_seq = checkpoint["sequence"], trusted_checkpoint["sequence"]
    if current_seq < trusted_seq:
        return bad(f"checkpoint sequence rolled back from {trusted_seq} to "
                   f"{current_seq}", "not-monotonic")
    if current_seq == trusted_seq:
        try:
            same = canonical(checkpoint) == canonical(trusted_checkpoint)
        except (TypeError, ValueError):
            same = False
        if not same:
            return bad("a different checkpoint was presented at an already trusted "
                       "sequence", "not-monotonic")
    else:
        old_ids = trusted_checkpoint["key_ids"]
        if checkpoint["key_ids"][:len(old_ids)] != old_ids:
            return bad("checkpoint truncated or rewrote the trusted succession "
                       "prefix", "not-monotonic")
        for previous, current in zip(
                trusted_checkpoint["entries"], checkpoint["entries"]):
            transition_error = _entry_transition_error(previous, current)
            if transition_error:
                return bad(transition_error, "not-monotonic")
        old_revoked = set(trusted_checkpoint["revoked_key_ids"])
        if not old_revoked.issubset(set(checkpoint["revoked_key_ids"])):
            return bad("checkpoint stripped a previously trusted revocation",
                       "not-monotonic")
        old_superseded = set(trusted_checkpoint["superseded_key_ids"])
        still_retired = (set(checkpoint["superseded_key_ids"])
                         | set(checkpoint["revoked_key_ids"]))
        if not old_superseded.issubset(still_retired):
            return bad("checkpoint restored a previously retired key to active "
                       "service", "not-monotonic")
        if float(checkpoint["manifest_published_at"]) < \
                float(trusted_checkpoint["manifest_published_at"]):
            return bad("checkpoint moved manifest publication time backwards",
                       "not-monotonic")
        if float(checkpoint["issued_at"]) < float(trusted_checkpoint["issued_at"]):
            return bad("checkpoint issuance time moved backwards", "not-monotonic")

    return {"ok": True, "reason": None, "failure": None,
            "manifest": verified, "next_checkpoint": checkpoint}


def _check_succession(st, prev: dict, kid: str, pub: str,
                      nb: float) -> Optional[str]:
    if not isinstance(st, dict) or st.get("type") != SUCCESSION_TYPE:
        return "no succession statement"
    if st.get("predecessor_key_id") != prev["key_id"]:
        return ("succession names a different predecessor than the previous entry — "
                "the chain must be a single line, not a graph")
    if st.get("key_id") != kid or st.get("public_key") != pub:
        return "succession attests a different key than the entry carries"
    if not _finite_number(st.get("not_before")):
        return "succession has no numeric not_before"
    if float(st["not_before"]) != nb:
        return "succession not_before disagrees with the entry"
    body = {k: v for k, v in st.items() if k != "signature"}
    try:
        encoded = canonical(body)
    except (TypeError, ValueError):
        return "succession statement is not canonically encodable"
    if not KeyStore.verify(prev["public_key"], encoded, st.get("signature", "")):
        return "succession signature is not valid under the PREDECESSOR's key"
    return None


def _check_revocation(rev, kid: str, keys: List[dict], i: int) -> Optional[str]:
    if not isinstance(rev, dict) or rev.get("type") != REVOCATION_TYPE:
        return "revocation is not a revocation statement"
    if rev.get("key_id") != kid:
        return "revocation names a different key"
    if rev.get("reason") not in REVOCATION_REASONS:
        return (f"revocation reason {rev.get('reason')!r} is not one of "
                f"{list(REVOCATION_REASONS)} — an unstated cause cannot be acted on")
    if not _finite_number(rev.get("revoked_at")):
        return "revocation has no numeric revoked_at"

    # WHO may revoke. A successor, or the key itself for a voluntary retirement.
    # Not an arbitrary key: otherwise anyone in the chain could retire anyone else.
    signer_id = rev.get("signer_key_id")
    permitted = {str(keys[j].get("key_id")): keys[j]
                 for j in range(i, len(keys))}          # itself + successors
    signer = permitted.get(signer_id)
    if signer is None:
        return ("revocation is signed by a key that is neither this key nor one of "
                "its successors, so it is not entitled to withdraw it")
    body = {k: v for k, v in rev.items() if k != "signature"}
    try:
        encoded = canonical(body)
    except (TypeError, ValueError):
        return "revocation statement is not canonically encodable"
    if not KeyStore.verify(signer["public_key"], encoded,
                           rev.get("signature", "")):
        return "revocation signature is not valid under its stated signer"
    # A compromised key can sign its own revocation just as well as its holder can,
    # so a self-signed revocation is evidence of intent, not of custody. Accepted,
    # and the caller is told which kind it got.
    return None


def key_status_at(verified: dict, key_id: Optional[str],
                  when: Optional[float]) -> Tuple[str, Optional[dict]]:
    """`(status, entry)` for a key at a moment, given a VERIFIED manifest.

    Statuses returned:
      `valid`             in service at `when`
      `revoked`           withdrawn for cause — never a bare ok, whenever issued
      `outside-validity`  in the chain, but not in service at `when`
      `unknown`           not in the manifest at all

    `revoked` ignores `when` deliberately. If the cause was compromise, the key was
    in an attacker's hands for an unknown period before anyone noticed, so "issued
    before the revocation date" establishes nothing. A verifier that waved those
    through would make revocation advisory, which is the opposite of its purpose.
    """
    if not verified.get("ok"):
        return "unknown", None
    e = (verified.get("keys") or {}).get(str(key_id))
    if e is None:
        return "unknown", None
    if e.get("status") == "revoked":
        return "revoked", e
    if when is None:
        # No issuance time to place the key against. Do not guess in the permissive
        # direction: an envelope with no `created` cannot be shown to be inside a
        # validity window.
        return "outside-validity", e
    w = float(when)
    if w < float(e["not_before"]):
        return "outside-validity", e
    na = e.get("not_after")
    if na is not None and w > float(na):
        return "outside-validity", e
    return "valid", e


def rotate(manifest_keys: List[dict], predecessor: KeyStore, successor: KeyStore, *,
           at: float) -> List[dict]:
    """The ordinary rotation: close the current key's window at `at`, mark it
    SUPERSEDED (not revoked — history must keep verifying), and append the successor
    attested by the predecessor.

    Returns a new entry list; the caller republishes the manifest.
    """
    if not manifest_keys:
        raise ValueError("cannot rotate an empty chain")
    keys = [dict(e) for e in manifest_keys]
    cur = keys[-1]
    if cur["key_id"] != predecessor.key_id():
        raise ValueError("the predecessor is not the chain's current key")
    if cur["status"] != "active":
        raise ValueError(f"the chain's current key is {cur['status']!r}, not active")
    cur["status"] = "superseded"
    cur["not_after"] = float(at)
    st = key_succession_statement(predecessor, successor.public_key_hex(),
                                 not_before=float(at))
    keys.append(manifest_entry(successor.key_id(), successor.public_key_hex(),
                              not_before=float(at), status="active", succession=st))
    return keys
