"""Attestation — the layer that turns the certified pipeline from ASSERTED into
THIRD-PARTY-VERIFIABLE (data-verification foothold, W2b + W3).

Two capabilities, both built on the never-pruned `cert_log` hash chain in
`history.py`:

  W2b — ed25519 SIGNING. A `KeyStore` holds a signing key OUTSIDE the capture
  writer (a gitignored PEM). A Signed Tree Head (STH) binds a scope's whole
  capture history to one signature: once a verifier has witnessed an STH, no
  earlier leaf can be rewritten without either detection (the chain) or an
  invalid signature (the key). tamper-EVIDENCE (the chain) becomes tamper-PROOF
  (the signature).

  W3 — a Merkle TRANSPARENCY LOG. The per-scope cert_log leaves are the leaves of
  an RFC-6962-style Merkle tree (domain-separated leaf/internal hashing prevents
  second-preimage forgery). `inclusion_proof` gives an O(log n) proof that one
  capture is in the signed tree; `verify_inclusion` / `verify_tree_head` need
  ONLY the public key and the proof — no store, no engine. This is the
  Certificate-Transparency / Sigstore-Rekor analogue for computed-data captures.

This module is PURE: it imports no store and no engine — the STH/proof builders
take the leaves (or a store that yields them) as an argument, so the verify path
pulls in only `cryptography` + stdlib. That is what lets an independent third
party run the checks.
"""
from __future__ import annotations

from contextlib import contextmanager
import errno
import hashlib
import json
import os
import struct
import tempfile
import threading
import time
from pathlib import Path
import math
from typing import Dict, List, Optional, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)
from cryptography.exceptions import InvalidSignature


def canonical(obj) -> bytes:
    """Deterministic JSON serialization for signing: sorted keys, compact
    separators, non-ASCII preserved. The same object always signs to the same
    bytes, on any machine.

    The encoding must be INJECTIVE — two different objects must never sign to
    the same bytes, and one object must never have two encodings — or a
    signature over it means less than it appears to. Two defaults of json.dumps
    break that, so both are disabled here:

    - allow_nan would emit bare NaN/Infinity, which is not JSON at all (RFC 8259)
      and cannot be re-parsed interoperably. A non-finite value in a certificate
      is a refusal, never something to sign.
    - non-string keys are coerced to strings AFTER sorting, so {1:'a'} and
      {'1':'a'} collide, and {10:..,9:..} sorts numerically while its string
      form sorts lexically — the same document, two different signed byte
      strings.
    """
    _reject_uncanonical(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _reject_uncanonical(obj, _path: str = "$") -> None:
    """Depth-first check that `obj` has exactly one canonical encoding."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"canonical(): non-string key {k!r} at {_path} — mapping keys "
                    f"must be strings or the encoding is not injective")
            _reject_uncanonical(v, f"{_path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _reject_uncanonical(v, f"{_path}[{i}]")
    elif isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            raise ValueError(
                f"canonical(): non-finite float at {_path} — NaN/Infinity are not "
                f"JSON and must not be signed")


# ── ed25519 key management ────────────────────────────────────────────────────
#: Where a key directory comes from when the caller does not name one.
#:
#: Resolution order, and why: (1) `$ALELYON_KEY_DIR`, so a standalone deployment
#: names its own location without a repo anywhere in the picture — a customer-run
#: witness (PLATFORM.md W2 option ii) installs this module from a wheel and has no
#: `globals/`; (2) the platform's on-disk state home, when this module is running
#: inside the monorepo; (3) refuse, naming what to pass.
#:
#: Step (3) matters. `alelyon.runtime.common.paths` locates the state home by
#: walking up for a `pyproject.toml`, and inside site-packages that walk either
#: finds an unrelated project or falls back to site-packages itself — which would
#: silently write a SIGNING KEY into the Python installation. An explicit refusal is
#: the only safe answer.
_KEY_DIR_ENV = "ALELYON_KEY_DIR"
_KEY_INIT_THREAD_LOCK = threading.RLock()
_KEY_INIT_LOCK_TIMEOUT = 30.0
_WITNESS_STATE_THREAD_LOCK = threading.RLock()
STH_TYPE = "alelyon.sth/v0"


def _default_key_dir() -> str:
    env = os.environ.get(_KEY_DIR_ENV)
    if env:
        return env
    try:
        from alelyon.runtime.common.paths import GLOBALS_DIR
    except ImportError:
        raise RuntimeError(
            f"no key directory given and none can be inferred: this looks like a "
            f"standalone install with no platform state home. Pass "
            f"key_dir=... explicitly, or set ${_KEY_DIR_ENV}. (Refusing to guess "
            f"— a wrong guess writes a signing key somewhere you did not choose.)"
        ) from None
    return str(Path(GLOBALS_DIR) / "dqc_keys")


@contextmanager
def _exclusive_key_init(lock_path: Path):
    """Cross-process lock for the one-time creation of a signing key.

    The lock is owned by the operating-system file descriptor rather than by the
    presence of a lock file. A crashed initializer therefore releases it without
    another process having to guess whether a lock file is stale. The file itself
    may remain and is harmless.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    windows_lock = None
    try:
        try:
            os.chmod(lock_path, 0o600)
        except OSError:
            pass

        if os.name == "nt":
            import msvcrt

            # Lock byte zero. Windows permits a byte-range lock past current EOF,
            # so the coordination file can remain empty and no pre-lock sentinel
            # write introduces a second initialization race.
            deadline = time.monotonic() + _KEY_INIT_LOCK_TIMEOUT
            retryable = {errno.EACCES, errno.EAGAIN, errno.EDEADLK}
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    windows_lock = msvcrt
                    locked = True
                    break
                except OSError as exc:
                    if exc.errno not in retryable:
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out waiting for signing-key initialization lock "
                            f"{lock_path}") from exc
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            locked = True

        yield
    finally:
        if locked:
            try:
                if windows_lock is not None:
                    os.lseek(fd, 0, os.SEEK_SET)
                    windows_lock.locking(fd, windows_lock.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                # Closing the descriptor releases an OS-owned lock even if an
                # explicit unlock fails during interpreter/process teardown.
                pass
        os.close(fd)


def _atomic_private_key_write(path: Path, pem: bytes) -> None:
    """Durably write a complete PEM and atomically publish it at ``path``."""
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(raw_tmp)
    fd_open = True
    try:
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "wb") as fh:
            fd_open = False
            fh.write(pem)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        if fd_open:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


class KeyStore:
    """An ed25519 signing key, generated on first use and persisted to a
    gitignored PEM. `key_id` is a short public-key fingerprint that a verifier
    uses to pin which key it trusts."""

    def __init__(self, key_dir: Optional[str] = None) -> None:
        if key_dir is None:
            key_dir = _default_key_dir()
        self.key_dir = Path(key_dir)
        self._priv: Optional[Ed25519PrivateKey] = None

    @property
    def _path(self) -> Path:
        return self.key_dir / "signing_ed25519.pem"

    @property
    def _lock_path(self) -> Path:
        return self.key_dir / ".signing_ed25519.pem.lock"

    def private_key(self) -> Ed25519PrivateKey:
        if self._priv is not None:
            return self._priv
        # The process lock protects both one KeyStore shared by many threads and
        # many KeyStore instances aimed at the same directory. The file lock then
        # serializes independent worker processes. Every waiter re-checks and
        # loads the persisted winner while holding both locks.
        with _KEY_INIT_THREAD_LOCK:
            if self._priv is not None:
                return self._priv
            self.key_dir.mkdir(parents=True, exist_ok=True)
            with _exclusive_key_init(self._lock_path):
                if not self._path.exists():
                    generated = Ed25519PrivateKey.generate()
                    pem = generated.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption())
                    _atomic_private_key_write(self._path, pem)
                # Load from the published file even in the winning process. This
                # makes the persisted identity—not a transient generated object—
                # the single source of truth returned by every contender.
                self._priv = serialization.load_pem_private_key(
                    self._path.read_bytes(), password=None)
        return self._priv

    def public_key_bytes(self) -> bytes:
        return self.private_key().public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)

    def public_key_hex(self) -> str:
        return self.public_key_bytes().hex()

    def key_id(self) -> str:
        return "ed25519:" + hashlib.blake2b(
            self.public_key_bytes(), digest_size=8).hexdigest()

    def sign(self, msg: bytes) -> bytes:
        return self.private_key().sign(msg)

    @staticmethod
    def key_id_of(public_key_hex: str) -> str:
        return "ed25519:" + hashlib.blake2b(
            bytes.fromhex(public_key_hex), digest_size=8).hexdigest()

    @staticmethod
    def verify(public_key_hex: str, msg: bytes, signature_hex: str) -> bool:
        """True iff `signature_hex` is a valid ed25519 signature of `msg` under
        the public key. Needs only the public key — the third-party check."""
        try:
            pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
            pk.verify(bytes.fromhex(signature_hex), msg)
            return True
        except (InvalidSignature, ValueError, TypeError):
            return False


# ── cert_log leaf hash (PURE — so a verifier can recompute it) ────────────────
def cert_leaf_hash(table_name: str, scope1: str, scope2: str, seq: int,
                   value_digest: str, n: int, lo_ts: float, hi_ts: float,
                   bits: int, payload: str, prev_hash: str) -> str:
    """The cert_log chain link: commits this capture's envelope (its value_digest
    AND its per-column payload — which carries the quantization Δ) plus the prior
    leaf's hash. This lives in the PURE attest layer (history.py delegates to it)
    so a third-party verifier can RE-DERIVE a leaf_hash from a leaf record and thus
    bind the committed Δ to the signed transparency tree — the basis of a
    transparency-anchored width. This proves continuity of the signer's capture-time
    commitments, not that the signer captured truthful values or deltas. Byte-format
    is frozen: any change invalidates every stored chain."""
    h = hashlib.blake2b(digest_size=32)
    parts = [table_name, scope1, scope2, str(int(seq)), value_digest, str(int(n)),
             repr(float(lo_ts)), repr(float(hi_ts)), str(int(bits)), payload,
             prev_hash]
    h.update("\x1f".join(parts).encode("utf-8"))
    return h.hexdigest()


# ── keyed-table canonical form + digest (PURE — the store and the verifier share it) ─
def canonical_table_rows(rows) -> List[Tuple[str, float]]:
    """The canonical form of a keyed table: duplicate keys resolve keep-LAST (the
    store's INSERT OR REPLACE rule), then sort ascending by key.

    Key order is by Unicode code point, which for UTF-8 is byte order — the two
    agree across all of Unicode, so a second implementation cannot pick the wrong
    one. No normalisation is applied: two Unicode spellings of a claim id are
    DIFFERENT keys, because merging them would silently combine rows a
    reconciliation exists to distinguish, and choosing a normal form would make the
    digest depend on a Unicode version. Canonicalising keys belongs to the ingestion
    adapter, upstream of any commitment.
    """
    keep: Dict[str, float] = {}
    for k, v in rows:
        if not isinstance(k, str):
            raise ValueError(f"table row key {k!r} is {type(k).__name__}, not a "
                             f"string — keyed-table ordering is defined over text")
        keep[k] = float("nan") if v is None else float(v)     # keep-LAST
    return sorted(keep.items(), key=lambda kv: kv[0])


def table_rows_digest(rows) -> str:
    """BLAKE2b-256 over canonical keyed-table rows, keys LENGTH-PREFIXED.

    Per row: `<Q` byte length of the UTF-8 key, the key bytes, then `<d` the value
    (NaN packs to a fixed bit pattern and is never compared).

    The length prefix is load-bearing, not tidiness. Concatenating keys without it
    makes the encoding non-injective: rows `("ab",1),("c",2)` and `("a",1),("bc",2)`
    feed the hasher identical bytes, so two different tables would share a digest and
    a commitment to one would be a commitment to the other.

    This lives in the PURE attest layer, like `cert_leaf_hash`, so the capture writer
    and a store-free third-party verifier compute it from ONE definition. Two
    definitions of a digest is two formats.
    """
    h = hashlib.blake2b(digest_size=32)
    for k, v in canonical_table_rows(rows):
        kb = k.encode("utf-8")
        h.update(struct.pack("<Q", len(kb)))
        h.update(kb)
        h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


#: The CLOSED vocabulary of corroboration outcomes, in the PURE layer because a
#: store-free verifier has to classify a carried attempt record the same way the
#: store did. Closed on purpose: the outcome IS the claim, so it must be auditable.
#: A free-text status would let a silent source be filed under something that reads
#: like success. Anything outside this set is a malformed record, not a new outcome.
CORROBORATION_OUTCOMES = frozenset({
    "answered",           # returned data that passed the quality gate
    "unavailable",        # asked, returned nothing
    "quality-rejected",   # returned data the quality gate refused
    "error",              # raised
})

#: The outcomes that count as a source having ANSWERED. Everything else is silence
#: of one kind or another, which is exactly the distinction an issuer would be
#: tempted to blur.
CORROBORATION_ANSWERED = frozenset({"answered"})


def _pack_nullable_f(v) -> bytes:
    """Canonical 8-byte encoding of a nullable float (None -> NaN). Exact bits, so
    a digest over these is reproducible byte-for-byte on any implementation."""
    return struct.pack("<d", float("nan") if v is None else float(v))


def corroboration_digest(attempts) -> str:
    """BLAKE2b over one probe's attempts, sorted by provider.

    `attempts` are `(provider, origin, outcome, value)` tuples. The OUTCOME is
    committed alongside the value, so a recorded `unavailable` cannot be edited into
    an `answered` — and because the row COUNT feeds the leaf too, a deleted attempt
    is detectable rather than merely absent.

    This lives in the pure attest layer, not in the store, for the same reason
    `payload_deltas` does: the issuer and a third-party verifier holding only the
    wheel must apply the byte-identical rule, or W3's provider anchoring would be
    checkable only by the party it is meant to constrain.
    """
    h = hashlib.blake2b(digest_size=32)
    for provider, origin, outcome, value in sorted(attempts, key=lambda x: str(x[0])):
        h.update(f"{provider}\x1f{origin}\x1f{outcome}\x1f".encode("utf-8"))
        h.update(_pack_nullable_f(value))
    return h.hexdigest()


def corroboration_tally(attempts) -> Dict[str, int]:
    """Reduce carried attempts to `{asked, answered, silent}`.

    The one number an issuer has an incentive to overstate is `answered`, so it is
    derived here from the outcome vocabulary rather than read from whatever the
    envelope claims. `silent` is everything asked that did not answer — deliberately
    NOT a third independent count, so the three cannot be made to disagree.
    """
    rows = list(attempts)
    answered = sum(1 for a in rows if str(a[2]) in CORROBORATION_ANSWERED)
    return {"asked": len(rows), "answered": answered, "silent": len(rows) - answered}


def payload_deltas(payload) -> Tuple[Dict[str, float], List[str]]:
    """Parse a capture payload into `({column: Δ}, [columns whose Δ is UNUSABLE])`.

    Both the store and the verifier used to read this as
    `float(entry.get("delta", 0.0))`, at seven separate sites. Δ=0.0 is a
    LEGITIMATE value — `certify.py` defines it as "the column was stored exactly"
    — so defaulting an ABSENT field to 0.0 silently converted *unknown* into
    *claimed exact*, which is the fake zero this program forbids.

    It was also exploitable by the adversary a transparency-anchored width is meant
    to constrain: a signer trying to rewrite its own earlier capture commitments.
    A signer holding the key writes a capture payload with the `delta` field simply
    omitted; the leaf_hash commits that payload, so the chain and every inclusion
    proof stay valid. The omission then yields the SAME 0.0 in the committed
    deltas and in the verifier's independent re-derivation — they agree, and a
    ZERO-width bound verifies as `width_trust="transparency-anchored"`. The
    cycle-5h falsifier only covered a signer SHRINKING Δ, which the anchor caught;
    dropping the field walked past it.

    A Δ that was never written is unknown, and unknown must never read as exact.
    Unusable means absent, null, non-numeric, non-finite, or negative. Callers get
    the malformed column names back so they can say WHICH column failed rather
    than reporting a generic miss.
    """
    good, bad = _parse_payload(payload)
    return {c: d for c, (d, _law) in good.items()}, bad


def payload_laws(payload) -> Dict[str, Optional[str]]:
    """`{column: capture law}` for every column whose Δ is USABLE. `None` means the
    relative-dither law (the member is absent), which is every leaf written before
    capture laws existed.

    A verifier needs this because the Δ=0 plausibility invariant is law-specific:
    under relative dither, Δ=0 means the column was all zero; under exact-cents it
    means the values are whole cents. Reading a Δ without knowing which rule produced
    it is how the fake-zero forgery would come back under a new law.
    """
    good, _ = _parse_payload(payload)
    return {c: law for c, (_d, law) in good.items()}


#: Capture laws this build understands. An UNRECOGNISED law makes the column
#: UNUSABLE — never a permissive default — because the Δ semantics of a law nobody
#: implemented are unknown, not lenient. A signer who could name an unknown law and
#: have it treated as "probably fine" would hold exactly the free pass the fake-zero
#: forgery exploited, one level up. `None` is the relative-dither law.
KNOWN_CAPTURE_LAWS = frozenset({None, "dither-relative/v0", "exact-cents/v0"})

# Exact capture-row membership lives inside the already hash-bound payload rather
# than changing cert_leaf_hash's frozen field layout. The entry intentionally has
# no ``column`` member, so old delta parsers ignore it instead of mistaking it for
# another numerical certificate.
_MEMBERSHIP_ENCODINGS = {
    "bars": "i64-epoch-seconds/v0",
    "series": "f64-epoch-seconds/v0",
}
_MAX_MEMBERSHIP_ROWS = 1_000_000
_MAX_CAPTURE_PAYLOAD_BYTES = 32 * 1024 * 1024
_MAX_CAPTURE_PAYLOAD_DEPTH = 64
_MAX_CAPTURE_PAYLOAD_NODES = 1_100_000
_MAX_CAPTURE_PAYLOAD_CONTAINER_ITEMS = 1_000_000
_MAX_CAPTURE_PAYLOAD_STRING_BYTES = 1024 * 1024
_MAX_CAPTURE_PAYLOAD_INTEGER_DIGITS = 1024


def _reject_payload_constant(value: str):
    raise ValueError(f"non-finite JSON constant in capture payload: {value}")


def _parse_payload_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("capture-payload number is outside finite f64 range")
    return parsed


def _parse_payload_int(value: str) -> int:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > _MAX_CAPTURE_PAYLOAD_INTEGER_DIGITS:
        raise ValueError("capture-payload integer exceeds the resource limit")
    return int(value)


def _unique_payload_object(pairs):
    if len(pairs) > _MAX_CAPTURE_PAYLOAD_CONTAINER_ITEMS:
        raise ValueError("capture-payload object exceeds the resource limit")
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate capture-payload object member: {name}")
        result[name] = value
    return result


def _payload_nesting_within_limit(text: str) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for value in text:
        if in_string:
            if escaped:
                escaped = False
            elif value == "\\":
                escaped = True
            elif value == '"':
                in_string = False
            continue
        if value == '"':
            in_string = True
        elif value in ("[", "{"):
            depth += 1
            if depth > _MAX_CAPTURE_PAYLOAD_DEPTH:
                return False
        elif value in ("]", "}"):
            depth = max(0, depth - 1)
    return True


def _validate_payload_resources(root) -> None:
    nodes = 0
    stack = [root]
    while stack:
        value = stack.pop()
        nodes += 1
        if nodes > _MAX_CAPTURE_PAYLOAD_NODES:
            raise ValueError("capture payload exceeds the node limit")
        if isinstance(value, dict):
            if len(value) > _MAX_CAPTURE_PAYLOAD_CONTAINER_ITEMS:
                raise ValueError("capture-payload object exceeds the resource limit")
            for name, child in value.items():
                if not isinstance(name, str):
                    raise ValueError("capture-payload object key is not a string")
                if len(name.encode("utf-8")) > _MAX_CAPTURE_PAYLOAD_STRING_BYTES:
                    raise ValueError("capture-payload key exceeds the string limit")
                stack.append(child)
        elif isinstance(value, (list, tuple)):
            if len(value) > _MAX_CAPTURE_PAYLOAD_CONTAINER_ITEMS:
                raise ValueError("capture-payload array exceeds the resource limit")
            stack.extend(value)
        elif isinstance(value, str) and \
                len(value.encode("utf-8")) > _MAX_CAPTURE_PAYLOAD_STRING_BYTES:
            raise ValueError("capture-payload string exceeds the resource limit")
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("capture-payload number is non-finite")
        elif isinstance(value, int) and not isinstance(value, bool) and \
                value.bit_length() > 3403:
            raise ValueError("capture-payload integer exceeds the resource limit")


def _strict_payload_json(payload):
    """Parse the hash-bound payload without ambiguous JSON or unbounded shapes."""
    if isinstance(payload, (str, bytes, bytearray)):
        try:
            if isinstance(payload, str):
                text = payload
                raw = text.encode("utf-8")
            else:
                raw = bytes(payload)
                text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError("capture payload is not valid UTF-8") from exc
        if len(raw) > _MAX_CAPTURE_PAYLOAD_BYTES:
            raise ValueError("capture payload exceeds the byte limit")
        if not _payload_nesting_within_limit(text):
            raise ValueError("capture payload exceeds the nesting limit")
        try:
            parsed = json.loads(
                text, parse_constant=_reject_payload_constant,
                parse_float=_parse_payload_float, parse_int=_parse_payload_int,
                object_pairs_hook=_unique_payload_object)
        except (OverflowError, RecursionError, UnicodeError) as exc:
            raise ValueError("capture payload is not bounded valid JSON") from exc
    else:
        parsed = payload
    _validate_payload_resources(parsed)
    return parsed


def _canonical_membership_rows(table_name: str, row_ts) -> tuple:
    """Return sorted, unique timestamps in the table's frozen digest encoding."""
    table = str(table_name)
    if table not in _MEMBERSHIP_ENCODINGS:
        raise ValueError(f"row membership is undefined for table {table!r}")
    values = []
    for raw in row_ts:
        if isinstance(raw, bool):
            raise ValueError("a row timestamp must be numeric, not boolean")
        if table == "bars":
            if not isinstance(raw, int):
                raise ValueError("a bars row timestamp must be an integer")
            value = int(raw)
            if not -(2 ** 63) <= value < 2 ** 63:
                raise ValueError("a bars row timestamp is outside int64")
        else:
            if not isinstance(raw, (int, float)):
                raise ValueError("a series row timestamp must be numeric")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("a series row timestamp must be finite")
        values.append(value)
    if len(values) > _MAX_MEMBERSHIP_ROWS:
        raise ValueError("capture row membership exceeds the resource limit")
    ordered = sorted(values)
    if any(ordered[i] == ordered[i - 1] for i in range(1, len(ordered))):
        raise ValueError("capture row membership contains duplicate timestamps")
    return tuple(ordered)


def payload_with_membership(payload, table_name: str, row_ts) -> str:
    """Append exact timestamp membership to a capture-certificate payload.

    The returned JSON string is what cert_leaf_hash commits. A malformed or
    pre-populated membership entry is refused so an injected CaptureCert-like
    object cannot choose a second, ambiguous row set.
    """
    entries = _strict_payload_json(payload)
    if not isinstance(entries, list):
        raise ValueError("capture certificate payload must be a JSON array")
    if any(isinstance(entry, dict) and "membership" in entry for entry in entries):
        raise ValueError("capture certificate payload already carries membership")
    rows = _canonical_membership_rows(table_name, row_ts)
    out = list(entries)
    out.append({"membership": {
        "encoding": _MEMBERSHIP_ENCODINGS[str(table_name)],
        "rows": list(rows),
    }})
    return json.dumps(out)


def payload_membership(payload, table_name: str) -> Optional[tuple]:
    """Parse exact signed row membership, or return None fail-closed.

    Absence is intentionally not inferred from ``n``/``lo_ts``/``hi_ts``:
    legacy leaves predate exact membership and therefore cover no rows for a
    transparency claim.
    """
    try:
        entries = _strict_payload_json(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(entries, list):
        return None
    found = [entry.get("membership") for entry in entries
             if isinstance(entry, dict) and "membership" in entry]
    if len(found) != 1 or not isinstance(found[0], dict):
        return None
    block = found[0]
    table = str(table_name)
    if table not in _MEMBERSHIP_ENCODINGS or \
            block.get("encoding") != _MEMBERSHIP_ENCODINGS[table]:
        return None
    rows = block.get("rows")
    if not isinstance(rows, list):
        return None
    try:
        canonical_rows = _canonical_membership_rows(table, rows)
    except (TypeError, ValueError):
        return None
    # Reject alternate ordering/number spellings instead of repairing signed
    # bytes into a different semantic row set.
    if list(canonical_rows) != rows:
        return None
    return canonical_rows


def validated_payload_membership(payload, table_name: str, n: int,
                                  lo_ts: float, hi_ts: float) -> Optional[tuple]:
    """Return membership only when it agrees with the leaf's signed summary."""
    rows = payload_membership(payload, table_name)
    if rows is None or not rows or len(rows) != int(n):
        return None
    if float(rows[0]) != float(lo_ts) or float(rows[-1]) != float(hi_ts):
        return None
    return rows


def _parse_payload(payload) -> Tuple[Dict[str, Tuple[float, Optional[str]]],
                                     List[str]]:
    """`({column: (Δ, law)}, [unusable columns])` — the single parse both public
    readers share, so the store and a store-free verifier cannot apply different
    rules to the same bytes."""
    # This parser is also on the store-free public verification path.  Bound
    # attacker-controlled bytes before ``json.loads`` so a transparency leaf
    # cannot bypass the membership parser's resource limit through its delta
    # or law fields.
    try:
        entries = _strict_payload_json(payload)
    except (TypeError, ValueError):
        return {}, []
    if not isinstance(entries, list):
        return {}, []
    good: Dict[str, Tuple[float, Optional[str]]] = {}
    bad: List[str] = []
    for e in entries:
        if not isinstance(e, dict) or "column" not in e:
            continue
        col = str(e["column"]).lower()
        law = e.get("law")
        if law is not None and not isinstance(law, str):
            bad.append(col)
            continue
        if law not in KNOWN_CAPTURE_LAWS:
            bad.append(col)
            continue
        raw = e.get("delta")
        # A Δ must be a JSON NUMBER. certify.py only ever writes one, so accepting
        # a numeric string would widen the type surface of a security-relevant
        # field for no benefit — the same leniency that made the 0.0 default
        # dangerous. `bool` is an `int` in Python, so it is excluded explicitly.
        if not isinstance(raw, (int, float)) or isinstance(raw, bool) \
                or not math.isfinite(float(raw)) or float(raw) < 0.0:
            bad.append(col)
            continue
        # Duplicate entries for one column resolve to the LARGEST Δ, matching
        # every other attribution in this system (max over covering leaves, max
        # into the watermark). Last-wins would let a payload carry the honest Δ
        # for an auditor to read while every reader used a smaller one written
        # after it. The LAW travels with the winning Δ, since it is the rule that
        # produced that particular number.
        d = float(raw)
        if col not in good or d > good[col][0]:
            good[col] = (d, law)
    # A column that was EVER unusable is unusable, even if another entry for it
    # parsed. Otherwise the store (which reads `good`) anchors a column the
    # verifier (which checks `bad`) rejects — the two disagreeing about the same
    # bytes. Fail closed, and fail identically on both sides.
    for col in bad:
        good.pop(col, None)
    return good, sorted(set(bad))


# ── RFC-6962-style Merkle tree (domain-separated) ─────────────────────────────
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def _leaf_node(leaf_hash_hex: str) -> bytes:
    return hashlib.sha256(_LEAF_PREFIX + bytes.fromhex(leaf_hash_hex)).digest()


def _internal_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def merkle_root(leaf_hashes: List[str]) -> Optional[str]:
    """RFC-6962 Merkle root over the leaf hashes (hex), or None for an empty
    tree. Odd nodes are promoted unhashed to the next level (the CT convention),
    and leaf vs internal nodes are domain-separated so no internal node can be
    forged to equal a leaf."""
    if not leaf_hashes:
        return None
    level = [_leaf_node(h) for h in leaf_hashes]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                nxt.append(_internal_node(level[i], level[i + 1]))
            else:
                nxt.append(level[i])           # odd one out: promote unhashed
        level = nxt
    return level[0].hex()


def merkle_proof(leaf_hashes: List[str], index: int) -> List[str]:
    """Audit path for leaf `index`: an ordered list of sibling hashes (hex),
    leaf-to-root. The SIDE of each sibling is NOT stored — it is derived from the
    index at verify time (RFC-6962), so a proof only recomputes the root at its
    true position; replaying it at a different index fails."""
    n = len(leaf_hashes)
    if not (0 <= index < n):
        raise IndexError(f"index {index} out of range for {n} leaves")
    level = [_leaf_node(h) for h in leaf_hashes]
    idx = index
    path: List[str] = []
    while len(level) > 1:
        sib = idx ^ 1
        if sib < len(level):               # a real sibling exists at this level
            path.append(level[sib].hex())
        # else: this node was promoted unhashed — no sibling on the path
        nxt = [(_internal_node(level[i], level[i + 1]) if i + 1 < len(level)
                else level[i]) for i in range(0, len(level), 2)]
        idx //= 2
        level = nxt
    return path


def verify_merkle_path(leaf_hash_hex: str, index: int, tree_size: int,
                       proof: List[str], root_hex: str) -> bool:
    """Recompute the root from a leaf + its audit path and compare. The sibling
    SIDE at each level comes from the index bit, so the proof is bound to
    (index, tree_size): a proof lifted to a different position or tree fails."""
    if not (0 <= index < tree_size) or root_hex is None:
        return False
    node = _leaf_node(leaf_hash_hex)
    idx, size, pi = index, tree_size, 0
    while size > 1:
        if (idx ^ 1) < size:               # this node has a sibling at this level
            if pi >= len(proof):
                return False
            sib = bytes.fromhex(proof[pi]); pi += 1
            node = (_internal_node(node, sib) if idx % 2 == 0
                    else _internal_node(sib, node))
        # else promoted: node carries up unchanged, no proof element consumed
        idx //= 2
        size = (size + 1) // 2
    return pi == len(proof) and node.hex() == root_hex


# ── RFC-6962 consistency proofs (APPEND-ONLY verification) ────────────────────
def _largest_pow2_below(n: int) -> int:
    """Largest power of two strictly less than n (n >= 2)."""
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def consistency_proof(leaf_hashes: List[str], m: int) -> List[str]:
    """RFC-6962 consistency proof that the current tree (size n = len(leaf_hashes))
    is an APPEND-ONLY extension of the tree of its first `m` leaves — the first m
    leaves are unchanged and only new leaves were appended. A short list of subtree
    hashes; verify_consistency recomputes BOTH roots from it. Empty when m == n.
    (merkle_root here is provably the RFC-6962 MTH — checked for all n.)"""
    n = len(leaf_hashes)
    if not (0 < m <= n):
        raise ValueError(f"consistency needs 0 < m <= n, got m={m}, n={n}")
    if m == n:
        return []
    return _subproof(m, leaf_hashes, True)


def _subproof(m: int, leaves: List[str], b: bool) -> List[str]:
    n = len(leaves)
    if m == n:
        return [] if b else [merkle_root(leaves)]
    k = _largest_pow2_below(n)
    if m <= k:
        return _subproof(m, leaves[:k], b) + [merkle_root(leaves[k:])]
    return _subproof(m - k, leaves[k:], False) + [merkle_root(leaves[:k])]


def verify_consistency(m: int, n: int, first_root: Optional[str],
                       second_root: Optional[str], proof: List[str]) -> bool:
    """Verify an RFC-6962 consistency proof: the size-n tree (root second_root)
    contains the size-m tree (root first_root) as an append-only PREFIX. A rewrite
    of any of the first m leaves, or a size rollback, fails. Needs only the two
    roots + the proof — no store."""
    if first_root is None or second_root is None:
        return False
    if m == n:
        return first_root == second_root and not proof
    if not (0 < m < n) or not proof:
        return False
    try:
        path = [bytes.fromhex(p) for p in proof]
        first = bytes.fromhex(first_root)
        second = bytes.fromhex(second_root)
    except (ValueError, TypeError):
        return False
    if _is_pow2(m):
        path = [first] + path
    fn, sn = m - 1, n - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1
    fr = sr = path[0]
    for c in path[1:]:
        if sn == 0:
            return False
        if (fn & 1) or (fn == sn):
            fr = _internal_node(c, fr)
            sr = _internal_node(c, sr)
            while fn and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            sr = _internal_node(sr, c)
        fn >>= 1
        sn >>= 1
    return sn == 0 and fr == first and sr == second


# ── signed tree heads + inclusion proofs over a scope's cert_log ──────────────
def _scope_leaves(store, table: str, scope1: str, scope2: str) -> List[dict]:
    return store.cert_log_leaves(table, scope1, scope2)


def signed_tree_head(store, table: str, scope1: str, scope2: str,
                     keystore: KeyStore, *, now: float) -> Optional[dict]:
    """Build and SIGN a tree head for a scope: {tree_size, root, key_id, ...,
    signature}. None when the scope has no certified captures. `now` is passed
    in (never wall-clock-read here) so the STH is reproducible in tests."""
    leaves = _scope_leaves(store, table, scope1, scope2)
    if not leaves:
        return None
    root = merkle_root([lf["leaf_hash"] for lf in leaves])
    sth = {
        "type": STH_TYPE,
        "table": table, "scope": [scope1, scope2],
        "tree_size": len(leaves), "root": root,
        "head_leaf": leaves[-1]["leaf_hash"],
        "key_id": keystore.key_id(),
        "public_key": keystore.public_key_hex(),
        "timestamp": float(now),
    }
    sth["signature"] = keystore.sign(canonical(sth)).hex()
    return sth


def _is_hex_bytes(value, length: int) -> bool:
    return (isinstance(value, str) and len(value) == 2 * length
            and all(char in "0123456789abcdef" for char in value))


def _is_finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def _tree_head_schema_error(sth: dict) -> Optional[str]:
    """Return the first structural defect in an STH, else ``None``.

    A valid Ed25519 signature authenticates bytes; it does not make those bytes a
    tree head. Every field used as a trust or routing boundary is therefore
    validated before signature verification. Unknown signed members remain
    permitted by the v0 additive-extension rule.
    """
    if not isinstance(sth, dict):
        return "not a signed tree head"
    if sth.get("type") != STH_TYPE:
        return f"tree head type is not {STH_TYPE}"
    if not isinstance(sth.get("table"), str) or not sth["table"]:
        return "tree head table must be a non-empty string"
    scope = sth.get("scope")
    if not (isinstance(scope, list) and len(scope) == 2
            and all(isinstance(part, str) and part for part in scope)):
        return "tree head scope must be an array of two non-empty strings"
    if type(sth.get("tree_size")) is not int or sth["tree_size"] <= 0:
        return "tree head tree_size must be a positive integer"
    if not _is_hex_bytes(sth.get("root"), 32):
        return "tree head root must be 32 bytes of hex"
    if not _is_hex_bytes(sth.get("head_leaf"), 32):
        return "tree head head_leaf must be 32 bytes of hex"
    if not isinstance(sth.get("key_id"), str):
        return "tree head key_id is required"
    if not _is_hex_bytes(sth.get("public_key"), 32):
        return "tree head public_key must be 32 bytes of hex"
    if sth["key_id"] != KeyStore.key_id_of(sth["public_key"]):
        return "tree head key_id does not identify its embedded public_key"
    if not _is_finite_number(sth.get("timestamp")):
        return "tree head timestamp must be a finite number"
    if not _is_hex_bytes(sth.get("signature"), 64):
        return "tree head signature must be 64 bytes of hex"
    # For a one-leaf tree the relationship is locally decidable. For larger
    # trees an inclusion proof, not the bare STH, establishes membership.
    if sth["tree_size"] == 1 and \
            merkle_root([sth["head_leaf"]]) != sth["root"].lower():
        return "single-leaf tree root does not commit the stated head_leaf"
    return None


def verify_tree_head(sth: dict, *, public_key_hex: Optional[str] = None) -> dict:
    """Verify an STH's signature against a TRUSTED (pinned) public key. A pinned
    key is REQUIRED: verifying against the key embedded in the same untrusted STH
    authenticates nothing (an attacker self-signs), so `public_key_hex` is
    mandatory for ok=True. Returns {ok, reason}."""
    schema_error = _tree_head_schema_error(sth)
    if schema_error:
        return {"ok": False, "reason": schema_error}
    if public_key_hex is None:
        return {"ok": False, "reason": "no trusted public key pinned — the "
                "embedded key cannot authenticate the STH"}
    if not _is_hex_bytes(public_key_hex, 32):
        return {"ok": False, "reason": "pinned public key is not 32 bytes of hex"}
    if sth["key_id"] != KeyStore.key_id_of(public_key_hex):
        return {"ok": False, "reason": "key_id does not match the pinned key"}
    if bytes.fromhex(sth["public_key"]) != bytes.fromhex(public_key_hex):
        return {"ok": False, "reason": "embedded public_key does not match the pinned key"}
    body = {k: v for k, v in sth.items() if k != "signature"}
    try:
        encoded = canonical(body)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "tree head is not canonically encodable"}
    if not KeyStore.verify(public_key_hex, encoded, sth["signature"]):
        return {"ok": False, "reason": "invalid signature under the pinned key"}
    return {"ok": True, "reason": None}


def inclusion_proof(store, table: str, scope1: str, scope2: str,
                    seq: int) -> Optional[dict]:
    """A proof that the capture at `seq` is in the scope's current tree:
    {leaf_hash, index, tree_size, proof, root, value_digest}. None if absent."""
    leaves = _scope_leaves(store, table, scope1, scope2)
    if not leaves or not (0 <= seq < len(leaves)):
        return None
    hashes = [lf["leaf_hash"] for lf in leaves]
    return {
        "leaf_hash": leaves[seq]["leaf_hash"],
        "value_digest": leaves[seq]["value_digest"],
        "index": seq, "tree_size": len(leaves),
        "proof": merkle_proof(hashes, seq),
        "root": merkle_root(hashes),
    }


def transparency_bundle(store, table: str, scope1: str, scope2: str,
                        seqs: List[int], keystore: KeyStore, *, now: float) -> Optional[dict]:
    """Bundle {sth, leaves:[full record + inclusion_proof]} for the given capture
    seqs of a scope, so a third party can bind each leaf's committed Δ (in its
    payload) to the SIGNED transparency tree: recompute cert_leaf_hash from the
    record, verify inclusion against the STH root, then trust the payload Δ. This
    is what turns the DRC width from signer-attested into transparency-anchored.
    None if the scope has no log, or any requested seq is missing a record/proof.
    `now` is passed in (never wall-clock-read) so the bundle is reproducible."""
    sth = signed_tree_head(store, table, scope1, scope2, keystore, now=now)
    if sth is None:
        return None
    records = {int(r["seq"]): r for r in store.cert_log_records(table, scope1, scope2)}
    leaves: List[dict] = []
    for seq in sorted({int(s) for s in seqs}):
        rec = records.get(seq)
        if rec is None:
            return None
        proof = inclusion_proof(store, table, scope1, scope2, seq)
        if proof is None or proof.get("leaf_hash") != rec.get("leaf_hash"):
            return None
        leaves.append({
            "seq": seq, "value_digest": rec["value_digest"], "n": rec["n"],
            "lo_ts": rec["lo_ts"], "hi_ts": rec["hi_ts"], "bits": rec["bits"],
            "payload": rec["payload"], "prev_hash": rec["prev_hash"],
            "inclusion_proof": proof,
        })
    return {"sth": sth, "leaves": leaves}


def verify_inclusion(proof: dict, *, expected_root: Optional[str] = None) -> dict:
    """Recompute the root from the leaf + audit path and bind it to a WITNESSED
    root. `expected_root` (the root of a signed tree head you have verified) is
    REQUIRED for a meaningful ok=True: without it the proof only shows a leaf,
    path, and root are internally consistent — which an attacker can fabricate —
    not that the leaf is in the REAL (signed) tree. Returns {ok, reason}; needs
    only the proof, no store."""
    try:
        ok = verify_merkle_path(proof["leaf_hash"], int(proof["index"]),
                                int(proof["tree_size"]),
                                list(proof["proof"]), proof["root"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "reason": "malformed inclusion proof"}
    if not ok:
        return {"ok": False, "reason": "audit path does not recompute the root"}
    if expected_root is None:
        return {"ok": False, "reason": "no witnessed root supplied — proof is only "
                "internally consistent, not bound to a signed tree; pass the STH root"}
    if proof["root"] != expected_root:
        return {"ok": False, "reason": "root does not match the witnessed STH"}
    return {"ok": True, "reason": None}


# ── external anchoring: an independent append-only WITNESS (equivocation guard) ─
class Witness:
    """An append-only-log WITNESS — a co-signer that refuses to vouch
    for a tree head unless it is a consistent, append-only extension of the last
    head it saw for that scope. This is what defeats EQUIVOCATION (the log signer
    showing different histories to different verifiers) and silent REWRITES: a
    client that trusts the witness's key can detect a fork or rewind relative to the
    witness's retained state. Independence is a deployment property, not a property
    of this class: it exists only when a party other than the log signer operates the
    witness. This reference persists its seen-heads locally and enforces monotonic,
    consistency-PROVEN growth."""

    TYPE = "alelyon.cosign/v0"

    def __init__(self, keystore: KeyStore, state_path: Optional[str] = None) -> None:
        self._ks = keystore
        if state_path is None:
            state_path = str(Path(_default_key_dir()) / "witness_state.json")
        self._path = Path(state_path)
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")
        self._seen = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        loaded = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("witness state must be a JSON object")
        return loaded

    def _save(self, seen: dict) -> None:
        """Durably publish state; failure is a refusal, never a co-signature."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_tmp = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=str(self._path.parent),
        )
        tmp = Path(raw_tmp)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(seen, fh, sort_keys=True, separators=(",", ":"))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _scope_key(sth: dict) -> str:
        sc = sth.get("scope") or [None, None]
        return f"{sth.get('table')}\x1f{sc[0]}\x1f{sc[1]}"

    def public_key_hex(self) -> str:
        return self._ks.public_key_hex()

    def key_id(self) -> str:
        return self._ks.key_id()

    def last_seen_size(self, sth: dict) -> Optional[int]:
        """The tree_size this witness last co-signed for the STH's scope, or None
        if it has never seen that scope. The issuer uses this to build the exact
        RFC-6962 consistency proof `cosign` will demand for a growth (the witness
        never trusts a proof it did not check itself)."""
        with _WITNESS_STATE_THREAD_LOCK:
            with _exclusive_key_init(self._lock_path):
                self._seen = self._load()
                prev = self._seen.get(self._scope_key(sth))
                return int(prev["tree_size"]) if prev else None

    def cosign(self, sth: dict, *, consistency: Optional[List[str]] = None,
               now: float) -> Optional[dict]:
        """Co-sign an STH IFF it is an append-only extension of the last head this
        witness saw for its scope. Returns the co-signature statement, or None
        (REFUSED) on a size rollback, a same-size fork (different root), or a growth
        without a valid consistency proof from the prior head. The first head for a
        scope is trusted-on-first-use and recorded."""
        if not isinstance(sth, dict) or "root" not in sth or "tree_size" not in sth:
            return None
        try:
            size, root = int(sth["tree_size"]), str(sth["root"])
        except (TypeError, ValueError):
            return None
        witness_key_id = self._ks.key_id()
        if sth.get("key_id") == witness_key_id:
            return None
        log_public_key = sth.get("public_key")
        if _is_hex_bytes(log_public_key, 32) and bytes.fromhex(log_public_key) == \
                bytes.fromhex(self._ks.public_key_hex()):
            return None
        key = self._scope_key(sth)
        try:
            with _WITNESS_STATE_THREAD_LOCK:
                with _exclusive_key_init(self._lock_path):
                    # Reload inside the cross-process critical section. An
                    # instance's constructor snapshot is never authority.
                    current = self._load()
                    prev = current.get(key)
                    if prev is not None:
                        psize, proot = int(prev["tree_size"]), str(prev["root"])
                        if size < psize:
                            return None                    # rollback
                        if size == psize:
                            if root != proot:
                                return None                # same-size fork
                        elif consistency is None or not verify_consistency(
                                psize, size, proot, root, list(consistency)):
                            return None                    # unproven growth

                    statement = {
                        "type": self.TYPE,
                        "table": sth.get("table"),
                        "scope": sth.get("scope"),
                        "tree_size": size,
                        "root": root,
                        # This binds every signed STH member, including its log
                        # signature, head leaf, timestamp, table, scope, and key.
                        "sth_digest": hashlib.sha256(canonical(sth)).hexdigest(),
                        "log_key_id": sth.get("key_id"),
                        "witness_key_id": witness_key_id,
                        "witness_public_key": self._ks.public_key_hex(),
                        "cosigned_ts": float(now),
                    }
                    statement["signature"] = self._ks.sign(
                        canonical(statement),
                    ).hex()
                    updated = dict(current)
                    updated[key] = {"tree_size": size, "root": root}
                    self._save(updated)
                    self._seen = updated
                    return statement
        except Exception:  # malformed/unwritable state is a hard refusal
            return None


def _is_key_hex(s) -> bool:
    return _is_hex_bytes(s, 32)


def verify_cosignature(statement: dict, *, witness_key_hex: Optional[str] = None,
                       expected_root: Optional[str] = None,
                       expected_sth: Optional[dict] = None) -> dict:
    """Verify a witness co-signature against a PINNED witness key distinct from the
    log key. ``ok=True`` means that key signed this exact complete tree head. It does
    not establish that an independent party controls the key; that is a deployment
    fact. A co-signature checked against its own embedded key vouches for nothing,
    so a pinned key is required.

    Both `expected_root` and the complete `expected_sth` are REQUIRED for ok=True.
    A root-only check cannot establish which timestamp, scope, log key, signature,
    or extension members the witness saw, so it is an explicitly partial check and
    fails closed. ``sth_digest`` binds every member of the complete signed tree head."""
    if not isinstance(statement, dict) or statement.get("type") != Witness.TYPE:
        return {"ok": False, "reason": "not a witness co-signature"}
    if witness_key_hex is None:
        return {"ok": False, "reason": "no pinned witness key — a co-signature "
                "verified against its own embedded key vouches for nothing"}
    if not _is_key_hex(witness_key_hex):
        return {"ok": False, "reason": "pinned witness key is not 32 bytes of hex"}
    if expected_root is None:
        return {"ok": False, "reason": "no tree head supplied — a co-signature not "
                "bound to a root vouches for nothing"}
    if not _is_hex_bytes(expected_root, 32):
        return {"ok": False, "reason": "expected tree root is not 32 bytes of hex"}
    if expected_sth is None:
        return {"ok": False, "reason": "no complete tree head supplied — a root-only "
                "check cannot bind the co-signature to the STH metadata or signature"}
    if not isinstance(expected_sth, dict):
        return {"ok": False, "reason": "expected tree head is not an object"}
    witness_key_id = KeyStore.key_id_of(witness_key_hex)
    if statement.get("log_key_id") == witness_key_id or \
            expected_sth.get("key_id") == witness_key_id:
        return {"ok": False, "reason": "witness and log key identifiers overlap; "
                "co-signing requires distinct cryptographic key roles"}
    expected_log_public_key = expected_sth.get("public_key")
    if not _is_hex_bytes(expected_log_public_key, 32):
        return {"ok": False, "reason": "expected tree head log public key is not "
                "32 bytes of lowercase hex"}
    if bytes.fromhex(expected_log_public_key) == bytes.fromhex(witness_key_hex):
        return {"ok": False, "reason": "witness and log key material overlaps; "
                "co-signing requires distinct cryptographic key roles"}
    if not isinstance(statement.get("table"), str) or not statement["table"]:
        return {"ok": False, "reason": "co-signature table must be a non-empty string"}
    scope = statement.get("scope")
    if not (isinstance(scope, list) and len(scope) == 2
            and all(isinstance(part, str) and part for part in scope)):
        return {"ok": False, "reason": "co-signature scope must be an array of two non-empty strings"}
    if type(statement.get("tree_size")) is not int or statement["tree_size"] <= 0:
        return {"ok": False, "reason": "co-signature tree_size must be a positive integer"}
    if not _is_hex_bytes(statement.get("root"), 32):
        return {"ok": False, "reason": "co-signature root must be 32 bytes of hex"}
    if not _is_hex_bytes(statement.get("sth_digest"), 32):
        return {"ok": False, "reason": "co-signature sth_digest must be 32 bytes of hex"}
    if not isinstance(statement.get("log_key_id"), str):
        return {"ok": False, "reason": "co-signature log_key_id is required"}
    if statement.get("witness_key_id") != witness_key_id:
        return {"ok": False, "reason": "witness key_id does not match the pinned key"}
    if not _is_hex_bytes(statement.get("witness_public_key"), 32) or \
            bytes.fromhex(statement["witness_public_key"]) != bytes.fromhex(witness_key_hex):
        return {"ok": False, "reason": "embedded witness_public_key does not match the pinned key"}
    if not _is_finite_number(statement.get("cosigned_ts")):
        return {"ok": False, "reason": "co-signature timestamp must be a finite number"}
    if not _is_hex_bytes(statement.get("signature"), 64):
        return {"ok": False, "reason": "co-signature signature must be 64 bytes of hex"}
    body = {k: v for k, v in statement.items() if k != "signature"}
    try:
        encoded = canonical(body)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "co-signature is not canonically encodable"}
    if not KeyStore.verify(witness_key_hex, encoded, statement["signature"]):
        return {"ok": False, "reason": "invalid witness signature under the pinned key"}
    if statement.get("root") != expected_root:
        return {"ok": False, "reason": "co-signed root does not match the tree head"}
    try:
        expected_digest = hashlib.sha256(canonical(expected_sth)).hexdigest()
    except (TypeError, ValueError):
        return {"ok": False, "reason": "expected tree head is not canonically encodable"}
    if statement["sth_digest"] != expected_digest:
        return {"ok": False, "reason": "co-signature vouches for a different complete tree head"}
    if statement["tree_size"] != expected_sth.get("tree_size"):
        return {"ok": False, "reason": "co-signed tree_size does not match the head"}
    if statement.get("log_key_id") != expected_sth.get("key_id"):
        return {"ok": False, "reason": "co-signature vouches for a different log key"}
    if statement.get("table") != expected_sth.get("table"):
        return {"ok": False, "reason": "co-signature vouches for a different table"}
    if statement.get("scope") != expected_sth.get("scope"):
        return {"ok": False, "reason": "co-signature vouches for a different scope"}
    return {"ok": True, "reason": None}


def verify_witnessed_head(sth: dict, cosignature: dict, *,
                          public_key_hex: Optional[str] = None,
                          witness_key_hex: Optional[str] = None) -> dict:
    """Check that the LOG signed this complete STH under the pinned log key and the
    pinned witness key co-signed that same complete STH. Independence and stronger
    equivocation-resistance depend on who operates the witness and preserves its
    state; this function cannot establish either deployment fact. Returns
    ``{ok, log, witness}``."""
    log = verify_tree_head(sth, public_key_hex=public_key_hex)
    root = sth.get("root") if isinstance(sth, dict) else None
    wit = verify_cosignature(cosignature, witness_key_hex=witness_key_hex,
                             expected_root=root, expected_sth=sth)
    return {"ok": bool(log["ok"] and wit["ok"]), "log": log, "witness": wit}
