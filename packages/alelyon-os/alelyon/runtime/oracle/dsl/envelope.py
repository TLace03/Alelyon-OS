"""The Certified Number Envelope (CNE) — the portable, signed artifact that makes
a computed number VERIFIABLE BY REPLAY (data-verification foothold, W3).

A CNE binds, in one signed JSON object: the DSL program, the computed scalar, the
storage-quantization error budget (the DRC certificate), and a COMMITMENT to each
input it consumed (a digest of the exact series + the per-row capture deltas + the
resample seed). Given the CNE, the public key, and its own copy of the input data,
a third party re-derives the number AND the bound on the deterministic Rust kernel
and confirms every field — trusting only the key and its own data, never the
engine that produced it (see `verify.py`).

The envelope carries ONLY the quantization term today; the sampling / provider /
model terms of the total error budget are named slots reserved for W4, never
fabricated. A refusal is a first-class, signable outcome — "we honestly could not
bound this" is itself an attestation.
"""
from __future__ import annotations

import hashlib
import math
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from alelyon.runtime.oracle.dsl.execcert import (FetchedSeries, StoreCertifiedFetcher,
                                                 certified_run, data_refs)

ENVELOPE_TYPE = "alelyon.cne/v0"


# ── ONE canonical series form for both the digest and the replay ─────────────
def canonical_series(series: pd.Series) -> pd.Series:
    """The single canonical form the digest AND the replay both consume: unique
    timestamps (dedup keep-LAST, matching the store's INSERT OR REPLACE) sorted
    ascending. This removes the digest/replay order mismatch and the duplicate-ts
    / NaN nondeterminism the red-team found — the digest sorts, so replay must
    consume the same order, and a value comparison (which NaN breaks) is never
    needed once ts are unique."""
    if series.index.has_duplicates:
        series = series[~series.index.duplicated(keep="last")]
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def input_digest(series: pd.Series) -> str:
    """BLAKE2b over the CANONICAL series' (epoch_ts, value) bytes: unique ts,
    ascending, so ordering is total and NaN-safe (NaN packs to fixed bits, never
    compared). The verifier recomputes this from its own copy — a mismatch means
    different data. Binds every ts and value; duplicate-ts value swaps are
    resolved the same way the store resolves them (last-wins)."""
    s = canonical_series(series)
    ts = [float(pd.Timestamp(i).timestamp()) for i in s.index]
    vals = s.to_numpy(dtype=float)
    h = hashlib.blake2b(digest_size=32)
    for t, v in zip(ts, vals):
        h.update(struct.pack("<d", t))
        h.update(struct.pack("<d", float(v)))   # NaN → fixed bit pattern
    return h.hexdigest()


#: The keyed-table input kind (insurance reconciliation: claim_id, policy_id,
#: accident-year × development-age cells). Its rows are keyed by TEXT, not stamped
#: with a time, which is why it needs its own digest layout and its own coverage
#: rule (see `_anchor_table_input`).
TABLE_KIND = "table"

#: A row key long enough to be a denial-of-service and short enough to be no real
#: key. Bounded before any allocation, like the delta-block row count.
_MAX_ROW_KEY_BYTES = 4096


def _row_key_bytes(k) -> bytes:
    """A row key's canonical bytes: UTF-8, no normalisation applied.

    NO normalisation is deliberate and must be stated wherever this is documented:
    two Unicode spellings of the same-looking claim id are DIFFERENT keys here.
    Silently normalising would merge rows a reconciliation is trying to tell apart,
    and picking a normal form would make the digest depend on a Unicode version.
    Canonicalising keys is the ingestion adapter's job, upstream of the commitment.
    """
    if not isinstance(k, str):
        raise MalformedEnvelope(
            f"table row key {k!r} is {type(k).__name__}, not a string — a keyed "
            f"table's ordering and digest are defined over text keys only")
    kb = k.encode("utf-8")
    if not kb:
        raise MalformedEnvelope("table row key is empty")
    if len(kb) > _MAX_ROW_KEY_BYTES:
        raise MalformedEnvelope(
            f"table row key is {len(kb)} bytes, over the {_MAX_ROW_KEY_BYTES} limit")
    return kb


def table_digest(series: pd.Series) -> str:
    """BLAKE2b over a CANONICAL keyed table's (key, value) bytes, with every key
    LENGTH-PREFIXED.

    The length prefix is the whole reason this is a separate function rather than a
    string concatenation. Without it, the rows `("ab", 1), ("c", 2)` and
    `("a", 1), ("bc", 2)` feed the hasher identical bytes, so two different tables
    would share a digest and a commitment to one would be a commitment to the other.
    With an 8-byte little-endian length before each key, the encoding is injective:
    every row contributes `8 + len(key) + 8` bytes and the framing is unambiguous.

    Canonical form is the same discipline as a time series (`canonical_series`):
    duplicate keys resolve keep-LAST, then sort ascending. String ordering is by
    Unicode code point, which for UTF-8 is byte order — the two agree for all of
    Unicode, so there is no ambiguity for a second implementation to get wrong.

    The byte layout itself lives in `attest.table_rows_digest`, which the CAPTURE
    writer also calls. One definition on purpose: the table kind's anchoring rule is
    equality between the input digest and a capture batch's `value_digest`, so if the
    two sides computed it separately they could disagree and the equality would mean
    nothing.
    """
    from alelyon.runtime.atlas.data.attest import table_rows_digest
    s = canonical_series(series)
    rows = []
    for k, v in zip(s.index, s.to_numpy(dtype=float)):
        _row_key_bytes(k)          # validate type/length, fail closed before hashing
        rows.append((k, float(v)))
    return table_rows_digest(rows)


def commitment_digest(kind: str, series: pd.Series) -> str:
    """The digest for an input of the given kind. Dispatch is EXPLICIT and an
    unknown kind raises rather than defaulting: an input whose commitment rule we
    cannot name is not something to hash with whichever layout happens to be
    first."""
    if kind in ("price", "series"):
        return input_digest(series)
    if kind == TABLE_KIND:
        return table_digest(series)
    raise MalformedEnvelope(
        f"no commitment layout defined for input kind {kind!r}")


def _runs(d: np.ndarray):
    """Run-length encode a delta array as [[value_or_None, count], ...] (None =
    uncertified). Batch-derived per-row deltas are piecewise-constant, so this is
    O(#batches), not O(#rows) — the difference between a 1 KB and a multi-MB
    envelope at enterprise scale."""
    out = []
    for x in d:
        key = None if not np.isfinite(x) else float(x)
        if out and out[-1][0] == key:
            out[-1][1] += 1
        else:
            out.append([key, 1])
    return out


def _compress_deltas(d: np.ndarray) -> dict:
    """Compact per-row delta encoding. Pure-uniform Δ → a constant (the common,
    minimal case); piecewise-constant (multi-batch spans, uncertified runs) →
    run-length; a truly per-row-varying array → the full list. None marks an
    uncertified (no-capture-bound) row."""
    d = np.asarray(d, dtype=float)
    finite = d[np.isfinite(d)]
    if finite.size == len(d) and finite.size and bool(np.all(finite == finite[0])):
        return {"const": float(finite[0]), "n": int(len(d)), "uncertified_idx": []}
    runs = _runs(d)
    if len(runs) * 2 <= len(d):
        return {"runs": runs, "n": int(len(d))}
    return {"list": [None if not np.isfinite(x) else float(x) for x in d]}


#: Refuse absurd row counts before allocating. Capture scopes are daily bars and
#: macro series; nothing legitimate approaches this.
_MAX_DELTA_ROWS = 10_000_000


class MalformedEnvelope(ValueError):
    """A commitment block in an untrusted envelope is structurally invalid.
    Raised rather than decoded, so verification fails closed instead of
    proceeding over a partly-decoded array."""


def _delta_n(c: dict) -> int:
    if "n" not in c or isinstance(c["n"], bool) or not isinstance(c["n"], int):
        raise MalformedEnvelope("delta block n is not a JSON integer")
    n = c["n"]
    if not 0 <= n <= _MAX_DELTA_ROWS:
        raise MalformedEnvelope(f"delta block declares an implausible length {n}")
    return n


def _decompress_deltas(c: dict) -> np.ndarray:
    """Decode a committed per-row Δ block from an UNTRUSTED envelope.

    Every branch is validated before use. The run-length branch in particular
    must never allocate with np.empty: a run list that does not sum to `n` would
    otherwise leave uninitialised heap memory in the array, which both makes
    verification nondeterministic and lets stray zeros shrink the bound.
    """
    if not isinstance(c, dict):
        raise MalformedEnvelope("delta commitment is not an object")

    branches = [name for name in ("const", "runs", "list") if name in c]
    if len(branches) != 1:
        raise MalformedEnvelope(
            "delta commitment must carry exactly one const/runs/list branch")

    def _value(raw):
        if raw is None:
            return np.nan
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise MalformedEnvelope("delta is not a JSON number or null")
        try:
            value = float(raw)
        except (OverflowError, ValueError):
            raise MalformedEnvelope("delta is not a finite JSON number") from None
        if not math.isfinite(value) or value < 0.0:
            raise MalformedEnvelope("delta is non-finite or negative")
        return value

    if "const" in c:
        n = _delta_n(c)
        value = _value(c["const"])
        if not math.isfinite(value):
            raise MalformedEnvelope("const delta must be finite")
        d = np.full(n, value, dtype=float)
        if "uncertified_idx" not in c:
            raise MalformedEnvelope("const delta omits uncertified_idx")
        idx = c["uncertified_idx"]
        if not isinstance(idx, (list, tuple)):
            raise MalformedEnvelope("uncertified_idx is not a list")
        for i in idx:
            if isinstance(i, bool) or not isinstance(i, int):
                raise MalformedEnvelope("uncertified_idx contains a non-integer")
            if not 0 <= i < n:
                raise MalformedEnvelope(
                    f"uncertified_idx {i} out of range for {n} rows")
            d[i] = np.nan
        return d

    if "runs" in c:
        n = _delta_n(c)
        runs = c["runs"]
        if not isinstance(runs, (list, tuple)):
            raise MalformedEnvelope("runs is not a list")
        total = 0
        for run in runs:
            if not isinstance(run, (list, tuple)) or len(run) != 2:
                raise MalformedEnvelope("run entry is not a [value, count] pair")
            cnt = run[1]
            if isinstance(cnt, bool) or not isinstance(cnt, int):
                raise MalformedEnvelope("run count is not a JSON integer")
            if cnt < 0:
                raise MalformedEnvelope("run count is negative")
            total += cnt
        if total != n:
            raise MalformedEnvelope(
                f"runs cover {total} rows but the block declares {n}")
        # np.full, not np.empty: a validated block fills completely, and an
        # unvalidated one would surface as uncertified rather than as garbage.
        out = np.full(n, np.nan, dtype=float)
        pos = 0
        for v, cnt in runs:
            out[pos:pos + cnt] = _value(v)
            pos += cnt
        return out

    if "list" in c:
        lst = c["list"]
        if not isinstance(lst, (list, tuple)):
            raise MalformedEnvelope("list is not a list")
        if len(lst) > _MAX_DELTA_ROWS:
            raise MalformedEnvelope("delta list is implausibly long")
        if "n" in c and _delta_n(c) != len(lst):
            raise MalformedEnvelope(
                f"list has {len(lst)} rows but the block declares {c['n']}")
        return np.array([_value(x) for x in lst], dtype=float)

    raise MalformedEnvelope("delta commitment has no const/runs/list branch")


class _DictFetcher:
    """A fetcher over pre-resolved FetchedSeries (so the committed inputs are
    EXACTLY the certified inputs, and the verifier can replay from data alone)."""

    def __init__(self, table: Dict[Tuple[str, str], FetchedSeries]) -> None:
        self._t = table

    def get(self, kind: str, key: str) -> FetchedSeries:
        return self._t[(kind, key)]


def _program_hash(src: str) -> str:
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def _scope_of(ref: Tuple[str, str]) -> Tuple[str, str, str, str]:
    """(table, scope1, scope2, column) for a data ref — the SAME scoping the
    StoreCertifiedFetcher / capture writer use, so the cert_log lookups line up.

    Dispatch is exhaustive and an unknown kind RAISES. It used to fall through to
    the FRED series scope for anything that was not `price`, which was a latent
    scope-confusion bug of exactly the class `_verify_one_anchor` guards against: a
    new input kind would silently claim the scope of an unrelated capture, and be
    checked against whatever Δ lived there. A kind we cannot scope is a kind we
    cannot anchor.
    """
    kind, key = ref
    if kind == "price":
        return "bars", str(key).upper(), "1d", "close"
    if kind == "series":
        return "series", "fred", str(key).upper(), "value"
    if kind == TABLE_KIND:
        # A table ref's key is "<dataset>|<column>": one capture scope per
        # (dataset, column), which is also the granularity Δ is certified at.
        dataset, sep, column = str(key).partition("|")
        if not sep or not dataset or not column:
            raise MalformedEnvelope(
                f"table input key {key!r} must be '<dataset>|<column>'")
        return TABLE_KIND, dataset.upper(), column.lower(), column.lower()
    raise MalformedEnvelope(f"no capture scope defined for input kind {kind!r}")


def _witness_cosign(witness, store, table: str, s1: str, s2: str, sth: dict,
                    now: float) -> Optional[dict]:
    """Ask a witness to co-sign this scope's STH, supplying the exact
    RFC-6962 consistency proof the witness demands for a growth beyond the head it
    last saw (the witness never trusts a proof it did not re-check). Returns the
    co-signature statement, or None if the witness refuses (a fork / rollback /
    unproven growth it will not vouch for) — co-signing is best-effort and never
    blocks issuing the certificate itself. The equivocation guarantee is only real
    when a party other than the signer operates the witness; a co-located witness
    demonstrates the wire format but not organizational independence. The log and
    witness keys must still be cryptographically distinct."""
    from alelyon.runtime.atlas.data.attest import consistency_proof
    try:
        hashes = [lf["leaf_hash"] for lf in store.cert_log_leaves(table, s1, s2)]
        prev = witness.last_seen_size(sth)
        cons = (consistency_proof(hashes, prev)
                if prev is not None and 0 < prev < len(hashes) else None)
        return witness.cosign(sth, consistency=cons, now=now)
    except Exception:  # noqa: BLE001 — best-effort; a refusal/absence is honest
        return None


def _matching_table_leaves(store, table: str, s1: str, s2: str, col: str,
                           digest: str):
    """(Δ, law) for the capture leaves of a table scope whose `value_digest` equals
    `digest`, or None when no usable leaf matches.

    This is the table kind's COVERAGE RULE, and it replaces the timestamped
    interval test rather than approximating it. A cert_log leaf spans
    `[lo_ts, hi_ts]`, and a keyed table has no time axis to span — so the options
    were to invent an ordinal for each row (attacker-influenced, since the writer
    chooses the row set) or to require exact identity between the input and a
    captured batch. Identity is strictly stronger: the batch digest and the input
    digest are the SAME function over the same canonical form, so equality means
    "this is that batch", decided on the verifier's own copy of the data.

    The cost, which must be stated rather than discovered: an input assembled from
    SEVERAL capture batches cannot anchor, and its width degrades honestly to
    signer-authenticated. For a quarterly extract — one file, one batch — that is
    the normal case, not a limitation.

    Δ is the MAX over matching leaves and the law travels with the winner, matching
    every other attribution here. A matching leaf whose column is UNUSABLE fails the
    whole thing closed: a leaf that commits no readable Δ for this column cannot be
    the one that vouches for it.
    """
    from alelyon.runtime.atlas.data.attest import payload_deltas, payload_laws
    best, best_law, seqs = None, None, []
    for rec in store.cert_log_records(table, s1, s2):
        if str(rec.get("value_digest")) != digest:
            continue
        cols, unusable = payload_deltas(rec["payload"])
        if col in unusable or col not in cols:
            return None
        d = float(cols[col])
        seqs.append(int(rec["seq"]))
        if best is None or d > best:
            best, best_law = d, payload_laws(rec["payload"]).get(col)
    if best is None:
        return None
    return best, best_law, sorted(set(seqs))


def _anchor_table_input(store, keystore, ref: Tuple[str, str], series: pd.Series,
                        now: float, witness=None) -> Optional[dict]:
    """Anchor a keyed-table input by digest identity against a signed capture leaf."""
    table, s1, s2, col = _scope_of(ref)
    s = canonical_series(series)
    found = _matching_table_leaves(store, table, s1, s2, col, table_digest(s))
    if found is None:
        return None
    delta, law, seqs = found
    from alelyon.runtime.atlas.data.attest import transparency_bundle
    bundle = transparency_bundle(store, table, s1, s2, seqs, keystore, now=now)
    if bundle is None:
        return None
    block = {"table": table, "scope": [s1, s2], "column": col, **bundle}
    if witness is not None:
        cosig = _witness_cosign(witness, store, table, s1, s2, bundle["sth"], now)
        if cosig is not None:
            block["cosignature"] = cosig
    return {"deltas": np.full(len(s), float(delta)), "canonical": s,
            "block": block, "law": law}


def _anchor_input(store, keystore, ref: Tuple[str, str], series: pd.Series,
                  now: float, witness=None) -> Optional[dict]:
    """Attribute the input's per-row Δ from the NEVER-PRUNED, SIGNED cert_log
    through exact current timestamp-to-leaf membership
    and package the covering leaves + inclusion proofs + STH. Returns
    {"deltas", "block"} only when EVERY consumed row is covered by a signed
    capture leaf (fully anchorable) — otherwise None, and the width stays
    signer-authenticated (honest, never fabricated as trustless). When a `witness`
    is supplied its co-signature over the STH is attached to the block, so a client
    that pins the witness key can confirm the log was not forked or rewound.

    A keyed table has no time axis, so it takes the digest-identity path instead of
    the interval-covering one (`_anchor_table_input`)."""
    if ref[0] == TABLE_KIND:
        return _anchor_table_input(store, keystore, ref, series, now, witness)
    table, s1, s2, col = _scope_of(ref)
    leaves = store.cert_log_batches(table, s1, s2)
    if not leaves:
        return None
    s = canonical_series(series)                 # the form the digest commits
    ts = np.array([float(pd.Timestamp(i).timestamp()) for i in s.index])
    deltas = np.full(len(ts), np.nan)
    covering: set = set()
    attributed = {}
    for lf in leaves:
        if col not in lf["columns"]:
            continue
        delta = lf["columns"][col]
        for member_ts in lf.get("rows", ()):
            prior = attributed.get(member_ts)
            if prior is None or delta > prior[0]:
                attributed[member_ts] = (delta, {int(lf["seq"])})
            elif delta == prior[0]:
                prior[1].add(int(lf["seq"]))
    for i, t in enumerate(ts):
        found = attributed.get(t)
        if found is not None:
            deltas[i] = found[0]
            covering.update(found[1])
    if bool(np.isnan(deltas).any()):
        return None                              # a row no signed leaf covers
    from alelyon.runtime.atlas.data.attest import transparency_bundle
    bundle = transparency_bundle(store, table, s1, s2, sorted(covering),
                                 keystore, now=now)
    if bundle is None:
        return None
    block = {"table": table, "scope": [s1, s2], "column": col, **bundle}
    if witness is not None:
        cosig = _witness_cosign(witness, store, table, s1, s2, bundle["sth"], now)
        if cosig is not None:
            block["cosignature"] = cosig
    return {"deltas": deltas, "canonical": s, "block": block,
            "law": _law_of_block(block, col)}


def _law_of_block(block: dict, column: str) -> Optional[str]:
    """The capture law attributed to `column` across a transparency block's leaves:
    the law of the leaf carrying the LARGEST Δ, matching how the Δ itself is
    attributed. Returns None when no leaf names one (the relative-dither law).

    Read from the leaves rather than carried as its own envelope member on purpose.
    A member the SIGNER writes would be a law id the adversary chooses freely; the
    leaf payload is committed in `cert_leaf_hash` and proven into the signed tree, so
    the law is bound to capture the same way the Δ is.
    """
    from alelyon.runtime.atlas.data.attest import payload_deltas, payload_laws
    col = str(column).lower()
    best, best_law = None, None
    for lf in block.get("leaves", []):
        cols, unusable = payload_deltas(lf.get("payload"))
        if col in unusable or col not in cols:
            continue
        d = float(cols[col])
        if best is None or d > best:
            best, best_law = d, payload_laws(lf.get("payload")).get(col)
    return best_law


def _provider_evidence(refs, data_service) -> dict:
    """The certificate's statement about cross-source evidence for its inputs.

    Always an object with a `status`, never null — a null slot beside a filled one
    is read as "checked, fine", which is the one thing this must never say by
    omission. `trust` is always "signer-attested": it describes the ISSUER'S
    DEPLOYMENT (which upstreams are configured), and no verifier can confirm that
    from outside — labeled the way `width_trust` separates authenticated from
    transparency-anchored, so an attested claim never wears a verified one's
    clothes.

    Imported lazily: `alelyon.verify` must stay extractable, and budget.py pulls in
    the bootstrap machinery an offline verifier has no use for.
    """
    try:
        from alelyon.runtime.oracle.dsl.budget import provider_status
        st = provider_status(list(refs), data_service)
    except Exception as exc:  # noqa: BLE001 — evidence is never fatal to issuance
        return {"status": "unmeasured", "trust": "signer-attested",
                "reason": f"origin catalogue unavailable: {type(exc).__name__}"}

    # DEDUPLICATE before signing. Every input of the same kind carries an identical
    # provider/adjustment structure, so the raw catalogue repeated it per input —
    # 13 KB of a 57 KB envelope at 20 inputs, for one diagnostic. The same reason
    # deltas are run-length encoded: envelopes must stay KB-scale. Nothing is lost;
    # the shared structure is emitted once and each input keeps what differs.
    cat = st.pop("catalogue", {}) or {}
    origins, per_input = {}, {}
    for ref, c in cat.items():
        for o, provs in (c.get("providers") or {}).items():
            origins.setdefault(o, {"providers": sorted(provs),
                                   "adjustment": (c.get("adjustments") or {}).get(o)})
        per_input[ref] = {"origin_ids": list(c.get("origin_ids") or []),
                          "comparable": bool(c.get("comparable"))}
    st["origins"] = origins
    st["inputs"] = per_input
    return st


def _provider_anchors(refs, data_service, keystore, *, now: float) -> dict:
    """Carry each input's corroboration probes into the envelope, provable (W3).

    Until this existed, the provider block was `trust: "signer-attested"` and there
    was nothing a reader could do about it: `verify_corroboration` detects a deleted
    silence or an edited outcome, but the ISSUER runs it on its own books. Attaching
    the probe scope's STH, each leaf's inclusion proof, and the attempt records lets
    a third party re-derive the digest and check the summary against leaves proven
    to be in the signed tree.

    The machinery is scope-generic since cycle 5k, so `transparency_bundle` needs no
    change — the corroboration table is just another scope with its own chain.

    Returns `{}` (never a partial or a placeholder) when the store, the key, or the
    probes are missing: an absent anchor leaves the slot `signer-attested`, which is
    honest, whereas a half-filled one would read as checked.
    """
    if keystore is None or data_service is None:
        return {}
    store = getattr(data_service, "store", None)
    if store is None or not getattr(store, "ok", False):
        return {}
    from alelyon.runtime.atlas.data.attest import transparency_bundle

    anchors = {}
    for ref in refs:
        # `data_refs` yields (kind, key) TUPLES, not "kind|key" strings. Parsing
        # `str(ref)` produced "('price', 'SYN')" and silently scoped every anchor to
        # a ticker that does not exist, so nothing anchored and nothing complained —
        # caught only because the vector generator's goldens came back unanchored.
        if isinstance(ref, (tuple, list)) and len(ref) == 2:
            kind, key = str(ref[0]), str(ref[1])
        else:
            kind, _, key = str(ref).partition("|")
            if not key:
                kind, key = "price", str(ref)
        if kind not in ("price", "bars"):
            continue
        tick, _, interval = key.partition("@")
        s1, s2 = tick.upper(), (interval or "1d")
        try:
            records = store.cert_log_records("corroboration", s1, s2)
        except Exception:  # noqa: BLE001 — evidence is never fatal to issuance
            continue
        if not records:
            continue
        seqs = [int(r["seq"]) for r in records]
        bundle = transparency_bundle(store, "corroboration", s1, s2, seqs,
                                     keystore, now=now)
        if bundle is None:
            continue
        # The attempt ROWS, keyed by the seq of the leaf that committed them. The
        # verifier re-derives each leaf's digest from these, so they are the
        # evidence rather than a convenience copy — which is why they are keyed by
        # seq and not merged: a merged list could not be attributed to a leaf.
        attempts = {}
        for rec, lf in zip(records, bundle["leaves"]):
            rows = store.corroborations(s1, s2, limit=10_000)
            probe_ts = float(lf["lo_ts"])
            attempts[str(lf["seq"])] = [
                [r["provider"], r["origin"], r["outcome"], r["value"]]
                for r in rows if float(r["probe_ts"]) == probe_ts]
        # Keyed `kind|key` — canonical JSON needs string keys, and this is the form
        # the verifier re-derives the scope from (SPEC §9.6). A tuple repr here would
        # be unparseable by any implementation that is not CPython.
        anchors[f"{kind}|{key}"] = {
            "table": "corroboration",
            "scope": [s1, s2],
            "sth": bundle["sth"],
            "leaves": bundle["leaves"],
            "attempts": attempts,
        }
    return anchors


def build_envelope(src: str, data_service=None, *, keystore=None,
                   fetcher=None, seed: Optional[int] = None, K: int = 63,
                   alpha: float = 0.05, strict: bool = True,
                   witness=None, now: float = 0.0,
                   require_tier: Optional[str] = None) -> dict:
    """Run the DRC certificate for `src` and package it as a CNE. The inputs are
    fetched ONCE and both certified and committed, so the envelope's bound and
    its input commitments are guaranteed consistent. Signs with `keystore` when
    given. When a keystore AND a readable capture store are both present, each
    input is TRANSPARENCY-ANCHORED: its per-row Δ is attributed from the signed
    cert_log and the covering leaves + inclusion proofs + STH are attached, so a
    verifier can confirm the width rests on capture-time-committed deltas, not the
    signer's later word. When a `witness` is also supplied, each anchored input
    carries a co-signature over its complete STH under a cryptographically distinct
    key. A client pinning that key can detect a fork or rewind relative to the
    witness's retained state; organizational independence exists only when another
    party operates the witness. `now` is passed in (never wall-clock-read) so
    envelopes are reproducible; the caller stamps real time if it wants it."""
    refs = data_refs(src)
    src_fetcher = fetcher or StoreCertifiedFetcher(data_service)
    fetched: Dict[Tuple[str, str], FetchedSeries] = {}
    fetch_failed = False
    for r in refs:
        try:
            fs = src_fetcher.get(*r)
        except Exception:  # noqa: BLE001 - normalized by certified replay below
            fetch_failed = True
            break
        fetched[r] = fs

    # transparency anchoring (opt-in: needs a keystore AND a store to read the log)
    store = getattr(data_service, "store", None) if data_service is not None else None
    anchors: Dict[Tuple[str, str], dict] = {}
    if keystore is not None and store is not None and not fetch_failed:
        for r, fs in list(fetched.items()):
            try:
                a = _anchor_input(store, keystore, r, fs.series, now, witness)
            except Exception:  # noqa: BLE001 — anchoring is best-effort, never fatal
                a = None
            if a is not None:
                anchors[r] = a
                # commit the SIGNED-log-attributed deltas (anchorable) for the
                # width, aligned to the same canonical series the digest commits.
                fetched[r] = FetchedSeries(
                    series=a["canonical"], deltas=a["deltas"],
                    uncertified=int(np.isnan(a["deltas"]).sum()),
                    law=a.get("law"))

    inputs: List[dict] = []
    for r, fs in fetched.items():
        c = {
            "kind": r[0], "key": r[1],
            "digest": commitment_digest(r[0], fs.series),
            "n": int(len(fs.series)),
            "deltas": _compress_deltas(fs.deltas),
            "uncertified": int(fs.uncertified),
        }
        if r in anchors:
            c["transparency"] = anchors[r]["block"]
        inputs.append(c)

    # Always classify and normalize refusal metadata through ``certified_run``.
    # A provider exception is not independently reproducible by a verifier and
    # therefore cannot safely become signed refusal prose.  The cached fetcher
    # contains every successfully fetched input and deterministically refuses at
    # the first missing reference, without calling the provider a second time.
    cert = certified_run(src, fetcher=_DictFetcher(fetched), seed=seed, K=K,
                         alpha=alpha, strict=strict,
                         require_tier=require_tier)

    cne: dict = {
        "type": ENVELOPE_TYPE,
        "program": src,
        "program_hash": _program_hash(src),
        "inputs": inputs,
        # `require_tier` is recorded ONLY when set, so an envelope issued without a
        # floor is byte-identical to one from before the option existed. It must be
        # in the SIGNED params: a refusal the verifier cannot reproduce is a refusal
        # it reports as a replay mismatch (SPEC section 7.1).
        "params": ({"K": int(K), "alpha": float(alpha), "strict": bool(strict)}
                   if require_tier is None else
                   {"K": int(K), "alpha": float(alpha), "strict": bool(strict),
                    "require_tier": str(require_tier)}),
        "kernel": _kernel_id(),
        "created": float(now),
    }
    if cert.refused or not cert.ok:
        cne["refused"] = True
        cne["reason"] = cert.reason
        cne["scalar"] = None
        cne["program_class"] = cert.program_class
        cne["error_budget"] = {"quantization": None}
    else:
        cne["refused"] = False
        cne["reason"] = None
        cne["scalar"] = float(cert.base_value)
        cne["program_class"] = cert.program_class
        cne["seed"] = int(cert.seed)
        cne["error_budget"] = {
            "quantization": {
                "width": float(cert.width), "level": float(cert.level),
                "exact": bool(cert.level_exact), "tier": cert.program_class,
            },
            # reserved, never fabricated — filled by W4
            "sampling": None,
            # NOT null. A blank slot beside a filled one reads as "checked, fine";
            # the certificate must instead NAME what cross-source evidence exists
            # about its inputs, and when none does, why. Network-free (capability
            # declarations only), so issuing stays a local operation.
            "provider": _provider_evidence(refs, data_service),
            "model": None,
        }
        # W3: make the provider slot CHECKABLE rather than merely attested. The
        # anchors ride inside the provider block so a reader finds the evidence
        # beside the claim it supports, and the trust label is set from whether
        # anchoring actually happened — never asserted independently of it.
        _anchors = _provider_anchors(refs, data_service, keystore, now=now)
        if _anchors:
            cne["error_budget"]["provider"]["anchors"] = _anchors
            cne["error_budget"]["provider"]["trust"] = "transparency-anchored"
        cne["assumptions"] = list(cert.assumptions)
        if cert.branch_sites:
            cne["branch_sites"] = cert.branch_sites

    if keystore is not None:
        from alelyon.runtime.atlas.data.attest import canonical
        cne["key_id"] = keystore.key_id()
        cne["public_key"] = keystore.public_key_hex()
        cne["signature"] = keystore.sign(canonical(_unsigned(cne))).hex()
    return cne


def _unsigned(cne: dict) -> dict:
    return {k: v for k, v in cne.items() if k != "signature"}


def _kernel_id() -> str:
    from alelyon.runtime.vector import native
    return f"alelyon-vector/{native.version()}" if native.available() else "numpy-fallback"


def rebuild_fetcher(cne: dict, input_data: Dict) -> _DictFetcher:
    """Reconstruct the certified fetcher from a CNE's commitments + a caller's
    copy of the input series, so certified_run reproduces the exact bound. Keys
    of `input_data` may be (kind, key) tuples or the bare key string."""
    table: Dict[Tuple[str, str], FetchedSeries] = {}
    for c in cne.get("inputs", []):
        ref = (c["kind"], c["key"])
        series = input_data.get(ref)
        if series is None:
            series = input_data.get(c["key"])
        if series is None:
            raise KeyError(f"input data missing for {ref}")
        # replay the SAME canonical form the digest committed (sort + dedup),
        # so a correctly-committed input replays to the identical scalar.
        series = canonical_series(series)
        deltas = _decompress_deltas(c["deltas"])
        # DERIVE the capture law from the PROVEN leaves, never from a field the
        # signer wrote. Without this the exact-cents aggregate guard would fire at
        # issuance and not on replay, so an envelope claiming Δ=0 over a total past
        # 2^53 cents would re-derive its own rounded number and verify — a false
        # exactness claim that every other check waves through.
        # The COLUMN is derived from the input's canonical scope, never read from
        # the block — the block's own `column` is attacker-chosen, and this runs
        # before `_verify_one_anchor` has validated it.
        law = None
        if c.get("transparency"):
            try:
                law = _law_of_block(c["transparency"], _scope_of(ref)[3])
            except MalformedEnvelope:
                law = None            # unscopeable kind: the anchor check refuses
        # DERIVE the uncertified count from the committed deltas — never read it
        # from the envelope. certified_run gates strict-mode refusal and the
        # branch-stability precondition on this number, so trusting the field
        # would let a one-word lie switch both off.
        table[ref] = FetchedSeries(series=series, deltas=deltas,
                                   uncertified=int(np.isnan(deltas).sum()),
                                   law=law)
    return _DictFetcher(table)
