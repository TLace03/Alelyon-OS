# Certified Number Envelope — wire-format specification, `alelyon.cne/v0`

> **spec version:** `alelyon.cne-spec/0.3.0` · **envelope type specified:** `alelyon.cne/v0`
> **status:** DRAFT — normative for implementations, not yet frozen. Freezes per §10.1 at
> the first external verification ([PLATFORM.md](PLATFORM.md) §5).
> **last-refereed:** 2026-07-29 · Bound by [CLAIMS.md](CLAIMS.md); the do-not-say list
> overrides anything below.
> **workstream:** Track 0 / W1(a), with the W7 versioning policy inlined as §10.

## 0. What this document is, and what it is not

This is the artifact a second implementer builds from. Until it existed, the
authoritative format was "whatever the Python does" ([PLATFORM.md](PLATFORM.md)
W4: *"One implementation (Python), which means the spec is whatever the Python
does."*). Every construction below is stated so that an implementation written
without reading `alelyon/`'s source can produce and check byte-identical
artifacts.

**It is not a proof of anything.** The guarantee an envelope carries is narrow and
stated in §11: replay of an arithmetic result plus a **storage-quantization** error
term, under a key the verifier pinned out of band. It says nothing about whether
the input data is true. Per [CLAIMS.md](CLAIMS.md) rule 8, we certify arithmetic
and storage error, not truth.

**It is not complete over the whole DSL.** §8.6 defines two profiles. Profile 1
(linear-exact) is frozen here and is what a second implementation must reproduce.
Profile 2 is defined by reference behaviour and is explicitly *not* frozen — a
spec that claimed otherwise would be overclaiming, because the smooth and
branch-guarded tiers inherit pandas' rolling/ewm semantics, which no prose here
pins down.

### 0.1 Requirement levels

MUST / MUST NOT / SHOULD / MAY per RFC 2119. A **MUST-fail** condition is one
where a verifier is required to return `ok=false`; failing closed on anything not
described here is always permitted and usually correct.

### 0.2 Conformance

An implementation conforms if, over the vector suite of §12, it produces for every
case the exact `ok`, the exact value of every `checks` slot including `null`, the
exact `width_trust`, the exact `provider_trust`, and the exact set of
`reason_classes` (§9.4). Prose reasons are **not** part of conformance and may be
reworded freely.

---

## 1. Conventions

| Term | Meaning in this document |
|---|---|
| byte order | little-endian wherever a float or int is packed (`<` in Python `struct` terms). Explicit at every site. |
| hex | lowercase, no prefix, no separators. A 32-byte key is exactly 64 hex characters. |
| BLAKE2b-256 | BLAKE2b with a 32-byte digest, no key, no salt, no personalisation. |
| SHA-256 | FIPS 180-4 SHA-256. |
| ed25519 | RFC 8032 Ed25519, 32-byte public keys, 64-byte signatures, hex-encoded on the wire. |
| epoch seconds | seconds since 1970-01-01T00:00:00Z as an IEEE-754 binary64, **not** an integer. Fractional values are legal and are packed as-is. |
| f64 | IEEE-754 binary64. |

Two hash constructions appear and MUST NOT be confused: **BLAKE2b-256** for
content digests and cert-log leaf links, **SHA-256** for the RFC-6962 Merkle tree.
This split is historical, not principled; it is frozen because changing it
invalidates every stored chain.

---

## 2. Canonical JSON — the signing encoding

Everything that is signed is signed over the canonical encoding of a JSON object.
Reference: `alelyon/runtime/atlas/data/attest.py::canonical`.

### 2.1 The encoding

Serialise the object as JSON with:

1. object members sorted by key, ascending, comparing the keys' **Unicode code
   point sequences** (Python `str` ordering — i.e. UTF-16-agnostic code-point
   order, not UTF-8 byte order; these differ only for astral-plane keys, which
   §2.2 does not forbid but which no producer emits);
2. no insignificant whitespace: member separator `,`, key/value separator `:`;
3. non-ASCII characters emitted literally as UTF-8, **not** `\u`-escaped;
4. the result encoded UTF-8, with no byte-order mark.

### 2.2 Injectivity constraints (MUST)

An encoding that is not injective makes a signature over it mean less than it
appears to, so a producer MUST reject, and a verifier MUST treat as malformed,
any object containing:

- **duplicate raw member names**, including names that become equal only after
  JSON escape decoding. Every JSON ingestion boundary MUST reject duplicates at
  every depth before converting the document to an ordinary map; last-wins and
  first-wins parsing are not permitted.
- **a non-string member key.** `{1:"a"}` and `{"1":"a"}` would otherwise encode
  identically, and `{10:…, 9:…}` sorts numerically before stringification but
  lexically after it — one document, two signed byte strings.
- **a non-finite float** (NaN, +Infinity, −Infinity). These are not JSON
  (RFC 8259) and cannot be re-parsed interoperably. A non-finite value in a
  certificate is a refusal, never something to sign.

The check is depth-first over the whole object graph, including inside arrays.

### 2.3 Number formatting (MUST — the interop trap)

This is the single most likely place for a second implementation to diverge, and
it diverges silently: a wrongly formatted float changes the signed bytes, so the
signature fails with no indication of *why*.

**Integers** are emitted as exact decimal with no exponent, no fraction, and no
size limit. `2**70` emits `1180591620717411303424`.

**Floats** are emitted using Python's `repr` semantics, which are:

1. the **shortest** decimal string that round-trips to the same f64;
2. rendered in **exponential** form iff the decimal exponent is `>= 16` or
   `<= -5`, otherwise in positional form;
3. exponential form is `<mantissa>e<sign><digits>`, with the sign always present
   and the exponent zero-padded to **at least two digits**;
4. positional form always contains a `.`; an integral value gets a trailing `.0`;
5. `-0.0` emits `-0.0`, distinct from `0.0`.

Verified boundaries (this repo, CPython 3.12):

| value | emitted |
|---|---|
| `1.0` | `1.0` |
| `1e15` | `1000000000000000.0` |
| `1e16` | `1e+16` |
| `1e-4` | `0.0001` |
| `1e-5` | `1e-05` |
| `1.5e-5` | `1.5e-05` |
| `2.0**53` | `9007199254740992.0` |
| `1/3` | `0.3333333333333333` |
| `5e-324` | `5e-324` |
| `1.7976931348623157e308` | `1.7976931348623157e+308` |
| `-0.0` | `-0.0` |

> **Known cost, recorded rather than hidden.** This is *not* RFC 8785 (JCS): JCS
> mandates ECMAScript `Number::toString`, which renders `1e16` as
> `10000000000000000` and never pads an exponent. Rust's `f64` `Display` differs
> again (no exponential form at all). A second-language implementation must
> therefore implement Python-repr formatting deliberately; it will not get it
> from its standard library. §10.4 records adopting JCS as a v1 candidate. The
> cost of migrating is that every signature over every existing envelope becomes
> invalid, which is why it is a v1 question and not a v0 patch.

### 2.4 String escaping (MUST)

Escape `"` as `\"` and `\` as `\\`. Escape U+0008, U+000C, U+000A, U+000D, U+0009
as `\b`, `\f`, `\n`, `\r`, `\t` respectively. Escape every other code point below
U+0020 as `\u` followed by **four lowercase hex digits**. Do **not** escape `/`,
U+007F (DEL), U+2028, or U+2029. Emit every other code point literally.

### 2.5 Signing

A signature over object `O` is `ed25519(sk, canonical(O_unsigned))` where
`O_unsigned` is `O` with the top-level `signature` member removed and nothing else
changed. Note that this means **unknown members are covered by the signature**
(§10.2).

---

## 3. Input commitment

### 3.1 Canonical series form (MUST)

Reference: `envelope.py::canonical_series`. Both the digest and the replay consume
the same form, so a correctly committed input replays to the identical scalar:

1. if the index has duplicates, drop all but the **last** occurrence of each
   index value, in the series' original order (matching the capture store's
   `INSERT OR REPLACE` resolution);
2. if the index is not monotonically increasing, sort ascending by index;
3. otherwise leave untouched.

Step 1 precedes step 2. After it the index is unique, so ordering is total and no
value comparison — which NaN would break — is ever needed.

### 3.2 `input_digest` byte layout (MUST)

Reference: `envelope.py::input_digest`. Over the canonical series of §3.1, in
index order, feed a BLAKE2b-256 hasher:

```
for each row (index i, value v):
    8 bytes  <d  float64( epoch_seconds(i) )
    8 bytes  <d  float64( v )
```

`epoch_seconds(i)` is the row's timestamp as f64 seconds since the Unix epoch.
The digest is the hasher's 32-byte output, lowercase hex.

Non-finite values are packed as their IEEE bit patterns and never compared, so a
NaN row contributes fixed, reproducible bytes. There is **no** length prefix and
**no** field separator: injectivity rests on every row contributing exactly 16
bytes, so no two distinct row sequences can produce the same byte stream.

> **Scope limit (MUST be preserved in prose).** The digest binds timestamps and
> values only. It does **not** bind the series *name*, the input `kind`, or the
> `key`. Those are carried as separate signed members of the input commitment
> (§7.3) and are bound by the envelope signature, not by the digest. An
> implementation MUST NOT describe the digest as identifying which series the data
> is.

### 3.3 Data supply at verification

A verifier is given the caller's own copy of each input. Lookup MUST try the
`(kind, key)` pair first and fall back to the bare `key`. A missing input is
reason class `input-missing`; a present but differing input is
`input-digest-mismatch`.

---

### 3.4 Keyed tables — the `table` input kind

A reconciliation input is a keyed table, not a time series: rows identified by a
claim id, a policy id, or an accident-year × development-age cell. Its canonical
form and its digest are separate constructions, and a verifier MUST dispatch on the
input's `kind` rather than assume a layout.

**Canonical form.** Duplicate keys resolve keep-**LAST**, then sort **ascending by
key**. Key order is by Unicode code point, which for UTF-8 is byte order — the two
agree over all of Unicode, so there is no choice to get wrong. Keys are compared as
raw code-point sequences with **no normalisation**: two Unicode spellings of the
same-looking id are different rows. Normalising would merge rows a reconciliation
exists to distinguish, and fixing a normal form would make the digest depend on a
Unicode version. Canonicalising keys belongs to the ingestion adapter, upstream of
any commitment.

A row key MUST be a string (never a coerced number — `1` and `"1"` must not become
one row), non-empty, and at most 4096 UTF-8 bytes.

**`table_digest` byte layout (MUST).** Over the canonical rows, in key order:

```
for each row (key k, value v):
    8 bytes  <Q  uint64( byte length of utf8(k) )
    N bytes      utf8(k)
    8 bytes  <d  float64( v )
```

The **length prefix is load-bearing**, not tidiness. Concatenating keys without it
is not injective: rows `("ab",1),("c",2)` and `("a",1),("bc",2)` feed the hasher
identical bytes, so two different tables would share a digest and a commitment to
one would be a commitment to the other.

**Scope derivation.** A table input's `key` is `"<dataset>|<column>"` — one capture
scope per (dataset, column), which is also the granularity Δ is certified at:

| `kind` | table | scope1 | scope2 | column |
|---|---|---|---|---|
| `table` | `table` | `upper(dataset)` | `lower(column)` | `lower(column)` |

A key with no `|`, an empty dataset, or an empty column MUST be rejected.

**Coverage is DIGEST IDENTITY, not interval containment (MUST).** A cert-log leaf
spans `[lo_ts, hi_ts]`, and a keyed table has no time axis to span. A table scope
therefore writes `lo_ts = hi_ts = 0.0`, and a verifier **MUST NOT** interval-test a
table leaf. Coverage instead requires the input's `table_digest` to **equal** a
proven leaf's `value_digest` — the two are the same function over the same canonical
form, so equality means "this is that captured batch", decided on the verifier's own
copy of the data. Δ is the maximum over matching leaves, and the capture law (§5.4)
travels with the winner.

This is strictly stronger than an interval test, and it avoids the alternative:
mapping rows onto a synthetic ordinal, which the writer would control.

> **Scope limit.** An input assembled from SEVERAL capture batches cannot anchor,
> and its width degrades honestly to `authenticated`. For a quarterly extract — one
> file, one batch — that is the normal case, not a limitation. An implementation MUST
> NOT paper over it by relaxing identity to a subset match.

Unknown kinds MUST fail closed: there is no default commitment layout and no default
scope. (An earlier revision of the reference implementation fell through to the FRED
series scope for any kind that was not `price`, so a new kind would silently claim an
unrelated capture scope and be checked against whatever Δ lived there.)

## 4. Per-row Δ commitment (the delta block)

Each input commits one Δ (quantization step) per row of its canonical series.
`null` marks a row with **no capture bound** — uncertified, not zero. Reference:
`envelope.py::_compress_deltas` / `_decompress_deltas`.

### 4.1 The three encodings

Exactly one of `const`, `runs`, `list` MUST be present.

```json
{"const": 9.5367431640625e-07, "n": 120, "uncertified_idx": []}
{"runs": [[9.5e-07, 40], [null, 5], [1.1e-06, 75]], "n": 120}
{"list": [9.5e-07, null, 1.1e-06]}
```

- **`const`** — every row shares one finite Δ, except the rows named in
  `uncertified_idx`, which are `null`. `n` is the row count. A producer emits this
  form only when **every** row is finite and equal, so `uncertified_idx` is `[]`
  in practice; the field exists because a decoder MUST honour it.
- **`runs`** — run-length pairs `[value_or_null, count]`, in row order. Batch-derived
  Δ are piecewise-constant, so this is O(#batches) rather than O(#rows) — the
  difference between a 1 KB and a multi-MB envelope at enterprise scale. A producer
  chooses `runs` over `list` iff `2·len(runs) <= n`.
- **`list`** — the full per-row array, `null` for uncertified.

### 4.2 Decoding an untrusted block (MUST fail closed)

A decoder MUST validate before allocating and MUST reject, not repair. Every
condition below is a MUST-reject:

| Condition | Rejected because |
|---|---|
| block is not a JSON object | — |
| anything other than exactly one `const`/`runs`/`list` member | ambiguous |
| `n` outside `[0, 10_000_000]` | resource guard; nothing legitimate approaches it |
| `n`, a run count, or an `uncertified_idx` member is not a JSON integer | cross-language coercion would be ambiguous |
| `uncertified_idx` is not an array | — |
| `const` omits `uncertified_idx` | the const form's row exceptions are not explicit |
| an index in `uncertified_idx` outside `[0, n)` | — |
| `runs` is not an array | — |
| a run entry is not a two-element `[value, count]` | — |
| a run `count` is negative | — |
| run counts do not sum **exactly** to `n` | see below |
| `list` is not an array | — |
| `len(list) > 10_000_000` | resource guard |
| `list` present with `n` and `len(list) != n` | — |
| a non-null Δ is not a finite, non-negative JSON number | Δ cannot be negative, non-finite, boolean, or text |
| `const` is `null` | uncertified const rows are represented by `uncertified_idx`, not an all-null shortcut |

The run-sum check is load-bearing and MUST NOT be relaxed to "at least `n`". An
implementation that pre-allocated an uninitialised buffer and filled it from
unvalidated runs would leave whichever bytes the runs did not cover holding
arbitrary memory — nondeterministic verification, and stray zeros there would
**shrink** the bound. Implementations SHOULD initialise the array to `null`
(uncertified) rather than zero, so that a validation gap degrades to "unbounded"
rather than to "exact".

Rejection is a distinct failure mode from a check returning false: it aborts
verification with reason class `malformed-envelope`.

### 4.3 Derived, never read

`uncertified` (§7.3) is a count the envelope *states*. A verifier MUST
**re-derive** it as the number of `null` entries in the decoded block and MUST NOT
use the stated value for any decision. The stated value is compared against the
derived one and a mismatch is `uncertified-count-mismatch`. This matters because
the uncertified count gates strict-mode refusal and the branch-stability
precondition, so trusting the field would let a one-word edit switch both guards
off.

---

## 5. Capture log — leaves, payloads, and the chain

The capture store keeps an append-only, never-pruned log with one leaf per capture
batch, per scope. A scope is `(table, scope1, scope2)`.

### 5.1 `value_digest` (MUST)

BLAKE2b-256 over the batch's rows **sorted ascending by timestamp**.

For a bars batch, per row:

```
8 bytes  <q  int64( ts )                     # integer epoch seconds
8 bytes  <d  float64( open   )   # None -> NaN
8 bytes  <d  float64( high   )
8 bytes  <d  float64( low    )
8 bytes  <d  float64( close  )
8 bytes  <d  float64( volume )
```

For a series batch, per row:

```
8 bytes  <d  float64( ts    )                # REAL epoch seconds, not int64
8 bytes  <d  float64( value )   # None -> NaN
```

`None` maps to NaN before packing. Note the deliberate asymmetry: bars timestamps
are `<q` int64, series timestamps are `<d` f64. This is frozen.

### 5.2 `cert_leaf_hash` — the chain link (FROZEN byte format)

Reference: `attest.py::cert_leaf_hash`. Any change invalidates every stored chain,
so this format is frozen independently of the rest of v0.

Build the following eleven fields, join them with a single **U+001F** (unit
separator) between adjacent fields, encode UTF-8, and take BLAKE2b-256, lowercase
hex:

| # | field | rendering |
|---|---|---|
| 1 | `table_name` | as-is |
| 2 | `scope1` | as-is |
| 3 | `scope2` | as-is |
| 4 | `seq` | decimal integer |
| 5 | `value_digest` | as-is (lowercase hex) |
| 6 | `n` | decimal integer |
| 7 | `lo_ts` | **f64 repr** per §2.3 |
| 8 | `hi_ts` | **f64 repr** per §2.3 |
| 9 | `bits` | decimal integer |
| 10 | `payload` | the payload **string**, verbatim |
| 11 | `prev_hash` | as-is |

Fields 7 and 8 use the §2.3 float rendering, *not* a fixed-precision format. A
second implementation that formats `lo_ts` as `1704153600` instead of
`1704153600.0` computes a different leaf hash and rejects every genuine leaf.

`prev_hash` for the first leaf of a scope is the genesis constant; for every
subsequent leaf it is the previous leaf's `leaf_hash`. `seq` is a per-scope
monotonic counter starting at 0, with a uniqueness constraint so two concurrent
captures cannot fork the chain at one `seq`.

The payload is committed as an opaque string. A verifier MUST recompute the leaf
hash from the leaf record's own fields and compare it to the hash the inclusion
proof commits to (§6.3) — this is what binds the payload's Δ to the signed tree.

### 5.3 Payload schema

The payload is a JSON **array** serialised as a string. It contains per-column
certificate objects and, for `bars`/`series` leaves, exactly one row-membership
object. Each column object:

```json
{"column": "close", "n": 120, "delta": 9.5367431640625e-07,
 "seed": 4611686018427387904, "error_var": 7.57e-14}
```

`column` names are lowercase-canonical. `error_var` is Δ²/12, and is `0.0` when
Δ is `0.0`.

The membership object has no `column` member, so pre-membership Δ parsers ignore
it rather than treating it as a numerical certificate:

```json
{"membership":{"encoding":"i64-epoch-seconds/v0","rows":[1704153600,1704240000]}}
{"membership":{"encoding":"f64-epoch-seconds/v0","rows":[1.0,3.0]}}
```

`bars` uses JSON integers representing the same signed `<q` epoch seconds as
§5.1; `series` uses finite JSON numbers representing the same `<d` timestamps.
The embedded payload is a second raw JSON ingestion boundary: a verifier MUST
apply §2.2's duplicate-member and non-finite-number rejections recursively before
using any delta, law, or membership field. A malformed payload yields no partial
claim; it MUST NOT be repaired with last-wins semantics.
Rows MUST be strictly increasing and unique, MUST number exactly the leaf's `n`,
and their first/last values MUST equal `lo_ts`/`hi_ts` after the corresponding
numeric conversion. Membership is limited to 1,000,000 rows and the payload to
32 MiB before JSON parsing. Wrong encoding, duplicate/unsorted/non-finite rows,
multiple membership objects, or a count/extrema mismatch makes membership
unusable.

The complete payload string, including this object, is already field 10 of the
frozen leaf hash (§5.2); no hash-layout change is required. A legacy leaf without
this object has **no exact row membership** and covers no time-series row. Its
`lo_ts`/`hi_ts` remain historical diagnostics, never a coverage fallback.

### 5.4 Parsing Δ out of a payload (MUST — the fake-zero rule)

Reference: `attest.py::payload_deltas`. This returns two things: a map of usable
`{column: Δ}`, and a sorted list of columns whose Δ is **UNUSABLE**. Both the
producer and the verifier MUST apply the identical rule, or they will disagree
about the same bytes.

A column's Δ is **usable** iff its `delta` member is:

- **present**, and
- a JSON **number** — not a numeric string, not a boolean (note that a boolean is
  an integer in some languages and MUST be excluded explicitly), not an array, not
  an object, and
- **finite**, and
- **`>= 0`**,

**and** its `law` member (§5.4.1) names a capture law this implementation
recognises.

Anything else — absent `delta`, `null`, non-numeric, non-finite, negative, or an
unrecognised `law` — is UNUSABLE.

#### 5.4.1 Capture laws (MUST)

A **capture law** is the rule that produced a column's Δ. It has to travel with the
certificate, because what makes Δ *checkable* by someone holding only the data is
law-specific — and a verifier that assumed one law would either reject legitimate
captures under another or, worse, accept a claim the named law could not have
produced.

`law` is a member of each per-column payload object (§5.3). It lives there, rather
than in a new leaf field, because `cert_leaf_hash` is frozen (§5.2) and cannot gain a
field — while the payload string is committed verbatim as field 10, so a law named
there is bound to the signed leaf at no cost to the frozen construction. It is
**not** a member of the envelope: a member the signer writes would be a law id the
adversary picks freely, whereas the payload is proven into the transparency tree the
same way the Δ is.

| `law` | meaning | Δ = 0 is admissible iff |
|---|---|---|
| *absent* | the relative-dither law (every leaf written before laws existed) | the value is `0` |
| `dither-relative/v0` | Δ = `scale · 2^(1-bits)`, so Δ > 0 for any nonzero scale | the value is `0` |
| `exact-cents/v0` | the stored value **is** an integer count of cents | the value is an exact integer with `|v| ≤ 2^53` |
| anything else | **unrecognised** | never — the column is UNUSABLE |

An unrecognised law MUST make the column unusable rather than fall through to a
permissive default. A signer able to name a law nobody implemented and have its Δ
read anyway would hold exactly the free pass §7.6's forgery exploited, one level up.

`absent` maps to the relative-dither law, which is safe because that law carries the
**stricter** invariant: an adversary gains nothing by omitting the member. For the
same reason a producer MUST NOT begin emitting `dither-relative/v0` explicitly —
the leaf hash commits the payload string, so a gratuitous new member would change
every future leaf for no gain.

**The exact-cents law stores the cent COUNT, not the dollar amount.** This is the
whole law. Storing dollars as `cents / 100.0` is a correctly-rounded f64
approximation of a generally non-dyadic rational, so its storage error is up to half
an ULP — small, but not zero, and `Δ = 0` over it would be false. Storing the count
makes `Δ = 0` literally true for every `|c| ≤ 2^53`. Capture MUST **refuse** (never
degrade) a value that is not a whole number of cents, or whose magnitude exceeds
`2^53`.

**Aggregate representability (MUST).** Per-element exactness does not imply the
RESULT is exact: `2^53` cents is ~$90 trillion for one element, but a total over many
elements can pass every per-element check and land where f64 no longer counts by
ones. At that point a sum returns a rounded number while Δ = 0 still claims "stored
exactly" — a receipt that reads as a penny-exact control total and is not one. A
producer MUST refuse when a result over exact-cents inputs exceeds `2^53`, and a
verifier that can determine the law from a proven leaf MUST apply the same rule on
replay. A guard that binds each element is not a guard on their sum.

**Zero width is not exactness of the answer.** Under this law the
storage-quantization term is genuinely zero, which invites reading the number as
exact. It is not: the computation's own floating-point rounding — the division in a
`mean`, for instance — is a compute-side error this certificate does not cover. That
boundary MUST be stated in the receipt's assumptions; it is the difference between
"certified exact storage" and the bare "certified" §1 forbids.

Two further rules, each of which exists because its absence was exploitable:

1. **Duplicate entries for one column resolve to the LARGEST Δ.** Every other Δ
   attribution in this system is conservative (max over covering leaves, max into
   the watermark). Last-wins would let a payload carry the honest Δ for an auditor
   to read while every reader used a smaller one written after it.
2. **A column that is EVER unusable is unusable**, even if another entry for it
   parsed cleanly. Otherwise the producer (reading the usable map) anchors a column
   the verifier (reading the unusable list) rejects.

> **Why absence may never default to zero.** Δ = 0 is a *legitimate* value: it
> means the column was stored exactly. Reading an **absent** `delta` as `0.0`
> therefore converts *unknown* into *claimed exact*. That was exploitable by the
> exact adversary this layer exists to resist: a signer holding the key omits the
> field at capture; the leaf hash commits whatever payload was written, so the
> chain and every inclusion proof stay valid; the fabricated zero then appears on
> **both** sides of the verifier's independent re-derivation, they agree, and a
> zero-width bound verifies as transparency-anchored. See §7.6 for the invariant
> that catches the explicit-`0.0` form of the same attack.

### 5.5 Δ attribution over rows (MUST)

Given a set of proven leaves for a scope and column, the Δ for a row at timestamp
`t` is the **maximum** `Δ` over every leaf whose signed exact membership contains
`t`:

```
covering(t) = { leaf : t ∈ leaf.membership.rows  and  column is USABLE in leaf.payload }
Δ(t)        = max { leaf.Δ[column] : leaf in covering(t) }      # undefined if empty
```

A row with no exact member leaf is uncertified. In particular, a sparse batch
capturing timestamps `{1,3}` does not cover timestamp `2`, even though `2` lies
between its extrema. A watermark is a scope diagnostic and MUST NOT supply Δ to
an unmatched row.

**Current-row state.** The reference store separately maps each currently served
row to the sequence that wrote its current bytes. Every overwrite first clears
that mapping in the data transaction; the certificate/log transaction assigns its
sequence only while the capture token still matches. Failed certification, raw
capture, and stale concurrent completion therefore leave the row uncovered.
Existing databases begin with no mapping and are not backfilled by interval.

**Residual value-binding limit (MITIGATED, not fixed).** Exact timestamps prevent
an omitted sparse row from borrowing a leaf. They do not, by themselves, prove
that a value now served at a previously captured timestamp is byte-identical to
the old batch. Honest producer paths clear current membership on every overwrite,
and `verify_data_integrity` re-derives a leaf digest only while all exact members
remain current. A key-holding signer can nevertheless attach an old bars leaf to
changed current values because the bars `value_digest` commits all OHLCV columns
while a `price` input supplies only `close`; the public verifier cannot re-derive
that digest. Closing this requires per-row/per-column value commitments or another
completeness proof and is not claimed by this mitigation.

> **Scope limit — anchoring proves matching, not maximality.** Attribution takes
> the max over the covering leaves the verifier is *shown*. When overlapping
> re-captures of the same timestamp exist, an issuer that presents a subset of the
> covering leaves presents a smaller max. Anchoring therefore proves that the
> committed Δ **match signed leaves covering every consumed row** — which defeats
> an invented Δ — and does **not** prove the committed Δ is the maximum over *all*
> leaves in the log. An implementation MUST NOT describe the anchor as proving
> maximality. Closing this requires a completeness proof obligation (that the
> verifier's re-derivation sees the same leaf set the producer did), which v0 does
> not carry. Recorded as [PLATFORM.md](PLATFORM.md) §3 ledger item 4.

---

## 6. Transparency log

### 6.1 Merkle tree (RFC 6962)

Over an ordered list of leaf hashes (hex strings from §5.2):

```
leaf node     = SHA256( 0x00 || bytes.fromhex(leaf_hash) )
internal node = SHA256( 0x01 || left || right )
```

Build bottom-up in pairs, left to right. When a level has an odd count, the last
node is **promoted to the next level unhashed** (the CT convention). The root of
an empty list is *absent* (`null`), not the hash of nothing. Domain separation
between leaf and internal nodes is what prevents an internal node being forged to
equal a leaf.

### 6.2 Inclusion proof

An audit path is an ordered list of sibling hashes, leaf to root. The **side** of
each sibling is NOT stored: it is derived from the index bit at each level, so a
proof only recomputes the root at its true position and replaying it at a
different index fails.

Verification of `(leaf_hash, index, tree_size, path, root)`:

```
if not (0 <= index < tree_size):        return false
node = leaf_node(leaf_hash);  idx = index;  size = tree_size;  pi = 0
while size > 1:
    if (idx XOR 1) < size:              # this node has a sibling at this level
        if pi >= len(path):             return false
        sib = path[pi];  pi += 1
        node = internal(node, sib) if idx is even else internal(sib, node)
    # else: promoted — carries up unchanged, consumes no path element
    idx  = idx // 2
    size = (size + 1) // 2
return pi == len(path) and node == root
```

The trailing `pi == len(path)` is required: a proof with **extra** unconsumed
elements MUST be rejected, not ignored.

### 6.3 Binding a proof to a signed head (MUST)

An inclusion proof carries its own `tree_size` and `root`. Checked in isolation it
shows only that a leaf, a path and a root are mutually consistent — which an
attacker can fabricate wholesale. A verifier MUST therefore:

1. verify the STH's signature under the **pinned** key (§6.5);
2. require `proof.tree_size == sth.tree_size`;
3. require `0 <= proof.index < sth.tree_size`;
4. recompute the root from the path and compare it against **`sth.root`**.

### 6.4 Consistency proof

RFC 6962 consistency proves the size-`n` tree contains the size-`m` tree as an
append-only prefix. Generation for `0 < m <= n`, over the leaf list `L`:

```
consistency(L, m):
    n = len(L)
    if m == n:  return []
    return subproof(m, L, true)

subproof(m, L, b):
    n = len(L)
    if m == n:   return []            if b else [merkle_root(L)]
    k = largest power of two strictly less than n
    if m <= k:   return subproof(m, L[:k], b)      + [merkle_root(L[k:])]
    else:        return subproof(m - k, L[k:], false) + [merkle_root(L[:k])]
```

Verification of `(m, n, first_root, second_root, path)`:

```
if first_root is absent or second_root is absent:  return false
if m == n:                       return first_root == second_root and path is empty
if not (0 < m < n) or path is empty:               return false
if m is a power of two:          path = [first_root] + path
fn = m - 1;  sn = n - 1
while fn is odd:  fn >>= 1;  sn >>= 1
fr = sr = path[0]
for c in path[1:]:
    if sn == 0:                  return false
    if fn is odd or fn == sn:
        fr = internal(c, fr);  sr = internal(c, sr)
        while fn != 0 and fn is even:  fn >>= 1;  sn >>= 1
    else:
        sr = internal(sr, c)
    fn >>= 1;  sn >>= 1
return sn == 0 and fr == first_root and sr == second_root
```

A rewrite of any of the first `m` leaves, or a size rollback, fails.

### 6.5 Signed tree head — `alelyon.sth/v0`

```json
{
  "type": "alelyon.sth/v0",
  "table": "bars",
  "scope": ["SYN", "1d"],
  "tree_size": 3,
  "root": "<64 hex>",
  "head_leaf": "<64 hex>",
  "key_id": "ed25519:<16 hex>",
  "public_key": "<64 hex>",
  "timestamp": 1000.0,
  "signature": "<128 hex>"
}
```

`scope` is a two-element array. `timestamp` is supplied by the caller, never read
from a wall clock, so a head is reproducible.

The schema is strict at the trust boundary. A verifier MUST require: `type` exactly
`alelyon.sth/v0`; a non-empty string `table`; `scope` as a JSON array of exactly two
non-empty strings; `tree_size` as a positive JSON integer (not a boolean, float, or
numeric string); `root`, `head_leaf`, and `public_key` as exactly 32 bytes of
lowercase hex; `key_id` equal to §6.6's identifier for `public_key`; `timestamp` as a
finite JSON number; and `signature` as exactly 64 bytes of lowercase hex. For a
one-leaf tree, `root` MUST equal `merkle_root([head_leaf])`; a larger bare STH does
not contain enough information to prove the head leaf's membership. Unknown members
MAY be present and are covered by the signature, per §10.2's additive rule.

Verification MUST also require a **pinned**, exactly 32-byte lowercase-hex
`public_key_hex` and MUST fail when it is absent — verifying against the key embedded
in the same untrusted object authenticates nothing. The embedded `public_key` MUST
equal the pin and `key_id` MUST equal `key_id(pinned)`. The signature is over
`canonical(sth without "signature")`. Malformed, non-finite, or non-canonically
encodable signed material is a refusal, never an exception from the verifier.

### 6.6 Key identifier

```
key_id = "ed25519:" + hex( BLAKE2b( public_key_bytes, digest_size=8 ) )
```

`public_key_bytes` is the 32-byte raw ed25519 public key. The result is the
literal prefix `ed25519:` followed by 16 lowercase hex characters.

### 6.7 Co-signature — `alelyon.cosign/v0`

```json
{
  "type": "alelyon.cosign/v0",
  "table": "bars",
  "scope": ["SYN", "1d"],
  "tree_size": 3,
  "root": "<64 hex>",
  "sth_digest": "<64 hex>",
  "log_key_id": "ed25519:<16 hex>",
  "witness_key_id": "ed25519:<16 hex>",
  "witness_public_key": "<64 hex>",
  "cosigned_ts": 1000.0,
  "signature": "<128 hex>"
}
```

A verifier MUST require **all** of: `type` exactly `alelyon.cosign/v0`; non-empty
`table`; `scope` as an array of exactly two non-empty strings; a positive integer
`tree_size`; lowercase 32-byte-hex `root` and `sth_digest`; non-empty `log_key_id`;
lowercase 32-byte-hex `witness_public_key`; finite `cosigned_ts`; lowercase
64-byte-hex `signature`; a pinned witness key as exactly 32 bytes of lowercase hex;
and an `expected_root` to bind against. `witness_public_key` MUST equal the pin and
`witness_key_id` MUST equal its §6.6 identifier. The signature is over
`canonical(statement without "signature")`. A verifier MUST also receive the
complete expected STH; a root alone does not identify its timestamp, scope, log key,
signature, or extension members.

The pinned witness key identifier and key material MUST differ from the complete
STH's log `key_id` and `public_key`. One key producing both signatures is not a
co-signing trust boundary. Distinct cryptographic roles are necessary but do not
establish organizational independence; that additionally requires another party to
operate the witness and preserve its retained state.

`sth_digest` is exactly
`hex(SHA-256(canonical(complete_sth)))`, where `complete_sth` includes its log
signature and every extension member. A verifier MUST recompute and compare that
digest and MUST also compare the repeated `root`, `tree_size`, `log_key_id`, `table`,
and `scope` fields. This binds the witness to the complete signed tree head, not
merely to a root that could be transplanted between heads. Missing, noncanonical, or
malformed expected-head material is a refusal.

Without both the root and complete-head binding, a valid co-signature over *some
other* head can be replayed as evidence about a head the witness never saw.

**Co-signing behaviour (producer side).** A witness maintains, per scope, the last
`(tree_size, root)` it co-signed. It MUST refuse — emit nothing — on a size
rollback (`size < prev`), on a same-size fork (`size == prev` with a different
root), and on growth (`size > prev`) that is not accompanied by a **valid
consistency proof** from the prior head. The first head for a scope is
trust-on-first-use.

> **Naming discipline ([CLAIMS.md](CLAIMS.md) §1).** This is a **co-signing
> witness seam**. It is independent only when a party other than the signer
> operates it. The reference witness ships co-located with the signer, and
> `checks.witness=true` therefore means "the pinned witness key signed this root",
> not "an independent party did". The guarantee is created by the deployment, not
> by the function.

---

### 6.8 Key lifecycle — succession, revocation, manifest

A verifier authenticates against a key it pinned out of band. That is only workable
over time if the issuer publishes a key **history** the client can follow forward, so
a pin does not have to be re-established on every rotation, and so a client can learn
that a key it holds has been withdrawn.

**The succession chain is the cryptographic content.** Key #1 is anchored out of band.
Every later key is introduced by a statement signed by its **immediate predecessor**,
so a key enters the chain only if the key already in it says so. A client pins the
root once and can authenticate every succession statement it receives.

**The manifest is transport, and is deliberately NOT signed as a whole.** Its
authority is the per-entry chain. An aggregate signature under the current key would
invite readers to treat that as the authority — and would let a compromised current
key rewrite the history of its own predecessors.

**The manifest alone is not fresh.** The succession checks below prove only the
statements present in the supplied manifest. A transport can otherwise serve an older
manifest, truncate a valid successor tail, or omit a signed revocation. The v0
verifier therefore accepts a supplied manifest only with the signed monotonic
checkpoint defined below, its separately pinned checkpoint key, and a previously
trusted signed checkpoint retained by that verifier. An aggregate manifest signature
without verifier-held rollback state would not solve the omission problem.

```json
{
  "type": "alelyon.keymanifest/v0",
  "issuer": "Alelyon",
  "root_key_id": "ed25519:<16 hex>",
  "published_at": 1700000000.0,
  "keys": [
    {"key_id": "ed25519:…", "public_key": "<64 hex>", "not_before": 1.0,
     "not_after": 2000.0, "status": "superseded",
     "succession": null, "revocation": null},
    {"key_id": "ed25519:…", "public_key": "<64 hex>", "not_before": 2000.0,
     "not_after": null, "status": "active",
     "succession": {"type": "alelyon.keysuccession/v0",
                    "predecessor_key_id": "ed25519:…", "key_id": "ed25519:…",
                    "public_key": "<64 hex>", "not_before": 2000.0,
                    "signature": "<128 hex>"},
     "revocation": null}
  ]
}
```

The checkpoint is a separate signed object:

```json
{
  "type": "alelyon.keymanifest-checkpoint/v0",
  "issuer": "Alelyon",
  "root_key_id": "ed25519:<16 hex>",
  "sequence": 2,
  "manifest_digest": "<64 lowercase hex>",
  "manifest_published_at": 1700000000.0,
  "entries": [ /* exact deep copy of manifest.keys */ ],
  "key_ids": ["ed25519:…"],
  "superseded_key_ids": [],
  "revoked_key_ids": [],
  "checkpoint_key_id": "ed25519:<16 hex>",
  "issued_at": 1700000001.0,
  "signature": "<128 lowercase hex>"
}
```

`manifest_digest` is
`hex(SHA-256(canonical(the complete keymanifest object)))`. The `entries` member is
an exact copy of the complete ordered key entries, including signed succession and
revocation statements and any extension fields. The three id arrays MUST agree
exactly with those entries and their statuses. `issued_at` MUST be finite and no
earlier than `manifest_published_at`. The checkpoint signature covers every member
except `signature` and MUST verify under a separately pinned, exactly 32-byte
lowercase-hex checkpoint key whose §6.6 identifier is `checkpoint_key_id`. That
identifier and key material MUST differ from every `key_id` and `public_key` in the
manifest entries. Reusing a manifest signing key collapses the authority that signs
history and the authority that protects rollback state; builders and verifiers MUST
refuse that role collision. Cryptographic separation does not by itself prove that
different people or systems hold the keys.

**Monotonic checkpoint verification (MUST).** A verifier retains the complete last
accepted signed checkpoint, not merely an unsigned sequence number. For a candidate:

1. verify both candidate and retained checkpoint signatures under the pinned
   checkpoint key and verify the candidate manifest's succession chain under its
   independently pinned root; reject either checkpoint if its checkpoint key
   identifier or material overlaps any manifest signing key;
2. require the candidate checkpoint's issuer, root, digest, publication time,
   complete entries, ordered ids, and status summaries to match that exact manifest;
3. reject sequence rollback; at an equal sequence accept only the byte-equivalent
   canonical checkpoint, rejecting same-sequence equivocation;
4. on sequence growth, preserve the prior ordered key-id prefix; preserve each prior
   key's identity, public key, `not_before`, and succession statement; never remove or
   rewrite a prior non-null `not_after`, revocation, or extension member;
5. permit only `active → active|superseded|revoked`,
   `superseded → superseded|revoked`, and `revoked → revoked`; never restore a
   retired key to active service; and
6. require non-decreasing manifest publication and checkpoint issuance times.

On success the exact candidate checkpoint is the next rollback state and MUST be
persisted atomically by the caller. This proves non-regression relative to retained
state. It does **not** tell a newly bootstrapping verifier that its first checkpoint
is globally newest; that initial checkpoint needs an out-of-band channel. It also
does not survive compromise of the separately trusted checkpoint signing key.

**Statuses (closed vocabulary).**

| status | meaning | envelopes it signed |
|---|---|---|
| `active` | currently signing | verify |
| `superseded` | rotated out in the ordinary course | **still verify**, inside the window |
| `revoked` | withdrawn for cause | **never a bare `ok`**, whenever issued |

The `superseded` / `revoked` split is load-bearing. If routine rotation invalidated
history, nobody would ever rotate. And `revoked` ignores the issuance time on purpose:
if the cause was compromise, the key was in someone else's hands for an unknown period
before anyone noticed, so "issued before the revocation date" establishes nothing.
A verifier that waved those through would make revocation advisory.

**Revocation reasons** are a closed set — `compromise`, `retired`,
`superseded-early`, `lost`. An unstated cause cannot be acted on, and `compromise`
versus `retired` is the difference between distrusting everything a key signed and
noting that it stopped signing.

**Who may revoke (MUST).** A revocation statement MUST be signed by the revoked key
itself or by one of its **successors** — never by an arbitrary key in the chain, or
any holder could retire any other. A self-signed revocation is accepted for a
voluntary retirement and MUST be reported as such: whoever stole a key can sign its
revocation just as well as its owner, so a self-signed revocation is evidence of
intent, not of custody.

**Manifest verification (MUST).** Against a pinned root:

1. a pinned root is **required** — refuse without one;
2. entry 0's `key_id` and `public_key` equal the pinned root, and it carries **no**
   succession statement;
3. every entry's `key_id` is the fingerprint (§6.6) of its own `public_key`;
4. for `i > 0`: the succession statement names `keys[i-1]` as predecessor, attests
   exactly this entry's key id and public key and `not_before`, and verifies **under
   the predecessor's key**;
5. `not_before` is non-decreasing along the chain;
6. `status == "revoked"` iff a revocation statement is present — in both directions;
7. a `superseded` entry carries a non-null `not_after`, or "superseded" means nothing;
8. every revocation present satisfies the who-may-revoke rule above.

**Verifier integration.** `key_status` (§9.1) is `null` when no manifest is supplied —
honestly not judged, like an absent witness key. Supplying a manifest opts in to a
stricter check. Supplying one **without** the root it chains to is refused
(`key-manifest-unrooted`). Supplying one without its candidate checkpoint, separately
pinned checkpoint key, or retained signed checkpoint is refused
(`key-manifest-checkpoint-required`). Invalid checkpoint material is
`key-manifest-checkpoint-invalid`; rollback or equivocation relative to retained state
is `key-manifest-checkpoint-not-monotonic`.

Envelopes carry `key_id`, so a rotation needs no re-issuance: an old envelope stays
verifiable under the pin that was current when it was signed.

## 7. The CNE object

### 7.1 Envelope skeleton

```json
{
  "type": "alelyon.cne/v0",
  "program": "show mean(returns(price(\"SYN\")))",
  "program_hash": "<64 hex>",
  "inputs": [ /* §7.3 */ ],
  "params": {"K": 63, "alpha": 0.05, "strict": true, "require_tier": "linear-exact"},
  "kernel": "alelyon-vector/0.1.0",
  "created": 1000.0,
  "refused": false,
  "reason": null,
  "scalar": 0.0004321,
  "program_class": "linear-exact",
  "seed": 7,
  "error_budget": { /* §7.4 */ },
  "assumptions": ["…"],
  "branch_sites": [ /* §7.5, present only when non-empty */ ],
  "key_id": "ed25519:<16 hex>",
  "public_key": "<64 hex>",
  "signature": "<128 hex>"
}
```

`program_hash` is `hex(SHA256(program.encode("utf-8")))` — SHA-256 here, not
BLAKE2b. `created` is caller-supplied, never wall-clock-read.

`params` MUST be a JSON object. `K` MUST be an unsigned 64-bit JSON integer,
`alpha` MUST be a finite JSON number, and `strict` MUST be a JSON boolean; omitted
members default to `63`, `0.05`, and `true` respectively. Numeric strings,
booleans in numeric positions, integral-looking floats for `K`, and non-finite
numbers MUST be rejected as malformed rather than coerced. A verifier MUST produce
the deterministic replay refusal before resampling when `K > 10,000`.

The complete carried `program` MUST also pass the Profile-1 resource envelope before
recursive parsing or execution: at most 65,536 UTF-8 bytes, 16,384 lexical tokens
(excluding the synthetic end token), 8,192 statement-plus-expression AST nodes, and
expression depth at most 64. These limits apply before final-output pruning so an
unreachable prefix cannot be used as a parser resource attack. After pruning, let `A`
be the retained statement-plus-expression node count and `R` the total consumed input
rows. A verifier MUST refuse before the base execution or any resample when
`(K + 1) * R * A > 50,000,000`; the `+1` accounts for the base execution. Products
MUST be evaluated with checked or saturating arithmetic so the guard itself cannot
overflow into acceptance.

`program` is hashed **verbatim, comments included**. Comments are discarded by the
lexer, so they cannot affect the tier or the number, and they are covered by
`program_hash` and therefore by the signature. That makes a comment the correct place
for a statement an issuer must be held to but which must not influence execution — for
example the row-selection rule behind a control total, where selection happens upstream
because it needs comparisons the exact tier does not admit.

`params.require_tier` is **OPTIONAL** and, when present, MUST be a string naming the weakest tier the
issuer was willing to certify at (§8.6). A program classifying weaker is REFUSED rather
than certified one tier down. It MUST be recorded in `params`, because `params` is
signed and is what a verifier replays with: an issuer that demanded a floor produced a
refusal under it, and a verifier that ignored the floor would replay to a *certificate*
and report `replay-refusal-mismatch`. An envelope issued without a floor omits the
member entirely, so it is byte-identical to one from before the option existed. Tier
order for the comparison, strictest first: `linear-exact`, `branch-stable-exact`,
`smooth-first-order`, `branch-stable-first-order`, `branch-sensitive`.

An unsigned envelope omits `key_id`, `public_key` and `signature`. It can never
reach `ok=true` (§9.2).

### 7.2 Refusal shape

A refusal is a first-class, signable outcome — "we honestly could not bound this"
is itself an attestation. When `refused` is `true`:

```json
{"refused": true, "reason": "<the real reason>", "scalar": null,
 "program_class": "<class or ?>", "error_budget": {"quantization": null}}
```

`seed`, `assumptions` and `branch_sites` are absent. The `reason` MUST be the
actual cause, never a generic sentence.

The verifier MUST replay the same program, committed inputs and signed `params` and
require all of the following: replay also refuses; the carried `reason` equals the
replayed reason byte-for-byte; `program_class` equals the replayed class; `scalar` is
present and null; `error_budget.quantization` is present and null; and the success-only
members above, plus success-only `sampling`, `provider`, and `model` budget members, are
absent. v0 carries no separate refusal-cause code, so the exact reason comparison is the
only way it can independently enforce “actual cause”; changing producer refusal prose is
therefore protocol-visible and requires matching vectors. A replay that produces a bound
or a different reason reports `replay-refusal-mismatch`; a different class reports
`tier-mismatch`; success-only bound members report `budget-mismatch`; malformed shape
reports `malformed-envelope`.

### 7.3 Input commitment

```json
{
  "kind": "price",
  "key": "SYN",
  "digest": "<64 hex>",
  "n": 120,
  "deltas": { /* §4 */ },
  "uncertified": 0,
  "transparency": { /* §7.6, optional */ }
}
```

`kind` is one of `price`, `series`, `table`. `n` is the canonical form's row
count. A `table` input commits under the keyed-table digest and anchors by
digest identity (§3.4); its `key` is `"<dataset>|<column>"`.

### 7.4 Error budget

```json
"error_budget": {
  "quantization": {"width": 1.23e-09, "level": 0.953125,
                   "exact": true, "tier": "linear-exact"},
  "sampling": null,
  "provider": {"status": "…", "trust": "signer-attested", "…": "…"},
  "model": null
}
```

Four named slots, always all four present.

- **`quantization`** — the DRC bound (§8.5). `width` is the conformal order
  statistic, `level` is `m/(K+1)`, `exact` says whether that level is a theorem
  or a first-order claim, `tier` restates `program_class`.
- **`sampling`** — `null` means **not filled**, and a verifier MUST NOT read it as
  zero. It is carried, never re-derived (§9.3).
- **`provider`** — MUST be an object with a `status`, never `null`. A blank slot
  beside a filled one reads as "checked, fine", which is the one thing this must
  never say by omission. `trust` is `signer-attested`: the slot describes the
  issuer's deployment, which no verifier can confirm from outside.
- **`model`** — `null` when the program *is* the computation and no model exists.

> **Never sum these into one number called "certified."** The composite is a
> labeled composition and the **dominant** term must be named. At 24-bit capture,
> sampling error typically dominates quantization by ~1e4, and the certificate
> says so ([CLAIMS.md](CLAIMS.md) rule 1).

### 7.5 Branch sites

Present only for the branch-guarded tiers. Each entry:

```json
{"op": "max", "winner": "2024-03-05 00:00:00", "margin": 0.41,
 "perturb_scale": 1.2e-07, "stable": true, "guard": "deterministic"}
```

`guard` is `deterministic`, `empirical`, or `failed`. `margin` is `null` when
infinite.

`perturb_scale` is the largest movement of that site's margin observed over the
K dither resamples **and** the two worst-case systematic probes of §8.5 step 7.
The `empirical` guard passes only where `margin > 3 × perturb_scale` at every
site.

> **Amended 2026-08-05 — `perturb_scale` was resample-only and that was unsound.**
> It previously ranged over the K dither resamples alone. Independent dither is a
> property of the *law*, not of the declaration: over an aggregate of n rows it
> cancels, so a mean moves by ≈Δ/√(12n), while a systematic rounding obeying the
> identical per-element promise |stored − true| ≤ Δ/2 moves it by Δ/2. The ratio
> grows like √(3n), so the fixed 3× factor was defeated by lengthening the series,
> and `sign(mean(x) − c)` certified at `branch-stable-first-order` with **width
> 0.0 and the decision inverted**. Measured at n = 25, 100, 400 and 1600.
>
> This is §2 rule 3 of [CLAIMS.md](CLAIMS.md) — validate against an
> independently-held invariant, never against the shape of what the writer emitted
> — reappearing one level up: the guard was validating against the spread of an
> assumed resampling law rather than against the bound the declaration actually
> gives.
>
> **Compatibility.** `branch_sites` is replay-compared (§9), so an envelope issued
> before this amendment carries the old, smaller `perturb_scale` and will not
> match a replay under this version. That is intended: those envelopes asserted a
> guard that did not hold. No golden vector carries a branch tier, and no external
> verification of one is recorded, so nothing outside this repository is
> invalidated. The `exact-cents/v0` law is unaffected — with Δ = 0 both probes are
> the identity — and so is the `branch-stable-exact` deterministic guard, which
> was always worst-case and never consulted the resamples.

### 7.6 Transparency block

Attached per input when the producer could attribute every consumed row's Δ to a
signed capture leaf.

```json
"transparency": {
  "table": "bars",
  "scope": ["SYN", "1d"],
  "column": "close",
  "sth": { /* §6.5 */ },
  "leaves": [
    {"seq": 0, "value_digest": "<64 hex>", "n": 120,
     "lo_ts": 1704153600.0, "hi_ts": 1718323200.0, "bits": 24,
     "payload": "[{\"column\":\"close\",…}]", "prev_hash": "<64 hex>",
     "inclusion_proof": {"leaf_hash": "<64 hex>", "value_digest": "<64 hex>",
                         "index": 0, "tree_size": 1, "proof": [], "root": "<64 hex>"}}
  ],
  "cosignature": { /* §6.7, optional */ }
}
```

**Anchor scope derivation (MUST).** The scope a block claims MUST NOT be trusted.
A verifier derives the canonical scope from the input commitment's `(kind, key)`:

| `kind` | table | scope1 | scope2 | column |
|---|---|---|---|---|
| `price` | `bars` | `upper(key)` | `1d` | `close` |
| `series` | `series` | `fred` | `upper(key)` | `value` |
| `table` | `table` | `upper(dataset)` | `lower(column)` | `lower(column)` |
| anything else | — | — | — | MUST fail closed |

and MUST reject when the block's `(table, scope, column)` differs, and separately
when the **STH's** `(table, scope)` differs. Without both checks the log identity
is attacker-chosen: a signer could anchor a claim about one series to an unrelated
capture scope whose Δ happen to be smaller, and produce an arbitrarily tighter
"anchored" width that still verifies.

**The Δ=0 plausibility invariant (MUST), dispatched per capture law.** After
re-deriving each row's Δ from the proven leaves and matching it against the committed
Δ, a verifier MUST additionally reject a leaf whose `Δ = 0` claim is impossible for
the value it can see, **per element**, under the law that leaf names (§5.4.1):

| law of the winning Δ | reject `Δ = 0` when the value is |
|---|---|
| absent / `dither-relative/v0` | finite and **nonzero** |
| `exact-cents/v0` | finite and **not an exact integer**, or `|v| > 2^53` |

This is checkable **without trusting anyone**, because the verifier holds its own copy
of the data. Under relative dither, Δ is `scale · 2^(1-bits)` and is zero only when
the column's scale is zero — every finite value was `0`. Under exact-cents, Δ = 0 is
the normal case, but only for values that really are whole cents inside f64's
exact-integer range; a fractional or oversized value refutes the claim whoever signed
it. A genuinely all-zero column still verifies, honestly, at width `0`, under either
law.

**Per element, not per column (MUST).** One benign value must never license the rest.
A check that looked at the first row, or at any aggregate, would let a single genuine
zero in a column carry an exactness claim across every other row — the element-wise
guard lesson, applied here before a red team had to find it.

An unrecognised law never reaches this check: §5.4 has already made its column
unusable, failing the anchor closed one step earlier with `anchor-delta-unusable`.

> **Why the dispatch exists at all.** The relative-dither invariant was, for a while,
> the *only* one, and it was written as though it were a property of certificates
> rather than of one quantizer. The exact-cents law walks straight through it: Δ = 0
> over nonzero data is exactly what that law produces. Had the law not been bound
> into the signed leaf, naming `exact-cents/v0` over ordinary fractional data would
> have reinstated the zero-width anchored bound this guard was written to kill.

---

## 8. Deterministic kernel — numeric semantics

Without this section frozen, a second implementation cannot be *built from the
spec*; it can only copy the reference implementation's behaviour, which would make
the format a description of one program rather than a specification
([PLATFORM.md](PLATFORM.md) Risks, item 2).

### 8.1 The substrate identifier

`kernel` is the substrate that produced the envelope:

- `alelyon-vector/<version>` — the deterministic native kernel. **The only
  specified substrate.** Reductions are fixed-order and compensated, hence
  bit-identical across runs, thread counts, machines and OSes.
- `numpy-fallback` — the portable path. **Not a specified substrate.**

A verifier compares its own substrate identifier to the envelope's **as strings**.
Equal ⇒ the width is compared bit-for-bit. Different ⇒ the width is left
**unverified** (`checks.width = null`, reason class `substrate-mismatch`) and the
scalar is verified to a relative tolerance of `1e-9`; this is an honest partial,
never a silent accept or a false reject.

**Exception — a width of exactly zero is substrate-independent (MUST).** When every
committed Δ is `0` the resample perturbation is `uniform(...) · 0`, identically zero,
so every pivot is exactly `0` and the order statistic is exactly `0.0` on any kernel,
with no near-cancellation anywhere. A verifier MUST therefore check such a width
exactly even on a mismatched substrate, reporting the advisory class
`width-substrate-independent`. All **three** of the committed Δ, the stated width, and
the replayed width must be exactly `0.0` — this is a triviality check, not a
relaxation, and the SCALAR is still only compared to tolerance.

This is what makes the exact-cents law (§5.4.1) deployable: a receipt whose storage
error is genuinely zero verifies to `ok=true` on a bare `pip install`, with no
deterministic kernel required.

**The SCALAR's off-substrate comparison (MUST).** Off-substrate the scalar is compared to
a relative tolerance of `1e-9`. A relative window on a large figure is a large *absolute*
window — on a control total of 156,588,400,000 cents it is ±157 cents, i.e. ±$1.57 — so
two rules apply:

1. **Integral figures MUST compare exactly, on any substrate.** A sum of integers is
   exact in f64 up to `2^53` under every implementation (neither Neumaier nor pairwise
   summation rounds it), so when the replayed and stated values are both integral and in
   range, any difference is a real difference and no tolerance is warranted. This is not
   an optimisation: without it, editing a $1.5bn integer-cent control total by one dollar
   VERIFIES on a fallback verifier.
2. **When the tolerance path is taken, the absolute window MUST be reported** (advisory
   class `scalar-tolerance-window`). A verdict whose sensitivity is undisclosed cannot be
   weighed, and concealing it is the same defect class as a blank budget slot.

The general non-integral case keeps its tolerance. An ill-conditioned smooth-tier program
legitimately differs by more than a few ULPs across kernels, and a false reject on a
genuine receipt is its own failure; a conditioning-aware bound is a recorded v1 candidate
(§10.4) rather than something to approximate here.

> **A measured warning, not a theoretical one.** Two machines both reporting
> `numpy-fallback` are not guaranteed to agree: `np.sum` is pairwise summation
> whose association order can vary with build and SIMD width. Measured in this
> repo, the Neumaier kernel and `np.sum` disagree on **146 of 200** random
> 500-element reductions, worst absolute difference `1.1e-08`. An implementation
> MUST therefore report reason class `unspecified-substrate` when a width matched
> on a non-specified substrate, and MUST NOT describe such a match as portable.
> **Consequence for deployment:** an envelope intended for third-party
> verification MUST be issued on `alelyon-vector/<version>`, and that kernel must
> be installable by the verifying party, or `ok=true` is unreachable for them.

### 8.2 Compensated summation (MUST, exactly)

Neumaier summation in fixed sequential order. No parallelism, no reassociation, no
FMA contraction:

```
sum = 0.0;  comp = 0.0
for x in xs:                       # source order, no sorting, no blocking
    t = sum + x
    if |sum| >= |x|:  comp += (sum - t) + x
    else:             comp += (x - t) + sum
    sum = t
return sum + comp
```

Empty input returns `0.0`.

### 8.3 Mean, variance, correlation (MUST)

Two-pass, compensated, using §8.2 for every reduction:

```
det_mean_var(xs):
    n = len(xs)
    if n == 0:  return (NaN, NaN)
    mean = neumaier(xs) / n
    if n == 1:  return (mean, NaN)
    var  = neumaier([ (v-mean)*(v-mean) for v in xs ]) / (n - 1)     # ddof = 1
    return (mean, var)

det_corr(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:  return NaN
    mx = neumaier(xs)/n;  my = neumaier(ys)/n
    cx = [x-mx …];  cy = [y-my …]
    sxy = neumaier([cx_i*cy_i …]);  sxx = neumaier([cx_i^2 …]);  syy = neumaier([cy_i^2 …])
    denom = sqrt(sxx * syy)
    return NaN if denom == 0 or not finite(denom) else sxy/denom
```

`std` is `sqrt(var)`, and NaN when `var` is NaN.

### 8.4 Dither generation from the committed seed (MUST)

Both the capture quantizer and the DRC resampler draw dither. The draw is
deterministic in `(seed, n, delta)` so encoder and decoder agree without storing
the draw.

For v0, `seed` and each resample index `k` are unsigned 64-bit integers. The
producer currently draws seeds from `[0, 2^63)`. A verifier MUST reject a seed
outside the unsigned-64 range rather than truncate or wrap it.

**Capture dither** (`certkit._draw_dither`):

```
rng = PCG64(SeedSequence.scalar(seed))
u   = rng.uniform(-0.5*delta, +0.5*delta, size=n)
```

**Quantize / reconstruct:**

```
q_i = round_half_to_even( (x_i + u_i) / delta )      # non-finite x_i -> level 0
y_i = q_i * delta - u_i
```

Rounding is **round-half-to-even**, matching `np.round` exactly including at exact
ties. The float→int64 cast **saturates**; it MUST NOT wrap.

**DRC resample dither** (`execcert.certified_run`), per resample `k`:

```
rng  = PCG64(SeedSequence.sequence([seed, k]))
pert = rng.uniform(-0.5, +0.5, size=n) * delta_effective
```

The distinction is load-bearing: capture seeds with a scalar, the resampler seeds
with the two-element sequence `[seed, k]`, and `uniform(-0.5, 0.5) * Δ` is drawn in
that order — not `uniform(-Δ/2, Δ/2)`. These produce different streams.

#### 8.4.1 Frozen SeedSequence and PCG64 mechanics

The names above do not delegate semantics to an installed NumPy version. The v0
stream is frozen here. All `u32`/`u128` operations below wrap modulo `2^32`/`2^128`.
Expand each unsigned-64 entropy integer into the shortest non-empty sequence of
little-endian `u32` words (`0 -> [0]`); concatenate expansions for sequence entropy.
The pool has four `u32` slots. Define:

```text
INIT_A=0x43b0d7e5  MULT_A=0x931e8875  INIT_B=0x8b51f9dd
MULT_B=0x58f38ded  MIX_L=0xca01f9dd  MIX_R=0x4973f715

hashmix(x, h): x ^= h; h *= MULT_A; x *= h; return x ^ (x >> 16), h
mix(a, b):     x = MIX_L*a - MIX_R*b; return x ^ (x >> 16)
```

Starting with `h=INIT_A`, fill pool slot `i=0..3` with `hashmix(entropy[i]
or 0, h)`. Then, for each source slot in order, snapshot its current value and mix
a fresh `hashmix(source, h)` into every *other* destination slot in order. Finally,
for every entropy word beyond the first four, mix a fresh `hashmix(word, h)` into
all four destinations in order. To generate state words, set `h=INIT_B`; for output
index `i`, take `x=pool[i mod 4]`, then `x ^= h; h *= MULT_B; x *= h;
x ^= x >> 16`. Adjacent `u32` outputs form `u64` as `lo | (hi << 32)`.

PCG64 is XSL-RR 128/64 with multiplier
`0x2360ed051fc65da44385df649fccf645`. Generate four SeedSequence `u64` words,
set `initial_state=(w0<<64)|w1`, `stream=(w2<<64)|w3`, and
`increment=(stream<<1)|1`. Initialize `state=0`; advance once; add
`initial_state`; advance once. Each raw output advances first, then returns
`rotate_right64(high64(state) XOR low64(state), state >> 122)`. A random double is
`(raw >> 11) * 2^-53`. `uniform(low, high)` performs the two IEEE-754 binary64
operations `(high-low) * random`, then `low + scaled`; fused multiply-add is
forbidden.

One `[seed,k]` generator is shared across every input in signed input order for that
resample. It MUST NOT be restarted per input. These frozen fixtures are normative:

| entropy / output | first values |
|---|---|
| scalar `42`, SeedSequence `u32` | `3444837047, 2669555309, 2046530742, 3581440988` |
| scalar `42`, PCG64 raw `u64` | `14276969152011380360, 8095878257575067585` |
| sequence `[7,1]`, SeedSequence `u32` | `369571992, 1544939151, 3514839603, 1195211032` |
| sequence `[7,1]`, random-double bits | `0x3fe8a4fea2bbd92b, 0x3fbca7438de03818` |

An implementation that disagrees with any fixture does not implement the v0
specified substrate, even if its library also calls itself PCG64.

### 8.5 The DRC certificate (MUST)

Given a program, per-input series and per-row Δ, `K` (default 63), `alpha`
(default 0.05), `strict`, and a `seed`:

1. Parse; prune the program to the statement producing the **final** output plus
   the let-bindings it transitively needs. A branch site that cannot affect the
   certified scalar is not classified, executed, or guarded.
2. Classify statically into a tier (§8.6). In `strict` mode a hard branch op
   refuses.
3. Refuse if the program reads no certified data; if `alpha` is not in `(0,1)`;
   if `K > 10_000`; if any complete-program source/token/AST/depth limit is
   exceeded; if the pruned aggregate work `(K + 1) * R * A` is greater than
   `50,000,000`; or if `m = ceil((1-alpha)(K+1))` exceeds `K`.
4. In `strict` mode, refuse if **any** consumed row is uncertified. In non-strict
   mode, replace each uncertified row's Δ with the **maximum finite Δ that input
   carries** — never zero, which would collapse the width — and refuse outright if
   an input is *fully* uncertified, since there is then no Δ to stand in.
5. Execute once on the stored values → `base_value`. Refuse if execution fails or
   the scalar is not finite.
6. For `k` in `0 … K-1`: perturb every input per §8.4, execute, and record
   `D_k = |f(perturbed) − base_value|`. Refuse if any resample fails.
7. **Branch-guarded tiers only.** Execute twice more, with every element of every
   input shifted by `+Δ/2` and then by `−Δ/2` — the corners the declaration
   permits. Refuse if either fails. These runs update the branch decision
   signature and `perturb_scale` (§7.5); they MUST NOT contribute to `D`, because
   the width is a conformal order statistic over the dither law of §8.4 and a
   worst-case corner is not a draw from it. A verifier that skips this step
   computes a smaller `perturb_scale` and admits a decision a systematic rounding
   can invert.
8. `width = sorted(D)[m-1]`; `level = m/(K+1)`.
9. `exact` is true iff no row was uncertified **and** the tier is `linear-exact`
   or `branch-stable-exact`.

The final scalar is extracted as: take the last output; if it is a series, drop
NaNs and take the last remaining element (or nothing if empty); coerce to f64;
reject if not finite.

### 8.6 Tiers and profiles

| Tier | Ops | Level |
|---|---|---|
| `linear-exact` | `price`, `series`, `diff`, `lag`, `sma`, `ema`, `rolling_mean`, `sum`, `mean`, `last`, `first`, `count`, and `+` `-` and numeric-literal scaling | **EXACT** — exchangeability is a theorem |
| `smooth-first-order` | adds `returns`, `logret`, `sqrt`, `zscore`, `std`, `rolling_std`, `corr`, `abs`, `clip`, and `*` `/` `^` | approximate, harness-validated |
| `branch-stable-exact` / `branch-stable-first-order` | `min`, `max`, `sign`, `where`, numeric comparisons — un-refused only by the margin guards | EXACT under Δ-separation; else first-order |
| `branch-sensitive` | `rsi`, `%`, unknown ops, truthiness branches on non-comparisons | **refused** in strict mode |

**Profile 1 (frozen here).** The `linear-exact` tier. A conforming second
implementation MUST reproduce it exactly. Its source grammar uses only the six ASCII
whitespace characters (space, tab, line feed, vertical tab, form feed, and carriage
return); line feed separates statements while the other five are insignificant
spacing. A semicolon also separates statements. Decimal digits are `0` through `9`;
and identifiers match
`[A-Za-z_][A-Za-z0-9_]*`. The words `let`, `signal`,
`show`, `when`, `and`, `or`, and `not` are reserved and MUST NOT be accepted as
identifiers. String literals MUST be double-quoted JSON strings decoded to Unicode
scalar values; single-quoted strings and unpaired UTF-16 surrogate escapes MUST be
rejected. A valid JSON surrogate pair decodes to its single scalar value.

A bare expression is equivalent to `show EXPR`; unary `+` is an identity operator.
`let NAME = EXPR` and `signal NAME when EXPR` both bind `NAME`. The last
output-producing statement, whether `show`, bare output, or `signal`, supplies the
certified output, and pruning retains only that statement plus the bindings it
transitively references. An unused earlier `signal` therefore has no effect. Its
reducer semantics:

| op | semantics |
|---|---|
| `sum` | drop NaN, then §8.2. All-NaN ⇒ `0.0` |
| `mean` | drop NaN, then §8.3 mean. All-NaN ⇒ NaN |
| `count` | number of non-NaN elements, as f64 |
| `last` / `first` | drop NaN, then last / first element; empty ⇒ NaN |
| `diff(s, n=1)` | `s − shift(s, n)`, index-aligned, first `n` rows NaN |
| `lag(s, n=1)` | `shift(s, n)` |
| `sma(s, w)` / `rolling_mean(s, w)` | trailing window of `w`, NaN until `w` observations |
| `ema(s, w)` | exponentially weighted mean, `span = max(w,1)`, `adjust=false` |

**Binary operators align on the index**, not on position: `a + b` over two series
produces the union of their indices with NaN where either is missing. An
implementation that zips positionally will diverge on any pair of unequal indices.

**Profile 2 (NOT frozen).** Every other tier. Their semantics are those of the
reference implementation's `rolling`/`ewm`/`pct_change`, which this document does
not pin down. A second implementation MAY decline Profile 2 entirely; declining it
is conformant, and claiming to implement it without a frozen definition is not.

---

## 9. The verify algorithm

### 9.1 Check slots

Eleven slots, each `true`, `false`, or `null` (= not performed / honestly unknown).

| slot | meaning |
|---|---|
| `authenticity` | signed by the **pinned** key |
| `program` | `program_hash` matches the carried `program` |
| `inputs` | caller's data matches every committed digest, and each block's bookkeeping is self-consistent |
| `scalar` | the value was **re-derived** from the data and matched |
| `tier` | the replayed tier matches |
| `budget` | level, exactness flag, tier, assumptions and branch sites all matched |
| `width` | the bound re-derived bit-for-bit (`null` on substrate mismatch) |
| `transparency` | committed Δ proven against the signed capture log |
| `witness` | co-signature valid under the pinned witness key |
| `key_status` | the signing key was in service in a manifest chained to a pinned root and proven non-regressing against the verifier's retained signed checkpoint (§6.8) |
| `provider` | the error budget's provider block carries corroboration anchors, and every carried attempt record was proven to be in the signed probe log and re-derives the stated summary (§9.6) |

`transparency` and `witness` use `null` for a genuine partial: *some* inputs
anchored or co-signed. That is honest, not a failure.

### 9.2 `ok` (MUST, exactly)

```
performed = [ v for v in checks.values() if v is not null ]
replayed  = checks.scalar is not null
bound_ok  = envelope.refused == true  or  checks.width is true
ok        = checks.authenticity is true  and  replayed
                                         and  all(performed)
                                         and  bound_ok
```

Consequences that MUST hold:

- **A pinned public key is mandatory.** Without it, `authenticity` is `false` and
  `ok` is `false`. Verifying against the key embedded in the same untrusted
  envelope authenticates nothing.
- **Without the caller's data, `ok` is false.** The number was not re-derived.
- **A substrate mismatch yields `ok=false`**, because `checks.width` is `null` and
  `bound_ok` fails. This is deliberate: an unverified bound must not present as a
  verified one.
- **A present-but-invalid transparency anchor yields `ok=false`.** A missing anchor
  (`null`) does not.
- **A present-but-invalid co-signature under a pinned witness key yields
  `ok=false`.** Present-but-unpinned (`null`) does not.
- **An unknown envelope `type` yields `ok=false`** with reason class
  `not-a-cne-v0`, immediately, before any other work.
- **A supplied key manifest that does not place the signing key in service yields
  `ok=false`** — revoked, absent, or outside its validity window. Not supplying one
  leaves `key_status=null` and does not fail. Supplying one without its pinned root,
  candidate signed checkpoint, separately pinned checkpoint key, and retained signed
  checkpoint is refused rather than half-performed; a stale or equivocal checkpoint
  also yields `ok=false`.

### 9.3 `width_trust`

```
refused                       -> "refusal"
checks.width is null          -> "unverified"
checks.transparency is true   -> "transparency-anchored"
checks.width is true          -> "authenticated"
otherwise                     -> "unverified"
```

`authenticated` means a trusted signature vouches for the Δ. `transparency-anchored`
means the Δ were re-derived from signed capture leaves whose Merkle inclusion was
proven under the pinned key. Neither label extends to the **sampling** term, which
the verifier **carries and does not recompute**. The provider slot carries its own
label, `provider_trust`, on the identical scale — see §9.6.

### 9.4 Reason classes (frozen vocabulary)

Every reason carries a stable class. The class is part of the wire contract; the
prose is not. Conformance is judged on the **set** of classes, sorted and
de-duplicated, because discovery order depends on input iteration order.

The vocabulary is exhaustive and closed. A verifier MUST NOT emit a class outside
it; adding one is a spec change. The block below is the normative list, one class
per line, machine-readable on purpose — CI parses it and fails if it disagrees with
the implementation's own declared vocabulary, so this document and the code cannot
drift apart.

```reason-classes
# authenticity and structure
not-a-cne-v0
no-pinned-key
malformed-pinned-key
unsigned
key-id-mismatch
bad-signature
program-hash-mismatch
malformed-envelope
# inputs
no-input-data
input-missing
input-digest-mismatch
delta-count-mismatch
uncertified-count-mismatch
# replay
replay-refusal-mismatch
scalar-mismatch
tier-mismatch
budget-mismatch
no-seed
width-mismatch
substrate-mismatch
unspecified-substrate
width-substrate-independent
scalar-tolerance-window
# transparency anchor
transparency-no-pinned-key
transparency-partial
anchor-sth-invalid
anchor-malformed-scope
anchor-scope-mismatch
anchor-sth-scope-mismatch
anchor-malformed-leaf
anchor-leaf-hash-mismatch
anchor-proof-tree-size-mismatch
anchor-proof-index-out-of-range
anchor-inclusion-failed
anchor-delta-unusable
anchor-no-data
anchor-length-mismatch
anchor-row-uncovered
anchor-delta-mismatch
anchor-delta-zero-implausible
# witness
witness-unpinned
witness-cosignature-invalid
witness-partial
witness-malformed
# key lifecycle
key-manifest-unrooted
key-manifest-invalid
key-manifest-checkpoint-required
key-manifest-checkpoint-invalid
key-manifest-checkpoint-not-monotonic
key-not-in-manifest
key-revoked
key-outside-validity
# provider corroboration anchor (W3)
provider-no-pinned-key
provider-partial
provider-sth-invalid
provider-malformed-scope
provider-scope-mismatch
provider-sth-scope-mismatch
provider-malformed-leaf
provider-leaf-hash-mismatch
provider-proof-tree-size-mismatch
provider-proof-index-out-of-range
provider-inclusion-failed
provider-attempt-count-mismatch
provider-attempts-digest-mismatch
provider-outcome-unknown
provider-summary-mismatch
```

Six of these are **advisory**: they are reported without changing `ok`.

```advisory-reason-classes
unspecified-substrate
width-substrate-independent
scalar-tolerance-window
transparency-partial
witness-partial
witness-unpinned
```

### 9.5 Never crash

A verifier is handed untrusted input. It MUST return a verdict for every input,
including structurally absurd ones. A malformed pinned key returns
`malformed-pinned-key`, not an exception. Any exception escaping replay is caught
and reported as `malformed-envelope` with `ok=false`.

---

### 9.6 The provider corroboration anchor (W3)

`error_budget.provider` states what cross-source evidence exists about the inputs.
An issuer MAY make that statement checkable by attaching an `anchors` object, keyed
by input reference:

```json
"anchors": {
  "price|AAPL@1d": {
    "table": "corroboration",
    "scope": ["AAPL", "1d"],
    "sth": { "...": "alelyon.sth/v0 over the corroboration scope" },
    "leaves": [ { "seq": 1, "value_digest": "...", "n": 2, "...": "",
                  "inclusion_proof": { "...": "" } } ],
    "attempts": { "1": [["yahoo", "nasdaq", "answered", 101.5],
                        ["stooq", "nasdaq", "unavailable", null]] }
  }
}
```

Each attempt is `[provider, origin, outcome, value]`; `value` is `null` unless the
outcome is `answered`. `outcome` MUST be one of the closed vocabulary
`answered | unavailable | quality-rejected | error`; anything else is a malformed
record (`provider-outcome-unknown`), not a new outcome.

A verifier presented with `anchors` MUST, per anchor:

1. verify the STH under the **pinned** key (`provider-sth-invalid`);
2. **derive** the scope `(scope1, scope2)` from the input reference and require the
   block's and the STH's scope to equal it (`provider-scope-mismatch`,
   `provider-sth-scope-mismatch`). The scope MUST NOT be read from the block: a
   scope the envelope chooses lets a signer anchor a claim about one input to a
   probe log where the sources happened to answer. A reference the verifier cannot
   scope fails closed;
3. recompute each `cert_leaf_hash` from the carried record and require it to equal
   the proof's `leaf_hash` (`provider-leaf-hash-mismatch`), then bind the proof's
   `tree_size` and `index` to the signed head and verify the audit path
   (`provider-proof-tree-size-mismatch`, `provider-proof-index-out-of-range`,
   `provider-inclusion-failed`);
4. require carried attempt records for every leaf (`provider-malformed-leaf`) — an
   absent record set is a refusal, never "nothing to check", or omission becomes the
   cheapest forgery;
5. require `len(attempts) == leaf.n` (`provider-attempt-count-mismatch`). This is
   what catches a **deleted silence**: the count is committed separately, so
   removing a row changes `n` before it changes anything else;
6. require `corroboration_digest(attempts) == leaf.value_digest`
   (`provider-attempts-digest-mismatch`). This is what catches a **rewritten
   outcome**, since the outcome is committed alongside the value; and
7. **re-derive** `{asked, answered, silent}` from the proven records and require any
   value the envelope states to equal it (`provider-summary-mismatch`). The summary
   MUST NOT be taken from the envelope: a summary checked against numbers the issuer
   also wrote is checked against itself.

`corroboration_digest` is BLAKE2b-256 over the attempts sorted by `provider`,
feeding `provider \x1f origin \x1f outcome \x1f` as UTF-8 followed by the value as
eight bytes little-endian IEEE-754 (`null` → NaN). `silent` is defined as
`asked - answered` and is never counted independently, so the three cannot be shown
disagreeing.

`provider_trust` is `transparency-anchored` when `checks.provider` is `true`, and
`signer-attested` in every other case including an envelope carrying no anchors.

**What this establishes, and what it does not.** It anchors *the record of having
asked*: which upstreams were probed and what each one did, non-repudiably and
checkable by a party that is not the issuer. With one answering origin it is **not**
a dispersion measurement and **not** a cross-source agreement claim, and no label
here may be presented as one. A present-but-invalid anchor fails `ok`; an absent one
does not.

## 10. Versioning and stability policy (W7)

### 10.1 v0 freezes at the first external verification

Until the [PLATFORM.md](PLATFORM.md) §5 milestone passes, `alelyon.cne/v0` is a
DRAFT and may change. **From that moment on**, within v0: no change to
canonicalization, to any digest or hash layout, to any hash construction, or to
the meaning of any existing member. The `cert_leaf_hash` byte format (§5.2) is
frozen already and independently, because changing it invalidates every stored
chain.

### 10.2 Additive changes

Unknown members are covered by the signature (§2.5 signs the whole object) and
MUST be **ignored semantically** by a v0 verifier. A member whose *absence* would
change a verdict is **not** additive and MUST NOT be added to v0.

Two vectors enforce this: an envelope carrying an unknown extra member still
verifies; a future-typed envelope is refused.

### 10.3 Anything else is `alelyon.cne/v1`

A v0-only verifier MUST refuse a v1 envelope with reason class `not-a-cne-v0` and
MUST NOT attempt to verify it under v0 rules. A v1 verifier supports both during a
published dual-issue window. The spec artifact, the wheel version, and the vector
suite are versioned **together**; each wheel release states exactly which envelope
types it verifies, and the declaration and the behaviour are checked against each
other in CI.

### 10.4 Recorded v1 candidates

Not decisions — a list so they are not rediscovered:

1. **Adopt RFC 8785 (JCS)** for the signing encoding, retiring the Python-repr
   dependency of §2.3. Cost: every existing signature becomes invalid.
2. **Unify the hash functions.** BLAKE2b for content, SHA-256 for the tree, is
   historical.
3. **Carry the capture law id explicitly** in every leaf, rather than defaulting
   an unmarked leaf to AUTO_BITS (§7.6).
4. **Make the substrate identifier precise** for the fallback path, so two
   non-specified substrates cannot appear to match (§8.1).
5. **A conditioning-aware scalar tolerance.** Integral figures now compare exactly, which
   closes the case the actuarial product depends on, but a non-integral figure of large
   magnitude still carries a proportionally large absolute window (§8.1). A bound derived
   from the committed row counts and the program's tier would be tighter without risking
   false rejects on ill-conditioned programs.

### 10.5 CI obligations

- The declared supported envelope types and the type string the verifier accepts
  MUST agree; a change to either without the other fails the build.
- A change to `ENVELOPE_TYPE` MUST land in the same commit as a spec update, a
  vector update, and a policy update.
- No verifier release may ship if any MUST-fail vector verifies or any golden
  vector fails.

---

## 11. What is proven, and what is not

Stated at exactly the verified strength, per [CLAIMS.md](CLAIMS.md).

**A holder of the envelope, the pinned public key, and their own copy of the
inputs can:**

- confirm the envelope is signed by the pinned key;
- confirm the data they hold is the data the number was computed over;
- **re-derive the scalar** by replay at ~1× cost;
- **re-derive the storage-quantization width**, bit-for-bit on the specified
  substrate;
- check the stated decomposition's tier, level and assumptions for consistency
  with the replay;
- when an anchor is present, confirm the committed Δ match signed capture leaves
  whose Merkle inclusion is proven under the pinned key;
- when a co-signature and a pinned witness key are present, confirm that witness
  signed the complete same STH, including its log signature and extension members;
- when a key manifest and both pinned lifecycle keys are present, reject regression
  relative to the verifier's retained signed checkpoint.

**They cannot, and no wording here may suggest otherwise:**

- **learn whether the data is true.** We detect revision, not invention. A producer
  who fabricates data at capture signs a receipt that verifies perfectly.
- **re-derive the sampling term.** It is carried, not recomputed.
- **confirm the provider slot.** It describes the issuer's deployment and is
  `signer-attested`.
- **conclude the Δ is maximal** over all leaves in the log (§5.5).
- **conclude the witness is independent.** That is a property of the deployment
  (§6.7).
- **conclude a bootstrap checkpoint is globally newest.** Monotonicity is relative
  to verifier-held state; the initial checkpoint needs an out-of-band channel (§6.8).
- **treat the width as a bound on anything but storage quantization.** Sampling,
  provider and model error are separate, separately named terms; at 24-bit capture
  sampling typically dominates by ~1e4.

Against validated numerics — Arb, INTLAB, IntervalArithmetic.jl, CAPD, Taylor
models — which give *guaranteed* enclosures over rounding, truncation and
discretization, this width is not a competitor and MUST NOT be described as tight.
It is a stated error decomposition, each term labeled by how it was obtained.

The attestation layer is a competent re-derivation in the SCITT / Sigstore-Rekor /
Sigsum / RFC-6962 family, not an invention.

**As of this document's date, no party outside the development machine has ever
verified a CNE.** The Apache-2.0 Python verifier is publicly available as
`alelyon-verify` 0.2.1, but no owner-approved public deterministic-kernel
distribution exists for replaying substrate-dependent nonzero widths. Nothing in
this specification may be read as a claim that third-party verification has
occurred.

---

## 12. Conformance vectors

The suite lives at `alelyon/verify/vectors/`, ships inside the wheel and in the
repo, and is modelled on RFC 7520's worked JOSE examples and
sigstore-conformance's must-pass / must-fail structure.

Each case is a single self-contained JSON document:

```json
{
  "slug": "golden-anchored",
  "must": "pass",
  "description": "…",
  "spec_version": "alelyon.cne-spec/0.3.0",
  "pins": {
    "public_key_hex": "<64 hex>",
    "witness_key_hex": null,
    "key_manifest": null,
    "manifest_root_hex": null,
    "manifest_checkpoint": null,
    "checkpoint_public_key_hex": null,
    "trusted_manifest_checkpoint": null
  },
  "inputs": {"price|SYN": {"index": [1704153600.0, …], "values": [100.1, …]}},
  "envelope": { /* a full CNE */ },
  "expect": {
    "ok": true,
    "checks": {"authenticity": true, "…": "…"},
    "width_trust": "transparency-anchored",
    "provider_trust": "transparency-anchored",
    "reason_classes": []
  }
}
```

`inputs` keys are `"<kind>|<key>"`. Index values are epoch seconds as JSON
numbers. `expect.checks` gives every slot including `null`s. Both trust labels are
claim-bearing conformance outputs: `provider_trust` MUST be compared even when no
provider anchor is present, in which case its value is `signer-attested`.

Because a golden vector's `width` reproduces only on the substrate that produced
it (§8.1), each case records the producing `kernel` in its envelope, and a runner
on a different substrate MUST expect the substrate-mismatch outcome rather than
the recorded one. The suite therefore carries, for cases whose verdict is
substrate-dependent, an `expect_off_substrate` block.

**Required goldens:** transparency-anchored · authenticated (no anchor) · signed
refusal · substrate-mismatch · witnessed · unknown-extra-member-still-verifies ·
keyed-table anchored by digest identity.

**Required MUST-fail forgeries**, each tracing to a defect this program shipped or
caught:

| slug | what it does |
|---|---|
| `delta-fake-zero` | explicit `"delta": 0.0` over nonzero data |
| `delta-omitted` | `delta` member removed at capture |
| `shrunk-delta-resigned` | Δ scaled down, width recomputed to match, re-signed under the real key |
| `tampered-scalar` | stated scalar edited |
| `tampered-cell` | one data cell differs from the committed digest |
| `stripped-seed` | dither seed removed |
| `mismatched-key-id` | `key_id` does not match the pinned key |
| `undercounted-uncertified` | `uncertified` understates the `null` Δ count |
| `anchor-scope-substitution` | anchor points at a different capture scope |
| `wrong-index-inclusion-proof` | valid path replayed at the wrong index |
| `proof-tree-size-mismatch` | proof's `tree_size` differs from the STH's |
| `cosignature-replay` | valid co-signature bound to a different root |
| `forgery-cosignature-key-role-collision` | the log key is reused as the witness key |
| `rollback-head` | witnessed head rewound |
| `same-size-fork` | two different roots at one `tree_size` |
| `forgery-key-manifest-checkpoint-missing` | manifest supplied without its signed checkpoint and retained rollback state |
| `forgery-key-manifest-checkpoint-role-collision` | a manifest signing key is reused as its checkpoint key |
| `forgery-key-manifest-rollback` | an older valid signed checkpoint replayed against newer retained state |
| `forgery-key-manifest-checkpoint-equivocation` | a different valid checkpoint at an already retained sequence |
| `future-version` | `type` is `alelyon.cne/v1` |
| `runs-undercover` | delta `runs` summing to fewer than `n` |
| `cents-law-over-fractional-data` | `law: exact-cents/v0` with Δ=0 over values that are not whole cents |
| `cents-law-one-bad-element` | the same, refuted by a single non-integral row among valid ones |
| `unknown-capture-law` | a `law` no implementation recognises, which must fail closed rather than default |
| `table-digest-not-captured` | a keyed table whose digest matches no signed capture batch |
| `tier-floor-refusal` | a smooth program under `require_tier: linear-exact` — a golden refusal, not a forgery |

Adding a newly confirmed forgery to the suite is a release-blocking event by
policy.

---

## 13. Open / proprietary boundary

**Apache-2.0 interoperability surface** — this specification, the envelope format,
the Python verifier source/package, and the conformance vectors. The base verifier
can check signatures, commitments, scalars, anchors, and forgery classes; replay of
a substrate-dependent nonzero width additionally requires the specified
deterministic kernel. No owner-approved public distribution of that kernel exists
as of this revision.

The in-repository Rust verifier is a private, unpublished second-language
implementation. It is not part of the public export: its Apache-2.0 crate has a
mandatory dependency on separately marked proprietary `vector_core`, and both are
`publish=false` pending an explicit licensing and distribution decision.

**Product** — the capture engine and its quantization, the analytics, the
DSL-authoring model, and the signing authority. Issuing trusted certificates at
scale is the business.

The reference producer (`build_envelope`, `signed_tree_head`, `inclusion_proof`)
ships in the open verifier for interop testing only; a commercial issuer runs it
behind its own key.

---

## 14. Sources

Repo, in the order this document consumes them:
`alelyon/verify/__init__.py` ·
`alelyon/runtime/oracle/dsl/envelope.py` ·
`alelyon/runtime/oracle/dsl/verify.py` ·
`alelyon/runtime/oracle/dsl/execcert.py` ·
`alelyon/runtime/oracle/dsl/interpreter.py` ·
`alelyon/runtime/atlas/data/attest.py` ·
`alelyon/runtime/atlas/data/history.py` ·
`alelyon/runtime/atlas/data/certify.py` ·
`alelyon/runtime/vector/codec/certkit.py` ·
`alelyon/runtime/vector/native.py` ·
`alelyon/languages/vector_native/src/lib.rs` ·
`alelyon/languages/vector_core/src/lib.rs` ·
`alelyon/languages/cne_verify/src/` ·
`tools/cne_mutation_parity.py` ·
`docs/cne/CLAIMS.md` · `docs/cne/PLATFORM.md` · `docs/RESEARCH.md`

External: [RFC 6962](https://www.rfc-editor.org/rfc/rfc6962) ·
[RFC 8032](https://www.rfc-editor.org/rfc/rfc8032) ·
[RFC 8259](https://www.rfc-editor.org/rfc/rfc8259) ·
[RFC 8785 (JCS)](https://www.rfc-editor.org/rfc/rfc8785) ·
[RFC 7520](https://www.rfc-editor.org/rfc/rfc7520) ·
[sigstore-conformance](https://github.com/sigstore/sigstore-conformance) ·
[C2SP tlog-checkpoint](https://github.com/C2SP/C2SP/blob/main/tlog-checkpoint.md) ·
[C2SP tlog-cosignature](https://github.com/C2SP/C2SP/blob/main/tlog-cosignature.md)

## Referee record

Findings from writing this document against the code, each of which changed the
program's understanding rather than only its prose:

- **The signing encoding depends on Python's `repr` for floats** (§2.3). This was
  nowhere recorded. A Rust verifier built from a naive reading of "canonical JSON"
  would fail every signature, and would fail it *silently* — a signature mismatch
  names no cause. Boundaries measured and tabulated; JCS recorded as a v1
  candidate with its migration cost stated.
- **`numpy-fallback` is not a substrate, it is a name** (§8.1). Measured: the
  Neumaier kernel and `np.sum` disagree on 146/200 random 500-element reductions.
  Two machines both reporting `numpy-fallback` were being treated as an exact
  substrate match, so a width could verify bit-for-bit by luck rather than by
  specification. A new advisory reason class `unspecified-substrate` now says so
  without changing `ok`.
- **W1's acceptance test and the W8 milestone are unreachable as written unless
  the native kernel is installable by the verifying party.** A substrate mismatch
  makes `checks.width = null`, which makes `bound_ok` false, which makes
  `ok=false`. [PLATFORM.md](PLATFORM.md) W1(b) says the Rust extension is
  "optional" and "a certificate never depends on the extension being installed" —
  true of *issuing*, false of *verifying a width to `ok=true`*. Either the
  deterministic kernel ships as a binary wheel across the platform matrix, or the
  milestone's genuine envelope must be issued on the fallback, whose portability
  the point above disproves. Recorded here; it needs a PLATFORM.md amendment, not
  a spec workaround.
- **Reproducing NumPy's PCG64/SeedSequence is a hard prerequisite for W4** (§8.4),
  and the capture path and the DRC resampler seed *differently* (scalar seed vs
  the two-element sequence `[seed, k]`, and `uniform(-0.5,0.5)*Δ` rather than
  `uniform(-Δ/2, Δ/2)`). A second implementation that missed either detail would
  produce a plausible but wrong width.
- **The input digest does not bind the series identity** (§3.2) — only timestamps
  and values. `kind` and `key` are bound by the envelope signature instead. Worth
  stating, because "the digest identifies the input" is the natural misreading.
- **The Δ=0 plausibility invariant is quantizer-specific** (§7.6). It is sound for
  the AUTO_BITS relative law only. This is the exact obstacle the insurance
  product's integer-cent capture law must clear, and the spec now names the
  requirement — law id bound into the signed leaf, per-law dispatch, unknown law
  fails closed — rather than leaving it to be discovered during implementation.
- **A spec claiming to cover the whole DSL would be overclaiming**, so §8.6 splits
  Profile 1 (frozen) from Profile 2 (reference behaviour, not frozen). The smooth
  and branch tiers inherit pandas `rolling`/`ewm`/`pct_change` semantics that no
  prose here pins down, and the product mandate is linear-exact only.
- **Anchoring proves matching, not maximality** (§5.5) — carried through from
  [PLATFORM.md](PLATFORM.md) ledger item 4 into normative text, with an explicit
  MUST NOT on describing it as maximality.
