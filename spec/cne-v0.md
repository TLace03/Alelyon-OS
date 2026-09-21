# Certified Number Envelope v0

Specification identifier: `alelyon.cne-spec/0.3.0`. Envelope type:
`alelyon.cne/v0`. Python verifier package version: `0.3.0`.

This document describes the current wire contract implemented by
`alelyon.runtime.oracle.dsl.envelope`, `alelyon.runtime.oracle.dsl.verify`,
`alelyon.runtime.atlas.data.attest`, and `alelyon.runtime.atlas.data.keylife`.
It was reconstructed from those implementations and their conformance tests on
2026-09-06. This documentation revision changes no protocol identifiers, numeric
algorithms, reason classes, or acceptance rules.

## 0. Scope and conformance

A CNE binds a restricted computation, input commitments, replay parameters, and
storage-quantization result. The verifier consumes a reader's input data and
separately trusted public keys. It returns individual check results and a combined
verdict. A signature alone does not establish the truth of the inputs, the
appropriateness of the program, or an all-source bound on its answer.

An implementation claiming this specification must run the bundled language-neutral
vectors, preserve failed and unperformed checks, and identify its actual numeric
substrate. An unsupported profile or unavailable substrate is a reported limitation;
it must not become a successful full verification. The Python runner is
`alelyon.verify.conformance`; Rust replay and wire checks live under
`alelyon/languages/cne_verify/src/`.

## 1. Values and representations

Wire objects are JSON. Times in envelope metadata are finite epoch-second numbers.
Cryptographic keys are lowercase hexadecimal encodings of 32 raw Ed25519 bytes;
signatures encode 64 bytes. SHA-256 and BLAKE2b-256 digests encode 32 bytes.
An integer parameter excludes a JSON boolean. Unknown signed object members remain
part of the signature; an extension cannot replace the meaning of an existing member.

Notation below uses `LE64` for little-endian unsigned 64-bit integers, `LEi64` for
signed integers, `LEf64` for IEEE-754 binary64, and `UTF8` for unnormalised UTF-8.
Hash descriptions concatenate bytes in the stated order.

## 2. Canonical signing bytes

`attest.canonical` recursively sorts string object keys, preserves array order,
uses comma and colon separators without whitespace, preserves non-ASCII text, and
encodes UTF-8. It rejects non-string mapping keys and non-finite numeric values.
The reference serialization is Python's
`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)` after recursive validation.

Numeric spelling is part of the signature. Preserve integer versus floating-point
representation: `1` and `1.0` are different signed bytes. Binary64 output uses the
reference shortest round-trip representation, including `.0` for integral floats
in fixed notation, `-0.0`, lowercase `e`, an explicit exponent sign, and at least
two exponent digits. For floats, fixed notation covers decimal exponents from
`-4` through `15`; smaller or larger exponents use scientific notation. A generic
JSON serializer that emits `1` for `1.0` is insufficient. Exercise the canonical
serialization vectors rather than relying on a language's defaults.

Escape quotes, backslashes, and control characters using the reference JSON
encoding; do not escape ordinary non-ASCII characters or normalize Unicode.
For an envelope, STH, succession, revocation, checkpoint, or co-signature, sign
the canonical object with its `signature` member removed. Keep every other
member, including embedded public keys and extension fields, in the signed body.

`key_id = "ed25519:" + hex(BLAKE2b-64(raw_public_key))`.
An embedded key is descriptive. Authentication requires a pinned key obtained
outside the untrusted object.

## 3. Input commitments

### 3.1 Time series

Resolve duplicate indexes by keeping the last occurrence, then sort ascending.
Digest and replay use this same canonical form. For `price` and `series`, compute
BLAKE2b-256 over each row's `LEf64(epoch_seconds) || LEf64(value)` in order.
The digest commits the stored numeric data, not its provenance or correctness.

The Python API accepts inputs keyed by `(kind, key)`; bare `key` is a compatibility
lookup. CLI/vector inputs use `kind|key`, with a bare reference interpreted as
`price`. A time-series JSON value has `index` epoch seconds and aligned `values`.
JSON null values become NaN in the supplied numeric series; non-finite JSON
constants themselves are rejected by the CLI loader.

### 3.2 Keyed tables

`table` inputs use a `<dataset>|<column>` key. Row keys must be nonempty strings
whose UTF-8 encoding occupies at most 4096 bytes. Do not normalize Unicode or
coerce non-string row identities. Keep the last duplicate key and sort by Unicode
code-point order. Hash each canonical row as:

```text
LE64(byte_length(UTF8(row_key))) || UTF8(row_key) || LEf64(value)
```

The complete input digest is BLAKE2b-256 over the concatenated rows. The CLI/vector
representation is `{ "keys": [...], "values": [...] }`. A table has no time axis;
its capture coverage is checked by matching its complete keyed-row digest.

## 4. Per-row delta commitments

An input's `deltas` object encodes the storage step for each canonical row. A null
element denotes an uncertified row, not a zero step. The decoder accepts exactly
one of these representations:

| Representation | Shape and constraints |
|---|---|
| Constant | `{"const": d, "n": n, "uncertified_idx": [...]}`; `d` is finite and nonnegative, and listed integer indexes replace that position with null. |
| Runs | `{"runs": [[d_or_null, count], ...], "n": n}`; counts are nonnegative integers whose sum equals `n`. |
| Explicit | `{"list": [d_or_null, ...]}`; an optional `n` must equal the list length. |

`n` is a nonnegative integer and decoding is capped at 10,000,000 rows. Reject
numeric strings, booleans, negative or non-finite deltas, conflicting encodings,
invalid indexes, and count/length disagreement. The constant form requires an
index list; an empty list means no uncertified positions.

Verification requires decoded length to equal canonical data length. Derive the
uncertified count from decoded nulls; when `uncertified` is supplied, compare it
with that count. Never let the signer's count override the decoded evidence.

## 5. Capture records

### 5.1 Captured data and scope

An input anchor's scope is derived from its input identity:

| Input | Capture table | Scope | Column |
|---|---|---|---|
| `price`, key `T` | `bars` | `[uppercase(T), "1d"]` | `close` |
| `series`, key `S` | `series` | `["fred", uppercase(S)]` | `value` |
| `table`, key `D|C` | `table` | `[uppercase(D), lowercase(C)]` | `lowercase(C)` |

An unknown input kind or malformed table reference has no assumed capture scope.
The anchor and its signed tree head must identify this derived scope. A valid
signature over another scope is insufficient.

### 5.2 Frozen certificate leaf hash

`cert_leaf_hash` computes BLAKE2b-256 over these UTF-8 strings joined by the single
byte `0x1f`, with no trailing separator:

```text
table, scope1, scope2, decimal_integer(seq), value_digest,
decimal_integer(n), repr_binary64(lo_ts), repr_binary64(hi_ts),
decimal_integer(bits), payload, prev_hash
```

`payload` is the exact stored JSON string, not a reserialized object. Float
representations follow the reference Python `repr(float(...))`. The leaf commits
both the data digest and delta payload together with its prior-chain hash.

### 5.3 Delta payloads and exact row membership

Numeric payload entries name a `column`, numeric `delta`, and optional `law`.
Usable deltas are finite, nonnegative JSON numbers. Supported laws are an absent
law/null, `dither-relative/v0`, and `exact-cents/v0`. Unknown laws are unusable.
Duplicate usable column entries select the largest delta and its associated law;
any unusable entry for that column makes the column unusable overall.

Timestamp membership is a separate payload entry. Its `membership` object uses
`encoding` and `rows`; bars use `i64-epoch-seconds/v0`, series use
`f64-epoch-seconds/v0`. Membership must be unique, correctly typed, in canonical
order, and agree with the leaf's row count and timestamp endpoints. A min/max
interval alone never establishes that an intermediate row was captured.

#### 5.3.1 Uncertified capture outcome

An uncertified-capture payload contains null delta entries for the affected
columns, exact row membership, and a single `capture_outcome` object with schema
`alelyon.capture-outcome/v0`, `reason`, and `columns`. Its closed reason set is:

```text
certification-disabled
certification-declined
certification-not-supplied
certificate-rejected
certificate-write-failed
```

Such a leaf records the absence of a certificate. It cannot anchor a width and
produces `capture-uncertified-leaf`. Do not turn an omitted delta into zero or
erase a failed capture because the data write remained available.

### 5.4 Attributing a delta

Verify each covering leaf's hash, inclusion proof, signed head, scope, membership,
and current-value commitment before using its delta. For each time-series row,
take the largest usable delta among proven leaves that still cover that row's
current value. Require exact equality with the envelope's committed delta.
For a keyed table, select among proven leaves whose complete data digest equals
the reader's table digest; timestamps are irrelevant to table coverage.

Under relative dither, a zero delta cannot cover a finite nonzero value. Under
`exact-cents/v0`, a finite value covered by zero delta must be an integral cent
count with absolute value at most `2^53`. Nonzero delta does not bypass the
scope, membership, or value-identity checks.

## 6. Transparency, witnesses, and keys

### 6.1 Merkle tree

For a certificate leaf digest `h`, the Merkle leaf is `SHA256(0x00 || raw(h))`.
An internal node is `SHA256(0x01 || left || right)`. Pair nodes from left to
right; promote an unpaired final node unchanged. An empty tree returns null.

### 6.2 Inclusion

An inclusion proof carries `leaf_hash`, zero-based `index`, `tree_size`, and a
leaf-to-root array `proof` of sibling hashes. Starting at the leaf, consume a
sibling only when `(index xor 1) < size`; hash left or right according to index
parity, then set `index = floor(index/2)` and `size = ceil(size/2)`. Require
exact proof exhaustion and equality with the trusted root. Bind proof size to the
signed head's size and require `0 <= index < tree_size`.

### 6.3 Binding proofs to a head

The root and tree geometry used for an anchored input come from its verified
signed head. An inclusion proof's self-declared root, size, or scope is not an
alternative trust source. Verify the derived input scope before using a leaf's
delta, even when that leaf has a valid signature and inclusion path.

### 6.4 Append-only consistency

The reference `consistency_proof` recursively splits at the largest power of two
strictly below the new tree size. `verify_consistency` reconstructs both the old
and new roots, not only the newer root. Reject rollback, malformed proofs, and
extra proof elements. Equal sizes require equal roots and an empty proof.
The old root and size must come from retained trusted state.

### 6.5 Signed head

An `alelyon.sth/v0` object has `table`, two-string `scope`, positive integer
`tree_size`, `root`, `head_leaf`, `key_id`, `public_key`, finite `timestamp`, and
`signature`. The embedded key and key identifier must match the pinned key.
For a one-leaf head, the root must commit its stated head leaf. Larger-tree
membership requires an inclusion proof; a head alone does not establish it.

### 6.6 Key identifier

Use the `ed25519:` identifier derived in section 2 from the raw public key bytes.
An identifier is a fingerprint, not a certificate for its operator. Verification
must still use the separately pinned full public key.

### 6.7 Witness co-signature

An `alelyon.cosign/v0` statement carries `table`, `scope`, `tree_size`, `root`,
`sth_digest`, `log_key_id`, `witness_key_id`, `witness_public_key`, `cosigned_ts`,
and `signature`. `sth_digest` is SHA-256 of the canonical **complete signed STH**,
including its signature and extension members.

Verification requires the separately pinned witness key, expected root, and
complete expected STH. Compare all scope and identity fields and the complete-head
digest. The witness key identifier and material must differ from the log key.
Root-only verification is refused.

The reference witness stores its last size/root per scope. It refuses smaller
trees, same-size different roots, and growth without a valid consistency proof.
It persists the accepted state before returning a co-signature; state failure
refuses the operation. A first observation starts that scope's history. These
mechanics do not establish that a separate organization operates the witness or
that its initial observation was globally current.

### 6.8 Key lifecycle and checkpoints

`alelyon.keysuccession/v0` binds a successor's `key_id`, `public_key`, and
`not_before` to `predecessor_key_id`, signed by that predecessor.
`alelyon.keyrevocation/v0` carries `key_id`, `revoked_at`, `reason`, and
`signer_key_id`, signed by a key entitled to attest the revocation.
The closed revocation reasons are `compromise`, `retired`, `superseded-early`,
and `lost`.

An `alelyon.keymanifest/v0` has `issuer`, `root_key_id`, ordered `keys`, and
`published_at`. Entries contain `key_id`, `public_key`, `not_before`,
`not_after`, `status`, `succession`, and `revocation`. Status is `active`,
`superseded`, or `revoked`. Authenticate the first entry against an out-of-band
root. Every later entry must have a valid immediate-predecessor succession.
Reject duplicate identities, invalid validity windows, status/revocation
disagreement, more than one active key, or an active key before the chain's end.
A superseded key requires `not_after`; the root cannot carry succession evidence.

The manifest itself is unsigned transport for its chain. Lifecycle verification
in a CNE additionally requires an `alelyon.keymanifest-checkpoint/v0` object,
separately pinned checkpoint key, and previously trusted signed checkpoint.
The checkpoint contains `issuer`, `root_key_id`, positive integer `sequence`,
`manifest_digest` (SHA-256 of canonical manifest), `manifest_published_at`, full
`entries`, ordered `key_ids`, `superseded_key_ids`, `revoked_key_ids`,
`checkpoint_key_id`, finite `issued_at`, and `signature`.

Verify the checkpoint's exact manifest commitment, summaries, signature, and key
role separation from every manifest key. Relative to retained state, reject
sequence rollback, different bytes at the same sequence, issuer/root changes,
chain-prefix truncation, rewritten prior identity/succession fields, backwards
timestamps, restored retired keys, and removed or rewritten revocations. Allowed
status progress is active to active/superseded/revoked, superseded to
superseded/revoked, and revoked to revoked. Existing extension values and closed
validity endpoints cannot be rewritten. Persist `next_checkpoint` only after
successful verification; the function itself does not persist client state.

Routine supersession preserves signatures issued within the inclusive validity
window. Revocation rejects all signatures from that key regardless of stated
issuance time. A retained checkpoint establishes non-regression relative to that
state, not globally latest status or protection from checkpoint-key compromise.

## 7. Envelope shape

### 7.1 Common members

The producer emits `type`, source-text `program`, `program_hash` (SHA-256 of its
UTF-8 bytes), `inputs`, `params`, `kernel`, `created`, `refused`, `reason`,
`scalar`, `program_class`, and `error_budget`. `params` contains `K`, `alpha`,
and `strict`; optional `require_tier` is signed when supplied. A signed envelope
also contains `key_id`, `public_key`, and `signature`.

Current replay defaults are `K=63`, `alpha=0.05`, and `strict=true`. Validate
parameter types before replay: `K` and `seed` are unsigned 64-bit integers,
`alpha` is finite numeric, `strict` is Boolean, and a supplied tier floor is a
string. `certified_run` enforces supported floors and `0 < alpha < 1`.

### 7.2 Refusal

A refusal has `refused=true`, `scalar=null`, a nonempty reproducible `reason`,
the replayed `program_class`, and `error_budget={"quantization": null}`.
Success-only `sampling`, `provider`, or `model` budget members and top-level
`seed`, `assumptions`, or `branch_sites` are invalid on a refusal. Unknown
extension members remain signed. Verification requires the replay to refuse for
the exact stated reason and class. A valid refusal is evidence of refusal, not
a certified numeric answer.

### 7.3 Inputs

Each input contains `kind`, `key`, `digest`, `n`, `deltas`, and `uncertified`;
`transparency` is optional. The producer fetches each referenced input once and
uses that same captured input for commitment and replay. A reader supplies its
own matching data. The verifier checks the digest and decoded row count rather
than using `n` as evidence that the supplied series is complete.

### 7.4 Success budget

A success has `refused=false`, a numeric scalar, committed `seed`, replayed
`assumptions`, and `error_budget.quantization` containing `width`, `level`,
`exact`, and `tier`. It may carry `branch_sites`.

The current reference producer includes null `sampling` and `model` slots and a
provider status object. Other application budget surfaces can calculate sampling
diagnostics; this verifier does not recompute those diagnostics. Quantization
level, exactness label, tier, assumptions, and branch-site trace must agree with
replay. The verifier does not promote a carried sampling/provider/model value
to a replayed uncertainty bound.

### 7.5 Branch sites

A branch-site record has `op`, `winner`, `margin`, `perturb_scale`, `stable`,
and `guard`. Non-finite/infinite margin is represented by null in the signed
record. The verifier compares the branch trace with replay exactly. It also
compares the sorted assumptions. These are part of the signed claim, not UI text
that may be edited independently.

### 7.6 Transparency block

An input's optional `transparency` contains `table`, `scope`, `column`, `sth`,
and `leaves`; a `cosignature` is optional. Each leaf carries `seq`,
`value_digest`, `n`, `lo_ts`, `hi_ts`, `bits`, exact `payload`, `prev_hash`,
and `inclusion_proof`. Recompute the certificate leaf hash before checking its
inclusion. A present invalid anchor fails the envelope. Absence or complete
validity on only some inputs remains an explicitly partial check.

### 7.7 Value commitments

An eligible bars/series payload has exactly one `value_commitments` block using
`encoding="blake2b-256-row/merkle-v0"` and a `columns` mapping of lowercase
column names to Merkle roots. A row leaf hashes:

```text
UTF8("alelyon.cne/value-row/v0")
|| length_prefixed(table) || length_prefixed(scope1)
|| length_prefixed(scope2) || length_prefixed(lowercase(column))
|| encoded_timestamp || tagged_value
```

Here each length prefix is `LE64(UTF8 byte length)`, followed by UTF-8 bytes.
Bars encode timestamps as `LEi64`; series use `LEf64`.
An absent value is the byte `0x00`; a present value is `0x01 || LEf64(value)`.
Compute BLAKE2b-256 for each row, sort uniquely by timestamp, then use the Merkle
tree from section 6.1 over those row digests.

The current opening requires every row of the leaf's committed batch for that
column. A narrower window is `value-commitment-unopenable`. Missing/invalid roots
are `value-commitment-absent`; a root differing from the reader's current values
is `value-commitment-mismatch`. Such a leaf stops covering those values. A fresh
valid leaf can cover a legitimate revision; an old timestamp membership alone
cannot vouch for a replacement value.

## 8. Numeric replay

### 8.1 Substrate and comparisons

The Python kernel identifier is `alelyon-vector/<native.version()>` when its
native extension is available, otherwise `numpy-fallback`. The deterministic
native family is the specified substrate; matching fallback identifiers do not
become an independent specification and produce `unspecified-substrate`.

Matching kernel identifiers use exact numeric equality for scalar and width.
Across different kernels, integral scalars within absolute `2^53` compare
exactly. Other scalars use relative tolerance `1e-9 * max(abs(a), abs(b), 1e-300)`;
an accepted comparison reports `scalar-tolerance-window` and the absolute window.
This is numeric equality, not an assertion that signed zero encodings are equal
for hashing or signatures.

A differing kernel leaves nonzero width unperformed with `substrate-mismatch`.
The cross-substrate zero-width exception requires nonempty committed inputs,
every decoded delta exactly zero, and both stated and replayed widths exactly
zero; it reports `width-substrate-independent`. Never substitute a scalar match
for missing width verification.

### 8.2 Fixed-order accumulation

Use sequential binary64 Neumaier accumulation, with no reordering, parallel
reduction, reassociation, or fused operations:

```text
s = 0; c = 0
for x in source_order:
    t = s + x
    if abs(s) >= abs(x): c = c + ((s - t) + x)
    else:                c = c + ((x - t) + s)
    s = t
result = s + c
```

Empty sum is positive zero. Non-finite handling belongs to the calling DSL
operation; the primitive does not silently discard values.

### 8.3 Mean, variance, and correlation

Mean is the compensated sum divided by count. Sample variance computes the mean
first, then compensated sum of `(x-mean)*(x-mean)`, divided by `n-1`.
Empty mean and variance are NaN; one value has its own mean and NaN variance.
Pearson correlation centers both inputs, separately accumulates pairwise products
and squared deviations, then divides the cross-product sum by
`sqrt(sum_xx * sum_yy)`. Mismatched sizes, fewer than two rows, or zero/non-finite
denominator yield NaN. DSL alignment and drop-NaN rules precede these primitives.

### 8.4 Reproducible random stream

DRC replicate `k` seeds NumPy-compatible SeedSequence/PCG64 from the entropy
sequence `[seed, k]`, beginning at `k=0`. Expand each nonnegative integer
independently into low-first 32-bit words; zero contributes one zero word.
The following constants and wrapping operations match `vector_core`:

```text
POOL_SIZE=4; INIT_A=0x43b0d7e5; MULT_A=0x931e8875
INIT_B=0x8b51f9dd; MULT_B=0x58f38ded
MIX_MULT_L=0xca01f9dd; MIX_MULT_R=0x4973f715; XSHIFT=16
PCG64_MULTIPLIER=0x2360ed051fc65da44385df649fccf645
```

All mixer arithmetic wraps modulo `2^32`. With evolving hash constant `h`,
`hashmix(v)` sets `v = v xor h`, then `h = h * MULT_A`, then `v = v * h`,
and returns `v xor (v >> 16)`. `mix(a,b)` computes
`v = MIX_MULT_L*a - MIX_MULT_R*b`, then `v xor (v >> 16)`.

Initialize the four pool words by hashmix of corresponding entropy words (zero
padding), starting `h=INIT_A`. For each source pool index in order, keep that
source value fixed for its inner loop and mix its hashmix into every different
destination. Feed each extra entropy word through hashmix/mix into every pool
destination. The hash constant continues evolving across all these steps.

Generate state words with a new `h=INIT_B`: take successive cyclic pool words,
xor with `h`, update `h *= MULT_B`, multiply by the updated `h`, then xor with
the value shifted right 16. Generate eight 32-bit words and combine each adjacent
pair low-first into four 64-bit words `w0..w3`.

PCG64 uses 128-bit wrapping arithmetic. Set initial state to `(w0 << 64) | w1`,
initial stream to `(w2 << 64) | w3`, and increment to `(stream << 1) | 1`.
Initialize running state to zero, advance once, add initial state, and advance
again. Each output advances first by `state = state * multiplier + increment`,
then rotates `(high64(state) xor low64(state))` right by `state >> 122`.
Convert a raw word to a uniform double by `(word >> 11) * 2^-53`.

One generator serves all referenced input arrays in replay traversal order for a
replicate; do not restart per input. Resampling uses `(-0.5 + uniform) * delta`
with distinct binary64 operations, then adds it to the stored value. Capture
dither uses `low + (high-low)*uniform` with `low=-0.5*delta` and
`high=0.5*delta`; these operations must not be algebraically merged.
`vector_core/tests/spec_parity.rs` fixes streams and rounding behavior against
stored reference fixtures.

### 8.5 DRC result and refusal

Replay prunes the parsed DSL program to the final output's dependencies,
classifies its operations, and fetches only those inputs. Arbitrary Python,
shell commands, imports, and network/filesystem operations are not DSL features.
Refuse malformed programs, unsupported tier floors, absent data, invalid
parameters, non-finite execution, and excessive replay work.

For each successful replicate, calculate the absolute deviation from the base
scalar. With `m = ceil((1-alpha)*(K+1))`, refuse when `m > K`; otherwise width
is the one-based `m`th smallest deviation and level is `m/(K+1)`.
Replay caps are `K <= 10,000` and
`(K+1) * total_input_rows * pruned_AST_nodes <= 50,000,000`.

Strict mode refuses any consumed uncertified row. In non-strict mode, missing
deltas use the largest known delta for that input as an explicitly conditional
proxy; a fully uncertified input still refuses. If all input deltas are zero and
values integral, a result with absolute value above `2^53` refuses.

### 8.6 Tiers and branch conditions

`linear-exact` and `branch-stable-exact` can report an exact conformal level only
with no uncertified rows. `smooth-first-order` and `branch-stable-first-order`
carry conditional first-order claims. Hard branch-sensitive operations refuse
in strict mode. Classification and exact assumption strings are implemented in
`execcert.py`; implementations must reproduce the selected profile and its
assumptions, not upgrade it from a display label.

For salvageable branches, trace decisions across the base, every dither replicate,
and both uniform-sign systematic probes at plus/minus delta/2. A deterministic
upgrade requires every eligible direct-data decision to be delta-separated:
extremum gaps exceed `(delta_winner+delta_competitor)/2`; threshold distances
exceed delta/2. Otherwise an empirical upgrade requires complete certification,
unchanged decisions, and each positive margin greater than three times the
largest observed margin perturbation (infinite margins pass). Failed guards
refuse even in non-strict mode. The systematic probes affect branch guards only,
not the conformal order statistic. Mixed-sign perturbations beyond these probes
are not established by the empirical guard.

## 9. Verdict contract

### 9.1 Check slots

`checks` contains `authenticity`, `inputs`, `scalar`, `width`, `budget`,
`program`, `tier`, `transparency`, `witness`, `key_status`, and `provider`.
Each is true, false, or null. Null means unperformed/partial; it is not false and
must never be rendered as passed. Human `reasons` explain outcomes; stable
`reason_classes` are the machine interface.

### 9.2 Combined acceptance

The combined rule is exactly:

```text
performed = every check whose value is not null
replayed = checks.scalar is not null
bound_ok = (refused is true) or (checks.width is true)
ok = bool(checks.authenticity) and replayed and all(performed) and bound_ok
```

No-input and no-pin cases cannot pass. A program hash, when present, must match
the exact source text. Absence of optional anchoring/witness/lifecycle material
does not create evidence for those checks. Present invalid evidence fails its
check and therefore the combined verdict.

### 9.3 Trust labels

`width_trust` is `refusal` for a refusal; otherwise `unverified` when width is
unperformed, `transparency-anchored` when capture transparency passed,
`authenticated` when width passed without full anchoring, and `unverified`
otherwise. These labels are not independent acceptance decisions: always read
them with `ok` and all check slots.

`provider_trust` is `transparency-anchored` only when `checks.provider` is true;
otherwise it is `signer-attested`. The conformance normalizer derives that value
from the check for legacy early-return objects that omit the label.

### 9.4 Reason classes

These exact machine vocabularies are checked against `verify.py`. Classes are
reported as a sorted unique list. Advisory classes qualify a result without
themselves asserting a failed check; acceptance still follows section 9.2.

```reason-classes
anchor-delta-mismatch
anchor-delta-unusable
anchor-delta-zero-implausible
anchor-inclusion-failed
anchor-leaf-hash-mismatch
anchor-length-mismatch
anchor-malformed-leaf
anchor-malformed-scope
anchor-no-data
anchor-proof-index-out-of-range
anchor-proof-tree-size-mismatch
anchor-row-uncovered
anchor-scope-mismatch
anchor-sth-invalid
anchor-sth-scope-mismatch
bad-signature
budget-mismatch
capture-uncertified-leaf
delta-count-mismatch
input-digest-mismatch
input-missing
key-id-mismatch
key-manifest-checkpoint-invalid
key-manifest-checkpoint-not-monotonic
key-manifest-checkpoint-required
key-manifest-invalid
key-manifest-unrooted
key-not-in-manifest
key-outside-validity
key-revoked
malformed-envelope
malformed-pinned-key
no-input-data
no-pinned-key
no-seed
not-a-cne-v0
program-hash-mismatch
provider-attempt-count-mismatch
provider-attempts-digest-mismatch
provider-inclusion-failed
provider-leaf-hash-mismatch
provider-malformed-leaf
provider-malformed-scope
provider-no-pinned-key
provider-outcome-unknown
provider-partial
provider-proof-index-out-of-range
provider-proof-tree-size-mismatch
provider-scope-mismatch
provider-sth-invalid
provider-sth-scope-mismatch
provider-summary-mismatch
replay-refusal-mismatch
scalar-mismatch
scalar-tolerance-window
substrate-mismatch
tier-mismatch
transparency-no-pinned-key
transparency-partial
uncertified-count-mismatch
unsigned
unspecified-substrate
value-commitment-absent
value-commitment-mismatch
value-commitment-unopenable
width-mismatch
width-substrate-independent
witness-cosignature-invalid
witness-malformed
witness-partial
witness-unpinned
```

```advisory-reason-classes
scalar-tolerance-window
transparency-partial
unspecified-substrate
width-substrate-independent
witness-partial
witness-unpinned
```

### 9.5 Untrusted input boundary

The CLI's JSON loader rejects duplicate object keys, non-finite constants or
overflowed numbers, and resource-limit violations before verification. Current
limits are 64 MiB per JSON file, depth 64, 2,000,000 nodes, 1,000,000 items per
container, 1 MiB per string, and 1024 integer digits. Capture-payload parsing has
its own 32 MiB/depth-64 limits and rejects ambiguity. Do not generalize these
CLI limits to arbitrary direct Python object callers. Malformed evidence must
not be converted into a successful result or cause unbounded replay allocation.

### 9.6 Provider corroboration anchors

`error_budget.provider.anchors` maps input references to objects containing
`table="corroboration"`, `scope`, `sth`, `leaves`, and `attempts` keyed by leaf
sequence. Derive the scope from `price`/`bars` input reference as uppercase ticker
and interval, defaulting to `1d`; do not trust the block's proposed scope.

Each carried attempt is `[provider, origin, outcome, value_or_null]`. Allowed
outcomes are `answered`, `unavailable`, `quality-rejected`, and `error`.
Sort attempts by provider; hash each UTF-8
`provider + 0x1f + origin + 0x1f + outcome + 0x1f`, followed by the nullable
binary64 value, with BLAKE2b-256. Null is encoded as the reference NaN value.
Check count and digest against the proven leaf, then derive asked/answered/silent
counts; only `answered` counts as answered. Compare any carried summary counts.

A passed provider check authenticates the carried attempt records and summaries
against their log. It does not establish provider truth, independent origins,
complete probe history, or an output-error bound. An absent anchor leaves the
check null. Invalid presented evidence fails it.

## 10. Version and compatibility discipline

Keep envelope type, spec version, package version, producer, verifier, vectors,
and packaging expectations consistent. Changes to signing bytes, digest layouts,
numeric order, refusal semantics, or trust meaning require explicit protocol
review and coordinated fixtures; they are not editorial changes. Do not repurpose
an existing reason class or accept unknown capture laws as a fallback. Additive
signed fields remain committed by canonical signing; their presence must not
silently change existing-field meaning.

## 11. Claim boundaries

Read CLAIMS.md (internal) before describing results. Replay establishes the
specified computation over matching supplied inputs. Anchors detect inconsistency
with signed capture commitments; they cannot prove honest initial capture.
A width covers storage quantization under the replay's stated assumptions, not
sampling variation, provider bias, model misspecification, or all floating-point
computation error. Zero storage width does not make every computed statistic exact.
Witness independence, key custody, and external verification require separate
operational evidence.

## 12. Conformance and local checks

Vectors under `alelyon/verify/vectors/` identify the specification, input data,
envelope, trusted pins, optional witness/key-history material, category, and
substrate-dependent expectations. The runner compares `ok`, all check slots,
trust labels, and exact reason-class sets; prose reasons are diagnostic.
Goldens must fully verify on their declared substrate, forgeries must fail on
every substrate, and partial cases must not pass. Missing coverage is reported.

From the repository's development environment:

```powershell
.venv312\Scripts\python.exe -m alelyon.verify.cli selftest
.venv312\Scripts\python.exe -m pytest -q tests\oracle\test_cne_conformance.py
.venv312\Scripts\python.exe tools\gen_cne_vectors.py --check
.venv312\Scripts\python.exe packaging\alelyon-verify\build_wheel.py --check-only
```

These checks do not publish an artifact or establish an external deployment.
The wheel build includes this specification and conformance vectors; packaging
tests verify that the included specification matches the source bytes.

## 13. Implementation and publication boundaries

The Python facade is `alelyon.verify`; its extracted implementation is selected
by the explicit allowlist in `packaging/alelyon-verify/build_wheel.py`. The
private store, data providers, broker products, frontend, and signing state are
outside that verifier artifact. Presence of source or a local native extension
does not establish an approved public distribution. Consult the current package
guides and inspect staged contents before any owner-authorized release.

The executable definitions for a port are the wire modules named at the start,
`execcert.py`, the DSL parser/interpreter, `vector_core`, `vector_native`, and
the conformance fixtures. Existing section numbers 5.3.1, 6.8, 7.7, and 8.4
remain navigation targets for source comments and tests.
