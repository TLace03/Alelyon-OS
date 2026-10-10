//! Capture-log, Merkle, signed-head, witness, and provider primitives.

use std::collections::{BTreeMap, BTreeSet};

use blake2::Blake2b;
use blake2::digest::{Digest, consts::U32};
use serde_json::Value;

use crate::canonical::{
    JsonResourceLimits, canonical_json, canonical_json_without_signature, format_python_f64,
    parse_json_strict_bounded,
};
use crate::crypto::{
    blake2b_256_hex, decode_lower_hex, encode_lower_hex, key_id_from_public_key_hex, sha256,
    sha256_hex, verify_ed25519_strict,
};

const STH_TYPE: &str = "alelyon.sth/v0";
const COSIGN_TYPE: &str = "alelyon.cosign/v0";
const MAX_CAPTURE_PAYLOAD_BYTES: usize = 32 * 1024 * 1024;
const MAX_MEMBERSHIP_ROWS: usize = 1_000_000;
const MAX_CAPTURE_PAYLOAD_STRING_BYTES: usize = 1024 * 1024;
const MAX_CAPTURE_PAYLOAD_INTEGER_DIGITS: usize = 1024;
const CAPTURE_PAYLOAD_LIMITS: JsonResourceLimits = JsonResourceLimits {
    max_depth: 64,
    // One million membership scalars plus bounded metadata and column claims.
    max_nodes: MAX_MEMBERSHIP_ROWS + 100_000,
    max_container_items: MAX_MEMBERSHIP_ROWS,
    max_string_bytes: MAX_CAPTURE_PAYLOAD_STRING_BYTES,
};

#[allow(clippy::too_many_arguments)] // Frozen wire format has eleven ordered fields.
pub fn cert_leaf_hash(
    table: &str,
    scope1: &str,
    scope2: &str,
    seq: i128,
    value_digest: &str,
    n: i128,
    lo_ts: f64,
    hi_ts: f64,
    bits: i128,
    payload: &str,
    prev_hash: &str,
) -> Option<String> {
    let parts = [
        table.to_owned(),
        scope1.to_owned(),
        scope2.to_owned(),
        seq.to_string(),
        value_digest.to_owned(),
        n.to_string(),
        format_python_f64(lo_ts).ok()?,
        format_python_f64(hi_ts).ok()?,
        bits.to_string(),
        payload.to_owned(),
        prev_hash.to_owned(),
    ];
    let digest = Blake2b::<U32>::digest(parts.join("\u{001f}").as_bytes());
    Some(encode_lower_hex(&digest))
}

fn leaf_node(leaf_hash: &str) -> Option<[u8; 32]> {
    let leaf = decode_lower_hex::<32>(leaf_hash).ok()?;
    let mut bytes = Vec::with_capacity(33);
    bytes.push(0);
    bytes.extend_from_slice(&leaf);
    Some(sha256(&bytes))
}

fn internal_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(65);
    bytes.push(1);
    bytes.extend_from_slice(left);
    bytes.extend_from_slice(right);
    sha256(&bytes)
}

pub fn merkle_root(leaf_hashes: &[String]) -> Option<String> {
    let mut level = leaf_hashes
        .iter()
        .map(|hash| leaf_node(hash))
        .collect::<Option<Vec<_>>>()?;
    if level.is_empty() {
        return None;
    }
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            next.push(if pair.len() == 2 {
                internal_node(&pair[0], &pair[1])
            } else {
                pair[0]
            });
        }
        level = next;
    }
    Some(encode_lower_hex(&level[0]))
}

pub fn verify_merkle_path(
    leaf_hash: &str,
    index: usize,
    tree_size: usize,
    proof: &[String],
    root: &str,
) -> bool {
    if index >= tree_size || tree_size == 0 || decode_lower_hex::<32>(root).is_err() {
        return false;
    }
    let Some(mut node) = leaf_node(leaf_hash) else {
        return false;
    };
    let (mut idx, mut size, mut proof_index) = (index, tree_size, 0usize);
    while size > 1 {
        if (idx ^ 1) < size {
            let Some(raw) = proof.get(proof_index) else {
                return false;
            };
            let Ok(sibling) = decode_lower_hex::<32>(raw) else {
                return false;
            };
            proof_index += 1;
            node = if idx % 2 == 0 {
                internal_node(&node, &sibling)
            } else {
                internal_node(&sibling, &node)
            };
        }
        idx /= 2;
        size = size.div_ceil(2);
    }
    proof_index == proof.len() && encode_lower_hex(&node) == root
}

pub fn verify_consistency(
    first_size: usize,
    second_size: usize,
    first_root: &str,
    second_root: &str,
    proof: &[String],
) -> bool {
    if first_size == second_size {
        return first_root == second_root && proof.is_empty();
    }
    if first_size == 0 || first_size >= second_size || proof.is_empty() {
        return false;
    }
    let Ok(first) = decode_lower_hex::<32>(first_root) else {
        return false;
    };
    let Ok(second) = decode_lower_hex::<32>(second_root) else {
        return false;
    };
    let mut path = proof
        .iter()
        .map(|item| decode_lower_hex::<32>(item))
        .collect::<Result<Vec<_>, _>>()
        .ok();
    let Some(ref mut path) = path else {
        return false;
    };
    if first_size.is_power_of_two() {
        path.insert(0, first);
    }
    let (mut first_node, mut second_node) = (first_size - 1, second_size - 1);
    while first_node & 1 == 1 {
        first_node >>= 1;
        second_node >>= 1;
    }
    let Some(mut first_hash) = path.first().copied() else {
        return false;
    };
    let mut second_hash = first_hash;
    for component in &path[1..] {
        if second_node == 0 {
            return false;
        }
        if first_node & 1 == 1 || first_node == second_node {
            first_hash = internal_node(component, &first_hash);
            second_hash = internal_node(component, &second_hash);
            while first_node != 0 && first_node & 1 == 0 {
                first_node >>= 1;
                second_node >>= 1;
            }
        } else {
            second_hash = internal_node(&second_hash, component);
        }
        first_node >>= 1;
        second_node >>= 1;
    }
    second_node == 0 && first_hash == first && second_hash == second
}

fn nonempty_string(value: Option<&Value>) -> Option<&str> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
}

fn positive_usize(value: Option<&Value>) -> Option<usize> {
    value
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
}

pub fn verify_tree_head(sth: &Value, pinned_key: &str) -> bool {
    let Some(object) = sth.as_object() else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some(STH_TYPE)
        || nonempty_string(object.get("table")).is_none()
    {
        return false;
    }
    let Some(scope) = object.get("scope").and_then(Value::as_array) else {
        return false;
    };
    if scope.len() != 2
        || scope
            .iter()
            .any(|part| nonempty_string(Some(part)).is_none())
    {
        return false;
    }
    let Some(tree_size) = positive_usize(object.get("tree_size")) else {
        return false;
    };
    let (Some(root), Some(head_leaf), Some(embedded), Some(key_id), Some(signature)) = (
        object.get("root").and_then(Value::as_str),
        object.get("head_leaf").and_then(Value::as_str),
        object.get("public_key").and_then(Value::as_str),
        object.get("key_id").and_then(Value::as_str),
        object.get("signature").and_then(Value::as_str),
    ) else {
        return false;
    };
    if decode_lower_hex::<32>(root).is_err()
        || decode_lower_hex::<32>(head_leaf).is_err()
        || decode_lower_hex::<32>(embedded).is_err()
        || decode_lower_hex::<32>(pinned_key).is_err()
        || decode_lower_hex::<64>(signature).is_err()
        || embedded != pinned_key
        || key_id_from_public_key_hex(embedded).ok().as_deref() != Some(key_id)
        || key_id_from_public_key_hex(pinned_key).ok().as_deref() != Some(key_id)
        || object
            .get("timestamp")
            .and_then(Value::as_f64)
            .is_none_or(|value| !value.is_finite())
    {
        return false;
    }
    if tree_size == 1 && merkle_root(&[head_leaf.to_owned()]).as_deref() != Some(root) {
        return false;
    }
    let Ok(message) = canonical_json_without_signature(sth) else {
        return false;
    };
    verify_ed25519_strict(pinned_key, &message, signature).unwrap_or(false)
}

pub fn verify_cosignature(statement: &Value, witness_key: &str, expected_sth: &Value) -> bool {
    let (Some(object), Some(sth)) = (statement.as_object(), expected_sth.as_object()) else {
        return false;
    };
    if object.get("type").and_then(Value::as_str) != Some(COSIGN_TYPE)
        || decode_lower_hex::<32>(witness_key).is_err()
    {
        return false;
    }
    let Some(witness_id) = key_id_from_public_key_hex(witness_key).ok() else {
        return false;
    };
    let (Some(root), Some(sth_root), Some(log_id), Some(sth_log_id), Some(sth_log_key)) = (
        object.get("root").and_then(Value::as_str),
        sth.get("root").and_then(Value::as_str),
        object.get("log_key_id").and_then(Value::as_str),
        sth.get("key_id").and_then(Value::as_str),
        sth.get("public_key").and_then(Value::as_str),
    ) else {
        return false;
    };
    if root != sth_root
        || log_id != sth_log_id
        || log_id == witness_id
        || sth_log_id == witness_id
        || decode_lower_hex::<32>(sth_log_key).ok() == decode_lower_hex::<32>(witness_key).ok()
        || object.get("witness_key_id").and_then(Value::as_str) != Some(&witness_id)
        || object.get("witness_public_key").and_then(Value::as_str) != Some(witness_key)
        || object.get("tree_size") != sth.get("tree_size")
        || object.get("table") != sth.get("table")
        || object.get("scope") != sth.get("scope")
    {
        return false;
    }
    let Ok(sth_bytes) = canonical_json(expected_sth) else {
        return false;
    };
    if object.get("sth_digest").and_then(Value::as_str) != Some(sha256_hex(&sth_bytes).as_str()) {
        return false;
    }
    if object
        .get("cosigned_ts")
        .and_then(Value::as_f64)
        .is_none_or(|value| !value.is_finite())
    {
        return false;
    }
    let Some(signature) = object.get("signature").and_then(Value::as_str) else {
        return false;
    };
    let Ok(message) = canonical_json_without_signature(statement) else {
        return false;
    };
    verify_ed25519_strict(witness_key, &message, signature).unwrap_or(false)
}

#[derive(Debug, Clone, Default)]
pub struct PayloadClaims {
    pub deltas: BTreeMap<String, (f64, Option<String>)>,
    pub unusable: BTreeSet<String>,
    pub membership: Option<Membership>,
    /// Per-column value roots (SPEC §7.7). `None` means the leaf makes no
    /// statement about its values, which every leaf written before that section
    /// does — never that its values are fine.
    pub value_commitments: Option<BTreeMap<String, String>>,
    /// `(reason, columns)` when this leaf RECORDS an uncertified capture
    /// (SPEC §5.3.1). Such a leaf commits no Δ for any column and can never
    /// anchor a width.
    pub capture_outcome: Option<(String, Vec<String>)>,
}

/// SPEC §5.3.1. The closed vocabulary of capture outcomes, mirroring
/// `attest.CAPTURE_OUTCOMES`. Closed on purpose: a reason outside it is a
/// malformed record, never a new outcome, or a signer could file a certificate
/// WRITE FAILURE under a spelling that reads like a policy choice.
const CAPTURE_OUTCOMES: [&str; 5] = [
    "certification-disabled",
    "certification-declined",
    "certification-not-supplied",
    "certificate-rejected",
    "certificate-write-failed",
];
const CAPTURE_OUTCOME_SCHEMA: &str = "alelyon.capture-outcome/v0";

/// SPEC §7.7. Frozen: the encoding name is compared exactly, so a leaf written
/// under a scheme this build cannot check reads as making no statement rather
/// than as satisfying one.
pub const VALUE_COMMITMENT_ENCODING: &str = "blake2b-256-row/merkle-v0";
const VALUE_ROW_DOMAIN: &[u8] = b"alelyon.cne/value-row/v0";
const VALUE_ABSENT_TAG: u8 = 0;
const VALUE_F64_TAG: u8 = 1;

fn push_length_prefixed(bytes: &mut Vec<u8>, text: &str) {
    bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
    bytes.extend_from_slice(text.as_bytes());
}

/// One row's value commitment. Mirrors `attest.value_row_leaf` byte for byte:
/// the domain separator, then table/scope1/scope2/column each as `<Q` length and
/// bytes, then the timestamp in the table's frozen membership encoding, then the
/// absent tag or the f64 tag followed by the value.
///
/// The length prefixes make the encoding injective — without them ("AA","PL")
/// and ("A","APL") commit identical bytes, and a commitment to one scope would
/// be a commitment to another.
pub fn value_row_leaf(
    table: &str,
    scope1: &str,
    scope2: &str,
    column: &str,
    row: ValueRow,
    value: Option<f64>,
) -> String {
    let mut bytes = Vec::with_capacity(128);
    bytes.extend_from_slice(VALUE_ROW_DOMAIN);
    for part in [table, scope1, scope2, &column.to_lowercase()] {
        push_length_prefixed(&mut bytes, part);
    }
    match row {
        ValueRow::Bars(timestamp) => bytes.extend_from_slice(&timestamp.to_le_bytes()),
        ValueRow::Series(timestamp) => bytes.extend_from_slice(&timestamp.to_le_bytes()),
    }
    match value {
        None => bytes.push(VALUE_ABSENT_TAG),
        Some(value) => {
            bytes.push(VALUE_F64_TAG);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    blake2b_256_hex(&bytes)
}

/// A row key in the table's own frozen encoding, so a bars row cannot be
/// re-presented as a series row at the same instant.
#[derive(Debug, Clone, Copy)]
pub enum ValueRow {
    Bars(i64),
    Series(f64),
}

/// The Merkle root over one column's per-row commitments, rows ASCENDING by
/// timestamp. `rows` must already be in that order — the membership parser
/// guarantees it, and re-sorting here would hide a leaf that was not.
pub fn value_column_root(
    table: &str,
    scope1: &str,
    scope2: &str,
    column: &str,
    rows: &[(ValueRow, Option<f64>)],
) -> Option<String> {
    if rows.is_empty() {
        return None;
    }
    let leaves = rows
        .iter()
        .map(|(row, value)| value_row_leaf(table, scope1, scope2, column, *row, *value))
        .collect::<Vec<_>>();
    merkle_root(&leaves)
}

/// `{reason, columns}` for an uncertified-capture leaf, else None. Mirrors
/// `attest.payload_capture_outcome`, whose refusals this must reproduce exactly:
/// two markers return None rather than picking one, and a reason outside the
/// closed set is malformed rather than a new outcome.
fn parse_capture_outcome(value: &Value) -> Option<(String, Vec<String>)> {
    let object = value.as_object()?;
    if object.get("schema")?.as_str()? != CAPTURE_OUTCOME_SCHEMA {
        return None;
    }
    let reason = object.get("reason")?.as_str()?;
    if !CAPTURE_OUTCOMES.contains(&reason) {
        return None;
    }
    let columns = object
        .get("columns")?
        .as_array()?
        .iter()
        .map(|column| column.as_str().map(str::to_owned))
        .collect::<Option<Vec<_>>>()?;
    if columns.iter().collect::<BTreeSet<_>>().len() != columns.len() {
        return None;
    }
    Some((reason.to_owned(), columns))
}

fn parse_value_commitments(value: &Value) -> Option<BTreeMap<String, String>> {
    let object = value.as_object()?;
    if object.get("encoding")?.as_str()? != VALUE_COMMITMENT_ENCODING {
        return None;
    }
    let columns = object.get("columns")?.as_object()?;
    if columns.is_empty() {
        return None;
    }
    let mut out = BTreeMap::new();
    for (name, root) in columns {
        let root = root.as_str()?;
        // `attest._is_hex_bytes(root, 32)`: exactly 64 chars, LOWERCASE hex.
        // Not the lenient whitespace-tolerant reader used for proof paths — an
        // uppercase root must read as malformed on both sides or the two
        // verifiers disagree about the same signed bytes.
        if root.len() != 64 || !root.bytes().all(|ch| ch.is_ascii_digit() || (b'a'..=b'f').contains(&ch)) {
            return None;
        }
        if out.insert(name.to_lowercase(), root.to_owned()).is_some() {
            return None; // two roots for one column
        }
    }
    Some(out)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Membership {
    Bars(Vec<i64>),
    Series(Vec<f64>),
}

pub fn parse_capture_payload(payload: &str, table: &str) -> PayloadClaims {
    if payload.len() > MAX_CAPTURE_PAYLOAD_BYTES {
        return PayloadClaims::default();
    }
    // The payload is a signed JSON string nested inside the outer envelope.
    // Reparse it under the same no-duplicate rule as the JSONL boundary so two
    // verifier implementations cannot assign first-wins/last-wins meanings to
    // the same signed leaf.
    let Ok(parsed) = parse_json_strict_bounded(payload, CAPTURE_PAYLOAD_LIMITS) else {
        return PayloadClaims::default();
    };
    if !capture_payload_numbers_valid(&parsed) {
        return PayloadClaims::default();
    }
    let Value::Array(entries) = parsed else {
        return PayloadClaims::default();
    };
    let mut result = PayloadClaims::default();
    let mut memberships = Vec::new();
    let mut commitments = Vec::new();
    let mut outcomes = Vec::new();
    for entry in entries {
        let Some(object) = entry.as_object() else {
            continue;
        };
        if let Some(block) = object.get("membership") {
            memberships.push(block.clone());
        }
        if let Some(block) = object.get("value_commitments") {
            commitments.push(block.clone());
        }
        if let Some(block) = object.get("capture_outcome") {
            outcomes.push(block.clone());
        }
        let Some(column) = object.get("column") else {
            continue;
        };
        let column = value_to_python_string(column).to_lowercase();
        let law = match object.get("law") {
            None | Some(Value::Null) => None,
            Some(Value::String(value)) => Some(value.clone()),
            _ => {
                result.unusable.insert(column);
                continue;
            }
        };
        if !matches!(
            law.as_deref(),
            None | Some("dither-relative/v0") | Some("exact-cents/v0")
        ) {
            result.unusable.insert(column);
            continue;
        }
        let Some(delta) = object.get("delta").and_then(Value::as_f64) else {
            result.unusable.insert(column);
            continue;
        };
        if !delta.is_finite() || delta < 0.0 {
            result.unusable.insert(column);
            continue;
        }
        match result.deltas.get(&column) {
            Some((current, _)) if delta <= *current => {}
            _ => {
                result.deltas.insert(column, (delta, law));
            }
        }
    }
    for column in &result.unusable {
        result.deltas.remove(column);
    }
    if memberships.len() == 1 {
        result.membership = parse_membership(&memberships[0], table);
    }
    // Exactly one, like membership: a leaf proposing two has committed neither.
    if commitments.len() == 1 {
        result.value_commitments = parse_value_commitments(&commitments[0]);
    }
    if outcomes.len() == 1 {
        result.capture_outcome = parse_capture_outcome(&outcomes[0]);
    }
    result
}

fn capture_payload_numbers_valid(value: &Value) -> bool {
    match value {
        Value::Number(number) => {
            let rendered = number.to_string();
            if rendered.contains(['.', 'e', 'E']) {
                // `arbitrary_precision` keeps 1e400 as a Number even though it
                // is outside finite f64. Python's strict payload decoder rejects
                // that spelling before interpreting any sibling claim.
                number.as_f64().is_some()
            } else {
                rendered.trim_start_matches('-').len() <= MAX_CAPTURE_PAYLOAD_INTEGER_DIGITS
            }
        }
        Value::Array(values) => values.iter().all(capture_payload_numbers_valid),
        Value::Object(values) => values.values().all(capture_payload_numbers_valid),
        Value::Null | Value::Bool(_) | Value::String(_) => true,
    }
}

fn value_to_python_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => value.to_string(),
        other => other.to_string(),
    }
}

fn parse_membership(value: &Value, table: &str) -> Option<Membership> {
    let object = value.as_object()?;
    let rows = object.get("rows")?.as_array()?;
    if rows.is_empty() || rows.len() > MAX_MEMBERSHIP_ROWS {
        return None;
    }
    match table {
        "bars" if object.get("encoding")?.as_str()? == "i64-epoch-seconds/v0" => {
            let values = rows.iter().map(Value::as_i64).collect::<Option<Vec<_>>>()?;
            if !strictly_increasing(&values) {
                return None;
            }
            Some(Membership::Bars(values))
        }
        "series" if object.get("encoding")?.as_str()? == "f64-epoch-seconds/v0" => {
            let values = rows.iter().map(Value::as_f64).collect::<Option<Vec<_>>>()?;
            if values.iter().any(|value| !value.is_finite())
                || values.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return None;
            }
            Some(Membership::Series(values))
        }
        _ => None,
    }
}

fn strictly_increasing<T: Ord>(values: &[T]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

pub fn membership_matches_summary(
    membership: &Membership,
    n: usize,
    lo_ts: f64,
    hi_ts: f64,
) -> bool {
    match membership {
        Membership::Bars(rows) => {
            rows.len() == n
                && rows.first().is_some_and(|value| *value as f64 == lo_ts)
                && rows.last().is_some_and(|value| *value as f64 == hi_ts)
        }
        Membership::Series(rows) => {
            rows.len() == n
                && rows.first().copied() == Some(lo_ts)
                && rows.last().copied() == Some(hi_ts)
        }
    }
}

pub fn membership_contains(membership: &Membership, timestamp: f64) -> bool {
    match membership {
        Membership::Bars(rows) => {
            // Rust's float-to-int cast saturates out-of-range values. Without
            // the explicit half-open range check, 1e30 could falsely match an
            // encoded i64::MAX row (and -1e30 could match i64::MIN), unlike the
            // Python integer conversion frozen by the reference verifier.
            const I64_EXCLUSIVE_UPPER: f64 = 9_223_372_036_854_775_808.0;
            timestamp.is_finite()
                && timestamp >= i64::MIN as f64
                && timestamp < I64_EXCLUSIVE_UPPER
                && timestamp.fract() == 0.0
                && rows.binary_search(&(timestamp as i64)).is_ok()
        }
        Membership::Series(rows) => {
            if !timestamp.is_finite() {
                return false;
            }
            let needle = if timestamp == 0.0 { 0.0 } else { timestamp };
            rows.binary_search_by(|probe| {
                let probe = if *probe == 0.0 { 0.0 } else { *probe };
                probe.total_cmp(&needle)
            })
            .is_ok()
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderAttempt {
    pub provider: String,
    pub origin: String,
    pub outcome: String,
    pub value: Option<f64>,
}

pub fn corroboration_digest(attempts: &[ProviderAttempt]) -> String {
    let mut rows = attempts.to_vec();
    rows.sort_by(|left, right| left.provider.cmp(&right.provider));
    let mut hasher = Blake2b::<U32>::new();
    for attempt in rows {
        hasher.update(
            format!(
                "{}\u{001f}{}\u{001f}{}\u{001f}",
                attempt.provider, attempt.origin, attempt.outcome
            )
            .as_bytes(),
        );
        hasher.update(attempt.value.unwrap_or(f64::NAN).to_le_bytes());
    }
    encode_lower_hex(&hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_single_leaf_root_matches_python() {
        let leaf = "c82e5437edd17ccb7c7cb0477ab43f79dbdcd80c7b929deabdf360e3c74fbb74";
        assert_eq!(
            merkle_root(&[leaf.to_owned()]).unwrap(),
            "9aea59208290aa2120203459a3a4a167425b7275bec660cc46500b557d8d8c33"
        );
        assert!(verify_merkle_path(
            leaf,
            0,
            1,
            &[],
            "9aea59208290aa2120203459a3a4a167425b7275bec660cc46500b557d8d8c33"
        ));
    }

    #[test]
    fn cert_leaf_hash_uses_python_float_spelling() {
        assert_eq!(
            cert_leaf_hash(
                "corroboration",
                "SYN",
                "1d",
                0,
                "0e15b88aade417e4d7fe618965a66827b7fb9f02d49d5a06dc7089da2258d8ab",
                2,
                900.0,
                900.0,
                0,
                "[{\"origin\": \"primary\", \"outcome\": \"answered\", \"provider\": \"alpha\", \"value\": 100.0}, {\"origin\": \"secondary\", \"outcome\": \"unavailable\", \"provider\": \"beta\", \"value\": null}]",
                "0000000000000000000000000000000000000000000000000000000000000000",
            ).unwrap(),
            "c82e5437edd17ccb7c7cb0477ab43f79dbdcd80c7b929deabdf360e3c74fbb74"
        );
    }

    #[test]
    fn deleted_provider_silence_changes_the_digest() {
        let rows = vec![
            ProviderAttempt {
                provider: "alpha".into(),
                origin: "primary".into(),
                outcome: "answered".into(),
                value: Some(100.0),
            },
            ProviderAttempt {
                provider: "beta".into(),
                origin: "secondary".into(),
                outcome: "unavailable".into(),
                value: None,
            },
        ];
        assert_eq!(
            corroboration_digest(&rows),
            "0e15b88aade417e4d7fe618965a66827b7fb9f02d49d5a06dc7089da2258d8ab"
        );
        assert_ne!(
            corroboration_digest(&rows[..1]),
            corroboration_digest(&rows)
        );
    }

    #[test]
    fn capture_payload_rejects_duplicate_members_at_every_depth() {
        for payload in [
            r#"[{"column":"close","delta":0.1,"delta":0.2}]"#,
            r#"[{"column":"close","delta":0.1,"\u0064elta":0.2}]"#,
            r#"[{"membership":{"encoding":"i64-epoch-seconds/v0","rows":[1],"rows":[2]}}]"#,
        ] {
            let claims = parse_capture_payload(payload, "bars");
            assert!(claims.deltas.is_empty(), "accepted {payload}");
            assert!(claims.unusable.is_empty(), "partially parsed {payload}");
            assert!(claims.membership.is_none(), "accepted {payload}");
        }
    }

    #[test]
    fn capture_payload_rejects_nonfinite_and_oversized_numbers_atomically() {
        let huge_integer = "9".repeat(MAX_CAPTURE_PAYLOAD_INTEGER_DIGITS + 1);
        for payload in [
            r#"[{"column":"close","delta":0.1},{"note":1e400}]"#.to_owned(),
            format!(r#"[{{"column":"close","delta":0.1}},{{"note":{huge_integer}}}]"#),
        ] {
            let claims = parse_capture_payload(&payload, "bars");
            assert!(claims.deltas.is_empty(), "partially parsed {payload}");
            assert!(claims.unusable.is_empty(), "partially parsed {payload}");
            assert!(claims.membership.is_none(), "partially parsed {payload}");
        }
    }

    #[test]
    fn capture_payload_still_accepts_distinct_escaped_members() {
        let claims = parse_capture_payload(
            r#"[{"column":"close","delta":0.1,"law":"dither-relative/v0","\u006eote":"ok"}]"#,
            "bars",
        );
        assert_eq!(
            claims.deltas.get("close"),
            Some(&(0.1, Some("dither-relative/v0".to_owned())))
        );
    }

    #[test]
    fn bar_membership_rejects_saturating_float_to_int_aliases() {
        let membership = Membership::Bars(vec![i64::MIN, 0, i64::MAX]);
        assert!(membership_contains(&membership, 0.0));
        assert!(membership_contains(&membership, i64::MIN as f64));
        assert!(!membership_contains(
            &membership,
            9_223_372_036_854_775_808.0
        ));
        assert!(!membership_contains(&membership, 1e30));
        assert!(!membership_contains(&membership, -1e30));
    }

    #[test]
    fn series_membership_uses_numeric_signed_zero_equality() {
        assert!(membership_contains(&Membership::Series(vec![-0.0]), 0.0));
        assert!(membership_contains(&Membership::Series(vec![0.0]), -0.0));
        assert!(!membership_contains(
            &Membership::Series(vec![0.0]),
            f64::NAN
        ));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Value commitments and the capture-outcome marker (SPEC §7.7, §5.3.1).
    //
    // These four functions are the verifier's anti-forgery core, and until now
    // nothing but the cross-language parity gate stood behind them — the same
    // gate that could measure the wrong tree. A mechanism whose only guard is a
    // guard we found broken is not defended. Measured before these were written:
    // `parse_value_commitments` stubbed to always-None left cargo test at
    // 51 passed / 0 failed while the parity gate showed 27 mismatches.
    //
    // EVERY expectation below is a FROZEN CONSTANT PRODUCED BY THE PYTHON
    // REFERENCE and pasted here. None is recomputed in Rust: an expectation
    // Rust derives would agree with a Rust bug, which is how a differential
    // mechanism ends up defending nothing.
    //
    // To regenerate, with the Python reference installed (the alelyon-os package):
    //
    //     python -c "from alelyon.runtime.atlas.data.attest import \
    //       value_row_leaf, value_column_root; \
    //       print(value_row_leaf('bars','SYN','1d','close',1704153600,101.25)); \
    //       print(value_column_root('bars','SYN','1d','close',[(1,1.0),(2,2.0),(3,3.0)]))"
    //
    // Each constant carries the exact call that produced it.
    // ─────────────────────────────────────────────────────────────────────────

    // python: value_row_leaf('bars', 'SYN', '1d', 'close', 1704153600, 101.25)
    const LEAF_BARS_VALUE: &str =
        "9ec9c2e5a64d32b2922843a4a2ef0a5246148bf72bb8c8e13414a0ed784a28f9";
    // python: value_row_leaf('bars', 'SYN', '1d', 'close', 1704153600, None)
    const LEAF_BARS_ABSENT: &str =
        "631c87ff392d8af964922068d2388a52be31b1d6229421d875b86250171a017c";
    // python: value_row_leaf('bars', 'SYN', '1d', 'close', 1704153600, 0.0)
    const LEAF_BARS_ZERO: &str =
        "a9693a12e5bd9dcb7624c4985ac396c40745466d71472712416f9109d8c3f9a8";
    // python: value_row_leaf('bars', 'SYN', '1d', 'close', 1704153600, -0.0)
    const LEAF_BARS_NEG_ZERO: &str =
        "d771f1ce3b75e6e0e2376da35ba14ec3967a60f65ad718c0464634c7504b08b6";
    // python: value_row_leaf('bars', 'SYN', '1d', 'close', 1704153600, nan)
    const LEAF_BARS_NAN: &str =
        "e37c7b4f1a1a2b9aebcf7ef9438b3fe776adf43d24f468bb15079e1dd881c970";
    // python: value_row_leaf('bars', 'SYN', '1d', 'close', 1704153600, inf)
    const LEAF_BARS_INF: &str =
        "5f99310949b2a03923aa786e9764d62d1f85d9fccfb358e1374ec892d8b62ba8";
    // python: value_row_leaf('series', 'fred', 'DGS10', 'value', 1704153600.0, 101.25)
    const LEAF_SERIES_VALUE: &str =
        "6a6c8baa8760305bff3d48cb0f7ba63864e576492a837c8ff49b413f1afd1d31";
    // python: value_row_leaf('bars', 'AA', 'PL', 'close', 1, 1.0)
    const LEAF_SCOPE_AA_PL: &str =
        "918ad6cc0e391b8034eddc7510f61b6d4d1ebb7485b8ec9d878571bdc4ff17bf";
    // python: value_row_leaf('bars', 'A', 'APL', 'close', 1, 1.0)
    const LEAF_SCOPE_A_APL: &str =
        "fa83a994d35cce746a5b7b1aaac401625cb3361028659d8f941f4d1e137d8f49";

    // python: value_column_root('bars','SYN','1d','close',[(1,1.0),(2,2.0),(3,3.0)])
    const ROOT_THREE_BARS: &str =
        "3a554b772eed378a65d2c57a490549cc657af9df817362202c168b48caf98936";
    // python: value_column_root('bars', 'SYN', '1d', 'close', [(1, 1.0)])
    const ROOT_SINGLE_ROW: &str =
        "3a6ede54413ecc03ca9431f4754b7c557d1041d0ae08e8563f253f217dfe3e1f";
    // python: value_column_root('bars','SYN','1d','close',[(1,None),(2,2.0)])
    const ROOT_WITH_ABSENT: &str =
        "44a16120667e4b0fae8eace58cc05e901d62d5a6fa54800b92b2e66bd8c59fb1";
    // python: value_column_root('series','fred','DGS10','value',[(1.0,1.5),(2.0,2.5)])
    const ROOT_SERIES: &str =
        "92ee302b631255268e3f749d31d029b4145f8d158eff05325fc31036a9642742";
    // python: value_column_root('bars','SYN','1d','close',[(1,1.0),(2,2.0)])
    const ROOT_TWO_BARS: &str =
        "b969d36bed489fbc6d1f0bf7ccfeee2354bc96073279ec93b951455b848b299f";

    fn json_of(text: &str) -> Value {
        serde_json::from_str(text).expect("test fixture must be valid JSON")
    }

    // ── value_row_leaf ───────────────────────────────────────────────────────

    #[test]
    fn value_row_leaf_matches_the_python_reference() {
        assert_eq!(
            value_row_leaf("bars", "SYN", "1d", "close",
                           ValueRow::Bars(1_704_153_600), Some(101.25)),
            LEAF_BARS_VALUE
        );
        assert_eq!(
            value_row_leaf("series", "fred", "DGS10", "value",
                           ValueRow::Series(1_704_153_600.0), Some(101.25)),
            LEAF_SERIES_VALUE
        );
    }

    #[test]
    fn an_absent_value_is_not_a_fabricated_zero() {
        // The defect this program was already bitten by once, one layer down
        // (`payload_deltas`): an absent field reading as a committed 0.0. A bar
        // with no volume must not commit what a bar with volume 0 commits.
        let absent = value_row_leaf("bars", "SYN", "1d", "close",
                                    ValueRow::Bars(1_704_153_600), None);
        let zero = value_row_leaf("bars", "SYN", "1d", "close",
                                  ValueRow::Bars(1_704_153_600), Some(0.0));
        let neg_zero = value_row_leaf("bars", "SYN", "1d", "close",
                                      ValueRow::Bars(1_704_153_600), Some(-0.0));
        assert_eq!(absent, LEAF_BARS_ABSENT);
        assert_eq!(zero, LEAF_BARS_ZERO);
        assert_eq!(neg_zero, LEAF_BARS_NEG_ZERO);
        assert_ne!(absent, zero);
        // -0.0 == 0.0 numerically; their f64 BITS differ, and the commitment is
        // over the bits. Comparing as numbers would let a signer swap one for
        // the other under a single commitment.
        assert_ne!(zero, neg_zero);
    }

    #[test]
    fn nonfinite_values_commit_their_bits_rather_than_being_rejected() {
        assert_eq!(
            value_row_leaf("bars", "SYN", "1d", "close",
                           ValueRow::Bars(1_704_153_600), Some(f64::NAN)),
            LEAF_BARS_NAN
        );
        assert_eq!(
            value_row_leaf("bars", "SYN", "1d", "close",
                           ValueRow::Bars(1_704_153_600), Some(f64::INFINITY)),
            LEAF_BARS_INF
        );
        assert_ne!(LEAF_BARS_NAN, LEAF_BARS_INF);
    }

    #[test]
    fn the_column_name_is_case_folded_exactly_as_python_folds_it() {
        // The same frozen digest as the lower-case call: `str(column).lower()`.
        assert_eq!(
            value_row_leaf("bars", "SYN", "1d", "CLOSE",
                           ValueRow::Bars(1_704_153_600), Some(101.25)),
            LEAF_BARS_VALUE
        );
    }

    #[test]
    fn length_prefixes_keep_the_scope_encoding_injective() {
        // Without the <Q length prefixes, ("AA","PL") and ("A","APL") commit
        // identical bytes and a commitment to one scope is a commitment to
        // another. Both constants come from Python; the assertion is that Rust
        // reproduces each AND that they differ.
        assert_eq!(
            value_row_leaf("bars", "AA", "PL", "close", ValueRow::Bars(1), Some(1.0)),
            LEAF_SCOPE_AA_PL
        );
        assert_eq!(
            value_row_leaf("bars", "A", "APL", "close", ValueRow::Bars(1), Some(1.0)),
            LEAF_SCOPE_A_APL
        );
        assert_ne!(LEAF_SCOPE_AA_PL, LEAF_SCOPE_A_APL);
    }

    #[test]
    fn a_bars_row_cannot_be_re_presented_as_a_series_row() {
        // The same instant under a different table encoding (<q vs <d).
        assert_ne!(LEAF_BARS_VALUE, LEAF_SERIES_VALUE);
    }

    // ── value_column_root ────────────────────────────────────────────────────

    #[test]
    fn value_column_root_matches_the_python_reference() {
        let rows = [
            (ValueRow::Bars(1), Some(1.0)),
            (ValueRow::Bars(2), Some(2.0)),
            (ValueRow::Bars(3), Some(3.0)),
        ];
        assert_eq!(
            value_column_root("bars", "SYN", "1d", "close", &rows).as_deref(),
            Some(ROOT_THREE_BARS)
        );
        assert_eq!(
            value_column_root("bars", "SYN", "1d", "close",
                              &[(ValueRow::Bars(1), Some(1.0))]).as_deref(),
            Some(ROOT_SINGLE_ROW)
        );
        assert_eq!(
            value_column_root("bars", "SYN", "1d", "close",
                              &[(ValueRow::Bars(1), None),
                                (ValueRow::Bars(2), Some(2.0))]).as_deref(),
            Some(ROOT_WITH_ABSENT)
        );
        assert_eq!(
            value_column_root("series", "fred", "DGS10", "value",
                              &[(ValueRow::Series(1.0), Some(1.5)),
                                (ValueRow::Series(2.0), Some(2.5))]).as_deref(),
            Some(ROOT_SERIES)
        );
    }

    #[test]
    fn an_empty_row_set_commits_nothing_rather_than_an_empty_root() {
        // ABSENCE FAILS CLOSED. A root over zero rows would be a fixed value
        // vouching for every empty column at once; None forces the caller to say
        // what an uncommitted column may still anchor.
        assert_eq!(value_column_root("bars", "SYN", "1d", "close", &[]), None);
    }

    #[test]
    fn value_column_root_commits_the_row_order_it_was_given() {
        // This function does NOT sort — its doc comment says the membership
        // parser guarantees ascending order and that re-sorting would hide a
        // leaf that was not ordered. The Python reference DOES sort, so the two
        // agree only while the caller supplies ascending rows. Measured:
        // python(ascending) == python(descending) == ROOT_TWO_BARS, so a
        // descending call here must NOT produce ROOT_TWO_BARS. That difference
        // is a property of the contract; it is reported, not "fixed" here.
        let ascending = [
            (ValueRow::Bars(1), Some(1.0)),
            (ValueRow::Bars(2), Some(2.0)),
        ];
        let descending = [
            (ValueRow::Bars(2), Some(2.0)),
            (ValueRow::Bars(1), Some(1.0)),
        ];
        assert_eq!(
            value_column_root("bars", "SYN", "1d", "close", &ascending).as_deref(),
            Some(ROOT_TWO_BARS)
        );
        assert_ne!(
            value_column_root("bars", "SYN", "1d", "close", &descending).as_deref(),
            Some(ROOT_TWO_BARS),
            "if this ever passes, this build started re-sorting and the doc \
             comment's guarantee moved without the comment moving"
        );
    }

    // ── parse_value_commitments ──────────────────────────────────────────────

    /// A real producer block, from `attest.payload_with_value_commitments`.
    const PRODUCER_BLOCK: &str = concat!(
        r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {"close": "#,
        r#""b969d36bed489fbc6d1f0bf7ccfeee2354bc96073279ec93b951455b848b299f"}}"#
    );

    #[test]
    fn parse_value_commitments_accepts_the_producers_own_block() {
        let parsed = parse_value_commitments(&json_of(PRODUCER_BLOCK))
            .expect("the reference producer's block must parse");
        assert_eq!(parsed.get("close").map(String::as_str), Some(ROOT_TWO_BARS));
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn parse_value_commitments_folds_the_column_name() {
        let block = PRODUCER_BLOCK.replace(r#""close""#, r#""CLOSE""#);
        let parsed = parse_value_commitments(&json_of(&block)).unwrap();
        assert_eq!(parsed.get("close").map(String::as_str), Some(ROOT_TWO_BARS));
    }

    /// Absence and malformation both fail closed. This is the property an
    /// always-None stub satisfies FOR FREE, which is why it is asserted
    /// alongside the positive tests above — those are what a stub cannot
    /// survive, and together they are the pair.
    #[test]
    fn parse_value_commitments_fails_closed_on_every_malformed_shape() {
        let cases: [(&str, &str); 10] = [
            (r#"{}"#, "an empty object makes no statement"),
            (r#"[]"#, "not an object at all"),
            (
                r#"{"columns": {"close": "b969d36bed489fbc6d1f0bf7ccfeee2354bc96073279ec93b951455b848b299f"}}"#,
                "no encoding: an unnamed scheme cannot be checked",
            ),
            (
                r#"{"encoding": "other/v0", "columns": {"close": "b969d36bed489fbc6d1f0bf7ccfeee2354bc96073279ec93b951455b848b299f"}}"#,
                "an encoding this build cannot check",
            ),
            (
                r#"{"encoding": "blake2b-256-row/merkle-v0"}"#,
                "no columns member",
            ),
            (
                r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {}}"#,
                "zero columns vouches for nothing",
            ),
            (
                r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {"close": 7}}"#,
                "a root that is not a string",
            ),
            (
                r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {"close": "b969d3"}}"#,
                "a root of the wrong length",
            ),
            (
                r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {"close": "B969D36BED489FBC6D1F0BF7CCFEEE2354BC96073279EC93B951455B848B299F"}}"#,
                "UPPERCASE hex: attest._is_hex_bytes is not lenient, and a \
                 verifier accepting it would disagree with Python about the \
                 same signed bytes",
            ),
            (
                r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {"close": "g969d36bed489fbc6d1f0bf7ccfeee2354bc96073279ec93b951455b848b299f"}}"#,
                "a non-hex character",
            ),
        ];
        for (text, why) in cases {
            assert!(parse_value_commitments(&json_of(text)).is_none(), "{why}");
        }
    }

    #[test]
    fn two_roots_for_one_column_commit_neither() {
        // Distinct JSON keys that fold to one column. A reader that kept either
        // would let a signer carry an honest root for an auditor to read beside
        // the one every verifier actually used.
        let block = concat!(
            r#"{"encoding": "blake2b-256-row/merkle-v0", "columns": {"#,
            r#""close": "b969d36bed489fbc6d1f0bf7ccfeee2354bc96073279ec93b951455b848b299f", "#,
            r#""CLOSE": "3a554b772eed378a65d2c57a490549cc657af9df817362202c168b48caf98936"}}"#
        );
        assert!(parse_value_commitments(&json_of(block)).is_none());
    }

    // ── parse_capture_outcome ────────────────────────────────────────────────

    /// A real marker, byte for byte out of the committed vector
    /// `forgery-uncertified-capture-leaf`, which the Python reference generated.
    const PRODUCER_MARKER: &str = concat!(
        r#"{"schema": "alelyon.capture-outcome/v0", "#,
        r#""reason": "certificate-write-failed", "#,
        r#""columns": ["open", "high", "low", "close", "volume"]}"#
    );

    #[test]
    fn parse_capture_outcome_accepts_the_producers_own_marker() {
        let (reason, columns) = parse_capture_outcome(&json_of(PRODUCER_MARKER))
            .expect("the reference producer's marker must parse");
        assert_eq!(reason, "certificate-write-failed");
        // Neither sorted nor case-folded: the reference keeps the names as
        // written and rejects only exact duplicates.
        assert_eq!(columns, vec!["open", "high", "low", "close", "volume"]);
    }

    #[test]
    fn every_frozen_capture_outcome_is_accepted_and_nothing_else_is() {
        for reason in CAPTURE_OUTCOMES {
            let text = format!(
                r#"{{"schema": "alelyon.capture-outcome/v0", "reason": "{reason}", "columns": ["close"]}}"#
            );
            assert_eq!(
                parse_capture_outcome(&json_of(&text)).map(|(r, _)| r),
                Some(reason.to_owned()),
                "{reason}"
            );
        }
        let unknown = r#"{"schema": "alelyon.capture-outcome/v0", "reason": "probably-fine", "columns": ["close"]}"#;
        assert!(
            parse_capture_outcome(&json_of(unknown)).is_none(),
            "a reason outside the closed set is malformed, never a new outcome"
        );
    }

    #[test]
    fn parse_capture_outcome_fails_closed_on_every_malformed_shape() {
        let cases: [(&str, &str); 7] = [
            (r#"{}"#, "an empty object"),
            (r#"[]"#, "not an object at all"),
            (
                r#"{"schema": "other/v0", "reason": "certification-disabled", "columns": ["close"]}"#,
                "an unrecognised schema",
            ),
            (
                r#"{"schema": "alelyon.capture-outcome/v0", "reason": 7, "columns": ["close"]}"#,
                "a reason that is not a string",
            ),
            (
                r#"{"schema": "alelyon.capture-outcome/v0", "reason": "certification-disabled"}"#,
                "no columns member",
            ),
            (
                r#"{"schema": "alelyon.capture-outcome/v0", "reason": "certification-disabled", "columns": [7]}"#,
                "a column that is not a string",
            ),
            (
                r#"{"schema": "alelyon.capture-outcome/v0", "reason": "certification-disabled", "columns": ["close", "close"]}"#,
                "an exact duplicate column",
            ),
        ];
        for (text, why) in cases {
            assert!(parse_capture_outcome(&json_of(text)).is_none(), "{why}");
        }
    }

    #[test]
    fn capture_outcome_columns_are_compared_as_written() {
        // Measured against the reference rather than assumed: Python checks
        // `len(set(cols)) != len(cols)` over the names AS WRITTEN, so "A" and
        // "a" are distinct and the marker is ACCEPTED. A parser that folded case
        // here would reject a marker Python accepts.
        let text = r#"{"schema": "alelyon.capture-outcome/v0", "reason": "certification-disabled", "columns": ["A", "a"]}"#;
        assert_eq!(
            parse_capture_outcome(&json_of(text)).map(|(_, c)| c),
            Some(vec!["A".to_owned(), "a".to_owned()])
        );
    }
}
