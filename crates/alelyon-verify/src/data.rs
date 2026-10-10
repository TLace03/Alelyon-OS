//! Language-neutral input commitments and untrusted delta decoding.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use blake2::Blake2b;
use blake2::digest::{Digest, consts::U32};
use serde_json::{Map, Value};

use crate::crypto::encode_lower_hex;

const MAX_DELTA_ROWS: usize = 10_000_000;
const MAX_ROW_KEY_BYTES: usize = 4_096;

/// A caller-supplied input after the §3 keep-last/sort canonicalization.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalInput {
    TimeSeries { index: Vec<f64>, values: Vec<f64> },
    KeyedTable { keys: Vec<String>, values: Vec<f64> },
}

impl CanonicalInput {
    pub fn len(&self) -> usize {
        match self {
            Self::TimeSeries { values, .. } | Self::KeyedTable { values, .. } => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn values(&self) -> &[f64] {
        match self {
            Self::TimeSeries { values, .. } | Self::KeyedTable { values, .. } => values,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataError {
    ExpectedObject(&'static str),
    MissingField(&'static str),
    InvalidField(&'static str),
    UnsupportedKind(String),
    LengthMismatch,
    RowLimit,
    InvalidRowKey,
    AmbiguousDeltaEncoding,
    DeltaRunCoverage,
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpectedObject(name) => write!(f, "{name} is not an object"),
            Self::MissingField(name) => write!(f, "missing field {name}"),
            Self::InvalidField(name) => write!(f, "invalid field {name}"),
            Self::UnsupportedKind(kind) => write!(f, "unsupported input kind {kind:?}"),
            Self::LengthMismatch => f.write_str("input index and values have different lengths"),
            Self::RowLimit => f.write_str("declared row count exceeds the resource limit"),
            Self::InvalidRowKey => f.write_str("invalid keyed-table row key"),
            Self::AmbiguousDeltaEncoding => {
                f.write_str("delta block must contain exactly one encoding branch")
            }
            Self::DeltaRunCoverage => {
                f.write_str("delta runs do not cover the declared row count exactly")
            }
        }
    }
}

impl std::error::Error for DataError {}

/// Parse the vector/JSONL input map (`kind|key` -> portable series object).
pub fn parse_input_map(
    value: &Value,
) -> Result<BTreeMap<(String, String), CanonicalInput>, DataError> {
    let object = value
        .as_object()
        .ok_or(DataError::ExpectedObject("inputs"))?;
    let mut out = BTreeMap::new();
    for (reference, spec) in object {
        let (kind, key) = reference
            .split_once('|')
            .ok_or(DataError::InvalidField("input reference"))?;
        if kind.is_empty() || key.is_empty() {
            return Err(DataError::InvalidField("input reference"));
        }
        out.insert((kind.to_owned(), key.to_owned()), parse_input(kind, spec)?);
    }
    Ok(out)
}

pub fn parse_input(kind: &str, spec: &Value) -> Result<CanonicalInput, DataError> {
    let object = spec
        .as_object()
        .ok_or(DataError::ExpectedObject("input data"))?;
    let values = parse_values(
        object
            .get("values")
            .ok_or(DataError::MissingField("values"))?,
    )?;
    match kind {
        "price" | "series" => {
            let index = parse_numbers(
                object
                    .get("index")
                    .ok_or(DataError::MissingField("index"))?,
                "index",
            )?;
            canonical_time_series(index, values)
        }
        "table" => {
            let raw_keys = object
                .get("keys")
                .and_then(Value::as_array)
                .ok_or(DataError::InvalidField("keys"))?;
            if raw_keys.len() > MAX_DELTA_ROWS {
                return Err(DataError::RowLimit);
            }
            if raw_keys.len() != values.len() {
                return Err(DataError::LengthMismatch);
            }
            let mut keys = Vec::with_capacity(raw_keys.len());
            for value in raw_keys {
                let key = value.as_str().ok_or(DataError::InvalidRowKey)?;
                let bytes = key.as_bytes();
                if bytes.is_empty() || bytes.len() > MAX_ROW_KEY_BYTES {
                    return Err(DataError::InvalidRowKey);
                }
                keys.push(key.to_owned());
            }
            canonical_keyed_table(keys, values)
        }
        other => Err(DataError::UnsupportedKind(other.to_owned())),
    }
}

fn parse_values(value: &Value) -> Result<Vec<f64>, DataError> {
    let array = value.as_array().ok_or(DataError::InvalidField("values"))?;
    if array.len() > MAX_DELTA_ROWS {
        return Err(DataError::RowLimit);
    }
    array
        .iter()
        .map(|value| match value {
            Value::Null => Ok(f64::NAN),
            Value::Number(number) => number
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or(DataError::InvalidField("values")),
            _ => Err(DataError::InvalidField("values")),
        })
        .collect()
}

fn parse_numbers(value: &Value, field: &'static str) -> Result<Vec<f64>, DataError> {
    let array = value.as_array().ok_or(DataError::InvalidField(field))?;
    if array.len() > MAX_DELTA_ROWS {
        return Err(DataError::RowLimit);
    }
    array
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or(DataError::InvalidField(field))
        })
        .collect()
}

fn canonical_time_series(index: Vec<f64>, values: Vec<f64>) -> Result<CanonicalInput, DataError> {
    if index.len() != values.len() {
        return Err(DataError::LengthMismatch);
    }
    // Keep the last occurrence under ordinary numeric equality, matching pandas'
    // duplicated(keep="last") behavior (including +0.0 == -0.0). A suffix scan
    // is quadratic and lets a large unique index monopolize the verifier; map
    // replacement preserves keep-last in O(n log n).
    let mut rows = BTreeMap::new();
    for (timestamp, value) in index.into_iter().zip(values) {
        rows.insert(FiniteIndexKey::new(timestamp), (timestamp, value));
    }
    let (index, values) = rows.into_values().unzip();
    Ok(CanonicalInput::TimeSeries { index, values })
}

#[derive(Debug, Clone, Copy)]
struct FiniteIndexKey(f64);

impl FiniteIndexKey {
    fn new(value: f64) -> Self {
        debug_assert!(value.is_finite());
        // pandas treats both signed zeros as one duplicate key.
        Self(if value == 0.0 { 0.0 } else { value })
    }
}

impl PartialEq for FiniteIndexKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for FiniteIndexKey {}

impl PartialOrd for FiniteIndexKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FiniteIndexKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

fn canonical_keyed_table(keys: Vec<String>, values: Vec<f64>) -> Result<CanonicalInput, DataError> {
    if keys.len() != values.len() {
        return Err(DataError::LengthMismatch);
    }
    // BTreeMap's UTF-8 lexical order is Unicode scalar-value order, and insert is
    // deliberately keep-last for duplicate keys.
    let mut rows = BTreeMap::new();
    for (key, value) in keys.into_iter().zip(values) {
        rows.insert(key, value);
    }
    let (keys, values) = rows.into_iter().unzip();
    Ok(CanonicalInput::KeyedTable { keys, values })
}

/// The BLAKE2b-256 commitment defined by SPEC §§3.2 and 3.4.
pub fn commitment_digest(kind: &str, input: &CanonicalInput) -> Result<String, DataError> {
    let mut hasher = Blake2b::<U32>::new();
    match (kind, input) {
        ("price" | "series", CanonicalInput::TimeSeries { index, values }) => {
            for (&timestamp, &value) in index.iter().zip(values) {
                hasher.update(timestamp.to_le_bytes());
                hasher.update(value.to_le_bytes());
            }
        }
        ("table", CanonicalInput::KeyedTable { keys, values }) => {
            for (key, &value) in keys.iter().zip(values) {
                let bytes = key.as_bytes();
                hasher.update((bytes.len() as u64).to_le_bytes());
                hasher.update(bytes);
                hasher.update(value.to_le_bytes());
            }
        }
        (other, _) if !matches!(other, "price" | "series" | "table") => {
            return Err(DataError::UnsupportedKind(other.to_owned()));
        }
        _ => return Err(DataError::InvalidField("input layout")),
    }
    Ok(encode_lower_hex(&hasher.finalize()))
}

/// Decode an untrusted const/runs/list delta block without partial allocation.
pub fn decompress_deltas(value: &Value) -> Result<Vec<Option<f64>>, DataError> {
    let object = value
        .as_object()
        .ok_or(DataError::ExpectedObject("delta commitment"))?;
    let branches = ["const", "runs", "list"]
        .into_iter()
        .filter(|name| object.contains_key(*name))
        .collect::<Vec<_>>();
    if branches.len() != 1 {
        return Err(DataError::AmbiguousDeltaEncoding);
    }
    match branches[0] {
        "const" => decode_const(object),
        "runs" => decode_runs(object),
        "list" => decode_list(object),
        _ => unreachable!(),
    }
}

fn row_count(object: &Map<String, Value>) -> Result<usize, DataError> {
    let n = object
        .get("n")
        .and_then(Value::as_u64)
        .ok_or(DataError::InvalidField("n"))?;
    usize::try_from(n)
        .ok()
        .filter(|n| *n <= MAX_DELTA_ROWS)
        .ok_or(DataError::RowLimit)
}

fn finite_delta(value: &Value) -> Result<Option<f64>, DataError> {
    if value.is_null() {
        return Ok(None);
    }
    let number = value
        .as_f64()
        .filter(|number| number.is_finite() && *number >= 0.0)
        .ok_or(DataError::InvalidField("delta"))?;
    Ok(Some(number))
}

fn decode_const(object: &Map<String, Value>) -> Result<Vec<Option<f64>>, DataError> {
    let n = row_count(object)?;
    let value = finite_delta(
        object
            .get("const")
            .ok_or(DataError::MissingField("const"))?,
    )?;
    if value.is_none() {
        return Err(DataError::InvalidField("const"));
    }
    let raw_indices = object
        .get("uncertified_idx")
        .and_then(Value::as_array)
        .ok_or(DataError::InvalidField("uncertified_idx"))?;
    if raw_indices.len() > MAX_DELTA_ROWS {
        return Err(DataError::RowLimit);
    }
    let mut indices = Vec::with_capacity(raw_indices.len());
    for raw in raw_indices {
        let index = raw
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .filter(|i| *i < n)
            .ok_or(DataError::InvalidField("uncertified_idx"))?;
        indices.push(index);
    }
    let mut out = vec![value; n];
    for index in indices {
        out[index] = None;
    }
    Ok(out)
}

fn decode_runs(object: &Map<String, Value>) -> Result<Vec<Option<f64>>, DataError> {
    let n = row_count(object)?;
    let raw_runs = object
        .get("runs")
        .and_then(Value::as_array)
        .ok_or(DataError::InvalidField("runs"))?;
    if raw_runs.len() > MAX_DELTA_ROWS {
        return Err(DataError::RowLimit);
    }
    let mut runs = Vec::with_capacity(raw_runs.len());
    let mut total = 0usize;
    for raw in raw_runs {
        let pair = raw
            .as_array()
            .filter(|pair| pair.len() == 2)
            .ok_or(DataError::InvalidField("runs"))?;
        let count = pair[1]
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .ok_or(DataError::InvalidField("run count"))?;
        total = total.checked_add(count).ok_or(DataError::RowLimit)?;
        if total > MAX_DELTA_ROWS {
            return Err(DataError::RowLimit);
        }
        runs.push((finite_delta(&pair[0])?, count));
    }
    if total != n {
        return Err(DataError::DeltaRunCoverage);
    }
    let mut out = Vec::with_capacity(n);
    for (value, count) in runs {
        out.extend(std::iter::repeat_n(value, count));
    }
    Ok(out)
}

fn decode_list(object: &Map<String, Value>) -> Result<Vec<Option<f64>>, DataError> {
    let raw = object
        .get("list")
        .and_then(Value::as_array)
        .ok_or(DataError::InvalidField("list"))?;
    if raw.len() > MAX_DELTA_ROWS {
        return Err(DataError::RowLimit);
    }
    if let Some(n) = object.get("n") {
        if n.as_u64() != Some(raw.len() as u64) {
            return Err(DataError::LengthMismatch);
        }
    }
    raw.iter().map(finite_delta).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn time_series_is_keep_last_then_sorted() {
        let input = parse_input(
            "price",
            &json!({"index": [2.0, 1.0, 2.0], "values": [20.0, 10.0, 21.0]}),
        )
        .unwrap();
        assert_eq!(
            input,
            CanonicalInput::TimeSeries {
                index: vec![1.0, 2.0],
                values: vec![10.0, 21.0],
            }
        );
    }

    #[test]
    fn table_digest_has_length_framing_and_keep_last() {
        let input = parse_input(
            "table",
            &json!({"keys": ["β", "a", "a"], "values": [2.0, 1.0, 3.0]}),
        )
        .unwrap();
        let digest = commitment_digest("table", &input).unwrap();
        assert_eq!(digest.len(), 64);
        assert_eq!(input.values(), &[3.0, 2.0]);
    }

    #[test]
    fn delta_decoder_rejects_ambiguous_and_undercover_blocks() {
        assert_eq!(
            decompress_deltas(&json!({"const": 1.0, "runs": [], "n": 0, "uncertified_idx": []})),
            Err(DataError::AmbiguousDeltaEncoding)
        );
        assert_eq!(
            decompress_deltas(&json!({"runs": [[1.0, 1]], "n": 2})),
            Err(DataError::DeltaRunCoverage)
        );
    }

    #[test]
    fn null_delta_is_uncertified_not_zero() {
        assert_eq!(
            decompress_deltas(&json!({"list": [0.0, null, 1.0]})).unwrap(),
            vec![Some(0.0), None, Some(1.0)]
        );
    }

    #[test]
    fn delta_decoder_rejects_type_confusion_and_null_const() {
        for block in [
            json!({"const": null, "n": 1, "uncertified_idx": []}),
            json!({"const": true, "n": 1, "uncertified_idx": []}),
            json!({"const": 1.0, "n": true, "uncertified_idx": []}),
            json!({"const": 1.0, "n": 1, "uncertified_idx": [false]}),
            json!({"runs": [[1.0, 1.0]], "n": 1}),
            json!({"list": [-1.0]}),
            json!({"list": ["0.0"]}),
        ] {
            assert!(decompress_deltas(&block).is_err(), "accepted {block}");
        }
    }

    #[test]
    fn signed_zero_duplicate_keeps_the_last_index_bits() {
        let input = parse_input(
            "series",
            &json!({"index": [0.0, -0.0], "values": [1.0, 2.0]}),
        )
        .unwrap();
        let CanonicalInput::TimeSeries { index, values } = input else {
            panic!("series parser returned a table")
        };
        assert_eq!(index, vec![-0.0]);
        assert!(index[0].is_sign_negative());
        assert_eq!(values, vec![2.0]);
    }
}
