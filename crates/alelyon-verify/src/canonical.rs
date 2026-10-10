//! The signing encoding frozen by CNE specification section 2.
//!
//! This is deliberately not RFC 8785/JCS. In particular, f64 values use CPython
//! 3.12 `repr` notation thresholds and exponent spelling. Serde JSON is used only
//! as a JSON value/parser layer; this module owns every emitted byte.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt::{self, Write as _};

use serde_json::{Map, Number, Value};

/// Resource policy for the persistent JSONL boundary. The canonical encoder
/// itself remains usable for larger in-memory values; these bounds apply only
/// while accepting untrusted wire text.
#[derive(Debug, Clone, Copy)]
pub struct JsonResourceLimits {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_container_items: usize,
    pub max_string_bytes: usize,
}

/// A canonical-encoding failure.
#[derive(Debug)]
pub enum CanonicalError {
    /// The input text was not valid JSON.
    InvalidJson(serde_json::Error),
    /// JSON cannot carry NaN or either infinity, and signing one is forbidden.
    NonFiniteFloat,
    /// A JSON number could not be interpreted under the CNE integer/f64 rules.
    InvalidNumber(String),
    /// RFC 8259 permits parsers to choose how duplicate names are handled, but
    /// that choice is unsafe at a signed interoperability boundary.
    DuplicateObjectMember,
    /// The JSON text exceeded an explicit pre-parse resource bound.
    ResourceLimit(&'static str),
    /// The lightweight pre-parser found malformed JSON before allocation.
    InvalidStructure(&'static str),
    /// Removing a signature is defined only for a top-level JSON object.
    TopLevelNotObject,
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(error) => write!(f, "invalid JSON: {error}"),
            Self::NonFiniteFloat => f.write_str("non-finite floats are not canonical JSON"),
            Self::InvalidNumber(number) => {
                write!(f, "number {number:?} has no canonical CNE representation")
            }
            Self::DuplicateObjectMember => {
                f.write_str("duplicate JSON object member is not allowed")
            }
            Self::ResourceLimit(limit) => write!(f, "JSON exceeds the {limit} resource limit"),
            Self::InvalidStructure(reason) => write!(f, "invalid JSON structure: {reason}"),
            Self::TopLevelNotObject => {
                f.write_str("signature removal requires a top-level JSON object")
            }
        }
    }
}

impl std::error::Error for CanonicalError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidJson(error) => Some(error),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for CanonicalError {
    fn from(value: serde_json::Error) -> Self {
        Self::InvalidJson(value)
    }
}

/// Encode a JSON value using the exact CNE signing representation.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    let mut output = String::new();
    write_value(value, &mut output, false)?;
    Ok(output.into_bytes())
}

/// Encode a signed object after removing only its top-level `signature` member.
///
/// Nested members named `signature` and all unknown extension members remain in
/// the encoded object and are therefore covered by the signature.
pub fn canonical_json_without_signature(value: &Value) -> Result<Vec<u8>, CanonicalError> {
    if !value.is_object() {
        return Err(CanonicalError::TopLevelNotObject);
    }
    let mut output = String::new();
    write_value(value, &mut output, true)?;
    Ok(output.into_bytes())
}

/// Parse JSON text and encode the resulting value canonically.
///
/// `serde_json`'s `arbitrary_precision` feature preserves integers beyond 64 bits;
/// they are emitted as exact decimal integers instead of being rounded through
/// f64. JSON spellings that parse to a non-finite f64 are refused.
pub fn parse_and_canonicalize(input: &str) -> Result<Vec<u8>, CanonicalError> {
    let value = parse_json_strict(input)?;
    canonical_json(&value)
}

/// Parse JSON while rejecting duplicate object members at every depth.
///
/// `serde_json::Value` normally keeps the last duplicate member. That is a
/// reasonable generic parser policy but is ambiguous for signed input: another
/// implementation may keep the first member or reject the document. The
/// preflight walk compares decoded member names, so `"id"` and `"\u0069d"`
/// are duplicates too.
pub fn parse_json_strict(input: &str) -> Result<Value, CanonicalError> {
    // Match serde_json's default recursion ceiling so the duplicate-member
    // preflight never becomes the first stack-unbounded parser.
    JsonPreflight::new(
        input,
        Some(JsonResourceLimits {
            max_depth: 128,
            max_nodes: usize::MAX,
            max_container_items: usize::MAX,
            max_string_bytes: usize::MAX,
        }),
    )
    .validate()?;
    Ok(serde_json::from_str(input)?)
}

/// Parse untrusted JSON under explicit structural limits, before constructing a
/// potentially much larger `serde_json::Value` tree.
pub fn parse_json_strict_bounded(
    input: &str,
    limits: JsonResourceLimits,
) -> Result<Value, CanonicalError> {
    JsonPreflight::new(input, Some(limits)).validate()?;
    Ok(serde_json::from_str(input)?)
}

struct JsonPreflight<'a> {
    input: &'a str,
    bytes: &'a [u8],
    position: usize,
    nodes: usize,
    limits: Option<JsonResourceLimits>,
}

impl<'a> JsonPreflight<'a> {
    fn new(input: &'a str, limits: Option<JsonResourceLimits>) -> Self {
        Self {
            input,
            bytes: input.as_bytes(),
            position: 0,
            nodes: 0,
            limits,
        }
    }

    fn validate(mut self) -> Result<(), CanonicalError> {
        self.skip_whitespace();
        self.parse_value(1)?;
        self.skip_whitespace();
        if self.position != self.bytes.len() {
            return Err(CanonicalError::InvalidStructure(
                "trailing non-whitespace data",
            ));
        }
        Ok(())
    }

    fn parse_value(&mut self, depth: usize) -> Result<(), CanonicalError> {
        if let Some(limits) = self.limits {
            if depth > limits.max_depth {
                return Err(CanonicalError::ResourceLimit("JSON depth"));
            }
            self.nodes = self
                .nodes
                .checked_add(1)
                .ok_or(CanonicalError::ResourceLimit("JSON node count"))?;
            if self.nodes > limits.max_nodes {
                return Err(CanonicalError::ResourceLimit("JSON node count"));
            }
        }
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.parse_object(depth),
            Some(b'[') => self.parse_array(depth),
            Some(b'"') => self.parse_string().map(|_| ()),
            Some(b't') => self.parse_literal(b"true"),
            Some(b'f') => self.parse_literal(b"false"),
            Some(b'n') => self.parse_literal(b"null"),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(CanonicalError::InvalidStructure("unexpected token")),
            None => Err(CanonicalError::InvalidStructure("missing value")),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<(), CanonicalError> {
        self.position += 1;
        self.skip_whitespace();
        if self.consume(b'}') {
            return Ok(());
        }
        let mut names = HashSet::new();
        let mut items = 0usize;
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(CanonicalError::InvalidStructure(
                    "object member name is not a string",
                ));
            }
            let range = self.parse_string()?;
            let name: String = serde_json::from_str(&self.input[range])?;
            if !names.insert(name) {
                return Err(CanonicalError::DuplicateObjectMember);
            }
            items = items
                .checked_add(1)
                .ok_or(CanonicalError::ResourceLimit("container item count"))?;
            self.check_container_items(items)?;
            self.skip_whitespace();
            if !self.consume(b':') {
                return Err(CanonicalError::InvalidStructure("object member omits ':'"));
            }
            self.parse_value(depth + 1)?;
            self.skip_whitespace();
            if self.consume(b'}') {
                return Ok(());
            }
            if !self.consume(b',') {
                return Err(CanonicalError::InvalidStructure(
                    "object members are not comma-separated",
                ));
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<(), CanonicalError> {
        self.position += 1;
        self.skip_whitespace();
        if self.consume(b']') {
            return Ok(());
        }
        let mut items = 0usize;
        loop {
            items = items
                .checked_add(1)
                .ok_or(CanonicalError::ResourceLimit("container item count"))?;
            self.check_container_items(items)?;
            self.parse_value(depth + 1)?;
            self.skip_whitespace();
            if self.consume(b']') {
                return Ok(());
            }
            if !self.consume(b',') {
                return Err(CanonicalError::InvalidStructure(
                    "array items are not comma-separated",
                ));
            }
        }
    }

    /// Scan one JSON string and return its complete source range, quotes
    /// included. `serde_json` performs the final Unicode/surrogate validation.
    fn parse_string(&mut self) -> Result<std::ops::Range<usize>, CanonicalError> {
        let start = self.position;
        self.position += 1;
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.position += 1;
                    let range = start..self.position;
                    if let Some(limits) = self.limits {
                        // Resource limits apply to the decoded UTF-8 string,
                        // matching Python's `len(value.encode("utf-8"))`.
                        // Counting source escapes would reject compact values
                        // such as `"\u0061"` differently across languages.
                        let decoded: String = serde_json::from_str(&self.input[range.clone()])?;
                        if decoded.len() > limits.max_string_bytes {
                            return Err(CanonicalError::ResourceLimit("JSON string byte count"));
                        }
                    }
                    return Ok(range);
                }
                b'\\' => {
                    self.position += 1;
                    match self.peek() {
                        Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => {
                            self.position += 1;
                        }
                        Some(b'u') => {
                            self.position += 1;
                            for _ in 0..4 {
                                if !self.peek().is_some_and(|digit| digit.is_ascii_hexdigit()) {
                                    return Err(CanonicalError::InvalidStructure(
                                        "invalid Unicode escape",
                                    ));
                                }
                                self.position += 1;
                            }
                        }
                        _ => return Err(CanonicalError::InvalidStructure("invalid string escape")),
                    }
                }
                0x00..=0x1f => {
                    return Err(CanonicalError::InvalidStructure(
                        "unescaped control character in string",
                    ));
                }
                _ => self.position += 1,
            }
        }
        Err(CanonicalError::InvalidStructure("unterminated string"))
    }

    fn parse_literal(&mut self, literal: &[u8]) -> Result<(), CanonicalError> {
        if self.bytes[self.position..].starts_with(literal) {
            self.position += literal.len();
            Ok(())
        } else {
            Err(CanonicalError::InvalidStructure("invalid literal"))
        }
    }

    fn parse_number(&mut self) -> Result<(), CanonicalError> {
        if self.consume(b'-') && !self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            return Err(CanonicalError::InvalidStructure("invalid number"));
        }
        match self.peek() {
            Some(b'0') => self.position += 1,
            Some(b'1'..=b'9') => {
                self.position += 1;
                while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                    self.position += 1;
                }
            }
            _ => return Err(CanonicalError::InvalidStructure("invalid number")),
        }
        if self.consume(b'.') {
            if !self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                return Err(CanonicalError::InvalidStructure("invalid number fraction"));
            }
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.position += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.position += 1;
            }
            if !self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                return Err(CanonicalError::InvalidStructure("invalid number exponent"));
            }
            while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
                self.position += 1;
            }
        }
        Ok(())
    }

    fn check_container_items(&self, items: usize) -> Result<(), CanonicalError> {
        if self
            .limits
            .is_some_and(|limits| items > limits.max_container_items)
        {
            Err(CanonicalError::ResourceLimit("container item count"))
        } else {
            Ok(())
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn consume(&mut self, byte: u8) -> bool {
        if self.peek() == Some(byte) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
}

fn write_value(
    value: &Value,
    output: &mut String,
    omit_top_level_signature: bool,
) -> Result<(), CanonicalError> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => output.push_str(&canonical_number(number)?),
        Value::String(value) => write_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_value(item, output, false)?;
            }
            output.push(']');
        }
        Value::Object(values) => write_object(values, output, omit_top_level_signature)?,
    }
    Ok(())
}

fn write_object(
    values: &Map<String, Value>,
    output: &mut String,
    omit_signature: bool,
) -> Result<(), CanonicalError> {
    let mut keys: Vec<&str> = values
        .keys()
        .map(String::as_str)
        .filter(|key| !(omit_signature && *key == "signature"))
        .collect();
    // State the contract directly instead of relying on a map implementation's
    // byte or UTF-16 ordering. Rust chars compare by Unicode scalar value, which
    // is the Python-str/code-point order frozen by the specification.
    keys.sort_unstable_by(|left, right| unicode_codepoint_cmp(left, right));

    output.push('{');
    for (index, key) in keys.into_iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write_string(key, output);
        output.push(':');
        // Every key came from this map; failure would indicate an internal bug.
        let value = values
            .get(key)
            .expect("a key collected from a JSON object must remain present");
        write_value(value, output, false)?;
    }
    output.push('}');
    Ok(())
}

fn unicode_codepoint_cmp(left: &str, right: &str) -> Ordering {
    left.chars().cmp(right.chars())
}

fn write_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            control if control <= '\u{001f}' => {
                write!(output, "\\u{:04x}", control as u32)
                    .expect("writing to a String cannot fail");
            }
            other => output.push(other),
        }
    }
    output.push('"');
}

fn canonical_number(number: &Number) -> Result<String, CanonicalError> {
    let raw = number.to_string();
    if raw.contains(['.', 'e', 'E']) {
        let value = raw
            .parse::<f64>()
            .map_err(|_| CanonicalError::InvalidNumber(raw.clone()))?;
        format_python_f64(value)
    } else {
        normalize_integer(&raw)
    }
}

fn normalize_integer(raw: &str) -> Result<String, CanonicalError> {
    let (negative, digits) = match raw.strip_prefix('-') {
        Some(digits) => (true, digits),
        None => (false, raw),
    };
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CanonicalError::InvalidNumber(raw.to_owned()));
    }
    let significant = digits.trim_start_matches('0');
    if significant.is_empty() {
        return Ok("0".to_owned());
    }
    Ok(if negative {
        format!("-{significant}")
    } else {
        significant.to_owned()
    })
}

/// Render one finite f64 exactly as CPython 3.12 `repr` does for CNE v0.
///
/// Ryu supplies only the shortest round-tripping digit sequence. This function
/// deliberately re-renders those digits using Python's notation threshold,
/// mandatory exponent sign/padding, positional decimal point, and signed zero.
pub fn format_python_f64(value: f64) -> Result<String, CanonicalError> {
    if !value.is_finite() {
        return Err(CanonicalError::NonFiniteFloat);
    }

    let negative = value.is_sign_negative();
    if value == 0.0 {
        return Ok(if negative { "-0.0" } else { "0.0" }.to_owned());
    }

    let mut buffer = ryu::Buffer::new();
    let shortest = buffer.format_finite(value.abs());
    let (mut digits, exponent) = decompose_shortest(shortest)?;

    // Trailing significant zeros can be represented by the decimal exponent.
    // Removing them makes the mantissa shortest while preserving the f64 value.
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }

    let mut output = String::new();
    if negative {
        output.push('-');
    }

    if exponent >= 16 || exponent <= -5 {
        output.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            output.push('.');
            output.push_str(&digits[1..]);
        }
        output.push('e');
        output.push(if exponent < 0 { '-' } else { '+' });
        write!(output, "{:02}", exponent.abs()).expect("writing to a String cannot fail");
        return Ok(output);
    }

    if exponent >= 0 {
        let integer_digits = exponent as usize + 1;
        if digits.len() <= integer_digits {
            output.push_str(&digits);
            output.extend(std::iter::repeat_n('0', integer_digits - digits.len()));
            output.push_str(".0");
        } else {
            output.push_str(&digits[..integer_digits]);
            output.push('.');
            output.push_str(&digits[integer_digits..]);
        }
    } else {
        output.push_str("0.");
        output.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
        output.push_str(&digits);
    }
    Ok(output)
}

/// Convert Ryu's shortest representation into `(significant digits,
/// scientific decimal exponent)` without changing the digit sequence.
fn decompose_shortest(shortest: &str) -> Result<(String, i32), CanonicalError> {
    let (mantissa, explicit_exponent) = match shortest.find(['e', 'E']) {
        Some(position) => {
            let exponent = shortest[position + 1..]
                .parse::<i32>()
                .map_err(|_| CanonicalError::InvalidNumber(shortest.to_owned()))?;
            (&shortest[..position], exponent)
        }
        None => (shortest, 0),
    };
    let decimal_position = mantissa.find('.').unwrap_or(mantissa.len());
    let raw_digits: String = mantissa
        .chars()
        .filter(|character| *character != '.')
        .collect();
    if raw_digits.is_empty() || !raw_digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(CanonicalError::InvalidNumber(shortest.to_owned()));
    }
    let first_nonzero = raw_digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or_else(|| CanonicalError::InvalidNumber(shortest.to_owned()))?;
    let exponent = explicit_exponent + decimal_position as i32 - first_nonzero as i32 - 1;
    Ok((raw_digits[first_nonzero..].to_owned(), exponent))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_text(bytes: Vec<u8>) -> String {
        String::from_utf8(bytes).expect("canonical JSON is UTF-8")
    }

    #[test]
    fn matches_every_frozen_cpython_float_boundary() {
        let cases = [
            (1.0, "1.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (2.0_f64.powi(53), "9007199254740992.0"),
            (1.0 / 3.0, "0.3333333333333333"),
            (f64::from_bits(1), "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (-0.0, "-0.0"),
        ];
        for (value, expected) in cases {
            assert_eq!(format_python_f64(value).unwrap(), expected, "{value:?}");
        }
    }

    #[test]
    fn rejects_every_non_finite_float() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                format_python_f64(value),
                Err(CanonicalError::NonFiniteFloat)
            ));
        }
        assert!(parse_and_canonicalize("1e400").is_err());
        assert!(parse_and_canonicalize("NaN").is_err());
    }

    #[test]
    fn preserves_arbitrary_precision_integers() {
        assert_eq!(
            as_text(parse_and_canonicalize("1180591620717411303424").unwrap()),
            "1180591620717411303424"
        );
        assert_eq!(as_text(parse_and_canonicalize("-0").unwrap()), "0");
    }

    #[test]
    fn sorts_by_unicode_codepoint_and_formats_nested_values() {
        let value: Value = serde_json::from_str(
            r#"{"\ud800\udc00":"astral","\ue000":"bmp","z":[1e16,1e-5],"a":true}"#,
        )
        .unwrap();
        assert_eq!(
            as_text(canonical_json(&value).unwrap()),
            "{\"a\":true,\"z\":[1e+16,1e-05],\"\u{e000}\":\"bmp\",\"\u{10000}\":\"astral\"}"
        );
    }

    #[test]
    fn implements_the_frozen_string_escape_table() {
        let value = Value::String(
            "\"\\\u{0008}\u{000c}\n\r\t\u{0001}/\u{007f}\u{2028}\u{2029}é".to_owned(),
        );
        assert_eq!(
            as_text(canonical_json(&value).unwrap()),
            "\"\\\"\\\\\\b\\f\\n\\r\\t\\u0001/\u{007f}\u{2028}\u{2029}é\""
        );
    }

    #[test]
    fn removes_only_the_top_level_signature_member() {
        let value: Value =
            serde_json::from_str(r#"{"x":1.0,"signature":"drop","nested":{"signature":"keep"}}"#)
                .unwrap();
        assert_eq!(
            as_text(canonical_json_without_signature(&value).unwrap()),
            r#"{"nested":{"signature":"keep"},"x":1.0}"#
        );
        assert!(matches!(
            canonical_json_without_signature(&Value::Null),
            Err(CanonicalError::TopLevelNotObject)
        ));
    }

    #[test]
    fn strict_parser_rejects_duplicate_members_after_escape_decoding() {
        for input in [
            r#"{"a":1,"a":2}"#,
            r#"{"nested":{"a":1,"\u0061":2}}"#,
            r#"{"\ud83d\ude00":1,"😀":2}"#,
        ] {
            assert!(matches!(
                parse_json_strict(input),
                Err(CanonicalError::DuplicateObjectMember)
            ));
        }
    }

    #[test]
    fn strict_parser_preserves_distinct_names_and_arbitrary_precision() {
        let value =
            parse_json_strict(r#"{"a":1180591620717411303424,"nested":{"A":1,"\u0061":2}}"#)
                .unwrap();
        assert_eq!(
            as_text(canonical_json(&value).unwrap()),
            r#"{"a":1180591620717411303424,"nested":{"A":1,"a":2}}"#
        );
    }

    #[test]
    fn bounded_parser_rejects_depth_and_string_amplification() {
        let limits = JsonResourceLimits {
            max_depth: 2,
            max_nodes: 10,
            max_container_items: 10,
            max_string_bytes: 3,
        };
        assert!(matches!(
            parse_json_strict_bounded("[[null]]", limits),
            Err(CanonicalError::ResourceLimit("JSON depth"))
        ));
        assert!(matches!(
            parse_json_strict_bounded(r#""four""#, limits),
            Err(CanonicalError::ResourceLimit("JSON string byte count"))
        ));
        assert!(parse_json_strict_bounded(r#""\u0061\u0062\u0063""#, limits).is_ok());
    }
}
