//! Complete fail-closed CNE verdict construction.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use serde_json::{Map, Value, json};

use crate::ENVELOPE_TYPE;
use crate::canonical::canonical_json_without_signature;
use crate::crypto::{
    decode_lower_hex, key_id_from_public_key_hex, sha256_hex, verify_ed25519_strict,
};
use crate::data::{CanonicalInput, commitment_digest, decompress_deltas, parse_input_map};
use crate::kernel::ReplayKernel;
use crate::keylife::{KeyStatus, KeylifeErrorKind, key_status_at, verify_manifest_checkpoint};
use crate::replay::{FetchedInput, ReplayParams, ReplayResult, non_text_program_refusal, replay};
use crate::transparency::{
    Membership, ProviderAttempt, ValueRow, cert_leaf_hash, corroboration_digest,
    membership_contains, membership_matches_summary, parse_capture_payload, value_column_root,
    verify_cosignature, verify_merkle_path, verify_tree_head,
};

const CHECK_SLOTS: [&str; 11] = [
    "authenticity",
    "inputs",
    "scalar",
    "width",
    "budget",
    "program",
    "tier",
    "transparency",
    "witness",
    "key_status",
    "provider",
];
const REL_TOL: f64 = 1e-9;
const SAFE_EXACT_INT: f64 = 9_007_199_254_740_992.0;

#[derive(Debug, Clone)]
struct VerdictState {
    checks: BTreeMap<&'static str, Option<bool>>,
    reasons: BTreeSet<String>,
}

impl VerdictState {
    fn new() -> Self {
        Self {
            checks: CHECK_SLOTS.into_iter().map(|slot| (slot, None)).collect(),
            reasons: BTreeSet::new(),
        }
    }

    fn set(&mut self, slot: &'static str, value: bool) {
        self.checks.insert(slot, Some(value));
    }

    fn reason(&mut self, class: &str) {
        self.reasons.insert(class.to_owned());
    }

    fn result(&self, envelope: &Value, force_false: bool) -> Value {
        let refused = envelope.get("refused").and_then(Value::as_bool) == Some(true);
        let performed = self.checks.values().flatten().all(|value| *value);
        let replayed = self.checks["scalar"].is_some();
        let bound_ok = refused || self.checks["width"] == Some(true);
        let ok = !force_false
            && self.checks["authenticity"] == Some(true)
            && replayed
            && performed
            && bound_ok;
        let width_trust = if force_false {
            "unverified"
        } else if refused {
            "refusal"
        } else if self.checks["width"].is_none() {
            "unverified"
        } else if self.checks["transparency"] == Some(true) {
            "transparency-anchored"
        } else if self.checks["width"] == Some(true) {
            "authenticated"
        } else {
            "unverified"
        };
        let provider_trust = if self.checks["provider"] == Some(true) {
            "transparency-anchored"
        } else {
            "signer-attested"
        };
        json!({
            "ok": ok,
            "checks": self.checks,
            "reasons": self.reasons.iter().cloned().collect::<Vec<_>>(),
            "reason_classes": self.reasons.iter().cloned().collect::<Vec<_>>(),
            "width_trust": width_trust,
            "provider_trust": provider_trust,
        })
    }
}

/// Verify a self-contained language-neutral conformance case, replaying on
/// `kernel` (None: no replay, see [`verify_envelope`]).
pub fn verify_case(case: &Value, kernel: Option<&dyn ReplayKernel>) -> Value {
    let Some(object) = case.as_object() else {
        let mut state = VerdictState::new();
        state.reason("malformed-envelope");
        return state.result(&Value::Null, true);
    };
    verify_envelope(
        object.get("envelope").unwrap_or(&Value::Null),
        object.get("inputs"),
        object.get("pins"),
        kernel,
    )
}

/// Verify an envelope against portable caller inputs and out-of-band pins.
///
/// Replay runs on `kernel`. With None, nothing is replayed: `scalar`, `tier`,
/// `budget` and `width` stay null (not performed), no reason class is added for
/// their absence (the frozen vocabulary has none, and a null slot already says
/// it), and `ok` is false, as it is for every verdict whose replay did not run.
/// Every other check is made as usual. [`crate::kernel`] says why.
pub fn verify_envelope(
    envelope: &Value,
    input_spec: Option<&Value>,
    pins: Option<&Value>,
    kernel: Option<&dyn ReplayKernel>,
) -> Value {
    let mut state = VerdictState::new();
    let Some(cne) = envelope.as_object() else {
        state.reason("not-a-cne-v0");
        return state.result(envelope, true);
    };
    if cne.get("type").and_then(Value::as_str) != Some(ENVELOPE_TYPE) {
        state.reason("not-a-cne-v0");
        return state.result(envelope, true);
    }
    let pin_object = pins.and_then(Value::as_object);
    let mut pinned_key = pin_object
        .and_then(|pins| pins.get("public_key_hex"))
        .and_then(Value::as_str);
    if let Some(pin) = pinned_key {
        if decode_lower_hex::<32>(pin).is_err() {
            state.set("authenticity", false);
            state.reason("malformed-pinned-key");
            pinned_key = None;
        }
    } else {
        state.set("authenticity", false);
        state.reason("no-pinned-key");
    }
    if state.checks["authenticity"].is_none() {
        if !cne.contains_key("signature") {
            state.set("authenticity", false);
            state.reason("unsigned");
        } else if cne
            .get("key_id")
            .filter(|key_id| python_truthy(key_id))
            .is_some_and(|key_id| {
                pinned_key
                    .and_then(|pin| key_id_from_public_key_hex(pin).ok())
                    .as_deref()
                    != key_id.as_str()
            })
        {
            state.set("authenticity", false);
            state.reason("key-id-mismatch");
        } else {
            let valid = pinned_key
                .zip(cne.get("signature").and_then(Value::as_str))
                .and_then(|(pin, signature)| {
                    canonical_json_without_signature(envelope)
                        .ok()
                        .map(|message| {
                            verify_ed25519_strict(pin, &message, signature).unwrap_or(false)
                        })
                })
                .unwrap_or(false);
            state.set("authenticity", valid);
            if !valid {
                state.reason("bad-signature");
            }
        }
    }

    if cne.get("program_hash").is_some() {
        let valid = cne
            .get("program")
            .and_then(Value::as_str)
            .zip(cne.get("program_hash").and_then(Value::as_str))
            .is_some_and(|(program, digest)| sha256_hex(program.as_bytes()) == digest);
        state.set("program", valid);
        if !valid {
            state.reason("program-hash-mismatch");
        }
    }

    let supplied = input_spec
        .and_then(Value::as_object)
        .filter(|inputs| !inputs.is_empty());
    if supplied.is_some() {
        let parsed = match parse_input_map(input_spec.unwrap_or(&Value::Null)) {
            Ok(parsed) => parsed,
            Err(_) => {
                state.reason("malformed-envelope");
                return state.result(envelope, true);
            }
        };
        let fetched = match check_inputs_and_build(cne, &parsed, &mut state) {
            Ok(fetched) => fetched,
            Err(_) => {
                state.reason("malformed-envelope");
                return state.result(envelope, true);
            }
        };
        let program_value = match cne.get("program") {
            Some(program) => program,
            None => {
                state.reason("malformed-envelope");
                return state.result(envelope, true);
            }
        };
        let params = match replay_params(cne) {
            Ok(params) => params,
            Err(_) => {
                state.reason("malformed-envelope");
                return state.result(envelope, true);
            }
        };
        if params.seed_invalid {
            state.reason("malformed-envelope");
            return state.result(envelope, true);
        }
        if let Some(kernel) = kernel {
            let replayed = match program_value.as_str() {
                Some(program) => replay(program, &fetched, &params, kernel),
                None => non_text_program_refusal(),
            };
            if replayed.fatal {
                state.reason("malformed-envelope");
                return state.result(envelope, true);
            }
            if replay_checks(cne, &fetched, &replayed, kernel.id(), &mut state).is_err() {
                state.reason("malformed-envelope");
                return state.result(envelope, true);
            }
        }
        if let Err(class) = check_transparency(cne, &parsed, pinned_key) {
            if class == "malformed-envelope" {
                state.reason(class);
                return state.result(envelope, true);
            } else if class != "absent" {
                state.set("transparency", false);
                state.reason(class);
            }
        } else if transparency_present(cne) {
            let input_count = input_commitments(cne).len();
            let anchored = input_commitments(cne)
                .iter()
                .filter(|input| truthy(input.get("transparency")))
                .count();
            if anchored == input_count {
                state.set("transparency", true);
            } else {
                state.reason("transparency-partial");
            }
        }
    } else {
        state.reason("no-input-data");
    }

    check_witness(cne, pin_object, &mut state);
    check_provider(cne, pinned_key, &mut state);
    // keylife.rs owns the full manifest/checkpoint logic; it is wired below once
    // all of its pins are present. Until then, absence remains the specified null.
    let manifest_present = pin_object
        .and_then(|pins| pins.get("key_manifest"))
        .is_some_and(|value| !value.is_null());
    let checkpoint_without_manifest = !manifest_present
        && pin_object.is_some_and(|pins| {
            [
                "manifest_checkpoint",
                "checkpoint_public_key_hex",
                "trusted_manifest_checkpoint",
            ]
            .iter()
            .any(|field| pins.get(*field).is_some_and(|value| !value.is_null()))
        });
    if checkpoint_without_manifest {
        state.set("key_status", false);
        state.reason("key-manifest-checkpoint-required");
    } else if manifest_present {
        check_key_lifecycle(cne, pin_object, &mut state);
    }
    state.result(envelope, false)
}

fn replay_params(cne: &Map<String, Value>) -> Result<ReplayParams, ()> {
    let params = match cne.get("params") {
        None => None,
        Some(value) => Some(value.as_object().ok_or(())?),
    };
    let k = match params.and_then(|params| params.get("K")) {
        Some(Value::Number(value)) => value.as_u64().map(i128::from).ok_or(())?,
        Some(_) => return Err(()),
        None => 63,
    };
    let alpha = match params.and_then(|params| params.get("alpha")) {
        Some(Value::Number(value)) => {
            let value = value.as_f64().ok_or(())?;
            if !value.is_finite() {
                return Err(());
            }
            value
        }
        Some(_) => return Err(()),
        None => 0.05,
    };
    let strict = match params.and_then(|params| params.get("strict")) {
        Some(Value::Bool(value)) => *value,
        Some(_) => return Err(()),
        None => true,
    };
    let require_tier = match params.and_then(|params| params.get("require_tier")) {
        Some(Value::String(value)) => Some(value.clone()),
        Some(_) => return Err(()),
        None => None,
    };
    let (seed, seed_invalid) = match cne.get("seed") {
        None | Some(Value::Null) => (None, false),
        Some(Value::Bool(_)) => (None, true),
        Some(Value::Number(value)) => match value.as_u64() {
            Some(seed) => (Some(seed), false),
            None => (None, true),
        },
        Some(Value::String(_) | Value::Array(_) | Value::Object(_)) => (None, true),
    };
    Ok(ReplayParams {
        seed,
        seed_invalid,
        k,
        alpha,
        strict,
        require_tier,
    })
}

fn input_commitments(cne: &Map<String, Value>) -> Vec<&Map<String, Value>> {
    cne.get("inputs")
        .and_then(Value::as_array)
        .map(|inputs| inputs.iter().filter_map(Value::as_object).collect())
        .unwrap_or_default()
}

fn check_inputs_and_build(
    cne: &Map<String, Value>,
    supplied: &BTreeMap<(String, String), CanonicalInput>,
    state: &mut VerdictState,
) -> Result<BTreeMap<(String, String), FetchedInput>, ()> {
    let inputs = cne.get("inputs").and_then(Value::as_array).ok_or(())?;
    let mut valid = true;
    let mut replay_shape_error = false;
    let mut fetched = BTreeMap::new();
    for commitment in inputs {
        let object = commitment.as_object().ok_or(())?;
        let kind_value = object.get("kind").ok_or(())?;
        let key_value = object.get("key").ok_or(())?;
        let digest = object.get("digest").ok_or(())?;
        let (Some(kind), Some(key)) = (kind_value.as_str(), key_value.as_str()) else {
            // The Python reference forms its lookup key from the untyped JSON
            // values first. A non-text kind/key therefore records a missing
            // input, then the replay fetcher rejects the malformed reference.
            valid = false;
            replay_shape_error = true;
            state.reason("input-missing");
            continue;
        };
        let reference = (kind.to_owned(), key.to_owned());
        let Some(data) = supplied.get(&reference) else {
            valid = false;
            replay_shape_error = true;
            state.reason("input-missing");
            continue;
        };
        let digest_matches = digest.as_str().is_some_and(|digest| {
            commitment_digest(kind, data).is_ok_and(|actual| actual == digest)
        });
        if !digest_matches {
            valid = false;
            state.reason("input-digest-mismatch");
        }
        let deltas = decompress_deltas(object.get("deltas").ok_or(())?).map_err(|_| ())?;
        if deltas.len() != data.len() {
            valid = false;
            state.reason("delta-count-mismatch");
            fetched.insert(
                reference,
                FetchedInput {
                    data: data.clone(),
                    deltas,
                },
            );
            continue;
        }
        let actual_uncertified = deltas.iter().filter(|delta| delta.is_none()).count() as i64;
        if digest_matches {
            if let Some(declared) = object.get("uncertified") {
                if !python_int_equals(declared, actual_uncertified)? {
                    valid = false;
                    state.reason("uncertified-count-mismatch");
                }
            }
        }
        fetched.insert(
            reference,
            FetchedInput {
                data: data.clone(),
                deltas,
            },
        );
    }
    state.set("inputs", valid);
    if replay_shape_error {
        Err(())
    } else {
        Ok(fetched)
    }
}

fn replay_checks(
    cne: &Map<String, Value>,
    fetched: &BTreeMap<(String, String), FetchedInput>,
    replayed: &ReplayResult,
    local_kernel: &str,
    state: &mut VerdictState,
) -> Result<(), ()> {
    let Some(refused) = cne.get("refused").and_then(Value::as_bool) else {
        state.set("scalar", false);
        state.reason("malformed-envelope");
        return Ok(());
    };
    if refused {
        replay_refusal_checks(cne, replayed, state);
        return Ok(());
    }
    let quantization_value = cne
        .get("error_budget")
        .and_then(Value::as_object)
        .and_then(|budget| budget.get("quantization"));
    let exact = cne.get("kernel").and_then(Value::as_str) == Some(local_kernel);
    let scalar = if replayed.ok && replayed.base_value.is_some() {
        scalar_eq(replayed.base_value, cne.get("scalar"), exact, state)?
    } else {
        false
    };
    state.set("scalar", scalar);
    let tier =
        cne.get("program_class").and_then(Value::as_str) == Some(replayed.program_class.as_str());
    state.set("tier", tier);
    if cne
        .get("assumptions")
        .and_then(Value::as_array)
        .is_some_and(|assumptions| assumptions.iter().any(|value| !value.is_string()))
    {
        return Err(());
    }
    let quantization = match quantization_value {
        Some(Value::Object(budget)) => Some(budget),
        Some(value) if python_truthy(value) => return Err(()),
        Some(_) | None => None,
    };
    let stated_level = optional_python_float(quantization.and_then(|budget| budget.get("level")))?;
    let tier_matches = quantization
        .and_then(|budget| budget.get("tier"))
        .is_none_or(|value| {
            value.is_null() || value.as_str() == Some(replayed.program_class.as_str())
        });
    let budget = num_eq(replayed.level, stated_level, true)
        && replayed.level_exact
            == quantization
                .and_then(|budget| budget.get("exact"))
                .is_some_and(python_truthy)
        && tier_matches
        && sorted_strings(cne.get("assumptions")) == sorted_owned(&replayed.assumptions)
        && cne
            .get("branch_sites")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            == replayed.branch_sites;
    state.set("budget", budget);
    if !budget {
        state.reason("budget-mismatch");
    }
    if cne.get("seed").is_none_or(Value::is_null) {
        state.set("width", false);
        state.reason("no-seed");
        return Ok(());
    }
    let stated_width = optional_python_float(quantization.and_then(|budget| budget.get("width")))?;
    let all_zero = !fetched.is_empty()
        && fetched.values().all(|input| {
            !input.deltas.is_empty() && input.deltas.iter().all(|delta| *delta == Some(0.0))
        });
    let zero_width = all_zero && stated_width == Some(0.0) && replayed.width == Some(0.0);
    if exact || zero_width {
        let width = num_eq(replayed.width, stated_width, true);
        state.set("width", width);
        if !width {
            state.reason("width-mismatch");
        } else if !exact {
            state.reason("width-substrate-independent");
        } else if !local_kernel.starts_with("alelyon-vector/") {
            state.reason("unspecified-substrate");
        }
    } else {
        state.reason("substrate-mismatch");
    }
    if !scalar {
        state.reason("scalar-mismatch");
    }
    if !tier {
        state.reason("tier-mismatch");
    }
    Ok(())
}

fn replay_refusal_checks(
    cne: &Map<String, Value>,
    replayed: &ReplayResult,
    state: &mut VerdictState,
) {
    let scalar_shape = cne.contains_key("scalar") && cne.get("scalar").is_some_and(Value::is_null);
    let stated_reason = cne
        .get("reason")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    let reason_matches = replayed.refused && stated_reason == replayed.reason.as_deref();
    state.set("scalar", replayed.refused && scalar_shape && reason_matches);
    if !replayed.refused || (stated_reason.is_some() && !reason_matches) {
        state.reason("replay-refusal-mismatch");
    } else if stated_reason.is_none() {
        state.reason("malformed-envelope");
    }
    if !scalar_shape {
        state.reason("malformed-envelope");
    }
    let tier = replayed.refused
        && cne.get("program_class").and_then(Value::as_str)
            == Some(replayed.program_class.as_str());
    state.set("tier", tier);
    if !tier {
        state.reason("tier-mismatch");
    }
    let budget = cne.get("error_budget").and_then(Value::as_object);
    let quantization_ok = budget.is_some_and(|budget| {
        budget.contains_key("quantization")
            && budget.get("quantization").is_some_and(Value::is_null)
    });
    let no_success_budget = budget.is_some_and(|budget| {
        ["sampling", "provider", "model"]
            .iter()
            .all(|field| !budget.contains_key(*field))
    });
    let no_success_top = ["seed", "assumptions", "branch_sites"]
        .iter()
        .all(|field| !cne.contains_key(*field));
    state.set(
        "budget",
        quantization_ok && no_success_budget && no_success_top,
    );
    if state.checks["budget"] != Some(true) {
        state.reason("budget-mismatch");
    }
}

fn scalar_eq(
    replayed: Option<f64>,
    stated: Option<&Value>,
    exact: bool,
    state: &mut VerdictState,
) -> Result<bool, ()> {
    let stated = optional_python_float(stated)?;
    let (Some(left), Some(right)) = (replayed, stated) else {
        return Ok(replayed.is_none() && stated.is_none());
    };
    if exact {
        return Ok(left == right || (left.is_nan() && right.is_nan()));
    }
    if left.abs() <= SAFE_EXACT_INT
        && right.abs() <= SAFE_EXACT_INT
        && left.fract() == 0.0
        && right.fract() == 0.0
    {
        return Ok(left == right);
    }
    let ok = num_eq(Some(left), Some(right), false);
    if ok {
        state.reason("scalar-tolerance-window");
    }
    Ok(ok)
}

fn num_eq(left: Option<f64>, right: Option<f64>, exact: bool) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(left), Some(right)) if left.is_nan() || right.is_nan() => {
            left.is_nan() && right.is_nan()
        }
        (Some(left), Some(right)) if exact => left == right,
        (Some(left), Some(right)) => {
            (left - right).abs() <= REL_TOL * left.abs().max(right.abs()).max(1e-300)
        }
        _ => false,
    }
}

fn sorted_strings(value: Option<&Value>) -> Vec<String> {
    let mut values: Vec<String> = value
        .and_then(Value::as_array)
        .map(|values| values.iter().map(value_to_string).collect())
        .unwrap_or_default();
    values.sort();
    values
}

fn sorted_owned(values: &[String]) -> Vec<String> {
    let mut values = values.to_vec();
    values.sort();
    values
}

fn transparency_present(cne: &Map<String, Value>) -> bool {
    input_commitments(cne)
        .iter()
        .any(|input| truthy(input.get("transparency")))
}

fn check_transparency(
    cne: &Map<String, Value>,
    supplied: &BTreeMap<(String, String), CanonicalInput>,
    pinned_key: Option<&str>,
) -> Result<(), &'static str> {
    let present = input_commitments(cne)
        .into_iter()
        .filter(|input| truthy(input.get("transparency")))
        .collect::<Vec<_>>();
    if present.is_empty() {
        return Err("absent");
    }
    let Some(pin) = pinned_key else {
        return Err("transparency-no-pinned-key");
    };
    for input in present {
        verify_one_anchor(input, supplied, pin)?;
    }
    Ok(())
}

fn canonical_scope(kind: &str, key: &str) -> Option<(String, String, String, String)> {
    match kind {
        "price" => Some((
            "bars".into(),
            key.to_uppercase(),
            "1d".into(),
            "close".into(),
        )),
        "series" => Some((
            "series".into(),
            "fred".into(),
            key.to_uppercase(),
            "value".into(),
        )),
        "table" => {
            let (dataset, column) = key.split_once('|')?;
            Some((
                "table".into(),
                dataset.to_uppercase(),
                column.to_lowercase(),
                column.to_lowercase(),
            ))
        }
        _ => None,
    }
}

fn verify_one_anchor(
    input: &Map<String, Value>,
    supplied: &BTreeMap<(String, String), CanonicalInput>,
    pin: &str,
) -> Result<(), &'static str> {
    let block = input
        .get("transparency")
        .and_then(Value::as_object)
        .ok_or("anchor-malformed-scope")?;
    let sth = block.get("sth").ok_or("anchor-sth-invalid")?;
    if !verify_tree_head(sth, pin) {
        return Err("anchor-sth-invalid");
    }
    let kind = input
        .get("kind")
        .and_then(Value::as_str)
        .ok_or("anchor-scope-mismatch")?;
    let key = input
        .get("key")
        .and_then(Value::as_str)
        .ok_or("anchor-scope-mismatch")?;
    let expected = canonical_scope(kind, key).ok_or("anchor-scope-mismatch")?;
    let scope = block
        .get("scope")
        .and_then(Value::as_array)
        .ok_or("anchor-malformed-scope")?;
    if scope.len() != 2 {
        return Err("anchor-malformed-scope");
    }
    let actual = (
        block.get("table").map(value_to_string).unwrap_or_default(),
        value_to_string(&scope[0]),
        value_to_string(&scope[1]),
        block.get("column").map(value_to_string).unwrap_or_default(),
    );
    if actual != expected {
        return Err("anchor-scope-mismatch");
    }
    let sth_object = sth.as_object().ok_or("anchor-sth-invalid")?;
    if sth_object.get("table").and_then(Value::as_str) != Some(expected.0.as_str())
        || sth_object
            .get("scope")
            .and_then(Value::as_array)
            .is_none_or(|scope| {
                scope.len() != 2
                    || value_to_string(&scope[0]) != expected.1
                    || value_to_string(&scope[1]) != expected.2
            })
    {
        return Err("anchor-sth-scope-mismatch");
    }
    let root = sth_object
        .get("root")
        .and_then(Value::as_str)
        .ok_or("anchor-sth-invalid")?;
    let tree_size = sth_object
        .get("tree_size")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or("anchor-sth-invalid")?;
    let leaves = block
        .get("leaves")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut claims = Vec::new();
    for leaf in leaves {
        let object = leaf.as_object().ok_or("anchor-malformed-leaf")?;
        let recomputed = leaf_hash_from_record(object, &expected.0, &expected.1, &expected.2)
            .ok_or("anchor-malformed-leaf")?;
        let proof = object
            .get("inclusion_proof")
            .and_then(Value::as_object)
            .ok_or("anchor-malformed-leaf")?;
        if proof.get("leaf_hash").and_then(Value::as_str) != Some(recomputed.as_str()) {
            return Err("anchor-leaf-hash-mismatch");
        }
        let proof_size = proof
            .get("tree_size")
            .map(python_int)
            .transpose()
            .map_err(|_| "malformed-envelope")?
            .unwrap_or(-1);
        let proof_size = usize::try_from(proof_size).unwrap_or(usize::MAX);
        if proof_size != tree_size {
            return Err("anchor-proof-tree-size-mismatch");
        }
        let index = proof
            .get("index")
            .map(python_int)
            .transpose()
            .map_err(|_| "malformed-envelope")?
            .unwrap_or(-1);
        let index = usize::try_from(index).unwrap_or(usize::MAX);
        if index >= tree_size {
            return Err("anchor-proof-index-out-of-range");
        }
        let path_values = proof.get("proof").and_then(Value::as_array);
        let path = path_values
            .map(|values| string_array(values))
            .unwrap_or_default();
        if path_values.is_some_and(|values| {
            values.iter().any(|value| {
                value
                    .as_str()
                    .is_none_or(|value| !python_hex_bytes_valid(value))
            })
        }) {
            return Err("malformed-envelope");
        }
        if !verify_merkle_path(&recomputed, index, tree_size, &path, root) {
            return Err("anchor-inclusion-failed");
        }
        let payload = object
            .get("payload")
            .and_then(Value::as_str)
            .ok_or("anchor-malformed-leaf")?;
        let parsed = parse_capture_payload(payload, &expected.0);
        // A leaf that RECORDS an uncertified capture can never vouch for a Δ, and
        // it is refused before the Δ is looked at (SPEC §5.3.1). Checked ahead of
        // the unusable test, which it would otherwise fall out as, because the
        // two say different things: that one reports a malformed commitment, this
        // one reports a correct commitment to the fact that no certificate
        // exists. The ORDER is the wire contract — Python refuses here too, and a
        // verifier reporting the generic class would describe an issuer's honest
        // record of its own fail-open capture in the words used for a forgery.
        if parsed.capture_outcome.is_some() {
            return Err("capture-uncertified-leaf");
        }
        if parsed.unusable.contains(&expected.3) || !parsed.deltas.contains_key(&expected.3) {
            return Err("anchor-delta-unusable");
        }
        let (delta, law) = parsed.deltas[&expected.3].clone();
        let n = object
            .get("n")
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or("anchor-malformed-leaf")?;
        let lo = object
            .get("lo_ts")
            .and_then(Value::as_f64)
            .ok_or("anchor-malformed-leaf")?;
        let hi = object
            .get("hi_ts")
            .and_then(Value::as_f64)
            .ok_or("anchor-malformed-leaf")?;
        let membership = if kind == "table" {
            None
        } else {
            let membership = parsed.membership.ok_or("anchor-row-uncovered")?;
            if !membership_matches_summary(&membership, n, lo, hi) {
                return Err("anchor-row-uncovered");
            }
            Some(membership)
        };
        claims.push((
            membership,
            delta,
            law,
            object
                .get("value_digest")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            parsed.value_commitments.clone(),
        ));
    }
    let data = supplied
        .get(&(kind.to_owned(), key.to_owned()))
        .ok_or("anchor-no-data")?;
    let committed = decompress_deltas(input.get("deltas").ok_or("anchor-delta-unusable")?)
        .map_err(|_| "anchor-delta-unusable")?;
    if committed.len() != data.len() {
        return Err("anchor-length-mismatch");
    }
    if kind == "table" {
        let digest = commitment_digest("table", data).map_err(|_| "anchor-row-uncovered")?;
        let matched = claims
            .iter()
            .filter(|claim| claim.3 == digest)
            .collect::<Vec<_>>();
        let best = matched
            .into_iter()
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .ok_or("anchor-row-uncovered")?;
        for (value, committed) in data.values().iter().zip(&committed) {
            if *committed != Some(best.1) {
                return Err("anchor-delta-mismatch");
            }
            zero_delta_plausible(best.1, best.2.as_deref(), *value)
                .map_err(|_| "anchor-delta-zero-implausible")?;
        }
        return Ok(());
    }
    let CanonicalInput::TimeSeries { index, values } = data else {
        return Err("anchor-no-data");
    };

    // SPEC §7.7: a leaf may vouch for a row only if it still commits that row's
    // CURRENT value. This is a coverage test rather than a separate verdict, and
    // it is the same rule `attest.leaf_value_coverage` applies on the Python
    // side — the two verifiers must refuse the same signed bytes or the format
    // means whatever the implementation you happened to run means.
    let mut own_values: HashMap<u64, f64> = HashMap::with_capacity(index.len());
    for (timestamp, value) in index.iter().zip(values) {
        own_values.insert(normalized_bits(*timestamp), *value);
    }
    let mut absent: HashSet<u64> = HashSet::new();
    let mut unopenable: HashSet<u64> = HashSet::new();
    let mut mismatched: HashSet<u64> = HashSet::new();
    let mut covering = Vec::with_capacity(claims.len());
    for claim in &claims {
        let Some(membership) = claim.0.as_ref() else {
            continue;
        };
        let rows = membership_rows(membership);
        let keys = rows
            .iter()
            .map(|row| normalized_bits(value_row_timestamp(*row)))
            .collect::<Vec<_>>();
        let root = claim
            .4
            .as_ref()
            .and_then(|roots| roots.get(&expected.3).cloned());
        let Some(root) = root else {
            absent.extend(keys);
            continue;
        };
        let mut held = Vec::with_capacity(rows.len());
        let mut openable = true;
        for (row, key) in rows.iter().zip(&keys) {
            match own_values.get(key) {
                Some(value) => held.push((*row, Some(*value))),
                None => {
                    openable = false;
                    break;
                }
            }
        }
        if !openable {
            unopenable.extend(keys);
            continue;
        }
        let recomputed = value_column_root(
            &expected.0,
            &expected.1,
            &expected.2,
            &expected.3,
            &held,
        );
        if recomputed.as_deref() == Some(root.as_str()) {
            covering.push(claim);
        } else {
            mismatched.extend(keys);
        }
    }

    for ((timestamp, value), committed) in index.iter().zip(values).zip(committed) {
        let best = covering
            .iter()
            .filter(|claim| {
                claim
                    .0
                    .as_ref()
                    .is_some_and(|membership| membership_contains(membership, *timestamp))
            })
            .max_by(|left, right| left.1.total_cmp(&right.1));
        let Some(best) = best else {
            // Name the most specific cause, in the same order Python does: a row
            // that WAS covered by a timestamp and lost it to a value commitment
            // is a different fact from one no capture leaf ever touched.
            let key = normalized_bits(*timestamp);
            return Err(if mismatched.contains(&key) {
                "value-commitment-mismatch"
            } else if unopenable.contains(&key) {
                "value-commitment-unopenable"
            } else if absent.contains(&key) {
                "value-commitment-absent"
            } else {
                "anchor-row-uncovered"
            });
        };
        if committed != Some(best.1) {
            return Err("anchor-delta-mismatch");
        }
        zero_delta_plausible(best.1, best.2.as_deref(), *value)
            .map_err(|_| "anchor-delta-zero-implausible")?;
    }
    Ok(())
}

/// `-0.0` and `0.0` are the same instant. Keying a lookup on raw bits would make
/// them different rows, so normalise before taking bits.
fn normalized_bits(timestamp: f64) -> u64 {
    (timestamp + 0.0).to_bits()
}

fn membership_rows(membership: &Membership) -> Vec<ValueRow> {
    match membership {
        Membership::Bars(rows) => rows.iter().map(|row| ValueRow::Bars(*row)).collect(),
        Membership::Series(rows) => rows.iter().map(|row| ValueRow::Series(*row)).collect(),
    }
}

fn value_row_timestamp(row: ValueRow) -> f64 {
    match row {
        ValueRow::Bars(timestamp) => timestamp as f64,
        ValueRow::Series(timestamp) => timestamp,
    }
}

fn leaf_hash_from_record(
    object: &Map<String, Value>,
    table: &str,
    scope1: &str,
    scope2: &str,
) -> Option<String> {
    cert_leaf_hash(
        table,
        scope1,
        scope2,
        python_int(object.get("seq")?).ok()?,
        object.get("value_digest")?.as_str()?,
        python_int(object.get("n")?).ok()?,
        python_float(object.get("lo_ts")?).ok()?,
        python_float(object.get("hi_ts")?).ok()?,
        python_int(object.get("bits")?).ok()?,
        object.get("payload")?.as_str()?,
        object.get("prev_hash")?.as_str()?,
    )
}

fn zero_delta_plausible(delta: f64, law: Option<&str>, value: f64) -> Result<(), ()> {
    if delta != 0.0 || !value.is_finite() {
        return Ok(());
    }
    if law == Some("exact-cents/v0") {
        if value.abs() > SAFE_EXACT_INT || value.fract() != 0.0 {
            return Err(());
        }
    } else if value != 0.0 {
        return Err(());
    }
    Ok(())
}

fn check_witness(
    cne: &Map<String, Value>,
    pins: Option<&Map<String, Value>>,
    state: &mut VerdictState,
) {
    let mut cosigned = Vec::new();
    for input in input_commitments(cne) {
        let Some(block) = input.get("transparency").and_then(Value::as_object) else {
            continue;
        };
        if truthy(block.get("cosignature")) {
            cosigned.push((
                block.get("cosignature").unwrap(),
                block.get("sth").unwrap_or(&Value::Null),
            ));
        }
    }
    if cosigned.is_empty() {
        return;
    }
    let witness_key = pins
        .and_then(|pins| pins.get("witness_key_hex"))
        .and_then(Value::as_str);
    let Some(witness_key) = witness_key else {
        state.reason("witness-unpinned");
        return;
    };
    let verified = cosigned
        .iter()
        .filter(|(cosignature, sth)| verify_cosignature(cosignature, witness_key, sth))
        .count();
    if verified != cosigned.len() {
        state.set("witness", false);
        state.reason("witness-cosignature-invalid");
    } else if verified == input_commitments(cne).len() {
        state.set("witness", true);
    } else {
        state.reason("witness-partial");
    }
}

fn corroboration_scope(reference: &str) -> Option<(String, String)> {
    let (kind, key) = reference.split_once('|').unwrap_or(("price", reference));
    if !matches!(kind, "price" | "bars") {
        return None;
    }
    let (ticker, interval) = key.split_once('@').unwrap_or((key, "1d"));
    Some((ticker.to_uppercase(), interval.to_owned()))
}

fn check_provider(cne: &Map<String, Value>, pinned_key: Option<&str>, state: &mut VerdictState) {
    let provider = cne
        .get("error_budget")
        .and_then(Value::as_object)
        .and_then(|budget| budget.get("provider"))
        .and_then(Value::as_object);
    let Some(provider) = provider else {
        return;
    };
    let anchors = provider
        .get("anchors")
        .and_then(Value::as_object)
        .filter(|anchors| !anchors.is_empty());
    let Some(anchors) = anchors else {
        return;
    };
    let Some(pin) = pinned_key else {
        state.set("provider", false);
        state.reason("provider-no-pinned-key");
        return;
    };
    let stated = provider.get("inputs").and_then(Value::as_object);
    for (reference, block) in anchors {
        let claimed = stated.and_then(|stated| stated.get(reference));
        if let Err(class) = verify_one_provider_anchor(reference, block, claimed, pin) {
            state.set("provider", false);
            state.reason(class);
            return;
        }
    }
    state.set("provider", true);
}

fn verify_one_provider_anchor(
    reference: &str,
    block: &Value,
    stated: Option<&Value>,
    pin: &str,
) -> Result<(), &'static str> {
    let block = block.as_object().ok_or("provider-malformed-scope")?;
    let sth = block.get("sth").ok_or("provider-sth-invalid")?;
    if !verify_tree_head(sth, pin) {
        return Err("provider-sth-invalid");
    }
    let expected = corroboration_scope(reference).ok_or("provider-scope-mismatch")?;
    let scope = block
        .get("scope")
        .and_then(Value::as_array)
        .ok_or("provider-malformed-scope")?;
    if scope.len() != 2 {
        return Err("provider-malformed-scope");
    }
    if value_to_string(&scope[0]) != expected.0 || value_to_string(&scope[1]) != expected.1 {
        return Err("provider-scope-mismatch");
    }
    let sth_object = sth.as_object().ok_or("provider-sth-invalid")?;
    if sth_object.get("table").and_then(Value::as_str) != Some("corroboration")
        || sth_object
            .get("scope")
            .and_then(Value::as_array)
            .is_none_or(|scope| {
                scope.len() != 2
                    || value_to_string(&scope[0]) != expected.0
                    || value_to_string(&scope[1]) != expected.1
            })
    {
        return Err("provider-sth-scope-mismatch");
    }
    let root = sth_object
        .get("root")
        .and_then(Value::as_str)
        .ok_or("provider-sth-invalid")?;
    let tree_size = sth_object
        .get("tree_size")
        .and_then(Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or("provider-sth-invalid")?;
    let leaves = block
        .get("leaves")
        .and_then(Value::as_array)
        .filter(|leaves| !leaves.is_empty())
        .ok_or("provider-malformed-leaf")?;
    let attempts_by_seq = block
        .get("attempts")
        .and_then(Value::as_object)
        .ok_or("provider-malformed-leaf")?;
    let mut all_attempts = Vec::new();
    for leaf in leaves {
        let object = leaf.as_object().ok_or("provider-malformed-leaf")?;
        let recomputed = leaf_hash_from_record(object, "corroboration", &expected.0, &expected.1)
            .ok_or("provider-malformed-leaf")?;
        let proof = object
            .get("inclusion_proof")
            .and_then(Value::as_object)
            .ok_or("provider-malformed-leaf")?;
        if proof.get("leaf_hash").and_then(Value::as_str) != Some(recomputed.as_str()) {
            return Err("provider-leaf-hash-mismatch");
        }
        let proof_size = proof
            .get("tree_size")
            .ok_or("provider-malformed-leaf")
            .and_then(|value| python_int(value).map_err(|_| "provider-malformed-leaf"))?;
        let proof_size = usize::try_from(proof_size).unwrap_or(usize::MAX);
        if proof_size != tree_size {
            return Err("provider-proof-tree-size-mismatch");
        }
        let index = proof
            .get("index")
            .ok_or("provider-malformed-leaf")
            .and_then(|value| python_int(value).map_err(|_| "provider-malformed-leaf"))?;
        let index = usize::try_from(index).unwrap_or(usize::MAX);
        if index >= tree_size {
            return Err("provider-proof-index-out-of-range");
        }
        let path = proof
            .get("proof")
            .and_then(Value::as_array)
            .map(|values| string_array(values))
            .unwrap_or_default();
        if !verify_merkle_path(&recomputed, index, tree_size, &path, root) {
            return Err("provider-inclusion-failed");
        }
        let sequence = json_integer(object.get("seq").ok_or("provider-malformed-leaf")?)
            .ok_or("provider-malformed-leaf")?;
        let carried = attempts_by_seq
            .get(&sequence.to_string())
            .and_then(Value::as_array)
            .ok_or("provider-malformed-leaf")?;
        let mut attempts = Vec::new();
        for attempt in carried {
            let fields = attempt
                .as_array()
                .filter(|fields| fields.len() >= 4)
                .ok_or("provider-malformed-leaf")?;
            let value = if fields[3].is_null() {
                None
            } else {
                Some(python_float(&fields[3]).map_err(|_| "provider-malformed-leaf")?)
            };
            let attempt = ProviderAttempt {
                provider: value_to_string(&fields[0]),
                origin: value_to_string(&fields[1]),
                outcome: value_to_string(&fields[2]),
                value,
            };
            if !matches!(
                attempt.outcome.as_str(),
                "answered" | "unavailable" | "quality-rejected" | "error"
            ) {
                return Err("provider-outcome-unknown");
            }
            attempts.push(attempt);
        }
        let expected_count = json_integer(object.get("n").ok_or("provider-malformed-leaf")?)
            .ok_or("provider-malformed-leaf")?;
        if attempts.len() as i64 != expected_count {
            return Err("provider-attempt-count-mismatch");
        }
        if object.get("value_digest").and_then(Value::as_str)
            != Some(corroboration_digest(&attempts).as_str())
        {
            return Err("provider-attempts-digest-mismatch");
        }
        all_attempts.extend(attempts);
    }
    if let Some(stated) = stated.and_then(Value::as_object) {
        let asked = all_attempts.len() as i64;
        let answered = all_attempts
            .iter()
            .filter(|attempt| attempt.outcome == "answered")
            .count() as i64;
        let actual = [
            ("asked", asked),
            ("answered", answered),
            ("silent", asked - answered),
        ];
        for (name, value) in actual {
            if let Some(claimed) = stated.get(name) {
                if !claimed.is_null() && json_integer(claimed) != Some(value) {
                    return Err("provider-summary-mismatch");
                }
            }
        }
    }
    Ok(())
}

fn check_key_lifecycle(
    cne: &Map<String, Value>,
    pins: Option<&Map<String, Value>>,
    state: &mut VerdictState,
) {
    let Some(pins) = pins else {
        return;
    };
    let manifest = pins.get("key_manifest").filter(|value| !value.is_null());
    let Some(manifest) = manifest else {
        return;
    };
    let root = pins.get("manifest_root_hex").and_then(Value::as_str);
    if root.is_none() {
        state.set("key_status", false);
        state.reason("key-manifest-unrooted");
        return;
    }
    let checkpoint = pins
        .get("manifest_checkpoint")
        .filter(|value| !value.is_null());
    let checkpoint_key = pins
        .get("checkpoint_public_key_hex")
        .and_then(Value::as_str);
    let trusted = pins
        .get("trusted_manifest_checkpoint")
        .filter(|value| !value.is_null());
    let verified =
        match verify_manifest_checkpoint(manifest, checkpoint, root, checkpoint_key, trusted) {
            Ok(verified) => verified,
            Err(error) => {
                state.set("key_status", false);
                state.reason(match error.kind {
                    KeylifeErrorKind::RootRequired => "key-manifest-unrooted",
                    KeylifeErrorKind::CheckpointRequired => "key-manifest-checkpoint-required",
                    KeylifeErrorKind::ManifestInvalid => "key-manifest-invalid",
                    KeylifeErrorKind::CheckpointInvalid => "key-manifest-checkpoint-invalid",
                    KeylifeErrorKind::NotMonotonic => "key-manifest-checkpoint-not-monotonic",
                });
                return;
            }
        };
    let issued_at = match optional_python_float(cne.get("created")) {
        Ok(value) => value,
        Err(_) => {
            state.set("key_status", false);
            state.reason("key-manifest-invalid");
            return;
        }
    };
    let status = key_status_at(
        &verified.manifest,
        cne.get("key_id").and_then(Value::as_str),
        issued_at,
    )
    .0;
    match status {
        KeyStatus::Valid => state.set("key_status", true),
        KeyStatus::Revoked => {
            state.set("key_status", false);
            state.reason("key-revoked");
        }
        KeyStatus::Unknown => {
            state.set("key_status", false);
            state.reason("key-not-in-manifest");
        }
        KeyStatus::OutsideValidity => {
            state.set("key_status", false);
            state.reason("key-outside-validity");
        }
    }
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
        Some(Value::Number(value)) => value.as_f64() != Some(0.0),
    }
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) => value.to_string(),
        other => other.to_string(),
    }
}

fn string_array(values: &[Value]) -> Vec<String> {
    values.iter().map(value_to_string).collect()
}

fn json_integer(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
}

fn python_int_equals(value: &Value, expected: i64) -> Result<bool, ()> {
    python_int(value).map(|integer| integer == i128::from(expected))
}

fn python_int(value: &Value) -> Result<i128, ()> {
    match value {
        Value::Bool(value) => Ok(i128::from(*value)),
        Value::Number(value) => {
            let rendered = value.to_string();
            if !rendered.contains(['.', 'e', 'E']) {
                return rendered.parse::<i128>().map_err(|_| ());
            }
            let number = value.as_f64().ok_or(())?;
            if !number.is_finite() || number < i128::MIN as f64 || number > i128::MAX as f64 {
                return Err(());
            }
            Ok(number.trunc() as i128)
        }
        Value::String(value) => {
            let value = value.trim();
            if value.is_empty() {
                return Err(());
            }
            value.parse::<i128>().map_err(|_| ())
        }
        Value::Null | Value::Array(_) | Value::Object(_) => Err(()),
    }
}

fn python_float(value: &Value) -> Result<f64, ()> {
    match value {
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Number(value) => value.as_f64().ok_or(()),
        Value::String(value) => value.trim().parse::<f64>().map_err(|_| ()),
        Value::Null | Value::Array(_) | Value::Object(_) => Err(()),
    }
}

fn optional_python_float(value: Option<&Value>) -> Result<Option<f64>, ()> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => python_float(value).map(Some),
    }
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_hex_bytes_valid(value: &str) -> bool {
    let digits = value
        .chars()
        .filter(|ch| !ch.is_ascii_whitespace())
        .collect::<String>();
    digits.len() % 2 == 0 && digits.chars().all(|ch| ch.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::TestKernel;

    /// The public build replays nothing: a golden receipt that passes on the
    /// specified kernel is not passed here, its replay
    /// slots are null rather than true or false, and no reason class is invented
    /// for their absence. Signature, inputs and transparency are still checked.
    #[test]
    fn without_a_kernel_a_golden_receipt_is_not_replayed_and_does_not_pass() {
        let case: Value =
            serde_json::from_str(include_str!("../tests/vectors/golden-anchored.json")).unwrap();
        assert_eq!(
            case["expect"]["ok"], true,
            "the vector passes on its kernel"
        );
        let result = verify_case(&case, None);
        assert_eq!(result["ok"], false);
        for slot in ["scalar", "tier", "budget", "width"] {
            assert!(result["checks"][slot].is_null(), "{slot} is not performed");
        }
        for slot in ["authenticity", "inputs", "transparency"] {
            assert_eq!(result["checks"][slot], true, "{slot} is still checked");
        }
        assert_eq!(result["reason_classes"], json!([]));
        assert_eq!(result["width_trust"], "unverified");
    }

    #[test]
    fn unknown_version_fails_before_any_check_runs() {
        let result = verify_envelope(
            &json!({"type": "alelyon.cne/v1"}),
            None,
            None,
            Some(&TestKernel),
        );
        assert_eq!(result["ok"], false);
        assert_eq!(result["reason_classes"], json!(["not-a-cne-v0"]));
        assert!(
            result["checks"]
                .as_object()
                .unwrap()
                .values()
                .all(Value::is_null)
        );
    }

    #[test]
    fn no_data_and_no_pin_are_explicit() {
        let result = verify_envelope(
            &json!({"type": ENVELOPE_TYPE, "refused": false}),
            None,
            None,
            Some(&TestKernel),
        );
        assert_eq!(result["ok"], false);
        assert_eq!(result["checks"]["authenticity"], false);
        assert!(
            result["reason_classes"]
                .as_array()
                .unwrap()
                .contains(&json!("no-input-data"))
        );
        assert!(
            result["reason_classes"]
                .as_array()
                .unwrap()
                .contains(&json!("no-pinned-key"))
        );
    }

    #[test]
    fn replay_parameters_reject_python_style_type_coercion() {
        let invalid = [
            json!({"params": {"K": -1}}),
            json!({"params": {"K": 63.0}}),
            json!({"params": {"K": true}}),
            json!({"params": {"K": "63"}}),
            json!({"params": {"alpha": true}}),
            json!({"params": {"alpha": "0.05"}}),
            json!({"params": {"strict": 1}}),
            json!({"params": {"strict": "false"}}),
            json!({"params": {"require_tier": null}}),
            json!({"params": {"require_tier": 1}}),
        ];
        for envelope in invalid {
            assert!(replay_params(envelope.as_object().unwrap()).is_err());
        }

        let valid = json!({
            "params": {
                "K": 63,
                "alpha": 0.05,
                "strict": true,
                "require_tier": "linear-exact"
            },
            "seed": 7
        });
        let parsed = replay_params(valid.as_object().unwrap()).unwrap();
        assert_eq!(parsed.k, 63);
        assert_eq!(parsed.alpha, 0.05);
        assert!(parsed.strict);
        assert_eq!(parsed.require_tier.as_deref(), Some("linear-exact"));
        assert_eq!(parsed.seed, Some(7));
    }

    #[test]
    fn non_text_program_is_a_replay_refusal_not_a_coerced_program() {
        let input = CanonicalInput::TimeSeries {
            index: vec![1.0],
            values: vec![2.0],
        };
        let digest = commitment_digest("price", &input).unwrap();
        let envelope = json!({
            "type": ENVELOPE_TYPE,
            "program": -5464422301462229765_i64,
            "program_class": "linear-exact",
            "refused": false,
            "seed": 7,
            "inputs": [{
                "kind": "price",
                "key": "SYN",
                "digest": digest,
                "deltas": {"const": 0.0, "n": 1, "uncertified_idx": []}
            }],
            "error_budget": {
                "quantization": {
                    "level": null,
                    "exact": false,
                    "tier": "?",
                    "width": null
                }
            },
            "assumptions": [],
            "branch_sites": []
        });
        let supplied = json!({
            "price|SYN": {"index": [1.0], "values": [2.0]}
        });

        let result = verify_envelope(&envelope, Some(&supplied), None, Some(&TestKernel));

        assert_eq!(result["checks"]["tier"], false);
        assert!(
            result["reason_classes"]
                .as_array()
                .unwrap()
                .contains(&json!("tier-mismatch"))
        );
    }
}
