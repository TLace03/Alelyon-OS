//! Independent verification of the CNE v0 key-lifecycle protocol (SPEC §6.8).
//!
//! The manifest is transport, not authority. Authority comes from a root pinned
//! out of band, immediate-predecessor succession signatures, authorized
//! revocations, and a separately signed checkpoint compared with verifier-held
//! rollback state. This module verifies those objects; it never creates, signs,
//! fetches, or persists them.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde_json::{Map, Value};

use crate::canonical::{canonical_json, canonical_json_without_signature};
use crate::crypto::{
    decode_lower_hex, key_id_from_public_key_hex, sha256_hex, verify_ed25519_strict,
};

pub const SUCCESSION_TYPE: &str = "alelyon.keysuccession/v0";
pub const REVOCATION_TYPE: &str = "alelyon.keyrevocation/v0";
pub const MANIFEST_TYPE: &str = "alelyon.keymanifest/v0";
pub const CHECKPOINT_TYPE: &str = "alelyon.keymanifest-checkpoint/v0";

/// Key-history arrays are intentionally small in practice. Bound them before
/// canonicalization or signature work so an untrusted manifest cannot become an
/// allocation/CPU amplifier.
pub const MAX_CONTAINER_ITEMS: usize = 4_096;
pub const MAX_JSON_DEPTH: usize = 64;
pub const MAX_JSON_NODES: usize = 100_000;
pub const MAX_STRING_BYTES: usize = 1_048_576;

const STATUSES: [&str; 3] = ["active", "superseded", "revoked"];
const REVOCATION_REASONS: [&str; 4] = ["compromise", "retired", "superseded-early", "lost"];
const KNOWN_ENTRY_FIELDS: [&str; 7] = [
    "key_id",
    "public_key",
    "not_before",
    "not_after",
    "status",
    "succession",
    "revocation",
];

/// Stable failure categories for integration with the envelope reason classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeylifeErrorKind {
    RootRequired,
    CheckpointRequired,
    ManifestInvalid,
    CheckpointInvalid,
    NotMonotonic,
}

/// A fail-closed lifecycle-verification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeylifeError {
    pub kind: KeylifeErrorKind,
    pub reason: String,
}

impl KeylifeError {
    fn new(kind: KeylifeErrorKind, reason: impl Into<String>) -> Self {
        Self {
            kind,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for KeylifeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason)
    }
}

impl std::error::Error for KeylifeError {}

/// A manifest whose complete chain and revocations verified under the pinned root.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedManifest {
    pub issuer: String,
    pub root_key_id: String,
    pub published_at: f64,
    pub order: Vec<String>,
    pub keys: BTreeMap<String, Value>,
}

impl VerifiedManifest {
    pub fn entry(&self, key_id: &str) -> Option<&Value> {
        self.keys.get(key_id)
    }
}

/// Successful checkpoint verification. `next_checkpoint` is the exact signed
/// object the caller may atomically retain as its new rollback state.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedCheckpoint {
    pub manifest: VerifiedManifest,
    pub next_checkpoint: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyStatus {
    Valid,
    Revoked,
    OutsideValidity,
    Unknown,
}

#[derive(Debug, Clone)]
struct EntryMeta {
    key_id: String,
    public_key: String,
    not_before: f64,
    status: String,
}

#[derive(Debug, Clone)]
struct CheckedCheckpoint {
    issuer: String,
    root_key_id: String,
    sequence: String,
    manifest_published_at: f64,
    issued_at: f64,
    entries: Vec<Value>,
    key_ids: Vec<String>,
    superseded_key_ids: Vec<String>,
    revoked_key_ids: Vec<String>,
}

/// Verify a complete key manifest against a root obtained out of band.
pub fn verify_key_manifest(
    manifest: &Value,
    root_public_key_hex: Option<&str>,
) -> Result<VerifiedManifest, KeylifeError> {
    let root_public_key_hex = root_public_key_hex.ok_or_else(|| {
        KeylifeError::new(
            KeylifeErrorKind::RootRequired,
            "no pinned root key; a manifest verified against nothing vouches for nothing",
        )
    })?;
    enforce_resource_bounds(manifest, KeylifeErrorKind::ManifestInvalid, "manifest")?;

    let object = expect_object(manifest, KeylifeErrorKind::ManifestInvalid, "manifest")?;
    require_type(
        object,
        MANIFEST_TYPE,
        KeylifeErrorKind::ManifestInvalid,
        "manifest",
    )?;
    let issuer = nonempty_string(
        object.get("issuer"),
        KeylifeErrorKind::ManifestInvalid,
        "manifest issuer",
    )?
    .to_owned();
    let published_at = finite_number(
        object.get("published_at"),
        KeylifeErrorKind::ManifestInvalid,
        "manifest published_at",
    )?;
    let root_key_id = key_id_from_public_key_hex(root_public_key_hex).map_err(|error| {
        KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("pinned root key is not valid lowercase hex: {error}"),
        )
    })?;
    let carried_root = nonempty_string(
        object.get("root_key_id"),
        KeylifeErrorKind::ManifestInvalid,
        "manifest root_key_id",
    )?;
    if carried_root != root_key_id {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            "manifest root_key_id does not match the pinned root",
        ));
    }

    let entries = bounded_nonempty_array(
        object.get("keys"),
        KeylifeErrorKind::ManifestInvalid,
        "manifest keys",
    )?;
    let mut metas = Vec::with_capacity(entries.len());
    let mut seen = BTreeSet::new();
    for (index, entry) in entries.iter().enumerate() {
        let map = expect_object(
            entry,
            KeylifeErrorKind::ManifestInvalid,
            &format!("manifest entry {index}"),
        )?;
        let key_id = nonempty_string(
            map.get("key_id"),
            KeylifeErrorKind::ManifestInvalid,
            &format!("manifest entry {index} key_id"),
        )?
        .to_owned();
        if !seen.insert(key_id.clone()) {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} duplicates key_id {key_id:?}"),
            ));
        }
        let public_key = nonempty_string(
            map.get("public_key"),
            KeylifeErrorKind::ManifestInvalid,
            &format!("manifest entry {index} public_key"),
        )?
        .to_owned();
        let derived = key_id_from_public_key_hex(&public_key).map_err(|error| {
            KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} public_key is invalid: {error}"),
            )
        })?;
        if derived != key_id {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} key_id does not identify its public_key"),
            ));
        }
        let not_before = finite_number(
            map.get("not_before"),
            KeylifeErrorKind::ManifestInvalid,
            &format!("manifest entry {index} not_before"),
        )?;
        let not_after = optional_finite_number(
            map.get("not_after"),
            KeylifeErrorKind::ManifestInvalid,
            &format!("manifest entry {index} not_after"),
        )?;
        if not_after.is_some_and(|end| end < not_before) {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} not_after precedes not_before"),
            ));
        }
        let status = nonempty_string(
            map.get("status"),
            KeylifeErrorKind::ManifestInvalid,
            &format!("manifest entry {index} status"),
        )?;
        if !STATUSES.contains(&status) {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} has unknown status {status:?}"),
            ));
        }
        let has_revocation = map.get("revocation").is_some_and(|value| !value.is_null());
        if (status == "revoked") != has_revocation {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} status {status:?} disagrees with its revocation"),
            ));
        }
        if status == "superseded" && not_after.is_none() {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} is superseded but has no not_after"),
            ));
        }
        metas.push(EntryMeta {
            key_id,
            public_key,
            not_before,
            status: status.to_owned(),
        });
    }

    if metas[0].key_id != root_key_id || metas[0].public_key != root_public_key_hex {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            "the manifest's first key is not the pinned root",
        ));
    }
    let root_entry = entries[0]
        .as_object()
        .expect("entries were validated above");
    if root_entry
        .get("succession")
        .is_some_and(|value| !value.is_null())
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            "the root key must not carry a succession statement",
        ));
    }

    for index in 1..entries.len() {
        if metas[index].not_before < metas[index - 1].not_before {
            return Err(KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!("manifest entry {index} not_before precedes its predecessor"),
            ));
        }
        verify_succession(
            entries[index]
                .as_object()
                .expect("entries were validated above")
                .get("succession"),
            &metas[index - 1],
            &metas[index],
            index,
        )?;
    }

    for (index, entry) in entries.iter().enumerate() {
        let map = entry.as_object().expect("entries were validated above");
        if let Some(revocation) = map.get("revocation").filter(|value| !value.is_null()) {
            verify_revocation(revocation, &metas, index)?;
        }
    }

    let active: Vec<usize> = metas
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| (entry.status == "active").then_some(index))
        .collect();
    if active.len() > 1 {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            "manifest has more than one active key",
        ));
    }
    if active
        .first()
        .is_some_and(|index| *index != metas.len() - 1)
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            "only the final key in the succession chain may be active",
        ));
    }

    let order = metas.iter().map(|entry| entry.key_id.clone()).collect();
    let keys = metas
        .iter()
        .zip(entries)
        .map(|(meta, entry)| (meta.key_id.clone(), entry.clone()))
        .collect();
    Ok(VerifiedManifest {
        issuer,
        root_key_id,
        published_at,
        order,
        keys,
    })
}

/// Verify a manifest and candidate checkpoint relative to retained signed state.
pub fn verify_manifest_checkpoint(
    manifest: &Value,
    checkpoint: Option<&Value>,
    root_public_key_hex: Option<&str>,
    checkpoint_public_key_hex: Option<&str>,
    trusted_checkpoint: Option<&Value>,
) -> Result<VerifiedCheckpoint, KeylifeError> {
    if root_public_key_hex.is_none() {
        return Err(KeylifeError::new(
            KeylifeErrorKind::RootRequired,
            "key manifest was supplied without a pinned root",
        ));
    }
    let (checkpoint, checkpoint_public_key_hex, trusted_checkpoint) = match (
        checkpoint,
        checkpoint_public_key_hex,
        trusted_checkpoint,
    ) {
        (Some(checkpoint), Some(key), Some(trusted)) => (checkpoint, key, trusted),
        _ => {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointRequired,
                "manifest freshness requires a candidate checkpoint, a separately pinned checkpoint key, and retained signed checkpoint state",
            ));
        }
    };

    let verified_manifest =
        verify_key_manifest(manifest, root_public_key_hex).map_err(|error| {
            if error.kind == KeylifeErrorKind::RootRequired {
                error
            } else {
                KeylifeError::new(
                    KeylifeErrorKind::ManifestInvalid,
                    format!("key manifest is invalid: {}", error.reason),
                )
            }
        })?;
    let current = check_checkpoint(
        checkpoint,
        checkpoint_public_key_hex,
        Some(manifest),
        "candidate checkpoint",
    )?;
    let trusted = check_checkpoint(
        trusted_checkpoint,
        checkpoint_public_key_hex,
        None,
        "trusted checkpoint",
    )?;
    verify_monotonic(checkpoint, &current, trusted_checkpoint, &trusted)?;

    Ok(VerifiedCheckpoint {
        manifest: verified_manifest,
        next_checkpoint: checkpoint.clone(),
    })
}

/// Place a key at an issuance time after manifest/checkpoint verification.
///
/// Revocation ignores issuance time deliberately: compromise may predate its
/// discovery, so a revoked key never receives a bare valid status.
pub fn key_status_at<'a>(
    manifest: &'a VerifiedManifest,
    key_id: Option<&str>,
    issued_at: Option<f64>,
) -> (KeyStatus, Option<&'a Value>) {
    let Some(entry) = key_id.and_then(|key| manifest.keys.get(key)) else {
        return (KeyStatus::Unknown, None);
    };
    let Some(map) = entry.as_object() else {
        return (KeyStatus::Unknown, None);
    };
    if map.get("status").and_then(Value::as_str) == Some("revoked") {
        return (KeyStatus::Revoked, Some(entry));
    }
    let Some(issued_at) = issued_at.filter(|value| value.is_finite()) else {
        return (KeyStatus::OutsideValidity, Some(entry));
    };
    let Some(not_before) = map.get("not_before").and_then(Value::as_f64) else {
        return (KeyStatus::OutsideValidity, Some(entry));
    };
    if issued_at < not_before {
        return (KeyStatus::OutsideValidity, Some(entry));
    }
    if map
        .get("not_after")
        .filter(|value| !value.is_null())
        .and_then(Value::as_f64)
        .is_some_and(|not_after| issued_at > not_after)
    {
        return (KeyStatus::OutsideValidity, Some(entry));
    }
    (KeyStatus::Valid, Some(entry))
}

fn verify_succession(
    succession: Option<&Value>,
    predecessor: &EntryMeta,
    successor: &EntryMeta,
    index: usize,
) -> Result<(), KeylifeError> {
    let succession = succession.ok_or_else(|| {
        KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("manifest entry {index} has no succession statement"),
        )
    })?;
    let object = expect_object(
        succession,
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {index} succession"),
    )?;
    require_type(
        object,
        SUCCESSION_TYPE,
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {index} succession"),
    )?;
    if object.get("predecessor_key_id").and_then(Value::as_str) != Some(predecessor.key_id.as_str())
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("manifest entry {index} succession does not name its immediate predecessor"),
        ));
    }
    if object.get("key_id").and_then(Value::as_str) != Some(successor.key_id.as_str())
        || object.get("public_key").and_then(Value::as_str) != Some(successor.public_key.as_str())
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("manifest entry {index} succession attests a different key"),
        ));
    }
    let not_before = finite_number(
        object.get("not_before"),
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {index} succession not_before"),
    )?;
    if not_before != successor.not_before {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("manifest entry {index} succession not_before disagrees with the entry"),
        ));
    }
    verify_signed_object(
        succession,
        &predecessor.public_key,
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {index} succession"),
    )
}

fn verify_revocation(
    revocation: &Value,
    entries: &[EntryMeta],
    revoked_index: usize,
) -> Result<(), KeylifeError> {
    let object = expect_object(
        revocation,
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {revoked_index} revocation"),
    )?;
    require_type(
        object,
        REVOCATION_TYPE,
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {revoked_index} revocation"),
    )?;
    if object.get("key_id").and_then(Value::as_str) != Some(entries[revoked_index].key_id.as_str())
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("manifest entry {revoked_index} revocation names a different key"),
        ));
    }
    let reason = nonempty_string(
        object.get("reason"),
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {revoked_index} revocation reason"),
    )?;
    if !REVOCATION_REASONS.contains(&reason) {
        return Err(KeylifeError::new(
            KeylifeErrorKind::ManifestInvalid,
            format!("manifest entry {revoked_index} has unknown revocation reason {reason:?}"),
        ));
    }
    finite_number(
        object.get("revoked_at"),
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {revoked_index} revoked_at"),
    )?;
    let signer_id = nonempty_string(
        object.get("signer_key_id"),
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {revoked_index} revocation signer_key_id"),
    )?;
    let signer = entries[revoked_index..]
        .iter()
        .find(|entry| entry.key_id == signer_id)
        .ok_or_else(|| {
            KeylifeError::new(
                KeylifeErrorKind::ManifestInvalid,
                format!(
                    "manifest entry {revoked_index} revocation signer is neither that key nor a successor"
                ),
            )
        })?;
    verify_signed_object(
        revocation,
        &signer.public_key,
        KeylifeErrorKind::ManifestInvalid,
        &format!("manifest entry {revoked_index} revocation"),
    )
}

fn check_checkpoint(
    checkpoint: &Value,
    checkpoint_public_key_hex: &str,
    manifest: Option<&Value>,
    label: &str,
) -> Result<CheckedCheckpoint, KeylifeError> {
    enforce_resource_bounds(checkpoint, KeylifeErrorKind::CheckpointInvalid, label)?;
    let object = expect_object(checkpoint, KeylifeErrorKind::CheckpointInvalid, label)?;
    require_type(
        object,
        CHECKPOINT_TYPE,
        KeylifeErrorKind::CheckpointInvalid,
        label,
    )?;
    let issuer = nonempty_string(
        object.get("issuer"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} issuer"),
    )?
    .to_owned();
    let root_key_id = nonempty_string(
        object.get("root_key_id"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} root_key_id"),
    )?
    .to_owned();
    let sequence = positive_integer(
        object.get("sequence"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} sequence"),
    )?;
    let manifest_digest = nonempty_string(
        object.get("manifest_digest"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} manifest_digest"),
    )?;
    decode_lower_hex::<32>(manifest_digest).map_err(|error| {
        KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} manifest_digest is invalid: {error}"),
        )
    })?;
    let manifest_published_at = finite_number(
        object.get("manifest_published_at"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} manifest_published_at"),
    )?;
    let issued_at = finite_number(
        object.get("issued_at"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} issued_at"),
    )?;
    if issued_at < manifest_published_at {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} issued_at precedes manifest_published_at"),
        ));
    }
    let entries = bounded_nonempty_array(
        object.get("entries"),
        KeylifeErrorKind::CheckpointInvalid,
        &format!("{label} entries"),
    )?
    .to_vec();
    let key_ids = unique_string_array(object.get("key_ids"), label, "key_ids")?;
    if key_ids.is_empty() {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} key_ids must not be empty"),
        ));
    }
    let superseded_key_ids = unique_string_array(
        object.get("superseded_key_ids"),
        label,
        "superseded_key_ids",
    )?;
    let revoked_key_ids =
        unique_string_array(object.get("revoked_key_ids"), label, "revoked_key_ids")?;
    if entries.len() != key_ids.len() {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} entries do not match ordered key_ids"),
        ));
    }

    let mut expected_superseded = Vec::new();
    let mut expected_revoked = Vec::new();
    let mut entry_public_keys = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let entry = expect_object(
            entry,
            KeylifeErrorKind::CheckpointInvalid,
            &format!("{label} entry {index}"),
        )?;
        let key_id = nonempty_string(
            entry.get("key_id"),
            KeylifeErrorKind::CheckpointInvalid,
            &format!("{label} entry {index} key_id"),
        )?;
        if key_id != key_ids[index] {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} entries do not match ordered key_ids"),
            ));
        }
        let public_key = nonempty_string(
            entry.get("public_key"),
            KeylifeErrorKind::CheckpointInvalid,
            &format!("{label} entry {index} public_key"),
        )?;
        let derived = key_id_from_public_key_hex(public_key).map_err(|error| {
            KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} entry {index} public_key is invalid: {error}"),
            )
        })?;
        if derived != key_id {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} entry {index} key_id does not identify public_key"),
            ));
        }
        entry_public_keys.push(public_key.to_owned());
        match entry.get("status").and_then(Value::as_str) {
            Some("active") => {}
            Some("superseded") => expected_superseded.push(key_id.to_owned()),
            Some("revoked") => expected_revoked.push(key_id.to_owned()),
            other => {
                return Err(KeylifeError::new(
                    KeylifeErrorKind::CheckpointInvalid,
                    format!("{label} entry {index} has unknown status {other:?}"),
                ));
            }
        }
    }
    if root_key_id != key_ids[0] {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} root_key_id does not identify its first entry"),
        ));
    }
    if superseded_key_ids != expected_superseded || revoked_key_ids != expected_revoked {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} status summaries disagree with its entries"),
        ));
    }
    let known: BTreeSet<&str> = key_ids.iter().map(String::as_str).collect();
    if superseded_key_ids
        .iter()
        .chain(&revoked_key_ids)
        .any(|key| !known.contains(key.as_str()))
        || superseded_key_ids
            .iter()
            .any(|key| revoked_key_ids.contains(key))
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} status arrays are inconsistent"),
        ));
    }

    let checkpoint_key_id =
        key_id_from_public_key_hex(checkpoint_public_key_hex).map_err(|error| {
            KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("pinned checkpoint key is invalid: {error}"),
            )
        })?;
    if object.get("checkpoint_key_id").and_then(Value::as_str) != Some(checkpoint_key_id.as_str()) {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} checkpoint_key_id does not match the pinned key"),
        ));
    }
    if known.contains(checkpoint_key_id.as_str())
        || entry_public_keys
            .iter()
            .any(|public_key| public_key == checkpoint_public_key_hex)
    {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} reuses a manifest signing key as the checkpoint key"),
        ));
    }

    verify_signed_object(
        checkpoint,
        checkpoint_public_key_hex,
        KeylifeErrorKind::CheckpointInvalid,
        label,
    )?;

    if let Some(manifest) = manifest {
        let manifest_object = expect_object(
            manifest,
            KeylifeErrorKind::CheckpointInvalid,
            "checkpoint manifest",
        )?;
        let manifest_bytes = canonical_json(manifest).map_err(|error| {
            KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("manifest is not canonically encodable: {error}"),
            )
        })?;
        if manifest_digest != sha256_hex(&manifest_bytes) {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} manifest_digest does not commit this manifest"),
            ));
        }
        if manifest_object.get("issuer").and_then(Value::as_str) != Some(issuer.as_str())
            || manifest_object.get("root_key_id").and_then(Value::as_str)
                != Some(root_key_id.as_str())
        {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} issuer or root does not match the manifest"),
            ));
        }
        let manifest_time = finite_number(
            manifest_object.get("published_at"),
            KeylifeErrorKind::CheckpointInvalid,
            "manifest published_at",
        )?;
        if manifest_time != manifest_published_at {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} published_at does not match the manifest"),
            ));
        }
        let manifest_entries = bounded_nonempty_array(
            manifest_object.get("keys"),
            KeylifeErrorKind::CheckpointInvalid,
            "manifest keys",
        )?;
        if !canonical_equal(
            &Value::Array(entries.clone()),
            &Value::Array(manifest_entries.to_vec()),
        )? || key_ids
            != manifest_entries
                .iter()
                .map(|entry| {
                    entry
                        .get("key_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                })
                .collect::<Vec<_>>()
        {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} entries or key_ids do not match the manifest"),
            ));
        }
    }

    Ok(CheckedCheckpoint {
        issuer,
        root_key_id,
        sequence,
        manifest_published_at,
        issued_at,
        entries,
        key_ids,
        superseded_key_ids,
        revoked_key_ids,
    })
}

fn verify_monotonic(
    current_value: &Value,
    current: &CheckedCheckpoint,
    trusted_value: &Value,
    trusted: &CheckedCheckpoint,
) -> Result<(), KeylifeError> {
    if current.issuer != trusted.issuer || current.root_key_id != trusted.root_key_id {
        return Err(not_monotonic("checkpoint changed issuer or root identity"));
    }
    match compare_positive_integers(&current.sequence, &trusted.sequence) {
        Ordering::Less => {
            return Err(not_monotonic(format!(
                "checkpoint sequence rolled back from {} to {}",
                trusted.sequence, current.sequence
            )));
        }
        Ordering::Equal => {
            if !canonical_equal(current_value, trusted_value)? {
                return Err(not_monotonic(
                    "a different checkpoint was presented at an already trusted sequence",
                ));
            }
            return Ok(());
        }
        Ordering::Greater => {}
    }

    if current.key_ids.len() < trusted.key_ids.len()
        || current.key_ids[..trusted.key_ids.len()] != trusted.key_ids
    {
        return Err(not_monotonic(
            "checkpoint truncated or rewrote the trusted succession prefix",
        ));
    }
    for (previous, next) in trusted.entries.iter().zip(&current.entries) {
        verify_entry_transition(previous, next)?;
    }
    let current_revoked: BTreeSet<&str> =
        current.revoked_key_ids.iter().map(String::as_str).collect();
    if trusted
        .revoked_key_ids
        .iter()
        .any(|key| !current_revoked.contains(key.as_str()))
    {
        return Err(not_monotonic(
            "checkpoint stripped a previously trusted revocation",
        ));
    }
    let still_retired: BTreeSet<&str> = current
        .superseded_key_ids
        .iter()
        .chain(&current.revoked_key_ids)
        .map(String::as_str)
        .collect();
    if trusted
        .superseded_key_ids
        .iter()
        .any(|key| !still_retired.contains(key.as_str()))
    {
        return Err(not_monotonic(
            "checkpoint restored a previously retired key to active service",
        ));
    }
    if current.manifest_published_at < trusted.manifest_published_at {
        return Err(not_monotonic(
            "checkpoint moved manifest publication time backwards",
        ));
    }
    if current.issued_at < trusted.issued_at {
        return Err(not_monotonic("checkpoint issuance time moved backwards"));
    }
    Ok(())
}

fn verify_entry_transition(previous: &Value, next: &Value) -> Result<(), KeylifeError> {
    let previous = expect_object(
        previous,
        KeylifeErrorKind::NotMonotonic,
        "trusted checkpoint entry",
    )?;
    let next = expect_object(
        next,
        KeylifeErrorKind::NotMonotonic,
        "candidate checkpoint entry",
    )?;
    let key_id = previous
        .get("key_id")
        .and_then(Value::as_str)
        .unwrap_or("<unknown>");
    for field in ["key_id", "public_key", "not_before", "succession"] {
        if !canonical_optional_equal(previous.get(field), next.get(field))? {
            return Err(not_monotonic(format!(
                "checkpoint rewrote prior key {key_id}'s {field}"
            )));
        }
    }

    let old_status = previous.get("status").and_then(Value::as_str);
    let new_status = next.get("status").and_then(Value::as_str);
    let permitted = matches!(
        (old_status, new_status),
        (Some("active"), Some("active" | "superseded" | "revoked"))
            | (Some("superseded"), Some("superseded" | "revoked"))
            | (Some("revoked"), Some("revoked"))
    );
    if !permitted {
        return Err(not_monotonic(format!(
            "checkpoint moved prior key {key_id} from {old_status:?} to {new_status:?}"
        )));
    }

    let old_end = previous.get("not_after").filter(|value| !value.is_null());
    let new_end = next.get("not_after").filter(|value| !value.is_null());
    if old_end.is_some() && !canonical_optional_equal(old_end, new_end)? {
        return Err(not_monotonic(format!(
            "checkpoint rewrote prior key {key_id}'s not_after"
        )));
    }
    if old_end.is_none() && new_end.is_some() && new_status == Some("active") {
        return Err(not_monotonic(format!(
            "checkpoint gave active key {key_id} a closed validity window"
        )));
    }

    let old_revocation = previous.get("revocation").filter(|value| !value.is_null());
    let new_revocation = next.get("revocation").filter(|value| !value.is_null());
    if old_revocation.is_some() && !canonical_optional_equal(old_revocation, new_revocation)? {
        return Err(not_monotonic(format!(
            "checkpoint removed or rewrote prior key {key_id}'s revocation"
        )));
    }
    if old_revocation.is_none() && new_revocation.is_some() && new_status != Some("revoked") {
        return Err(not_monotonic(format!(
            "checkpoint added revocation evidence without revoking key {key_id}"
        )));
    }

    for (field, value) in previous {
        if KNOWN_ENTRY_FIELDS.contains(&field.as_str()) {
            continue;
        }
        let Some(next_value) = next.get(field) else {
            return Err(not_monotonic(format!(
                "checkpoint stripped prior key {key_id}'s extension {field:?}"
            )));
        };
        if !canonical_equal(value, next_value)? {
            return Err(not_monotonic(format!(
                "checkpoint rewrote prior key {key_id}'s extension {field:?}"
            )));
        }
    }
    Ok(())
}

fn verify_signed_object(
    value: &Value,
    public_key_hex: &str,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<(), KeylifeError> {
    let signature = value
        .get("signature")
        .and_then(Value::as_str)
        .ok_or_else(|| KeylifeError::new(kind, format!("{label} has no signature")))?;
    let bytes = canonical_json_without_signature(value)
        .map_err(|error| KeylifeError::new(kind, format!("{label} is not canonical: {error}")))?;
    match verify_ed25519_strict(public_key_hex, &bytes, signature) {
        Ok(true) => Ok(()),
        Ok(false) => Err(KeylifeError::new(
            kind,
            format!("{label} signature is not valid under its pinned/authorized key"),
        )),
        Err(error) => Err(KeylifeError::new(
            kind,
            format!("{label} carries malformed cryptographic material: {error}"),
        )),
    }
}

fn expect_object<'a>(
    value: &'a Value,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<&'a Map<String, Value>, KeylifeError> {
    value
        .as_object()
        .ok_or_else(|| KeylifeError::new(kind, format!("{label} is not an object")))
}

fn require_type(
    object: &Map<String, Value>,
    expected: &str,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<(), KeylifeError> {
    if object.get("type").and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(KeylifeError::new(
            kind,
            format!("{label} type is not {expected}"),
        ))
    }
}

fn nonempty_string<'a>(
    value: Option<&'a Value>,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<&'a str, KeylifeError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| KeylifeError::new(kind, format!("{label} must be a non-empty string")))
}

fn finite_number(
    value: Option<&Value>,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<f64, KeylifeError> {
    value
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or_else(|| KeylifeError::new(kind, format!("{label} must be a finite number")))
}

fn optional_finite_number(
    value: Option<&Value>,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<Option<f64>, KeylifeError> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => finite_number(Some(value), kind, label).map(Some),
    }
}

fn bounded_nonempty_array<'a>(
    value: Option<&'a Value>,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<&'a [Value], KeylifeError> {
    let values = value
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty())
        .ok_or_else(|| KeylifeError::new(kind, format!("{label} must be a non-empty array")))?;
    if values.len() > MAX_CONTAINER_ITEMS {
        return Err(KeylifeError::new(
            kind,
            format!("{label} exceeds the {MAX_CONTAINER_ITEMS}-item resource limit"),
        ));
    }
    Ok(values)
}

fn unique_string_array(
    value: Option<&Value>,
    label: &str,
    field: &str,
) -> Result<Vec<String>, KeylifeError> {
    let values = value.and_then(Value::as_array).ok_or_else(|| {
        KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} {field} must be an array"),
        )
    })?;
    if values.len() > MAX_CONTAINER_ITEMS {
        return Err(KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("{label} {field} exceeds the {MAX_CONTAINER_ITEMS}-item resource limit"),
        ));
    }
    let mut result = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for value in values {
        let value = value.as_str().ok_or_else(|| {
            KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} {field} contains a non-string value"),
            )
        })?;
        if !seen.insert(value) {
            return Err(KeylifeError::new(
                KeylifeErrorKind::CheckpointInvalid,
                format!("{label} {field} contains duplicate key ids"),
            ));
        }
        result.push(value.to_owned());
    }
    Ok(result)
}

fn positive_integer(
    value: Option<&Value>,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<String, KeylifeError> {
    let raw = value
        .and_then(Value::as_number)
        .map(ToString::to_string)
        .ok_or_else(|| KeylifeError::new(kind, format!("{label} must be a positive integer")))?;
    if raw.starts_with('-')
        || raw.contains(['.', 'e', 'E'])
        || !raw.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(KeylifeError::new(
            kind,
            format!("{label} must be a positive integer"),
        ));
    }
    let normalized = raw.trim_start_matches('0');
    if normalized.is_empty() {
        return Err(KeylifeError::new(
            kind,
            format!("{label} must be a positive integer"),
        ));
    }
    Ok(normalized.to_owned())
}

fn compare_positive_integers(left: &str, right: &str) -> Ordering {
    left.len().cmp(&right.len()).then_with(|| left.cmp(right))
}

fn canonical_equal(left: &Value, right: &Value) -> Result<bool, KeylifeError> {
    let left = canonical_json(left).map_err(|error| {
        KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("checkpoint value is not canonical: {error}"),
        )
    })?;
    let right = canonical_json(right).map_err(|error| {
        KeylifeError::new(
            KeylifeErrorKind::CheckpointInvalid,
            format!("checkpoint value is not canonical: {error}"),
        )
    })?;
    Ok(left == right)
}

fn canonical_optional_equal(
    left: Option<&Value>,
    right: Option<&Value>,
) -> Result<bool, KeylifeError> {
    canonical_equal(left.unwrap_or(&Value::Null), right.unwrap_or(&Value::Null))
}

fn not_monotonic(reason: impl Into<String>) -> KeylifeError {
    KeylifeError::new(KeylifeErrorKind::NotMonotonic, reason)
}

fn enforce_resource_bounds(
    value: &Value,
    kind: KeylifeErrorKind,
    label: &str,
) -> Result<(), KeylifeError> {
    fn walk(
        value: &Value,
        depth: usize,
        nodes: &mut usize,
        kind: KeylifeErrorKind,
        label: &str,
    ) -> Result<(), KeylifeError> {
        if depth > MAX_JSON_DEPTH {
            return Err(KeylifeError::new(
                kind,
                format!("{label} exceeds the JSON depth limit {MAX_JSON_DEPTH}"),
            ));
        }
        *nodes += 1;
        if *nodes > MAX_JSON_NODES {
            return Err(KeylifeError::new(
                kind,
                format!("{label} exceeds the JSON node limit {MAX_JSON_NODES}"),
            ));
        }
        match value {
            Value::String(value) if value.len() > MAX_STRING_BYTES => {
                return Err(KeylifeError::new(
                    kind,
                    format!("{label} contains a string over {MAX_STRING_BYTES} bytes"),
                ));
            }
            Value::Array(values) => {
                if values.len() > MAX_CONTAINER_ITEMS {
                    return Err(KeylifeError::new(
                        kind,
                        format!("{label} contains an array over {MAX_CONTAINER_ITEMS} items"),
                    ));
                }
                for value in values {
                    walk(value, depth + 1, nodes, kind, label)?;
                }
            }
            Value::Object(values) => {
                if values.len() > MAX_CONTAINER_ITEMS {
                    return Err(KeylifeError::new(
                        kind,
                        format!("{label} contains an object over {MAX_CONTAINER_ITEMS} members"),
                    ));
                }
                for (key, value) in values {
                    if key.len() > MAX_STRING_BYTES {
                        return Err(KeylifeError::new(
                            kind,
                            format!("{label} contains a key over {MAX_STRING_BYTES} bytes"),
                        ));
                    }
                    walk(value, depth + 1, nodes, kind, label)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    walk(value, 0, &mut 0, kind, label)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOLDEN: &str = include_str!("../tests/vectors/golden-key-in-service.json");
    const REVOKED: &str = include_str!("../tests/vectors/forgery-key-revoked.json");
    const UNROOTED: &str = include_str!("../tests/vectors/forgery-key-manifest-unrooted.json");
    const MISSING_CHECKPOINT: &str =
        include_str!("../tests/vectors/forgery-key-manifest-checkpoint-missing.json");
    const ROLE_COLLISION: &str =
        include_str!("../tests/vectors/forgery-key-manifest-checkpoint-role-collision.json");
    const ROLLBACK: &str = include_str!("../tests/vectors/forgery-key-manifest-rollback.json");
    const EQUIVOCATION: &str =
        include_str!("../tests/vectors/forgery-key-manifest-checkpoint-equivocation.json");
    const UNKNOWN_KEY: &str = include_str!("../tests/vectors/forgery-key-not-in-manifest.json");

    fn case(text: &str) -> Value {
        serde_json::from_str(text).expect("checked-in conformance vector is JSON")
    }

    fn pin<'a>(case: &'a Value, name: &str) -> Option<&'a Value> {
        case.get("pins")?.get(name)
    }

    fn pin_str<'a>(case: &'a Value, name: &str) -> Option<&'a str> {
        pin(case, name)?.as_str()
    }

    fn verify_fixture(case: &Value) -> Result<VerifiedCheckpoint, KeylifeError> {
        verify_manifest_checkpoint(
            pin(case, "key_manifest").expect("fixture manifest"),
            pin(case, "manifest_checkpoint"),
            pin_str(case, "manifest_root_hex"),
            pin_str(case, "checkpoint_public_key_hex"),
            pin(case, "trusted_manifest_checkpoint"),
        )
    }

    #[test]
    fn golden_fixture_verifies_and_places_key_inside_window() {
        let case = case(GOLDEN);
        let verified = verify_fixture(&case).unwrap();
        assert_eq!(
            verified.next_checkpoint,
            pin(&case, "manifest_checkpoint").unwrap().clone()
        );
        let key_id = case["envelope"]["key_id"].as_str().unwrap();
        let issued_at = case["envelope"]["created"].as_f64().unwrap();
        assert_eq!(
            key_status_at(&verified.manifest, Some(key_id), Some(issued_at)).0,
            KeyStatus::Valid
        );
        assert_eq!(
            key_status_at(&verified.manifest, Some(key_id), Some(2_001.0)).0,
            KeyStatus::OutsideValidity
        );
        assert_eq!(
            key_status_at(&verified.manifest, Some(key_id), None).0,
            KeyStatus::OutsideValidity
        );
    }

    #[test]
    fn revoked_fixture_is_never_valid_even_before_revocation() {
        let case = case(REVOKED);
        let verified = verify_fixture(&case).unwrap();
        let key_id = case["envelope"]["key_id"].as_str().unwrap();
        assert_eq!(
            key_status_at(&verified.manifest, Some(key_id), Some(1.0)).0,
            KeyStatus::Revoked
        );
        assert_eq!(
            key_status_at(&verified.manifest, Some(key_id), Some(1_000.0)).0,
            KeyStatus::Revoked
        );
    }

    #[test]
    fn fixture_without_root_fails_before_any_partial_check() {
        let case = case(UNROOTED);
        let error = verify_fixture(&case).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::RootRequired);
    }

    #[test]
    fn fixture_without_checkpoint_state_fails_required() {
        let case = case(MISSING_CHECKPOINT);
        let error = verify_fixture(&case).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::CheckpointRequired);
    }

    #[test]
    fn checkpoint_key_role_collision_fixture_fails_closed() {
        let case = case(ROLE_COLLISION);
        let error = verify_fixture(&case).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::CheckpointInvalid);
    }

    #[test]
    fn signed_sequence_rollback_fixture_is_not_monotonic() {
        let case = case(ROLLBACK);
        let error = verify_fixture(&case).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::NotMonotonic);
    }

    #[test]
    fn signed_same_sequence_equivocation_fixture_is_not_monotonic() {
        let case = case(EQUIVOCATION);
        let error = verify_fixture(&case).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::NotMonotonic);
    }

    #[test]
    fn absent_signing_key_is_unknown_after_valid_checkpoint() {
        let case = case(UNKNOWN_KEY);
        let verified = verify_fixture(&case).unwrap();
        let key_id = case["envelope"]["key_id"].as_str().unwrap();
        assert_eq!(
            key_status_at(&verified.manifest, Some(key_id), Some(1_000.0)).0,
            KeyStatus::Unknown
        );
    }

    #[test]
    fn succession_and_revocation_authority_are_independently_checked() {
        let mut golden = case(GOLDEN);
        let root = pin_str(&golden, "manifest_root_hex").unwrap().to_owned();
        golden["pins"]["key_manifest"]["keys"][1]["succession"]["predecessor_key_id"] =
            Value::String("ed25519:0000000000000000".to_owned());
        let error = verify_key_manifest(&golden["pins"]["key_manifest"], Some(&root)).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::ManifestInvalid);

        let mut revoked = case(REVOKED);
        let root = pin_str(&revoked, "manifest_root_hex").unwrap().to_owned();
        revoked["pins"]["key_manifest"]["keys"][0]["revocation"]["signer_key_id"] =
            Value::String("ed25519:0000000000000000".to_owned());
        let error = verify_key_manifest(&revoked["pins"]["key_manifest"], Some(&root)).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::ManifestInvalid);
    }

    #[test]
    fn nested_arrays_are_resource_bounded_before_signature_work() {
        let mut case = case(GOLDEN);
        let oversized = Value::Array(vec![Value::Null; MAX_CONTAINER_ITEMS + 1]);
        case["pins"]["key_manifest"]["extension"] = oversized;
        let root = pin_str(&case, "manifest_root_hex").unwrap().to_owned();
        let error = verify_key_manifest(&case["pins"]["key_manifest"], Some(&root)).unwrap_err();
        assert_eq!(error.kind, KeylifeErrorKind::ManifestInvalid);
        assert!(error.reason.contains("resource limit") || error.reason.contains("items"));
    }
}
