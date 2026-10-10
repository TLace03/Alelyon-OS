//! Network-free primitives for independently verifying Alelyon Certified Number
//! Envelopes (CNEs).
//!
//! This crate intentionally contains no producer, key-generation, filesystem,
//! process, or network functionality. It is the beginning of an independent
//! verifier for `alelyon.cne-spec/0.3.0`.

#![forbid(unsafe_code)]

pub mod canonical;
pub mod crypto;
pub mod data;
pub mod kernel;
pub mod keylife;
pub mod replay;
pub mod transparency;
pub mod verifier;

pub use canonical::{
    CanonicalError, canonical_json, canonical_json_without_signature, format_python_f64,
    parse_and_canonicalize,
};
pub use crypto::{
    CryptoError, blake2b_256, blake2b_256_hex, decode_lower_hex, encode_lower_hex,
    key_id_from_public_key_hex, sha256, sha256_hex, verify_ed25519_strict,
};
pub use kernel::{DitherStream, ReplayKernel};

/// This crate's version: what a verifier program built on it reports as the verifier's.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The specification revision implemented by this verifier crate.
pub const SPEC_VERSION: &str = "alelyon.cne-spec/0.3.0";

/// The only envelope type this verifier revision may interpret.
pub const ENVELOPE_TYPE: &str = "alelyon.cne/v0";

/// Where this crate's copy of the published conformance vectors is, for the tests
/// of crates that link it (a path on the machine that built it; not for programs).
#[doc(hidden)]
pub const TEST_VECTORS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors");
