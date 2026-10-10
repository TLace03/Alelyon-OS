//! Narrow cryptographic helpers used by CNE verification.
//!
//! This module verifies signatures but deliberately provides no key generation,
//! signing, key persistence, remote lookup, or network functionality.

use std::fmt;

use blake2::Blake2b;
use blake2::digest::{Digest, consts::U8, consts::U32};
use ed25519_dalek::{Signature, VerifyingKey};
use sha2::Sha256;

/// A malformed cryptographic input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CryptoError {
    /// Hex material did not have the exact byte length required by its field.
    InvalidHexLength { expected: usize, actual: usize },
    /// Hex material contained uppercase or a non-hexadecimal byte.
    InvalidHexCharacter { index: usize, byte: u8 },
    /// The 32 bytes did not encode an Ed25519 verification key.
    InvalidPublicKey,
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHexLength { expected, actual } => {
                write!(
                    f,
                    "expected {expected} lowercase hex characters, got {actual}"
                )
            }
            Self::InvalidHexCharacter { index, byte } => write!(
                f,
                "invalid lowercase hex byte 0x{byte:02x} at character {index}"
            ),
            Self::InvalidPublicKey => f.write_str("invalid Ed25519 public key"),
        }
    }
}

impl std::error::Error for CryptoError {}

/// Decode exactly `N` bytes of lowercase hexadecimal text.
///
/// Uppercase is rejected rather than normalized because CNE structures freeze
/// lowercase, fixed-width hex as part of their schema.
pub fn decode_lower_hex<const N: usize>(value: &str) -> Result<[u8; N], CryptoError> {
    let expected = N * 2;
    if value.len() != expected {
        return Err(CryptoError::InvalidHexLength {
            expected,
            actual: value.len(),
        });
    }
    let mut output = [0_u8; N];
    let bytes = value.as_bytes();
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let high = decode_nibble(pair[0], index * 2)?;
        let low = decode_nibble(pair[1], index * 2 + 1)?;
        output[index] = (high << 4) | low;
    }
    Ok(output)
}

fn decode_nibble(byte: u8, index: usize) -> Result<u8, CryptoError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        _ => Err(CryptoError::InvalidHexCharacter { index, byte }),
    }
}

/// Encode bytes as fixed lowercase hexadecimal text.
pub fn encode_lower_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

/// BLAKE2b-256, used for CNE content commitments and cert-log leaf links.
pub fn blake2b_256(input: &[u8]) -> [u8; 32] {
    let digest = Blake2b::<U32>::digest(input);
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}

/// Lowercase-hex BLAKE2b-256.
pub fn blake2b_256_hex(input: &[u8]) -> String {
    encode_lower_hex(&blake2b_256(input))
}

/// SHA-256, used by CNE program hashes and RFC-6962 Merkle nodes.
pub fn sha256(input: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(input);
    let mut output = [0_u8; 32];
    output.copy_from_slice(&digest);
    output
}

/// Lowercase-hex SHA-256.
pub fn sha256_hex(input: &[u8]) -> String {
    encode_lower_hex(&sha256(input))
}

/// Derive the frozen CNE key identifier from a raw Ed25519 public key.
///
/// This is BLAKE2b configured for an 8-byte output, not a truncation of a
/// BLAKE2b-256 digest; those are different constructions.
pub fn key_id_from_public_key_hex(public_key_hex: &str) -> Result<String, CryptoError> {
    let public_key = decode_lower_hex::<32>(public_key_hex)?;
    let digest = Blake2b::<U8>::digest(public_key);
    Ok(format!("ed25519:{}", encode_lower_hex(&digest)))
}

/// Strictly verify an Ed25519 signature under a pinned raw public key.
///
/// Malformed key/hex material is an error. Well-formed material whose signature
/// is not valid for `message` returns `Ok(false)`. Dalek's strict verifier rejects
/// non-canonical scalar encodings and weak-key edge cases rather than accepting a
/// merely equation-valid signature.
pub fn verify_ed25519_strict(
    public_key_hex: &str,
    message: &[u8],
    signature_hex: &str,
) -> Result<bool, CryptoError> {
    let public_key_bytes = decode_lower_hex::<32>(public_key_hex)?;
    let signature_bytes = decode_lower_hex::<64>(signature_hex)?;
    let public_key =
        VerifyingKey::from_bytes(&public_key_bytes).map_err(|_| CryptoError::InvalidPublicKey)?;
    let signature = Signature::from_bytes(&signature_bytes);
    Ok(public_key.verify_strict(message, &signature).is_ok())
}
#[cfg(test)]
mod tests {
    use super::*;

    const RFC8032_PUBLIC_KEY: &str =
        "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a";
    const RFC8032_EMPTY_SIGNATURE: &str = concat!(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155",
        "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    );

    #[test]
    fn accepts_the_rfc8032_strict_verification_vector() {
        assert_eq!(
            verify_ed25519_strict(RFC8032_PUBLIC_KEY, b"", RFC8032_EMPTY_SIGNATURE),
            Ok(true)
        );
        assert_eq!(
            verify_ed25519_strict(RFC8032_PUBLIC_KEY, b"changed", RFC8032_EMPTY_SIGNATURE),
            Ok(false)
        );
    }

    #[test]
    fn enforces_lowercase_and_exact_hex_lengths() {
        assert_eq!(decode_lower_hex::<2>("00af").unwrap(), [0x00, 0xaf]);
        assert!(matches!(
            decode_lower_hex::<2>("00AF"),
            Err(CryptoError::InvalidHexCharacter { .. })
        ));
        assert_eq!(
            decode_lower_hex::<2>("00a"),
            Err(CryptoError::InvalidHexLength {
                expected: 4,
                actual: 3
            })
        );
        assert!(matches!(
            verify_ed25519_strict(
                &RFC8032_PUBLIC_KEY.to_uppercase(),
                b"",
                RFC8032_EMPTY_SIGNATURE
            ),
            Err(CryptoError::InvalidHexCharacter { .. })
        ));
    }

    #[test]
    fn matches_frozen_hash_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            blake2b_256_hex(b""),
            "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
        );
    }

    #[test]
    fn key_id_uses_blake2b_with_an_eight_byte_output() {
        assert_eq!(
            key_id_from_public_key_hex(RFC8032_PUBLIC_KEY).unwrap(),
            "ed25519:c46607cab5cd8e90"
        );
    }
}
