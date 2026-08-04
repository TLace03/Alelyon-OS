"""Registration Certificates for the exact Lattice slice.

A certificate is the artifact a party who was not present at registration is
given: it names the two coordinate spaces, the transform chain that maps one to
the other, the conditions under which that chain was produced, and a signature
over all of it. `verify_registration_certificate` re-derives every commitment and
replays the chain, so a reader checks the record rather than trusting the issuer's
database.

Scope, stated rather than implied
---------------------------------
The governing specification (MODEL-MORPHOMETRY.md §25.1) declares 34 certificate
fields. This slice populates 13 of them. The other 21 refer to mechanisms that do
not exist here -- there is no artifact manifest, no template registry, no search
plan, no metric registry, no objective, no payload remapping, no uncertainty
propagation and no execution trace -- and each is carried in the certificate as a
named absence with its reason and its kind: NOT_APPLICABLE where the mechanism
cannot apply to exact registration, UNMEASURED where it could apply and nothing
measured it. The absences are part of the signed bytes, so a certificate cannot
understate what it leaves out, and `DECLARED_ABSENCES` is asserted complete
against the spec's field list by
`tests/vector/test_lattice_certificate.py::test_every_spec_field_is_populated_or_named_absent`.

What a CERTIFICATE_VERIFIED report establishes
----------------------------------------------
* The certificate bytes are canonical, hash to the reference they are addressed
  by, and carry a valid signature under the public key **the caller pinned**.
* The committed transform chain has not been revised, is the canonical encoding
  of the chain it decodes to, satisfies every contract invariant, and reproduces
  each claimed source coordinate (delegated to `verify.verify_transform_chain`,
  which is the one replay implementation).
* The chain the caller holds is the one this certificate is about: the declared
  space references, loss class and invertibility are cross-checked against what
  the replay independently re-derived.

What it does not establish
--------------------------
* Not that the registration is *correct* for any dataset. A semantically wrong
  but internally consistent chain verifies cleanly, exactly as it replays cleanly.
* Not optimality. Nothing here searches an objective, so no field claims a bound.
* Nothing about payload values. No artifact is read and no value is remapped.
* Not an independent implementation. Verification shares the contract, transform
  and canonical modules with the issuer, so a defect in those is reproduced
  rather than caught.
* Not that the signer is anyone in particular. A signature binds bytes to a key;
  who holds that key is a fact the pinning party establishes out of band. The
  certificate carries only the key's *fingerprint*, never public key material, so
  there is nothing inside it a verifier could mistake for a trust input
  (specification §25.2).

Pillar note: this module imports `KeyStore` from `alelyon.runtime.atlas`, which is
the repository's only ed25519 implementation and is a boundary the pure core does
not cross -- `contracts`, `transforms`, `canonical` and `verify` remain free of
any key, signature or filesystem dependency.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import hmac
from types import MappingProxyType

from alelyon.runtime.atlas.data.attest import KeyStore
from alelyon.runtime.vector.lattice.canonical import (
    CanonicalEncodingError,
    ContractViolationError,
    _Reader,
    _content_ref,
    _domain,
    _sequence,
    _string,
    coordinate_space_ref,
    transform_chain_ref,
)
from alelyon.runtime.vector.lattice.contracts import (
    CoordinateSpace,
    _bounded_tuple,
    _text,
)
from alelyon.runtime.vector.lattice.registration import (
    COMPOSED_STAGE_ORDER,
    EXACT_CHAIN_SHAPES,
    CompatibilityCode,
    CompatibilityReport,
    chain_shape_refusal,
    composed_intermediate_ids,
)
from alelyon.runtime.vector.lattice.transforms import (
    Invertibility,
    LossClass,
    TransformChain,
)
from alelyon.runtime.vector.lattice.verify import (
    ReplayReport,
    verify_transform_chain,
)


CERTIFICATE_SCHEMA = "alelyon.lattice.registration-certificate/0.1"
CERTIFICATE_DOMAIN = "alelyon.lattice.canonical.registration-certificate/0.1"
ABSENCE_DOMAIN = "alelyon.lattice.canonical.field-absence/0.1"

#: Declared build identifiers for the two halves the specification separates
#: (§25.1 `candidate_generator_build`, `reference_verifier_build`). These are
#: *declared* names, not a measured attestation of the running code: nothing here
#: hashes the interpreter, the installed package or the machine.
CANDIDATE_GENERATOR_BUILD = "alelyon.lattice.registration/0.1"
REFERENCE_VERIFIER_BUILD = "alelyon.lattice.verify/0.1"

MAX_CERTIFICATE_WARNINGS = 64
MAX_CERTIFICATE_SIGNATURES = 8
MAX_CERTIFICATE_BYTES = 1 << 20

_U64_MAX = 0xFFFFFFFFFFFFFFFF
_SIGNATURE_HEX_LENGTH = 128
_KEY_ID_PREFIX = "ed25519:"


class DeterminismProfile(str, Enum):
    """Specification §20. Certificates are issued under STRICT_REFERENCE only."""

    STRICT_REFERENCE = "STRICT_REFERENCE"


class FieldStatus(str, Enum):
    """Why a specification field carries no value here.

    The two are not interchangeable, and collapsing them is the defect the CNE
    claim discipline names: a blank slot beside a filled one reads as
    "checked, fine".
    """

    #: The mechanism cannot apply to exact registration. Nothing will ever fill it
    #: at this schema version, and that is a design fact.
    NOT_APPLICABLE = "NOT_APPLICABLE"
    #: The field could carry a value and nothing measured one. This is a gap.
    UNMEASURED = "UNMEASURED"


class CertificateCode(str, Enum):
    CERTIFICATE_VERIFIED = "CERTIFICATE_VERIFIED"
    MALFORMED_CERTIFICATE = "MALFORMED_CERTIFICATE"
    NONCANONICAL_CERTIFICATE = "NONCANONICAL_CERTIFICATE"
    CERTIFICATE_COMMITMENT_MISMATCH = "CERTIFICATE_COMMITMENT_MISMATCH"
    KEY_NOT_PINNED = "KEY_NOT_PINNED"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    CHAIN_REPLAY_FAILED = "CHAIN_REPLAY_FAILED"
    DECLARATION_MISMATCH = "DECLARATION_MISMATCH"


class CertificateError(ValueError):
    """A certificate could not be issued from the inputs supplied."""


@dataclass(frozen=True, slots=True)
class FieldAbsence:
    """One specification field this schema version does not populate."""

    field_name: str
    status: FieldStatus
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_name",
            _text(self.field_name, "field_name", identifier=True, maximum=128),
        )
        if not isinstance(self.status, FieldStatus):
            raise TypeError("status must be a FieldStatus")
        object.__setattr__(
            self, "reason", _text(self.reason, "reason", identifier=False, maximum=512)
        )


#: Every field the specification's §25.1 schema declares, in its own order. The
#: certificate is checked against this list rather than against a summary of it,
#: so a field the spec declares and this slice forgot cannot pass unnoticed.
SPEC_CERTIFICATE_FIELDS = (
    "certificate_id",
    "certificate_schema_version",
    "issued_at",
    "source_artifact_ref",
    "source_commitment",
    "template_ref",
    "template_commitment",
    "request_commitment",
    "search_plan_commitment",
    "metric_profile_commitment",
    "transform_chain_ref",
    "transform_chain_commitment",
    "output_artifact_ref",
    "output_commitment",
    "morphometry_refs",
    "morphometry_commitments",
    "proof_status",
    "objective_value",
    "objective_lower_bound",
    "objective_upper_bound",
    "epsilon",
    "hard_constraint_results",
    "residual_bounds",
    "conservation_bounds",
    "uncertainty_bounds",
    "inverse_consistency_bounds",
    "determinism_profile",
    "candidate_generator_build",
    "reference_verifier_build",
    "hardware_profile",
    "execution_trace_commitment",
    "warnings",
    "refusal_or_failure_reason",
    "signatures",
)

#: Which certificate attribute carries each specification field this slice does
#: populate. Two spec fields may map to one attribute: in a content-addressed
#: system a reference *is* the commitment, so `transform_chain_ref` and
#: `transform_chain_commitment` are one value rather than two names for a hash
#: computed twice.
POPULATED_FIELDS: Mapping[str, str] = MappingProxyType({
    "certificate_id": "certificate_ref (derived, not stored: a field cannot be "
                      "inside the bytes it hashes)",
    "certificate_schema_version": "schema_version",
    "issued_at": "issued_at_unix_seconds",
    "source_commitment": "source_space_ref",
    "template_commitment": "template_space_ref",
    "transform_chain_ref": "transform_chain_ref",
    "transform_chain_commitment": "transform_chain_ref",
    "proof_status": "compatibility_code",
    "determinism_profile": "determinism_profile",
    "candidate_generator_build": "candidate_generator_build",
    "reference_verifier_build": "reference_verifier_build",
    "warnings": "warnings",
    "signatures": "SignedRegistrationCertificate.signatures",
})

#: Why each remaining §25.1 field is empty. These are committed in the signed
#: bytes: a later build that starts populating one of them produces a different
#: certificate encoding rather than quietly filling a slot readers had been told
#: was empty.
DECLARED_ABSENCES = (
    FieldAbsence(
        "source_artifact_ref",
        FieldStatus.NOT_APPLICABLE,
        "no artifact manifest exists in this slice; the registered source is a "
        "coordinate space, committed as source_commitment",
    ),
    FieldAbsence(
        "template_ref",
        FieldStatus.NOT_APPLICABLE,
        "no template registry exists; the template is a coordinate space, "
        "committed as template_commitment",
    ),
    FieldAbsence(
        "request_commitment",
        FieldStatus.NOT_APPLICABLE,
        "no registration request object exists; compatibility analysis takes the "
        "two spaces directly",
    ),
    FieldAbsence(
        "search_plan_commitment",
        FieldStatus.NOT_APPLICABLE,
        "exact correspondence is decided by bounded exhaustive matching, so there "
        "is no search plan to commit to",
    ),
    FieldAbsence(
        "metric_profile_commitment",
        FieldStatus.NOT_APPLICABLE,
        "no metric registry exists and exact registration evaluates no metric",
    ),
    FieldAbsence(
        "output_artifact_ref",
        FieldStatus.NOT_APPLICABLE,
        "no payload is remapped, so no output artifact is produced",
    ),
    FieldAbsence(
        "output_commitment",
        FieldStatus.NOT_APPLICABLE,
        "no payload is remapped, so there is no output to commit to",
    ),
    FieldAbsence(
        "morphometry_refs",
        FieldStatus.NOT_APPLICABLE,
        "the exact registration path computes no morphometry; measurements from "
        "the Model Morphometry engine are not part of this certificate",
    ),
    FieldAbsence(
        "morphometry_commitments",
        FieldStatus.NOT_APPLICABLE,
        "no morphometry output exists on this path to commit to",
    ),
    FieldAbsence(
        "objective_value",
        FieldStatus.NOT_APPLICABLE,
        "no objective is optimized; correspondence is decided combinatorially",
    ),
    FieldAbsence(
        "objective_lower_bound",
        FieldStatus.NOT_APPLICABLE,
        "no objective is optimized, so no bound on one exists",
    ),
    FieldAbsence(
        "objective_upper_bound",
        FieldStatus.NOT_APPLICABLE,
        "no objective is optimized, so no bound on one exists",
    ),
    FieldAbsence(
        "epsilon",
        FieldStatus.NOT_APPLICABLE,
        "no tolerance is admitted: an exact correspondence holds or is refused",
    ),
    FieldAbsence(
        "hard_constraint_results",
        FieldStatus.NOT_APPLICABLE,
        "no constraint registry exists; admissibility is enforced at construction "
        "and a violation is a refusal rather than a recorded result",
    ),
    FieldAbsence(
        "residual_bounds",
        FieldStatus.NOT_APPLICABLE,
        "no payload is remapped, so no residual field exists to bound",
    ),
    FieldAbsence(
        "conservation_bounds",
        FieldStatus.NOT_APPLICABLE,
        "no payload is remapped, so no extensive quantity is transported",
    ),
    FieldAbsence(
        "uncertainty_bounds",
        FieldStatus.NOT_APPLICABLE,
        "no uncertainty propagation exists in this slice",
    ),
    FieldAbsence(
        "inverse_consistency_bounds",
        FieldStatus.UNMEASURED,
        "the chain declares EXACT invertibility and inversion is implemented, but "
        "no inverse-consistency measurement is performed or recorded",
    ),
    FieldAbsence(
        "hardware_profile",
        FieldStatus.NOT_APPLICABLE,
        "the exact path is integer and string comparison with no floating-point "
        "kernel, so no hardware attribute can change its result",
    ),
    FieldAbsence(
        "execution_trace_commitment",
        FieldStatus.UNMEASURED,
        "no execution trace is recorded, so none is committed to",
    ),
    FieldAbsence(
        "refusal_or_failure_reason",
        FieldStatus.NOT_APPLICABLE,
        "this schema certifies an exact correspondence only; a compatibility "
        "refusal is returned as a CompatibilityReport and is never certified",
    ),
)

#: The correspondence classes a certificate may be issued for. A refusal is not a
#: certificate, so the remaining `CompatibilityCode` members cannot appear.
#:
#: `CompatibilityReport` already enforces the class-to-chain pairing when the
#: report is built, but a verifier holds the certificate and the chain rather
#: than the report, so the pairing is re-derived from the replayed chain rather
#: than assumed. Without it a certificate could declare EXACT_IDENTITY over a
#: chain that reindexes labels, and every other check would pass. Both sides now
#: ask `registration.chain_shape_refusal`, which is the single statement of that
#: rule; this set only decides *which* classes are certifiable at all.
CERTIFIABLE_CODES: frozenset[CompatibilityCode] = frozenset(
    set(EXACT_CHAIN_SHAPES) | {CompatibilityCode.EXACT_COMPOSED_CHAIN}
)

#: Every transform type a certifiable chain may contain, derived rather than
#: listed. `test_every_certifiable_class_weaker_than_lossless_has_a_policy_test`
#: reads this to decide which loss classes owe a policy test.
CERTIFIABLE_TRANSFORM_TYPES: frozenset[str] = frozenset(
    set(COMPOSED_STAGE_ORDER).union(
        *(set(shape) for shape in EXACT_CHAIN_SHAPES.values())
    )
)


def _u64(value: int) -> bytes:
    if type(value) is not int or not 0 <= value <= _U64_MAX:
        raise CanonicalEncodingError("value is outside the unsigned 64-bit domain")
    return value.to_bytes(8, "big")


def _read_u64(reader: _Reader) -> int:
    return int.from_bytes(reader.take(8), "big")


def _content_reference(value: object, field_name: str) -> str:
    text = _text(value, field_name, identifier=True, maximum=128)
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field_name} must be a sha256 content reference")
    if any(character not in "0123456789abcdef" for character in text[7:]):
        raise ValueError(f"{field_name} must be a lowercase hex content reference")
    return text


@dataclass(frozen=True, slots=True)
class RegistrationCertificate:
    """The unsigned record. Its canonical bytes are what a signature covers."""

    issued_at_unix_seconds: int
    compatibility_code: CompatibilityCode
    source_space_ref: str
    template_space_ref: str
    transform_chain_ref: str
    loss_class: LossClass
    invertibility: Invertibility
    schema_version: str = CERTIFICATE_SCHEMA
    determinism_profile: DeterminismProfile = DeterminismProfile.STRICT_REFERENCE
    candidate_generator_build: str = CANDIDATE_GENERATOR_BUILD
    reference_verifier_build: str = REFERENCE_VERIFIER_BUILD
    warnings: tuple[str, ...] = ()
    absences: tuple[FieldAbsence, ...] = DECLARED_ABSENCES

    def __post_init__(self) -> None:
        if type(self.issued_at_unix_seconds) is not int or isinstance(
            self.issued_at_unix_seconds, bool
        ):
            raise TypeError("issued_at_unix_seconds must be an int")
        if not 0 <= self.issued_at_unix_seconds <= _U64_MAX:
            raise ValueError("issued_at_unix_seconds is outside the u64 domain")
        if self.schema_version != CERTIFICATE_SCHEMA:
            raise ValueError(
                f"schema_version must be {CERTIFICATE_SCHEMA!r}; a different "
                "version is a different encoding and needs its own reader"
            )
        if not isinstance(self.compatibility_code, CompatibilityCode):
            raise TypeError("compatibility_code must be a CompatibilityCode")
        if self.compatibility_code not in CERTIFIABLE_CODES:
            raise ValueError(
                f"{self.compatibility_code.value} is a refusal, not an exact "
                "correspondence; refusals are not certified"
            )
        if not isinstance(self.determinism_profile, DeterminismProfile):
            raise TypeError("determinism_profile must be a DeterminismProfile")
        if not isinstance(self.loss_class, LossClass):
            raise TypeError("loss_class must be a LossClass")
        if not isinstance(self.invertibility, Invertibility):
            raise TypeError("invertibility must be an Invertibility")
        for name in ("source_space_ref", "template_space_ref", "transform_chain_ref"):
            object.__setattr__(
                self, name, _content_reference(getattr(self, name), name)
            )
        for name in ("candidate_generator_build", "reference_verifier_build"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, identifier=True, maximum=256),
            )
        warnings = _bounded_tuple(
            self.warnings, "warnings", MAX_CERTIFICATE_WARNINGS
        )
        object.__setattr__(
            self,
            "warnings",
            tuple(
                _text(item, "warning", identifier=False, maximum=512)
                for item in warnings
            ),
        )
        # The absences are the honesty of the record, so they are fixed by the
        # schema rather than chosen per issuance: a producer cannot issue a
        # certificate that admits to leaving out less than this build leaves out.
        absences = tuple(self.absences)
        if absences != DECLARED_ABSENCES:
            raise ValueError(
                "absences must be exactly DECLARED_ABSENCES for this schema "
                "version; understating what the certificate omits is the one "
                "edit this record refuses"
            )
        object.__setattr__(self, "absences", absences)


def _absence_bytes(absence: FieldAbsence) -> bytes:
    return (
        _domain(ABSENCE_DOMAIN)
        + _string(absence.field_name)
        + _string(absence.status.value)
        + _string(absence.reason)
    )


def certificate_bytes(certificate: RegistrationCertificate) -> bytes:
    """Encode the unsigned certificate. Pure: no clock, no key, no environment."""

    if type(certificate) is not RegistrationCertificate:
        raise CanonicalEncodingError("certificate must be a RegistrationCertificate")
    payload = (
        _domain(CERTIFICATE_DOMAIN)
        + _string(certificate.schema_version)
        + _u64(certificate.issued_at_unix_seconds)
        + _string(certificate.determinism_profile.value)
        + _string(certificate.compatibility_code.value)
        + _string(certificate.source_space_ref)
        + _string(certificate.template_space_ref)
        + _string(certificate.transform_chain_ref)
        + _string(certificate.loss_class.value)
        + _string(certificate.invertibility.value)
        + _string(certificate.candidate_generator_build)
        + _string(certificate.reference_verifier_build)
        + _sequence(_string(warning) for warning in certificate.warnings)
        + _sequence(_absence_bytes(absence) for absence in certificate.absences)
    )
    if len(payload) > MAX_CERTIFICATE_BYTES:
        raise CanonicalEncodingError(
            f"certificate encodes to {len(payload)} bytes, above the "
            f"{MAX_CERTIFICATE_BYTES}-byte limit"
        )
    return payload


def certificate_ref(certificate: RegistrationCertificate) -> str:
    """The certificate's content reference. This is §25.1's `certificate_id`.

    Derived rather than stored: a field naming the digest of the bytes it sits
    inside cannot be computed, so the identifier is the digest itself.
    """

    return _content_ref(certificate_bytes(certificate))


def _read_enum(reader: _Reader, enum_type, field_name: str):
    raw = reader.string()
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise CanonicalEncodingError(
            f"{field_name} {raw!r} is not a known {enum_type.__name__}"
        ) from exc


def read_certificate(payload: bytes) -> RegistrationCertificate:
    """Recover a certificate from canonical bytes, refusing anything ambiguous."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise CanonicalEncodingError("certificate input must be bytes")
    if len(payload) > MAX_CERTIFICATE_BYTES:
        raise CanonicalEncodingError(
            f"certificate input is {len(payload)} bytes, above the "
            f"{MAX_CERTIFICATE_BYTES}-byte limit"
        )
    reader = _Reader(bytes(payload))
    reader.expect_domain(CERTIFICATE_DOMAIN)
    schema_version = reader.string()
    issued_at = _read_u64(reader)
    determinism_profile = _read_enum(reader, DeterminismProfile, "determinism_profile")
    compatibility_code = _read_enum(reader, CompatibilityCode, "compatibility_code")
    source_space_ref = reader.string()
    template_space_ref = reader.string()
    chain_ref = reader.string()
    loss_class = _read_enum(reader, LossClass, "loss_class")
    invertibility = _read_enum(reader, Invertibility, "invertibility")
    candidate_generator_build = reader.string()
    reference_verifier_build = reader.string()
    warnings = tuple(
        reader.string()
        for _ in range(reader.count("warnings", MAX_CERTIFICATE_WARNINGS))
    )
    absences = tuple(
        _read_absence(reader)
        for _ in range(reader.count("absences", len(SPEC_CERTIFICATE_FIELDS)))
    )
    reader.expect_end()
    try:
        return RegistrationCertificate(
            issued_at_unix_seconds=issued_at,
            compatibility_code=compatibility_code,
            source_space_ref=source_space_ref,
            template_space_ref=template_space_ref,
            transform_chain_ref=chain_ref,
            loss_class=loss_class,
            invertibility=invertibility,
            schema_version=schema_version,
            determinism_profile=determinism_profile,
            candidate_generator_build=candidate_generator_build,
            reference_verifier_build=reference_verifier_build,
            warnings=warnings,
            absences=absences,
        )
    except (TypeError, ValueError) as exc:
        raise ContractViolationError(
            f"the recovered certificate violates its own contract: {exc}"
        ) from exc


def _read_absence(reader: _Reader) -> FieldAbsence:
    reader.expect_domain(ABSENCE_DOMAIN)
    field_name = reader.string()
    status = _read_enum(reader, FieldStatus, "absence status")
    reason = reader.string()
    try:
        return FieldAbsence(field_name, status, reason)
    except (TypeError, ValueError) as exc:
        raise ContractViolationError(f"absence record is invalid: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CertificateSignature:
    """A key fingerprint and a signature over the unsigned certificate bytes.

    Deliberately no public key material. §25.2: a key carried inside the
    certificate is not an authenticated trust input, so there is nothing here a
    verifier could mistake for one. The fingerprint says *which* key to go and
    obtain; obtaining it is the verifier's job.
    """

    key_id: str
    signature_hex: str

    def __post_init__(self) -> None:
        key_id = _text(self.key_id, "key_id", identifier=True, maximum=128)
        if not key_id.startswith(_KEY_ID_PREFIX):
            raise ValueError(f"key_id must begin with {_KEY_ID_PREFIX!r}")
        object.__setattr__(self, "key_id", key_id)
        signature = _text(
            self.signature_hex, "signature_hex", identifier=True, maximum=256
        )
        if len(signature) != _SIGNATURE_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in signature
        ):
            raise ValueError(
                "signature_hex must be a lowercase 64-byte ed25519 signature"
            )
        object.__setattr__(self, "signature_hex", signature)


@dataclass(frozen=True, slots=True)
class SignedRegistrationCertificate:
    """A certificate together with the signatures over its canonical bytes."""

    certificate: RegistrationCertificate
    signatures: tuple[CertificateSignature, ...]

    def __post_init__(self) -> None:
        if type(self.certificate) is not RegistrationCertificate:
            raise TypeError("certificate must be a RegistrationCertificate")
        signatures = _bounded_tuple(
            self.signatures, "signatures", MAX_CERTIFICATE_SIGNATURES
        )
        if not signatures:
            raise ValueError("a signed certificate must carry a signature")
        for signature in signatures:
            if type(signature) is not CertificateSignature:
                raise TypeError("signatures must be CertificateSignature values")
        key_ids = [signature.key_id for signature in signatures]
        if len(set(key_ids)) != len(key_ids):
            raise ValueError("one key may sign a certificate at most once")
        object.__setattr__(self, "signatures", tuple(signatures))

    @property
    def certificate_ref(self) -> str:
        return certificate_ref(self.certificate)


@dataclass(frozen=True, slots=True)
class CertificateReport:
    """The outcome of checking one certificate."""

    code: CertificateCode
    explanation: str
    certificate_ref: str | None = None
    key_id: str | None = None
    failing_constraint: str | None = None
    evidence: tuple[str, ...] = ()
    replay: ReplayReport | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, CertificateCode):
            raise TypeError("code must be a CertificateCode")
        if self.code is CertificateCode.CERTIFICATE_VERIFIED:
            if self.failing_constraint is not None:
                raise ValueError("a verified certificate cannot name a failure")
            if self.certificate_ref is None or self.key_id is None:
                raise ValueError(
                    "a verified certificate must report what it verified and "
                    "under which key"
                )
            if self.replay is None or not self.replay.verified:
                raise ValueError(
                    "a verified certificate requires a matched chain replay"
                )
        elif self.failing_constraint is None:
            raise ValueError("a certificate refusal must name failing_constraint")

    @property
    def verified(self) -> bool:
        return self.code is CertificateCode.CERTIFICATE_VERIFIED


def _refusal(
    code: CertificateCode,
    explanation: str,
    *,
    failing_constraint: str,
    evidence: tuple[str, ...] = (),
    **facts: object,
) -> CertificateReport:
    return CertificateReport(
        code=code,
        explanation=explanation,
        failing_constraint=failing_constraint,
        evidence=evidence,
        **facts,  # type: ignore[arg-type]
    )


def issue_registration_certificate(
    report: CompatibilityReport,
    *,
    source_space: CoordinateSpace,
    template_space: CoordinateSpace,
    signer: KeyStore,
    issued_at_unix_seconds: int,
    warnings: Iterable[str] = (),
) -> SignedRegistrationCertificate:
    """Issue and sign a certificate for an exact compatibility result.

    ``source_space`` and ``template_space`` are the spaces the report was
    produced from. They are supplied rather than taken from the chain so that the
    two can be checked against each other: a caller who hands over a report for a
    different pair is refused here instead of certified.

    The clock is a parameter. Reading one inside would make the record depend on
    when it was built rather than on what it says, and the encoding is a pure
    function of the record.
    """

    if type(report) is not CompatibilityReport:
        raise TypeError("report must be a CompatibilityReport")
    if not report.compatible:
        raise CertificateError(
            f"{report.code.value} is a refusal, not an exact correspondence: "
            f"{report.explanation}"
        )
    if report.code not in CERTIFIABLE_CODES:
        # An exact result this schema cannot certify. Saying "not an exact
        # correspondence" here would be false, and saying nothing would leave a
        # caller to conclude the registration was defective.
        #
        # Currently unreachable: every exact code is certifiable, which
        # `test_every_exact_correspondence_is_certifiable` asserts rather than
        # assumes. It fires when a new exact class arrives, so whoever adds one
        # decides deliberately whether a certificate may name it instead of
        # inheriting a yes.
        raise CertificateError(
            f"{report.code.value} is an exact correspondence that certificate "
            f"schema {CERTIFICATE_SCHEMA} does not declare"
        )
    chain = report.transform
    if type(chain) is not TransformChain:      # unreachable via CompatibilityReport
        raise CertificateError("an exact report must carry a transform chain")
    if (
        type(source_space) is not CoordinateSpace
        or type(template_space) is not CoordinateSpace
    ):
        raise TypeError("source_space and template_space must be CoordinateSpace")

    declared_source = coordinate_space_ref(source_space)
    declared_template = coordinate_space_ref(template_space)
    chain_source = coordinate_space_ref(chain.source_space)
    chain_target = coordinate_space_ref(chain.target_space)
    if declared_source != chain_source or declared_template != chain_target:
        raise CertificateError(
            "the supplied spaces are not the pair this chain connects; the "
            "chain maps target->source, so template_space must be its target"
        )

    certificate = RegistrationCertificate(
        issued_at_unix_seconds=issued_at_unix_seconds,
        compatibility_code=report.code,
        source_space_ref=declared_source,
        template_space_ref=declared_template,
        transform_chain_ref=transform_chain_ref(chain),
        loss_class=chain.loss_class,
        invertibility=chain.invertibility,
        warnings=tuple(warnings),
    )
    payload = certificate_bytes(certificate)
    signature = CertificateSignature(
        key_id=signer.key_id(), signature_hex=signer.sign(payload).hex()
    )
    return SignedRegistrationCertificate(certificate, (signature,))


def verify_registration_certificate(
    signed: SignedRegistrationCertificate,
    *,
    pinned_public_key_hex: str,
    spaces: Mapping[str, CoordinateSpace],
    chain_bytes: bytes,
    expected_certificate_ref: str | None = None,
    replay_cases: Iterable[tuple[tuple[object, ...], tuple[object, ...]]] = (),
    maximum_loss_class: LossClass | None = None,
) -> CertificateReport:
    """Check a certificate against a key the caller pinned out of band.

    ``pinned_public_key_hex`` is required and is never read from the certificate:
    the certificate carries no public key at all, so the trust input can only come
    from the caller. Read the module docstring for what a verified report does and
    does not establish.
    """

    if type(signed) is not SignedRegistrationCertificate:
        raise TypeError("signed must be a SignedRegistrationCertificate")
    if not isinstance(pinned_public_key_hex, str):
        raise TypeError("pinned_public_key_hex must be a string")
    try:
        pinned_key_id = KeyStore.key_id_of(pinned_public_key_hex)
    except ValueError as exc:
        raise ValueError("pinned_public_key_hex must be hex-encoded key bytes") from exc

    try:
        payload = certificate_bytes(signed.certificate)
    except CanonicalEncodingError as exc:
        return _refusal(
            CertificateCode.MALFORMED_CERTIFICATE,
            f"the certificate could not be encoded: {exc}",
            failing_constraint="canonical_encoding",
        )
    actual_ref = _content_ref(payload)

    if expected_certificate_ref is not None:
        if not isinstance(expected_certificate_ref, str):
            raise TypeError("expected_certificate_ref must be a string")
        if not hmac.compare_digest(actual_ref, expected_certificate_ref):
            return _refusal(
                CertificateCode.CERTIFICATE_COMMITMENT_MISMATCH,
                "the certificate does not hash to the pinned reference",
                failing_constraint="certificate_commitment",
                evidence=(
                    f"expected:{expected_certificate_ref}",
                    f"actual:{actual_ref}",
                ),
            )

    # Round-tripping proves exactly one byte string can present this record, so a
    # second encoding of the same certificate cannot carry a second signature.
    try:
        if certificate_bytes(read_certificate(payload)) != payload:
            raise CanonicalEncodingError("re-encoding produced different bytes")
    except CanonicalEncodingError as exc:
        return _refusal(
            CertificateCode.NONCANONICAL_CERTIFICATE,
            f"the certificate bytes are not canonical: {exc}",
            failing_constraint="canonical_encoding",
            certificate_ref=actual_ref,
        )

    matching = [
        signature
        for signature in signed.signatures
        if hmac.compare_digest(signature.key_id, pinned_key_id)
    ]
    if not matching:
        return _refusal(
            CertificateCode.KEY_NOT_PINNED,
            "no signature on this certificate was made by the pinned key",
            failing_constraint="pinned_key",
            certificate_ref=actual_ref,
            evidence=(f"pinned:{pinned_key_id}",)
            + tuple(f"present:{s.key_id}" for s in signed.signatures),
        )
    if not any(
        KeyStore.verify(pinned_public_key_hex, payload, signature.signature_hex)
        for signature in matching
    ):
        return _refusal(
            CertificateCode.SIGNATURE_INVALID,
            "the pinned key did not sign these certificate bytes",
            failing_constraint="signature",
            certificate_ref=actual_ref,
            key_id=pinned_key_id,
        )

    replay = verify_transform_chain(
        chain_bytes,
        expected_chain_ref=signed.certificate.transform_chain_ref,
        spaces=spaces,
        replay_cases=replay_cases,
        maximum_loss_class=maximum_loss_class,
    )
    if not replay.verified:
        return _refusal(
            CertificateCode.CHAIN_REPLAY_FAILED,
            f"the committed chain did not replay: {replay.explanation}",
            failing_constraint=replay.failing_constraint or "chain_replay",
            certificate_ref=actual_ref,
            key_id=pinned_key_id,
            replay=replay,
        )

    # The chain replayed, but is it the chain this certificate is about? Every
    # fact the certificate declares about it is re-derived by the replay, so the
    # two are compared rather than one being read from the other.
    declared = (
        signed.certificate.source_space_ref,
        signed.certificate.template_space_ref,
        signed.certificate.loss_class,
        signed.certificate.invertibility,
    )
    replayed = (
        replay.source_space_ref,
        replay.target_space_ref,
        replay.loss_class,
        replay.invertibility,
    )
    # The declared class is checked against the replayed *shape* by the same rule
    # the report side uses, rather than against a fixed tuple: a composed chain's
    # shape depends on which stages the registration needed, and the certificate
    # already binds the exact chain through `transform_chain_ref`.
    shape_problem = chain_shape_refusal(
        signed.certificate.compatibility_code, replay.transform_types
    )
    # And the spaces *between* the stages are re-derived rather than accepted.
    # Their identities are a pure function of the two ends the certificate names
    # and the stage each one follows, so a verifier holding the certificate can
    # rebuild them from what it already has. Checking the chain's own ends would
    # be circular — a chain that named its own derived spaces consistently would
    # pass — so the refs come from the certificate, which the pinned key signed.
    expected_intermediates = composed_intermediate_ids(
        replay.transform_types,
        target_ref=signed.certificate.template_space_ref,
        source_ref=signed.certificate.source_space_ref,
    )
    intermediate_problem = (
        None
        if declared != replayed
        or shape_problem is not None
        or expected_intermediates == replay.intermediate_space_ids
        else (
            "the chain's derived spaces are not the ones the two declared ends "
            f"produce: expected {expected_intermediates}, chain names "
            f"{replay.intermediate_space_ids}"
        )
    )
    if declared != replayed or shape_problem is not None or intermediate_problem:
        names = (
            "source_space_ref",
            "template_space_ref",
            "loss_class",
            "invertibility",
        )
        evidence = [
            f"{name}: declared={declared_value!r} replayed={replayed_value!r}"
            for name, declared_value, replayed_value in zip(
                names, declared, replayed
            )
            if declared_value != replayed_value
        ]
        if shape_problem is not None:
            evidence.append(f"compatibility_code: {shape_problem}")
        if intermediate_problem is not None:
            evidence.append(f"intermediate_space_ids: {intermediate_problem}")
        return _refusal(
            CertificateCode.DECLARATION_MISMATCH,
            "the certificate describes a chain other than the one replayed",
            failing_constraint="chain_declaration",
            certificate_ref=actual_ref,
            key_id=pinned_key_id,
            replay=replay,
            evidence=tuple(evidence),
        )

    return CertificateReport(
        code=CertificateCode.CERTIFICATE_VERIFIED,
        explanation=(
            "the certificate is canonical, hashes to its reference, is signed by "
            "the pinned key, and its committed chain replayed to every claimed "
            "source coordinate"
        ),
        certificate_ref=actual_ref,
        key_id=pinned_key_id,
        replay=replay,
    )
