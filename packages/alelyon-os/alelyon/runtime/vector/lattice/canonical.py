"""Canonical byte encoding and content commitments for Lattice records.

Every Lattice record that another party may need to re-derive has exactly one
byte encoding here. The encoding is domain-separated, explicitly length-
prefixed, and free of locale, floating-point, map-ordering and absent-versus-
null ambiguity, so a content hash over these bytes is stable.

What a commitment produced here does and does not establish:

* It detects revision of the committed record. Re-deriving the reference from a
  record and comparing it to a stored reference will fail if either changed.
* It does not authenticate an issuer. There is no key, signature or witness in
  this module, so a reference alone carries no statement about who produced it.
* It does not establish that the declared coordinate meaning is true. A space
  that misdescribes its own axes commits to that misdescription faithfully.

Coordinate spaces are committed in full. A transform commits to its *references*
to the spaces it connects rather than embedding them, so a transform chain stays
small and a verifier resolves each space from a content-addressed table, as the
governing specification requires of externally committed blobs.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from types import MappingProxyType

from alelyon.runtime.vector.lattice.contracts import (
    MAX_AXES,
    MAX_LABEL_ITEMS,
    MAX_METADATA_ITEMS,
    MAX_POLICY_ITEMS,
    MAX_SPACE_ENCODED_BYTES,
    AxisKind,
    CoordinateAxis,
    CoordinateOrdering,
    CoordinateSpace,
    Periodicity,
    ScalarType,
    TopologyType,
)
from alelyon.runtime.vector.lattice.transforms import (
    MAX_TRANSFORM_CHAIN_DEPTH,
    MAX_LABEL_REINDEX_ITEMS,
    AxisOrderingTransform,
    AxisOrientationTransform,
    AxisReferenceShift,
    AxisPermutationTransform,
    AxisTimezoneOffset,
    AxisUnitConversion,
    CalendarTransform,
    ExactTransform,
    IdentityTransform,
    LabelReindexTransform,
    ReferenceBasisTransform,
    TimezoneTransform,
    TransformChain,
    TransformDirection,
    UnitAffineTransform,
)


CANONICAL_SCHEMA = "alelyon.lattice.canonical/0.1"
AXIS_DOMAIN = "alelyon.lattice.canonical.axis/0.1"
SPACE_DOMAIN = "alelyon.lattice.canonical.coordinate-space/0.1"
TRANSFORM_DOMAIN = "alelyon.lattice.canonical.transform/0.1"
CHAIN_DOMAIN = "alelyon.lattice.canonical.transform-chain/0.1"

# A chain embeds only space references, so this bounds the parameter payload
# (chiefly label maps). Encoding or decoding beyond it is a refusal, never a
# truncation.
MAX_CANONICAL_BYTES = 64 * 1024 * 1024
MAX_SPACE_CANONICAL_BYTES = MAX_SPACE_ENCODED_BYTES + (1024 * 1024)

_ABSENT = b"\x00"
_PRESENT = b"\x01"
_U32_MAX = 0xFFFFFFFF


class CanonicalEncodingError(ValueError):
    """A record could not be encoded to, or recovered from, canonical bytes."""


def _u32(value: int) -> bytes:
    if not 0 <= value <= _U32_MAX:
        raise CanonicalEncodingError(f"length {value} is outside the u32 domain")
    return value.to_bytes(4, "big")


def _blob(payload: bytes) -> bytes:
    return _u32(len(payload)) + payload


def _string(value: str) -> bytes:
    return _blob(value.encode("utf-8"))


def _optional_string(value: str | None) -> bytes:
    if value is None:
        return _ABSENT
    return _PRESENT + _string(value)


def _sequence(items: Iterable[bytes]) -> bytes:
    materialized = list(items)
    return _u32(len(materialized)) + b"".join(materialized)


def _domain(identifier: str) -> bytes:
    return identifier.encode("ascii") + b"\x00"


class _Reader:
    """A strict, bounded cursor over canonical bytes."""

    __slots__ = ("_data", "_offset")

    def __init__(self, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise CanonicalEncodingError("canonical input must be bytes")
        self._data = bytes(data)
        self._offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self._offset + count > len(self._data):
            raise CanonicalEncodingError("canonical input ended early")
        chunk = self._data[self._offset : self._offset + count]
        self._offset += count
        return chunk

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "big")

    def blob(self) -> bytes:
        return self.take(self.u32())

    def string(self) -> str:
        raw = self.blob()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalEncodingError("canonical text must be UTF-8") from exc

    def marker(self) -> bool:
        """Read a one-byte absent/present marker, refusing any other value.

        `optional_string` has validated this byte since the decoder was written.
        The three inline marker reads in `_read_axis` did not: they tested
        ``== _PRESENT`` and treated the other 255 values as absent, so 254
        substitutions at each of those offsets decoded to a byte-identical
        record while presenting bytes the encoder cannot emit. A content
        reference means nothing unless exactly one byte string presents a
        record, so the rule lives here once and every marker read goes through
        it.
        """

        marker = self.take(1)
        if marker == _PRESENT:
            return True
        if marker == _ABSENT:
            return False
        raise CanonicalEncodingError("invalid absent/present marker")

    def optional_string(self) -> str | None:
        return self.string() if self.marker() else None

    def count(self, field_name: str, maximum: int) -> int:
        value = self.u32()
        if value > maximum:
            raise CanonicalEncodingError(
                f"{field_name} declares {value} items, above the {maximum} limit"
            )
        return value

    def expect_domain(self, identifier: str) -> None:
        expected = _domain(identifier)
        if self.take(len(expected)) != expected:
            raise CanonicalEncodingError(f"expected the {identifier} domain tag")

    def expect_end(self) -> None:
        if self._offset != len(self._data):
            raise CanonicalEncodingError("canonical input has trailing bytes")


def _content_ref(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Coordinate axes
# --------------------------------------------------------------------------


def axis_bytes(axis: CoordinateAxis) -> bytes:
    """Encode one axis in declared field order."""

    if type(axis) is not CoordinateAxis:
        raise CanonicalEncodingError("axis must be a CoordinateAxis")
    parts = [
        _domain(AXIS_DOMAIN),
        _string(axis.axis_id),
        _string(axis.semantic_id),
        _string(axis.kind.value),
        _string(axis.scalar_type.value),
        _string(axis.ordering.value),
        _optional_string(axis.unit),
        _optional_string(axis.reference_frame),
        _optional_string(axis.calendar),
        _optional_string(axis.timezone),
        _optional_string(axis.orientation),
        _optional_string(axis.origin),
        _optional_string(axis.resolution),
    ]
    if axis.bounds is None:
        parts.append(_ABSENT)
    else:
        parts.append(_PRESENT + _string(axis.bounds[0]) + _string(axis.bounds[1]))
    if axis.periodicity is None:
        parts.append(_ABSENT)
    else:
        parts.append(
            _PRESENT
            + _string(axis.periodicity.period)
            + _string(axis.periodicity.phase)
        )
    parts.append(_optional_string(axis.labels_ref))
    if axis.labels is None:
        parts.append(_ABSENT)
    else:
        parts.append(_PRESENT + _sequence(_string(label) for label in axis.labels))
    parts.append(_string(axis.missingness_policy))
    parts.append(_sequence(_string(item) for item in axis.interpolation_policy))
    parts.append(_sequence(_string(item) for item in axis.transform_policy))
    parts.append(
        _sequence(_string(key) + _string(value) for key, value in axis.metadata)
    )
    return b"".join(parts)


def _read_axis(reader: _Reader) -> CoordinateAxis:
    reader.expect_domain(AXIS_DOMAIN)
    axis_id = reader.string()
    semantic_id = reader.string()
    kind = _read_enum(reader, AxisKind, "axis kind")
    scalar_type = _read_enum(reader, ScalarType, "scalar type")
    ordering = _read_enum(reader, CoordinateOrdering, "ordering")
    unit = reader.optional_string()
    reference_frame = reader.optional_string()
    calendar = reader.optional_string()
    timezone = reader.optional_string()
    orientation = reader.optional_string()
    origin = reader.optional_string()
    resolution = reader.optional_string()
    bounds: tuple[str, str] | None = None
    if reader.marker():
        bounds = (reader.string(), reader.string())
    periodicity: Periodicity | None = None
    if reader.marker():
        periodicity = Periodicity(reader.string(), reader.string())
    labels_ref = reader.optional_string()
    labels: tuple[str, ...] | None = None
    if reader.marker():
        labels = tuple(
            reader.string()
            for _ in range(reader.count("labels", MAX_LABEL_ITEMS))
        )
    missingness_policy = reader.string()
    interpolation_policy = tuple(
        reader.string()
        for _ in range(reader.count("interpolation_policy", MAX_POLICY_ITEMS))
    )
    transform_policy = tuple(
        reader.string()
        for _ in range(reader.count("transform_policy", MAX_POLICY_ITEMS))
    )
    metadata = tuple(
        (reader.string(), reader.string())
        for _ in range(reader.count("metadata", MAX_METADATA_ITEMS))
    )
    # Reconstruction runs every contract validator again, so malformed bytes
    # cannot produce an axis that the constructor would have refused.
    return CoordinateAxis(
        axis_id=axis_id,
        semantic_id=semantic_id,
        kind=kind,
        scalar_type=scalar_type,
        ordering=ordering,
        unit=unit,
        reference_frame=reference_frame,
        calendar=calendar,
        timezone=timezone,
        orientation=orientation,
        origin=origin,
        resolution=resolution,
        bounds=bounds,
        periodicity=periodicity,
        labels_ref=labels_ref,
        labels=labels,
        missingness_policy=missingness_policy,
        interpolation_policy=interpolation_policy,
        transform_policy=transform_policy,
        metadata=metadata,
    )


def _read_enum(reader: _Reader, enum_type, field_name: str):
    raw = reader.string()
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise CanonicalEncodingError(f"unknown {field_name} {raw!r}") from exc


# --------------------------------------------------------------------------
# Coordinate spaces
# --------------------------------------------------------------------------


def coordinate_space_bytes(space: CoordinateSpace) -> bytes:
    """Encode a complete coordinate space."""

    if type(space) is not CoordinateSpace:
        raise CanonicalEncodingError("space must be a CoordinateSpace")
    payload = b"".join(
        (
            _domain(SPACE_DOMAIN),
            _string(space.schema_version),
            _string(space.space_id),
            _string(space.version),
            _string(space.topology.value),
            _sequence(axis_bytes(axis) for axis in space.axes),
            _string(space.index_convention),
            _optional_string(space.unit_system),
            _optional_string(space.reference_frame),
            _string(space.valid_domain_rule),
            _sequence(_string(item) for item in space.region_atlas_refs),
            _sequence(
                _string(key) + _string(value) for key, value in space.metadata
            ),
        )
    )
    if len(payload) > MAX_SPACE_CANONICAL_BYTES:
        raise CanonicalEncodingError(
            f"coordinate space encodes to {len(payload)} bytes, above the "
            f"{MAX_SPACE_CANONICAL_BYTES}-byte canonical limit"
        )
    return payload


def coordinate_space_ref(space: CoordinateSpace) -> str:
    """Return the content reference for a coordinate space."""

    return _content_ref(coordinate_space_bytes(space))


def read_coordinate_space(payload: bytes, *, strict: bool = True) -> CoordinateSpace:
    """Recover a coordinate space from canonical bytes, or refuse.

    Under ``strict`` — the default — the recovered space is re-encoded and
    compared against ``payload``. Decoding alone is not enough to make a
    reference meaningful: several fields are reconstructed into a normal form
    (sets and metadata maps are re-sorted at construction), so bytes that are
    merely *decodable* can present a record whose canonical encoding is a
    different byte string. A caller who decoded such bytes and then committed
    to the result would hold a reference nobody else derives. ``strict=False``
    is for callers that need to inspect a non-canonical record in order to
    report on it — `verify.py` is the only one — and it must never be the
    default for a caller that goes on to trust what it decoded.

    Cost: one extra encode per decode, linear in the input.
    """

    if len(payload) > MAX_SPACE_CANONICAL_BYTES:
        raise CanonicalEncodingError(
            f"coordinate space input is {len(payload)} bytes, above the "
            f"{MAX_SPACE_CANONICAL_BYTES}-byte canonical limit"
        )
    reader = _Reader(payload)
    space = _read_coordinate_space(reader)
    reader.expect_end()
    if strict and coordinate_space_bytes(space) != bytes(payload):
        raise CanonicalEncodingError(
            "the input is not the canonical encoding of the space it decodes to"
        )
    return space


def _read_coordinate_space(reader: _Reader) -> CoordinateSpace:
    reader.expect_domain(SPACE_DOMAIN)
    schema_version = reader.string()
    space_id = reader.string()
    version = reader.string()
    topology = _read_enum(reader, TopologyType, "topology")
    axes = tuple(
        _read_axis(reader) for _ in range(reader.count("axes", MAX_AXES))
    )
    index_convention = reader.string()
    unit_system = reader.optional_string()
    reference_frame = reader.optional_string()
    valid_domain_rule = reader.string()
    region_atlas_refs = tuple(
        reader.string()
        for _ in range(reader.count("region_atlas_refs", MAX_METADATA_ITEMS))
    )
    metadata = tuple(
        (reader.string(), reader.string())
        for _ in range(reader.count("metadata", MAX_METADATA_ITEMS))
    )
    return CoordinateSpace(
        space_id=space_id,
        version=version,
        topology=topology,
        axes=axes,
        index_convention=index_convention,
        unit_system=unit_system,
        reference_frame=reference_frame,
        valid_domain_rule=valid_domain_rule,
        region_atlas_refs=region_atlas_refs,
        metadata=metadata,
        schema_version=schema_version,
    )


# --------------------------------------------------------------------------
# Transforms and chains
# --------------------------------------------------------------------------


def transform_bytes(transform: ExactTransform) -> bytes:
    """Encode one transform, binding its spaces by content reference."""

    transform_type = type(transform)
    if transform_type not in _TRANSFORM_ENCODERS:
        raise CanonicalEncodingError(
            f"unsupported transform type {transform_type.__name__}"
        )
    header = b"".join(
        (
            _domain(TRANSFORM_DOMAIN),
            _string(transform.transform_type),
            _string(transform.direction.value),
            _string(transform.loss_class.value),
            _string(transform.invertibility.value),
            _string(coordinate_space_ref(transform.target_space)),
            _string(coordinate_space_ref(transform.source_space)),
        )
    )
    payload = header + _TRANSFORM_ENCODERS[transform_type](transform)
    if len(payload) > MAX_CANONICAL_BYTES:
        raise CanonicalEncodingError(
            f"transform encodes to {len(payload)} bytes, above the "
            f"{MAX_CANONICAL_BYTES}-byte canonical limit"
        )
    return payload


def _identity_parameters(transform: IdentityTransform) -> bytes:
    return b""


def _permutation_parameters(transform: AxisPermutationTransform) -> bytes:
    return _sequence(_u32(index) for index in transform.source_order)


def _ordering_parameters(transform: AxisOrderingTransform) -> bytes:
    # Which axes are stored back to front. There is nothing else to commit: the
    # reversal has no parameter, and which ordering each side declares is
    # already committed by the two coordinate spaces the header names.
    return _sequence(_u32(index) for index in transform.axis_indexes)


def _calendar_parameters(transform: CalendarTransform) -> bytes:
    # Which axes were re-spelled, and nothing else. The declaration asserts that
    # two calendars admit the same instants; it has no magnitude, and the two
    # calendar names it relates are already committed by the spaces the header
    # binds. These bytes are the whole record of the assertion.
    return _sequence(_u32(index) for index in transform.axis_indexes)


def _orientation_parameters(transform: AxisOrientationTransform) -> bytes:
    # The whole parameter set: which axes were reflected. There is no factor to
    # commit to because the map is negation and nothing else, so these bytes are
    # the entire declaration.
    return _sequence(_u32(index) for index in transform.axis_indexes)


def _label_reindex_parameters(transform: LabelReindexTransform) -> bytes:
    return _u32(transform.axis_index) + _sequence(
        _string(target) + _string(source) for target, source in transform.label_map
    )


def _unit_affine_parameters(transform: UnitAffineTransform) -> bytes:
    # The declared scale and offset are committed as their canonical reduced
    # strings, not as a parsed number: re-deriving them through Fraction would
    # let two spellings share one encoding, and the transform already refuses any
    # spelling that is not canonical.
    return _sequence(
        _u32(conversion.axis_index)
        + _string(conversion.scale)
        + _string(conversion.offset)
        for conversion in transform.conversions
    )


def _reference_basis_parameters(transform: ReferenceBasisTransform) -> bytes:
    # The declared offset is committed as its canonical reduced string, for the
    # same reason a unit factor is: re-deriving it through Fraction would let two
    # spellings share one encoding.
    return _sequence(
        _u32(shift.axis_index) + _string(shift.offset)
        for shift in transform.shifts
    )


def _timezone_parameters(transform: TimezoneTransform) -> bytes:
    # Offsets are signed minutes, encoded as their canonical decimal text rather
    # than a two's-complement integer: the reader already refuses a
    # non-canonical numeric spelling, and text keeps one encoding per offset
    # without introducing a signed-integer width to the format.
    return _sequence(
        _u32(offset.axis_index)
        + _string(str(offset.target_offset_minutes))
        + _string(str(offset.source_offset_minutes))
        for offset in transform.offsets
    )


# Dispatch tables are read-only views. A plain dict here is shared mutable state
# on the commitment path: any code in the process could install an entry and
# change what these bytes commit to, with no refusal and no trace. The proxy
# refuses the mapping API, which closes the accidental route — a stray
# `_TRANSFORM_ENCODERS[X] = f` raises instead of silently re-deciding what these
# bytes commit to.
#
# It does not make the mapping immutable. `gc.get_referents()` on a proxy hands
# back the underlying dict, which is writable, so what holds here is that the
# proxy is the mapping's only holder: it wraps an anonymous literal, nothing
# else is bound to it, and `test_the_dispatch_tables_refuse_mutation` asserts
# that rather than trusting it. Deliberate reflection, and rebinding this module
# attribute, both remain possible — as they do for any Python global — and are
# the loud form of that failure rather than the silent one.
_TRANSFORM_ENCODERS = MappingProxyType(
    {
        IdentityTransform: _identity_parameters,
        AxisPermutationTransform: _permutation_parameters,
        AxisOrderingTransform: _ordering_parameters,
        AxisOrientationTransform: _orientation_parameters,
        CalendarTransform: _calendar_parameters,
        LabelReindexTransform: _label_reindex_parameters,
        UnitAffineTransform: _unit_affine_parameters,
        ReferenceBasisTransform: _reference_basis_parameters,
        TimezoneTransform: _timezone_parameters,
    }
)


def transform_ref(transform: ExactTransform) -> str:
    """Return the content reference for a single transform."""

    return _content_ref(transform_bytes(transform))


def transform_chain_bytes(chain: TransformChain) -> bytes:
    """Encode a chain as an ordered sequence of transform encodings."""

    if type(chain) is not TransformChain:
        raise CanonicalEncodingError("chain must be a TransformChain")
    payload = _domain(CHAIN_DOMAIN) + _sequence(
        transform_bytes(transform) for transform in chain.transforms
    )
    if len(payload) > MAX_CANONICAL_BYTES:
        raise CanonicalEncodingError(
            f"transform chain encodes to {len(payload)} bytes, above the "
            f"{MAX_CANONICAL_BYTES}-byte canonical limit"
        )
    return payload


def transform_chain_ref(chain: TransformChain) -> str:
    """Return the content reference for a transform chain."""

    return _content_ref(transform_chain_bytes(chain))


def read_transform_chain(
    payload: bytes,
    spaces: Mapping[str, CoordinateSpace],
    *,
    strict: bool = True,
) -> TransformChain:
    """Recover a chain from canonical bytes and a content-addressed space table.

    ``spaces`` maps a coordinate-space reference to the space it commits to.
    Every entry is re-derived here, so a table that misfiles a space under the
    wrong reference is refused rather than trusted.

    ``strict`` has the same meaning and the same reason as it does on
    `read_coordinate_space`: the chain is re-encoded and compared against
    ``payload``, because a chain that decodes is not yet a chain these exact
    bytes canonically present. `verify.py` passes ``strict=False`` so that it
    keeps ownership of the distinction between bytes that cannot be decoded
    (`MALFORMED_ENCODING`) and bytes that decode but are not canonical
    (`NONCANONICAL_ENCODING`); a refusal raised here would collapse the second
    report into the first.
    """

    if len(payload) > MAX_CANONICAL_BYTES:
        raise CanonicalEncodingError(
            f"transform chain input is {len(payload)} bytes, above the "
            f"{MAX_CANONICAL_BYTES}-byte canonical limit"
        )
    if not isinstance(spaces, Mapping):
        raise CanonicalEncodingError("spaces must be a mapping of reference to space")
    resolved: dict[str, CoordinateSpace] = {}
    for declared_ref, space in spaces.items():
        if type(space) is not CoordinateSpace:
            raise CanonicalEncodingError("space table values must be CoordinateSpace")
        actual_ref = coordinate_space_ref(space)
        if actual_ref != declared_ref:
            raise CanonicalEncodingError(
                f"space table entry {declared_ref!r} does not commit to its space"
            )
        resolved[actual_ref] = space
    reader = _Reader(payload)
    reader.expect_domain(CHAIN_DOMAIN)
    transforms = tuple(
        _read_transform(reader, resolved)
        for _ in range(reader.count("transforms", MAX_TRANSFORM_CHAIN_DEPTH))
    )
    reader.expect_end()
    chain = TransformChain(transforms)
    if strict and transform_chain_bytes(chain) != bytes(payload):
        raise CanonicalEncodingError(
            "the input is not the canonical encoding of the chain it decodes to"
        )
    return chain


def _resolve_space(
    reference: str,
    spaces: Mapping[str, CoordinateSpace],
) -> CoordinateSpace:
    space = spaces.get(reference)
    if space is None:
        raise CanonicalEncodingError(
            f"coordinate space {reference!r} is not available in the space table"
        )
    return space


def _read_transform(
    reader: _Reader,
    spaces: Mapping[str, CoordinateSpace],
) -> ExactTransform:
    reader.expect_domain(TRANSFORM_DOMAIN)
    transform_type = reader.string()
    direction = _read_enum(reader, TransformDirection, "transform direction")
    declared_loss_class = reader.string()
    declared_invertibility = reader.string()
    target_space = _resolve_space(reader.string(), spaces)
    source_space = _resolve_space(reader.string(), spaces)
    builder = _TRANSFORM_DECODERS.get(transform_type)
    if builder is None:
        raise CanonicalEncodingError(f"unknown transform type {transform_type!r}")
    transform = builder(reader, target_space, source_space)
    if transform.direction is not direction:
        raise CanonicalEncodingError("encoded direction contradicts the transform")
    # The capability surface is recomputed from the reconstructed transform, so
    # encoded bytes claiming a stronger loss class than the type actually has
    # are refused instead of adopted.
    if (
        transform.loss_class.value != declared_loss_class
        or transform.invertibility.value != declared_invertibility
    ):
        raise CanonicalEncodingError(
            f"encoded capability surface contradicts {transform_type}"
        )
    return transform


def _read_identity(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> IdentityTransform:
    return IdentityTransform(target_space, source_space)


def _read_permutation(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> AxisPermutationTransform:
    order = tuple(
        reader.u32() for _ in range(reader.count("source_order", MAX_AXES))
    )
    return AxisPermutationTransform(target_space, source_space, order)


def _read_ordering(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> AxisOrderingTransform:
    indexes = tuple(
        reader.u32() for _ in range(reader.count("axis_indexes", MAX_AXES))
    )
    return AxisOrderingTransform(target_space, source_space, indexes)


def _read_orientation(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> AxisOrientationTransform:
    indexes = tuple(
        reader.u32() for _ in range(reader.count("axis_indexes", MAX_AXES))
    )
    return AxisOrientationTransform(target_space, source_space, indexes)


def _read_calendar(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> CalendarTransform:
    indexes = tuple(
        reader.u32() for _ in range(reader.count("axis_indexes", MAX_AXES))
    )
    return CalendarTransform(target_space, source_space, indexes)


def _read_label_reindex(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> LabelReindexTransform:
    axis_index = reader.u32()
    label_map = tuple(
        (reader.string(), reader.string())
        for _ in range(reader.count("label_map", MAX_LABEL_REINDEX_ITEMS))
    )
    return LabelReindexTransform(target_space, source_space, axis_index, label_map)


def _read_unit_affine(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> UnitAffineTransform:
    conversions = tuple(
        AxisUnitConversion(reader.u32(), reader.string(), reader.string())
        for _ in range(reader.count("conversions", MAX_AXES))
    )
    return UnitAffineTransform(target_space, source_space, conversions)


def _read_reference_basis(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> ReferenceBasisTransform:
    shifts = tuple(
        AxisReferenceShift(reader.u32(), reader.string())
        for _ in range(reader.count("shifts", MAX_AXES))
    )
    return ReferenceBasisTransform(target_space, source_space, shifts)


def _read_signed_minutes(reader: _Reader, field_name: str) -> int:
    """Recover a signed minute count from its one canonical decimal spelling."""

    text = reader.string()
    try:
        value = int(text)
    except ValueError as exc:
        raise CanonicalEncodingError(
            f"{field_name} is not an integer number of minutes"
        ) from exc
    # "+240", "0240" and "-0" all parse to a value this encoder would write
    # differently, so accepting them would give one offset several encodings.
    if str(value) != text:
        raise CanonicalEncodingError(
            f"{field_name} must use its canonical decimal spelling"
        )
    return value


def _read_timezone(
    reader: _Reader,
    target_space: CoordinateSpace,
    source_space: CoordinateSpace,
) -> TimezoneTransform:
    offsets = tuple(
        AxisTimezoneOffset(
            reader.u32(),
            _read_signed_minutes(reader, "target_offset_minutes"),
            _read_signed_minutes(reader, "source_offset_minutes"),
        )
        for _ in range(reader.count("offsets", MAX_AXES))
    )
    return TimezoneTransform(target_space, source_space, offsets)


# Read-only for the same reason, and one step stronger: an installed decoder
# would let forged bytes reconstruct a transform this encoder can never produce,
# which is the replay checker's whole subject.
_TRANSFORM_DECODERS = MappingProxyType(
    {
        "IDENTITY": _read_identity,
        "AXIS_PERMUTATION": _read_permutation,
        "AXIS_ORDERING": _read_ordering,
        "AXIS_ORIENTATION": _read_orientation,
        "CALENDAR": _read_calendar,
        "LABEL_REINDEX": _read_label_reindex,
        "UNIT_AFFINE": _read_unit_affine,
        "REFERENCE_BASIS": _read_reference_basis,
        "TIMEZONE": _read_timezone,
    }
)
