"""Immutable coordinate contracts for Lattice's exact-registration core.

This module defines coordinate meaning only. It performs no payload I/O,
persistence, numerical optimization, or certificate issuance. Its sole
content commitment is a narrow, versioned label-dictionary SHA-256 encoding.
Exact transforms consume these records in transforms.py.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import hashlib
from itertools import islice
import re
from typing import TypeVar
import unicodedata


COORDINATE_SPACE_SCHEMA = "alelyon.lattice.coordinate-space/0.1"
LABEL_DICTIONARY_SCHEMA = "alelyon.lattice.label-dictionary/0.1"
MAX_AXES = 64
MAX_LABEL_ITEMS = 100_000
MAX_LABEL_BYTES = 16 * 1024 * 1024
MAX_POLICY_ITEMS = 128
MAX_METADATA_ITEMS = 256
MAX_LABEL_LENGTH = 4096

# Per-field limits alone permit a space whose declared parts each fit while the
# whole space does not: MAX_AXES axes at MAX_LABEL_BYTES each would accept a
# gigabyte of committed coordinate metadata. The space-wide ceiling below is the
# budget that actually bounds one CoordinateSpace.
MAX_SPACE_ENCODED_BYTES = 32 * 1024 * 1024

_CONTENT_REF = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RATIONAL = re.compile(
    r"-?(?:0|[1-9][0-9]{0,255})(?:/[1-9][0-9]{0,255})?\Z"
)
_T = TypeVar("_T")


class AxisKind(str, Enum):
    CONTINUOUS = "CONTINUOUS"
    DISCRETE_ORDINAL = "DISCRETE_ORDINAL"
    CATEGORICAL = "CATEGORICAL"
    TEMPORAL = "TEMPORAL"
    SPATIAL = "SPATIAL"
    ENTITY = "ENTITY"
    SCENARIO = "SCENARIO"
    ENSEMBLE = "ENSEMBLE"
    FREQUENCY = "FREQUENCY"
    SCALE = "SCALE"
    GRAPH_NODE = "GRAPH_NODE"
    GRAPH_EDGE = "GRAPH_EDGE"
    MESH_VERTEX = "MESH_VERTEX"
    MESH_CELL = "MESH_CELL"
    MODEL_LAYER = "MODEL_LAYER"
    MODEL_STATE = "MODEL_STATE"
    CUSTOM_TYPED = "CUSTOM_TYPED"


class ScalarType(str, Enum):
    INTEGER = "INTEGER"
    RATIONAL = "RATIONAL"
    DECIMAL = "DECIMAL"
    FLOAT = "FLOAT"
    TIMESTAMP = "TIMESTAMP"
    DURATION = "DURATION"
    LABEL = "LABEL"
    UUID = "UUID"
    HASH = "HASH"


class CoordinateOrdering(str, Enum):
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    CANONICAL_LABEL_ORDER = "CANONICAL_LABEL_ORDER"
    UNORDERED = "UNORDERED"


class TopologyType(str, Enum):
    DENSE_REGULAR_GRID = "DENSE_REGULAR_GRID"
    SPARSE_REGULAR_GRID = "SPARSE_REGULAR_GRID"
    RAGGED_LABELED_ARRAY = "RAGGED_LABELED_ARRAY"
    RECTANGULAR_TABLE = "RECTANGULAR_TABLE"
    IRREGULAR_POINT_CLOUD = "IRREGULAR_POINT_CLOUD"
    UNSTRUCTURED_MESH = "UNSTRUCTURED_MESH"
    EVENT_STREAM = "EVENT_STREAM"
    DIRECTED_GRAPH = "DIRECTED_GRAPH"
    UNDIRECTED_GRAPH = "UNDIRECTED_GRAPH"
    HYPERGRAPH = "HYPERGRAPH"
    PRODUCT_SPACE = "PRODUCT_SPACE"


INITIAL_EXACT_TOPOLOGIES = frozenset(
    {
        TopologyType.RECTANGULAR_TABLE,
    }
)


LABEL_COMPATIBLE_AXIS_KINDS = frozenset(
    {
        AxisKind.DISCRETE_ORDINAL,
        AxisKind.CATEGORICAL,
        AxisKind.ENTITY,
        AxisKind.SCENARIO,
        AxisKind.ENSEMBLE,
        AxisKind.GRAPH_NODE,
        AxisKind.GRAPH_EDGE,
        AxisKind.MESH_VERTEX,
        AxisKind.MESH_CELL,
        AxisKind.MODEL_LAYER,
        AxisKind.MODEL_STATE,
        AxisKind.CUSTOM_TYPED,
    }
)


def _bounded_tuple(
    values: Iterable[_T],
    field_name: str,
    maximum: int,
) -> tuple[_T, ...]:
    """Collect at most maximum + 1 items before accepting or refusing."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable, not text")
    try:
        materialized = tuple(islice(iter(values), maximum + 1))
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable") from exc
    if len(materialized) > maximum:
        raise ValueError(f"{field_name} exceeds the {maximum}-item limit")
    return materialized


def _text(
    value: object,
    field_name: str,
    *,
    identifier: bool,
    maximum: int = 4096,
) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty with no outer whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds the {maximum}-character limit")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError(f"{field_name} must use NFC-normalized Unicode")
    # Category "C" covers control (Cc), format (Cf), surrogate (Cs), private-use
    # (Co) and unassigned (Cn) code points. Format characters in particular are
    # invisible when rendered, so permitting them would let two semantic
    # identifiers or two labels look identical while committing to different
    # bytes. Coordinate meaning is decided by exact string equality here, so the
    # whole category is refused. The deliberate cost is that ZWNJ/ZWJ cannot
    # appear in a label; changing that requires a new schema version.
    if any(unicodedata.category(char)[0] == "C" for char in value):
        raise ValueError(
            f"{field_name} must not contain control, format, surrogate, "
            "private-use or unassigned code points"
        )
    if identifier and any(char.isspace() for char in value):
        raise ValueError(f"{field_name} must not contain whitespace")
    return value


def _optional_text(
    value: object | None,
    field_name: str,
    *,
    identifier: bool = False,
) -> str | None:
    if value is None:
        return None
    return _text(value, field_name, identifier=identifier)


def _name_set(
    values: Iterable[object],
    field_name: str,
    *,
    maximum: int = MAX_POLICY_ITEMS,
) -> tuple[str, ...]:
    raw_values = _bounded_tuple(values, field_name, maximum)
    materialized = tuple(
        _text(value, f"{field_name} item", identifier=True, maximum=256)
        for value in raw_values
    )
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(sorted(materialized))


def _metadata(
    value: Mapping[object, object] | Iterable[tuple[object, object]],
) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("metadata must be a mapping or an iterable of key/value pairs")
    raw_source = value.items() if isinstance(value, Mapping) else value
    raw_items = _bounded_tuple(raw_source, "metadata", MAX_METADATA_ITEMS)
    items: list[tuple[str, str]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, tuple) or len(raw_item) != 2:
            raise TypeError("metadata entries must be two-item tuples")
        raw_key, raw_value = raw_item
        key = _text(raw_key, "metadata key", identifier=True, maximum=256)
        item = _text(raw_value, f"metadata[{key!r}]", identifier=False)
        items.append((key, item))
    keys = [key for key, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError("metadata must not contain duplicate keys")
    return tuple(sorted(items))


def _label_values(values: Iterable[object], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable, not text")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable") from exc
    labels: list[str] = []
    seen: set[str] = set()
    encoded_bytes = 0
    for index, value in enumerate(iterator):
        if index >= MAX_LABEL_ITEMS:
            raise ValueError(
                f"{field_name} exceeds the {MAX_LABEL_ITEMS}-item limit"
            )
        label = _text(
            value,
            f"{field_name} item",
            identifier=False,
            maximum=MAX_LABEL_LENGTH,
        )
        encoded_bytes += 8 + len(label.encode("utf-8"))
        if encoded_bytes > MAX_LABEL_BYTES:
            raise ValueError(
                f"{field_name} exceeds the {MAX_LABEL_BYTES}-byte encoding limit"
            )
        if label in seen:
            raise ValueError(f"{field_name} must not contain duplicate labels")
        seen.add(label)
        labels.append(label)
    if not labels:
        raise ValueError(f"{field_name} must contain at least one label")
    return tuple(labels)


def label_dictionary_ref(labels: Iterable[object]) -> str:
    """Return the versioned SHA-256 commitment for an ordered label domain."""

    normalized = _label_values(labels, "labels")
    digest = hashlib.sha256()
    digest.update(LABEL_DICTIONARY_SCHEMA.encode("ascii") + b"\x00")
    for label in normalized:
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return "sha256:" + digest.hexdigest()


def _content_ref(value: object, field_name: str) -> str:
    normalized = _text(value, field_name, identifier=True, maximum=71)
    if _CONTENT_REF.fullmatch(normalized) is None:
        raise ValueError(f"{field_name} must be a lowercase sha256 content reference")
    return normalized


def _rational_value(value: object, field_name: str) -> Fraction:
    """Parse a bounded canonical integer-or-fraction string."""

    normalized = _text(value, field_name, identifier=True, maximum=513)
    if _RATIONAL.fullmatch(normalized) is None:
        raise ValueError(
            f"{field_name} must be a bounded canonical integer or fraction"
        )
    parsed = Fraction(normalized)
    if str(parsed) != normalized:
        raise ValueError(f"{field_name} must be in reduced canonical form")
    return parsed


@dataclass(frozen=True, slots=True)
class Periodicity:
    """An explicit exact period and phase representation."""

    period: str
    phase: str = "0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "period", _text(self.period, "period", identifier=False)
        )
        object.__setattr__(
            self, "phase", _text(self.phase, "phase", identifier=False)
        )
        period_value = _rational_value(self.period, "period")
        _rational_value(self.phase, "phase")
        if period_value <= 0:
            raise ValueError("period must be greater than zero")


@dataclass(frozen=True, slots=True)
class CoordinateAxis:
    """One ordered, semantic axis in a native or canonical coordinate space."""

    axis_id: str
    semantic_id: str
    kind: AxisKind
    scalar_type: ScalarType
    ordering: CoordinateOrdering
    unit: str | None = None
    reference_frame: str | None = None
    calendar: str | None = None
    timezone: str | None = None
    orientation: str | None = None
    origin: str | None = None
    resolution: str | None = None
    bounds: tuple[str, str] | None = None
    periodicity: Periodicity | None = None
    labels_ref: str | None = None
    labels: tuple[str, ...] | None = None
    missingness_policy: str = "TYPED"
    interpolation_policy: tuple[str, ...] = ()
    transform_policy: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    _label_membership: frozenset[str] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )
    _declared_bytes: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "axis_id",
            _text(self.axis_id, "axis_id", identifier=True, maximum=256),
        )
        object.__setattr__(
            self,
            "semantic_id",
            _text(self.semantic_id, "semantic_id", identifier=True, maximum=1024),
        )
        if type(self.kind) is not AxisKind:
            raise TypeError("kind must be an AxisKind")
        if type(self.scalar_type) is not ScalarType:
            raise TypeError("scalar_type must be a ScalarType")
        if type(self.ordering) is not CoordinateOrdering:
            raise TypeError("ordering must be a CoordinateOrdering")
        for name in (
            "unit",
            "reference_frame",
            "calendar",
            "timezone",
            "orientation",
            "origin",
            "resolution",
        ):
            object.__setattr__(
                self,
                name,
                _optional_text(
                    getattr(self, name),
                    name,
                    identifier=name in {"calendar", "timezone"},
                ),
            )
        if self.labels_ref is not None:
            object.__setattr__(
                self,
                "labels_ref",
                _content_ref(self.labels_ref, "labels_ref"),
            )
        if self.labels is not None:
            normalized_labels = _label_values(self.labels, "labels")
            object.__setattr__(self, "labels", normalized_labels)
            object.__setattr__(self, "_label_membership", frozenset(normalized_labels))
            if self.labels_ref is not None:
                expected_ref = label_dictionary_ref(normalized_labels)
                if self.labels_ref != expected_ref:
                    raise ValueError(
                        "labels_ref does not commit to the declared ordered labels"
                    )
        if (
            self.scalar_type is ScalarType.LABEL
            and self.kind not in LABEL_COMPATIBLE_AXIS_KINDS
        ):
            raise ValueError(
                "LABEL scalar_type is incompatible with the declared axis kind"
            )
        if self.scalar_type is not ScalarType.LABEL and (
            self.labels_ref is not None or self.labels is not None
        ):
            raise ValueError(
                "labels_ref and labels are supported only for LABEL scalar_type"
            )
        if self.bounds is not None:
            if type(self.bounds) is not tuple or len(self.bounds) != 2:
                raise TypeError("bounds must be a two-item tuple or None")
            object.__setattr__(
                self,
                "bounds",
                (
                    _text(self.bounds[0], "bounds lower", identifier=False),
                    _text(self.bounds[1], "bounds upper", identifier=False),
                ),
            )
        if self.periodicity is not None and type(self.periodicity) is not Periodicity:
            raise TypeError("periodicity must be Periodicity or None")
        object.__setattr__(
            self,
            "missingness_policy",
            _text(
                self.missingness_policy,
                "missingness_policy",
                identifier=True,
                maximum=256,
            ),
        )
        object.__setattr__(
            self,
            "interpolation_policy",
            _name_set(self.interpolation_policy, "interpolation_policy"),
        )
        object.__setattr__(
            self,
            "transform_policy",
            _name_set(self.transform_policy, "transform_policy"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "_declared_bytes", self._measure_declared_bytes())

    def _measure_declared_bytes(self) -> int:
        """Bytes this axis contributes to its space's declared-size budget."""

        total = 0
        for value in (
            self.axis_id,
            self.semantic_id,
            self.unit,
            self.reference_frame,
            self.calendar,
            self.timezone,
            self.orientation,
            self.origin,
            self.resolution,
            self.labels_ref,
            self.missingness_policy,
        ):
            if value is not None:
                total += 8 + len(value.encode("utf-8"))
        if self.bounds is not None:
            total += sum(8 + len(item.encode("utf-8")) for item in self.bounds)
        if self.periodicity is not None:
            total += 16 + len(self.periodicity.period.encode("utf-8"))
            total += len(self.periodicity.phase.encode("utf-8"))
        for group in (self.labels or (), self.interpolation_policy, self.transform_policy):
            total += sum(8 + len(item.encode("utf-8")) for item in group)
        for key, item in self.metadata:
            total += 16 + len(key.encode("utf-8")) + len(item.encode("utf-8"))
        return total

    @property
    def declared_bytes(self) -> int:
        """The measured contribution of this axis to the space-size budget."""

        return self._declared_bytes

    def registration_metadata_gaps(self) -> tuple[str, ...]:
        """Return metadata that strict exact registration would otherwise guess."""

        gaps: list[str] = []
        if not self.transform_policy:
            gaps.append("transform_policy")
        if self.scalar_type is ScalarType.LABEL:
            if self.labels_ref is None:
                gaps.append("labels_ref")
            if self.labels is None:
                gaps.append("labels")
        if self.kind is AxisKind.TEMPORAL or self.scalar_type is ScalarType.TIMESTAMP:
            if self.calendar is None:
                gaps.append("calendar")
            if self.timezone is None:
                gaps.append("timezone")
        if self.kind is AxisKind.SPATIAL:
            if self.unit is None:
                gaps.append("unit")
            if self.reference_frame is None:
                gaps.append("reference_frame")
            if self.orientation is None:
                gaps.append("orientation")
        if self.kind in {
            AxisKind.CONTINUOUS,
            AxisKind.FREQUENCY,
            AxisKind.SCALE,
        } and self.unit is None:
            gaps.append("unit")
        return tuple(gaps)

    def contains_label(self, value: str) -> bool:
        """Return membership in the resolved, content-checked label domain."""

        return self._label_membership is not None and value in self._label_membership

    def unsupported_exact_features(self) -> tuple[str, ...]:
        """Features declared by the schema but not enforced by this exact slice."""

        unsupported: list[str] = []
        if self.bounds is not None:
            unsupported.append("bounds")
        if self.origin is not None:
            unsupported.append("origin")
        if self.resolution is not None:
            unsupported.append("resolution")
        if self.periodicity is not None:
            unsupported.append("periodicity")
        return tuple(unsupported)

    def exact_semantics_key(
        self,
        *,
        include_labels: bool = True,
        include_unit: bool = True,
        include_timezone: bool = True,
        include_orientation: bool = True,
    ) -> tuple[object, ...]:
        """Fields that must agree for an exact coordinate correspondence.

        A caller excludes exactly one field when a transform family is allowed
        to change that field and nothing else: ``include_orientation=False`` for
        an orientation correction, ``include_labels=False`` for a label reindex,
        ``include_unit=False`` for a unit conversion, ``include_timezone=False``
        for a timezone conversion. Excluding a field here does not permit the
        change on its own — the transform still has to be declared, admitted by
        both axis policies, and exact.

        ``calendar`` is deliberately not relaxable here. A calendar decides which
        instants exist on an axis at all, so changing it is a different rung with
        a different admissibility question, not a re-spelling of one instant.

        ``ordering`` is not relaxable either, and for a different reason: it
        describes the axis's own storage direction, which no transform in this
        slice reads. Correspondence here is by coordinate *value*, so two spaces
        differing only in ``ordering`` still hold the same coordinates and are
        still refused. Relaxing it would require deciding whether by-position
        iteration is part of this contract, which it does not currently state.
        """

        return (
            self.semantic_id,
            self.kind,
            self.scalar_type,
            self.unit if include_unit else None,
            self.reference_frame,
            self.calendar,
            self.timezone if include_timezone else None,
            self.orientation if include_orientation else None,
            self.origin,
            self.resolution,
            self.bounds,
            self.periodicity,
            self.ordering,
            self.labels_ref if include_labels else None,
            self.labels if include_labels else None,
            self.missingness_policy,
            # A declared remapping policy is part of what an axis means, so it
            # belongs to exact correspondence. Leaving it out of both this key
            # and unsupported_exact_features() would make it declared-but-
            # unenforced metadata, which this contract does not allow.
            self.interpolation_policy,
            self.metadata,
        )


@dataclass(frozen=True, slots=True)
class CoordinateSpace:
    """An immutable, ordered coordinate-space definition."""

    space_id: str
    version: str
    topology: TopologyType
    axes: tuple[CoordinateAxis, ...]
    index_convention: str
    unit_system: str | None = None
    reference_frame: str | None = None
    valid_domain_rule: str = "ALL_DECLARED_COORDINATES"
    region_atlas_refs: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    schema_version: str = COORDINATE_SPACE_SCHEMA
    declared_bytes: int = field(
        default=0,
        init=False,
        repr=False,
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "space_id",
            _text(self.space_id, "space_id", identifier=True, maximum=512),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, "version", identifier=True, maximum=128),
        )
        if type(self.topology) is not TopologyType:
            raise TypeError("topology must be a TopologyType")
        axes = _bounded_tuple(self.axes, "axes", MAX_AXES)
        if not axes:
            raise ValueError("coordinate space must contain at least one axis")
        if any(type(axis) is not CoordinateAxis for axis in axes):
            raise TypeError("axes must contain only CoordinateAxis values")
        axis_ids = [axis.axis_id for axis in axes]
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("axis_id values must be unique within a coordinate space")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(
            self,
            "index_convention",
            _text(self.index_convention, "index_convention", identifier=True),
        )
        object.__setattr__(
            self,
            "unit_system",
            _optional_text(self.unit_system, "unit_system", identifier=True),
        )
        object.__setattr__(
            self,
            "reference_frame",
            _optional_text(self.reference_frame, "reference_frame"),
        )
        object.__setattr__(
            self,
            "valid_domain_rule",
            _text(self.valid_domain_rule, "valid_domain_rule", identifier=True),
        )
        object.__setattr__(
            self,
            "region_atlas_refs",
            _name_set(
                self.region_atlas_refs,
                "region_atlas_refs",
                maximum=MAX_METADATA_ITEMS,
            ),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.schema_version != COORDINATE_SPACE_SCHEMA:
            raise ValueError(
                f"schema_version must be {COORDINATE_SPACE_SCHEMA!r} "
                "for this implementation"
            )
        declared = sum(axis.declared_bytes for axis in self.axes)
        declared += sum(
            8 + len(value.encode("utf-8"))
            for value in (
                self.space_id,
                self.version,
                self.index_convention,
                self.valid_domain_rule,
                self.schema_version,
            )
        )
        for value in (self.unit_system, self.reference_frame):
            if value is not None:
                declared += 8 + len(value.encode("utf-8"))
        declared += sum(
            8 + len(item.encode("utf-8")) for item in self.region_atlas_refs
        )
        declared += sum(
            16 + len(key.encode("utf-8")) + len(item.encode("utf-8"))
            for key, item in self.metadata
        )
        if declared > MAX_SPACE_ENCODED_BYTES:
            raise ValueError(
                f"coordinate space declares {declared} bytes, which exceeds the "
                f"{MAX_SPACE_ENCODED_BYTES}-byte space budget"
            )
        object.__setattr__(self, "declared_bytes", declared)

    def axis_index(self, axis_id: str) -> int:
        """Return an axis position or raise a named lookup error."""

        axis_id = _text(axis_id, "axis_id", identifier=True, maximum=256)
        for index, axis in enumerate(self.axes):
            if axis.axis_id == axis_id:
                return index
        raise KeyError(f"unknown axis_id {axis_id!r}")

    def registration_metadata_gaps(self) -> tuple[str, ...]:
        gaps: list[str] = []
        for axis in self.axes:
            gaps.extend(
                f"{axis.axis_id}.{field}"
                for field in axis.registration_metadata_gaps()
            )
        return tuple(gaps)

    def unsupported_exact_features(self) -> tuple[str, ...]:
        unsupported: list[str] = []
        if self.valid_domain_rule != "ALL_DECLARED_COORDINATES":
            unsupported.append("valid_domain_rule")
        for axis in self.axes:
            unsupported.extend(
                f"{axis.axis_id}.{feature}"
                for feature in axis.unsupported_exact_features()
            )
        return tuple(unsupported)

    def exact_space_key(self) -> tuple[object, ...]:
        """Space-wide semantics that cannot be repaired by axis permutation."""

        return (
            self.topology,
            self.index_convention,
            self.unit_system,
            self.reference_frame,
            self.valid_domain_rule,
            self.region_atlas_refs,
            self.metadata,
        )
