"""Exact target-to-source coordinate transforms for Lattice."""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as fixed_timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import math
import re
from types import MappingProxyType
from typing import ClassVar, TypeAlias
from uuid import UUID

from alelyon.runtime.vector.lattice.contracts import (
    CoordinateAxis,
    CoordinateOrdering,
    CoordinateSpace,
    INITIAL_EXACT_TOPOLOGIES,
    MAX_AXES,
    ScalarType,
    _bounded_tuple,
    _rational_value,
    _text,
)


MAX_TRANSFORM_CHAIN_DEPTH = 64
MAX_LABEL_REINDEX_ITEMS = 100_000
MAX_LABEL_REINDEX_BYTES = 32 * 1024 * 1024

MAX_DECIMAL_LENGTH = 512

# A scale-and-offset map — and its parameterless special case, negation — is
# exact only over coordinates that carry their own exact arithmetic. INTEGER,
# RATIONAL and DECIMAL do; FLOAT does not, because the product of two
# representable binary floats generally is not representable, and TIMESTAMP,
# DURATION, LABEL, UUID and HASH are not a signed numeric line at all. One
# constant serves both the unit and orientation families rather than two copies,
# so the two cannot drift apart without someone editing this line.
EXACT_NUMERIC_SCALAR_TYPES = frozenset(
    {
        ScalarType.INTEGER,
        ScalarType.RATIONAL,
        ScalarType.DECIMAL,
    }
)

# `datetime.timezone` accepts strictly between -24h and +24h. The coordinate
# grammar already requires a minute-aligned offset, so minutes is the unit.
MAX_UTC_OFFSET_MINUTES = 1440

# An orientation correction has no declared parameters: it is exactly negation.
# These are named so the reflection reads as what it is at the call site, and so
# nothing can pass a different pair by accident.
_REFLECTION_SCALE = Fraction(-1)
_REFLECTION_OFFSET = Fraction(0)

# A reference-basis change is a translation: the coordinate keeps its scale and
# moves its origin, so the multiplier is fixed at one and only the offset is
# declared.
_TRANSLATION_SCALE = Fraction(1)

_CONTENT_REF = re.compile(r"sha256:[0-9a-f]{64}\Z")
# One coordinate must have exactly one spelling, or byte-distinct coordinates
# that denote the same position would produce distinct canonical addresses and
# commitments later. Exponents, leading '+', leading zeros, trailing fractional
# zeros and negative zero are therefore refused rather than normalized.
_CANONICAL_DECIMAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*[1-9])?\Z")
# Extended ISO-8601 with a minute-aligned numeric offset. 'Z', basic format and
# sub-minute offsets are rejected here; the exact spelling is then pinned by
# requiring datetime.isoformat() to reproduce the input byte for byte.
_CANONICAL_TIMESTAMP = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{6})?[+-][0-9]{2}:[0-9]{2}\Z"
)
_DAY_TIME_DURATION = re.compile(
    r"(?P<sign>[+-])?P"
    r"(?:(?P<days>0|[1-9][0-9]{0,18})D)?"
    r"(?:(?P<time>T)"
    r"(?:(?P<hours>0|[1-9][0-9]{0,18})H)?"
    r"(?:(?P<minutes>0|[1-9][0-9]{0,18})M)?"
    r"(?:(?P<seconds>0|[1-9][0-9]{0,18})"
    r"(?:\.[0-9]{1,18})?S)?"
    r")?\Z"
)


class TransformDirection(str, Enum):
    TARGET_TO_SOURCE = "TARGET_TO_SOURCE"


class LossClass(str, Enum):
    """What a transform may do to the information it carries.

    Ordered from strongest to weakest; a chain declares the weakest class any
    member declares.
    """

    LOSSLESS = "LOSSLESS"
    EXACT_WITH_REPRESENTATION_CHANGE = "EXACT_WITH_REPRESENTATION_CHANGE"
    BOUNDED_LOSS = "BOUNDED_LOSS"
    LOSSY_REQUIRES_EXPLICIT_APPROVAL = "LOSSY_REQUIRES_EXPLICIT_APPROVAL"
    PROHIBITED = "PROHIBITED"


# A read-only view, not a dict: this ordering decides the class a chain declares
# and gates the replay checker's maximum-loss refusal, so a mutable table would
# let a chain be re-ranked into passing a gate it should fail.
LOSS_CLASS_RANK: Mapping[LossClass, int] = MappingProxyType(
    {
        LossClass.LOSSLESS: 0,
        LossClass.EXACT_WITH_REPRESENTATION_CHANGE: 1,
        LossClass.BOUNDED_LOSS: 2,
        LossClass.LOSSY_REQUIRES_EXPLICIT_APPROVAL: 3,
        LossClass.PROHIBITED: 4,
    }
)


class Invertibility(str, Enum):
    EXACT = "EXACT"
    BOUNDED = "BOUNDED"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class ProhibitedTransformError(ValueError):
    """A declared axis policy rejected a requested transform family."""

    def __init__(self, role: str, axis_id: str, transform_type: str) -> None:
        self.role = role
        self.axis_id = axis_id
        self.transform_type = transform_type
        super().__init__(
            f"{role} axis {axis_id!r} does not allow {transform_type}"
        )


def _validate_coordinate_value(value: object, axis: CoordinateAxis) -> None:
    scalar_type = axis.scalar_type
    if scalar_type is ScalarType.INTEGER:
        if type(value) is not int:
            raise TypeError(f"axis {axis.axis_id!r} requires an INTEGER coordinate")
        return
    if scalar_type is ScalarType.RATIONAL:
        if not isinstance(value, str):
            raise TypeError(f"axis {axis.axis_id!r} requires a RATIONAL string")
        _rational_value(value, f"axis {axis.axis_id!r} rational")
        return
    if scalar_type is ScalarType.DECIMAL:
        if not isinstance(value, str):
            raise TypeError(f"axis {axis.axis_id!r} requires a DECIMAL string")
        _text(
            value,
            f"axis {axis.axis_id!r} decimal",
            identifier=True,
            maximum=MAX_DECIMAL_LENGTH,
        )
        # "-0.0" already fails the regex because a fraction must end in 1-9, so
        # the exact string "-0" is the only remaining negative-zero spelling.
        if _CANONICAL_DECIMAL.fullmatch(value) is None or value == "-0":
            raise ValueError(
                f"axis {axis.axis_id!r} requires a canonical DECIMAL coordinate "
                "with no exponent, leading sign, leading zero, trailing "
                "fractional zero or negative zero"
            )
        try:
            parsed_decimal = Decimal(value)
        except InvalidOperation as exc:  # pragma: no cover - regex precedes this
            raise ValueError(
                f"axis {axis.axis_id!r} has an invalid DECIMAL coordinate"
            ) from exc
        if not parsed_decimal.is_finite():  # pragma: no cover - regex precedes this
            raise ValueError(
                f"axis {axis.axis_id!r} requires a finite DECIMAL coordinate"
            )
        return
    if scalar_type is ScalarType.FLOAT:
        if type(value) is not float:
            raise TypeError(f"axis {axis.axis_id!r} requires a FLOAT coordinate")
        if not math.isfinite(value):
            raise ValueError(
                f"axis {axis.axis_id!r} requires a finite FLOAT coordinate"
            )
        if value == 0.0 and math.copysign(1.0, value) < 0.0:
            raise ValueError(
                f"axis {axis.axis_id!r} refuses negative zero; this slice has no "
                "signed-zero coordinate policy"
            )
        return
    if scalar_type is ScalarType.TIMESTAMP:
        if not isinstance(value, str):
            raise TypeError(f"axis {axis.axis_id!r} requires a TIMESTAMP string")
        _text(value, f"axis {axis.axis_id!r} timestamp", identifier=True, maximum=64)
        if _CANONICAL_TIMESTAMP.fullmatch(value) is None:
            raise ValueError(
                f"axis {axis.axis_id!r} requires an extended ISO-8601 TIMESTAMP "
                "coordinate with seconds and a minute-aligned numeric offset"
            )
        try:
            parsed_timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"axis {axis.axis_id!r} has an invalid ISO-8601 TIMESTAMP coordinate"
            ) from exc
        if (  # pragma: no cover - the canonical form already mandates an offset
            parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None
        ):
            raise ValueError(
                f"axis {axis.axis_id!r} requires an offset-aware TIMESTAMP coordinate"
            )
        # Equal instants written with different declared offsets stay distinct
        # coordinates. Unifying them is a TimezoneTransform, which does not exist
        # in this slice, so the offset must be carried explicitly rather than
        # silently normalized.
        if parsed_timestamp.isoformat() != value:
            raise ValueError(
                f"axis {axis.axis_id!r} requires a canonical TIMESTAMP coordinate "
                "spelling; fractional seconds must be absent or exactly six digits"
            )
        return
    if scalar_type is ScalarType.DURATION:
        if not isinstance(value, str):
            raise TypeError(f"axis {axis.axis_id!r} requires a DURATION string")
        _text(value, f"axis {axis.axis_id!r} duration", identifier=True)
        match = _DAY_TIME_DURATION.fullmatch(value)
        components = ("days", "hours", "minutes", "seconds")
        if match is None or not any(match.group(name) for name in components):
            raise ValueError(
                f"axis {axis.axis_id!r} has an unsupported DURATION coordinate"
            )
        if match.group("time") and not any(
            match.group(name) for name in ("hours", "minutes", "seconds")
        ):
            raise ValueError(
                f"axis {axis.axis_id!r} has an unsupported DURATION coordinate"
            )
        return
    if scalar_type is ScalarType.LABEL:
        label = _text(value, f"axis {axis.axis_id!r} label", identifier=False)
        if not axis.contains_label(label):
            raise ValueError(
                f"axis {axis.axis_id!r} label is outside the committed domain"
            )
        return
    if scalar_type is ScalarType.UUID:
        if not isinstance(value, str):
            raise TypeError(f"axis {axis.axis_id!r} requires a UUID string")
        _text(value, f"axis {axis.axis_id!r} uuid", identifier=True)
        try:
            parsed_uuid = UUID(value)
        except ValueError as exc:
            raise ValueError(f"axis {axis.axis_id!r} has an invalid UUID") from exc
        if str(parsed_uuid) != value:
            raise ValueError(f"axis {axis.axis_id!r} requires canonical UUID text")
        return
    if scalar_type is ScalarType.HASH:
        if type(value) is not str:
            raise TypeError(f"axis {axis.axis_id!r} requires a HASH string")
        _text(value, f"axis {axis.axis_id!r} hash", identifier=True)
        if _CONTENT_REF.fullmatch(value) is None:
            raise ValueError(
                f"axis {axis.axis_id!r} requires a lowercase sha256 HASH coordinate"
            )
        return
    raise TypeError(f"axis {axis.axis_id!r} has an unsupported scalar type")


def _coordinates(
    value: tuple[object, ...],
    space: CoordinateSpace,
) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError("coordinates must be an immutable tuple")
    if len(value) != len(space.axes):
        raise ValueError(
            "INSUFFICIENT_METADATA: coordinate arity "
            f"{len(value)} does not match target axis count {len(space.axes)}"
        )
    for coordinate, axis in zip(value, space.axes):
        _validate_coordinate_value(coordinate, axis)
    return value


def _spaces_match_exactly(
    target: CoordinateSpace,
    source: CoordinateSpace,
    *,
    include_labels: bool = True,
) -> bool:
    return (
        target.exact_space_key() == source.exact_space_key()
        and len(target.axes) == len(source.axes)
        and all(
            target_axis.exact_semantics_key(include_labels=include_labels)
            == source_axis.exact_semantics_key(include_labels=include_labels)
            for target_axis, source_axis in zip(target.axes, source.axes)
        )
    )


def _require_space_values(
    target: CoordinateSpace,
    source: CoordinateSpace,
) -> None:
    if type(target) is not CoordinateSpace or type(source) is not CoordinateSpace:
        raise TypeError("target_space and source_space must be CoordinateSpace values")
    if (
        target.topology not in INITIAL_EXACT_TOPOLOGIES
        or source.topology not in INITIAL_EXACT_TOPOLOGIES
    ):
        raise ValueError(
            "UNSUPPORTED_TOPOLOGY: this exact slice supports only rectangular tables"
        )
    if target.topology is not source.topology:
        raise ValueError("INCOMPATIBLE_TOPOLOGY: coordinate topologies differ")
    if len(target.axes) != len(source.axes):
        raise ValueError("INCOMPATIBLE_TOPOLOGY: axis counts differ")
    for role, space in (("target", target), ("source", source)):
        unsupported = space.unsupported_exact_features()
        if unsupported:
            raise ValueError(
                "UNSUPPORTED_VALUE_SEMANTICS: "
                f"{role} coordinate space declares {', '.join(unsupported)}"
            )


def _require_registration_metadata(*spaces: CoordinateSpace) -> None:
    for role, space in zip(("target", "source"), spaces):
        gaps = space.registration_metadata_gaps()
        if gaps:
            raise ValueError(
                "INSUFFICIENT_METADATA: "
                f"{role} coordinate space is missing {', '.join(gaps)}"
            )


def _require_axis_policy(
    transform_type: str,
    target_axis: CoordinateAxis,
    source_axis: CoordinateAxis,
) -> None:
    for role, axis in (("target", target_axis), ("source", source_axis)):
        if transform_type not in axis.transform_policy:
            raise ProhibitedTransformError(role, axis.axis_id, transform_type)


@dataclass(frozen=True, slots=True)
class IdentityTransform:
    """An exact identity sampling map between semantically equal spaces."""

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "IDENTITY"

    # Declared capability surface. These are class constants rather than fields
    # so a caller cannot construct a transform that misreports what it does.
    loss_class: ClassVar[LossClass] = LossClass.LOSSLESS
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "IDENTITY":
            raise ValueError("identity transform_type must be IDENTITY")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if not _spaces_match_exactly(self.target_space, self.source_space):
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: identity requires equal ordered "
                "coordinate semantics"
            )
        for target_axis, source_axis in zip(
            self.target_space.axes, self.source_space.axes
        ):
            _require_axis_policy("IDENTITY", target_axis, source_axis)

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        return _coordinates(target_coordinates, self.target_space)

    def invert(self) -> IdentityTransform:
        return IdentityTransform(self.source_space, self.target_space)


@dataclass(frozen=True, slots=True)
class AxisPermutationTransform:
    """Map a target coordinate tuple into the source's axis order.

    source_order[j] is the target-axis index supplying source axis j.
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    source_order: tuple[int, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "AXIS_PERMUTATION"

    # Reordering axes moves coordinates without changing any declared meaning.
    loss_class: ClassVar[LossClass] = LossClass.LOSSLESS
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "AXIS_PERMUTATION":
            raise ValueError(
                "axis permutation transform_type must be AXIS_PERMUTATION"
            )
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        order = _bounded_tuple(
            self.source_order,
            "source_order",
            len(self.target_space.axes),
        )
        axis_count = len(self.target_space.axes)
        if (
            len(order) != axis_count
            or any(
                type(index) is not int
                for index in order
            )
            or set(order) != set(range(axis_count))
        ):
            raise ValueError("source_order must be a complete, bijective permutation")
        if order == tuple(range(axis_count)):
            raise ValueError("identity source_order must use IdentityTransform")
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        for source_index, target_index in enumerate(order):
            target_axis = self.target_space.axes[target_index]
            source_axis = self.source_space.axes[source_index]
            if target_axis.exact_semantics_key() != source_axis.exact_semantics_key():
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: source_order maps "
                    f"target axis {target_axis.axis_id!r} to incompatible source "
                    f"axis {source_axis.axis_id!r}"
                )
            policy = "IDENTITY" if source_index == target_index else "AXIS_PERMUTATION"
            _require_axis_policy(policy, target_axis, source_axis)
        object.__setattr__(self, "source_order", order)

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = _coordinates(target_coordinates, self.target_space)
        return tuple(coordinates[target_index] for target_index in self.source_order)

    def invert(self) -> AxisPermutationTransform:
        inverse = [0] * len(self.source_order)
        for source_index, target_index in enumerate(self.source_order):
            inverse[target_index] = source_index
        return AxisPermutationTransform(
            self.source_space,
            self.target_space,
            tuple(inverse),
        )


#: The two orderings that are each other's reverse. `UNORDERED` has no direction
#: to reverse and `CANONICAL_LABEL_ORDER` is fixed by the label dictionary, so
#: neither can take part in a storage-direction correction.
REVERSIBLE_ORDERINGS = frozenset(
    {
        CoordinateOrdering.ASCENDING,
        CoordinateOrdering.DESCENDING,
    }
)


@dataclass(frozen=True, slots=True)
class AxisOrderingTransform:
    """Reconcile axes stored in opposite directions. No coordinate moves.

    §11.5 requires a canonical cell address to be derived from "canonical
    coordinate values or labels" and forbids it depending on "storage chunk
    placement". Addressing is therefore by *value*, and an axis stored back to
    front holds the same coordinates at the same addresses. `apply_coordinates`
    is the identity here, and that is the finding rather than a stub: a
    difference in `ordering` cannot move a coordinate in this model.

    So why a record at all? §8.4 — every transformation is explicit. Two spaces
    that disagree about storage direction *do* differ, and a consumer that walks
    an axis by position rather than by address has to reverse one of them. A
    chain that silently registered these as identical would take that difference
    out of the audit trail, which is the one thing it must not do.

    Alone among the families that relax a field, this one needs **no
    declaration**. `CoordinateOrdering` is a closed enum defined by this
    contract, and "the reverse of ASCENDING is DESCENDING" is a fact about the
    schema rather than about the world. The unit, timezone, label and
    orientation vocabularies are all open and carry no algebra, which is exactly
    why each of those needs a caller to supply the relationship and this one does
    not.
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    axis_indexes: tuple[int, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "AXIS_ORDERING"

    # Every coordinate corresponds to itself, unchanged, in the same alphabet.
    # Only a declared layout attribute differs, so this is as lossless as an axis
    # permutation — which likewise moves things without changing any of them.
    loss_class: ClassVar[LossClass] = LossClass.LOSSLESS
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "AXIS_ORDERING":
            raise ValueError("ordering transform_type must be AXIS_ORDERING")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        indexes = _bounded_tuple(self.axis_indexes, "axis_indexes", MAX_AXES)
        if any(
            type(index) is not int or isinstance(index, bool) for index in indexes
        ):
            raise TypeError("axis_indexes must contain only integers")
        if len(set(indexes)) != len(indexes):
            raise ValueError("axis_indexes must name each axis at most once")
        if not indexes:
            raise ValueError(
                "an ordering correction must reverse at least one axis; equal "
                "orderings must use IdentityTransform"
            )
        axis_count = len(self.target_space.axes)
        if any(index < 0 or index >= axis_count for index in indexes):
            raise ValueError("axis_indexes names an axis outside the target space")
        reversed_axes = frozenset(indexes)
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes, strict=True)
        ):
            include_ordering = index not in reversed_axes
            if target_axis.exact_semantics_key(
                include_ordering=include_ordering
            ) != source_axis.exact_semantics_key(
                include_ordering=include_ordering
            ):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: an ordering correction may change "
                    "only the declared ordering on the axes it names"
                )
            if include_ordering:
                _require_axis_policy("IDENTITY", target_axis, source_axis)
                continue
            pair = {target_axis.ordering, source_axis.ordering}
            if pair != REVERSIBLE_ORDERINGS:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: an ordering correction "
                    "reverses ASCENDING against DESCENDING; axis "
                    f"{target_axis.axis_id!r} declares "
                    f"{target_axis.ordering.value} against "
                    f"{source_axis.ordering.value}"
                )
            _require_axis_policy("AXIS_ORDERING", target_axis, source_axis)
        object.__setattr__(self, "axis_indexes", tuple(sorted(indexes)))

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        # Deliberately the identity. §11.5 addresses a cell by coordinate value,
        # so reversing an axis's storage direction leaves every address where it
        # was. The record carries the difference; the map has nothing to do.
        return _coordinates(target_coordinates, self.target_space)

    def invert(self) -> AxisOrderingTransform:
        return AxisOrderingTransform(
            self.source_space,
            self.target_space,
            self.axis_indexes,
        )


@dataclass(frozen=True, slots=True)
class CalendarTransform:
    """Re-spell the calendar an axis names. No coordinate moves.

    §11.3 makes ``calendar`` one field meaning "calendar/session definition", so
    a calendar here is the rule deciding **which instants exist** on an axis —
    an exchange session as much as a leap rule. It does not decide where an
    instant sits. Two calendars that admit the same instants therefore hold the
    same coordinates, and `apply_coordinates` is the identity, for the same
    reason `AxisOrderingTransform`'s is: the record carries a difference the map
    has nothing to do about. §8.4 still requires that difference to be explicit.

    What this rung refuses is the case the spec names directly — "calendar
    reindexing without inventing observations on closed days". Two calendars
    admitting *different* instants do not correspond exactly in either
    direction: one way invents observations on days the other closes, the other
    drops them. That is a resampling, rung 11 at best, and this slice does not
    do it. Nothing here can tell the two cases apart from the names, which is
    why a caller has to.

    So the declaration asserts set equality of the instants and nothing else.
    That it can stop there is a consequence of a limit this slice already has:
    `unsupported_exact_features` refuses any axis carrying `origin`, `bounds`,
    `resolution` or `periodicity`, which are the fields that would rest on the
    *separate* claim that both calendars number the same instants alike — Julian
    and Gregorian admit every real day, and disagree about what "1900-02-28"
    names. With those fields refused the axis says nothing a calendar
    reinterprets, so one assertion is sufficient. It would not be if that
    refusal were lifted, and
    `test_the_calendar_rung_rests_on_the_unsupported_feature_refusal` fails on
    purpose if it is.

    Where the instants *are* committed, this is the weaker path and a caller
    should not be on it: an axis that enumerates its sessions as a LABEL domain
    gets `LabelReindexTransform`, whose declared map is checked for bijectivity
    against both committed dictionaries. This family exists for the axis whose
    instant set is not committed, where the two calendar names are the only
    evidence and no check is possible.
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    axis_indexes: tuple[int, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "CALENDAR"

    # Every coordinate corresponds to itself, unchanged, in the same alphabet:
    # the declaration is that both calendars admit exactly these instants, so
    # there is nothing for a representation change to consist of. Unlike the
    # timezone rung next to it on rung 6, no number is re-spelled.
    loss_class: ClassVar[LossClass] = LossClass.LOSSLESS
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "CALENDAR":
            raise ValueError("calendar transform_type must be CALENDAR")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        indexes = _bounded_tuple(self.axis_indexes, "axis_indexes", MAX_AXES)
        if any(
            type(index) is not int or isinstance(index, bool) for index in indexes
        ):
            raise TypeError("axis_indexes must contain only integers")
        if len(set(indexes)) != len(indexes):
            raise ValueError("axis_indexes must name each axis at most once")
        if not indexes:
            raise ValueError(
                "a calendar correspondence must re-spell at least one axis; "
                "equal calendars must use IdentityTransform"
            )
        axis_count = len(self.target_space.axes)
        if any(index < 0 or index >= axis_count for index in indexes):
            raise ValueError("axis_indexes names an axis outside the target space")
        respelled = frozenset(indexes)
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes, strict=True)
        ):
            include_calendar = index not in respelled
            if target_axis.exact_semantics_key(
                include_calendar=include_calendar
            ) != source_axis.exact_semantics_key(
                include_calendar=include_calendar
            ):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: a calendar correspondence may "
                    "change only the declared calendar on the axes it names"
                )
            if include_calendar:
                _require_axis_policy("IDENTITY", target_axis, source_axis)
                continue
            # A calendar declared on one side only leaves nothing to assert set
            # equality *against*, so there is no correspondence to state.
            if target_axis.calendar is None or source_axis.calendar is None:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: axis "
                    f"{target_axis.axis_id!r} declares a calendar on only one "
                    "side, so no correspondence between them can be exact"
                )
            if target_axis.calendar == source_axis.calendar:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: axis "
                    f"{target_axis.axis_id!r} declares the same calendar on "
                    "both sides; equal calendars must use IdentityTransform"
                )
            _require_axis_policy("CALENDAR", target_axis, source_axis)
        object.__setattr__(self, "axis_indexes", tuple(sorted(indexes)))

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        # Deliberately the identity, and this is the finding rather than a stub.
        # The declaration is that both calendars admit the same instants; an
        # instant that exists under both is the same instant, so there is
        # nothing to move. A calendar that moved coordinates would be one that
        # renumbers them, which this rung does not admit.
        return _coordinates(target_coordinates, self.target_space)

    def invert(self) -> CalendarTransform:
        return CalendarTransform(
            self.source_space,
            self.target_space,
            self.axis_indexes,
        )


@dataclass(frozen=True, slots=True)
class AxisOrientationTransform:
    """Reflect one or more axes that point the opposite way: source = -target.

    The named axes declare *different* orientations on the two sides, and this
    transform is the statement that those two declarations are exact opposites
    of one another. It carries no parameters at all. §8.6 asks for the
    least-expressive sufficient transform, and "this axis points the other way"
    has no magnitude in it: a reflection about some other centre is a reflection
    *and* a translation, which is rung 5 or 7 and must be composed, not folded
    into this record.

    The flip is *declared*, like every other field this slice can relax, and for
    a reason of its own. ``orientation`` holds a free-text domain code — "RAS",
    "LPS", "up", "depth-positive-down" — and that vocabulary is open and has no
    algebra. Nothing here can compute that "LPS" is the reverse of "RAS" rather
    than an unrelated basis or a permutation of three axes, so a caller says so
    and the record commits to having been told. Nothing in the data can
    contradict it either: with `bounds`, `origin` and `resolution` all outside
    this slice, the coordinates carry no evidence about which way they run.
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    axis_indexes: tuple[int, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "AXIS_ORIENTATION"

    # Every coordinate survives and negation is its own inverse, but the numbers
    # change sign because the direction they are quoted against changed. That is
    # a representation change, not an identity.
    loss_class: ClassVar[LossClass] = LossClass.EXACT_WITH_REPRESENTATION_CHANGE
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "AXIS_ORIENTATION":
            raise ValueError(
                "orientation transform_type must be AXIS_ORIENTATION"
            )
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        indexes = _bounded_tuple(self.axis_indexes, "axis_indexes", MAX_AXES)
        if any(
            type(index) is not int or isinstance(index, bool) for index in indexes
        ):
            raise TypeError("axis_indexes must contain only integers")
        if len(set(indexes)) != len(indexes):
            raise ValueError("axis_indexes must name each axis at most once")
        if not indexes:
            raise ValueError(
                "an orientation correction must reflect at least one axis; equal "
                "orientations must use IdentityTransform"
            )
        axis_count = len(self.target_space.axes)
        if any(index < 0 or index >= axis_count for index in indexes):
            raise ValueError("axis_indexes names an axis outside the target space")
        flipped = frozenset(indexes)
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes, strict=True)
        ):
            include_orientation = index not in flipped
            if target_axis.exact_semantics_key(
                include_orientation=include_orientation
            ) != source_axis.exact_semantics_key(
                include_orientation=include_orientation
            ):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: an orientation correction may "
                    "change only the declared orientation on the axes it names"
                )
            if include_orientation:
                _require_axis_policy("IDENTITY", target_axis, source_axis)
                continue
            if target_axis.scalar_type not in EXACT_NUMERIC_SCALAR_TYPES:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: orientation correction "
                    "requires an INTEGER, RATIONAL or DECIMAL axis, not "
                    f"{target_axis.scalar_type.value}"
                )
            # An axis that never said which way it points cannot claim to point
            # the other way. This is the metadata gap named, not inferred away.
            if target_axis.orientation is None or source_axis.orientation is None:
                raise ValueError(
                    "INSUFFICIENT_METADATA: a reflected axis must declare an "
                    "orientation on both sides"
                )
            if target_axis.orientation == source_axis.orientation:
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: axis "
                    f"{target_axis.axis_id!r} declares the same orientation on "
                    "both sides, so no orientation correction applies"
                )
            _require_axis_policy("AXIS_ORIENTATION", target_axis, source_axis)
        object.__setattr__(self, "axis_indexes", tuple(sorted(indexes)))

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = list(_coordinates(target_coordinates, self.target_space))
        for index in self.axis_indexes:
            source_axis = self.source_space.axes[index]
            reflected = _exact_source_coordinate(
                coordinates[index],
                _REFLECTION_SCALE,
                _REFLECTION_OFFSET,
                source_axis,
            )
            _validate_coordinate_value(reflected, source_axis)
            coordinates[index] = reflected
        return tuple(coordinates)

    def invert(self) -> AxisOrientationTransform:
        # Negation is an involution, so the inverse reflects the same axes; only
        # the two spaces swap.
        return AxisOrientationTransform(
            self.source_space,
            self.target_space,
            self.axis_indexes,
        )


@dataclass(frozen=True, slots=True)
class LabelReindexTransform:
    """An explicit total bijection between two committed label domains."""

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    axis_index: int
    label_map: tuple[tuple[str, str], ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "LABEL_REINDEX"

    # The correspondence is total and bijective, so no coordinate is lost, but
    # the label alphabet itself changes: this is a representation change, not an
    # identity. Callers that require byte-identical labels must check the class.
    loss_class: ClassVar[LossClass] = LossClass.EXACT_WITH_REPRESENTATION_CHANGE
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "LABEL_REINDEX":
            raise ValueError("label reindex transform_type must be LABEL_REINDEX")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if type(self.axis_index) is not int:
            raise TypeError("axis_index must be an integer")
        if not 0 <= self.axis_index < len(self.target_space.axes):
            raise ValueError("axis_index is outside the target coordinate space")
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        selected_target_axis = self.target_space.axes[self.axis_index]
        selected_source_axis = self.source_space.axes[self.axis_index]
        if (
            selected_target_axis.scalar_type is not ScalarType.LABEL
            or selected_source_axis.scalar_type is not ScalarType.LABEL
        ):
            raise ValueError(
                "UNSUPPORTED_VALUE_SEMANTICS: label reindex requires LABEL axes"
            )
        if selected_target_axis.labels_ref == selected_source_axis.labels_ref:
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: identical label commitments must use "
                "IdentityTransform"
            )
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes)
        ):
            include_labels = index != self.axis_index
            if target_axis.exact_semantics_key(
                include_labels=include_labels
            ) != source_axis.exact_semantics_key(include_labels=include_labels):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: label reindex may change only the "
                    "committed label dictionary on its selected axis"
                )
            _require_axis_policy(
                "LABEL_REINDEX" if index == self.axis_index else "IDENTITY",
                target_axis,
                source_axis,
            )
        if isinstance(self.label_map, (str, bytes)):
            raise TypeError("label_map must be an iterable, not text")
        try:
            raw_map = iter(self.label_map)
        except TypeError as exc:
            raise TypeError("label_map must be iterable") from exc
        normalized: list[tuple[str, str]] = []
        encoded_bytes = 0
        for index, raw_item in enumerate(raw_map):
            if index >= MAX_LABEL_REINDEX_ITEMS:
                raise ValueError(
                    f"label_map exceeds the {MAX_LABEL_REINDEX_ITEMS}-item limit"
                )
            if type(raw_item) is not tuple or len(raw_item) != 2:
                raise TypeError("label_map entries must be exact two-item tuples")
            target_label = _text(
                raw_item[0],
                "target label",
                identifier=False,
            )
            source_label = _text(
                raw_item[1],
                "source label",
                identifier=False,
            )
            encoded_bytes += (
                16
                + len(target_label.encode("utf-8"))
                + len(source_label.encode("utf-8"))
            )
            if encoded_bytes > MAX_LABEL_REINDEX_BYTES:
                raise ValueError(
                    "label_map exceeds the "
                    f"{MAX_LABEL_REINDEX_BYTES}-byte encoding limit"
                )
            normalized.append((target_label, source_label))
        if not normalized:
            raise ValueError("label_map must contain at least one correspondence")
        targets = [target for target, _ in normalized]
        sources = [source for _, source in normalized]
        if len(set(targets)) != len(targets):
            raise ValueError("label_map target labels must be unique")
        if len(set(sources)) != len(sources):
            raise ValueError("label_map must be bijective over source labels")
        if set(targets) != set(selected_target_axis.labels or ()):
            raise ValueError("label_map must cover the complete target label domain")
        if set(sources) != set(selected_source_axis.labels or ()):
            raise ValueError("label_map must cover the complete source label domain")
        object.__setattr__(self, "label_map", tuple(sorted(normalized)))

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = list(_coordinates(target_coordinates, self.target_space))
        target_label = coordinates[self.axis_index]
        if not isinstance(target_label, str):
            raise TypeError("label coordinates must be strings")
        map_index = bisect_left(self.label_map, (target_label, ""))
        if (
            map_index == len(self.label_map)
            or self.label_map[map_index][0] != target_label
        ):
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: target label "
                f"{target_label!r} has no explicit source correspondence"
            )
        coordinates[self.axis_index] = self.label_map[map_index][1]
        return tuple(coordinates)

    def invert(self) -> LabelReindexTransform:
        return LabelReindexTransform(
            self.source_space,
            self.target_space,
            self.axis_index,
            tuple((source, target) for target, source in self.label_map),
        )


def _decimal_text(value: Fraction) -> str | None:
    """Return the one canonical decimal spelling of an exact value, or None.

    None means the value is not a terminating decimal, so no DECIMAL coordinate
    denotes it and the conversion has to be refused rather than rounded.
    """

    denominator = value.denominator
    twos = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    fives = 0
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return None
    places = max(twos, fives)
    # The rational grammar bounds each part to 256 digits, so `places` is already
    # bounded; this caps the widening multiplication below at a spelling the
    # coordinate validator could accept anyway.
    if places > MAX_DECIMAL_LENGTH:
        return None
    scaled = value.numerator * 2 ** (places - twos) * 5 ** (places - fives)
    # Padding to places + 1 digits guarantees a whole part exists, so the result
    # never acquires a leading zero beyond the single "0" the grammar allows.
    digits = str(abs(scaled)).rjust(places + 1, "0")
    whole, fraction = (
        (digits, "") if places == 0 else (digits[:-places], digits[-places:])
    )
    fraction = fraction.rstrip("0")
    sign = "-" if scaled < 0 else ""
    if not fraction:
        return sign + whole
    return f"{sign}{whole}.{fraction}"


def _exact_source_coordinate(
    value: object,
    scale: Fraction,
    offset: Fraction,
    axis: CoordinateAxis,
) -> object:
    """Evaluate scale * value + offset exactly in the axis's own number line."""

    if axis.scalar_type is ScalarType.INTEGER:
        exact = scale * value + offset  # type: ignore[operator]
        if exact.denominator != 1:
            raise ValueError(
                f"UNSUPPORTED_VALUE_SEMANTICS: axis {axis.axis_id!r} is INTEGER "
                f"and the exact converted value {exact} is not an integer"
            )
        return int(exact)
    if axis.scalar_type is ScalarType.RATIONAL:
        return str(scale * Fraction(value) + offset)  # type: ignore[arg-type]
    exact = scale * Fraction(Decimal(value)) + offset  # type: ignore[arg-type]
    text = _decimal_text(exact)
    if text is None:
        raise ValueError(
            f"UNSUPPORTED_VALUE_SEMANTICS: axis {axis.axis_id!r} is DECIMAL and "
            f"the exact converted value {exact} has no terminating decimal "
            "spelling"
        )
    return text


@dataclass(frozen=True, slots=True)
class AxisUnitConversion:
    """One declared exact scale-and-offset conversion for a single axis."""

    axis_index: int
    scale: str
    offset: str = "0"

    def __post_init__(self) -> None:
        if type(self.axis_index) is not int or isinstance(self.axis_index, bool):
            raise TypeError("axis_index must be an integer")
        if not 0 <= self.axis_index < MAX_AXES:
            raise ValueError("axis_index is outside the supported axis range")
        scale = _rational_value(self.scale, "scale")
        _rational_value(self.offset, "offset")
        # §8.6: reversing an axis is rung 3, and a two-parameter affine is not
        # the least-expressive way to say it. A negative scale is a reflection
        # composed with a rescale, so it is spelled as AxisOrientationTransform
        # followed by a positive UnitAffineTransform — two records that each say
        # one thing, rather than one that hides a direction change inside a
        # number. A zero scale collapses the axis and is not exact at all.
        if scale <= 0:
            raise ValueError(
                "scale must be greater than zero; a reversing conversion is an "
                "orientation correction (AxisOrientationTransform), optionally "
                "composed with a positive unit conversion"
            )

    @property
    def exact_scale(self) -> Fraction:
        return Fraction(self.scale)

    @property
    def exact_offset(self) -> Fraction:
        return Fraction(self.offset)


@dataclass(frozen=True, slots=True)
class UnitAffineTransform:
    """An exact declared unit conversion on one or more numeric axes.

    The conversion is *declared*, never inferred. Nothing here parses a unit
    string, consults a unit registry or derives a factor from magnitudes, so the
    record commits to the caller's declaration exactly as given. Replay detects
    revision of that declaration; it does not establish that "centimetre" really
    is a hundredth of "metre".
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    conversions: tuple[AxisUnitConversion, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "UNIT_AFFINE"

    # Every coordinate survives, and the map inverts exactly because the scale is
    # non-zero — but the unit the numbers are quoted in changes, so this is a
    # representation change rather than an identity.
    loss_class: ClassVar[LossClass] = LossClass.EXACT_WITH_REPRESENTATION_CHANGE
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "UNIT_AFFINE":
            raise ValueError("unit affine transform_type must be UNIT_AFFINE")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        conversions = _bounded_tuple(self.conversions, "conversions", MAX_AXES)
        if any(
            type(conversion) is not AxisUnitConversion for conversion in conversions
        ):
            raise TypeError("conversions must contain only AxisUnitConversion values")
        if not conversions:
            raise ValueError(
                "a unit conversion must convert at least one axis; equal units "
                "must use IdentityTransform"
            )
        converted = [conversion.axis_index for conversion in conversions]
        if len(set(converted)) != len(converted):
            raise ValueError("conversions must name each axis at most once")
        axis_count = len(self.target_space.axes)
        if any(index >= axis_count for index in converted):
            raise ValueError("a conversion names an axis outside the target space")
        converted_indexes = frozenset(converted)
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes, strict=True)
        ):
            include_unit = index not in converted_indexes
            if target_axis.exact_semantics_key(
                include_unit=include_unit
            ) != source_axis.exact_semantics_key(include_unit=include_unit):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: a unit conversion may change only "
                    "the declared unit on the axes it names"
                )
            if include_unit:
                _require_axis_policy("IDENTITY", target_axis, source_axis)
                continue
            if target_axis.scalar_type not in EXACT_NUMERIC_SCALAR_TYPES:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: unit conversion requires an "
                    f"INTEGER, RATIONAL or DECIMAL axis, not "
                    f"{target_axis.scalar_type.value}"
                )
            if target_axis.unit is None or source_axis.unit is None:
                raise ValueError(
                    "INSUFFICIENT_METADATA: a converted axis must declare a unit "
                    "on both sides"
                )
            if target_axis.unit == source_axis.unit:
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: axis "
                    f"{target_axis.axis_id!r} declares the same unit on both "
                    "sides, so no unit conversion applies"
                )
            _require_axis_policy("UNIT_AFFINE", target_axis, source_axis)
        object.__setattr__(
            self,
            "conversions",
            tuple(sorted(conversions, key=lambda item: item.axis_index)),
        )

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = list(_coordinates(target_coordinates, self.target_space))
        for conversion in self.conversions:
            index = conversion.axis_index
            source_axis = self.source_space.axes[index]
            converted = _exact_source_coordinate(
                coordinates[index],
                conversion.exact_scale,
                conversion.exact_offset,
                source_axis,
            )
            # The exact value can still fall outside what the source axis will
            # accept as a coordinate — a rational whose parts outgrow the bounded
            # grammar, or a decimal past the length limit. Validating here turns
            # that into a named refusal instead of a coordinate that the source
            # space would never have admitted.
            _validate_coordinate_value(converted, source_axis)
            coordinates[index] = converted
        return tuple(coordinates)

    def invert(self) -> UnitAffineTransform:
        return UnitAffineTransform(
            self.source_space,
            self.target_space,
            tuple(
                AxisUnitConversion(
                    conversion.axis_index,
                    str(1 / conversion.exact_scale),
                    str(-conversion.exact_offset / conversion.exact_scale),
                )
                for conversion in self.conversions
            ),
        )


@dataclass(frozen=True, slots=True)
class AxisReferenceShift:
    """One declared exact origin shift for a single axis, in the target's unit."""

    axis_index: int
    offset: str

    def __post_init__(self) -> None:
        if type(self.axis_index) is not int or isinstance(self.axis_index, bool):
            raise TypeError("axis_index must be an integer")
        if not 0 <= self.axis_index < MAX_AXES:
            raise ValueError("axis_index is outside the supported axis range")
        offset = _rational_value(self.offset, "offset")
        # A zero shift means the two frames put their origin in the same place,
        # so no coordinate changes and the only difference is the frame's name.
        # That is a metadata change, and this slice does not carry one — the
        # same boundary `AxisTimezoneOffset` draws for two zone names at one
        # offset.
        if offset == 0:
            raise ValueError(
                "offset must not be zero; two frames sharing an origin differ "
                "only in name, which is a metadata change rather than a "
                "coordinate transform"
            )

    @property
    def exact_offset(self) -> Fraction:
        return Fraction(self.offset)


@dataclass(frozen=True, slots=True)
class ReferenceBasisTransform:
    """Re-express a coordinate against a different declared reference frame.

    The map is `source = target + offset`: a translation, and nothing else.
    §8.6 asks for the least-expressive sufficient transform, and "measured from
    a different origin" has no scaling in it. A frame change that also rescales
    is a *unit* change as well, and the two compose — one record per thing said,
    which is also what keeps the offset unambiguous, because it is declared in
    the target axis's own unit and the shift happens while the coordinate is
    still expressed in it.

    Declared, like every family whose vocabulary is open. A reference-frame
    identifier is free text — "mean-sea-level", "drill-floor", "EPSG:4326" —
    and nothing here can compute how far apart two origins are. Ranked against
    the others by how much the data can push back, this sits with the unit rung:
    nothing in the coordinates can contradict a declared origin offset.
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    shifts: tuple[AxisReferenceShift, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "REFERENCE_BASIS"

    # Every coordinate survives and a translation inverts exactly, but the datum
    # the numbers are quoted against changed, so the numbers changed with it.
    loss_class: ClassVar[LossClass] = LossClass.EXACT_WITH_REPRESENTATION_CHANGE
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "REFERENCE_BASIS":
            raise ValueError("reference transform_type must be REFERENCE_BASIS")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        shifts = _bounded_tuple(self.shifts, "shifts", MAX_AXES)
        if any(type(shift) is not AxisReferenceShift for shift in shifts):
            raise TypeError("shifts must contain only AxisReferenceShift values")
        if not shifts:
            raise ValueError(
                "a reference conversion must shift at least one axis; equal "
                "frames must use IdentityTransform"
            )
        shifted = [shift.axis_index for shift in shifts]
        if len(set(shifted)) != len(shifted):
            raise ValueError("shifts must name each axis at most once")
        axis_count = len(self.target_space.axes)
        if any(index >= axis_count for index in shifted):
            raise ValueError("a shift names an axis outside the target space")
        shifted_indexes = frozenset(shifted)
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes, strict=True)
        ):
            include_reference_frame = index not in shifted_indexes
            if target_axis.exact_semantics_key(
                include_reference_frame=include_reference_frame
            ) != source_axis.exact_semantics_key(
                include_reference_frame=include_reference_frame
            ):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: a reference conversion may change "
                    "only the declared reference_frame on the axes it names"
                )
            if include_reference_frame:
                _require_axis_policy("IDENTITY", target_axis, source_axis)
                continue
            if target_axis.scalar_type not in EXACT_NUMERIC_SCALAR_TYPES:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: reference conversion requires "
                    "an INTEGER, RATIONAL or DECIMAL axis, not "
                    f"{target_axis.scalar_type.value}"
                )
            if (
                target_axis.reference_frame is None
                or source_axis.reference_frame is None
            ):
                raise ValueError(
                    "INSUFFICIENT_METADATA: a shifted axis must declare a "
                    "reference_frame on both sides"
                )
            if target_axis.reference_frame == source_axis.reference_frame:
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: axis "
                    f"{target_axis.axis_id!r} declares the same reference_frame "
                    "on both sides, so no reference conversion applies"
                )
            _require_axis_policy("REFERENCE_BASIS", target_axis, source_axis)
        object.__setattr__(
            self,
            "shifts",
            tuple(sorted(shifts, key=lambda item: item.axis_index)),
        )

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = list(_coordinates(target_coordinates, self.target_space))
        for shift in self.shifts:
            index = shift.axis_index
            source_axis = self.source_space.axes[index]
            shifted = _exact_source_coordinate(
                coordinates[index],
                _TRANSLATION_SCALE,
                shift.exact_offset,
                source_axis,
            )
            _validate_coordinate_value(shifted, source_axis)
            coordinates[index] = shifted
        return tuple(coordinates)

    def invert(self) -> ReferenceBasisTransform:
        return ReferenceBasisTransform(
            self.source_space,
            self.target_space,
            tuple(
                AxisReferenceShift(shift.axis_index, str(-shift.exact_offset))
                for shift in self.shifts
            ),
        )


def _reexpressed_instant(
    value: object,
    target_offset: int,
    source_offset: int,
    axis: CoordinateAxis,
) -> str:
    """Re-spell one instant at a different declared UTC offset."""

    parsed = datetime.fromisoformat(value)  # type: ignore[arg-type]
    declared = timedelta(minutes=target_offset)
    # The coordinate carries its own offset, so it can contradict the transform.
    # This is the check the unit rung has no counterpart for: there, nothing in
    # the data can disagree with a declared factor. Here it can, and does exactly
    # when the zone's offset moved.
    if parsed.utcoffset() != declared:
        raise ValueError(
            f"INCOMPATIBLE_SEMANTICS: axis {axis.axis_id!r} coordinate declares "
            f"offset {parsed.utcoffset()}, which contradicts the transform's "
            f"declared target offset {declared}. An offset that moves across the "
            "registered interval — a daylight-saving change — needs a "
            "per-instant offset that this slice does not carry"
        )
    try:
        converted = parsed.astimezone(fixed_timezone(timedelta(minutes=source_offset)))
    except (OverflowError, OSError, ValueError) as exc:
        # Re-spelling an instant near the representable boundary can leave the
        # domain entirely. Bounded refusal, not a wrapped date.
        raise ValueError(
            f"axis {axis.axis_id!r} instant leaves the representable timestamp "
            f"domain when re-spelled at offset {source_offset} minutes"
        ) from exc
    return converted.isoformat()


@dataclass(frozen=True, slots=True)
class AxisTimezoneOffset:
    """The declared fixed UTC offsets of one axis on each side of a conversion."""

    axis_index: int
    target_offset_minutes: int
    source_offset_minutes: int

    def __post_init__(self) -> None:
        if type(self.axis_index) is not int or isinstance(self.axis_index, bool):
            raise TypeError("axis_index must be an integer")
        if not 0 <= self.axis_index < MAX_AXES:
            raise ValueError("axis_index is outside the supported axis range")
        for name in ("target_offset_minutes", "source_offset_minutes"):
            offset = getattr(self, name)
            if type(offset) is not int or isinstance(offset, bool):
                raise TypeError(f"{name} must be an integer number of minutes")
            # datetime.timezone's own domain. Narrowing it further to the offsets
            # the IANA database currently uses would be an invented policy that a
            # future release of that database could falsify.
            if not -MAX_UTC_OFFSET_MINUTES < offset < MAX_UTC_OFFSET_MINUTES:
                raise ValueError(
                    f"{name} must be strictly within ±{MAX_UTC_OFFSET_MINUTES} "
                    "minutes of UTC"
                )
        if self.target_offset_minutes == self.source_offset_minutes:
            raise ValueError(
                "equal offsets re-spell nothing; a differently named zone at the "
                "same offset is a metadata change, not a coordinate transform"
            )


@dataclass(frozen=True, slots=True)
class TimezoneTransform:
    """Re-spell the same instant under a different declared UTC offset.

    The instant is not moved: the target coordinate already carries its own
    numeric offset, so it names an absolute instant, and this transform writes
    that same instant with the source axis's offset. No IANA timezone database is
    read. The offsets are *declared*, exactly as a unit conversion's factor is,
    because resolving ``America/New_York`` to ``-04:00`` on a given date is a
    lookup in external, versioned data that a verifier on another machine may not
    have the same edition of.

    What the data can still catch: a coordinate whose own offset contradicts the
    declared one is refused, which is precisely what a daylight-saving change
    looks like from in here.
    """

    target_space: CoordinateSpace
    source_space: CoordinateSpace
    offsets: tuple[AxisTimezoneOffset, ...]
    direction: TransformDirection = TransformDirection.TARGET_TO_SOURCE
    transform_type: str = "TIMEZONE"

    # The instant is preserved exactly and the map inverts exactly, but the
    # spelling — and the zone the reader will attribute it to — changes.
    loss_class: ClassVar[LossClass] = LossClass.EXACT_WITH_REPRESENTATION_CHANGE
    invertibility: ClassVar[Invertibility] = Invertibility.EXACT

    def __post_init__(self) -> None:
        if self.direction is not TransformDirection.TARGET_TO_SOURCE:
            raise ValueError("Lattice stores only TARGET_TO_SOURCE transforms")
        if self.transform_type != "TIMEZONE":
            raise ValueError("timezone transform_type must be TIMEZONE")
        _require_space_values(self.target_space, self.source_space)
        _require_registration_metadata(self.target_space, self.source_space)
        if self.target_space.exact_space_key() != self.source_space.exact_space_key():
            raise ValueError(
                "INCOMPATIBLE_SEMANTICS: space-wide coordinate metadata differs"
            )
        offsets = _bounded_tuple(self.offsets, "offsets", MAX_AXES)
        if any(type(offset) is not AxisTimezoneOffset for offset in offsets):
            raise TypeError("offsets must contain only AxisTimezoneOffset values")
        if not offsets:
            raise ValueError(
                "a timezone conversion must convert at least one axis; equal "
                "zones must use IdentityTransform"
            )
        converted = [offset.axis_index for offset in offsets]
        if len(set(converted)) != len(converted):
            raise ValueError("offsets must name each axis at most once")
        axis_count = len(self.target_space.axes)
        if any(index >= axis_count for index in converted):
            raise ValueError("an offset names an axis outside the target space")
        converted_indexes = frozenset(converted)
        for index, (target_axis, source_axis) in enumerate(
            zip(self.target_space.axes, self.source_space.axes, strict=True)
        ):
            include_timezone = index not in converted_indexes
            if target_axis.exact_semantics_key(
                include_timezone=include_timezone
            ) != source_axis.exact_semantics_key(include_timezone=include_timezone):
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: a timezone conversion may change "
                    "only the declared timezone on the axes it names"
                )
            if include_timezone:
                _require_axis_policy("IDENTITY", target_axis, source_axis)
                continue
            if target_axis.scalar_type is not ScalarType.TIMESTAMP:
                raise ValueError(
                    "UNSUPPORTED_VALUE_SEMANTICS: timezone conversion requires a "
                    f"TIMESTAMP axis, not {target_axis.scalar_type.value}"
                )
            if target_axis.timezone is None or source_axis.timezone is None:
                raise ValueError(
                    "INSUFFICIENT_METADATA: a converted axis must declare a "
                    "timezone on both sides"
                )
            if target_axis.timezone == source_axis.timezone:
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: axis "
                    f"{target_axis.axis_id!r} declares the same timezone on both "
                    "sides, so no timezone conversion applies"
                )
            _require_axis_policy("TIMEZONE", target_axis, source_axis)
        object.__setattr__(
            self,
            "offsets",
            tuple(sorted(offsets, key=lambda item: item.axis_index)),
        )

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = list(_coordinates(target_coordinates, self.target_space))
        for offset in self.offsets:
            index = offset.axis_index
            source_axis = self.source_space.axes[index]
            converted = _reexpressed_instant(
                coordinates[index],
                offset.target_offset_minutes,
                offset.source_offset_minutes,
                source_axis,
            )
            _validate_coordinate_value(converted, source_axis)
            coordinates[index] = converted
        return tuple(coordinates)

    def invert(self) -> TimezoneTransform:
        return TimezoneTransform(
            self.source_space,
            self.target_space,
            tuple(
                AxisTimezoneOffset(
                    offset.axis_index,
                    offset.source_offset_minutes,
                    offset.target_offset_minutes,
                )
                for offset in self.offsets
            ),
        )


ExactTransform: TypeAlias = (
    IdentityTransform
    | AxisPermutationTransform
    | AxisOrderingTransform
    | AxisOrientationTransform
    | CalendarTransform
    | LabelReindexTransform
    | ReferenceBasisTransform
    | UnitAffineTransform
    | TimezoneTransform
)


@dataclass(frozen=True, slots=True)
class TransformChain:
    """A typed, auditable composition in target-to-source execution order."""

    transforms: tuple[ExactTransform, ...]

    def __post_init__(self) -> None:
        transforms = _bounded_tuple(
            self.transforms,
            "transforms",
            MAX_TRANSFORM_CHAIN_DEPTH,
        )
        if not transforms:
            raise ValueError("transform chain must contain at least one transform")
        supported_types = {
            IdentityTransform,
            AxisPermutationTransform,
            AxisOrderingTransform,
            AxisOrientationTransform,
            CalendarTransform,
            LabelReindexTransform,
            ReferenceBasisTransform,
            UnitAffineTransform,
            TimezoneTransform,
        }
        if any(type(transform) not in supported_types for transform in transforms):
            raise TypeError("transform chain contains an unsupported transform type")
        for left, right in zip(transforms, transforms[1:]):
            if left.source_space != right.target_space:
                raise ValueError(
                    "INCOMPATIBLE_SEMANTICS: adjacent transform spaces do not match"
                )
        object.__setattr__(self, "transforms", transforms)

    @property
    def target_space(self) -> CoordinateSpace:
        return self.transforms[0].target_space

    @property
    def source_space(self) -> CoordinateSpace:
        return self.transforms[-1].source_space

    @property
    def loss_class(self) -> LossClass:
        """The weakest loss class any member declares."""

        return max(
            (transform.loss_class for transform in self.transforms),
            key=LOSS_CLASS_RANK.__getitem__,
        )

    @property
    def invertibility(self) -> Invertibility:
        """A chain inverts exactly only when every member does."""

        if all(
            transform.invertibility is Invertibility.EXACT
            for transform in self.transforms
        ):
            return Invertibility.EXACT
        return Invertibility.UNKNOWN

    def apply_coordinates(
        self, target_coordinates: tuple[object, ...]
    ) -> tuple[object, ...]:
        coordinates = target_coordinates
        for transform in self.transforms:
            coordinates = transform.apply_coordinates(coordinates)
        return coordinates

    def invert(self) -> TransformChain:
        return TransformChain(
            tuple(transform.invert() for transform in reversed(self.transforms))
        )
