"""Deterministic compatibility analysis for exact Lattice transforms."""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
from types import MappingProxyType

from alelyon.runtime.vector.lattice.canonical import coordinate_space_ref

from alelyon.runtime.vector.lattice.contracts import (
    CoordinateSpace,
    INITIAL_EXACT_TOPOLOGIES,
    _bounded_tuple,
    _rational_value,
    _text,
)
from alelyon.runtime.vector.lattice.transforms import (
    MAX_LABEL_REINDEX_ITEMS,
    AxisOrientationTransform,
    AxisPermutationTransform,
    AxisTimezoneOffset,
    AxisUnitConversion,
    IdentityTransform,
    LabelReindexTransform,
    ProhibitedTransformError,
    TimezoneTransform,
    TransformChain,
    UnitAffineTransform,
)


MAX_COMPATIBILITY_EVIDENCE_ITEMS = 128
MAX_DECLARED_CONVERSIONS = 1024

# Uniqueness checking re-runs the augmenting search once per matched edge, so
# the number of edge visits is cubic in the axis count even though every input
# is inside its declared limit. The budget converts that from an unbounded grind
# into a deterministic, named refusal. It is a fixed constant so the outcome
# never depends on machine speed or load.
MAX_MATCHING_EDGE_VISITS = 1 << 20


class CompatibilityCode(str, Enum):
    EXACT_IDENTITY = "EXACT_IDENTITY"
    EXACT_AXIS_PERMUTATION = "EXACT_AXIS_PERMUTATION"
    EXACT_AXIS_ORIENTATION = "EXACT_AXIS_ORIENTATION"
    EXACT_UNIT_AFFINE = "EXACT_UNIT_AFFINE"
    EXACT_TIMEZONE = "EXACT_TIMEZONE"
    EXACT_LABEL_REINDEX = "EXACT_LABEL_REINDEX"
    EXACT_COMPOSED_CHAIN = "EXACT_COMPOSED_CHAIN"
    AMBIGUOUS_MAPPING = "AMBIGUOUS_MAPPING"
    UNSUPPORTED_TOPOLOGY = "UNSUPPORTED_TOPOLOGY"
    UNSUPPORTED_VALUE_SEMANTICS = "UNSUPPORTED_VALUE_SEMANTICS"
    UNSUPPORTED_TRANSFORM_COMPOSITION = "UNSUPPORTED_TRANSFORM_COMPOSITION"
    INCOMPATIBLE_TOPOLOGY = "INCOMPATIBLE_TOPOLOGY"
    INCOMPATIBLE_SEMANTICS = "INCOMPATIBLE_SEMANTICS"
    INSUFFICIENT_METADATA = "INSUFFICIENT_METADATA"
    UNDECLARED_UNIT_CONVERSION = "UNDECLARED_UNIT_CONVERSION"
    UNDECLARED_TIMEZONE_CONVERSION = "UNDECLARED_TIMEZONE_CONVERSION"
    UNDECLARED_LABEL_REINDEX = "UNDECLARED_LABEL_REINDEX"
    UNDECLARED_ORIENTATION_FLIP = "UNDECLARED_ORIENTATION_FLIP"
    PROHIBITED_TRANSFORM = "PROHIBITED_TRANSFORM"
    RESOURCE_BUDGET_EXCEEDED = "RESOURCE_BUDGET_EXCEEDED"


_SUCCESS_CODES = frozenset(
    {
        CompatibilityCode.EXACT_IDENTITY,
        CompatibilityCode.EXACT_AXIS_PERMUTATION,
        CompatibilityCode.EXACT_AXIS_ORIENTATION,
        CompatibilityCode.EXACT_UNIT_AFFINE,
        CompatibilityCode.EXACT_TIMEZONE,
        CompatibilityCode.EXACT_LABEL_REINDEX,
        CompatibilityCode.EXACT_COMPOSED_CHAIN,
    }
)


@dataclass(frozen=True, slots=True)
class DeclaredOrientationFlip:
    """One caller-declared assertion that two orientation codes are opposites.

    Alone among these declarations it carries no parameter, because an
    orientation correction has none: the map is negation. Its whole content is
    the claim, and its absence is the refusal. Two different orientation codes
    are not evidence of a flip — "RAS" and "LPI" differ, but they differ by
    three reflections, and "RAS" and "LAS" by one. The vocabulary is free text
    from some domain, open and without an algebra, so nothing here can tell
    those cases apart and a caller has to say which it is.

    Nothing in the data can contradict the claim either. A timezone declaration
    is checked against the coordinate's own offset; a label map against both
    committed dictionaries; this one against nothing, because with `bounds`,
    `origin` and `resolution` outside this slice the coordinates carry no
    evidence of which way they run. What is checked is that both sides declare
    an orientation, that the two differ, and that both axis policies admit the
    family.
    """

    semantic_id: str
    target_orientation: str
    source_orientation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_id",
            _text(self.semantic_id, "semantic_id", identifier=True, maximum=1024),
        )
        for name in ("target_orientation", "source_orientation"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, identifier=False)
            )
        if self.target_orientation == self.source_orientation:
            raise ValueError(
                "a declared orientation flip must name two different orientations"
            )

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.semantic_id, self.target_orientation, self.source_orientation)


@dataclass(frozen=True, slots=True)
class DeclaredUnitConversion:
    """One caller-declared exact target-to-source unit conversion.

    Lattice never infers a conversion factor. It does not parse ``metre`` or
    ``percent``, hold a unit registry, or read magnitudes: a unit difference is
    an obstruction until a caller declares what the conversion is, and the
    declaration is what gets committed. Read as ``source = scale * target +
    offset``, matching the stored target-to-source direction.
    """

    semantic_id: str
    target_unit: str
    source_unit: str
    scale: str
    offset: str = "0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_id",
            _text(self.semantic_id, "semantic_id", identifier=True, maximum=1024),
        )
        for name in ("target_unit", "source_unit"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, identifier=False)
            )
        if self.target_unit == self.source_unit:
            raise ValueError(
                "a declared unit conversion must name two different units"
            )
        object.__setattr__(
            self, "scale", _text(self.scale, "scale", identifier=True, maximum=513)
        )
        object.__setattr__(
            self, "offset", _text(self.offset, "offset", identifier=True, maximum=513)
        )
        if _rational_value(self.scale, "scale") <= 0:
            raise ValueError("scale must be greater than zero")
        _rational_value(self.offset, "offset")

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.semantic_id, self.target_unit, self.source_unit)


@dataclass(frozen=True, slots=True)
class DeclaredTimezoneConversion:
    """One caller-declared pair of fixed UTC offsets for a timezone conversion.

    Lattice reads no IANA timezone database. Resolving ``America/New_York`` to
    ``-04:00`` on a given date is a lookup in external, versioned data, and a
    verifier on another machine may hold a different edition of it — so the
    offsets are declared, the declaration is committed, and the coordinate's own
    offset is checked against it.
    """

    semantic_id: str
    target_timezone: str
    source_timezone: str
    target_offset_minutes: int
    source_offset_minutes: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_id",
            _text(self.semantic_id, "semantic_id", identifier=True, maximum=1024),
        )
        for name in ("target_timezone", "source_timezone"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, identifier=True)
            )
        if self.target_timezone == self.source_timezone:
            raise ValueError(
                "a declared timezone conversion must name two different zones"
            )
        # Validated by the same record the transform will carry, so a declaration
        # cannot describe an offset pair the transform would refuse.
        AxisTimezoneOffset(
            0, self.target_offset_minutes, self.source_offset_minutes
        )

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.semantic_id, self.target_timezone, self.source_timezone)


@dataclass(frozen=True, slots=True)
class DeclaredLabelReindex:
    """One caller-declared total bijection between two committed label domains.

    Lattice does not guess that ``AAPL`` and ``BBG000B9XRY4`` are the same
    security. Resolving an alias is Nexus's job and its answer carries a
    confidence; a registration carries a correspondence that is exact or absent.
    So the map is declared, committed, and checked for bijectivity against both
    axes' committed label dictionaries when the transform is built.
    """

    semantic_id: str
    target_labels_ref: str
    source_labels_ref: str
    label_map: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "semantic_id",
            _text(self.semantic_id, "semantic_id", identifier=True, maximum=1024),
        )
        for name in ("target_labels_ref", "source_labels_ref"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), name, identifier=True, maximum=71),
            )
        if self.target_labels_ref == self.source_labels_ref:
            raise ValueError(
                "a declared label reindex must name two different label "
                "commitments; identical ones are an identity"
            )
        pairs = _bounded_tuple(
            self.label_map, "label_map", MAX_LABEL_REINDEX_ITEMS
        )
        if any(type(pair) is not tuple or len(pair) != 2 for pair in pairs):
            raise TypeError("label_map entries must be exact two-item tuples")
        if not pairs:
            raise ValueError("label_map must contain at least one correspondence")
        object.__setattr__(self, "label_map", pairs)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.semantic_id, self.target_labels_ref, self.source_labels_ref)


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """A self-consistent exact result or a structured refusal."""

    code: CompatibilityCode
    explanation: str
    transform: TransformChain | None = None
    failing_constraint: str | None = None
    evidence: tuple[str, ...] = ()
    can_retry_with_metadata_or_policy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, CompatibilityCode):
            raise TypeError("code must be a CompatibilityCode")
        object.__setattr__(
            self,
            "explanation",
            _text(self.explanation, "explanation", identifier=False),
        )
        evidence = _bounded_tuple(
            self.evidence,
            "evidence",
            MAX_COMPATIBILITY_EVIDENCE_ITEMS,
        )
        object.__setattr__(
            self,
            "evidence",
            tuple(
                _text(item, "evidence item", identifier=False)
                for item in evidence
            ),
        )
        if not isinstance(self.can_retry_with_metadata_or_policy, bool):
            raise TypeError("can_retry_with_metadata_or_policy must be a boolean")
        if self.code in _SUCCESS_CODES:
            if type(self.transform) is not TransformChain:
                raise ValueError("an exact compatibility code requires a transform")
            if self.failing_constraint is not None:
                raise ValueError("an exact compatibility result cannot name a failure")
            if self.can_retry_with_metadata_or_policy:
                raise ValueError("an exact compatibility result cannot be retryable")
            problem = chain_shape_refusal(
                self.code,
                tuple(
                    transform.transform_type
                    for transform in self.transform.transforms
                ),
            )
            if problem is not None:
                raise ValueError(problem)
            return
        if self.transform is not None:
            raise ValueError("a compatibility refusal cannot contain a transform")
        if self.failing_constraint is None:
            raise ValueError("a compatibility refusal must name failing_constraint")
        object.__setattr__(
            self,
            "failing_constraint",
            _text(
                self.failing_constraint,
                "failing_constraint",
                identifier=True,
                maximum=256,
            ),
        )

    @property
    def compatible(self) -> bool:
        return self.code in _SUCCESS_CODES


#: The chain shape each fixed-shape exact class requires, as transform-type
#: names in execution order.
#:
#: One table, not two. This fact used to be written twice — as a per-code type
#: check when a `CompatibilityReport` was built, and again as a lookup table in
#: `certificate.py` for a verifier that holds the chain but not the report. Two
#: statements of one rule can disagree, and the certificate side is the one a
#: third party relies on.
EXACT_CHAIN_SHAPES: Mapping[CompatibilityCode, tuple[str, ...]] = MappingProxyType({
    CompatibilityCode.EXACT_IDENTITY: ("IDENTITY",),
    CompatibilityCode.EXACT_AXIS_PERMUTATION: ("AXIS_PERMUTATION",),
    CompatibilityCode.EXACT_AXIS_ORIENTATION: ("AXIS_ORIENTATION",),
    CompatibilityCode.EXACT_LABEL_REINDEX: ("LABEL_REINDEX",),
    CompatibilityCode.EXACT_TIMEZONE: ("TIMEZONE",),
    CompatibilityCode.EXACT_UNIT_AFFINE: ("UNIT_AFFINE",),
})


def chain_shape_refusal(
    code: CompatibilityCode,
    shape: tuple[str, ...],
) -> str | None:
    """Why ``shape`` is not a chain ``code`` can name, or None if it is.

    The one authority on "does this declared class match this chain", used both
    where a report is built and where a certificate is verified against a
    replayed chain. A caller that holds only the chain gets the same answer as
    one that holds the report.

    Most classes fix their shape exactly. `EXACT_COMPOSED_CHAIN` does not — its
    shape depends on which stages the registration needed — so it is decided by
    a rule instead of by a table. That is not a weaker commitment: a certificate
    already binds the *exact* chain through `transform_chain_ref`, so the shape
    is pinned cryptographically either way. What this decides is only whether
    the declared class is honest about the chain, and for a variable-shape class
    a rule states that as precisely as an enumeration does.
    """

    if code is CompatibilityCode.EXACT_COMPOSED_CHAIN:
        # One stage is a single-family rung wearing the wrong class.
        if len(shape) < 2:
            return (
                "EXACT_COMPOSED_CHAIN requires at least two stages, not "
                f"{len(shape)}"
            )
        # IDENTITY is chainable but is not a composed stage: §14.5 asks
        # composition to *remove* redundant identities, so one appearing inside a
        # composed chain means the chain is not the plan it claims to be.
        outside = tuple(name for name in shape if name not in COMPOSED_STAGE_ORDER)
        if outside:
            return (
                "EXACT_COMPOSED_CHAIN admits only the composed stages "
                f"{COMPOSED_STAGE_ORDER}, not {outside}"
            )
        # An out-of-order shape is a chain whose derived intermediate spaces were
        # built against a different plan than the one it claims.
        if not _is_stage_order(shape, COMPOSED_STAGE_ORDER):
            return (
                "EXACT_COMPOSED_CHAIN requires its stages in "
                f"{COMPOSED_STAGE_ORDER} order, not {shape}"
            )
        return None
    expected = EXACT_CHAIN_SHAPES.get(code)
    if expected is None:
        return f"{code.value} is not an exact correspondence class"
    if shape != expected:
        return f"{code.value} requires the chain shape {expected}, not {shape}"
    return None


def _is_stage_order(candidate: tuple[str, ...], whole: tuple[str, ...]) -> bool:
    """Whether candidate runs the stages of whole in order.

    A family that carries one axis at a time contributes several adjacent stages
    of the same type, so a run of equal names collapses to one before the
    subsequence test. What this still refuses is a family appearing *after* one
    that must follow it, which would mean the chain's derived intermediates were
    built against a different plan than the one it claims.
    """

    collapsed: list[str] = []
    for name in candidate:
        if not collapsed or collapsed[-1] != name:
            collapsed.append(name)
    remaining = iter(whole)
    return all(name in remaining for name in collapsed)


def _refusal(
    code: CompatibilityCode,
    explanation: str,
    *,
    failing_constraint: str,
    evidence: tuple[str, ...] = (),
    can_retry: bool = False,
) -> CompatibilityReport:
    return CompatibilityReport(
        code=code,
        explanation=explanation,
        transform=None,
        failing_constraint=failing_constraint,
        evidence=evidence,
        can_retry_with_metadata_or_policy=can_retry,
    )


def _cap_evidence(items: tuple[str, ...]) -> tuple[str, ...]:
    if len(items) <= MAX_COMPATIBILITY_EVIDENCE_ITEMS:
        return items
    retained = items[: MAX_COMPATIBILITY_EVIDENCE_ITEMS - 1]
    omitted = len(items) - len(retained)
    return retained + (f"omitted:{omitted}",)


def _summarize_items(items: tuple[str, ...], *, character_budget: int = 1024) -> str:
    shown: list[str] = []
    used = 0
    for item in items:
        added = len(item) + (2 if shown else 0)
        if shown and used + added > character_budget:
            break
        shown.append(item)
        used += added
    summary = ", ".join(shown)
    if len(items) > len(shown):
        summary += f" (+{len(items) - len(shown)} more)"
    return summary


class _MatchingBudgetExceeded(Exception):
    """The bounded axis-matching search reached its fixed work budget."""


@dataclass(frozen=True, slots=True)
class _MatchingProblem:
    """A payload-free view of the axis-correspondence problem.

    Axis semantics are interned to small integers once, so the search compares
    class identifiers instead of re-comparing whole semantic keys — which carry
    complete label domains and metadata — at every candidate edge.
    """

    source_classes: tuple[int, ...]
    target_classes: tuple[int, ...]
    source_identity: tuple[bool, ...]
    source_permutation: tuple[bool, ...]
    target_identity: tuple[bool, ...]
    target_permutation: tuple[bool, ...]
    candidates: tuple[tuple[int, ...], ...]

    @property
    def axis_count(self) -> int:
        return len(self.source_classes)

    def allows_edge(self, source_index: int, target_index: int) -> bool:
        if source_index == target_index:
            return (
                self.source_identity[source_index]
                and self.target_identity[target_index]
            )
        return (
            self.source_permutation[source_index]
            and self.target_permutation[target_index]
        )


def _build_matching_problem(
    source_space: CoordinateSpace,
    target_space: CoordinateSpace,
    relaxed: Mapping[str, bool] = MappingProxyType({}),
) -> _MatchingProblem:
    """Intern axis semantics and build the admissible-edge sets.

    ``relaxed`` excludes fields that a declared transform family is allowed to
    change, so the same matcher decides the correspondence for a composed chain
    as for an exact one. It never widens which *edges* are admissible: a caller
    that relaxes a field has already checked that both axis policies admit the
    family that changes it.
    """

    classes: dict[object, int] = {}

    def class_of(axis) -> int:
        key = axis.exact_semantics_key(**relaxed)
        identifier = classes.get(key)
        if identifier is None:
            identifier = len(classes)
            classes[key] = identifier
        return identifier

    source_classes = tuple(class_of(axis) for axis in source_space.axes)
    target_classes = tuple(class_of(axis) for axis in target_space.axes)
    targets_by_class: dict[int, list[int]] = {}
    for target_index, target_class in enumerate(target_classes):
        targets_by_class.setdefault(target_class, []).append(target_index)
    return _MatchingProblem(
        source_classes=source_classes,
        target_classes=target_classes,
        source_identity=tuple(
            "IDENTITY" in axis.transform_policy for axis in source_space.axes
        ),
        source_permutation=tuple(
            "AXIS_PERMUTATION" in axis.transform_policy
            for axis in source_space.axes
        ),
        target_identity=tuple(
            "IDENTITY" in axis.transform_policy for axis in target_space.axes
        ),
        target_permutation=tuple(
            "AXIS_PERMUTATION" in axis.transform_policy
            for axis in target_space.axes
        ),
        # Only same-class targets can ever match, so the search never visits an
        # edge it would immediately reject on semantics.
        candidates=tuple(
            tuple(targets_by_class.get(source_class, ()))
            for source_class in source_classes
        ),
    )


def _find_policy_matching(
    problem: _MatchingProblem,
    budget: list[int],
    *,
    forbidden_edge: tuple[int, int] | None = None,
) -> tuple[int, ...] | None:
    """Find one deterministic perfect semantic/policy matching."""

    target_to_source: dict[int, int] = {}

    def augment(source_index: int, seen_targets: set[int]) -> bool:
        for target_index in problem.candidates[source_index]:
            budget[0] -= 1
            if budget[0] < 0:
                raise _MatchingBudgetExceeded
            if forbidden_edge == (source_index, target_index):
                continue
            if not problem.allows_edge(source_index, target_index):
                continue
            if target_index in seen_targets:
                continue
            seen_targets.add(target_index)
            previous_source = target_to_source.get(target_index)
            if previous_source is None or augment(previous_source, seen_targets):
                target_to_source[target_index] = source_index
                return True
        return False

    for source_index in range(problem.axis_count):
        if not augment(source_index, set()):
            return None
    order = [-1] * problem.axis_count
    for target_index, source_index in target_to_source.items():
        order[source_index] = target_index
    if any(index < 0 for index in order):
        return None
    return tuple(order)


def _unique_policy_matching(
    problem: _MatchingProblem,
) -> tuple[tuple[int, ...] | None, bool]:
    """Return one matching and whether it is the only admissible matching."""

    budget = [MAX_MATCHING_EDGE_VISITS]
    first = _find_policy_matching(problem, budget)
    if first is None:
        return None, False
    for source_index, target_index in enumerate(first):
        alternative = _find_policy_matching(
            problem,
            budget,
            forbidden_edge=(source_index, target_index),
        )
        if alternative is not None:
            return first, False
    return first, True


def _declaration_table(
    declarations: Iterable[object],
    expected_type: type,
    field_name: str,
) -> dict[tuple[str, str, str], object]:
    declared = _bounded_tuple(
        declarations,
        field_name,
        MAX_DECLARED_CONVERSIONS,
    )
    table: dict[tuple[str, str, str], object] = {}
    for conversion in declared:
        if type(conversion) is not expected_type:
            raise TypeError(
                f"{field_name} must contain only {expected_type.__name__} values"
            )
        existing = table.get(conversion.key)
        # Two declarations for one (semantic, target, source) triple would make
        # the emitted conversion depend on iteration order. That is a caller
        # error rather than a property of the data, so it raises instead of
        # becoming a report.
        if existing is not None and existing != conversion:
            raise ValueError(
                f"{field_name} declares two different conversions for "
                f"{conversion.key!r}"
            )
        table[conversion.key] = conversion
    return table


@dataclass(frozen=True, slots=True)
class _RelaxableField:
    """One axis field a declared transform family is allowed to change alone."""

    field: str
    key_argument: str
    policy: str
    #: Whether the family's transform carries one axis or all of them at once.
    #: A per-axis family needs one stage — and so one derived intermediate space
    #: — per differing axis, because its transform record names a single axis.
    per_axis: bool = False


# In §14.2 ladder order: orientation correction is rung 3, label reindexing 4,
# unit conversion 5, temporal conversion 6. Iteration order here is what decides
# which rung a pair of spaces takes, so it is this tuple's order, not a dict's.
_RELAXABLE_FIELDS = (
    _RelaxableField("orientation", "include_orientation", "AXIS_ORIENTATION"),
    _RelaxableField("labels_ref", "include_labels", "LABEL_REINDEX", per_axis=True),
    _RelaxableField("timezone", "include_timezone", "TIMEZONE"),
    _RelaxableField("unit", "include_unit", "UNIT_AFFINE"),
)

# Read-only views for the reason canonical.py's dispatch tables are: an installed
# entry would decide which transform a composed stage becomes, with no refusal.
_COMPOSED_STAGE_ITEM: Mapping[str, object] = MappingProxyType({
    # The declaration has no parameters, so the item is the axis position and
    # the presence of `declared` is the whole of what it contributed.
    "AXIS_ORIENTATION": lambda index, declared: index,
    "LABEL_REINDEX": lambda index, declared: (index, declared.label_map),
    "TIMEZONE": lambda index, declared: AxisTimezoneOffset(
        index, declared.target_offset_minutes, declared.source_offset_minutes
    ),
    "UNIT_AFFINE": lambda index, declared: AxisUnitConversion(
        index, declared.scale, declared.offset
    ),
})


def _build_label_reindex(current, following, items):
    """A label reindex names one axis, so a stage carries exactly one item."""

    (axis_index, label_map), = items
    return LabelReindexTransform(current, following, axis_index, label_map)


_COMPOSED_STAGE_TRANSFORM: Mapping[str, object] = MappingProxyType({
    "AXIS_ORIENTATION": AxisOrientationTransform,
    "LABEL_REINDEX": _build_label_reindex,
    "TIMEZONE": TimezoneTransform,
    "UNIT_AFFINE": UnitAffineTransform,
})

# What an axis becomes once its stage has run. A label axis carries two coupled
# fields — the ordered domain and its commitment — and `CoordinateAxis` refuses a
# pair that disagrees, so both move together or neither does.
_COMPOSED_STAGE_OVERRIDE: Mapping[str, object] = MappingProxyType({
    "AXIS_ORIENTATION": lambda axis: {"orientation": axis.orientation},
    "LABEL_REINDEX": lambda axis: {
        "labels_ref": axis.labels_ref,
        "labels": axis.labels,
    },
    "TIMEZONE": lambda axis: {"timezone": axis.timezone},
    "UNIT_AFFINE": lambda axis: {"unit": axis.unit},
})

_UNDECLARED_CODES: Mapping[str, "CompatibilityCode"] = MappingProxyType({
    "AXIS_ORIENTATION": CompatibilityCode.UNDECLARED_ORIENTATION_FLIP,
    "LABEL_REINDEX": CompatibilityCode.UNDECLARED_LABEL_REINDEX,
    "TIMEZONE": CompatibilityCode.UNDECLARED_TIMEZONE_CONVERSION,
    "UNIT_AFFINE": CompatibilityCode.UNDECLARED_UNIT_CONVERSION,
})


def _single_field_differences(
    source_space: CoordinateSpace,
    target_space: CoordinateSpace,
    relaxable: _RelaxableField,
) -> tuple[int, ...] | None:
    """Return the positionally aligned axes differing only in one declared field.

    None means the two spaces are not in single-field positional correspondence,
    so a transform of that family alone cannot connect them.
    """

    differing: list[int] = []
    # strict: the caller checked the axis counts. If that guard is ever removed,
    # a silent truncation here would report a correspondence over a prefix of the
    # axes, so it has to raise instead.
    for index, (target_axis, source_axis) in enumerate(
        zip(target_space.axes, source_space.axes, strict=True)
    ):
        relaxed = {relaxable.key_argument: False}
        if target_axis.exact_semantics_key(
            **relaxed
        ) != source_axis.exact_semantics_key(**relaxed):
            return None
        if getattr(target_axis, relaxable.field) == getattr(
            source_axis, relaxable.field
        ):
            continue
        # An axis whose policy does not admit the family is not a candidate at
        # all: reporting an undeclared-conversion refusal for it would invite a
        # caller to supply a declaration that the policy would still reject.
        if (
            relaxable.policy not in target_axis.transform_policy
            or relaxable.policy not in source_axis.transform_policy
        ):
            return None
        differing.append(index)
    return tuple(differing) if differing else None


#: The stage order a composed chain executes in, target to source. Field stages
#: run while the axes are still in the target's order, so each one is indexed by
#: target position; the permutation is last and moves the whole tuple into the
#: source's order. A chain is always a subsequence of this.
COMPOSED_STAGE_ORDER = (
    "AXIS_ORIENTATION",
    "LABEL_REINDEX",
    "TIMEZONE",
    "UNIT_AFFINE",
    "AXIS_PERMUTATION",
)

#: Prefix of the space_id given to a coordinate space Lattice derived rather than
#: received. It is deliberately loud: an intermediate is a computed record that a
#: verifier re-derives, not a space anybody declared, and an audit trail must not
#: read it as one.
INTERMEDIATE_SPACE_PREFIX = "lattice.intermediate"


#: Families whose transform record names one axis, so the planner emits one
#: stage — and one derived intermediate — per differing axis. Derived from the
#: field table rather than listed, so a family that becomes per-axis cannot be
#: forgotten here.
_PER_AXIS_POLICIES: frozenset[str] = frozenset(
    field.policy for field in _RELAXABLE_FIELDS if field.per_axis
)


def intermediate_space_id(*, target_ref: str, source_ref: str, stage: str) -> str:
    """The identity a derived intermediate coordinate space must carry.

    A pure function of the two *declared* ends and the stage it follows — which
    is what makes an intermediate re-derivable by a party that holds only those
    three things, rather than a record they are asked to trust.
    """

    digest = hashlib.sha256()
    digest.update(INTERMEDIATE_SPACE_PREFIX.encode("ascii") + b"\x00")
    for part in (target_ref, source_ref, stage):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{INTERMEDIATE_SPACE_PREFIX}:{digest.hexdigest()}"


def composed_intermediate_ids(
    shape: tuple[str, ...],
    *,
    target_ref: str,
    source_ref: str,
) -> tuple[str, ...]:
    """The space_id every space *between* the stages of this shape must have.

    One entry per join, so a single-stage chain has none. This is the whole of
    what a verifier needs to rebuild the derived spaces' identities: the stage
    name each intermediate follows is its stage's transform type, plus a
    position for a per-axis family whose steps are contiguous.

    Deliberately not a second copy of the planner — the planner decides *which*
    stages a registration needs and what each one carries. This decides only
    what a stage's intermediate is called, which is the part a third party has
    to be able to reproduce.
    """

    names: list[str] = []
    run: dict[str, int] = {}
    for name in shape[:-1]:
        index = run.get(name, 0)
        run[name] = index + 1
        names.append(f"{name}:{index}" if name in _PER_AXIS_POLICIES else name)
    return tuple(
        intermediate_space_id(
            target_ref=target_ref, source_ref=source_ref, stage=stage
        )
        for stage in names
    )


def _intermediate_space(
    base: CoordinateSpace,
    overrides: Mapping[int, Mapping[str, str | None]],
    *,
    target_ref: str,
    source_ref: str,
    stage: str,
) -> CoordinateSpace:
    """Derive the coordinate space that sits between two composed transforms.

    An intermediate is a pure function of the two declared spaces and the stage
    it follows, so a verifier holding both ends re-derives it rather than being
    asked to trust it. Its space_id is a commitment over exactly those three
    inputs, and carries a prefix saying it was derived.

    Space-level metadata is *not* touched: `exact_space_key()` includes it, so an
    intermediate tagged there would no longer match either end and every
    transform in the chain would refuse it.
    """

    return replace(
        base,
        space_id=intermediate_space_id(
            target_ref=target_ref, source_ref=source_ref, stage=stage
        ),
        axes=tuple(
            replace(axis, **overrides[index]) if index in overrides else axis
            for index, axis in enumerate(base.axes)
        ),
    )


def _plan_field_steps(
    current: CoordinateSpace,
    source_space: CoordinateSpace,
    partner: list[int],
    field: _RelaxableField,
    tables: Mapping[str, Mapping[tuple[str, str, str], object]],
) -> list[tuple[dict[int, dict[str, object]], list[object]]] | CompatibilityReport:
    """Plan one family's stages, or return the refusal that stopped it.

    A whole-space family plans a single stage carrying every differing axis. A
    per-axis family plans one stage per differing axis, because its transform
    record names a single axis and cannot carry more.
    """

    steps: list[tuple[dict[int, dict[str, object]], list[object]]] = []
    overrides: dict[int, dict[str, object]] = {}
    items: list[object] = []
    missing: list[str] = []
    for target_index, axis in enumerate(current.axes):
        partner_axis = source_space.axes[partner[target_index]]
        here = getattr(axis, field.field)
        there = getattr(partner_axis, field.field)
        if here == there:
            continue
        if here is None or there is None:
            return _refusal(
                CompatibilityCode.INSUFFICIENT_METADATA,
                f"axis {axis.axis_id!r} declares a {field.field} on only one "
                "side, so no conversion between them can be exact",
                failing_constraint=f"declared_{field.field}",
                evidence=(f"axis:{axis.axis_id}",),
                can_retry=True,
            )
        key = (axis.semantic_id, here, there)
        declared = tables[field.policy].get(key)
        if declared is None:
            missing.append(f"{key[0]}:{key[1]}->{key[2]}")
            continue
        item = _COMPOSED_STAGE_ITEM[field.policy](target_index, declared)
        override = {
            target_index: _COMPOSED_STAGE_OVERRIDE[field.policy](partner_axis)
        }
        if field.per_axis:
            steps.append((override, [item]))
            continue
        overrides.update(override)
        items.append(item)
    if missing:
        return _refusal(
            _UNDECLARED_CODES[field.policy],
            f"the axes correspond except for their declared {field.field}s, "
            "and no conversion was declared for "
            + _summarize_items(tuple(missing)),
            failing_constraint=f"declared_{field.field}_conversion",
            evidence=_cap_evidence(tuple(missing)),
            can_retry=True,
        )
    if items:
        steps.append((overrides, items))
    return steps


def _composed_rung(
    source_space: CoordinateSpace,
    target_space: CoordinateSpace,
    tables: Mapping[str, Mapping[tuple[str, str, str], object]],
) -> CompatibilityReport | None:
    """Compose the admitted single-field rungs with a permutation.

    Reached only when no single rung applied. Each stage is an ordinary
    transform over an ordinary coordinate space; the only new thing is that the
    spaces between stages are derived here instead of supplied, and derived
    deterministically so that they can be re-derived rather than trusted.
    """

    relaxed = {field.key_argument: False for field in _RELAXABLE_FIELDS}
    if Counter(
        axis.exact_semantics_key(**relaxed) for axis in source_space.axes
    ) != Counter(axis.exact_semantics_key(**relaxed) for axis in target_space.axes):
        return None
    required = []
    for field in _RELAXABLE_FIELDS:
        if Counter(
            getattr(axis, field.field) for axis in source_space.axes
        ) == Counter(getattr(axis, field.field) for axis in target_space.axes):
            continue
        # Only compose a family every axis admits. Otherwise the true obstruction
        # is policy, and falling through to the ordinary refusal is honest.
        if not all(
            field.policy in axis.transform_policy
            for space in (source_space, target_space)
            for axis in space.axes
        ):
            return None
        required.append(field)
    if not required:
        return None

    try:
        order, unique = _unique_policy_matching(
            _build_matching_problem(source_space, target_space, relaxed)
        )
    except _MatchingBudgetExceeded:
        return _refusal(
            CompatibilityCode.RESOURCE_BUDGET_EXCEEDED,
            "the bounded axis-correspondence search reached its fixed work "
            f"budget of {MAX_MATCHING_EDGE_VISITS} edge visits",
            failing_constraint="matching_work_budget",
            evidence=(f"axes:{len(source_space.axes)}",),
            can_retry=True,
        )
    if order is None:
        return _refusal(
            CompatibilityCode.PROHIBITED_TRANSFORM,
            "declared transform policies prohibit every complete semantic "
            "mapping, even with every declarable field relaxed",
            failing_constraint="transform_policy",
            can_retry=True,
        )
    if not unique:
        return _refusal(
            CompatibilityCode.AMBIGUOUS_MAPPING,
            "more than one semantic mapping satisfies the declared policies "
            "once every declarable field is relaxed",
            failing_constraint="unique_axis_correspondence",
        )
    # order[j] is the target axis supplying source axis j; the field stages are
    # indexed by target position, so invert it once here.
    partner = [0] * len(order)
    for source_index, target_index in enumerate(order):
        partner[target_index] = source_index

    target_ref = coordinate_space_ref(target_space)
    source_ref = coordinate_space_ref(source_space)
    permuted = order != tuple(range(len(order)))

    # Plan every family before emitting any stage. The last field stage has to
    # land on the *declared* source space when no permutation follows it, and
    # whether one follows is only knowable once the whole plan exists. Each
    # family is planned against the target because families touch disjoint
    # fields, so applying one does not change what another sees.
    plan: list[tuple[_RelaxableField, int, Mapping[int, Mapping[str, object]], list]]
    plan = []
    for field in required:
        planned = _plan_field_steps(
            target_space, source_space, partner, field, tables
        )
        if isinstance(planned, CompatibilityReport):
            return planned
        for step_index, (overrides, items) in enumerate(planned):
            plan.append((field, step_index, overrides, items))

    stages: list[object] = []
    current = target_space
    for position, (field, step_index, overrides, items) in enumerate(plan):
        if position == len(plan) - 1 and not permuted:
            # Without a permutation this is the final stage, so it must arrive at
            # the space the caller declared. Deriving one more intermediate here
            # would leave the chain a renamed copy short of its own destination:
            # every coordinate correct, and `chain.source_space` naming a space
            # nobody asked about.
            following = source_space
        else:
            following = _intermediate_space(
                current,
                overrides,
                target_ref=target_ref,
                source_ref=source_ref,
                # A per-axis family produces several stages, so the stage name
                # has to separate them or two intermediates would share one
                # identity. A whole-space family keeps its bare name, which is
                # also what keeps already-committed chains re-derivable.
                stage=f"{field.policy}:{step_index}" if field.per_axis
                else field.policy,
            )
        try:
            stages.append(
                _COMPOSED_STAGE_TRANSFORM[field.policy](
                    current, following, tuple(items)
                )
            )
        except ProhibitedTransformError as exc:
            return _refusal(
                CompatibilityCode.PROHIBITED_TRANSFORM,
                str(exc),
                failing_constraint="transform_policy",
                evidence=(
                    f"axis:{exc.axis_id}",
                    f"transform:{exc.transform_type}",
                ),
                can_retry=True,
            )
        except (TypeError, ValueError) as exc:
            return _refusal(
                CompatibilityCode.UNSUPPORTED_VALUE_SEMANTICS,
                f"the declared {field.field} conversion cannot be applied "
                f"exactly: {exc}",
                failing_constraint=f"exact_{field.field}_conversion",
            )
        current = following
    if permuted:
        try:
            stages.append(
                AxisPermutationTransform(current, source_space, order)
            )
        except ProhibitedTransformError as exc:
            return _refusal(
                CompatibilityCode.PROHIBITED_TRANSFORM,
                str(exc),
                failing_constraint="transform_policy",
                evidence=(f"axis:{exc.axis_id}", f"transform:{exc.transform_type}"),
                can_retry=True,
            )
    if len(stages) < 2:
        # Every single-stage shape belongs to a rung that already ran, so
        # arriving here means this function's reasoning disagrees with theirs.
        # Refusing is the safe answer; silently emitting a one-stage "composed"
        # chain would let a caller certify it under the wrong class.
        return _refusal(
            CompatibilityCode.UNSUPPORTED_TRANSFORM_COMPOSITION,
            "composition resolved to fewer than two stages, which a "
            "single-family rung should already have returned",
            failing_constraint="composition_stage_count",
            evidence=(f"stages:{len(stages)}",),
        )
    try:
        chain = TransformChain(tuple(stages))
    except (TypeError, ValueError) as exc:  # pragma: no cover - stages are typed
        return _refusal(
            CompatibilityCode.UNSUPPORTED_TRANSFORM_COMPOSITION,
            f"the composed stages do not form a typed chain: {exc}",
            failing_constraint="transform_composition",
        )
    return CompatibilityReport(
        code=CompatibilityCode.EXACT_COMPOSED_CHAIN,
        explanation=(
            "the axes correspond once every declarable field is relaxed, and "
            "each differing field has a declared exact conversion; the stages "
            "are composed through derived intermediate coordinate spaces"
        ),
        transform=chain,
    )


def _declared_rung(
    source_space: CoordinateSpace,
    target_space: CoordinateSpace,
    relaxable: _RelaxableField,
    table: Mapping[tuple[str, str, str], object],
    *,
    success_code: CompatibilityCode,
    undeclared_code: CompatibilityCode,
    noun: str,
    build_item,
    build_transform,
) -> CompatibilityReport | None:
    """Attempt one declared single-field rung, or return None to fall through.

    Shared by the timezone and unit rungs because their shape is identical:
    establish a positional correspondence with the field relaxed, look up what
    the caller declared for each differing axis, and refuse by name when a
    declaration is absent rather than deriving one.
    """

    differing = _single_field_differences(source_space, target_space, relaxable)
    if differing is None:
        return None
    # Positional correspondence is not the same as unique correspondence. With
    # the field relaxed, two axes can become interchangeable, and pairing them by
    # position would let *axis order* decide which conversion each one gets —
    # materially different numbers, chosen silently. The exact path refuses that
    # reasoning for equal semantics, and so must this. Where the axes do not
    # admit reordering, the positional pairing is the only admissible one and
    # this check confirms it rather than rejecting it.
    relaxed = {relaxable.key_argument: False}
    try:
        matching, unique = _unique_policy_matching(
            _build_matching_problem(source_space, target_space, relaxed)
        )
    except _MatchingBudgetExceeded:
        return _refusal(
            CompatibilityCode.RESOURCE_BUDGET_EXCEEDED,
            "the bounded axis-correspondence search reached its fixed work "
            f"budget of {MAX_MATCHING_EDGE_VISITS} edge visits",
            failing_constraint="matching_work_budget",
            evidence=(f"axes:{len(source_space.axes)}",),
            can_retry=True,
        )
    # `matching is None` means no admissible correspondence exists at all, which
    # is a policy failure rather than an ambiguity. Falling through lets the
    # transform's own constructor name the axis and family that refused it,
    # which is more use to a caller than a bare code.
    if matching is not None and not unique:
        return _refusal(
            CompatibilityCode.AMBIGUOUS_MAPPING,
            "more than one axis correspondence satisfies the declared policies "
            f"once {relaxable.field} is relaxed, so which axis receives which "
            "declared conversion is not determined",
            failing_constraint="unique_axis_correspondence",
        )
    items: list[object] = []
    missing: list[str] = []
    for index in differing:
        target_axis = target_space.axes[index]
        source_axis = source_space.axes[index]
        here = getattr(target_axis, relaxable.field)
        there = getattr(source_axis, relaxable.field)
        # A field present on one side only is a metadata gap, not a missing
        # declaration: every declaration type requires both names, so no caller
        # could supply one, and reporting UNDECLARED_* would invite a retry that
        # cannot succeed. The composed planner already answered this way; the
        # single-field rungs said something different for the same situation.
        if here is None or there is None:
            return _refusal(
                CompatibilityCode.INSUFFICIENT_METADATA,
                f"axis {target_axis.axis_id!r} declares a {relaxable.field} on "
                "only one side, so no conversion between them can be exact",
                failing_constraint=f"declared_{relaxable.field}",
                evidence=(f"axis:{target_axis.axis_id}",),
                can_retry=True,
            )
        key = (target_axis.semantic_id, here, there)
        declared = table.get(key)
        if declared is None:
            missing.append(f"{key[0]}:{key[1]}->{key[2]}")
            continue
        items.append(build_item(index, declared))
    if missing:
        return _refusal(
            undeclared_code,
            f"the axes correspond except for their declared {noun}s, and no "
            "conversion was declared for " + _summarize_items(tuple(missing)),
            failing_constraint=f"declared_{noun}_conversion",
            evidence=_cap_evidence(tuple(missing)),
            can_retry=True,
        )
    try:
        transform = build_transform(target_space, source_space, tuple(items))
    except ProhibitedTransformError as exc:
        # Reachable through an axis this rung did not pre-check: the converted
        # axes were screened for the family's policy, but an *unconverted* axis
        # still has to admit IDENTITY, and nothing before here looked at that.
        return _refusal(
            CompatibilityCode.PROHIBITED_TRANSFORM,
            str(exc),
            failing_constraint="transform_policy",
            evidence=(
                f"role:{exc.role}",
                f"axis:{exc.axis_id}",
                f"transform:{exc.transform_type}",
            ),
            can_retry=True,
        )
    except (TypeError, ValueError) as exc:
        # The declaration is well-formed but this pair of spaces cannot carry it
        # — a non-numeric axis, for instance. Name that rather than letting it
        # read as a missing declaration.
        return _refusal(
            CompatibilityCode.UNSUPPORTED_VALUE_SEMANTICS,
            f"the declared {noun} conversion cannot be applied exactly: {exc}",
            failing_constraint=f"exact_{noun}_conversion",
            evidence=tuple(f"axis:{target_space.axes[i].axis_id}" for i in differing),
        )
    return CompatibilityReport(
        code=success_code,
        explanation=(
            f"the axes correspond in order and every differing {noun} has a "
            "declared exact conversion"
        ),
        transform=TransformChain((transform,)),
    )


def _relaxable(policy: str) -> _RelaxableField:
    """Find a relaxable field by its family name.

    By name, not by position: `_RELAXABLE_FIELDS` is in ladder order, so
    inserting a rung renumbers it, and an index here would silently hand one
    rung another rung's field.
    """

    for field in _RELAXABLE_FIELDS:
        if field.policy == policy:
            return field
    raise KeyError(policy)  # pragma: no cover - callers pass literals


def _orientation_rung(source_space, target_space, table):
    return _declared_rung(
        source_space,
        target_space,
        _relaxable("AXIS_ORIENTATION"),
        table,
        success_code=CompatibilityCode.EXACT_AXIS_ORIENTATION,
        undeclared_code=CompatibilityCode.UNDECLARED_ORIENTATION_FLIP,
        noun="orientation",
        # The declaration carries nothing to copy across: an orientation
        # correction is negation, so the axis position is the whole item.
        build_item=lambda index, declared: index,
        build_transform=AxisOrientationTransform,
    )


def _timezone_rung(source_space, target_space, table):
    return _declared_rung(
        source_space,
        target_space,
        _relaxable("TIMEZONE"),
        table,
        success_code=CompatibilityCode.EXACT_TIMEZONE,
        undeclared_code=CompatibilityCode.UNDECLARED_TIMEZONE_CONVERSION,
        noun="timezone",
        build_item=lambda index, declared: AxisTimezoneOffset(
            index,
            declared.target_offset_minutes,
            declared.source_offset_minutes,
        ),
        build_transform=TimezoneTransform,
    )


def _label_reindex_rung(source_space, target_space, table):
    relaxable = _relaxable("LABEL_REINDEX")
    differing = _single_field_differences(source_space, target_space, relaxable)
    if differing is None or len(differing) != 1:
        # A LabelReindexTransform names one axis, so two differing label domains
        # need two transforms with a coordinate space between them. That is
        # composition, and the composed rung builds it.
        return None
    return _declared_rung(
        source_space,
        target_space,
        relaxable,
        table,
        success_code=CompatibilityCode.EXACT_LABEL_REINDEX,
        undeclared_code=CompatibilityCode.UNDECLARED_LABEL_REINDEX,
        noun="labels_ref",
        build_item=lambda index, declared: (index, declared.label_map),
        build_transform=_build_label_reindex,
    )


def _unit_affine_rung(source_space, target_space, table):
    return _declared_rung(
        source_space,
        target_space,
        _relaxable("UNIT_AFFINE"),
        table,
        success_code=CompatibilityCode.EXACT_UNIT_AFFINE,
        undeclared_code=CompatibilityCode.UNDECLARED_UNIT_CONVERSION,
        noun="unit",
        build_item=lambda index, declared: AxisUnitConversion(
            index, declared.scale, declared.offset
        ),
        build_transform=UnitAffineTransform,
    )


def analyze_exact_compatibility(
    *,
    source_space: CoordinateSpace,
    target_space: CoordinateSpace,
    unit_conversions: Iterable[DeclaredUnitConversion] = (),
    timezone_conversions: Iterable[DeclaredTimezoneConversion] = (),
    label_reindexes: Iterable[DeclaredLabelReindex] = (),
    orientation_flips: Iterable[DeclaredOrientationFlip] = (),
) -> CompatibilityReport:
    """Find identity, a unique exact axis permutation, or a declared conversion.

    The source is the native coordinate space and the target is the requested
    canonical coordinate space. This function never guesses a semantic mapping
    and never reports an optimality proof.

    Every declaration argument is caller-supplied and is consulted only after a
    correspondence has already been established with that one field relaxed.
    Supplying none of them leaves every result exactly as it was before these
    arguments existed: a unit, zone, label or orientation difference is then an
    obstruction, never something inferred away.

    Both spaces are keyword-only on purpose. Transforms are constructed
    ``(target_space, source_space)`` while analysis reads ``source`` first, so a
    positional call site that swapped the two would silently return the inverse
    permutation and still report it as exact.
    """

    if (
        type(source_space) is not CoordinateSpace
        or type(target_space) is not CoordinateSpace
    ):
        raise TypeError("source_space and target_space must be CoordinateSpace values")
    unit_table = _declaration_table(
        unit_conversions, DeclaredUnitConversion, "unit_conversions"
    )
    timezone_table = _declaration_table(
        timezone_conversions, DeclaredTimezoneConversion, "timezone_conversions"
    )
    label_table = _declaration_table(
        label_reindexes, DeclaredLabelReindex, "label_reindexes"
    )
    orientation_table = _declaration_table(
        orientation_flips, DeclaredOrientationFlip, "orientation_flips"
    )

    unsupported = tuple(
        f"{role}:{space.topology.value}"
        for role, space in (("source", source_space), ("target", target_space))
        if space.topology not in INITIAL_EXACT_TOPOLOGIES
    )
    if unsupported:
        return _refusal(
            CompatibilityCode.UNSUPPORTED_TOPOLOGY,
            "the bounded exact slice supports only rectangular tables",
            failing_constraint="supported_topology",
            evidence=unsupported,
        )
    if source_space.topology is not target_space.topology:
        return _refusal(
            CompatibilityCode.INCOMPATIBLE_TOPOLOGY,
            f"source topology {source_space.topology.value} does not match "
            f"target topology {target_space.topology.value}",
            failing_constraint="topology_identity",
            evidence=(
                f"source:{source_space.topology.value}",
                f"target:{target_space.topology.value}",
            ),
        )
    source_unsupported = source_space.unsupported_exact_features()
    target_unsupported = target_space.unsupported_exact_features()
    if source_unsupported or target_unsupported:
        details = []
        evidence = []
        if source_unsupported:
            details.append("source: " + _summarize_items(source_unsupported))
            evidence.extend("source." + item for item in source_unsupported)
        if target_unsupported:
            details.append("target: " + _summarize_items(target_unsupported))
            evidence.extend("target." + item for item in target_unsupported)
        return _refusal(
            CompatibilityCode.UNSUPPORTED_VALUE_SEMANTICS,
            "the bounded exact slice cannot enforce declared coordinate features ("
            + "; ".join(details)
            + ")",
            failing_constraint="supported_coordinate_domain",
            evidence=_cap_evidence(tuple(evidence)),
        )

    source_gaps = source_space.registration_metadata_gaps()
    target_gaps = target_space.registration_metadata_gaps()
    if source_gaps or target_gaps:
        details = []
        evidence = []
        if source_gaps:
            details.append("source: " + _summarize_items(source_gaps))
            evidence.extend("source." + gap for gap in source_gaps)
        if target_gaps:
            details.append("target: " + _summarize_items(target_gaps))
            evidence.extend("target." + gap for gap in target_gaps)
        return _refusal(
            CompatibilityCode.INSUFFICIENT_METADATA,
            "strict exact registration requires explicit metadata ("
            + "; ".join(details)
            + ")",
            failing_constraint="registration_metadata",
            evidence=_cap_evidence(tuple(evidence)),
            can_retry=True,
        )
    if len(source_space.axes) != len(target_space.axes):
        return _refusal(
            CompatibilityCode.INCOMPATIBLE_TOPOLOGY,
            "source and target axis counts differ",
            failing_constraint="axis_count",
            evidence=(
                f"source:{len(source_space.axes)}",
                f"target:{len(target_space.axes)}",
            ),
        )
    if source_space.exact_space_key() != target_space.exact_space_key():
        return _refusal(
            CompatibilityCode.INCOMPATIBLE_SEMANTICS,
            "space-wide declared coordinate metadata differs",
            failing_constraint="space_metadata_identity",
        )

    source_keys = tuple(axis.exact_semantics_key() for axis in source_space.axes)
    target_keys = tuple(axis.exact_semantics_key() for axis in target_space.axes)
    if Counter(source_keys) != Counter(target_keys):
        # The ladder tries identity and permutation on exact semantics first, so
        # arriving here means declared semantics differ. The single-field rungs
        # are the next-least-expressive ones that could still be exact, tried in
        # §14.2 order and only against what the caller explicitly declared.
        for rung, rung_table in (
            (_orientation_rung, orientation_table),
            (_label_reindex_rung, label_table),
            (_timezone_rung, timezone_table),
            (_unit_affine_rung, unit_table),
        ):
            report = rung(source_space, target_space, rung_table)
            if report is not None:
                return report
        composed = _composed_rung(
            source_space,
            target_space,
            {
                "AXIS_ORIENTATION": orientation_table,
                "LABEL_REINDEX": label_table,
                "TIMEZONE": timezone_table,
                "UNIT_AFFINE": unit_table,
            },
        )
        if composed is not None:
            return composed
        return _refusal(
            CompatibilityCode.INCOMPATIBLE_SEMANTICS,
            "source and target axis semantics do not admit a complete bijection",
            failing_constraint="axis_semantic_bijection",
        )
    try:
        order, unique = _unique_policy_matching(
            _build_matching_problem(source_space, target_space)
        )
    except _MatchingBudgetExceeded:
        return _refusal(
            CompatibilityCode.RESOURCE_BUDGET_EXCEEDED,
            "the bounded axis-correspondence search reached its fixed work "
            f"budget of {MAX_MATCHING_EDGE_VISITS} edge visits",
            failing_constraint="matching_work_budget",
            evidence=(
                f"axes:{len(source_space.axes)}",
                f"edge_visit_budget:{MAX_MATCHING_EDGE_VISITS}",
            ),
            can_retry=True,
        )
    if order is None:
        return _refusal(
            CompatibilityCode.PROHIBITED_TRANSFORM,
            "declared transform policies prohibit every complete semantic mapping",
            failing_constraint="transform_policy",
            can_retry=True,
        )
    if not unique:
        return _refusal(
            CompatibilityCode.AMBIGUOUS_MAPPING,
            "more than one semantic mapping satisfies the declared policies",
            failing_constraint="unique_axis_correspondence",
        )
    if order == tuple(range(len(order))):
        try:
            transform = IdentityTransform(target_space, source_space)
        except ProhibitedTransformError as exc:
            return _refusal(
                CompatibilityCode.PROHIBITED_TRANSFORM,
                str(exc),
                failing_constraint="transform_policy",
                evidence=(
                    f"role:{exc.role}",
                    f"axis:{exc.axis_id}",
                    f"transform:{exc.transform_type}",
                ),
                can_retry=True,
            )
        return CompatibilityReport(
            code=CompatibilityCode.EXACT_IDENTITY,
            explanation="ordered declared coordinate metadata matches exactly",
            transform=TransformChain((transform,)),
        )
    try:
        transform = AxisPermutationTransform(target_space, source_space, order)
    except ProhibitedTransformError as exc:
        return _refusal(
            CompatibilityCode.PROHIBITED_TRANSFORM,
            str(exc),
            failing_constraint="transform_policy",
            evidence=(
                f"role:{exc.role}",
                f"axis:{exc.axis_id}",
                f"transform:{exc.transform_type}",
            ),
            can_retry=True,
        )
    return CompatibilityReport(
        code=CompatibilityCode.EXACT_AXIS_PERMUTATION,
        explanation="one unique exact semantic axis permutation exists",
        transform=TransformChain((transform,)),
    )
