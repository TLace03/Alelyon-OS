"""The native-artifact manifest: what a payload is, and in which space.

§12 opens with the reason this record exists: *a registered payload cannot be
trusted if its native interpretation is incomplete*. A `CoordinateSpace` says
what an axis **means**; it says nothing about the bytes anyone claims live on
it. Until now the exact slice registered spaces against spaces, so an "artifact"
was a thing the certificate had to describe by its absence — `certificate.py`
carries `NOT_APPLICABLE: no artifact manifest exists in this slice`. This module
is that record.

What it does, exactly
---------------------
It binds a declared payload to a declared coordinate space and refuses the pairs
that cannot be interpreted: an extent list that does not match the space's axes,
a labelled axis whose declared extent contradicts its own label dictionary, a
declared cell count that overflows a budget, a parent reference that points at
the artifact itself.

What it does not do, and why saying so matters
----------------------------------------------
**This module never reads a payload.** Nothing in the exact slice does — the
morphometry path reads a model runtime's *declared metadata*, and this module
reads a caller's *declared manifest*. So §12's validation list splits cleanly in
two, and only one half is implementable here:

* checkable now, because both sides are declarations — extent/axis agreement,
  label-extent agreement, axis order, uniqueness, self-reference, and the
  arithmetic that turns a small declaration into an enormous one;
* **UNMEASURED until a payload reader exists** — that the payload's real shape
  matches the declared one, that its dtype is what `value_encoding` says, that
  masks line up with real coordinates, that no decompression bomb sits behind
  `payload_ref`. `ManifestGaps` names each of these rather than leaving a reader
  to assume the list was satisfied.

Two of §12's checks are refused for a third reason: they need registries this
engine deliberately does not have. "Units are parseable and dimensionally
valid" needs a unit registry, and §14.2's unit rung already established the rule
— a unit is *declared* here, never parsed, because nothing holds a unit table.
"Time metadata is internally consistent" needs calendar data for the same
reason. Both stay declarations, and `strict_refusals()` names them when a caller
asks for strict admission.

A manifest is therefore evidence of *what was claimed*, committed byte-exactly
so a later reader can detect revision of the claim. It is not evidence that the
claim is true. That is the same line the whole engine draws.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from alelyon.runtime.vector.lattice.contracts import (
    CoordinateSpace,
    MAX_AXES,
    _bounded_tuple,
    _content_ref,
    _metadata,
    _name_set,
    _optional_text,
    _text,
)


ARTIFACT_MANIFEST_SCHEMA = "alelyon.lattice.artifact-manifest/0.1"

MAX_PARENT_REFS = 64
MAX_EXTENT = 2**53 - 1

#: Ceiling on the product of an artifact's declared extents.
#:
#: A shape bomb is not a big manifest — it is a *small* one. Eight axes of
#: 10^6 extents encode in a few hundred bytes and declare 10^48 cells, so a
#: per-field limit cannot catch it and the byte budget that bounds a
#: `CoordinateSpace` is the wrong instrument. §12 names "integer overflow" and
#: "shape bomb" as distinct refusals; in Python the first cannot occur, so the
#: product is checked directly and incrementally, against this budget.
MAX_DECLARED_CELLS = 2**48


class ValueSemantics(str, Enum):
    """§13.1's value classes.

    The class decides which remapping operator is admissible, which is why it is
    a required field rather than metadata: §13.1 exists so that a price and a
    count cannot share one interpolation rule, and a default would be a silent
    choice of rule. No remapping operator exists in this slice, so nothing here
    consumes the class yet — it is recorded now so that when one does, the
    artifacts already carry the declaration instead of acquiring it by guess.
    """

    INTENSIVE_SCALAR = "INTENSIVE_SCALAR"
    EXTENSIVE_TOTAL = "EXTENSIVE_TOTAL"
    DENSITY = "DENSITY"
    COUNT = "COUNT"
    PROBABILITY_MASS = "PROBABILITY_MASS"
    PROBABILITY_DENSITY = "PROBABILITY_DENSITY"
    CATEGORICAL_LABEL = "CATEGORICAL_LABEL"
    IDENTIFIER = "IDENTIFIER"
    BOOLEAN_MASK = "BOOLEAN_MASK"
    VECTOR = "VECTOR"
    TENSOR = "TENSOR"
    EVENT = "EVENT"
    ORDINAL_SCORE = "ORDINAL_SCORE"
    COMPLEX = "COMPLEX"


#: Value classes §13.1 forbids continuous interpolation over.
#:
#: Held here rather than in a future remapping module because the manifest is
#: where the declaration is made, and a reader deciding whether a plan is
#: admissible should not have to import an operator registry to learn that a
#: security identifier must never be averaged.
NON_INTERPOLABLE_SEMANTICS = frozenset(
    {
        ValueSemantics.CATEGORICAL_LABEL,
        ValueSemantics.IDENTIFIER,
        ValueSemantics.BOOLEAN_MASK,
        ValueSemantics.EVENT,
    }
)


class MissingnessState(str, Enum):
    """§13.4's states. A numeric sentinel is explicitly not sufficient."""

    NOT_OBSERVED = "NOT_OBSERVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    OUTSIDE_DOMAIN = "OUTSIDE_DOMAIN"
    MASKED_BY_POLICY = "MASKED_BY_POLICY"
    INVALID_SOURCE = "INVALID_SOURCE"
    CENSORED = "CENSORED"
    BELOW_DETECTION = "BELOW_DETECTION"
    COMPUTATION_FAILED = "COMPUTATION_FAILED"
    UNKNOWN_REASON = "UNKNOWN_REASON"


class UncertaintyModel(str, Enum):
    """§13.5's representations, plus an explicit absence.

    ``NONE`` is a declaration that the artifact carries no uncertainty, which is
    a different statement from a field left unset. §8.9 requires propagation or
    refusal, so "there is nothing to propagate" has to be sayable.
    """

    NONE = "NONE"
    CERTIFIED_INTERVAL = "CERTIFIED_INTERVAL"
    PER_CELL_INTERVAL = "PER_CELL_INTERVAL"
    STANDARD_ERROR = "STANDARD_ERROR"
    COVARIANCE = "COVARIANCE"
    LOW_RANK_COVARIANCE = "LOW_RANK_COVARIANCE"
    DISTRIBUTION_SAMPLES = "DISTRIBUTION_SAMPLES"
    CONFIDENCE_WEIGHT = "CONFIDENCE_WEIGHT"


class ManifestValidationError(ValueError):
    """A manifest cannot be interpreted in the space it names."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class AxisExtent:
    """One axis's declared size, named by the axis it belongs to.

    Named rather than positional. §12 requires "axis count and order match the
    coordinate-space definition", and a bare tuple of integers can only express
    the count — a caller who reorders two axes produces a manifest that is
    equally valid against the same space and means something else. Carrying the
    `axis_id` makes the order checkable instead of assumed, which is the same
    reason ADR-0007 made the transform rungs check axis pairing for uniqueness
    rather than pair by position.
    """

    axis_id: str
    extent: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "axis_id",
            _text(self.axis_id, "axis_id", identifier=True, maximum=256),
        )
        if type(self.extent) is not int or type(self.extent) is bool:
            raise TypeError("extent must be an int")
        if self.extent < 0:
            raise ValueError("extent must not be negative")
        if self.extent > MAX_EXTENT:
            raise ValueError(f"extent exceeds the {MAX_EXTENT} limit")


@dataclass(frozen=True, slots=True)
class ManifestGaps:
    """What a manifest does not establish, enumerated rather than implied.

    Every field is a §12 validation item this slice cannot perform. Returning
    them as a record — instead of omitting them — is what stops "the manifest
    validated" from being read as "§12's list was satisfied".
    """

    #: True while no payload reader exists, which is always, in this slice.
    payload_unread: bool = True

    def reasons(self) -> tuple[str, ...]:
        if not self.payload_unread:
            return ()
        return (
            "PAYLOAD_SHAPE_UNMEASURED: the declared extents were checked against "
            "the coordinate space, not against payload bytes",
            "PAYLOAD_DTYPE_UNMEASURED: value_encoding is declared and unread",
            "PAYLOAD_COMMITMENT_UNVERIFIED: payload_commitment was not recomputed "
            "from payload bytes",
            "MASK_ALIGNMENT_UNMEASURED: masks were not compared against payload "
            "coordinates",
            "DECOMPRESSION_BOMB_UNMEASURED: nothing behind payload_ref was opened",
            "UNIT_DIMENSIONALITY_UNCHECKED: no unit registry exists; units are "
            "declared, never parsed",
            "TIME_METADATA_UNCHECKED: no calendar registry exists; calendar and "
            "timezone fields are declared, never resolved",
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """§12's `QuantitativeArtifactManifest`, bounded to what this slice can hold.

    Fields §12 lists that are **absent here on purpose**: `payload_format` is
    kept but `security_classification`, `tenant_id` and `parent_artifact_refs`'
    lineage semantics belong to Atlas (§7 gives it storage, lineage, security
    labels and tenant boundaries), and inventing local copies would be the
    "parallel infrastructure" §5 forbids. `tenant_id` and
    `security_classification` are therefore *not* fields: a manifest that
    carried them would imply this engine enforces them, and it does not.
    `parent_artifact_refs` is kept because a self-reference is checkable without
    Atlas and §12 names cyclic references explicitly.
    """

    artifact_id: str
    payload_ref: str
    payload_format: str
    payload_commitment: str
    native_space: CoordinateSpace
    extents: tuple[AxisExtent, ...]
    value_encoding: str
    value_semantics: ValueSemantics
    uncertainty_model: UncertaintyModel = UncertaintyModel.NONE
    missingness_states: tuple[MissingnessState, ...] = ()
    provenance: str | None = None
    producer_build: str | None = None
    parent_artifact_refs: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA
    declared_cells: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _text(self.artifact_id, "artifact_id", identifier=True, maximum=512),
        )
        object.__setattr__(
            self,
            "payload_ref",
            _text(self.payload_ref, "payload_ref", identifier=True, maximum=1024),
        )
        object.__setattr__(
            self,
            "payload_format",
            _text(self.payload_format, "payload_format", identifier=True, maximum=128),
        )
        object.__setattr__(
            self,
            "payload_commitment",
            _content_ref(self.payload_commitment, "payload_commitment"),
        )
        if type(self.native_space) is not CoordinateSpace:
            raise TypeError("native_space must be a CoordinateSpace")
        if type(self.value_semantics) is not ValueSemantics:
            raise TypeError("value_semantics must be a ValueSemantics")
        if type(self.uncertainty_model) is not UncertaintyModel:
            raise TypeError("uncertainty_model must be an UncertaintyModel")
        object.__setattr__(
            self,
            "value_encoding",
            _text(self.value_encoding, "value_encoding", identifier=True, maximum=128),
        )

        extents = _bounded_tuple(self.extents, "extents", MAX_AXES)
        if any(type(item) is not AxisExtent for item in extents):
            raise TypeError("extents must contain only AxisExtent values")
        object.__setattr__(self, "extents", extents)

        states = _bounded_tuple(
            self.missingness_states, "missingness_states", len(MissingnessState)
        )
        if any(type(item) is not MissingnessState for item in states):
            raise TypeError("missingness_states must contain only MissingnessState")
        if len(set(states)) != len(states):
            raise ValueError("missingness_states must not contain duplicates")
        # Sorted so two manifests declaring the same states commit to the same
        # bytes; a set's iteration order must never reach a content hash.
        object.__setattr__(
            self, "missingness_states", tuple(sorted(states, key=lambda s: s.value))
        )

        object.__setattr__(
            self, "provenance", _optional_text(self.provenance, "provenance")
        )
        object.__setattr__(
            self,
            "producer_build",
            _optional_text(self.producer_build, "producer_build", identifier=True),
        )
        object.__setattr__(
            self,
            "parent_artifact_refs",
            _name_set(
                self.parent_artifact_refs,
                "parent_artifact_refs",
                maximum=MAX_PARENT_REFS,
            ),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.schema_version != ARTIFACT_MANIFEST_SCHEMA:
            raise ValueError(
                f"schema_version must be {ARTIFACT_MANIFEST_SCHEMA!r} "
                "for this implementation"
            )

        object.__setattr__(self, "declared_cells", _declared_cells(self.extents))
        _validate_against_space(self)

    def gaps(self) -> ManifestGaps:
        return ManifestGaps()

    def strict_refusals(self) -> tuple[str, ...]:
        """Why §12 forbids guessing this artifact into a template in strict mode.

        Empty means only that nothing *this slice can see* refuses it. The gaps
        in :meth:`gaps` are unmeasured either way and are not repeated here,
        because a refusal a caller can act on and an absence of evidence are
        different things and merging them would let a caller clear the first by
        ignoring the second.
        """

        refusals: list[str] = []
        if self.provenance is None:
            refusals.append(
                "PROVENANCE_ABSENT: §12 requires provenance under strict policy"
            )
        if self.producer_build is None:
            refusals.append(
                "PRODUCER_BUILD_ABSENT: §12 requires the producing build under "
                "strict policy"
            )
        gaps = self.native_space.registration_metadata_gaps()
        if gaps:
            refusals.append(
                "SPACE_METADATA_INCOMPLETE: the native space leaves "
                f"{', '.join(gaps)} undeclared"
            )
        return tuple(refusals)


def _declared_cells(extents: tuple[AxisExtent, ...]) -> int:
    """Multiply extents, refusing before the product becomes unwieldy.

    Incremental rather than `math.prod(...) > budget`: the product of eight
    64-bit extents is a 512-bit integer, and computing it in order to reject it
    is the arithmetic the caller was trying to provoke. Checking each step keeps
    the work proportional to the axis count.
    """

    total = 1
    for item in extents:
        if item.extent == 0:
            # A zero extent makes the artifact empty. Short-circuiting keeps a
            # later huge extent from being multiplied out pointlessly, and an
            # empty artifact is a legitimate declaration rather than a bomb.
            return 0
        total *= item.extent
        if total > MAX_DECLARED_CELLS:
            raise ManifestValidationError(
                "SHAPE_BOMB",
                f"declared extents exceed the {MAX_DECLARED_CELLS}-cell budget",
            )
    return total


def _validate_against_space(manifest: ArtifactManifest) -> None:
    """§12's checkable half: the manifest against the space it names."""

    space = manifest.native_space
    extents = manifest.extents

    if len(extents) != len(space.axes):
        raise ManifestValidationError(
            "AXIS_COUNT_MISMATCH",
            f"{len(extents)} extents for {len(space.axes)} axes",
        )
    declared_ids = [item.axis_id for item in extents]
    if len(set(declared_ids)) != len(declared_ids):
        raise ManifestValidationError(
            "DUPLICATE_EXTENT_AXIS", "one axis is given an extent twice"
        )
    space_ids = [axis.axis_id for axis in space.axes]
    if declared_ids != space_ids:
        if set(declared_ids) == set(space_ids):
            raise ManifestValidationError(
                "AXIS_ORDER_MISMATCH",
                "extents name the space's axes in a different order; axis order "
                "is coordinate meaning, not presentation",
            )
        raise ManifestValidationError(
            "AXIS_ID_MISMATCH",
            "extents do not name the coordinate space's axes",
        )

    for axis, item in zip(space.axes, extents, strict=True):
        if axis.labels is not None and item.extent != len(axis.labels):
            raise ManifestValidationError(
                "LABEL_EXTENT_MISMATCH",
                f"axis {axis.axis_id!r} declares {len(axis.labels)} labels but "
                f"the manifest declares extent {item.extent}",
            )

    if manifest.artifact_id in manifest.parent_artifact_refs:
        raise ManifestValidationError(
            "CYCLIC_PARENT_REF",
            f"{manifest.artifact_id!r} is listed as its own parent",
        )


def declared_extent(manifest: ArtifactManifest, axis_id: str) -> int:
    """Return one axis's declared extent, or raise a named lookup error."""

    axis_id = _text(axis_id, "axis_id", identifier=True, maximum=256)
    for item in manifest.extents:
        if item.axis_id == axis_id:
            return item.extent
    raise KeyError(f"unknown axis_id {axis_id!r}")


def manifests_share_shape(
    left: ArtifactManifest, right: ArtifactManifest
) -> tuple[str, ...]:
    """Report why two artifacts are not directly comparable, cell for cell.

    Returns the differing axis identifiers, empty when the declared shapes
    agree. This answers a question morphometry will ask constantly and that a
    caller would otherwise answer by comparing tuples positionally — which is
    wrong for the same reason `AxisExtent` carries an id.
    """

    differing: list[str] = []
    left_by_id = {item.axis_id: item.extent for item in left.extents}
    right_by_id = {item.axis_id: item.extent for item in right.extents}
    for axis_id in sorted(set(left_by_id) | set(right_by_id)):
        if left_by_id.get(axis_id) != right_by_id.get(axis_id):
            differing.append(axis_id)
    return tuple(differing)


def validate_extents(
    space: CoordinateSpace, extents: Iterable[object]
) -> tuple[AxisExtent, ...]:
    """Coerce and check extents against a space without building a manifest.

    Exposed because compatibility analysis wants the shape check on its own,
    before a payload reference or commitment exists to put in a manifest.
    """

    materialized = _bounded_tuple(extents, "extents", MAX_AXES)
    if any(type(item) is not AxisExtent for item in materialized):
        raise TypeError("extents must contain only AxisExtent values")
    if len(materialized) != len(space.axes):
        raise ManifestValidationError(
            "AXIS_COUNT_MISMATCH",
            f"{len(materialized)} extents for {len(space.axes)} axes",
        )
    if [item.axis_id for item in materialized] != [
        axis.axis_id for axis in space.axes
    ]:
        raise ManifestValidationError(
            "AXIS_ORDER_MISMATCH", "extents do not match the space's axis order"
        )
    _declared_cells(materialized)
    return materialized
