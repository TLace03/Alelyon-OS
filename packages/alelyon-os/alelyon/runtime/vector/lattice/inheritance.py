"""Whether a child template narrows its parent, or quietly redefines it.

§11.6 ends with one sentence — *a child template may narrow constraints but must
not silently redefine parent semantics* — and the word carrying the weight is
**silently**. A redefinition that is declared and reviewed is a new base
contract; a redefinition that arrives wearing a child's tier is the failure the
sentence is about, because every artifact registered against the child is then
interpreted under a meaning the parent never agreed to.

Why this is not `analyze_exact_compatibility`
---------------------------------------------
[ADR-0027](../../../../docs/architecture/adr/ADR-0027-lattice-template-registry.md)
named `registration.py`'s space comparison as the next step for this check.
Building it showed that to be the wrong tool, and the reason is worth stating
because the two questions look alike:

* Registration asks **can a payload move from A to B exactly?** and answers yes
  whenever a *declared conversion* bridges the difference. A child that
  redeclares its parent's axis in feet instead of metres registers exactly, via
  a declared unit conversion — and has redefined the parent's unit. Exact
  registrability is not preservation of meaning.
* Inheritance asks **does the child leave the parent's declared meaning
  intact?**, which is directional. Registration is a search for *any* exact
  correspondence, so it happily reports the inverse permutation as exact.
* Every real narrowing changes `exact_semantics_key()` — restricting a cohort's
  labels, adding bounds the parent left open, fixing a resolution. Requiring
  `EXACT_IDENTITY` between child and parent would therefore be a rule that
  nothing may narrow, which is the opposite of §11.6.
* Registration refuses any space outside the bounded exact slice: a topology
  other than a rectangular table, or an axis carrying bounds, origin,
  resolution or periodicity. Templates are not restricted to that slice, so
  wiring it in would refuse inheritance *because this build's exact core is
  narrow*, and report an infrastructure limit as a policy verdict.

So the comparison here is its own thing: a field-by-field walk in which every
declared field of `CoordinateAxis` and `CoordinateSpace` is classified once, and
the classification is what a test asserts is total.

The three answers, and why "undecidable" is a refusal
------------------------------------------------------
A comparison returns findings in three buckets. `redefinitions` are checked and
violated. `narrowings` are checked and legitimate — reported rather than
discarded, so a reviewer can see what a child actually restricted.
`undecidable` is the honest third answer: a difference this build can see but
cannot order, such as two `bounds` pairs whose endpoints are opaque strings.

An undecidable difference makes `narrows()` false. Treating it as a pass would
publish an unproven narrowing under a rule whose whole purpose is to prevent
one, and the caller who hits it has a remedy that costs them nothing but
honesty: declare a new template rather than a derived one, or express the bound
as a rational this build can compare.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from fractions import Fraction

from alelyon.runtime.vector.lattice.contracts import (
    _RATIONAL,
    CoordinateAxis,
    CoordinateSpace,
)


class FieldClass(str, Enum):
    """How one declared field participates in the inheritance comparison."""

    #: Identifies the record or the axis; two templates differing here are
    #: expected to, and it says nothing about meaning.
    NAMING = "NAMING"
    #: Declares what the axis or space *means*. A child changing one has
    #: redefined its parent, whatever tier it claims.
    IDENTITY = "IDENTITY"
    #: A constraint a child may legitimately make stricter, in a direction this
    #: module can check.
    NARROWABLE = "NARROWABLE"
    #: Free annotation. A child may add; it may not overwrite or drop.
    ANNOTATION = "ANNOTATION"
    #: Not compared directly because it is walked as structure.
    STRUCTURAL = "STRUCTURAL"


#: Every ``init=True`` field of `CoordinateAxis`, classified exactly once.
#: `test_lattice_inheritance` asserts this covers the dataclass with nothing
#: left over, so an axis field added upstream cannot arrive unclassified and be
#: silently exempt from inheritance.
AXIS_FIELD_CLASSES: dict[str, FieldClass] = {
    "axis_id": FieldClass.NAMING,
    "semantic_id": FieldClass.IDENTITY,
    "kind": FieldClass.IDENTITY,
    "scalar_type": FieldClass.IDENTITY,
    "ordering": FieldClass.IDENTITY,
    "unit": FieldClass.IDENTITY,
    "reference_frame": FieldClass.IDENTITY,
    "calendar": FieldClass.IDENTITY,
    "timezone": FieldClass.IDENTITY,
    "orientation": FieldClass.IDENTITY,
    "origin": FieldClass.IDENTITY,
    "periodicity": FieldClass.IDENTITY,
    "missingness_policy": FieldClass.IDENTITY,
    "resolution": FieldClass.NARROWABLE,
    "bounds": FieldClass.NARROWABLE,
    "labels": FieldClass.NARROWABLE,
    "labels_ref": FieldClass.NARROWABLE,
    "interpolation_policy": FieldClass.NARROWABLE,
    "transform_policy": FieldClass.NARROWABLE,
    "metadata": FieldClass.ANNOTATION,
}

#: Every ``init=True`` field of `CoordinateSpace`, classified exactly once.
SPACE_FIELD_CLASSES: dict[str, FieldClass] = {
    "space_id": FieldClass.NAMING,
    "version": FieldClass.NAMING,
    "topology": FieldClass.IDENTITY,
    "index_convention": FieldClass.IDENTITY,
    "unit_system": FieldClass.IDENTITY,
    "reference_frame": FieldClass.IDENTITY,
    "valid_domain_rule": FieldClass.IDENTITY,
    "schema_version": FieldClass.IDENTITY,
    "region_atlas_refs": FieldClass.NARROWABLE,
    "metadata": FieldClass.ANNOTATION,
    "axes": FieldClass.STRUCTURAL,
}


def _identity_fields(classes: dict[str, FieldClass]) -> tuple[str, ...]:
    return tuple(
        sorted(name for name, kind in classes.items() if kind is FieldClass.IDENTITY)
    )


AXIS_IDENTITY_FIELDS = _identity_fields(AXIS_FIELD_CLASSES)
SPACE_IDENTITY_FIELDS = _identity_fields(SPACE_FIELD_CLASSES)

#: Identity fields whose redefinition earns its own code rather than the generic
#: one. Topology is the space's shape, and §11.6 singles it out: a reader
#: scanning a refusal should see *what kind* of redefinition happened without
#: parsing a locus out of the message.
_SPACE_FIELD_CODES = {"topology": "TOPOLOGY_REDEFINED"}


@dataclass(frozen=True, slots=True)
class InheritanceFinding:
    """One reason, attached to the place it was found."""

    code: str
    locus: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code} at {self.locus}: {self.detail}"


@dataclass(frozen=True, slots=True)
class InheritanceReport:
    """What comparing a child space against its parent established.

    `narrowings` is carried rather than discarded because "the child restricted
    the cohort to 12 of 500 entities" is the reviewable content of a derived
    template; a bare boolean would throw it away.
    """

    redefinitions: tuple[InheritanceFinding, ...] = ()
    undecidable: tuple[InheritanceFinding, ...] = ()
    narrowings: tuple[InheritanceFinding, ...] = ()

    @property
    def narrows(self) -> bool:
        """True only when nothing was redefined **and** nothing was undecidable.

        An undecidable difference is not a pass. See the module docstring.
        """

        return not self.redefinitions and not self.undecidable

    def reasons(self) -> tuple[str, ...]:
        """Every blocking finding, rendered for a refusal message."""

        return tuple(str(item) for item in self.redefinitions + self.undecidable)


def _maybe_rational(value: str) -> Fraction | None:
    """Parse a bound endpoint, or return None when it is not a rational.

    Bound endpoints are free text by contract — ``"2024-01-01"`` and ``"-inf"``
    are as admissible as ``"3/4"``. Only the rational ones can be ordered, and
    the rest become an undecidable finding rather than an assumed one.
    """

    if _RATIONAL.fullmatch(value) is None:
        return None
    return Fraction(value)


def _bounds_interval(
    bounds: tuple[str, str],
) -> tuple[Fraction, Fraction] | None:
    """A comparable interval, or None when the pair cannot be ordered.

    A pair whose parsed lower exceeds its parsed upper is rejected too:
    `CoordinateAxis` does not require ``lower <= upper``, so such a pair has no
    established reading, and guessing one would decide containment from a
    convention this contract never states.
    """

    low = _maybe_rational(bounds[0])
    high = _maybe_rational(bounds[1])
    if low is None or high is None or low > high:
        return None
    return (low, high)


def _is_subsequence(child: tuple[str, ...], parent: tuple[str, ...]) -> bool:
    """Whether ``child`` appears inside ``parent`` in the same relative order."""

    remaining = iter(parent)
    return all(label in remaining for label in child)


def _compare_optional_constraint(
    *,
    field_name: str,
    locus: str,
    child_value: object,
    parent_value: object,
    redefinitions: list[InheritanceFinding],
    narrowings: list[InheritanceFinding],
) -> bool:
    """Handle the declared/undeclared cases shared by every optional constraint.

    Returns True when the pair is settled here. The asymmetry is the point: a
    parent that left a constraint open may have it closed by a child, but a
    constraint the parent declared cannot be dropped — dropping it removes a
    guarantee every consumer of the parent already relies on, which is a
    widening wearing a child's tier.
    """

    if child_value == parent_value:
        return True
    if parent_value is None:
        narrowings.append(
            InheritanceFinding(
                f"{field_name.upper()}_DECLARED",
                locus,
                f"child declares {child_value!r} where the parent left it open",
            )
        )
        return True
    if child_value is None:
        redefinitions.append(
            InheritanceFinding(
                f"{field_name.upper()}_DROPPED",
                locus,
                f"parent declares {parent_value!r}; dropping it widens rather "
                "than narrows",
            )
        )
        return True
    return False


def _compare_labels(
    *,
    locus: str,
    child: CoordinateAxis,
    parent: CoordinateAxis,
    redefinitions: list[InheritanceFinding],
    undecidable: list[InheritanceFinding],
    narrowings: list[InheritanceFinding],
) -> None:
    """Compare a label domain, where narrowing means an ordered restriction."""

    if child.labels is None or parent.labels is None:
        # At least one side did not spell its domain out, so the ordered
        # comparison below cannot run and the commitment is what is left.
        if child.labels_ref != parent.labels_ref:
            # Consulting the reference here rather than only when *both* sides
            # omit their labels is deliberate: a parent committing to a
            # dictionary by reference, against a child spelling out a
            # *different* one, would otherwise read as a child closing a
            # constraint the parent left open. The parent left nothing open.
            if parent.labels_ref is None or child.labels_ref is None:
                _compare_optional_constraint(
                    field_name="labels",
                    locus=locus,
                    child_value=child.labels_ref,
                    parent_value=parent.labels_ref,
                    redefinitions=redefinitions,
                    narrowings=narrowings,
                )
                return
            undecidable.append(
                InheritanceFinding(
                    "LABEL_DICTIONARY_UNRESOLVED",
                    locus,
                    "a label dictionary is committed by reference only, and no "
                    f"dictionary store exists to resolve {child.labels_ref} "
                    f"against {parent.labels_ref}",
                )
            )
            return
        # The references agree, or neither side carries one. Any remaining
        # difference is in whether the domain was spelled out at all, and a
        # child that enumerates what its parent left open has narrowed
        # something — reporting it keeps `narrowings` a faithful account.
        _compare_optional_constraint(
            field_name="labels",
            locus=locus,
            child_value=child.labels,
            parent_value=parent.labels,
            redefinitions=redefinitions,
            narrowings=narrowings,
        )
        return

    if child.labels == parent.labels:
        return

    child_labels = child.labels
    parent_labels = parent.labels
    added = tuple(label for label in child_labels if label not in set(parent_labels))
    if added:
        redefinitions.append(
            InheritanceFinding(
                "LABEL_DOMAIN_WIDENED",
                locus,
                f"child admits {len(added)} label(s) the parent does not, "
                f"beginning with {added[0]!r}",
            )
        )
        return
    if not _is_subsequence(child_labels, parent_labels):
        redefinitions.append(
            InheritanceFinding(
                "LABEL_DOMAIN_REINDEXED",
                locus,
                "child's labels are a subset of the parent's but in a different "
                "order; reordering a label dictionary changes which coordinate "
                "each position addresses, which is a reindex, not a restriction",
            )
        )
        return
    narrowings.append(
        InheritanceFinding(
            "LABEL_DOMAIN_RESTRICTED",
            locus,
            f"child restricts {len(parent_labels)} label(s) to "
            f"{len(child_labels)}, preserving order",
        )
    )


def _compare_bounds(
    *,
    locus: str,
    child: CoordinateAxis,
    parent: CoordinateAxis,
    redefinitions: list[InheritanceFinding],
    undecidable: list[InheritanceFinding],
    narrowings: list[InheritanceFinding],
) -> None:
    """Compare declared bounds, deciding only what rationals make decidable."""

    if _compare_optional_constraint(
        field_name="bounds",
        locus=locus,
        child_value=child.bounds,
        parent_value=parent.bounds,
        redefinitions=redefinitions,
        narrowings=narrowings,
    ):
        return

    child_interval = _bounds_interval(child.bounds)
    parent_interval = _bounds_interval(parent.bounds)
    if child_interval is None or parent_interval is None:
        undecidable.append(
            InheritanceFinding(
                "BOUNDS_NOT_ORDERED",
                locus,
                f"child declares {child.bounds} and the parent {parent.bounds}; "
                "bound endpoints are free text by contract, and this build "
                "orders only endpoints that parse as canonical rationals",
            )
        )
        return
    if child_interval[0] >= parent_interval[0] and child_interval[1] <= parent_interval[1]:
        narrowings.append(
            InheritanceFinding(
                "BOUNDS_TIGHTENED",
                locus,
                f"child's {child.bounds} lies inside the parent's {parent.bounds}",
            )
        )
        return
    redefinitions.append(
        InheritanceFinding(
            "BOUNDS_WIDENED",
            locus,
            f"child's {child.bounds} is not contained in the parent's "
            f"{parent.bounds}",
        )
    )


def _compare_name_set(
    *,
    code_stem: str,
    field_name: str,
    locus: str,
    child_names: tuple[str, ...],
    parent_names: tuple[str, ...],
    redefinitions: list[InheritanceFinding],
    narrowings: list[InheritanceFinding],
) -> None:
    """Compare a policy set, where narrowing is exactly subset."""

    widened = tuple(name for name in child_names if name not in set(parent_names))
    if widened:
        redefinitions.append(
            InheritanceFinding(
                f"{code_stem}_WIDENED",
                locus,
                f"child's {field_name} admits {', '.join(widened)}, which the "
                "parent does not",
            )
        )
        return
    if len(child_names) < len(parent_names):
        narrowings.append(
            InheritanceFinding(
                f"{code_stem}_RESTRICTED",
                locus,
                f"child's {field_name} drops "
                f"{', '.join(sorted(set(parent_names) - set(child_names)))}",
            )
        )


def _compare_metadata(
    *,
    locus: str,
    child_metadata: tuple[tuple[str, str], ...],
    parent_metadata: tuple[tuple[str, str], ...],
    redefinitions: list[InheritanceFinding],
    narrowings: list[InheritanceFinding],
) -> None:
    """Compare annotations: a child may add, but not overwrite or drop."""

    child_map = dict(child_metadata)
    for key, value in parent_metadata:
        if key not in child_map:
            redefinitions.append(
                InheritanceFinding(
                    "METADATA_KEY_DROPPED",
                    locus,
                    f"parent annotates {key!r}; the child does not carry it",
                )
            )
        elif child_map[key] != value:
            redefinitions.append(
                InheritanceFinding(
                    "METADATA_VALUE_REDEFINED",
                    locus,
                    f"{key!r} is {child_map[key]!r} in the child and {value!r} "
                    "in the parent",
                )
            )
    parent_keys = {key for key, _ in parent_metadata}
    added = sorted(key for key in child_map if key not in parent_keys)
    if added:
        narrowings.append(
            InheritanceFinding(
                "METADATA_ADDED",
                locus,
                f"child adds annotation(s) {', '.join(repr(key) for key in added)}",
            )
        )


def compare_axes(
    child: CoordinateAxis, parent: CoordinateAxis, *, locus: str
) -> InheritanceReport:
    """Compare one child axis against the parent axis it corresponds to."""

    if type(child) is not CoordinateAxis or type(parent) is not CoordinateAxis:
        raise TypeError("compare_axes takes CoordinateAxis values")

    redefinitions: list[InheritanceFinding] = []
    undecidable: list[InheritanceFinding] = []
    narrowings: list[InheritanceFinding] = []

    for name in AXIS_IDENTITY_FIELDS:
        child_value = getattr(child, name)
        parent_value = getattr(parent, name)
        if child_value != parent_value:
            redefinitions.append(
                InheritanceFinding(
                    "AXIS_SEMANTICS_REDEFINED",
                    f"{locus}.{name}",
                    f"child declares {child_value!r} where the parent declares "
                    f"{parent_value!r}",
                )
            )

    _compare_labels(
        locus=locus,
        child=child,
        parent=parent,
        redefinitions=redefinitions,
        undecidable=undecidable,
        narrowings=narrowings,
    )
    _compare_bounds(
        locus=locus,
        child=child,
        parent=parent,
        redefinitions=redefinitions,
        undecidable=undecidable,
        narrowings=narrowings,
    )
    _compare_optional_constraint(
        field_name="resolution",
        locus=locus,
        child_value=child.resolution,
        parent_value=parent.resolution,
        redefinitions=redefinitions,
        narrowings=narrowings,
    )
    if (
        child.resolution is not None
        and parent.resolution is not None
        and child.resolution != parent.resolution
    ):
        # Not a narrowing in either direction. A resolution fixes which
        # coordinates exist on the axis, so a different one is a different
        # axis — §11.6's "resolution variant" tier sets a resolution the
        # parent left open, which is the case handled above.
        redefinitions.append(
            InheritanceFinding(
                "RESOLUTION_REDEFINED",
                locus,
                f"child declares {child.resolution!r} where the parent declares "
                f"{parent.resolution!r}; a resolution decides which coordinates "
                "exist, so replacing one is not a restriction of it",
            )
        )
    _compare_name_set(
        code_stem="TRANSFORM_POLICY",
        field_name="transform_policy",
        locus=locus,
        child_names=child.transform_policy,
        parent_names=parent.transform_policy,
        redefinitions=redefinitions,
        narrowings=narrowings,
    )
    _compare_name_set(
        code_stem="INTERPOLATION_POLICY",
        field_name="interpolation_policy",
        locus=locus,
        child_names=child.interpolation_policy,
        parent_names=parent.interpolation_policy,
        redefinitions=redefinitions,
        narrowings=narrowings,
    )
    _compare_metadata(
        locus=locus,
        child_metadata=child.metadata,
        parent_metadata=parent.metadata,
        redefinitions=redefinitions,
        narrowings=narrowings,
    )
    return InheritanceReport(
        redefinitions=tuple(redefinitions),
        undecidable=tuple(undecidable),
        narrowings=tuple(narrowings),
    )


def compare_spaces(
    child: CoordinateSpace, parent: CoordinateSpace
) -> InheritanceReport:
    """Compare a child coordinate space against its parent's.

    Axes correspond by ``axis_id`` **in declared order**. Both halves matter:
    correspondence by name means renaming an axis reads as removing one and
    adding another rather than as an unrelated pair silently compared, and
    requiring the order to agree means a reordering is reported instead of being
    matched away — §11.5 derives a cell address from axis position, so a child
    that reorders its parent's axes addresses different cells with the same
    coordinates.
    """

    if type(child) is not CoordinateSpace or type(parent) is not CoordinateSpace:
        raise TypeError("compare_spaces takes CoordinateSpace values")

    redefinitions: list[InheritanceFinding] = []
    undecidable: list[InheritanceFinding] = []
    narrowings: list[InheritanceFinding] = []

    for name in SPACE_IDENTITY_FIELDS:
        child_value = getattr(child, name)
        parent_value = getattr(parent, name)
        if child_value != parent_value:
            redefinitions.append(
                InheritanceFinding(
                    _SPACE_FIELD_CODES.get(name, "SPACE_SEMANTICS_REDEFINED"),
                    name,
                    f"child declares {child_value!r} where the parent declares "
                    f"{parent_value!r}",
                )
            )

    dropped = tuple(
        axis.axis_id
        for axis in parent.axes
        if axis.axis_id not in {item.axis_id for item in child.axes}
    )
    added = tuple(
        axis.axis_id
        for axis in child.axes
        if axis.axis_id not in {item.axis_id for item in parent.axes}
    )
    if dropped or added:
        redefinitions.append(
            InheritanceFinding(
                "AXIS_SET_REDEFINED",
                "axes",
                f"child drops {dropped or ()} and adds {added or ()}; a child "
                "narrows an axis, it does not change which axes exist",
            )
        )
    elif tuple(axis.axis_id for axis in child.axes) != tuple(
        axis.axis_id for axis in parent.axes
    ):
        redefinitions.append(
            InheritanceFinding(
                "AXIS_ORDER_REDEFINED",
                "axes",
                f"child orders axes {tuple(axis.axis_id for axis in child.axes)} "
                f"against the parent's "
                f"{tuple(axis.axis_id for axis in parent.axes)}",
            )
        )
    else:
        for child_axis, parent_axis in zip(child.axes, parent.axes):
            axis_report = compare_axes(
                child_axis, parent_axis, locus=f"axes[{child_axis.axis_id}]"
            )
            redefinitions.extend(axis_report.redefinitions)
            undecidable.extend(axis_report.undecidable)
            narrowings.extend(axis_report.narrowings)

    missing_atlases = tuple(
        ref for ref in parent.region_atlas_refs if ref not in set(child.region_atlas_refs)
    )
    if missing_atlases:
        redefinitions.append(
            InheritanceFinding(
                "REGION_ATLAS_DROPPED",
                "region_atlas_refs",
                f"child drops {len(missing_atlases)} atlas reference(s) the "
                f"parent declares, beginning with {missing_atlases[0]}",
            )
        )
    elif len(child.region_atlas_refs) > len(parent.region_atlas_refs):
        narrowings.append(
            InheritanceFinding(
                "REGION_ATLAS_ADDED",
                "region_atlas_refs",
                f"child adds {len(child.region_atlas_refs) - len(parent.region_atlas_refs)} "
                "atlas reference(s)",
            )
        )
    _compare_metadata(
        locus="metadata",
        child_metadata=child.metadata,
        parent_metadata=parent.metadata,
        redefinitions=redefinitions,
        narrowings=narrowings,
    )
    return InheritanceReport(
        redefinitions=tuple(redefinitions),
        undecidable=tuple(undecidable),
        narrowings=tuple(narrowings),
    )


def uncompared_semantics() -> tuple[str, ...]:
    """What a clean `compare_spaces` still does not establish.

    Every entry names a mechanism this build does not have, not a comparison it
    forgot. The list shrinks when the mechanism arrives, which is why it is a
    function rather than prose in a docstring.
    """

    return (
        "REGION_ATLAS_CONTENTS_UNRESOLVED: atlas references are compared as "
        "opaque strings; no atlas store exists, so retaining a parent's "
        "reference is a byte claim rather than a claim about the regions it "
        "names",
        "QUALITY_POLICY_ABSENT: §22.1's metric profile, quality policy, "
        "remapping policy and uncertainty policy have no representation, so "
        "there is nothing to inherit or narrow",
        "PAYLOAD_UNREAD: this compares declarations. A child whose payload "
        "violates its own declared bounds is not detected here, and nothing in "
        "this module reads a payload",
    )
