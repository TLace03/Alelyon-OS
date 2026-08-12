"""Canonical templates, and the registry that keeps a published one immutable.

§22.2 is one sentence — *a published template is immutable; any change creates a
new version* — and it is the reason this module is a registry rather than a
record. A dataclass can be frozen; only something that remembers what it already
published can refuse the second, different version of `v1`. §22.5's concern is
sharper than tidiness: a template decides how every artifact registered against
it is interpreted, so a template that changed under its own name would silently
reinterpret every result already derived from it.

What is checked here
--------------------
* **Immutability.** Re-publishing `(template_id, version)` with different bytes
  is refused. Re-publishing byte-identical content is not — that is the same
  template, and making it an error would punish idempotent callers for nothing.
* **A closed transform vocabulary.** `admissible_transforms` must name
  transforms this build implements. The vocabulary is derived from the
  `ExactTransform` union rather than restated, so a new rung joins it by
  existing. A template admitting a transform nobody implements would pass every
  review and then never apply.
* **The hierarchy has a shape.** §11.6 stacks base contract → domain family →
  application → study/cohort → resolution variant. A base contract has no
  parents; everything else has at least one.
* **A child narrows, and never widens.** §11.6 allows a child to narrow
  constraints and forbids it redefining parent semantics. Transform policy is a
  set, so "narrow" is exactly subset, and a child admitting a transform its
  parent forbids is refused by name.
* **The child's coordinate space is compared against the parent's**, field by
  classified field, by `inheritance.compare_spaces`. Redefining an axis's unit,
  calendar, timezone or ordering is refused; restricting a label domain, adding
  bounds or fixing a resolution the parent left open is admitted and reported.
  A difference that exists but cannot be *ordered* — two opaque bounds pairs —
  refuses rather than passes.

What is **not** checked, and would be mistaken for checked
----------------------------------------------------------
This compares **declarations**. Nothing here reads a payload, so a child whose
data violates its own declared bounds is not detected, and a retained region
atlas reference is a byte claim rather than a claim about the regions it names.
`unchecked_inheritance()` names what is left so a reader does not read
"published" as "reviewed".

Absent for want of mechanisms this engine does not have (§22.1's field list):
`metric_profile_ref`, `remapping_policy`, `quality_policy` and
`uncertainty_policy` all reference registries that do not exist, and are omitted
rather than carried empty — a field that is always `None` reads as a slot
somebody forgot to fill. `signatures` is omitted for a stronger reason: §22.5
asks for signed releases and role-based publication, and this registry is an
in-memory object with no key, no roles and no persistence. A `signatures` field
here would be the appearance of that control without the control.

§22.3 migrations and §22.4 population-derived templates are later phases and
have no representation here at all. `CreationMethod` admits only `DECLARED`, so
a cohort-derived template cannot be published by claiming to be one.
"""
from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, field
from enum import Enum
from typing import get_args

from alelyon.runtime.vector.lattice.contracts import (
    CoordinateSpace,
    _content_ref,
    _metadata,
    _name_set,
    _optional_text,
    _text,
)
from alelyon.runtime.vector.lattice.inheritance import (
    compare_spaces,
    uncompared_semantics,
)
from alelyon.runtime.vector.lattice.transforms import ExactTransform


CANONICAL_TEMPLATE_SCHEMA = "alelyon.lattice.canonical-template/0.1"

MAX_PARENT_TEMPLATE_REFS = 16


def _derive_transform_vocabulary() -> frozenset[str]:
    """Every ``transform_type`` this build can actually execute.

    Derived from the `ExactTransform` union rather than restated as a literal.
    A hand-maintained list is a second place for the truth to live, and the one
    that goes stale is always the list — a rung added to the union but missed
    here would be silently un-admittable by any template.
    """

    names: set[str] = set()
    for member in get_args(ExactTransform):
        for declared in dataclass_fields(member):
            if declared.name == "transform_type" and isinstance(declared.default, str):
                names.add(declared.default)
    return frozenset(names)


#: The closed vocabulary a template's `admissible_transforms` is checked against.
KNOWN_TRANSFORM_TYPES = _derive_transform_vocabulary()


class TemplateTier(str, Enum):
    """§11.6's hierarchy, in narrowing order."""

    BASE_CONTRACT = "BASE_CONTRACT"
    DOMAIN_FAMILY = "DOMAIN_FAMILY"
    APPLICATION = "APPLICATION"
    STUDY_COHORT = "STUDY_COHORT"
    RESOLUTION_VARIANT = "RESOLUTION_VARIANT"


class CreationMethod(str, Enum):
    """How a template came to exist.

    Only `DECLARED` is admissible. §22.4's population-derived construction needs
    a committed cohort, deterministic initialisation and iteration, a fixed
    stopping rule and bias diagnostics — none of which exist — so the enum has
    one member rather than several with four of them refused at construction.
    A second member arrives with the machinery that earns it.
    """

    DECLARED = "DECLARED"


class TemplateValidationError(ValueError):
    """A template cannot be published as described."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class CanonicalTemplate:
    """§22.1's `CanonicalTemplate`, bounded to what this slice can honour."""

    template_id: str
    version: str
    tier: TemplateTier
    coordinate_space: CoordinateSpace
    admissible_transforms: tuple[str, ...]
    parent_template_refs: tuple[str, ...] = ()
    region_atlas_refs: tuple[str, ...] = ()
    creation_method: CreationMethod = CreationMethod.DECLARED
    provenance: str | None = None
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    schema_version: str = CANONICAL_TEMPLATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "template_id",
            _text(self.template_id, "template_id", identifier=True, maximum=512),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, "version", identifier=True, maximum=128),
        )
        if type(self.tier) is not TemplateTier:
            raise TypeError("tier must be a TemplateTier")
        if type(self.coordinate_space) is not CoordinateSpace:
            raise TypeError("coordinate_space must be a CoordinateSpace")
        if type(self.creation_method) is not CreationMethod:
            raise TypeError("creation_method must be a CreationMethod")

        admissible = _name_set(
            self.admissible_transforms,
            "admissible_transforms",
            maximum=len(KNOWN_TRANSFORM_TYPES),
        )
        if not admissible:
            raise TemplateValidationError(
                "NO_ADMISSIBLE_TRANSFORM",
                "a template admitting nothing can never register an artifact",
            )
        unknown = tuple(name for name in admissible if name not in KNOWN_TRANSFORM_TYPES)
        if unknown:
            raise TemplateValidationError(
                "UNKNOWN_TRANSFORM_TYPE",
                f"{', '.join(unknown)} — this build implements "
                f"{', '.join(sorted(KNOWN_TRANSFORM_TYPES))}",
            )
        object.__setattr__(self, "admissible_transforms", admissible)

        parents = _name_set(
            self.parent_template_refs,
            "parent_template_refs",
            maximum=MAX_PARENT_TEMPLATE_REFS,
        )
        for ref in parents:
            _content_ref(ref, "parent_template_refs item")
        object.__setattr__(self, "parent_template_refs", parents)

        if self.tier is TemplateTier.BASE_CONTRACT:
            if parents:
                raise TemplateValidationError(
                    "BASE_CONTRACT_HAS_PARENT",
                    "a base contract is the root of §11.6's hierarchy",
                )
        elif not parents:
            raise TemplateValidationError(
                "DERIVED_TEMPLATE_HAS_NO_PARENT",
                f"tier {self.tier.value} narrows something, and must name it",
            )

        object.__setattr__(
            self,
            "region_atlas_refs",
            _name_set(self.region_atlas_refs, "region_atlas_refs", maximum=128),
        )
        object.__setattr__(
            self, "provenance", _optional_text(self.provenance, "provenance")
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if self.schema_version != CANONICAL_TEMPLATE_SCHEMA:
            raise ValueError(
                f"schema_version must be {CANONICAL_TEMPLATE_SCHEMA!r} "
                "for this implementation"
            )

    def admits(self, transform_type: str) -> bool:
        """Whether this template permits one transform type."""

        return transform_type in self.admissible_transforms

    def unchecked_inheritance(self) -> tuple[str, ...]:
        """Parent semantics a clean `narrows_parent` still does not establish.

        Named because an empty refusal list is easy to read as "the child is a
        valid narrowing of its parent", and it establishes something weaker.
        Every entry is now a *mechanism this build does not have* rather than a
        comparison it declines to make: axis semantics, label dictionaries,
        bounds, resolution and policy sets are compared, and the residue is
        delegated to `inheritance.uncompared_semantics()` so the list shrinks
        where the comparison grows instead of being maintained twice.
        """

        if not self.parent_template_refs:
            return ()
        return uncompared_semantics()


#: §11.6's tiers, ranked so "below" is a comparison rather than a convention.
TEMPLATE_TIER_ORDER = {
    TemplateTier.BASE_CONTRACT: 0,
    TemplateTier.DOMAIN_FAMILY: 1,
    TemplateTier.APPLICATION: 2,
    TemplateTier.STUDY_COHORT: 3,
    TemplateTier.RESOLUTION_VARIANT: 4,
}


def narrows_parent(
    child: CanonicalTemplate, parent: CanonicalTemplate
) -> tuple[str, ...]:
    """Report why ``child`` is not a valid narrowing of ``parent``.

    Three things are checked here — the template's own transform policy, its
    tier, and its coordinate space, the last by delegating to
    :func:`~alelyon.runtime.vector.lattice.inheritance.compare_spaces`. An empty
    result means each of those passed. What it still does not mean is listed by
    :meth:`CanonicalTemplate.unchecked_inheritance`, which now names mechanisms
    this build lacks rather than comparisons it skipped.

    A difference the space comparison can see but cannot *order* — two opaque
    bounds pairs, say — is reported here as a reason, not passed over. See that
    module's docstring for why an undecidable narrowing is refused rather than
    assumed.
    """

    reasons: list[str] = []
    widened = tuple(
        name
        for name in child.admissible_transforms
        if name not in parent.admissible_transforms
    )
    if widened:
        reasons.append(
            "TRANSFORM_POLICY_WIDENED: child admits "
            f"{', '.join(widened)}, which the parent forbids"
        )
    if TEMPLATE_TIER_ORDER[child.tier] <= TEMPLATE_TIER_ORDER[parent.tier]:
        reasons.append(
            f"TIER_NOT_BELOW_PARENT: {child.tier.value} does not sit under "
            f"{parent.tier.value} in §11.6's hierarchy"
        )
    reasons.extend(
        compare_spaces(child.coordinate_space, parent.coordinate_space).reasons()
    )
    return tuple(reasons)


class TemplateRegistry:
    """An in-memory, content-addressed store that enforces §22.2.

    Deliberately not persistent. Persistence is Atlas's (§7), and a registry
    that wrote its own store would be the parallel infrastructure §5 forbids.
    What this class owns is the *rule*, not the storage: a caller backing it
    with Atlas keeps the rule and gains durability.
    """

    __slots__ = ("_by_ref", "_by_version")

    def __init__(self) -> None:
        self._by_ref: dict[str, CanonicalTemplate] = {}
        self._by_version: dict[tuple[str, str], str] = {}

    def __len__(self) -> int:
        return len(self._by_ref)

    def __contains__(self, ref: object) -> bool:
        return ref in self._by_ref

    def publish(self, template: CanonicalTemplate) -> str:
        """Publish a template and return its content reference.

        Refuses to redefine an already-published `(template_id, version)`.
        Re-publishing byte-identical content returns the same reference and is
        not an error: it is the same template, and idempotent callers should not
        have to remember whether they already did it.
        """

        if type(template) is not CanonicalTemplate:
            raise TypeError("template must be a CanonicalTemplate")

        # Imported here, not at module scope: `canonical` imports this module
        # to encode a template, so a top-level import would be a cycle. The
        # registry is the only place that needs the encoder.
        from alelyon.runtime.vector.lattice.canonical import canonical_template_ref

        ref = canonical_template_ref(template)
        key = (template.template_id, template.version)
        existing = self._by_version.get(key)
        if existing is not None and existing != ref:
            raise TemplateValidationError(
                "TEMPLATE_VERSION_IMMUTABLE",
                f"{template.template_id}@{template.version} is already published "
                f"as {existing}; §22.2 requires a new version, not a new meaning",
            )

        for parent_ref in template.parent_template_refs:
            if parent_ref not in self._by_ref:
                raise TemplateValidationError(
                    "UNKNOWN_PARENT_TEMPLATE",
                    f"{parent_ref} is not published in this registry",
                )
            reasons = narrows_parent(template, self._by_ref[parent_ref])
            if reasons:
                raise TemplateValidationError(
                    "INVALID_NARROWING", "; ".join(reasons)
                )

        self._by_ref[ref] = template
        self._by_version[key] = ref
        return ref

    def resolve(self, ref: str) -> CanonicalTemplate:
        """Return a published template by content reference."""

        ref = _content_ref(ref, "ref")
        try:
            return self._by_ref[ref]
        except KeyError:
            raise KeyError(f"no template published at {ref}") from None

    def latest_ref(self, template_id: str, version: str) -> str:
        """Return the reference published for one ``(id, version)``."""

        template_id = _text(template_id, "template_id", identifier=True, maximum=512)
        version = _text(version, "version", identifier=True, maximum=128)
        try:
            return self._by_version[(template_id, version)]
        except KeyError:
            raise KeyError(f"no template published as {template_id}@{version}") from None

    def versions(self, template_id: str) -> tuple[str, ...]:
        """Every published version of one template id, in canonical order."""

        template_id = _text(template_id, "template_id", identifier=True, maximum=512)
        return tuple(
            sorted(
                version
                for (published_id, version) in self._by_version
                if published_id == template_id
            )
        )

    def ancestry(self, ref: str) -> tuple[str, ...]:
        """Every ancestor reference of a published template, nearest first.

        Iterative and visit-tracked. A registry cannot hold a cycle — a parent
        must be published before its child, and a template's reference covers
        its parents — but a traversal that assumed acyclicity would be relying
        on that argument rather than checking it.
        """

        start = _content_ref(ref, "ref")
        if start not in self._by_ref:
            raise KeyError(f"no template published at {start}")
        ordered: list[str] = []
        seen = {start}
        pending = list(self._by_ref[start].parent_template_refs)
        while pending:
            current = pending.pop(0)
            if current in seen:
                continue
            seen.add(current)
            ordered.append(current)
            parent = self._by_ref.get(current)
            if parent is not None:
                pending.extend(parent.parent_template_refs)
        return tuple(ordered)
