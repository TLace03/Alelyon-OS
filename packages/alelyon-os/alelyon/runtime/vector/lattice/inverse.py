"""Measured inverse consistency for an exact Lattice transform chain.

`TransformChain.invertibility` is a *structural* declaration: it reads EXACT
because every member type declares EXACT as a class constant. Nothing in that
derivation ever applies a transform to a coordinate, and nothing ever composes
`invert()` with `apply_coordinates`. A chain whose `invert()` were wrong -- an
off-by-one in a permutation, a reciprocal that does not spell back, a label map
inverted onto the wrong side -- would still report EXACT.

This module takes the measurement that declaration does not: it derives a probe
set, maps each probe through the chain, maps the result back through
`chain.invert()`, and counts how many probes came back byte-identical. That is
the §25.1 `inverse_consistency_bounds` field, which
[ADR-0004](../../../../docs/architecture/adr/ADR-0004-lattice-registration-certificate.md)
carried as UNMEASURED because inversion was implemented and never exercised
against the chain that declares it.

What a measurement bounds, and what it cannot
---------------------------------------------
`EXACT_ON_EVERY_PROBE` says: over the probes this chain admitted, the inverse
recovered the original coordinate exactly. It bounds inverse consistency **from
below**, on a finite sample, exactly as fuzzing bounds robustness from below. It
is not a proof that the chain inverts on every coordinate in its domain, and no
field here claims one. A chain can round-trip every probe and still fail on a
coordinate the probe set never reached.

Two properties are what make the sample worth signing:

* **The probes are derived, not chosen.** `derive_probe_coordinates` is a pure
  function of the chain -- its target axes' declared scalar types, label domains
  and the offsets the chain itself declares. An issuer cannot pick coordinates
  that happen to round-trip, because the issuer does not pick them. A verifier
  holding the replayed chain regenerates the identical probe set and re-runs the
  measurement rather than reading the issuer's number.
* **A vacuous result is a distinct answer.** A chain that admits no probe reports
  `NO_PROBE_ADMITTED`, and one whose axes yield no probe at all reports
  `NO_PROBE_DERIVED`. Neither is reported as success. "Zero of zero probes
  failed" is the shape of a test that asserts nothing, and it is named here
  instead of being counted as a clean run.

What it still shares with the issuer is the implementation: the verifier calls
the same `invert()` and the same `apply_coordinates`. A defect *inside* those is
reproduced on both sides rather than caught, which is the same boundary
`certificate` and `verify` already state for the replay itself. What this does
catch is a chain whose declared invertibility is not borne out by its own
implementation on its own coordinates.

What the counts do *not* bind, and the trace does
-------------------------------------------------
A tally of "4 offered, 4 admitted, 4 recovered" says nothing about *which* four.
Two builds whose probe derivation had drifted apart would each produce that
tally, compare equal, and report a verified certificate while having executed
disjoint sets of coordinates -- the verifier vouching for a round trip it never
reproduced. `ExecutionTrace` closes that: it records every probe, its forward
image and what came back, and the certificate carries the hash of that record as
§25.1's `execution_trace_commitment`. The verifier regenerates the trace from
the chain it decoded and compares commitments, so a derivation that drifted is a
refusal rather than a silent agreement about a number.

Pillar note: pure core. This module imports from `canonical`, `contracts` and
`transforms` only -- no key, no signature, no filesystem, no clock. The
`canonical` import is the encoding primitives the trace commitment hashes over,
so a trace is committed by the same length-prefixed, domain-separated rules as
every other record rather than by a second spelling of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as fixed_timezone
from enum import Enum

from alelyon.runtime.vector.lattice.canonical import (
    CanonicalEncodingError,
    _content_ref,
    _domain,
    _sequence,
    _string,
)
from alelyon.runtime.vector.lattice.contracts import (
    CoordinateAxis,
    CoordinateSpace,
    ScalarType,
)
from alelyon.runtime.vector.lattice.transforms import (
    Invertibility,
    TimezoneTransform,
    TransformChain,
)


#: Domain separator for the execution trace commitment. Its own domain, so a
#: trace's bytes can never be read as any other record's.
EXECUTION_TRACE_DOMAIN = "alelyon.lattice.canonical.execution-trace/0.1"

#: The most probes one measurement will offer a chain. Round-robin construction
#: already keeps the count at the widest single axis rather than the product of
#: every axis, so this is a backstop against a space with a very large declared
#: label domain rather than the usual limit.
MAX_PROBES = 64

#: Instants the temporal probes are spelled from, as plain (year, month, day,
#: hour, minute, second) parts. Two rather than one so a chain that happens to be
#: correct on a single date is not measured only there; both are far from any
#: representable boundary, so a re-spelling at any declared offset stays in the
#: domain.
#:
#: Integers rather than `datetime` objects deliberately.
#: `test_the_commitment_path_reads_no_mutable_shared_state` refuses a module
#: constant holding any object whose type it cannot see through, and the
#: `datetime` is built per call from these parts instead.
_PROBE_INSTANT_PARTS = (
    (2026, 8, 1, 9, 30, 0),
    (1999, 12, 31, 23, 59, 59),
)

#: Per-scalar-type probe values for axes that declare no domain of their own.
#: Chosen to exercise sign and magnitude on the numeric line -- zero, both units,
#: and a non-integral value -- because the numeric rungs (unit affine, reference
#: basis, orientation) are the ones whose inverses carry arithmetic.
_INTEGER_PROBES = (0, 1, -1, 7)
_RATIONAL_PROBES = ("0", "1", "-1", "3/4")
_DECIMAL_PROBES = ("0", "1", "-1", "1.5")
_FLOAT_PROBES = (0.0, 1.0, -1.0, 0.5)
_DURATION_PROBES = ("P1D", "PT1H", "P0D")
_UUID_PROBES = (
    "00000000-0000-4000-8000-000000000000",
    "11111111-2222-4333-8444-555555555555",
)
_HASH_PROBES = (
    "sha256:" + "00" * 32,
    "sha256:" + "ab" * 32,
)


class InverseConsistencyCode(str, Enum):
    """The outcome of one measurement. Every member is a measured answer."""

    #: Every probe the chain admitted came back byte-identical.
    EXACT_ON_EVERY_PROBE = "EXACT_ON_EVERY_PROBE"
    #: At least one admitted probe did not survive the round trip. The chain's
    #: declared invertibility is contradicted by its own implementation.
    RECOVERY_MISMATCH = "RECOVERY_MISMATCH"
    #: Probes were derived and the chain refused every one, so nothing was
    #: measured. Not a failure of inversion, and not a clean run either.
    NO_PROBE_ADMITTED = "NO_PROBE_ADMITTED"
    #: The target axes yielded no probe coordinate at all -- an axis declares a
    #: domain this module cannot spell a value in. Nothing was offered.
    NO_PROBE_DERIVED = "NO_PROBE_DERIVED"
    #: The chain does not declare EXACT invertibility, so there is no claim to
    #: measure against.
    NOT_DECLARED_INVERTIBLE = "NOT_DECLARED_INVERTIBLE"


@dataclass(frozen=True, slots=True)
class InverseConsistency:
    """The §25.1 `inverse_consistency_bounds` value: a sample, with its size."""

    code: InverseConsistencyCode
    probes_offered: int
    probes_admitted: int
    probes_recovered: int

    def __post_init__(self) -> None:
        if not isinstance(self.code, InverseConsistencyCode):
            raise TypeError("code must be an InverseConsistencyCode")
        for name in ("probes_offered", "probes_admitted", "probes_recovered"):
            value = getattr(self, name)
            if type(value) is not int or isinstance(value, bool):
                raise TypeError(f"{name} must be an int")
            if not 0 <= value <= MAX_PROBES:
                raise ValueError(f"{name} is outside the 0..{MAX_PROBES} range")
        if not self.probes_recovered <= self.probes_admitted <= self.probes_offered:
            raise ValueError(
                "probe counts must narrow: recovered <= admitted <= offered"
            )
        # The code and the counts are two spellings of one outcome, and a record
        # carrying a code its own counts contradict would be the defect this
        # field exists to prevent: a reader who trusts the code would be told
        # something the numbers deny.
        expected = _code_for(
            self.probes_offered, self.probes_admitted, self.probes_recovered
        )
        if self.code is not InverseConsistencyCode.NOT_DECLARED_INVERTIBLE:
            if self.code is not expected:
                raise ValueError(
                    f"code {self.code.value} contradicts its counts "
                    f"({self.probes_recovered}/{self.probes_admitted} recovered "
                    f"of {self.probes_offered} offered), which say "
                    f"{expected.value}"
                )
        elif self.probes_admitted or self.probes_recovered:
            raise ValueError(
                "NOT_DECLARED_INVERTIBLE means nothing was round-tripped, so no "
                "probe may be recorded as admitted or recovered"
            )

    @property
    def measured(self) -> bool:
        """Whether any probe was actually round-tripped.

        False for both vacuous outcomes. A caller must not read
        ``probes_recovered == probes_admitted`` as a clean result without this,
        because zero equals zero.
        """

        return self.probes_admitted > 0


def _code_for(offered: int, admitted: int, recovered: int) -> InverseConsistencyCode:
    if offered == 0:
        return InverseConsistencyCode.NO_PROBE_DERIVED
    if admitted == 0:
        return InverseConsistencyCode.NO_PROBE_ADMITTED
    if recovered == admitted:
        return InverseConsistencyCode.EXACT_ON_EVERY_PROBE
    return InverseConsistencyCode.RECOVERY_MISMATCH


def _declared_target_offsets(chain: TransformChain) -> tuple[int, ...]:
    """Every UTC offset any timezone stage in the chain declares on its target.

    A TIMESTAMP coordinate carries its own offset and a `TimezoneTransform`
    refuses one that contradicts its declaration, so a probe spelled at an
    arbitrary offset would be refused rather than measured. The offsets are read
    from the chain instead of resolved from a zone name, because resolving one
    would consult the IANA database that
    [ADR-0006](../../../../docs/architecture/adr/ADR-0006-lattice-declared-timezone-offsets.md)
    forbids this slice from reading.

    Every stage's target offsets are offered, not only the first stage's: in a
    composed chain an earlier stage can move an axis, so which offset a given
    timestamp axis needs is not decidable positionally here. Offering all of them
    lets the chain itself decide, and a probe it refuses is counted as refused.
    """

    offsets = {0}
    for transform in chain.transforms:
        if type(transform) is TimezoneTransform:
            for offset in transform.offsets:
                offsets.add(offset.target_offset_minutes)
    return tuple(sorted(offsets))


def _instant_probes(offsets: tuple[int, ...]) -> tuple[str, ...]:
    spellings = []
    for parts in _PROBE_INSTANT_PARTS:
        for minutes in offsets:
            zone = fixed_timezone(timedelta(minutes=minutes))
            spellings.append(datetime(*parts, tzinfo=zone).isoformat())
    return tuple(spellings)


def _axis_probes(axis: CoordinateAxis, offsets: tuple[int, ...]) -> tuple[object, ...]:
    """The values this axis can be probed at, derived from its declaration.

    Returns an empty tuple when the axis declares a domain no value can be
    spelled in -- a LABEL axis whose labels are committed by reference only, for
    instance. That propagates to NO_PROBE_DERIVED rather than to a silent
    substitution of some value the axis never admitted.
    """

    scalar_type = axis.scalar_type
    if scalar_type is ScalarType.INTEGER:
        return _INTEGER_PROBES
    if scalar_type is ScalarType.RATIONAL:
        return _RATIONAL_PROBES
    if scalar_type is ScalarType.DECIMAL:
        return _DECIMAL_PROBES
    if scalar_type is ScalarType.FLOAT:
        return _FLOAT_PROBES
    if scalar_type is ScalarType.TIMESTAMP:
        return _instant_probes(offsets)
    if scalar_type is ScalarType.DURATION:
        return _DURATION_PROBES
    if scalar_type is ScalarType.LABEL:
        # The committed domain is the only source of a legal label; inventing one
        # would be refused by the axis, and picking none would make every label
        # axis unmeasurable.
        return tuple(axis.labels) if axis.labels else ()
    if scalar_type is ScalarType.UUID:
        return _UUID_PROBES
    if scalar_type is ScalarType.HASH:
        return _HASH_PROBES
    return ()  # pragma: no cover - every ScalarType member is covered above


def derive_probe_coordinates(
    chain: TransformChain,
) -> tuple[tuple[object, ...], ...]:
    """Derive the probe coordinates for a chain. Pure, total and deterministic.

    The probes are combined round-robin rather than as a cartesian product: probe
    *k* takes each axis's *k*-th declared value, cycling the shorter axes. That
    visits every value of every axis in as many probes as the widest axis has
    values, instead of the product of all of them, and -- unlike holding one base
    coordinate fixed and varying a single axis -- it does not lose the whole
    measurement when the base happens to be a coordinate the chain refuses.
    """

    if type(chain) is not TransformChain:
        raise TypeError("chain must be a TransformChain")
    space = chain.target_space
    if type(space) is not CoordinateSpace:      # unreachable via TransformChain
        raise TypeError("the chain's target space is not a CoordinateSpace")
    offsets = _declared_target_offsets(chain)
    per_axis = [_axis_probes(axis, offsets) for axis in space.axes]
    if any(not values for values in per_axis):
        # A coordinate needs a value on every axis, so one unspellable axis
        # leaves no probe at all rather than a partial one.
        return ()
    width = min(max(len(values) for values in per_axis), MAX_PROBES)
    return tuple(
        tuple(values[index % len(values)] for values in per_axis)
        for index in range(width)
    )


class ProbeOutcome(str, Enum):
    """What happened to one probe. Three outcomes, never folded together."""

    #: The chain declined the coordinate. A statement about the probe's place in
    #: the domain, not about inversion.
    REFUSED = "REFUSED"
    #: Applied forward and back, and the result was the probe.
    RECOVERED = "RECOVERED"
    #: Applied forward and back, and the result was something else.
    MISMATCH = "MISMATCH"


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    """One probe's journey: what went in, what came out of each direction.

    A `REFUSED` probe has no images, and a probe that completed has both. The
    constructor holds that shape rather than trusting the caller, because these
    records are hashed into a commitment and a malformed one would commit
    cleanly to a fiction.
    """

    probe: tuple[object, ...]
    outcome: ProbeOutcome
    forward: tuple[object, ...] | None = None
    recovered: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if type(self.probe) is not tuple:
            raise TypeError("probe must be a tuple")
        if not isinstance(self.outcome, ProbeOutcome):
            raise TypeError("outcome must be a ProbeOutcome")
        completed = self.outcome is not ProbeOutcome.REFUSED
        for name in ("forward", "recovered"):
            image = getattr(self, name)
            if completed and type(image) is not tuple:
                raise ValueError(
                    f"a {self.outcome.value} probe must carry its {name} image"
                )
            if not completed and image is not None:
                raise ValueError("a REFUSED probe has no images to carry")
        # The outcome is a claim about the recovered image, so it is checked
        # against that image rather than accepted alongside it.
        if completed and (self.recovered == self.probe) != (
            self.outcome is ProbeOutcome.RECOVERED
        ):
            raise ValueError(
                f"outcome {self.outcome.value} contradicts the recovered image"
            )


@dataclass(frozen=True, slots=True)
class ExecutionTrace:
    """Every probe execution, in order: the §25.1 `execution_trace_commitment`.

    `InverseConsistency` is a tally of this, and is *derived* from it by
    `summary()` rather than counted alongside it. A count that could be computed
    independently of the executions it describes could disagree with them; this
    one cannot.
    """

    #: False only when the chain never declared itself invertible, in which case
    #: nothing was run. Carried explicitly because "no probe was executed" is
    #: true of that case and of a chain whose axes spell no value, and the two
    #: must not commit to identical bytes.
    declared_invertible: bool
    executions: tuple[ProbeExecution, ...] = ()

    def __post_init__(self) -> None:
        if type(self.declared_invertible) is not bool:
            raise TypeError("declared_invertible must be a bool")
        if type(self.executions) is not tuple:
            raise TypeError("executions must be a tuple")
        if not all(type(item) is ProbeExecution for item in self.executions):
            raise TypeError("every execution must be a ProbeExecution")
        if not self.declared_invertible and self.executions:
            raise ValueError(
                "a chain that declared no invertibility was never run, so it "
                "cannot carry executions"
            )

    def summary(self) -> InverseConsistency:
        """Tally this trace. The only place `InverseConsistency` is built."""

        if not self.declared_invertible:
            return InverseConsistency(
                InverseConsistencyCode.NOT_DECLARED_INVERTIBLE, 0, 0, 0
            )
        offered = len(self.executions)
        admitted = sum(
            1 for item in self.executions
            if item.outcome is not ProbeOutcome.REFUSED
        )
        recovered = sum(
            1 for item in self.executions
            if item.outcome is ProbeOutcome.RECOVERED
        )
        return InverseConsistency(
            _code_for(offered, admitted, recovered), offered, admitted, recovered
        )

    def commitment(self) -> str:
        """The content reference these executions hash to."""

        return _content_ref(execution_trace_bytes(self))


def _coordinate_value_bytes(value: object) -> bytes:
    """Encode one coordinate value, tagged by type.

    The tag is what stops `1` and `"1"` -- distinct coordinates on distinct axis
    kinds -- from committing to identical bytes.

    A FLOAT axis is spelled with `float.hex()` rather than as a number, because
    `canonical`'s encoding is deliberately free of floating-point fields. The
    hex form is exact, locale-independent, round-trips through `float.fromhex`,
    and keeps `-0.0` distinct from `0.0` -- which is right for a record of what
    executed, whatever an equality test would later make of the two.
    """

    # `bool` is a subclass of `int`, so exact type checks rather than isinstance:
    # True must not silently commit as 1.
    if type(value) is int:
        return b"i" + _string(str(value))
    if type(value) is str:
        return b"s" + _string(value)
    if type(value) is float:
        return b"f" + _string(value.hex())
    raise CanonicalEncodingError(
        f"a coordinate value of type {type(value).__name__!r} has no canonical "
        f"spelling in an execution trace; add one deliberately rather than "
        f"letting it hash by repr"
    )


def _coordinate_bytes(coordinate: tuple[object, ...] | None) -> bytes:
    if coordinate is None:
        return b"\x00"
    return b"\x01" + _sequence(
        _coordinate_value_bytes(value) for value in coordinate
    )


def execution_trace_bytes(trace: ExecutionTrace) -> bytes:
    """Encode a trace for commitment. Order is content, not presentation."""

    if type(trace) is not ExecutionTrace:
        raise TypeError("trace must be an ExecutionTrace")
    return (
        _domain(EXECUTION_TRACE_DOMAIN)
        + (b"\x01" if trace.declared_invertible else b"\x00")
        + _sequence(
            _string(item.outcome.value)
            + _coordinate_bytes(item.probe)
            + _coordinate_bytes(item.forward)
            + _coordinate_bytes(item.recovered)
            for item in trace.executions
        )
    )


def execute_probes(chain: TransformChain) -> ExecutionTrace:
    """Round-trip every derived probe through the chain and its inverse.

    The chain maps target to source, so a probe is a *target* coordinate: it is
    applied forward to a source coordinate, then through `chain.invert()` back to
    the target space, and compared to the probe it started as. Equality is the
    coordinate tuple's own -- byte-identical strings, equal integers -- because
    an exact slice admits no tolerance and §25.1's bound here is a count, not a
    residual.

    Returns the whole execution rather than its tally. The tally is `summary()`.
    """

    if type(chain) is not TransformChain:
        raise TypeError("chain must be a TransformChain")
    if chain.invertibility is not Invertibility.EXACT:
        # Unreachable while every transform type in this slice declares EXACT,
        # which `test_every_transform_type_declares_exact_invertibility` asserts
        # rather than assumes. It fires when a non-exact transform arrives, so
        # whoever adds one decides what a bound over it would mean instead of
        # inheriting a round trip that silently measures the wrong thing.
        return ExecutionTrace(declared_invertible=False)

    inverse = chain.invert()
    executions = []
    for probe in derive_probe_coordinates(chain):
        try:
            forward = chain.apply_coordinates(probe)
            back = inverse.apply_coordinates(forward)
        except (TypeError, ValueError):
            executions.append(ProbeExecution(probe, ProbeOutcome.REFUSED))
            continue
        outcome = (
            ProbeOutcome.RECOVERED if back == probe else ProbeOutcome.MISMATCH
        )
        executions.append(ProbeExecution(probe, outcome, forward, back))
    return ExecutionTrace(declared_invertible=True, executions=tuple(executions))


def measure_inverse_consistency(chain: TransformChain) -> InverseConsistency:
    """Tally the probe round trip. A summary of `execute_probes`.

    Kept as the name the certificate and the replay report have always called,
    and now a thin derivation: the counts and the committed trace are two views
    of one execution rather than two executions that might differ.
    """

    return execute_probes(chain).summary()
