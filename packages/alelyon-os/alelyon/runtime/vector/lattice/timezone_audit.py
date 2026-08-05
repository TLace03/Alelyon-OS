"""Advisory authoring-time audit of declared timezone offsets.

ADR-0006 commits timezone conversions with *declared* fixed offsets and reads
no IANA database, because resolving ``America/New_York`` to ``-04:00`` on a
date is a lookup in external, versioned data that two machines need not hold
the same edition of. That leaves a named measurement untaken: whether a
declared offset is the one the named zone actually had at a given instant.
This module takes that measurement — at authoring time, against the one
edition this machine holds — and nothing else.

What a report establishes:

* Each declared ``(zone, offset)`` pair either agrees or disagrees with the
  zoneinfo edition this process consulted, at each supplied instant, and the
  report names which edition that was when it can be determined.
* A disagreement is real evidence: the producer is about to commit an offset
  the local database says the zone did not have at that instant — which is
  exactly what declaring across a daylight-saving change looks like.

What it does not establish:

* Nothing about the zone's true history. Agreement with one edition is not
  agreement with the zone; the edition itself is revised several times a year
  and may be wrong or stale. The verdict is named against the edition, never
  against the world.
* Nothing about any committed chain. No transform, encoding, commitment,
  replay or certificate consults this module — `test_no_timezone_database_is
  _consulted` continues to hold, and the replay of a committed conversion
  succeeds on a machine where this audit cannot run at all.
* Nothing when there is no database. A machine without zoneinfo data yields
  ``DATABASE_UNAVAILABLE``, and the report says it measured nothing rather
  than passing vacuously.

The committed record is the authority; this audit is a lamp, not a gate.
Wiring it into registration or replay would reintroduce the tzdata-edition
dependence ADR-0006 exists to forbid.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from enum import Enum
import importlib.metadata
import zoneinfo

from alelyon.runtime.vector.lattice.registration import DeclaredTimezoneConversion
from alelyon.runtime.vector.lattice.transforms import TimezoneTransform


MAX_AUDIT_INSTANTS = 4096

#: How the report's ``edition`` field was determined. ``TZDATA_DISTRIBUTION``
#: is the installed `tzdata` package's version; ``DECLARED`` is a caller
#: override; ``UNDETERMINED`` means zoneinfo answers from data whose edition
#: the standard library cannot name (a system tz directory), so the report is
#: a measurement against an unnamed edition and says so.
EDITION_SOURCES = ("TZDATA_DISTRIBUTION", "DECLARED", "UNDETERMINED")

ZoneResolver = Callable[[str], tzinfo]


class TimezoneAuditVerdict(str, Enum):
    OFFSET_MATCHES_EDITION = "OFFSET_MATCHES_EDITION"
    OFFSET_DIFFERS_FROM_EDITION = "OFFSET_DIFFERS_FROM_EDITION"
    ZONE_NOT_IN_EDITION = "ZONE_NOT_IN_EDITION"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class TimezoneAuditFinding:
    """One declared ``(zone, offset)`` pair measured at one instant."""

    verdict: TimezoneAuditVerdict
    #: Which declared pair this is about, e.g. ``axis 'time' target side``.
    subject: str
    zone: str
    declared_offset_minutes: int
    #: The instant measured, ISO-8601 with offset. ``None`` only for verdicts
    #: that are not instant-specific (an unknown zone is unknown at every
    #: instant).
    instant: str | None
    #: What the consulted edition resolves the zone to at that instant, in
    #: whole minutes; ``None`` when the zone could not be resolved or the
    #: edition's offset is not a whole number of minutes (named in ``detail``).
    edition_offset_minutes: int | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, TimezoneAuditVerdict):
            raise TypeError("verdict must be a TimezoneAuditVerdict")
        if not self.detail:
            raise ValueError("a finding must say what it found")
        if self.verdict is TimezoneAuditVerdict.OFFSET_MATCHES_EDITION:
            if self.instant is None or self.edition_offset_minutes is None:
                raise ValueError(
                    "a match must name the instant and the edition's offset"
                )
            if self.edition_offset_minutes != self.declared_offset_minutes:
                raise ValueError(
                    "a match cannot carry an edition offset that differs from "
                    "the declared one"
                )
        elif self.verdict is TimezoneAuditVerdict.OFFSET_DIFFERS_FROM_EDITION:
            if self.instant is None:
                raise ValueError("a divergence must name the instant")
            if self.edition_offset_minutes == self.declared_offset_minutes:
                raise ValueError(
                    "a divergence cannot carry an edition offset equal to the "
                    "declared one"
                )
        elif self.edition_offset_minutes is not None:
            raise ValueError(
                "an unresolved zone cannot carry an edition offset"
            )


@dataclass(frozen=True, slots=True)
class TimezoneAuditReport:
    """Every declared pair of one conversion, measured against one edition."""

    #: The zoneinfo edition consulted, e.g. ``tzdata-2026.1``; ``None`` when
    #: it could not be determined (see ``edition_source``).
    edition: str | None
    edition_source: str
    findings: tuple[TimezoneAuditFinding, ...]

    def __post_init__(self) -> None:
        if self.edition_source not in EDITION_SOURCES:
            raise ValueError(
                f"edition_source must be one of {EDITION_SOURCES}"
            )
        if self.edition_source == "UNDETERMINED":
            if self.edition is not None:
                raise ValueError(
                    "an undetermined edition cannot carry an edition name"
                )
        elif self.edition is None:
            raise ValueError(
                "a determined edition_source must name the edition"
            )
        findings = tuple(self.findings)
        if not findings:
            raise ValueError(
                "a report must carry at least one finding; an empty audit "
                "measured nothing and must not exist to be mistaken for one "
                "that passed"
            )
        if any(type(f) is not TimezoneAuditFinding for f in findings):
            raise TypeError(
                "findings must contain only TimezoneAuditFinding values"
            )
        object.__setattr__(self, "findings", findings)

    @property
    def measured(self) -> bool:
        """Whether a database was available to measure against at all."""

        return all(
            f.verdict is not TimezoneAuditVerdict.DATABASE_UNAVAILABLE
            for f in self.findings
        )

    @property
    def agrees_with_edition(self) -> bool:
        """Every declared pair matches the consulted edition at every instant.

        This is a statement about the consulted edition only — not about the
        zone's true history, and never about any committed chain.
        """

        return all(
            f.verdict is TimezoneAuditVerdict.OFFSET_MATCHES_EDITION
            for f in self.findings
        )


def audit_timezone_transform(
    transform: TimezoneTransform,
    instants: Iterable[str],
    *,
    resolver: ZoneResolver | None = None,
    edition: str | None = None,
) -> TimezoneAuditReport:
    """Measure a transform's declared offsets against this machine's zoneinfo.

    Every converted axis is measured on both sides: the target axis's zone
    against the declared target offset and the source axis's zone against the
    declared source offset, at each supplied instant. ``instants`` are
    ISO-8601 timestamps carrying their own offsets — normally the coordinates
    about to be registered. Read the module docstring for what a report can
    and cannot establish.
    """

    if type(transform) is not TimezoneTransform:
        raise TypeError("transform must be a TimezoneTransform")
    pairs = []
    for offset in transform.offsets:
        target_axis = transform.target_space.axes[offset.axis_index]
        source_axis = transform.source_space.axes[offset.axis_index]
        pairs.append((
            f"axis {target_axis.axis_id!r} target side",
            target_axis.timezone,
            offset.target_offset_minutes,
        ))
        pairs.append((
            f"axis {source_axis.axis_id!r} source side",
            source_axis.timezone,
            offset.source_offset_minutes,
        ))
    return _audit_pairs(pairs, instants, resolver, edition)


def audit_declared_conversion(
    declaration: DeclaredTimezoneConversion,
    instants: Iterable[str],
    *,
    resolver: ZoneResolver | None = None,
    edition: str | None = None,
) -> TimezoneAuditReport:
    """Measure a caller's declaration before a transform is even built."""

    if type(declaration) is not DeclaredTimezoneConversion:
        raise TypeError("declaration must be a DeclaredTimezoneConversion")
    pairs = [
        (
            f"declaration {declaration.semantic_id!r} target side",
            declaration.target_timezone,
            declaration.target_offset_minutes,
        ),
        (
            f"declaration {declaration.semantic_id!r} source side",
            declaration.source_timezone,
            declaration.source_offset_minutes,
        ),
    ]
    return _audit_pairs(pairs, instants, resolver, edition)


def _audit_pairs(
    pairs: list[tuple[str, str | None, int]],
    instants: Iterable[str],
    resolver: ZoneResolver | None,
    edition: str | None,
) -> TimezoneAuditReport:
    parsed = _parsed_instants(instants)
    if edition is not None:
        if type(edition) is not str or not edition:
            raise TypeError("edition must be a non-empty string")
        edition_source = "DECLARED"
    else:
        edition, edition_source = _detect_edition()
    if resolver is None:
        resolver = _zoneinfo_resolver
    findings = []
    for subject, zone, declared in pairs:
        if zone is None:
            # The transform contract requires a zone on every converted axis,
            # so this is reachable only through a record the contract would
            # itself refuse; measuring nothing must still be named.
            raise ValueError(f"{subject} declares no timezone to measure")
        findings.extend(
            _measure_pair(subject, zone, declared, parsed, resolver)
        )
    return TimezoneAuditReport(
        edition=edition,
        edition_source=edition_source,
        findings=tuple(findings),
    )


def _measure_pair(
    subject: str,
    zone: str,
    declared: int,
    parsed: tuple[tuple[str, datetime], ...],
    resolver: ZoneResolver,
) -> list[TimezoneAuditFinding]:
    try:
        resolved = resolver(zone)
    except KeyError:
        # `zoneinfo.ZoneInfoNotFoundError` is a KeyError; an injected resolver
        # signals an unknown zone the same way. Which refusal this is depends
        # on whether any database answered at all.
        if _database_present():
            return [TimezoneAuditFinding(
                verdict=TimezoneAuditVerdict.ZONE_NOT_IN_EDITION,
                subject=subject,
                zone=zone,
                declared_offset_minutes=declared,
                instant=None,
                edition_offset_minutes=None,
                detail=(
                    f"the consulted edition carries no zone named {zone!r}; "
                    "a non-IANA label is legal on an axis and cannot be "
                    "measured here"
                ),
            )]
        return [TimezoneAuditFinding(
            verdict=TimezoneAuditVerdict.DATABASE_UNAVAILABLE,
            subject=subject,
            zone=zone,
            declared_offset_minutes=declared,
            instant=None,
            edition_offset_minutes=None,
            detail=(
                "no zoneinfo data is available on this machine, so whether "
                f"{zone!r} had the declared offset is UNMEASURED here — "
                "which is not evidence that it did"
            ),
        )]
    findings = []
    for text, instant in parsed:
        actual = instant.astimezone(resolved).utcoffset()
        whole, remainder = divmod(actual, timedelta(minutes=1))
        edition_minutes = whole if remainder == timedelta(0) else None
        if edition_minutes == declared:
            findings.append(TimezoneAuditFinding(
                verdict=TimezoneAuditVerdict.OFFSET_MATCHES_EDITION,
                subject=subject,
                zone=zone,
                declared_offset_minutes=declared,
                instant=text,
                edition_offset_minutes=edition_minutes,
                detail=(
                    f"the consulted edition also resolves {zone!r} to "
                    f"{declared:+d} minutes at this instant"
                ),
            ))
        else:
            spelled = (
                f"{edition_minutes:+d} minutes"
                if edition_minutes is not None
                else f"{actual!r}, not a whole number of minutes"
            )
            findings.append(TimezoneAuditFinding(
                verdict=TimezoneAuditVerdict.OFFSET_DIFFERS_FROM_EDITION,
                subject=subject,
                zone=zone,
                declared_offset_minutes=declared,
                instant=text,
                edition_offset_minutes=edition_minutes,
                detail=(
                    f"the consulted edition resolves {zone!r} to {spelled} at "
                    f"this instant, not the declared {declared:+d} minutes; "
                    "an offset that moves across the registered interval is "
                    "what a daylight-saving change looks like"
                ),
            ))
    return findings


def _parsed_instants(
    instants: Iterable[str],
) -> tuple[tuple[str, datetime], ...]:
    parsed = []
    for index, text in enumerate(instants):
        if index >= MAX_AUDIT_INSTANTS:
            raise ValueError(
                f"instants exceeds the {MAX_AUDIT_INSTANTS}-instant audit "
                "budget"
            )
        if type(text) is not str:
            raise TypeError("each instant must be an ISO-8601 string")
        try:
            value = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                f"instant {text!r} is not an ISO-8601 timestamp: {exc}"
            ) from exc
        if value.utcoffset() is None:
            raise ValueError(
                f"instant {text!r} carries no UTC offset, so it names no "
                "absolute instant to measure a zone at"
            )
        parsed.append((text, value))
    if not parsed:
        raise ValueError(
            "an audit needs at least one instant; a zone's offset is a "
            "function of time"
        )
    return tuple(parsed)


def _zoneinfo_resolver(zone: str) -> tzinfo:
    # Late attribute access on purpose: the database is consulted when an
    # audit runs, never when this module is imported.
    return zoneinfo.ZoneInfo(zone)


def _database_present() -> bool:
    try:
        return bool(zoneinfo.available_timezones())
    except Exception:
        return False


def _detect_edition() -> tuple[str | None, str]:
    try:
        version = importlib.metadata.version("tzdata")
    except importlib.metadata.PackageNotFoundError:
        # zoneinfo may still answer from a system tz directory whose edition
        # the standard library cannot name.
        return None, "UNDETERMINED"
    return f"tzdata-{version}", "TZDATA_DISTRIBUTION"
