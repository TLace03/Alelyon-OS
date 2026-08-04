"""Replay checking for committed Lattice transform chains.

This module deliberately does not import the compatibility analyser. It starts
from canonical bytes and a content-addressed coordinate-space table, recovers
the chain through the strict decoder, and re-executes it. Nothing here can
consult, re-run or trust the search that proposed the chain in the first place.

What a passing report establishes:

* The supplied bytes hash to the reference the caller pinned, so the committed
  chain has not been revised since that reference was recorded.
* The bytes are the canonical encoding of the chain they decode to, so no second
  byte string can present the same chain.
* Every reconstructed transform satisfies its own contract invariants, because
  decoding rebuilds the records through their ordinary constructors.
* Re-executing the chain on the supplied target coordinates reproduces the
  claimed source coordinates exactly.

What it does not establish:

* Nothing about who issued the chain. There is no key, signature or witness in
  this path, so an attacker who can replace both the bytes and the pinned
  reference is not detected here. An authenticated claim needs a public key
  pinned out of band, which this module does not implement.
* Nothing about whether the chain is the *correct* registration for any dataset.
  A semantically wrong but internally consistent chain replays cleanly.
* Nothing about payload values. No artifact is read and no value is remapped.
* This is not an independently written second implementation. It shares the
  contract and transform modules with the producer, so a defect in those modules
  would be reproduced rather than caught.

Nothing here issues or checks a Registration Certificate, and no optimality is
proven or claimed. `certificate.py` builds on this module rather than the reverse:
it delegates every chain replay here so that one implementation does that work,
and adds the signature and the certificate's own declarations on top.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac

from alelyon.runtime.vector.lattice.canonical import (
    CanonicalEncodingError,
    ContractViolationError,
    coordinate_space_ref,
    read_transform_chain,
    transform_chain_bytes,
    transform_chain_ref,
)
from alelyon.runtime.vector.lattice.contracts import CoordinateSpace
from alelyon.runtime.vector.lattice.transforms import (
    LOSS_CLASS_RANK,
    Invertibility,
    LossClass,
    TransformChain,
)


MAX_REPLAY_CASES = 4096


class ReplayCode(str, Enum):
    REPLAY_MATCHED = "REPLAY_MATCHED"
    CHAIN_COMMITMENT_MISMATCH = "CHAIN_COMMITMENT_MISMATCH"
    MALFORMED_ENCODING = "MALFORMED_ENCODING"
    NONCANONICAL_ENCODING = "NONCANONICAL_ENCODING"
    COORDINATE_REJECTED = "COORDINATE_REJECTED"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    LOSS_POLICY_NOT_SATISFIED = "LOSS_POLICY_NOT_SATISFIED"
    RESOURCE_BUDGET_EXCEEDED = "RESOURCE_BUDGET_EXCEEDED"


@dataclass(frozen=True, slots=True)
class ReplayReport:
    """The outcome of replaying one committed chain."""

    code: ReplayCode
    explanation: str
    chain_ref: str | None = None
    target_space_ref: str | None = None
    source_space_ref: str | None = None
    transform_types: tuple[str, ...] = ()
    #: The `space_id` of each space *between* two stages, in order — one per
    #: join, so a single-stage chain has none. Reported rather than judged: this
    #: module cannot say whether a derived space is the one some planner would
    #: have produced, because it does not import the planner and will not start.
    #: A caller that holds the two declared ends can decide that, and
    #: `certificate.py` does.
    intermediate_space_ids: tuple[str, ...] = ()
    loss_class: LossClass | None = None
    invertibility: Invertibility | None = None
    cases_replayed: int = 0
    failing_constraint: str | None = None
    evidence: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.code, ReplayCode):
            raise TypeError("code must be a ReplayCode")
        if self.code is ReplayCode.REPLAY_MATCHED:
            if self.failing_constraint is not None:
                raise ValueError("a matched replay cannot name a failing constraint")
            if self.chain_ref is None or self.loss_class is None:
                raise ValueError("a matched replay must report what it verified")
        elif self.failing_constraint is None:
            raise ValueError("a replay refusal must name failing_constraint")

    @property
    def verified(self) -> bool:
        return self.code is ReplayCode.REPLAY_MATCHED


def _refusal(
    code: ReplayCode,
    explanation: str,
    *,
    failing_constraint: str,
    evidence: tuple[str, ...] = (),
    **facts: object,
) -> ReplayReport:
    return ReplayReport(
        code=code,
        explanation=explanation,
        failing_constraint=failing_constraint,
        evidence=evidence,
        **facts,  # type: ignore[arg-type]
    )


def verify_transform_chain(
    chain_bytes: bytes,
    *,
    expected_chain_ref: str,
    spaces: Mapping[str, CoordinateSpace],
    replay_cases: Iterable[tuple[tuple[object, ...], tuple[object, ...]]] = (),
    maximum_loss_class: LossClass | None = None,
) -> ReplayReport:
    """Replay a committed chain against its pinned reference and claimed outputs.

    ``replay_cases`` pairs a target coordinate with the source coordinate the
    producer claims the chain maps it to. Read the module docstring for the exact
    boundary of what a ``REPLAY_MATCHED`` report does and does not establish.
    """

    if not isinstance(chain_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("chain_bytes must be bytes")
    if not isinstance(expected_chain_ref, str):
        raise TypeError("expected_chain_ref must be a string")
    chain_bytes = bytes(chain_bytes)

    actual_ref = "sha256:" + hashlib.sha256(chain_bytes).hexdigest()
    if not hmac.compare_digest(actual_ref, expected_chain_ref):
        return _refusal(
            ReplayCode.CHAIN_COMMITMENT_MISMATCH,
            "the supplied bytes do not hash to the pinned chain reference",
            failing_constraint="chain_commitment",
            evidence=(f"expected:{expected_chain_ref}", f"actual:{actual_ref}"),
        )

    try:
        # strict=False deliberately: the canonicality check below is this
        # module's to make and to name. Letting the reader refuse would report
        # non-canonical bytes as MALFORMED_ENCODING, losing the distinction a
        # replay report exists to draw.
        chain = read_transform_chain(chain_bytes, spaces, strict=False)
    except ContractViolationError as exc:
        # Bytes that parsed, into a record its own type refused. Ordered before
        # CanonicalEncodingError because it is a subclass of it; reversing these
        # two arms silently reports every contract failure as a parse failure.
        return _refusal(
            ReplayCode.MALFORMED_ENCODING,
            f"the recovered chain violates its own contract: {exc}",
            failing_constraint="contract_invariant",
            chain_ref=actual_ref,
        )
    except CanonicalEncodingError as exc:
        return _refusal(
            ReplayCode.MALFORMED_ENCODING,
            f"the committed bytes could not be recovered: {exc}",
            failing_constraint="canonical_decoding",
            chain_ref=actual_ref,
        )
    except (TypeError, ValueError) as exc:
        # Backstop for a contract refusal raised outside `_build`. Reachable only
        # if a construction site is added without going through it.
        return _refusal(
            ReplayCode.MALFORMED_ENCODING,
            f"the recovered chain violates its own contract: {exc}",
            failing_constraint="contract_invariant",
            chain_ref=actual_ref,
        )

    # Canonicality: exactly one byte string may present this chain.
    if transform_chain_bytes(chain) != chain_bytes:
        return _refusal(
            ReplayCode.NONCANONICAL_ENCODING,
            "the committed bytes are not the canonical encoding of the chain "
            "they decode to",
            failing_constraint="canonical_encoding",
            chain_ref=actual_ref,
        )

    facts = _chain_facts(chain, actual_ref)

    if maximum_loss_class is not None:
        if not isinstance(maximum_loss_class, LossClass):
            raise TypeError("maximum_loss_class must be a LossClass")
        if LOSS_CLASS_RANK[chain.loss_class] > LOSS_CLASS_RANK[maximum_loss_class]:
            return _refusal(
                ReplayCode.LOSS_POLICY_NOT_SATISFIED,
                f"the chain declares {chain.loss_class.value}, weaker than the "
                f"requested maximum {maximum_loss_class.value}",
                failing_constraint="loss_policy",
                **facts,
            )

    cases = []
    for index, case in enumerate(replay_cases):
        if index >= MAX_REPLAY_CASES:
            return _refusal(
                ReplayCode.RESOURCE_BUDGET_EXCEEDED,
                f"replay_cases exceeds the {MAX_REPLAY_CASES}-case limit",
                failing_constraint="replay_case_budget",
                **facts,
            )
        if type(case) is not tuple or len(case) != 2:
            raise TypeError("each replay case must be a two-item tuple")
        cases.append(case)

    for index, (target_coordinate, claimed_source) in enumerate(cases):
        try:
            replayed = chain.apply_coordinates(target_coordinate)
        except (TypeError, ValueError) as exc:
            return _refusal(
                ReplayCode.COORDINATE_REJECTED,
                f"replay case {index} was refused by the chain: {exc}",
                failing_constraint="coordinate_domain",
                evidence=(f"case:{index}",),
                cases_replayed=index,
                **facts,
            )
        if replayed != claimed_source:
            return _refusal(
                ReplayCode.REPLAY_MISMATCH,
                f"replay case {index} produced a different source coordinate "
                "than the producer claimed",
                failing_constraint="claimed_output",
                evidence=(
                    f"case:{index}",
                    f"replayed:{replayed!r}",
                    f"claimed:{claimed_source!r}",
                ),
                cases_replayed=index,
                **facts,
            )

    return ReplayReport(
        code=ReplayCode.REPLAY_MATCHED,
        explanation=(
            "the committed bytes are canonical, hash to the pinned reference, "
            "satisfy every contract invariant, and reproduce each claimed source "
            "coordinate"
        ),
        cases_replayed=len(cases),
        **facts,  # type: ignore[arg-type]
    )


def _chain_facts(chain: TransformChain, chain_ref: str) -> dict[str, object]:
    return {
        "chain_ref": chain_ref,
        "target_space_ref": coordinate_space_ref(chain.target_space),
        "source_space_ref": coordinate_space_ref(chain.source_space),
        "transform_types": tuple(
            transform.transform_type for transform in chain.transforms
        ),
        "intermediate_space_ids": tuple(
            transform.source_space.space_id for transform in chain.transforms[:-1]
        ),
        "loss_class": chain.loss_class,
        "invertibility": chain.invertibility,
    }


def chain_commitment(chain: TransformChain) -> tuple[bytes, str]:
    """Return the canonical bytes and reference a verifier should be given."""

    return transform_chain_bytes(chain), transform_chain_ref(chain)
