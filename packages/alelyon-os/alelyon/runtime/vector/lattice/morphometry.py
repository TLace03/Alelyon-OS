"""Model morphometry — a model's declared anatomy on a canonical lattice.

This is the first consumer of the registration core: it takes what a local model
runtime reports about a model it holds, arranges that inventory on a canonical
``(block, module)`` coordinate space, and emits per-cell measurements plus a
content commitment over the space it used.

**What is measured, and what is not.** Every figure here is derived from the
runtime's *declared* metadata — a tensor inventory (name, element type, shape)
or, when the runtime does not publish one, the architecture fields it does
publish. Nothing in this module reads a weight, runs a forward pass, or observes
an activation. So the fields it produces describe **architecture and storage
precision**, never learned behaviour, and the rendering built on top of them is
a picture of the model's *structure*. Calling that a scan of what the model
knows would be a claim nothing here supports.

**Two sources, never blended.** ``SOURCE_TENSOR_INVENTORY`` is the strong case:
each tensor's own shape and quantisation type, so per-cell parameter counts are
counted rather than inferred. ``SOURCE_DECLARED_ARCHITECTURE`` is the fallback:
per-role parameter counts computed analytically from block count, embedding
length, feed-forward length and head counts. The second is arithmetic over
declared dimensions and is correct only for the dense decoder shape it names —
so a mixture-of-experts model with no tensor inventory is **refused by name**
rather than approximated. ``ModelMorphometry.source`` always says which one ran,
and mixing them in one result is not possible.

**Bit widths are nominal.** ``NOMINAL_BITS_PER_WEIGHT`` is a static table of the
published bits-per-weight for each ``ggml`` element type. It is the format's
nominal cost, not a measurement of the file, and a type absent from the table
yields ``None`` and a named gap rather than a guess.

Pure and deterministic: no network, no filesystem, no clock, no Qt. The caller
supplies the payload; ``alelyon.frontend.desktop.lattice`` is what fetches it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import re

from alelyon.runtime.vector.lattice.canonical import coordinate_space_ref
from alelyon.runtime.vector.lattice.contracts import (
    AxisKind,
    CoordinateAxis,
    CoordinateOrdering,
    CoordinateSpace,
    ScalarType,
    TopologyType,
    label_dictionary_ref,
)
from alelyon.runtime.vector.lattice.certificate import (
    CertificateError,
    SignedRegistrationCertificate,
    issue_registration_certificate,
)
from alelyon.runtime.vector.lattice.registration import (
    CompatibilityReport,
    analyze_exact_compatibility,
)

MORPHOMETRY_SCHEMA = "alelyon.lattice.model-morphometry/0.1"

#: The canonical space every model is arranged on. Versioned because a change to
#: the module vocabulary below changes what a committed space means.
CANONICAL_SPACE_ID = "alelyon.model.transformer.canonical"
CANONICAL_SPACE_VERSION = "0.1.0"

SOURCE_TENSOR_INVENTORY = "TENSOR_INVENTORY"
SOURCE_DECLARED_ARCHITECTURE = "DECLARED_ARCHITECTURE"

#: Refusal reasons. A closed vocabulary, because each one names a different
#: thing the caller can do about it.
REFUSED_NO_METADATA = "NO_MODEL_METADATA"
REFUSED_NO_BLOCKS = "BLOCK_COUNT_NOT_DECLARED"
REFUSED_MIXTURE_OF_EXPERTS = "MIXTURE_OF_EXPERTS_WITHOUT_TENSOR_INVENTORY"
REFUSED_INSUFFICIENT_ARCHITECTURE = "ARCHITECTURE_FIELDS_INCOMPLETE"

#: Guards against an inventory large enough to make the UI unresponsive. These
#: are presentation budgets, not security boundaries — the contracts layer holds
#: the encoding limits.
MAX_TENSORS = 20_000
MAX_BLOCKS = 4_096


# ── the module vocabulary ────────────────────────────────────────────────────
#
# A module slot is (family, stage). The family is what the sub-module DOES; the
# stage is its position in execution order inside that family. Both are needed:
# grouping by family alone loses the ordering that makes a depth profile
# readable, and a flat ordinal loses the grouping that makes it interpretable.

FAMILY_EMBEDDING = "embedding"
FAMILY_NORM = "normalisation"
FAMILY_ATTENTION = "attention"
FAMILY_FEED_FORWARD = "feed-forward"
FAMILY_OUTPUT = "output"

#: Order is the y-axis order of the render, from input side to output side.
FAMILIES: tuple[str, ...] = (
    FAMILY_EMBEDDING,
    FAMILY_NORM,
    FAMILY_ATTENTION,
    FAMILY_FEED_FORWARD,
    FAMILY_OUTPUT,
)

#: ``module id -> (family, stage, human label)``. The module id is the canonical
#: axis label and is what the content commitment covers, so this table is part
#: of the space's meaning: adding a row changes ``CANONICAL_SPACE_VERSION``.
MODULES: dict[str, tuple[str, int, str]] = {
    "token_embd": (FAMILY_EMBEDDING, 0, "Token embedding"),
    "attn_norm": (FAMILY_NORM, 0, "Attention norm"),
    "attn_q_norm": (FAMILY_NORM, 1, "Query norm"),
    "attn_k_norm": (FAMILY_NORM, 2, "Key norm"),
    "ffn_norm": (FAMILY_NORM, 3, "Feed-forward norm"),
    "output_norm": (FAMILY_NORM, 4, "Output norm"),
    "attn_q": (FAMILY_ATTENTION, 0, "Query projection"),
    "attn_k": (FAMILY_ATTENTION, 1, "Key projection"),
    "attn_v": (FAMILY_ATTENTION, 2, "Value projection"),
    "attn_output": (FAMILY_ATTENTION, 3, "Attention output"),
    "ffn_gate": (FAMILY_FEED_FORWARD, 0, "Feed-forward gate"),
    "ffn_up": (FAMILY_FEED_FORWARD, 1, "Feed-forward up"),
    "ffn_down": (FAMILY_FEED_FORWARD, 2, "Feed-forward down"),
    "ffn_expert": (FAMILY_FEED_FORWARD, 3, "Expert bank"),
    "output": (FAMILY_OUTPUT, 0, "Output projection"),
    "other": (FAMILY_OUTPUT, 1, "Unclassified"),
}

#: Sorted so the axis labels — and therefore the committed bytes — do not depend
#: on dict insertion order.
MODULE_IDS: tuple[str, ...] = tuple(sorted(MODULES))

STAGE_COUNT = max(stage for _f, stage, _l in MODULES.values()) + 1

#: Tensor-name fragments that identify a module, longest first so ``attn_q_norm``
#: is not swallowed by ``attn_q``. Matching is on the ggml name segments, which
#: are stable across the llama-family architectures this table covers.
_NAME_TO_MODULE: tuple[tuple[str, str], ...] = (
    ("attn_q_norm", "attn_q_norm"),
    ("attn_k_norm", "attn_k_norm"),
    ("attn_norm", "attn_norm"),
    ("ffn_norm", "ffn_norm"),
    ("output_norm", "output_norm"),
    ("token_embd", "token_embd"),
    ("attn_output", "attn_output"),
    ("attn_out", "attn_output"),
    ("attn_q", "attn_q"),
    ("attn_k", "attn_k"),
    ("attn_v", "attn_v"),
    ("ffn_gate_exps", "ffn_expert"),
    ("ffn_up_exps", "ffn_expert"),
    ("ffn_down_exps", "ffn_expert"),
    ("ffn_gate_inp", "ffn_expert"),
    ("ffn_gate", "ffn_gate"),
    ("ffn_up", "ffn_up"),
    ("ffn_down", "ffn_down"),
    ("output", "output"),
)

_BLOCK_RE = re.compile(r"(?:^|\.)blk\.(\d{1,6})\.")

#: Tensor-name suffix marking a ROUTED expert stack -- one tensor holding every
#: expert's copy of a projection, of which a token reaches only
#: ``expert_used_count`` slices.
#:
#: Two neighbours are deliberately NOT here because both are always active:
#: the router (``ffn_gate_inp``), which is what decides the routing and must run
#: for every token, and shared experts (``ffn_*_shexp``), which every token uses
#: by definition. The naming table already sends ``_shexp`` tensors to the dense
#: ``ffn_gate``/``ffn_up``/``ffn_down`` modules, so they are counted as active
#: without a second rule -- asserted in the tests rather than left to luck.
_ROUTED_EXPERT_SUFFIX = "_exps"


def _is_routed_expert(name: str) -> bool:
    """True when this tensor holds routed expert weights.

    Matched on the ggml name segment rather than on the module, because the
    canonical space folds the router into the ``ffn_expert`` cell alongside the
    experts it selects. Splitting them at the tensor level keeps the arithmetic
    exact without changing what the committed coordinate space means.
    """
    for segment in str(name).split("."):
        if segment.endswith(_ROUTED_EXPERT_SUFFIX):
            return True
    return False

#: Published bits per weight for each ``ggml`` element type. NOMINAL: this is
#: what the format costs by construction, not a measurement of any file. A type
#: that is not here yields None and is named in ``gaps`` — an unknown precision
#: silently defaulted to 16 would put a wrong number under a real label.
NOMINAL_BITS_PER_WEIGHT: dict[str, Fraction] = {
    "F64": Fraction(64), "I64": Fraction(64),
    "F32": Fraction(32), "I32": Fraction(32),
    "F16": Fraction(16), "BF16": Fraction(16), "I16": Fraction(16),
    "Q8_0": Fraction(17, 2), "Q8_1": Fraction(9), "Q8_K": Fraction(17, 2),
    "I8": Fraction(8),
    "Q6_K": Fraction(105, 16),
    "Q5_0": Fraction(11, 2), "Q5_1": Fraction(6), "Q5_K": Fraction(11, 2),
    "Q4_0": Fraction(9, 2), "Q4_1": Fraction(5), "Q4_K": Fraction(9, 2),
    "IQ4_NL": Fraction(9, 2), "IQ4_XS": Fraction(17, 4),
    "Q3_K": Fraction(55, 16), "IQ3_S": Fraction(55, 16),
    "IQ3_XXS": Fraction(49, 16),
    "Q2_K": Fraction(21, 8), "IQ2_S": Fraction(5, 2),
    "IQ2_XS": Fraction(37, 16), "IQ2_XXS": Fraction(33, 16),
    "IQ1_M": Fraction(7, 4), "IQ1_S": Fraction(25, 16),
}


def module_of(tensor_name: str) -> str:
    """The module slot a ggml tensor name belongs to.

    Falls back to ``other`` rather than raising: an architecture this table does
    not know must still be counted somewhere, and a cell labelled
    ``Unclassified`` is visible in the render where a dropped tensor would not
    be.
    """
    name = str(tensor_name or "").lower()
    for fragment, module in _NAME_TO_MODULE:
        if fragment in name:
            return module
    return "other"


def block_of(tensor_name: str) -> int | None:
    """The transformer block index in a ggml tensor name, or None if it has no
    block — embeddings and the output head sit outside the stack."""
    match = _BLOCK_RE.search(str(tensor_name or ""))
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:  # pragma: no cover - the pattern guarantees digits
        return None


# ── records ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class TensorRecord:
    """One declared tensor, placed on the canonical lattice."""

    name: str
    module: str
    block: int | None
    shape: tuple[int, ...]
    element_type: str
    parameters: int

    @property
    def family(self) -> str:
        return MODULES.get(self.module, MODULES["other"])[0]

    @property
    def stage(self) -> int:
        return MODULES.get(self.module, MODULES["other"])[1]

    @property
    def bits_per_weight(self) -> Fraction | None:
        return NOMINAL_BITS_PER_WEIGHT.get(self.element_type.upper())

    @property
    def nominal_bytes(self) -> int | None:
        """Storage the format costs for this tensor, or None when the element
        type is not in the table."""
        bits = self.bits_per_weight
        if bits is None:
            return None
        return int(Fraction(self.parameters) * bits / 8)


def _active_path_gap(
    tensors: Sequence[TensorRecord],
    used: int | None,
    total: int | None,
) -> str | None:
    """Why routed active-parameter arithmetic is unavailable, if it is.

    GGML expert-bank tensors store the expert slices on their final axis.  A
    declared count is therefore not enough by itself: every routed tensor must
    expose an axis of exactly that length before dividing its parameters by the
    declaration.  Otherwise a contradictory payload would turn into an exact-
    looking number even though no partition in the inventory supports it.
    """
    routed = tuple(t for t in tensors if _is_routed_expert(t.name))
    if not routed:
        if used is not None or total is not None:
            return (
                "the runtime declares routing metadata "
                f"(expert_count={total}, expert_used_count={used}), but its "
                "tensor inventory contains no recognized routed _exps expert "
                "bank. Stored inventory counts remain measured; the active-"
                "parameter figure is UNMEASURED."
            )
        return None
    if used is None or total is None:
        return (
            "this model routes part of its feed-forward stack, and the runtime "
            "does not declare a complete expert_count/expert_used_count pair "
            f"(expert_count={total}, expert_used_count={used}). Parameter share "
            "is exact; the active-parameter figure is UNMEASURED and the "
            "totals here must not be read as the cost of one token."
        )
    if used <= 0 or total <= 0:
        return (
            f"the runtime declares non-positive routing counts "
            f"(expert_count={total}, expert_used_count={used}). Parameter share "
            "is exact; the active-parameter figure is UNMEASURED."
        )
    if used > total:
        return (
            f"the runtime declares expert_used_count={used}, greater than "
            f"expert_count={total}. Parameter share is exact; the active-"
            "parameter figure is UNMEASURED."
        )
    mismatched = sum(
        1 for tensor in routed
        if not tensor.shape or tensor.shape[-1] != total
    )
    if mismatched:
        return (
            f"the runtime declares expert_count={total}, but {mismatched} of "
            f"{len(routed)} routed tensor shape(s) do not end in an expert axis "
            f"of length {total}. Stored parameter counts still follow the "
            "inventory; the active-parameter figure is UNMEASURED."
        )
    return None


@dataclass(frozen=True, slots=True)
class Cell:
    """One ``(block, module)`` cell of the canonical space.

    ``block is None`` is the stack-external row: embeddings and the output head
    belong to the model but to no transformer block, and folding them into block
    0 would put their parameters — often a fifth of the model — under a layer
    that does not contain them.
    """

    block: int | None
    module: str
    parameters: int
    tensors: int
    element_types: tuple[str, ...]
    nominal_bytes: int | None
    #: How many of ``parameters`` sit in routed expert stacks. Zero for every
    #: dense cell, and for the router that shares this cell with the experts.
    routed_parameters: int = 0

    def active_parameters(self, used: int | None, total: int | None) -> int | None:
        """Parameters one token reaches in this cell.

        None is UNMEASURED: a routed cell whose runtime declared no routing is
        not the same as a dense cell, and returning ``parameters`` for it would
        report an idle expert bank as fully active. A routed pool that does not
        divide evenly by the declared expert count is likewise unavailable.

        CALLERS MUST CHECK ``ModelMorphometry.active_path_gap`` FIRST. A cell
        sees only its own tensors, so it cannot tell a genuinely dense cell from
        one whose model declared routing that the inventory never corroborated:
        both have ``routed_parameters == 0`` and both return ``parameters``
        here, which is right for the first and a fabrication for the second.
        The model-level fact that separates them is not reachable from a cell,
        so this method cannot defend itself and does not pretend to. That is
        why ``voxel_field`` blanks every cell when the gap is named rather than
        asking each one, and why a new consumer that reads cells directly has
        to repeat that check rather than inherit it.
        """
        if not self.routed_parameters:
            return self.parameters
        if (
            used is None
            or total is None
            or used <= 0
            or total <= 0
            or used > total
            or self.routed_parameters % total != 0
        ):
            return None
        dense = self.parameters - self.routed_parameters
        return dense + (self.routed_parameters * used) // total

    @property
    def family(self) -> str:
        return MODULES.get(self.module, MODULES["other"])[0]

    @property
    def stage(self) -> int:
        return MODULES.get(self.module, MODULES["other"])[1]

    @property
    def label(self) -> str:
        return MODULES.get(self.module, MODULES["other"])[2]

    @property
    def mean_bits_per_weight(self) -> Fraction | None:
        if self.nominal_bytes is None or self.parameters <= 0:
            return None
        return Fraction(self.nominal_bytes * 8, self.parameters)


@dataclass(frozen=True, slots=True)
class BlockProfile:
    """A transformer block's totals, for the depth profile."""

    block: int
    parameters: int
    nominal_bytes: int | None
    modules: int

    @property
    def mean_bits_per_weight(self) -> Fraction | None:
        if self.nominal_bytes is None or self.parameters <= 0:
            return None
        return Fraction(self.nominal_bytes * 8, self.parameters)


@dataclass(frozen=True, slots=True)
class ModelMorphometry:
    """The measured anatomy of one model.

    ``gaps`` is load-bearing. Every quantity this class could not establish is
    named there rather than defaulted, because the panel prints the gaps beside
    the figures and a silent zero is indistinguishable from a measurement.
    """

    model: str
    source: str
    architecture: str = ""
    family: str = ""
    quantization: str = ""
    declared_parameters: int | None = None
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    #: Routing, as the runtime declares it. Both None on a dense model, and both
    #: required before any active-parameter figure is produced.
    expert_count: int | None = None
    expert_used_count: int | None = None
    #: Present only when a routing key existed but could not be parsed. Keeping
    #: this state prevents malformed metadata from disappearing into the dense
    #: path when its parsed value is None.
    routing_metadata_error: str = ""
    tensors: tuple[TensorRecord, ...] = ()
    cells: tuple[Cell, ...] = ()
    gaps: tuple[str, ...] = ()
    refusal: str = ""
    schema_version: str = MORPHOMETRY_SCHEMA
    #: Axis order as the SOURCE presented it, which is what makes registration
    #: against the canonical space a real question rather than a formality.
    native_axis_order: tuple[str, str] = ("block", "module")

    @property
    def ok(self) -> bool:
        return not self.refusal and bool(self.cells)

    @property
    def counted_parameters(self) -> int:
        return sum(c.parameters for c in self.cells)

    @property
    def nominal_bytes(self) -> int | None:
        """Total storage, or None when any cell's element type was unknown —
        a partial total presented as a total is the error worth avoiding."""
        total = 0
        for cell in self.cells:
            if cell.nominal_bytes is None:
                return None
            total += cell.nominal_bytes
        return total

    @property
    def coverage(self) -> float | None:
        """Counted parameters over the runtime's declared parameter count.

        None when the runtime declared no count: that is UNMEASURED, not 1.0.
        A value below 1 means the inventory does not account for the whole
        model, which is exactly what a reader needs to know before trusting a
        per-cell breakdown.
        """
        declared = self.declared_parameters
        if not declared or declared <= 0:
            return None
        return self.counted_parameters / float(declared)

    # ── the active path ─────────────────────────────────────────────────────
    #
    # Parameter share answers "where does this model keep its weights". On a
    # mixture-of-experts model that is NOT the same question as "what does a
    # token cost", and reporting only the first invites the second to be read
    # off it. These figures separate the two.

    @property
    def is_mixture_of_experts(self) -> bool:
        """Whether any weights are routed. Measured from the inventory when one
        exists, so a runtime that declares experts without publishing them still
        reads as routed."""
        routed = self.routed_expert_parameters
        if routed:
            return True
        if self.active_path_gap is not None:
            return True
        return bool(self.expert_count and self.expert_count > 1)

    @property
    def routed_expert_parameters(self) -> int | None:
        """Parameters held in routed expert stacks.

        None when no inventory exists, or when routing metadata says an expert
        bank should exist but the inventory does not identify one. That is
        distinct from 0, which is the true answer for a dense model.
        """
        if not self.tensors:
            return None
        routed = sum(
            t.parameters for t in self.tensors if _is_routed_expert(t.name)
        )
        if not routed and self.active_path_gap is not None:
            return None
        return routed

    @property
    def always_active_parameters(self) -> int | None:
        """Parameters every token reaches: attention, norms, embeddings, the
        output head, the router, and any shared expert."""
        routed = self.routed_expert_parameters
        if routed is None:
            return None
        if not routed and self.active_path_gap is not None:
            return None
        return self.counted_parameters - routed

    @property
    def active_path_gap(self) -> str | None:
        """The named reason routed active arithmetic cannot be established."""
        if self.routing_metadata_error:
            return self.routing_metadata_error
        return _active_path_gap(
            self.tensors,
            self.expert_used_count,
            self.expert_count,
        )

    @property
    def active_parameters(self) -> int | None:
        """Parameters one token reaches, on the model's declared routing.

        A dense model returns its full count, because every weight is active.
        A routed model whose runtime did not declare ``expert_count`` and
        ``expert_used_count`` returns None -- UNMEASURED, never the total, since
        a total here would report an idle expert bank as fully active.

        The routed term is an average over tokens rather than a property of any
        one of them. It is produced only after every routed tensor's final GGML
        axis agrees with the declared expert count, establishing equal slices.
        """
        routed = self.routed_expert_parameters
        if routed is None:
            return None
        if self.active_path_gap is not None:
            return None
        if not routed:
            return self.counted_parameters
        used, total = self.expert_used_count, self.expert_count
        assert used is not None and total is not None
        return (self.counted_parameters - routed) + (routed * used) // total

    @property
    def active_fraction(self) -> float | None:
        """Active over counted parameters. None when either is unavailable."""
        active = self.active_parameters
        counted = self.counted_parameters
        if active is None or counted <= 0:
            return None
        return active / float(counted)

    @property
    def blocks(self) -> tuple[BlockProfile, ...]:
        totals: dict[int, list] = {}
        for cell in self.cells:
            if cell.block is None:
                continue
            entry = totals.setdefault(cell.block, [0, 0, 0, True])
            entry[0] += cell.parameters
            entry[1] += cell.nominal_bytes or 0
            entry[2] += 1
            if cell.nominal_bytes is None:
                entry[3] = False
        return tuple(
            BlockProfile(
                block=index,
                parameters=entry[0],
                nominal_bytes=entry[1] if entry[3] else None,
                modules=entry[2],
            )
            for index, entry in sorted(totals.items())
        )

    @property
    def stack_external(self) -> tuple[Cell, ...]:
        return tuple(c for c in self.cells if c.block is None)

    def by_family(self) -> tuple[tuple[str, int], ...]:
        """Parameters per family, in render order. Families with nothing in them
        are kept at zero so the breakdown has a stable shape across models."""
        totals = {name: 0 for name in FAMILIES}
        for cell in self.cells:
            totals[cell.family] = totals.get(cell.family, 0) + cell.parameters
        return tuple((name, totals.get(name, 0)) for name in FAMILIES)

    def block_dispersion(self) -> tuple[float | None, tuple[int, ...]]:
        """(relative median absolute deviation, blocks beyond 3 MADs).

        Transformer blocks are usually near-identical, so a block that is not is
        worth surfacing — a stack whose blocks differ is either a genuinely
        heterogeneous architecture or an inventory that lost tensors. Median and
        MAD rather than mean and sigma because a single outlier block would drag
        a mean-based threshold past itself and hide.
        """
        profiles = self.blocks
        if len(profiles) < 3:
            return None, ()
        values = sorted(p.parameters for p in profiles)
        median = _median(values)
        if median <= 0:
            return None, ()
        deviations = sorted(abs(p.parameters - median) for p in profiles)
        mad = _median(deviations)
        relative = mad / median
        if mad <= 0:
            outliers = tuple(
                p.block for p in profiles if p.parameters != median
            )
        else:
            outliers = tuple(
                p.block
                for p in profiles
                if abs(p.parameters - median) > 3 * mad
            )
        return relative, outliers


def _median(values: Sequence[float]) -> float:
    """Median of an ALREADY SORTED sequence."""
    count = len(values)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return float(values[middle])
    return (float(values[middle - 1]) + float(values[middle])) / 2.0


# ── parsing ──────────────────────────────────────────────────────────────────
def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _arch_key(info: Mapping[str, object], architecture: str, suffix: str):
    """Read ``<architecture>.<suffix>`` from a GGUF info map.

    The prefix is the architecture name, which is why it is read from the map
    rather than assumed: ``qwen3.block_count`` and ``llama.block_count`` are the
    same field under different keys, and a hard-coded prefix would silently
    return nothing for every model but one.
    """
    if architecture:
        direct = info.get(f"{architecture}.{suffix}")
        if direct is not None:
            return direct
    tail = "." + suffix
    for key, value in info.items():
        if isinstance(key, str) and key.endswith(tail) and not key.startswith("general."):
            return value
    return None


def _routing_declaration(
    info: Mapping[str, object],
    architecture: str,
    suffix: str,
) -> tuple[bool, object]:
    """Return ``(present, value)`` without erasing an unusable declaration.

    `_arch_key` intentionally returns only a value. Routing needs the stronger
    distinction because an absent key can describe a dense model, while a key
    present with ``None`` or a non-integer value is malformed evidence that must
    not silently select dense fallback arithmetic.
    """
    if architecture:
        direct = f"{architecture}.{suffix}"
        if direct in info:
            return True, info[direct]
    tail = "." + suffix
    for key, value in info.items():
        if (
            isinstance(key, str)
            and key.endswith(tail)
            and not key.startswith("general.")
        ):
            return True, value
    return False, None


def _cells_from_tensors(tensors: Sequence[TensorRecord]) -> tuple[Cell, ...]:
    buckets: dict[tuple[int | None, str], list] = {}
    for tensor in tensors:
        key = (tensor.block, tensor.module)
        entry = buckets.setdefault(key, [0, 0, set(), 0, True, 0])
        entry[0] += tensor.parameters
        entry[1] += 1
        entry[2].add(tensor.element_type)
        nominal = tensor.nominal_bytes
        if nominal is None:
            entry[4] = False
        else:
            entry[3] += nominal
        if _is_routed_expert(tensor.name):
            entry[5] += tensor.parameters
    # ``block is None`` sorts before every integer block so the stack-external
    # rows lead the table; Python will not order None against int for us.
    ordered = sorted(
        buckets.items(),
        key=lambda item: (item[0][0] is not None, item[0][0] or 0, item[0][1]),
    )
    return tuple(
        Cell(
            block=key[0],
            module=key[1],
            parameters=entry[0],
            tensors=entry[1],
            element_types=tuple(sorted(entry[2])),
            nominal_bytes=entry[3] if entry[4] else None,
            routed_parameters=entry[5],
        )
        for key, entry in ordered
    )


def _native_axis_order(tensors: Sequence[TensorRecord]) -> tuple[str, str]:
    """Which axis varies fastest in the inventory as the source reported it.

    Observed, not assumed. GGUF names are conventionally block-major, but the
    order is a property of the payload in hand, and reading it is what lets the
    compatibility analysis below answer a real question instead of restating an
    assumption.
    """
    block_changes = 0
    module_changes = 0
    previous: TensorRecord | None = None
    for tensor in tensors:
        if tensor.block is None:
            continue
        if previous is not None:
            if tensor.block != previous.block:
                block_changes += 1
            if tensor.module != previous.module:
                module_changes += 1
        previous = tensor
    if module_changes > block_changes:
        return ("block", "module")
    if block_changes > module_changes:
        return ("module", "block")
    # Undetermined — an inventory with one block, one module, or a tie carries
    # no evidence either way. Report the ggml convention rather than pick the
    # other one from nothing; a coin flip here would surface as a permutation
    # the source never actually declared.
    return ("block", "module")


def parse_tensor_inventory(payload: Mapping[str, object]) -> tuple[
    tuple[TensorRecord, ...], tuple[str, ...]
]:
    """(records, gaps) from a runtime's ``tensors`` list.

    A tensor with an unusable shape is DROPPED and counted in the gaps rather
    than assigned a zero: a cell that reads zero parameters is a statement about
    the model, and this would be a statement about the payload.
    """
    raw = payload.get("tensors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return (), ("the runtime published no tensor inventory",)
    records: list[TensorRecord] = []
    unusable = 0
    unknown_types: set[str] = set()
    truncated = len(raw) > MAX_TENSORS
    for item in list(raw)[:MAX_TENSORS]:
        if not isinstance(item, Mapping):
            unusable += 1
            continue
        name = str(item.get("name") or "").strip()
        shape_raw = item.get("shape")
        if not name or not isinstance(shape_raw, Sequence) or isinstance(shape_raw, (str, bytes)):
            unusable += 1
            continue
        dims: list[int] = []
        for value in shape_raw:
            dim = _int(value)
            if dim is None or dim <= 0:
                dims = []
                break
            dims.append(dim)
        if not dims:
            unusable += 1
            continue
        parameters = 1
        for dim in dims:
            parameters *= dim
        element_type = str(item.get("type") or item.get("dtype") or "").strip().upper()
        if element_type and element_type not in NOMINAL_BITS_PER_WEIGHT:
            unknown_types.add(element_type)
        records.append(
            TensorRecord(
                name=name,
                module=module_of(name),
                block=block_of(name),
                shape=tuple(dims),
                element_type=element_type,
                parameters=parameters,
            )
        )
    gaps: list[str] = []
    if truncated:
        gaps.append(
            f"the inventory was truncated at {MAX_TENSORS:,} tensors; totals "
            f"below are for the truncated set"
        )
    if unusable:
        gaps.append(f"{unusable} tensor entries had no usable name or shape and "
                    f"were dropped rather than counted as zero")
    if unknown_types:
        gaps.append(
            "no published bits-per-weight for element type(s) "
            + ", ".join(sorted(unknown_types))
            + " — storage for those cells is UNMEASURED"
        )
    return tuple(records), tuple(gaps)


# ── declared-architecture fallback ───────────────────────────────────────────
def _dense_decoder_cells(
    *,
    blocks: int,
    embedding: int,
    feed_forward: int,
    heads: int,
    kv_heads: int,
    head_dim: int,
    vocab: int | None,
) -> tuple[Cell, ...]:
    """Per-cell parameter counts for a dense decoder, from declared dimensions.

    Arithmetic only — the standard projection shapes for a llama-family decoder.
    No element types are known at this level, so ``nominal_bytes`` is None on
    every cell and the storage figures stay UNMEASURED rather than inheriting
    the file's headline quantisation, which does not apply uniformly.
    """
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    per_block = {
        "attn_norm": embedding,
        "attn_q": embedding * q_out,
        "attn_k": embedding * kv_out,
        "attn_v": embedding * kv_out,
        "attn_output": q_out * embedding,
        "ffn_norm": embedding,
        "ffn_gate": embedding * feed_forward,
        "ffn_up": embedding * feed_forward,
        "ffn_down": feed_forward * embedding,
    }
    cells: list[Cell] = []
    if vocab:
        cells.append(Cell(None, "token_embd", vocab * embedding, 1, (), None))
        cells.append(Cell(None, "output", vocab * embedding, 1, (), None))
    cells.append(Cell(None, "output_norm", embedding, 1, (), None))
    for index in range(blocks):
        for module, parameters in per_block.items():
            cells.append(Cell(index, module, parameters, 1, (), None))
    return tuple(cells)


# ── the entry point ──────────────────────────────────────────────────────────
def analyze(payload: Mapping[str, object], *, model: str = "") -> ModelMorphometry:
    """Measure one model from a runtime's ``show``-style payload.

    Never raises for a malformed payload: a refusal carrying a reason is the
    result the panel can render, and an exception at this boundary would reach
    the user as a traceback with less information in it.
    """
    if not isinstance(payload, Mapping):
        return ModelMorphometry(
            model=str(model or ""), source="", refusal=REFUSED_NO_METADATA,
            gaps=("the runtime returned no model metadata",))

    details = payload.get("details")
    details = details if isinstance(details, Mapping) else {}
    info = payload.get("model_info")
    info = info if isinstance(info, Mapping) else {}

    name = str(model or payload.get("model") or details.get("parent_model") or "").strip()
    architecture = str(info.get("general.architecture") or "").strip()
    family = str(details.get("family") or "").strip()
    quantization = str(details.get("quantization_level") or "").strip()
    declared = _int(info.get("general.parameter_count"))
    blocks = _int(_arch_key(info, architecture, "block_count"))
    embedding = _int(_arch_key(info, architecture, "embedding_length"))
    context = _int(_arch_key(info, architecture, "context_length"))
    # Read on BOTH paths. These used to be consulted only where the dense
    # fallback refuses, so a runtime that published an inventory had its routing
    # measured cell by cell and then described as if every expert ran.
    experts_present, experts_raw = _routing_declaration(
        info, architecture, "expert_count")
    experts_used_present, experts_used_raw = _routing_declaration(
        info, architecture, "expert_used_count")
    experts = _int(experts_raw)
    experts_used = _int(experts_used_raw)
    invalid_routing_fields = tuple(
        field for field, present, parsed in (
            ("expert_count", experts_present, experts),
            ("expert_used_count", experts_used_present, experts_used),
        )
        if present and parsed is None
    )
    routing_metadata_error = ""
    if invalid_routing_fields:
        invalid_names = ", ".join(invalid_routing_fields)
        routing_metadata_error = (
            f"the runtime declares unusable routing metadata for {invalid_names}; "
            "each value must be an integer. The expert partition is not "
            "established, so the active-parameter figure is UNMEASURED."
        )

    common = {
        "model": name,
        "architecture": architecture,
        "family": family,
        "quantization": quantization,
        "declared_parameters": declared,
        "context_length": context,
        "embedding_length": embedding,
        "block_count": blocks,
        "expert_count": experts,
        "expert_used_count": experts_used,
        "routing_metadata_error": routing_metadata_error,
    }

    tensors, gaps = parse_tensor_inventory(payload)
    if tensors:
        cells = _cells_from_tensors(tensors)
        observed_blocks = {c.block for c in cells if c.block is not None}
        if len(observed_blocks) > MAX_BLOCKS:
            return ModelMorphometry(
                source=SOURCE_TENSOR_INVENTORY,
                refusal=REFUSED_NO_BLOCKS,
                gaps=gaps + (f"the inventory declares more than {MAX_BLOCKS:,} "
                             f"blocks, beyond this view's budget",),
                **common)
        if blocks is not None and observed_blocks and len(observed_blocks) != blocks:
            gaps = gaps + (
                f"the runtime declares {blocks} blocks; the inventory contains "
                f"{len(observed_blocks)} — the breakdown follows the inventory",
            )
        # The breakdown remains exact about stored tensors even when the
        # runtime's routing declarations are absent or contradict their shapes.
        # Only the active path is withheld, and the reason travels with it.
        routing_gap = _active_path_gap(tensors, experts_used, experts)
        for gap in (routing_metadata_error, routing_gap):
            if gap and gap not in gaps:
                gaps = gaps + (gap,)
        return ModelMorphometry(
            source=SOURCE_TENSOR_INVENTORY,
            tensors=tensors,
            cells=cells,
            gaps=gaps,
            native_axis_order=_native_axis_order(tensors),
            **common)

    # No inventory. Fall back to arithmetic over declared dimensions — but only
    # for the shape that arithmetic describes.
    if experts_present or experts_used_present:
        routing_refusal = routing_metadata_error or (
            "this model declares routing metadata "
            f"(expert_count={experts}, expert_used_count={experts_used}). The "
            "dense decoder arithmetic below could understate it, so no "
            "breakdown is produced. A runtime that publishes a tensor "
            "inventory can establish the expert bank."
        )
        return ModelMorphometry(
            source=SOURCE_DECLARED_ARCHITECTURE,
            refusal=REFUSED_MIXTURE_OF_EXPERTS,
            gaps=gaps + (routing_refusal,),
            **common)
    if not blocks:
        return ModelMorphometry(
            source=SOURCE_DECLARED_ARCHITECTURE,
            refusal=REFUSED_NO_BLOCKS,
            gaps=gaps + ("the runtime declares no block count, so there is no "
                         "stack to arrange",),
            **common)
    feed_forward = _int(_arch_key(info, architecture, "feed_forward_length"))
    heads = _int(_arch_key(info, architecture, "attention.head_count"))
    kv_heads = _int(_arch_key(info, architecture, "attention.head_count_kv")) or heads
    missing = [
        label for label, value in (
            ("embedding_length", embedding),
            ("feed_forward_length", feed_forward),
            ("attention.head_count", heads),
        ) if not value
    ]
    if missing:
        return ModelMorphometry(
            source=SOURCE_DECLARED_ARCHITECTURE,
            refusal=REFUSED_INSUFFICIENT_ARCHITECTURE,
            gaps=gaps + ("the runtime declares no " + ", ".join(missing)
                         + " — the per-cell arithmetic needs them",),
            **common)
    key_length = _int(_arch_key(info, architecture, "attention.key_length"))
    head_dim = key_length or (embedding // heads if heads else 0)
    vocab = _int(_arch_key(info, architecture, "vocab_size"))
    cells = _dense_decoder_cells(
        blocks=blocks, embedding=embedding, feed_forward=feed_forward,
        heads=heads, kv_heads=kv_heads, head_dim=head_dim, vocab=vocab)
    derived_gaps = list(gaps)
    derived_gaps.append(
        "counted from declared dimensions, not from a tensor inventory: the "
        "figures are the dense-decoder shape this runtime declares, and any "
        "tensor the architecture adds beyond it is not represented")
    derived_gaps.append(
        "per-cell storage is UNMEASURED without an inventory — a file's headline "
        "quantisation is not applied uniformly across tensors")
    if not vocab:
        derived_gaps.append(
            "the runtime declares no vocabulary size, so the embedding and "
            "output projections are absent from the breakdown")
    if not key_length and heads:
        derived_gaps.append(
            "no declared attention.key_length; head dimension taken as "
            "embedding_length / head_count")
    return ModelMorphometry(
        source=SOURCE_DECLARED_ARCHITECTURE,
        cells=cells,
        gaps=tuple(derived_gaps),
        native_axis_order=("block", "module"),
        **common)


# ── the canonical space ──────────────────────────────────────────────────────
def _block_axis() -> CoordinateAxis:
    # No origin, resolution or bounds. The exact slice refuses a space whose
    # declared coordinate features it cannot enforce, and declaring depth here
    # would also make the template model-specific — the opposite of a shared
    # frame. Depth belongs to the artifact arranged on the space, not to the
    # space.
    return CoordinateAxis(
        axis_id="block",
        semantic_id="alelyon:model.transformer.block",
        kind=AxisKind.MODEL_LAYER,
        scalar_type=ScalarType.INTEGER,
        ordering=CoordinateOrdering.ASCENDING,
        transform_policy=("IDENTITY", "AXIS_PERMUTATION"),
        metadata={"stack_external": "block index is absent for embedding and "
                                    "output tensors"},
    )


def _module_axis() -> CoordinateAxis:
    return CoordinateAxis(
        axis_id="module",
        semantic_id="alelyon:model.transformer.module",
        kind=AxisKind.MODEL_STATE,
        scalar_type=ScalarType.LABEL,
        ordering=CoordinateOrdering.CANONICAL_LABEL_ORDER,
        labels_ref=label_dictionary_ref(MODULE_IDS),
        labels=MODULE_IDS,
        # AXIS_PERMUTATION is admissible on BOTH axes or on neither: a source
        # that reports module-major cannot be registered by permuting only the
        # block axis, and the policy intersection would refuse the whole
        # mapping. Reordering two axes changes no coordinate meaning, so it is
        # the least-expressive transform that resolves the difference.
        transform_policy=("AXIS_PERMUTATION", "IDENTITY", "LABEL_REINDEX"),
    )


def canonical_space() -> CoordinateSpace:
    """The immutable space every model is arranged on.

    Depends only on the module vocabulary, so every model commits to the same
    frame — which is the point: a shared frame is what makes two models
    comparable cell by cell, and a template that changed with the artifact would
    be no template at all.
    """
    return CoordinateSpace(
        space_id=CANONICAL_SPACE_ID,
        version=CANONICAL_SPACE_VERSION,
        topology=TopologyType.RECTANGULAR_TABLE,
        axes=(_block_axis(), _module_axis()),
        index_convention="ZERO_BASED",
        metadata={"schema": MORPHOMETRY_SCHEMA},
    )


def native_space(morph: ModelMorphometry) -> CoordinateSpace:
    """The model's own space, with the axes in the order its source presented
    them. Identical to the canonical space except possibly in axis ORDER — which
    is exactly the question the compatibility analysis answers."""
    axes = {"block": _block_axis(), "module": _module_axis()}
    ordered = tuple(axes[name] for name in morph.native_axis_order)
    # The metadata must match the template's exactly. Space-wide metadata is
    # part of `exact_space_key`, so recording the inventory SOURCE here would
    # make every registration refuse with INCOMPATIBLE_SEMANTICS — and it does
    # not belong on the space anyway: the source is a property of the artifact
    # being registered, not of the frame it is registered onto.
    return CoordinateSpace(
        space_id=f"{CANONICAL_SPACE_ID}.native",
        version=CANONICAL_SPACE_VERSION,
        topology=TopologyType.RECTANGULAR_TABLE,
        axes=ordered,
        index_convention="ZERO_BASED",
        metadata={"schema": MORPHOMETRY_SCHEMA},
    )


def register(morph: ModelMorphometry) -> CompatibilityReport:
    """Register the model's native arrangement against the canonical space.

    Returns the registration core's own report — an exact identity, an exact
    axis permutation, or a named refusal. This function signs nothing; `certify`
    is the step that does, and it is deliberately separate because signing
    touches a key and this does not.
    """
    return analyze_exact_compatibility(
        source_space=native_space(morph),
        target_space=canonical_space(),
    )


def certify(
    morph: ModelMorphometry,
    *,
    signer,
    issued_at_unix_seconds: int,
) -> SignedRegistrationCertificate:
    """Issue a signed certificate for this model's registration onto the frame.

    **What is certified is the REGISTRATION, not the measurements.** The
    certificate binds the model's native coordinate space, the canonical
    template space, and the exact transform between them. It says nothing about
    the parameter counts, the storage figures or the active path — those are
    measured from a payload this certificate never saw, and the schema records
    that by carrying `morphometry_refs` and `morphometry_commitments` as
    NOT_APPLICABLE rather than leaving them blank.

    So a verified certificate answers "was this model arranged on the frame
    everyone else is using, by a transform that replays exactly" and nothing
    further. That is a smaller claim than a reader might expect from the word
    certificate, which is why it is stated here rather than left to the schema.

    ``signer`` is anything with ``key_id()`` and ``sign(bytes)`` -- an
    `alelyon.runtime.atlas.data.attest.KeyStore` in practice. It is a parameter
    because issuing is a deliberate act: a key is generated on first use, and
    that must not happen because someone opened a view.

    Raises `CertificateError` when the model was not measured, or when the
    registration refused. A refusal is an answer, not an artifact to sign.

    The first of those guards is the less obvious one and it is deliberate. The
    frame registration would succeed for a refused measurement -- `native_space`
    is built from an axis order that survives any refusal -- so a certificate
    naming a model that produced no cells would verify perfectly while implying
    that model had been processed. Refusing costs nothing and removes the
    inference.
    """
    if not morph.ok:
        raise CertificateError(
            f"{morph.model or 'this model'} was not measured "
            f"({morph.refusal or 'no cells'}); its registration is not certified"
        )
    report = register(morph)
    return issue_registration_certificate(
        report,
        source_space=native_space(morph),
        template_space=canonical_space(),
        signer=signer,
        issued_at_unix_seconds=issued_at_unix_seconds,
    )


def space_commitment() -> str:
    """``sha256:…`` over the canonical bytes of the template space.

    It commits to the FRAME, not to any model's contents. Two readers who agree
    on this string are arranging models the same way; nothing about it says a
    model's figures are right.
    """
    return coordinate_space_ref(canonical_space())


# ── the voxel field ──────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Voxel:
    """One cell of the render volume.

    ``intensity`` is the cell's parameter count over the volume's maximum, so it
    is a share of the busiest cell — comparable within one model and, because
    both models are arranged on the same space, comparable between two only when
    the caller normalises them together.
    """

    x: int          # block index, or -1 for the stack-external plane
    y: int          # family index into FAMILIES
    z: int          # stage within the family
    module: str
    parameters: int
    intensity: float
    bits_per_weight: float | None
    #: Parameters one token reaches here, and that as a share of the busiest
    #: cell's active count. Both None when the model routes and its declarations
    #: are absent or inconsistent — a routed cell drawn at its stored size would
    #: render an idle expert bank as the brightest thing in the volume.
    active_parameters: int | None = None
    active_intensity: float | None = None


@dataclass(frozen=True, slots=True)
class VoxelField:
    """A deterministic sparse volume: the model's declared structure, not its
    learned weights. ``voxels`` is ordered by (x, y, z) so two runs over the
    same payload produce the same list."""

    width: int      # blocks along x
    height: int     # len(FAMILIES)
    depth: int      # STAGE_COUNT
    voxels: tuple[Voxel, ...] = ()
    max_parameters: int = 0
    has_stack_external: bool = False
    source: str = ""
    #: The busiest cell's ACTIVE parameter count, or None when the active path
    #: could not be established. A renderer offering "active share" as a
    #: brightness mode must offer it only when this is not None.
    max_active_parameters: int | None = None

    @property
    def has_active_path(self) -> bool:
        return self.max_active_parameters is not None

    @property
    def occupied(self) -> int:
        return len(self.voxels)

    @property
    def capacity(self) -> int:
        return max(0, self.width) * self.height * self.depth

    def plane(self, x: int) -> tuple[Voxel, ...]:
        return tuple(v for v in self.voxels if v.x == x)


def voxel_field(morph: ModelMorphometry) -> VoxelField:
    """Arrange a morphometry result as a sparse ``(block, family, stage)``
    volume.

    Empty voxels are simply absent. Filling them would draw a solid block for
    every model and hide the one thing the render is for — where the parameters
    actually sit.
    """
    cells = morph.cells
    if not cells:
        return VoxelField(width=0, height=len(FAMILIES), depth=STAGE_COUNT,
                          source=morph.source)
    maximum = max((c.parameters for c in cells), default=0)
    family_index = {name: i for i, name in enumerate(FAMILIES)}
    blocks = sorted({c.block for c in cells if c.block is not None})
    # Positions are the block's RANK, not its declared index, so an inventory
    # that skips a block draws a contiguous stack rather than a gap that looks
    # like a missing layer.
    rank = {block: i for i, block in enumerate(blocks)}
    used, total = morph.expert_used_count, morph.expert_count
    # One cell that cannot state its active count sinks the whole mode: a volume
    # where some cells are drawn by what a token reaches and others by what is
    # stored is a picture of neither.
    if morph.active_path_gap is None:
        active_by_cell = [cell.active_parameters(used, total) for cell in cells]
    else:
        active_by_cell = [None for _cell in cells]
    active_known = all(value is not None for value in active_by_cell)
    max_active = max(active_by_cell) if active_known else None  # type: ignore[type-var]
    voxels: list[Voxel] = []
    for cell, active in zip(cells, active_by_cell):
        x = -1 if cell.block is None else rank.get(cell.block, -1)
        bits = cell.mean_bits_per_weight
        voxels.append(Voxel(
            x=x,
            y=family_index.get(cell.family, len(FAMILIES) - 1),
            z=min(cell.stage, STAGE_COUNT - 1),
            module=cell.module,
            parameters=cell.parameters,
            intensity=(cell.parameters / maximum) if maximum > 0 else 0.0,
            bits_per_weight=float(bits) if bits is not None else None,
            active_parameters=active if active_known else None,
            active_intensity=(
                (active / max_active) if active_known and max_active else None
            ),
        ))
    voxels.sort(key=lambda v: (v.x, v.y, v.z, v.module))
    return VoxelField(
        width=len(blocks),
        height=len(FAMILIES),
        depth=STAGE_COUNT,
        voxels=tuple(voxels),
        max_parameters=maximum,
        has_stack_external=any(v.x < 0 for v in voxels),
        source=morph.source,
        max_active_parameters=max_active,
    )
