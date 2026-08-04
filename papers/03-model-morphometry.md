# Model Morphometry

### Measuring a model's declared anatomy without reading a weight

**Status:** systems paper. Ships in `alelyon-os`. The boundary described in §2 is
enforced by the module's own tests and must survive export verbatim.

---

## Abstract

Neuroimaging morphometry measures structure — cortical thickness, regional
volume, surface area — on a canonical coordinate space, so that two brains can be
compared cell by cell. We apply the same construction to language models: take
what a local runtime reports about a model it holds, arrange that inventory on a
canonical `(block, module)` coordinate space, and emit per-cell measurements plus
a content commitment over the space used.

The result is a structural picture: how parameters distribute across blocks and
across roles (attention projections, feed-forward, embeddings, norms), and at
what storage precision. Two models become comparable on a shared grid.

The paper's substance is a boundary. **Nothing here reads a weight, runs a
forward pass, or observes an activation.** Everything measured is derived from
declared metadata. We think this constraint makes the tool more useful rather
than less, and we explain why.

## 1. The construction

A canonical coordinate space with two axes:

- **block** — ordinal, one cell per transformer block, plus cells for the
  pre-block embedding and post-block head.
- **module** — categorical, the role a tensor plays: `attn_q`, `attn_k`,
  `attn_v`, `attn_o`, `ffn_gate`, `ffn_up`, `ffn_down`, `norm`, `embed`, `output`.

Every tensor a runtime reports is assigned to exactly one cell by a name-matching
rule, or to an explicitly unassigned bucket. Per-cell measurements are then
parameter count, byte footprint, nominal bits per weight, and the set of
quantisation types present.

The coordinate space itself is committed: a `coordinate_space_ref` over the axes,
their ordering, and the label dictionary. Two morphometries computed on the same
space reference are comparable; two computed on different ones are not, and the
reference is how you can tell without inspecting either.

## 2. Two sources, never blended

**`SOURCE_TENSOR_INVENTORY`** is the strong case. The runtime publishes each
tensor's name, element type and shape. Per-cell parameter counts are then
*counted*, and the quantisation type is each tensor's own.

**`SOURCE_DECLARED_ARCHITECTURE`** is the fallback. The runtime publishes only
architecture fields — block count, embedding length, feed-forward length, head
counts — and per-role parameter counts are computed analytically. This is
arithmetic over declared dimensions and is correct **only for the dense decoder
shape it names**. A mixture-of-experts model with no tensor inventory is
therefore **refused by name** rather than approximated. An MoE model's parameter
distribution is not the dense formula's answer, and returning the dense answer
with no flag would be a confident wrong number in the shape of a right one.

`ModelMorphometry.source` always says which one ran. Mixing them in one result is
not possible — not discouraged, not possible.

## 3. Bit widths are nominal

`NOMINAL_BITS_PER_WEIGHT` is a static table of published bits-per-weight for each
`ggml` element type. It is the **format's nominal cost**, not a measurement of the
file. A type absent from the table yields `None` and a named gap rather than a
guess.

This matters because a quantisation format's real footprint includes block
scales, zero points and padding, and the nominal figure understates it by a few
percent. Reporting nominal-as-measured would be a small lie that compounds when
someone sums across a model to predict memory.

## 4. Why the boundary is a feature

The obvious objection: a structural measurement that never reads a weight cannot
say anything about what the model *knows*. That is correct, and it is the point.

Three arguments:

**It is honest about what is derivable.** Nothing in a tensor inventory carries
information about learned behaviour. A tool that measured structure and *implied*
capability would be making a claim its inputs cannot support. Calling this "a
scan of what the model knows" is precisely the sentence the module's docstring
forbids.

**It is cheap and safe.** Reading declared metadata requires no GPU, no model
load, no inference, and touches no weights — so it can run against a model the
caller is not licensed to execute, in an environment with no accelerator, in
milliseconds. A structural comparison across twenty models is a table lookup.

**Structure is genuinely informative for the questions it answers.** Where does
this model spend its parameters? Which blocks are stored at lower precision than
their neighbours? Do two checkpoints claiming the same architecture actually have
the same shape? Those are answerable from declared metadata and are the questions
people ask before deciding what to run.

## 5. Relationship to the certificate

Morphometry is the first consumer of the exact registration core (paper 2). The
`(block, module)` space is a coordinate space in that engine's sense, so a
morphometry result can carry a content commitment and, where a caller wants one,
a signed registration certificate through the same `KeyStore` an envelope uses.

That shared substrate is why morphometry and the lattice engine ship in one
distribution rather than two: separating them would duplicate the certificate
machinery.

## 6. Purity

No network, no filesystem, no clock, no GUI toolkit. The caller supplies the
payload. The desktop surface is what fetches it.

This makes the module deterministic and trivially testable, and it is the reason
it could be published without dragging a runtime dependency behind it.

## 7. Reproduction

```bash
pip install "alelyon-os[lattice]"
python -c "from alelyon.runtime.vector.lattice import morphometry; print(morphometry.__doc__)"
```

The module accepts an Ollama-shaped `show` payload. A tensor inventory produces
the counted path; an architecture-only payload produces the analytic path with
`source` saying so; an MoE architecture with no inventory produces a named
refusal.

## 8. What we would want reviewed first

1. The name-matching rule that assigns tensors to `module` cells. It is
   regex-based and was written against a handful of architectures; a tensor it
   fails to place lands in the unassigned bucket, which is visible but not
   loud.
2. Whether `NOMINAL_BITS_PER_WEIGHT` should be replaced by a measured footprint
   where the runtime publishes byte sizes — currently we report both and let the
   caller compare, but that puts the reconciliation on them.
3. Whether the dense-decoder analytic formula is right for architectures that
   are dense but unusual (shared embeddings, tied output heads, grouped-query
   attention with unusual ratios).
