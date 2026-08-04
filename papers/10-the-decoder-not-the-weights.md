# The Decoder, Not the Weights

### Making fabrication unrepresentable rather than detectable

**Status:** systems paper with one sealed evaluation. Ships in `alelyon-os` as
`alelyon.runtime.oracle.assistant.constrain` and the `alelyon-workspace` CLI. The
evaluation's result is a caveat on the method's scope and is reported in §5.

---

## Abstract

A language model asked to report figures from a data source will sometimes invent
one. The standard mitigations are training (RLHF against hallucination,
retrieval-augmented fine-tuning) and post-hoc checking (verify each number in the
output against the source, regenerate on failure).

Both are probabilistic. Training moves probability mass; a mass of 10⁻⁹ on a
wrong token is still a token the sampler can draw. Post-hoc checking catches the
figure *after* the model has written it, which means the system's guarantee is
"we will usually notice".

We describe a third approach with a categorically different guarantee.
**Constrain the decoder so that the wrong figure is not in the grammar.** Then a
fabricated number is not unlikely — it is unreachable.

The paper also reports the evaluation that bounds what this buys, in which the
model copied a supplied exemplar 140 times out of 140. The win was the
certification layer, not the model's reasoning, and we say so.

## 1. Why weights cannot provide the guarantee

The claim "this model does not fabricate figures" is not the kind of statement a
set of weights can support.

Training shapes a distribution over tokens. Whatever the shaping, the
distribution assigns nonzero probability to token sequences containing a figure
that was never in the source, and sampling draws from that distribution. You can
push the mass down. You cannot push it to zero, and you cannot prove for a given
input that it *is* zero without enumerating the output space.

So a guarantee sourced from weights is a statement about a rate, measured on a
benchmark, extrapolated to inputs the benchmark did not contain.

## 2. The construction

Generate the answer as **structure, not prose**:

```json
{"sentences": [
  {"before": "NVDA carries ", "figure": "34.20%", "after": " of the book's risk."}
]}
```

Two constraints are compiled into the decoding grammar:

1. **`figure` is an enum of the exact rendered strings the tools returned.**
   There is no other legal value. A quoted figure is a tool figure *by
   construction*.
2. **`before` and `after` match `^[^0-9]*$`** — prose cannot contain a digit.
   Every number in the answer therefore comes through a `figure` slot.

The empty string is a legal `figure`, because most sentences carry no number and
without it the model would be forced to quote one in every clause.

The consequence is that the set of numerically-distinct outputs the decoder can
produce is exactly the set of tool results, in any arrangement. Fabrication is
not filtered. It is not in the output space.

## 3. Two guarantees, and they are not the same one

This distinction is the paper's most important practical content, because a
system that conflates them is claiming something it cannot deliver.

**Where the backend compiles the schema into a grammar** — llama.cpp and Ollama
do, via GBNF — the constraint is **structural**. A fabricated figure is literally
unrepresentable.

**Where a backend merely *suggests* a schema** — most hosted APIs' "JSON mode",
and in-process loaders that do not restrict the sampler — the constraint is a
request. A validator still rejects any reply containing a digit in prose or an
unknown figure, so the output is **detected and refused**.

The honest claim is therefore: **structurally impossible where the grammar is
enforced, detected and refused everywhere else.** Never "the model was trained
not to".

This forced a design change we would flag to anyone building similar systems.
Provider capability had been inferred as `return self.local` — on the reasoning
that local meant Ollama, which compiles grammars. That stopped being true the
moment a second local backend existed: an in-process Hugging Face loader and an
OpenAI-compatible server are both local and neither restricts the sampler. A
provider now **declares** `grammar` explicitly, and a provider without the seam
**refuses** a constrained request rather than silently answering it
unconstrained — because the caller would badge the result as guaranteed.

## 4. The cost, stated

Constrained decoding narrows what a small model can say. A model that cannot
phrase its answer inside the template will fail to produce one.

The system therefore treats a constrained attempt as **preferred, not
mandatory**, and falls back to the checked path rather than returning nothing.
The resulting answer carries `constrained: true` or `false`, so a reader knows
which guarantee they have. Two answers that look identical are not equally
trustworthy and the flag is how you tell.

There is also a mode distinction. The constrained contract is right beside a live
data source and wrong for a general assistant: it turns the model into a renderer
of somebody else's table. The published Workspace CLI runs an open mode where the
model may reason, calculate and explain — and the grounding check still runs, but
its output is presented as **provenance** (which figures came from a tool, which
are the model's own) rather than as a verdict. Both modes exist; neither is
described as the other.

## 5. The sealed evaluation, and what it actually showed

We ran a sealed evaluation of the certification pipeline. A local Qwen model was
asked to produce structured outputs under the constrained decoder.

**It reproduced the supplied exemplar 140 times out of 140.**

Read carefully, that result says the harness works and the model did not reason.
A 140/140 score on a task where the exemplar is in the prompt measures copying
fidelity, not capability. The probe was tracking **JSON format compliance**, which
the grammar guarantees by construction — so the evaluation was, in part, measuring
its own scaffolding.

**The win was certification, not model reasoning.** We report the number with that
sentence attached, because 140/140 detached from it is exactly the kind of figure
that gets quoted.

The general lesson: when you constrain a decoder hard enough to guarantee a
property, you have also made it much harder to evaluate whether the model
contributed anything. An evaluation of a constrained system needs a control the
constraint does not trivially satisfy.

## 6. Relationship to grounding

The two mechanisms are complementary and are not the same check.

`grounding.py` extracts numeric mentions from generated prose and matches them
against the facts the tools returned. It catches a fabricated figure **after** the
model writes it. That is a real check and it earns its place — it is what runs in
open mode, where the model is permitted to write figures of its own.

`constrain.py` makes the figure unwritable in the first place. It is what runs
where the guarantee matters.

A system with only grounding has a detector. A system with only constraint cannot
operate in open mode. The published package has both, and each answer records
which one applied.

## 7. What this does not claim

- **Not that the answer is correct.** The figures come from tools; if a tool is
  wrong, the answer is wrong and perfectly constrained. This bounds fabrication,
  not truth — the same boundary paper 1 draws for the envelope.
- **Not that the prose is right.** The model can arrange true figures into a
  misleading sentence. Nothing here checks argument.
- **Not that this is novel.** Grammar-constrained decoding is well established
  (GBNF, Outlines, Guidance, JSON-schema-constrained sampling). The contribution
  is the *specific grammar* — figures as an enum of tool outputs, prose as a
  digit-free pattern — and the discipline of not claiming the structural
  guarantee where the backend cannot provide it.

## 8. Reproduction

```bash
pip install "alelyon-os[workspace]"
ollama serve && ollama pull qwen2.5:7b
alelyon-workspace models      # reports which providers can constrain sampling
alelyon-workspace
```

`models` prints `constrains sampling` or `cannot constrain sampling` per provider.
That line is the difference between the two guarantees in §3, and it is printed
rather than assumed.

## 9. What we would want reviewed first

1. Whether the digit-free prose pattern has holes — spelled-out numbers
   ("thirty-four percent") pass it, and we treat that as acceptable because it
   cannot carry the precision that makes a fabricated figure dangerous. That
   judgement deserves challenge.
2. Whether the enum-of-rendered-strings approach degrades when the tool output
   set is large, and where the practical ceiling is.
3. A better evaluation design than the one in §5 — specifically, a control task
   the grammar does not trivially satisfy.
