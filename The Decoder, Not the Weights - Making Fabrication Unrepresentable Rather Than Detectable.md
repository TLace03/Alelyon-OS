# The Decoder, Not the Weights: Making Fabrication Unrepresentable Rather Than Detectable

**Alelyon Quantitative Services**
Working Paper V of V · Series: *Certified Computation over Compressed and Committed Data*

**Version:** 1.0 · **Date:** 2026-08-04 · **Status:** Working paper; not peer reviewed.
**Artifact:** `alelyon.runtime.oracle.assistant.constrain`, distributed in
`alelyon-os`; operator surface `alelyon-workspace`.
**Correspondence:** Alelyon Quantitative Services.

---

## Standing honesty contract

1. **Name the certified object.** An unqualified "certified" launders a small
   guarantee over large ones. This paper bounds *fabrication of figures*, not
   correctness of figures and not soundness of argument.
2. **Revision is detected; invention is not.** Where a guarantee derives from a
   commitment rather than from a construction, it establishes that inputs were not
   revised after commitment, never that they were right.
3. **An absent measurement is reported as UNMEASURED.**
4. **Refusal is a first-class outcome.** A provider that cannot enforce the
   grammar refuses a constrained request rather than answering it unconstrained.

---

## Abstract

A language model asked to report figures from a data source will sometimes emit a
figure the source does not contain. The established mitigations are training —
preference optimization against hallucination, retrieval-augmented fine-tuning —
and post-hoc verification, in which each emitted figure is checked against the
source and the generation is retried on failure. Both yield probabilistic
guarantees: training relocates probability mass without driving it to zero, and
verification acts after the figure has been written, so the system's guarantee is
that it will usually notice. This paper describes a construction with a
categorically different guarantee. The answer is generated as structure rather
than prose; the decoding grammar admits a numeric slot whose value is an
enumeration of exactly the rendered strings the tools returned, and prose slots
matched against a digit-free pattern. Where the backend compiles the schema into
a sampling grammar, the set of numerically distinct reachable outputs is exactly
the set of tool results, so a fabricated figure is not improbable but
unrepresentable. Where the backend only advises a schema, the same contract is
enforced by a validator and the guarantee degrades explicitly to detection and
refusal. The paper reports a sealed evaluation in which the model reproduced a
supplied exemplar 140 times out of 140, and argues that this measures the
certification harness and the grammar's own format compliance rather than the
model's reasoning. That reading is the paper's principal caveat and motivates a
stated requirement on evaluations of constrained systems.

**Keywords:** constrained decoding, grammar-constrained generation, hallucination,
structural guarantees, tool use, provenance, evaluation design.

---

## 1. Introduction

Consider a system that answers questions about a live data source by calling
deterministic tools and narrating the results. The property the operator requires
is that every figure appearing in the answer came from a tool. The property is
easy to state and, under the usual approaches, impossible to guarantee.

The difficulty is a property of what weights are. Training shapes a distribution
over token sequences. However that distribution is shaped, it assigns nonzero
probability to sequences containing a figure absent from the source, and sampling
draws from it. Probability mass on an incorrect token can be reduced; it cannot be
set to zero by training, and for a given input it cannot be shown to be zero
without enumerating the output space. A guarantee sourced from weights is
therefore a statement about a *rate*, measured on a benchmark and extrapolated to
inputs the benchmark did not contain.

Post-hoc verification changes the failure mode without changing its character.
A checker that extracts numeric mentions and matches them against tool results
acts after generation. It is a detector, and its guarantee is a detection rate.

This paper takes the third option. If the property is to be guaranteed rather
than estimated, it must be enforced at the point where tokens are chosen.

### 1.1 Contributions

1. A statement of why a non-fabrication guarantee is not the kind of claim a set
   of weights can support, and what a weights-sourced claim actually asserts (§2).
2. A specific decoding grammar — figures as an enumeration of rendered tool
   outputs, prose as a digit-free pattern — under which the set of numerically
   distinct reachable outputs equals the set of tool results (§3).
3. A distinction between two guarantees that are commonly conflated, structural
   impossibility and detection-and-refusal, together with the provider-capability
   discipline that keeps them apart (§4).
4. A sealed evaluation and its honest reading: a 140-of-140 result that measures
   the harness rather than the model, and the consequent requirement on controls
   for evaluations of constrained systems (§6).
5. A statement of what the construction does not establish, in particular that it
   bounds fabrication and not correctness (§8).

---

## 2. Why weights cannot supply the guarantee

**Proposition 2.1 (informal).** Let p_θ be an autoregressive distribution over
token sequences and let F be the set of sequences containing a figure not present
in the source. For any θ obtained by gradient-based training on a finite corpus,
p_θ(F) > 0, and sampling from p_θ realizes elements of F at rate p_θ(F).

The proposition is not deep and is not new; it is stated because systems are
routinely described as though it were false. Its practical content is that the
claim "this model does not fabricate figures" has no truth condition that
training can establish. What training can establish is that p_θ(F) is small on a
measured distribution of inputs. Deployment inputs are not that distribution.

The consequence for system design is that a guarantee must come from a component
whose behavior is not a learned distribution. The decoder is such a component: the
set of sequences it can emit is determined by the grammar it is given, not by the
weights it samples under.

---

## 3. Construction

The answer is generated as structure rather than prose. A response is a sequence
of sentence records, each split into a leading prose fragment, an optional
figure, and a trailing prose fragment:

```json
{"sentences": [
  {"before": "NVDA carries ", "figure": "34.20%", "after": " of the book's risk."}
]}
```

Two constraints are compiled into the decoding grammar.

**Constraint 3.1 (figure enumeration).** The `figure` field is an enumeration
whose members are exactly the rendered strings the tools returned for this
request. No other value is legal. A quoted figure is therefore a tool figure by
construction rather than by inspection.

**Constraint 3.2 (digit-free prose).** The `before` and `after` fields match
`^[^0-9]*$`. Prose cannot contain a digit, so every numeral in the rendered answer
necessarily arrives through a `figure` slot.

The empty string is a legal member of the `figure` enumeration. Without it, a
model would be obliged to quote a figure in every clause, which most sentences do
not require.

**Proposition 3.3.** Under Constraints 3.1 and 3.2, and assuming the decoder
enforces the grammar, the set of numerically distinct outputs reachable by any
sampling procedure is exactly the set of tool results, in arbitrary arrangement
and multiplicity.

*Proof sketch.* By 3.2 no numeral occurs outside a `figure` slot. By 3.1 every
`figure` slot takes a value in the tool-result set. The reachable numeral
multiset is therefore a sub-multiset of the tool-result set, and each element is
reachable by a legal completion. ∎

Fabrication is consequently not filtered from the output. It is absent from the
output space.

---

## 4. Two guarantees, kept distinct

The distinction in this section is the paper's principal practical content,
because a system that conflates the two asserts something it cannot deliver.

**Structural.** Where the backend compiles the schema into a sampling grammar —
as `llama.cpp` and Ollama do through GBNF, and as the constrained-decoding
libraries in §7 do — Proposition 3.3 applies and a fabricated figure is
unrepresentable.

**Detection and refusal.** Where a backend merely advises a schema — the "JSON
mode" of most hosted interfaces, and in-process loaders that do not restrict the
sampler — the grammar is a request rather than a constraint. A validator then
rejects any reply containing a digit in a prose slot or an unrecognized figure,
and the request is refused. The property still holds of everything the system
emits; it holds by rejection rather than by construction.

The claim this system makes is therefore: **structurally impossible where the
grammar is enforced; detected and refused everywhere else.** It is never
"the model was trained not to".

### 4.1 Provider capability must be declared, not inferred

An earlier implementation inferred the capability from locality, on the reasoning
that a local provider meant Ollama and Ollama compiles grammars. That inference
ceased to be true when a second local backend existed: an in-process Hugging Face
loader and an OpenAI-compatible local server are both local, and neither
restricts the sampler.

A provider now **declares** grammar support explicitly, and a provider lacking the
seam **refuses** a constrained request rather than answering it unconstrained. The
refusal is necessary rather than fastidious: the caller labels a constrained
result as carrying the structural guarantee, so an unconstrained answer returned
through that path would be mislabeled at the point of use. This is an instance of
a general rule — a capability that is inferred from a proxy will eventually be
inferred wrongly, and the failure is silent.

---

## 5. Cost, and the two operating modes

Constrained decoding narrows what a small model can express. A model unable to
phrase its answer inside the template produces no answer.

The system therefore treats a constrained attempt as **preferred rather than
mandatory** and falls back to the checked path instead of returning nothing. Every
answer carries a `constrained` flag recording which guarantee applies. Two answers
that read identically do not carry the same guarantee, and the flag is the only
way to distinguish them.

A mode distinction accompanies this. The constrained contract is appropriate
beside a live data source and inappropriate for a general assistant, because it
reduces the model to a renderer of another component's table. The distributed
Workspace surface therefore also offers an open mode in which the model may
reason, compute and explain. The grounding check still runs in open mode, but its
output is presented as **provenance** — which figures came from a tool and which
are the model's own — rather than as a verdict. Both modes are implemented and
neither is described as the other.

---

## 6. Sealed evaluation

**Design.** A sealed evaluation of the certification pipeline was conducted with a
local Qwen model producing structured outputs under the constrained decoder.

**Result.** The model reproduced the supplied exemplar in 140 of 140 cases.

**Reading.** Taken at face value the figure is a capability claim. It is not one.
The exemplar was present in the prompt, so a perfect score measures copying
fidelity rather than reasoning. The probe tracked JSON format compliance, which
the grammar guarantees by construction under Proposition 3.3; to that extent the
evaluation measured its own scaffolding.

**Table 1.** What the sealed evaluation establishes and what it does not.

| Question | Established? | Basis |
|---|---|---|
| Does the pipeline produce schema-valid structured output? | Yes | 140/140 |
| Is the format compliance attributable to the model? | **No** | Guaranteed by the grammar |
| Did the model reason about the task? | **No evidence** | Exemplar present in prompt |
| Would the model generalize past the exemplar? | **UNMEASURED** | No control task was run |

The result is reported with that reading attached, because the bare figure is
precisely the kind of number that is quoted without it.

**Generalization.** When a decoder is constrained tightly enough to guarantee a
property, it becomes correspondingly harder to determine whether the model
contributed anything. An evaluation of a constrained system requires a control
task that the constraint does not trivially satisfy. The evaluation reported here
did not have one, and that is a defect in the evaluation rather than a property
of the method.

---

## 7. Relationship to grounding, and to prior work

### 7.1 Grounding

The two mechanisms are complementary and are not the same check. The grounding
component extracts numeric mentions from generated prose and matches them against
the facts the tools returned; it detects a fabricated figure after the model has
written it. That is a genuine check and it is what operates in open mode, where
the model is permitted to write figures of its own. The constraint component makes
the figure unwritable and is what operates where the guarantee is required.

A system with only grounding possesses a detector. A system with only constraint
cannot operate in open mode. The distributed package implements both, and each
answer records which applied.

### 7.2 Prior work

Grammar-constrained decoding is established. Incremental parsing against a grammar
during autoregressive decoding was demonstrated for semantic parsing by PICARD
[5]; efficient guided generation via finite-state indexing of regular and
context-free constraints is developed in Outlines [3]; grammar-constrained
decoding for structured tasks without fine-tuning is treated by Geng et al. [4];
and GBNF in `llama.cpp` [6] together with the Guidance library [7] are the
implementations in common use. The general hallucination problem and its
mitigations are surveyed by Ji et al. [1]; retrieval augmentation is due to Lewis
et al. [2]; preference-optimization training is exemplified by Ouyang et al. [8].

Neither the technique nor the machinery is claimed here. What is claimed is the
**specific grammar** — figures as an enumeration of rendered tool outputs, prose
as a digit-free pattern — under which Proposition 3.3 yields the non-fabrication
property, together with the operational discipline of §4.1 that prevents the
structural guarantee from being asserted where the backend cannot supply it.

---

## 8. Limitations and open problems

1. **This bounds fabrication, not correctness.** The figures originate from tools.
   If a tool is wrong, the answer is wrong and perfectly constrained. This is the
   same boundary Paper I draws for the signed envelope, and it is the boundary at
   which every guarantee in this series lives.
2. **This bounds figures, not argument.** The model can arrange true figures into
   a misleading sentence. Nothing in the construction inspects the claim a
   sentence makes.
3. **Spelled-out numerals defeat the digit-free pattern.** "Thirty-four percent"
   satisfies `^[^0-9]*$`. The construction accepts this on the judgment that a
   spelled-out numeral cannot carry the precision that makes a fabricated figure
   consequential in this setting. The judgment is stated rather than assumed and
   is the first item this paper would put to a reviewer.
4. **The enumeration's behavior at scale is UNMEASURED.** The `figure` enumeration
   contains one member per rendered tool output. Whether grammar compilation and
   sampling degrade as that set grows, and where the practical ceiling lies, has
   not been measured.
5. **The evaluation lacked a control.** Per §6, no task was run that the grammar
   does not trivially satisfy, so the model's contribution is unquantified.
6. **The structural guarantee is conditional on the backend.** Proposition 3.3
   assumes the decoder enforces the grammar. That assumption is discharged by a
   declared provider capability, which is itself a configuration fact rather than
   a verified property of the running backend.

---

## 9. Conclusion

A non-fabrication guarantee cannot be sourced from weights, because weights define
a distribution and a distribution assigns nonzero mass to the outcome being
excluded. It can be sourced from the decoder, because the decoder's reachable set
is determined by a grammar. Under a grammar in which numeric slots enumerate
rendered tool outputs and prose slots exclude digits, the reachable numeric
content of an answer is exactly the tool results. Where a backend enforces that
grammar the guarantee is structural; where it does not, the same contract is
enforced by refusal, and the distinction is declared rather than inferred. The
sealed evaluation reported here validates the certification harness and does not
validate the model, and evaluations of constrained systems require controls that
the constraint does not satisfy by construction.

---

## Appendix A — Reproduction

```bash
pip install "alelyon-os[workspace]"
ollama serve && ollama pull qwen2.5:7b
alelyon-workspace models
alelyon-workspace
```

`alelyon-workspace models` prints, per provider, either `constrains sampling` or
`cannot constrain sampling`. That line is the §4 distinction as the running system
reports it, and it is printed rather than assumed.

**Environment.** Python 3.12. The constrained path requires a backend that
compiles a schema into a sampling grammar; Ollama serving a GGUF model through
`llama.cpp` satisfies this. No network access is required beyond retrieving the
model artifact.

**Data availability.** The sealed evaluation used a fixed prompt set with a
supplied exemplar; the harness is `alelyon.runtime.oracle.assistant`. Tool outputs
in the constrained path are produced by deterministic components described in
Papers I and II.

---

## Appendix B — The grammar contract

| Field | Type | Constraint | Rationale |
|---|---|---|---|
| `sentences` | array | — | The answer is structure, not prose |
| `before` | string | `^[^0-9]*$` | No numeral outside a figure slot |
| `figure` | enum | Exactly the rendered tool outputs, plus `""` | Constraint 3.1; `""` avoids forcing a figure per clause |
| `after` | string | `^[^0-9]*$` | As `before` |
| `constrained` | boolean | Set by the system, not the model | Records which of the two §4 guarantees applies |

---

## References

The identifiers below were recorded from the authors' working bibliography and
should be checked against the primary sources before external publication; this
series has not yet had a bibliographic review.

[1] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. Bang, A. Madotto,
P. Fung. *Survey of Hallucination in Natural Language Generation.* ACM Computing
Surveys 55(12), 2023. arXiv:2202.03629.

[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, D. Kiela. *Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020. arXiv:2005.11401.

[3] B. T. Willard, R. Louf. *Efficient Guided Generation for Large Language
Models.* 2023. arXiv:2307.09702.

[4] S. Geng, M. Josifoski, M. Peyrard, R. West. *Grammar-Constrained Decoding for
Structured NLP Tasks without Finetuning.* EMNLP 2023. arXiv:2305.13971.

[5] T. Scholak, N. Schucher, D. Bahdanau. *PICARD: Parsing Incrementally for
Constrained Auto-Regressive Decoding from Language Models.* EMNLP 2021.
arXiv:2109.05093.

[6] G. Gerganov and contributors. *llama.cpp*, GBNF grammar-constrained sampling.
Software.

[7] Guidance contributors. *Guidance: a guidance language for controlling large
language models.* Software.

[8] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang,
S. Agarwal, K. Slama, A. Ray, et al. *Training Language Models to Follow
Instructions with Human Feedback.* NeurIPS 2022. arXiv:2203.02155.

[9] Qwen Team. *Qwen2.5 Technical Report.* 2024. arXiv:2412.15115.