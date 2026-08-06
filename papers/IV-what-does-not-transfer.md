# What Does Not Transfer: Four Refutations of Structured Compression, and the Artifact That Survived

**Alelyon Quantitative Services**
Working Paper IV of V · Series: *Certified Computation over Compressed and Committed Data*

**Version:** 1.0 · **Date:** 2026-08-04 · **Status:** Working paper; not peer reviewed.
**Experiments:** EXP-1, EXP-2, EXP-LLM, EXP-KV, conducted 2026-07-19.
**Correspondence:** Alelyon Quantitative Services.

---

## Standing honesty contract

Every paper in this series is subordinate to a claim discipline that four rules
govern. They are restated in each paper because a reader of one paper never sees
the index.

1. **Name the certified object.** "Certified" without a noun launders a small
   term over large ones. The certificates in this series bound *storage
   quantization*, which at realistic capture precision is several orders of
   magnitude smaller than sampling error. The decomposed budget states this
   rather than concealing it.
2. **Revision is detected; invention is not.** A producer who fabricates data at
   capture signs a receipt that verifies perfectly. Verification establishes that
   committed inputs were not revised after commitment; it does not audit a source.
3. **An absent measurement is reported as UNMEASURED.** A blank beside a filled
   field reads as "checked, and fine". Every absent term is named with its reason.
4. **Refusal is a first-class outcome.** Several results in this series are
   certificates that refuse on real data. A refusal is the system operating
   correctly, not a failure to report.

---

## Abstract

A compression program was undertaken on the hypothesis that financial return
panels and neural network weight matrices are low-residual-rank: that after a
low-rank factor core is removed, the residual is near-isotropic and therefore
compresses to near its entropy under a scalar quantizer, so that the composite
representation dominates flat quantization at matched bits per element. Four
experiments were designed to refute this hypothesis, with acceptance criteria
registered before execution. The hypothesis is false for financial panels and
false for language model weights. In both classes the compression ratio is
governed by per-element bit width, which is precisely what commodity 4-bit and
8-bit integer formats already deliver. One positive transfer was found: the
key component of the attention cache is genuinely low-rank, and a low-rank core
with a quantized residual improves on flat cache quantization at 2 to 3.5 bit
budgets, subject to an acausal-basis caveat that makes the reported gain an upper
bound. A second result transferred to every data class tested: a calibrated error
bar on a quantity computed from compressed data, with predicted error tracking
realized error at R² ≈ 0.95 on a returns panel and with unit slope on weights.
That certificate, rather than any compression ratio, is the artifact the rest of
this series develops. The methodological contribution is that the precondition
test is inexpensive, decisive, and rarely run first.

**Keywords:** structured compression, low-rank decomposition, post-training
quantization, key-value cache, negative results, matched-bit comparison,
error certification.

---

## 1. Introduction

Structured compression proposes that data with exploitable algebraic structure
can be represented more compactly than a per-element code allows. For a matrix
X, the canonical form of the proposal is a decomposition X = LF + E in which L
is a low-rank factor core stored cheaply and E is a residual compressed by a
simple quantizer. The proposal is attractive because both financial return panels
and neural network weight matrices are widely described as low-rank or
approximately so.

The proposal carries a precondition that is frequently left implicit. The
composite representation improves on flat quantization only if the residual E is
*near-isotropic*. If E retains strong spectral structure after the factor core is
removed, then the core has been paid for and the residual remains a hard
compression problem; the composite is strictly worse than the flat code it was
meant to beat. The precondition is cheap to test — one singular value
decomposition on representative data — and it is decisive.

This paper reports four experiments that tested the hypothesis on real data, and
reports that it is refuted for the two data classes the program cared about. The
paper is published because a refutation of a widely assumed precondition is
useful to others considering the same design, and because the structured
compression literature is not well served by a record that reports only the cases
where the approach succeeded.

### 1.1 Contributions

1. A precondition test for residual isotropy, applied to real financial return
   panels before any codec was constructed, which refuted the hypothesis in every
   regime examined (§3).
2. A matched-bits-per-element comparison of structured decomposition against flat
   quantization on real language model weights, in which the structured method
   does not win (§5).
3. A positive transfer result: the key component of the attention cache is
   low-rank in the sense the hypothesis requires, and a low-rank core with a
   quantized residual improves on flat cache quantization at aggressive budgets,
   with the acausal-basis limitation stated explicitly (§6).
4. A transfer result orthogonal to compression: a calibrated error bar on
   analytics computed from compressed data holds across financial panels, weights
   and cache, and is the artifact the remainder of the series develops (§7).
5. A statement of scope boundaries that prevents the negative result from being
   read more broadly than the evidence supports (§8).

### 1.2 Position in the series

This paper reads first. It records why the compression objective was abandoned
and why a certificate became the surviving artifact, which is the premise the
other four papers assume. Papers II and I develop the certificate for dithered
storage and for committed computation respectively.

---

## 2. Notation and preliminaries

| Symbol | Meaning |
|---|---|
| X | Data matrix under compression; rows are observations, columns are coordinates |
| L, F | Low-rank factor core; L holds loadings and F holds factors |
| E | Residual, E = X − LF |
| r | Rank of the factor core |
| σ₁ ≥ σ₂ ≥ … | Singular values of the matrix under discussion |
| b | Bits per element of the storage code |
| R² | Coefficient of determination of realized error regressed on predicted error |

**Definition 2.1 (near-isotropy).** A residual E with empirical covariance Σ_E is
*near-isotropic* at tolerance τ when the spectrum of Σ_E is flat to within τ,
measured here as the ratio of the leading eigenvalue to the mean eigenvalue,
compared against the value that would arise from an equivalently shaped matrix of
independent entries. Isotropy is the condition under which a scalar quantizer
applied element-wise to E approaches the entropy of E; departure from it is
exactly the structure a scalar quantizer cannot exploit.

**Definition 2.2 (matched-bits comparison).** Two representations are compared at
*matched bits per element* when the total stored payload, including every
auxiliary quantity — factor core, scales, offsets, group metadata and codebooks —
is divided by the number of represented elements, and the two are evaluated at
equal values of that quotient. A comparison that omits auxiliary payload from one
side is not a matched-bits comparison.

**Definition 2.3 (pre-registered acceptance criterion).** A quantitative
threshold, together with the measurement that will be taken and the decision that
each outcome implies, recorded before the experiment is executed. The purpose is
to remove the analyst's discretion over what counts as success after the result
is visible.

---

## 3. EXP-1: the residual-isotropy precondition

**Design.** Real financial return panels were decomposed into a factor core of
several ranks, and the spectrum of the residual covariance was compared against
the isotropic reference. The pre-registered criterion was that the residual
spectrum be flat to within the stated tolerance in at least one regime, that
regime then defining the codec's operating point.

**Result.** The residual is not near-isotropic in any regime tested. Removing a
factor core leaves a residual whose spectrum remains far from flat. The
decomposition does not produce the condition the codec was designed to exploit.

**Interpretation.** This is a statement about the data, not about the estimator.
Financial return panels have a well-documented eigenvalue structure in which a
market-wide mode dominates and a band of further structured modes sits above the
noise bulk predicted by random matrix theory [8, 9]. Removing the leading mode
removes the largest of these and leaves the rest. The residual is therefore
structured by construction of the data, and no choice of rank recovers isotropy.

**Consequence.** The precondition failed before any codec existed. The
experiment cost one decomposition and its interpretation; the codec it prevented
would have cost substantially more. §9 returns to this as the paper's
methodological point.

---

## 4. EXP-2: certified analytics from compressed representations

**Design.** Two claims were tested separately. The first: that second-moment
analytics — covariance, market beta, correlation — can be computed directly from
a compressed representation with a calibrated error bar, rather than by
decompressing and recomputing. The second: that the compression achieves a ratio
advantage over flat quantization at equal accuracy.

**Result, first claim: upheld.** Predicted error tracked realized error with
R² ≈ 0.95 on a real returns panel. The error bar is calibrated in the sense that
its magnitude is informative about the realized error rather than merely bounding
it.

**Result, second claim: withdrawn.** The ratio advantage did not hold, and the
floor claim was retracted in the same review that retracted the corresponding
claim for certified quantiles (Paper II, §3).

**Consequence.** The certificate half of EXP-2 is the direct antecedent of the
decomposed error budget of Paper I and of the spectral certificate of Paper II.
The ratio half is closed.

---

## 5. EXP-LLM: transfer to language model weights

**Design.** Three parts. Part A applied the EXP-1 precondition test at scale to
real language model weight matrices. Part B was the decisive comparison:
structured decomposition against flat quantization **at matched bits per element**
in the sense of Definition 2.2. Part C measured perplexity against bit width on
a 0.5-billion-parameter model using a hand-written forward pass, so that the
measurement did not inherit an inference stack's own quantization behavior.

**Result: the hypothesis does not transfer.** Weight matrices are not
low-residual-rank in the sense the hypothesis requires. At matched bits, the
structured decomposition does not beat flat quantization.

**The governing statement**, recorded in the research ledger at the time of the
experiment:

> The compression ratio is governed by per-element bit width in both cases, which
> is captured by commodity int4/int8/fp16. For language models the additional
> lever is outlier and group handling, which is an established and heavily
> contested field.

That field is occupied by GPTQ [1], AWQ [2], QuIP and QuIP# [3, 4], AQLM [5] and
the LLM.int8() outlier decomposition [6], among others. A method entering it
competes on outlier handling, not on the presence of a factor core.

**A claim this work does not make.** Any figure of the form "31 GB reduced to
4 GB" attributable to this program describes commodity 4-bit quantization, which
achieves roughly a fourfold reduction and is a third-party result. It is not a
finding of this research and is not claimed as one.

**Certificate transfer.** The error bar transferred where the compression did
not. Predicted-versus-realized error had slope ≈ 1 and was monotonic with
perplexity. The certificate is informative about weights even though the
structured representation is not competitive on them.

---

## 6. EXP-KV: the one positive transfer

**Design.** The attention key-value cache was tested under the same hypothesis,
separately for keys and values, against flat cache quantization at aggressive bit
budgets.

**Result: positive, for keys, and modest.** The key component is genuinely
low-rank in the sense the hypothesis requires — unlike weights. A low-rank core
with a quantized residual improves on flat cache quantization at budgets of 2 to
3.5 bits per element, which is the regime in which long-context memory pressure
is binding.

**Limitation, stated as a bound rather than a caveat.** The low-rank basis in the
experiment was fitted on the whole sequence. A streaming decoder cannot do this,
because the basis would depend on tokens not yet emitted. The reported gain is
therefore an **upper bound** on what a causal implementation could achieve, and a
causal construction is a distinct and harder problem that was not attempted. Until
one exists, the result should not be read as a deployable ratio.

**Position against prior work.** The gain is incremental over KIVI [7] and
KVQuant [10] on ratio alone. If a differentiator exists in this direction it is
the certified sensitivity layer — which layers and positions tolerate aggressive
budgets, with a bound — rather than the compression itself.

---

## 7. Results summary

**Table 1.** Verdict by thread. Each row is the outcome against the criterion
registered before the corresponding experiment was run.

| Thread | Verdict | Evidence |
|---|---|---|
| Compression ratio as a novel structural advantage, financial panels | Refuted | EXP-1, EXP-2 |
| Compression ratio as a novel structural advantage, LM weights | Refuted | EXP-LLM Part B, matched bits |
| Low-rank core on weights | Not useful | EXP-LLM Part A |
| Low-rank core on returns residual | Not useful | EXP-1 |
| Low-rank core on attention key cache | Modestly useful | EXP-KV; acausal basis, upper bound |
| Calibrated error certificate | Transfers to all classes tested | R² ≈ 0.95 (panels); slope ≈ 1 (weights); monotonic (cache) |

The asymmetry in the final row is the paper's substantive result. The quantity
that transferred across every data class is not a representation but a *statement
about the error of a computation performed on a representation*.

---

## 8. Scope boundaries

The negative result concerns **weight matrices**, under **inference**. It was not
tested on activations, and it was not tested on training-time memory — optimizer
states, gradients and stored activations — which is a distinct regime with
different structure.

Reported gains from oblivious quantization methods in the literature live
substantially in cache and activations rather than weights, where the structural
story is stronger. EXP-KV probed one of those regimes and found the positive
result of §6. The refutation should not be generalized past weights on inference.

---

## 9. Limitations and open problems

1. **The precondition test is regime-dependent.** EXP-1 tested the panels and
   ranks that the intended application required. A different asset universe,
   sampling frequency or factor construction could in principle produce a residual
   closer to isotropy. The claim is that the precondition failed on the data the
   codec was for, not that it fails universally.
2. **The causal key-cache basis is UNMEASURED.** §6's gain assumes a basis fitted
   on the full sequence. No causal variant was constructed or evaluated, and the
   gap between the acausal bound and an achievable causal method is unknown.
3. **Part C used one model at one scale.** Perplexity against bit width was
   measured on a single 0.5-billion-parameter model. The matched-bits conclusion
   of Part B is the load-bearing result; the perplexity curve is corroborating,
   not independent, evidence.
4. **The certificate's calibration is measured, not proven.** R² ≈ 0.95 and unit
   slope are empirical properties of the tested panels and weights. Papers I and
   II construct the cases where a bound is a theorem; the EXP-2 certificate
   predates them and is a fitted relationship.
5. **No adversarial review of the compression results.** The four experiments
   were run against pre-registered criteria but were not subjected to the
   independent referee process that Papers I and II record. The refutations are
   the authors' own, and a refutation is the direction in which self-review is
   least likely to be biased, but the asymmetry is stated.

---

## 10. Methodological discussion

Three observations generalize past this program.

**The precondition test is the inexpensive experiment, and it is rarely run
first.** The question "is the residual actually near-isotropic on my data" costs
one decomposition. Run before construction, it would have prevented the codec
effort in its entirety. For any proposal of the form "structure S makes this data
compressible", the corresponding precondition test is the experiment worth
copying.

**"Commodity already achieves this" is a result, and reporting it requires
matched bits.** A structured method that does not state its total payload —
including cores, scales, group metadata and codebooks — divided by represented
elements is not comparable to a flat code. The Part B finding is that at matched
bits, on weights, the structured method loses. A matched-bits comparison should be
a reporting minimum rather than an unusual courtesy.

**The salvage exceeded the original objective in value.** A calibrated statement
of the form "here is the quantity, and here is a defensible error bar on it", or
"here is the layer that tolerates a two-bit budget, with a bound", is a narrower
claim than a compression advance. Unlike the compression advance, it survived
every test applied to it. The remaining four papers in this series develop it.

---

## 11. Conclusion

The central compression hypothesis was refuted for both data classes the program
targeted, by experiments designed to refute it and evaluated against criteria
registered in advance. The compression ratio is governed by per-element bit
width, which commodity integer formats already deliver, and the remaining lever
is outlier handling in an established and contested field. One positive transfer
was found in the attention key cache and is reported with an explicit upper-bound
caveat. The artifact that transferred to every data class is the certificate: a
calibrated statement about the error of a computation performed on compressed
data. That artifact, and not any representation, is the subject of the rest of
this series.

---

## Appendix A — Reproduction

Full result tables are recorded in the research ledger at `docs/RESEARCH.md`,
under the sections EXP-1, EXP-2, EXP-LLM and EXP-KV, with figures under
`docs/research/`. The certificate component that survived is implemented in
`alelyon.runtime.vector.compute` and in the decomposed error budget described in
Paper I.

> **Correction, 2026-08-05 — the scripts are not in the repository.** This
> appendix previously read "Experiment scripts and full result tables are
> recorded in the research ledger", which a reader would take as an offer to
> re-run the experiments. The ledger records the result tables and the method in
> detail, but it names the scripts as scratchpad files — the four exp1 pull,
> analyze, wall and figure scripts, the two exp2 codec and check scripts, the two
> exp_kv scripts, the instrumented qwen2 forward pass, and the tecert gate
> benchmark — and **none of the ten is tracked here.** They were run inline,
> outside version control, and are gone. They are described rather than named in
> backticks on purpose: a backticked filename reads as an artifact you can open,
> and these are not.
>
> What survives is therefore the ledger's prose: the construction of each panel,
> the baselines, the bit accounting and the measured numbers, which are enough to
> re-implement an experiment but not to re-run one. That is a materially weaker
> claim than this appendix made, and the distinction is exactly the one the
> program's own structural audit drew when it found a regression gate specified
> against 78 fixtures that were never committed: **do not leave a reproduction
> documented that the repository cannot execute.**
>
> This is recorded rather than repaired because reconstructing the scripts would
> produce *new* code attributed to a 2026-07-19 measurement it did not produce.
> The papers whose scripts ARE committed — Papers I and II, under
> `research/papers/` — are unaffected, and
> `tests/regression/test_paper_reproduction_claims.py` now fails if a paper names
> a script the repository does not carry.

> **Second correction, 2026-08-05 (EXP-3) — "gone" was wrong, and three of the ten
> are now tracked.** The correction above says the ten scripts "were run inline,
> outside version control, and are gone." The first half was true; the last word
> was not. They were still in the originating session's scratchpad under
> `%LOCALAPPDATA%\Temp\claude`, which is unswept rather than safe, and EXP-3 found
> them there while auditing the frontier programme's record. Recovering them
> raised no attribution problem at all — nothing was reconstructed, the 2026-07-19
> files were copied verbatim — so the reasoning above for recording rather than
> repairing applied to a reconstruction nobody needed to do.
>
> Committed at `research/experiments/expllm/` (in the private source repo):
> the two `exp_kv` scripts, the instrumented `qwen2_forward.py`, and the EXP-LLM
> Part A/B/C scripts and figure generators, with all nine raw result JSONs. Of the
> ten this appendix enumerates, **three are now tracked**; the four exp1 scripts,
> the two exp2 scripts and the tecert gate benchmark are not, and remain described
> rather than named. EXP-LLM Part C is still not re-runnable from a clean clone —
> it needs a 988 MB fp16 checkpoint that is deliberately not in git, recorded
> instead as a hashed `world-state` claim in `tools/claim_capsule.json`.
>
> **How this was caught is the point.** The sentence above became false the moment
> the files were committed, and no reader, linter or reviewer noticed — the test
> did, because it resolves against the *tracked file set* rather than against
> prose. A gate aimed at what produces the fact fires when the fact changes; one
> aimed at the narration cannot. That is the finding EXP-3 was running down when
> it walked into this appendix
> (docs/AI_INFRA_FRONTIER.md (in the private source repo), Part V).

**Environment.** Python 3.12; the canonical interpreter for this repository is
`.venv312/Scripts/python.exe`. Experiments used NumPy for the decompositions and a
hand-written forward pass for the perplexity measurement of EXP-LLM Part C, so
that the measurement did not inherit an inference framework's own quantization
behavior.

**Data availability.** The financial panels are drawn from the program's
committed history store (`globals/history.db`) and are not redistributable; the
panel construction is deterministic given the store, and the loaders are recorded
with the experiments. The language model weights are publicly available model
artifacts. No experiment in this paper requires network access at run time.

---

## Appendix B — Notation table

| Symbol | Definition | First use |
|---|---|---|
| X | Data matrix under compression | §2 |
| L, F | Low-rank factor core (loadings, factors) | §1 |
| E | Residual X − LF | §1 |
| r | Factor core rank | §2 |
| Σ_E | Empirical covariance of the residual | Def. 2.1 |
| τ | Isotropy tolerance | Def. 2.1 |
| b | Bits per element | Def. 2.2 |
| R² | Coefficient of determination, realized on predicted error | §4 |

---

## References

The identifiers below were recorded from the authors' working bibliography and
should be checked against the primary sources before external publication; this
series has not yet had a bibliographic review.

[1] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. *GPTQ: Accurate
Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023.
arXiv:2210.17323.

[2] J. Lin, J. Tang, H. Tang, S. Yang, W.-M. Chen, W.-C. Wang, G. Xiao, X. Dang,
C. Gan, S. Han. *AWQ: Activation-aware Weight Quantization for LLM Compression
and Acceleration.* MLSys 2024. arXiv:2306.00978.

[3] J. Chee, Y. Cai, V. Kuleshov, C. De Sa. *QuIP: 2-Bit Quantization of Large
Language Models With Guarantees.* NeurIPS 2023. arXiv:2307.13304.

[4] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. *QuIP#: Even Better LLM
Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024.
arXiv:2402.04396.

[5] V. Egiazarian, A. Panferov, D. Kuznedelev, E. Frantar, A. Babenko,
D. Alistarh. *Extreme Compression of Large Language Models via Additive
Quantization.* ICML 2024. arXiv:2401.06118.

[6] T. Dettmers, M. Lewis, Y. Belkada, L. Zettlemoyer. *LLM.int8(): 8-bit Matrix
Multiplication for Transformers at Scale.* NeurIPS 2022. arXiv:2208.07339.

[7] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, X. Hu.
*KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
arXiv:2402.02750.

[8] L. Laloux, P. Cizeau, J.-P. Bouchaud, M. Potters. *Noise Dressing of
Financial Correlation Matrices.* Physical Review Letters 83(7), 1999.

[9] V. A. Marchenko, L. A. Pastur. *Distribution of Eigenvalues for Some Sets of
Random Matrices.* Matematicheskii Sbornik 72(4), 1967.

[10] C. Hooper, S. Kim, H. Mohammadzadeh, M. W. Mahoney, Y. S. Shao, K. Keutzer,
A. Gholami. *KVQuant: Towards 10 Million Context Length LLM Inference with KV
Cache Quantization.* NeurIPS 2024. arXiv:2401.18079.

[11] C. Eckart, G. Young. *The Approximation of One Matrix by Another of Lower
Rank.* Psychometrika 1(3), 1936.
