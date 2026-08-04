# What Does Not Transfer

### Four kill-tests on structured compression, and the one thing that survived

**Status:** negative-results paper. Four experiments (EXP-1, EXP-2, EXP-LLM,
EXP-KV), run 2026-07-19. The program's central compression hypothesis was
**refuted** and the moat closed. Published because the refutation is the finding.

---

## Abstract

We set out to build a proprietary compression engine — a "certified
factor-residual codec" (CFRC) — on the hypothesis that financial return panels
and neural network weights are both low-residual-rank: that after removing a
small factor core, what remains is compressible in a structured way that
commodity per-element quantisation cannot reach.

We ran four kill-tests. The hypothesis is **false for financial panels and false
for LLM weights**. The compression ratio is governed by per-element bit width in
both cases, which is exactly what commodity int4/int8/fp16 already captures. The
"revolutionary ratio" ambition is closed.

One thing transferred to every data class we tested: **the certificate** — a
calibrated, honest error bar on a quantity computed from compressed data. That
is the salvage, and it is what became the rest of this program.

We publish this because the four negatives are individually useful, and because
the literature on structured compression is not well served by papers that only
report where it worked.

## 1. The hypothesis

If a data matrix decomposes as `X = LF + E` with `L` a low-rank factor core and
`E` a near-isotropic residual, then `E` compresses to near its entropy with a
simple scalar quantiser while `L` is stored cheaply, and the composite beats flat
quantisation at matched bits per element.

The condition that matters is **near-isotropy of the residual**. If `E` retains
strong structure, the factor core has not bought anything: you have paid for `L`
and still have a hard problem.

## 2. EXP-1 — the residual-isotropy kill-test

**Result: the residual is NOT near-isotropic, in any regime tested.**

This is the precondition for the entire thesis, tested first and deliberately, on
real return panels. It failed. Removing a factor core leaves a residual whose
spectrum is still far from flat, so the decomposition does not deliver the
regime the codec was designed for.

Running the precondition test before building the codec is the only thing we
would claim we did right in this sequence.

## 3. EXP-2 — certified compressed analytics

**Result: mixed — and the surviving half became the program.**

What survived: computing analytics (covariance, beta, correlation) directly from
compressed representations, **with a calibrated error bar**, works. The predicted
error tracked the realised error with **R² ≈ 0.95** on a real returns panel.

What did not: the compression-ratio advantage. The floor claim was retracted in
the same review that killed the equivalent claim in SRC-VaR (paper 5).

The certificate half is the direct ancestor of the decomposed error budget in
paper 1 and the spectral certificate in paper 6.

## 4. EXP-LLM — does it transfer to weights?

**Result: no. Decisively.**

Part A tested the precondition at scale on real LLM weights. Part B was the
decisive test: **CFRC versus FLAT quantisation at matched bits per element**. Part
C measured real perplexity against bits on Qwen-0.5B with a hand-written forward
pass.

The factor core does not help. LLM weight matrices are not low-residual-rank in
the way the hypothesis needs. At matched bits, structured decomposition does not
beat flat quantisation.

Meanwhile the **certificate transferred**: predicted-versus-realised error slope
≈ 1, monotonic with perplexity. The error bar works on weights even though the
compression does not.

**The blunt conclusion**, recorded verbatim in the research ledger:

> The compression RATIO is governed by per-element bit-width in both — captured by
> commodity int4/int8/fp16; for LLMs the extra lever is outlier/group handling, a
> solved and heavily contested field (GPTQ/AWQ/QuIP#/AQLM/GGUF). No revolutionary
> "31GB→4GB novel method" is reachable here; ~4× via 4-bit is commodity.

Any "31GB → 4GB" figure attributable to this work is a **third-party claim about
commodity 4-bit quantisation**, not a result of this research.

## 5. EXP-KV — the one positive transfer

**Result: the first and only positive transfer of the low-rank core to LLM data.**

The KV-cache — specifically the keys — **is** genuinely low-rank, unlike weights.
A low-rank-core plus quantised-residual codec beats flat KV quantisation at
aggressive 2–3.5 bit budgets, which is exactly where the long-context RAM
bottleneck lives.

Real, and modest. Caveated by an **acausal basis**: the low-rank basis in the
experiment was fitted using the whole sequence, which a streaming decoder cannot
do. A causal version is a different and harder problem, and the reported gain
should be read as an upper bound until one exists.

Incremental over KIVI and KVQuant on ratio. The differentiator, if there is one,
would again be the certified sensitivity layer rather than the compression.

## 6. Scope boundaries — do not over-generalise the negative

We tested **weights** (inference RAM). We did **not** test activations, and we did
not test training-RAM (optimizer states, gradients, activations), which is a
distinct regime.

TurboQuant's reported gains live substantially in KV-cache and activations, where
the structure is different and the oblivious-quantisation story is genuinely
stronger. EXP-KV probed one of those and found the positive result described
above. The negative result is about **weights**, on **inference**, and should not
be read more broadly than that.

## 7. The honest map

| thread | verdict |
|---|---|
| compression ratio as a novel-CS moat | **closed / negative** — financial panels AND LLM weights |
| low-rank core on weights | **useless** |
| low-rank core on returns residual | **useless** |
| low-rank core on KV-cache | **modestly useful** (acausal basis caveat) |
| **the certificate** | **transfers to all of them** — finance covariance R² ≈ 0.95; LLM weights slope ≈ 1; KV monotonic |

## 8. What we think the finding is worth

Three things.

**The precondition test is the cheap experiment and almost nobody runs it first.**
"Is the residual actually near-isotropic on my data?" is one SVD and an afternoon.
It would have saved the entire codec effort. If you are considering a structured
compression scheme, EXP-1 is the experiment to copy.

**"Commodity already does this" is a real result.** A substantial amount of
structured-compression work competes against flat quantisation without saying so
at matched bits per element. Our Part B result is that at matched bits, on
weights, the sophisticated method loses. Reporting the matched-bits comparison
should be table stakes.

**The salvage is worth more than the original target.** Nobody ships "here is the
number *and* a trustworthy error bar on it" or "here is which layer you can
safely push to two bits". That is a smaller claim than a compression
breakthrough, and unlike the compression breakthrough it survived every test we
put it through.

## 9. Reproduction

The experiment scripts and full result tables are in the research ledger
(`docs/RESEARCH.md`, EXP-1 / EXP-2 / EXP-LLM / EXP-KV sections), with figures in
`docs/research/`. The certificate that survived is the shipped
`alelyon.runtime.vector.compute` and the error budget in paper 1.
