# Certified Statistics from Subtractively Dithered Storage: A Ladder of Downstream Objects, and Where It Terminates

**Alelyon Quantitative Services**
Working Paper II of V · Series: *Certified Computation over Compressed and Committed Data*

**Version:** 1.0 · **Date:** 2026-08-04 · **Status:** Working paper; not peer reviewed.
**Artifacts:** `alelyon.runtime.vector.{codec.certkit, speccert, ditherdmd, qpregret}`.
**Adversarial review:** the decision-regret construction was refereed (verdict
*repairable*, repaired; §6.3); the scalar-functional headline was **retracted** in
review (§3.5).
**Correspondence:** Alelyon Quantitative Services.

---

## Standing honesty contract

1. **Name the certified object.** Every bound in this paper is over **storage
   quantization**, conditional on a fixed data path. None of them says anything
   about how far the clean sample is from the population it came from. That is a
   separate term, and §3.5 is the measurement of how much larger it is.
2. **Revision is detected; invention is not.** These certificates take Δ as a
   declaration. A false Δ yields a valid certificate over false premises.
3. **An absent measurement is reported as UNMEASURED.**
4. **Refusal is a first-class outcome.** Three of the four constructions below
   refuse on real data at realistic storage depths, and those refusals are the
   paper's principal empirical content.

---

## Abstract

When numeric data is stored with subtractive dither — a known pseudorandom offset
added before quantization and subtracted after — the storage error is uniform on
(−Δ/2, Δ/2), independent of the signal, and known in distribution rather than
assumed. This paper asks what can be certified downstream of that premise, and
constructs a ladder of four rungs: scalar functionals of the empirical
distribution (quantiles, value-at-risk, expected shortfall); second moments (the
sample covariance in operator norm); linear operators (the ridge dynamic-mode
decomposition fit); and decisions (the solution of a mean-variance quadratic
program). The rungs are not independent: the second-moment certificate's bound
enters the decision certificate's solution radius as a term. Every constant is
computable from stored values, and every construction refuses by name rather than
degrading. Two cross-cutting negative results are the paper's payload. The first is
a **conditioning ceiling**: certifying an intermediate object proved far easier
than certifying the decision depending on it, observed twice from independent
directions. New work reported here shows that one of those two observations was an
**artifact of the bound rather than of the problem** — replacing a Bauer–Fike
eigenvalue inclusion with a direct spectral inclusion, which is elementary and
provably optimal given an operator-norm radius, moves the contraction gate from
never firing to firing five to six bits earlier, a 32- to 64-fold relaxation in
required storage precision. The second observation, at the decision rung, is not
resolved by the same device, and whether it is similarly an artifact is now the
leading open question. The second negative is **allocation futility**: two
independently constructed bit allocators, optimizing the certificate directly,
failed to beat mean-squared-error-optimal allocation, by +0.01%/+0.18% in one
program and by −203% to −1294% in the other. Neither was released.

**Keywords:** subtractive dither, matrix concentration, conformal certification,
dynamic mode decomposition, pseudospectra, distance to instability, mean-variance
optimization, bit allocation, negative results.

---

## 1. Introduction

A certificate over stored data is only as useful as the object it certifies. A
bound on the storage error of a *number* is of limited interest; a bound on the
storage error of the *decision* that number feeds is what an operator needs. This
paper constructs the intervening ladder and reports where it stops.

The premise throughout is subtractive dither. Under conditions due to
Schuchman [1], adding a known pseudorandom offset before quantization and
subtracting it after makes the resulting error uniform, signal-independent and
independent across elements — properties that ordinary rounding does not have and
that no amount of analysis can recover from a merely interval-bounded error. That
premise is strong enough to certify things, and the four rungs below establish
what.

### 1.1 The ladder

| Rung | Object certified | Machinery | Section |
|---|---|---|---|
| 1 | Scalar functionals of the empirical distribution | Order statistics, DKW | §3 |
| 2 | Second moments (sample covariance, operator norm) | Matrix Bernstein | §4 |
| 3 | Linear operators (ridge DMD fit) | Resolvent bootstrap + matrix Bernstein | §5 |
| 4 | Decisions (mean-variance QP solution) | KKT margins under a certified radius | §6 |

The rungs are coupled. The certificate of Theorem 4.6 appears as the term ε_Σ in
Lemma 6.2, and the surrogate machinery of Lemmas 4.4 and 4.5 is reused unmodified
in the Gram bound of §5. This is one construction observed at four altitudes, not
four constructions.

### 1.2 Contributions

1. A single consolidated storage model and validation protocol, replacing four
   independently derived preliminaries, with the Schuchman conditions stated in
   full and the storage obligations they impose made explicit (§2).
2. Four rungs of certified downstream objects, each with its constants computable
   from stored values alone and each with named refusal paths (§§3–6).
3. **Theorem 5.4 and Corollary 5.5**, new in this paper: a direct spectral
   inclusion and a contraction gate that carry no eigenbasis conditioning factor,
   together with **Proposition 5.6**, that the resulting criterion is optimal
   given only an operator-norm radius (§5.4).
4. A revised account of the conditioning ceiling: one of its two observed
   instances is an artifact of Bauer–Fike and dissolves; the other does not, and
   the asymmetry between them is characterized (§7).
5. Allocation futility, with the mechanism identified: mean-squared-error-optimal
   allocation already minimizes the dominant term of both certificates (§8).
6. A retraction, reported at the prominence of a result: the claim that a
   certified population band could be narrower than the storage resolution was
   wrong, and the measured gap grows to 457× at 12 bits (§3.5).

### 1.3 Position in the series

Paper IV records why a certificate rather than a compression ratio became this
program's objective. Paper I describes the signed object that transports a
certificate to a third party. This paper is the mathematics.

---

## 2. Storage model, assumptions and validation protocol

This section replaces four separately derived preliminaries. It is stated once,
in full, because the difference between its hypotheses and their weaker
interval-bounded cousin is load-bearing in every section that follows and is the
subject of a confirmed defect reported in Paper I.

### 2.1 The storage model

Let x ∈ ℝ be a value to be stored at step Δ > 0. Let d be a dither signal,
uniform on (−Δ/2, Δ/2), independent of x, and reproducible by the decoder from a
committed seed. Subtractive dither stores

  y = Q(x + d) − d,  Q the uniform quantizer at step Δ.  (2.1)

**Lemma 2.1 (Schuchman; exact error model — cited, not claimed).** Under (2.1),
provided no code is clipped, the error e := y − x is uniform on (−Δ/2, Δ/2),
statistically independent of x, and independent across elements with independent
dither draws. Consequently

  E[e] = 0,  E[e²] = Δ²/12,  E[e⁴] = Δ⁴/80.  (2.2)

*Attribution.* Schuchman [1]; the modern treatment is Lipshitz, Wannamaker and
Vanderkooy [2] and Gray and Stockham [3]. Nothing in this paper claims this
result; everything in it depends on the hypotheses being met.

**Assumption 2.2 (no overload).** No stored code lies at a quantizer boundary. The
number of clipped codes is observable at decode; a nonzero count voids Lemma 2.1
and every construction below refuses on it.

**Assumption 2.3 (independent dither across elements).** Independence is *not*
automatic when the dither is a deterministic function of a seed and a shape. Where
a trajectory is stored snapshot by snapshot, the encoder must draw a separate,
independent seed per snapshot. A reused seed silently voids every concentration
bound and **no runtime observable exists** for it, so the property is asserted at
encode time and the encoder refuses a caller-forced reuse.

**Definition 2.4 (observable surrogate).** For a stored vector y with step Δ, the
entrywise surrogate is x̄ := |y| + Δ/2. By Lemma 2.1 and Assumption 2.2, x̄
dominates |x| entrywise with certainty. Every "conditional" constant below is
built from x̄, which is why it is computable from stored values.

**Definition 2.5 (per-column heterogeneity).** For a panel with columns j, the
step is Δ_j and s_j := Δ_j². Steps are heterogeneous in every real panel used
here; a certificate assuming a common Δ is not applicable.

### 2.2 Validation protocol

Unless stated otherwise, every empirical result below uses the following protocol,
which is the same in all four sections.

- The **data path is fixed**. Randomness is over the injected dither only. This is
  the probability space of the theorems: data fixed, dither redrawn.
- **500 dither redraws** per cell (100 in §5.4's extended sweep, where the
  additional quantity computed per redraw is a batched singular value
  decomposition over a grid).
- **δ = 0.05** unless stated.
- Data are read **read-only from committed history**; no experiment requires
  network access.
- **Acceptance criteria are registered before execution** (Definition 2.6).

**Definition 2.6 (pre-registered acceptance criterion).** A quantitative threshold,
the measurement that will be taken, and the decision each outcome implies, recorded
before the experiment runs. Two constructions below failed their criteria and were
not released; §8 is that result.

---

## 3. Rung 1: scalar functionals of the empirical distribution

Two targets must be distinguished, and conflating them is precisely the error the
retraction of §3.5 corrects: the **empirical** quantile of the clean sample held,
and the **population** quantile of the distribution the sample came from.

### 3.1 The deterministic sandwich

**Theorem 3.1 (empirical sandwich; certainty 1).** Under Lemma 2.1 and
Assumption 2.2, the clean empirical q-quantile of a sample lies in
[y_(k) − Δ/2, y_(k) + Δ/2] for **every** dither realization, where y_(k) is the
corresponding order statistic of the stored sample. The statement carries no
distributional assumption and holds with probability one.

**Table 1.** Coverage of Theorem 3.1 on a real price series, q ∈ {0.01, 0.05,
0.95, 0.99}, 500 dither redraws at each depth. 10,000 checks total. The value must
be exactly 1.0000; anything less falsifies a deterministic claim.

| b | Δ | Coverage |
|---|---|---|
| 4 | 1.258e-2 | 1.0000 |
| 6 | 3.146e-3 | 1.0000 |
| 8 | 7.865e-4 | 1.0000 |
| 10 | 1.966e-4 | 1.0000 |
| 12 | 4.916e-5 | 1.0000 |

This is the strongest result in the paper and the one with the fewest hypotheses.
It is also the least ambitious: it bounds error against the sample held, not
against the population.

### 3.2 Population smoothing bias

Dithering convolves the population distribution function with a uniform kernel,
which biases the population quantile.

**Theorem 3.2 (exact population smoothing bias).** Let F be the population
distribution function with density f satisfying |f′| ≤ L. Let F_Y be the
distribution function of the dithered variable. Then

  sup_t |F_Y(t) − F(t)| ≤ L·Δ²/24.  (3.1)

**Table 2.** Validation of (3.1) on a moment-matched two-component Gaussian
mixture with known L = 1061. The ratio approaching 1.000 from below shows the
constant is tight, not merely valid.

| b | Δ | Measured sup&#124;F_Y − F&#124; | Bound L·Δ²/24 | Ratio |
|---|---|---|---|---|
| 4 | 3.511e-2 | 4.724e-2 | 5.448e-2 | 0.867 |
| 6 | 8.776e-3 | 3.373e-3 | 3.405e-3 | 0.991 |
| 8 | 2.194e-3 | 2.127e-4 | 2.128e-4 | 0.999 |
| 10 | 5.485e-4 | 1.330e-5 | 1.330e-5 | 1.000 |
| 12 | 1.371e-4 | 8.314e-7 | 8.314e-7 | 1.000 |

### 3.3 The certified population band

**Theorem 3.3 (certified population band).** Composing a Dvoretzky–Kiefer–
Wolfowitz finite-sample band [4, 5] at level 1 − δ with the bias term of Theorem
3.2 yields a band containing the population quantile with probability at least
1 − δ, computable from stored values and δ.

Validated on synthetic independent data at q = 0.05, δ = 0.05, n = 5000 per
redraw, 400 redraws, b = 8: true population quantile −2.783e-2; bands emitted
400 of 400; coverage 1.0000 against a 0.93 pre-registered threshold.

### 3.4 Two-resolution extrapolation as a test of the declared regime

**Theorem 3.4 (Richardson two-resolution extrapolation).** Storing at two
resolutions and combining as (4F^{Δ/2} − F^{Δ})/3 cancels the Δ² term of Theorem
3.2. The residual order depends on the smoothness regime: O(Δ⁴) with constant
Δ⁴/4608 when f″ is Lipschitz, and O(Δ³) with constant Δ³/384 when only f′ is.

**Table 3.** Measured convergence order against Δ. The order is itself a
measurement of the smoothness declaration.

| Declared regime | Predicted order | Measured slope | Criterion |
|---|---|---|---|
| f″-Lipschitz (smooth Gaussian mixture) | O(Δ⁴) | 3.989 | ≥ 3.5, met |
| f′-Lipschitz (quadratic B-spline; f″ has jumps) | O(Δ³) | 2.973 | ≥ 2.8, met |

The consequence of practical interest is that a caller declaring more smoothness
than the data possesses can be detected by the slope.

**Proposition 3.5 (observable smoothness falsifier).** The population certificate
requires an L-Lipschitz density. The hypothesis is falsified from stored data by a
window-mass test: if the mass in the densest 2Δ window exceeds the maximum
compatible with the declared L, no such density exists.

Applied to a real price series on a coarse tick grid — 1500 closes with 637
distinct values, i.e. genuine point masses — at b = 8, Δ = 1.027e-1: densest 2Δ
window mass 0.2533 against an L-Lipschitz threshold of 0.1093. The falsifier
fired, the population band **refused** with reason `SMOOTHNESS_FALSIFIED`, and the
assumption-free sandwich of Theorem 3.1 remained valid and contained the clean
quantile across 200 redraws.

This is the rung's most useful output. The smoothness assumption is not declared
and hoped for; it is something the data can reject.

### 3.5 Retraction: dither buys the error model, not a smaller floor

The original framing of this rung claimed the certified population band could be
narrower than the storage resolution. Referee review refuted it. The corrected
comparison is reported as plain constant factors with no order-of-magnitude claim.

**Table 4.** Certified population band against the per-sample resolution. The band
is floored by the DKW sampling term √(ln(2/δ)/2n), which does not shrink with bit
depth at all, so the ratio *grows* with b.

| b | Δ | Band 4β/f_lb | Sandwich Δ | Asymptotic LΔ²/12f_lb | Band / sandwich |
|---|---|---|---|---|---|
| 4 | 3.511e-2 | 2.403e-1 | 3.511e-2 | 8.882e-2 | 6.84 |
| 6 | 8.776e-3 | 7.372e-2 | 8.776e-3 | 5.551e-3 | 8.40 |
| 8 | 2.194e-3 | 6.332e-2 | 2.194e-3 | 3.470e-4 | 28.9 |
| 10 | 5.485e-4 | 6.266e-2 | 5.485e-4 | 2.168e-5 | 114 |
| 12 | 1.371e-4 | 6.262e-2 | 1.371e-4 | 1.355e-6 | 457 |

**Dither buys the exact, checkable error model. It does not buy a smaller floor.**

This is the same conclusion the decomposed error budget of Paper I reaches from a
different direction — sampling binds, quantization does not — and the convergence
of two independent derivations on it is the most reliable finding in this research
line.

---

## 4. Rung 2: second moments

Given a panel stored with per-column subtractive dither at heterogeneous depths,
how far can the sample covariance computed from stored values be from the
covariance of the clean values, in operator norm?

### 4.1 Exact second-moment structure

**Lemma 4.1 (exact fourth-moment computation).** With E e_j² = s_j/12 and
E e_j⁴ = s_j²/80 from (2.2), all moments entering the matrix-variance computation
below are exact rather than bounded.

**Lemma 4.2 (positive-semidefinite per-column decomposition).** Define, per row t
and column j, the per-column contribution Z_{t,j} to the error matrix
Σ̂ − Σ_clean. The matrix variance decomposes into a sum of positive-semidefinite
per-column terms, each computable from the surrogate of Definition 2.4. This is
the load-bearing identity of the rung: without positive semidefiniteness the terms
cannot be bounded separately without loss.

**Corollary 4.3 (matrix-variance function).** The matrix-variance function ν(s)
assembled from Lemma 4.2 is computable from stored values. *A convexity claim
attached to this corollary in an earlier draft was corrected in referee review;
the corrected statement is what §4.4's allocator uses.*

**Lemma 4.4 (surrogate domination).** Let Ā_j^abs be the per-column matrix built
from the surrogate x̄_t of Definition 2.4 in place of x_t. Then Ā_j^abs dominates
its clean counterpart entrywise, and hence — by a Perron argument on nonnegative
matrices — in spectral norm, **surely** rather than with high probability.

**Lemma 4.5 (range term).** The range of the summands satisfies
R(s) ≤ R_obs(s) := 2·(max_t ‖x̄_t‖₂)·r(s) + r(s)² + s_max/12, with
r(s) := ½√(Σ_j s_j), computable from stored values.

### 4.2 The certificate

**Theorem 4.6 (conditional operator-norm certificate).** With probability at least
1 − δ over the injected dither, conditional on the fixed data path,

  ‖Σ̂ − Σ_clean‖_op ≤ B(δ, s) := √(2·ν̄_obs·ℓ) + (2/3)·R_obs·ℓ,  ℓ := ln(2p/δ),  (4.1)

with ν̄_obs from Corollary 4.3 and R_obs from Lemma 4.5. Every constant is computed
from (Y, Δ). The result is an instantiation of the matrix Bernstein inequality
[6, 7]; the concentration inequality is cited, and the computability of its
constants from stored values is what is claimed.

**Table 5.** Coverage of Theorem 4.6, 500 dither redraws, δ = 0.05. Two real
panels: EQUITY (T = 1499, p = 200, daily log-returns, column σ ∈ [9.3e-3,
3.65e-2]) and MACRO (T = 1199, p = 32, daily first differences, column σ ∈
[4.0e-3, 1.8475], deliberately harder because its column scales span three orders
of magnitude).

| Panel | Covered / K | Coverage | Floor | Mean B | Mean realized | Max realized |
|---|---|---|---|---|---|---|
| EQUITY | 500/500 | 1.0000 | 0.95 | 6.374e-4 | 6.328e-5 | 1.024e-4 |
| MACRO | 500/500 | 1.0000 | 0.95 | 1.167e-1 | 1.722e-2 | 3.598e-2 |

Coverage of 1.0000 against a 0.95 floor is the expected behavior of a conservative
two-term Bernstein bound. The mean bound is roughly 10× the mean realized error,
which is the price of that conservatism and is reported rather than smoothed over.

### 4.3 Conditioning is load-bearing

The pre-registered criterion for this rung asked whether *conditioning* — building
constants from the stored data via Definition 2.4 rather than from a
data-independent global envelope — does any work. The criterion was a ratio of at
least 10×.

**Table 6.** Conditional against unconditional bound. Identical Bernstein
machinery and δ; the only difference is the surrogate.

| Panel | B conditional | B unconditional | Ratio | Criterion |
|---|---|---|---|---|
| EQUITY | 6.377e-4 | 2.035e-2 | **31.9×** | ≥ 10× |
| MACRO | 1.1659e-1 | 2.64498 | **22.7×** | ≥ 10× |

This is the rung's positive result. A certificate computed from what was actually
stored is worth roughly an order of magnitude and a half over one computed from
what might have been stored.

### 4.4 Structure, closed form, and an observable regime test

**Theorem 4.7 (structure and closed form).** The bound (4.1) admits a
water-filling minimizer over per-column bit depths at fixed total budget, with
weights a_j = λ_max(Ā_j^abs). Part (a) — the objective's structure — was corrected
in referee review. Part (c) is an **observable regime condition**: the closed form
is valid for the objective only where the quadratic and range terms satisfy two
inequalities computable from stored values.

**Table 7.** Frequency with which the closed form is admissible on real panels.

| Panel | Closed form admissible | Detail |
|---|---|---|
| EQUITY | 1 of 8 budgets (12%) | Variance check passes everywhere; range check fails at every b_avg ≥ 3 |
| MACRO | 0 of 8 (0%) | Range check fails at every depth tested |

Where the regime condition fails the allocator ships a geometric-program minimizer
of the surrogate objective instead, and never the closed form out of regime. A
regime check that passed everywhere would be a rubber stamp; this one fails in
seven of eight and eight of eight cases on real data, which is the evidence that
it tests something.

**Proposition 4.8 (separation).** Certificate-optimal allocation is a minimizer of
(4.1) and mean-squared-error-optimal allocation is not, so the difference
B_mse − B_cert is nonnegative. This is a worked-example-grade statement, renamed
from "theorem" in referee review. §8 reports its measured magnitude.

---

## 5. Rung 3: linear operators

Let x_1, …, x_{W+1} ∈ ℝ^k be a **fixed, arbitrary bounded** snapshot trajectory —
no stationarity, mixing or noise-model assumption on the dynamics; randomness is
over injected dither only. Store it under (2.1) with per-snapshot independent
seeds (Assumption 2.3). Write X₋ = [x_1 … x_W], X₊ = [x_2 … x_{W+1}], and Y₋, Y₊
for their stored versions, with dither matrices N₋, N₊. Note the shift: N₊ and N₋
share the columns for t = 2 … W, which is the only dependence in the problem and
is confined to one term below.

**Clean target.** For a declared ridge λ ≥ 0,

  A_λ := X₊X₋ᵀ(X₋X₋ᵀ + λI)^{−1},  (5.1)

the ridge dynamic-mode-decomposition fit an analyst would compute from
full-precision data at the same λ [8, 9].

**Debiased estimator.** Ĝ := Y₋Y₋ᵀ − W·D with D := diag(s_i/12), so E[Ĝ] = X₋X₋ᵀ
exactly; Ĉ := Y₊Y₋ᵀ, so E[Ĉ] = X₊X₋ᵀ exactly because the shift makes
E[N₊N₋ᵀ] = 0; and Â := Ĉ(Ĝ + λI)^{−1}.

### 5.1 Error decomposition and the bootstrap

**Lemma 5.1 (resolvent identity).** Â − A_λ = (E_C − A_λE_G)(Ĝ + λI)^{−1}, where
E_G := Ĝ − X₋X₋ᵀ and E_C := Ĉ − X₊X₋ᵀ.

*Proof.* Â(Ĝ+λI) = Ĉ = C + E_C and A_λ(G+λI) = C, so
(Â − A_λ)(Ĝ+λI) = C + E_C − A_λ(G + λI + E_G) = E_C − A_λE_G. Multiply on the
right by (Ĝ+λI)^{−1}, which exists whenever m̂ := λ_min(Ĝ) + λ > 0, an observable
condition. ∎

Hence ‖Â − A_λ‖ ≤ (‖E_C‖ + ‖A_λ‖·‖E_G‖)/m̂. The quantity ‖A_λ‖ is clean-side and
unavailable. It is eliminated self-consistently:

**Theorem 5.2 (observable-surrogate bootstrap).** Let c ≥ ‖E_C‖ and g ≥ ‖E_G‖ hold
on an event of probability at least 1 − δ. Let a := ‖Â‖ and m̂ := λ_min(Ĝ) + λ,
both observable. If g < m̂, then on the same event

  ‖Â − A_λ‖ ≤ r̂ := (c + a·g)/(m̂ − g).  (5.2)

If g ≥ m̂, **refuse** with reason `INSUFFICIENT_REGULARIZATION`.

*Proof.* Write r := ‖Â − A_λ‖. Then ‖A_λ‖ ≤ a + r, and Lemma 5.1 gives
r ≤ (c + (a+r)g)/m̂, so r(1 − g/m̂) ≤ (c + ag)/m̂, and r ≤ (c + ag)/(m̂ − g)
when g < m̂. ∎

**Theorem 5.3 (operator-norm certificate).** With probability at least 1 − δ over
the injected dither, conditional on the realized trajectory, (5.2) holds with c
and g computable from (Y₋, Y₊, Δ, δ, λ) at decode time. The Gram bound g reuses
the machinery of Lemmas 4.4 and 4.5 unnormalized with T := W. The cross term
decomposes as E_C = X₊N₋ᵀ + N₊X₋ᵀ + N₊N₋ᵀ, of which the first two are independent
sums handled by Hermitian dilation, and the third is 1-dependent and handled by
classical even/odd Bernstein blocking; its constants involve no data at all.

*Proof.* Union bound over the four events; Lemma 5.1 and Theorem 5.2 on the
intersection. ∎

**Table 8.** Coverage and width of Theorem 5.3, 500 redraws, δ = 0.05. STATE is
the real product state trajectory (k = 14 macro features, weekly); returns is 20
liquid names' daily log returns (k = 20).

| Data | b | ok / refuse | Coverage | Median realized | Median r̂ | ‖A_λ‖ | Verdict |
|---|---|---|---|---|---|---|---|
| STATE | 6 | 500 / 0 | 1.0000 | 1.378e-2 | 5.094e0 | 1.0493 | vacuous |
| STATE | 8 | 500 / 0 | 1.0000 | 3.382e-3 | 5.074e-1 | 1.0493 | partial |
| STATE | 10 | 500 / 0 | 1.0000 | 8.652e-4 | 1.101e-1 | 1.0493 | informative |
| returns | 6 | 0 / 500 | refuses | — | — | 0.4789 | `INSUFFICIENT_REGULARIZATION` ×500 |
| returns | 8 | 500 / 0 | 1.0000 | 4.460e-2 | 3.798e1 | 0.4789 | vacuous |
| returns | 10 | 500 / 0 | 1.0000 | 1.088e-2 | 6.153e-1 | 0.4789 | vacuous |

At b = 2 every seed on both trajectories refuses with
`INSUFFICIENT_REGULARIZATION`, raised **before** Â is formed, so no radius is
produced off a numerically meaningless operator norm.

### 5.2 The spectral corollary as originally constructed

The product-facing question is not the operator norm but whether the clean fit is
*contracting*: ρ(A_λ) < 1. The original construction routed Theorem 5.3 through a
Bauer–Fike inclusion [10]: if Â = VΛV^{−1} with observable κ₂(V), every eigenvalue
of A_λ lies within κ₂(V)·r̂ of an eigenvalue of Â, giving the gate

  ρ(A_λ) ≤ max_i |λ_i(Â)| + κ₂(V)·r̂ < 1.  (5.3)

**Table 9.** Gate (5.3) on the STATE trajectory. The clean fit is contracting —
ρ(A_λ) = 0.99264 — but by a margin of 7.346e-3.

| b | Median max&#124;λ_i(Â)&#124; | Median κ₂(V) | Median r̂ | Median bound (5.3) | Certifies |
|---|---|---|---|---|---|
| 6 | 0.9926 | 7.185e1 | 5.094e0 | 3.676e2 | 0 / 500 |
| 8 | 0.9926 | 7.220e1 | 5.074e-1 | 3.755e1 | 0 / 500 |
| 10 | 0.9926 | 7.060e1 | 1.101e-1 | 8.769e0 | 0 / 500 |

The gate never fires. Because it never fires and the clean operator is
contracting, there are zero consistency violations across all redraws. This was
originally reported as a conditioning ceiling: correct machinery, perfect
coverage, a conditioning quantity placing the required margin out of reach.

### 5.3 That reading was wrong

The ceiling was an artifact of the inclusion, not of the problem. The eigenbasis
condition number κ₂(V) ≈ 70.5 is a *bound* on eigenvalue sensitivity, and on this
operator it overstates the truth by a factor of 53.

### 5.4 Direct spectral inclusion

**Theorem 5.4 (direct inclusion).** If ‖Â − A‖₂ ≤ r, then every eigenvalue μ of A
satisfies σ_min(μI − Â) ≤ r.

*Proof.* μI − A is singular, so (μI − A)v = 0 for some unit vector v. Then
(μI − Â)v = (A − Â)v, whence σ_min(μI − Â) ≤ ‖(A − Â)v‖ ≤ r. ∎

The theorem requires no diagonalizability, no eigenbasis and no conditioning
factor. It states that the spectrum of A is contained in the r-pseudospectrum of
Â [11].

**Corollary 5.5 (contraction gate without eigenbasis conditioning).** Define

  s⋆ := min_{|z| = 1} σ_min(zI − Â).  (5.4)

If ρ(Â) < 1 and r < s⋆, then ρ(A) < 1.

*Proof.* Put A_t := Â + t(A − Â) for t ∈ [0,1]. Then ‖A_t − Â‖ ≤ t·r ≤ r < s⋆, so
by Theorem 5.4 no eigenvalue of any A_t lies on the unit circle. Eigenvalues
depend continuously on t, and at t = 0 all lie strictly inside the unit disk. A
continuous eigenvalue path leaving the open disk must meet the circle. Hence
ρ(A_1) = ρ(A) < 1. ∎

Both statements consume exactly the premise Theorem 5.3 already delivers, so
Corollary 5.5 is a drop-in replacement for (5.3) and introduces no new
probabilistic content.

**Computability.** The map z ↦ σ_min(zI − Â) is 1-Lipschitz in z, so over grid
arcs of chord c the true minimum is at least (grid minimum) − c/2. A coarse pass
certifies most of the circle and candidate arcs are refined, yielding a rigorous
lower bound on s⋆. The gate uses the lower bound, never the observed minimum.

**Proposition 5.6 (optimality).** Corollary 5.5 is unimprovable given only an
operator-norm radius. At a minimizing z⋆ = e^{iθ⋆} with singular vectors u, v
satisfying (z⋆I − Â)v = s⋆u, the perturbation E := s⋆uv^H has ‖E‖₂ = s⋆ and makes
z⋆ an eigenvalue of Â + E, with |z⋆| = 1. Hence no criterion depending only on
‖Â − A‖ ≤ r can accept r ≥ s⋆.

*Remark.* s⋆ is exactly the complex distance from Â to the set of matrices with an
eigenvalue on the unit circle — the discrete-time stability radius [12, 13, 14].
Proposition 5.6 is therefore a restatement of a known identity in the form the
certificate needs.

### 5.5 Measured effect

**Table 10.** Both gates evaluated on the same certificate, 100 redraws per cell,
δ = 0.05. The direct gate uses the rigorous lower bound on s⋆.

| Data | b | Median r̂ | Median κ₂(V) | Bauer–Fike fires | Median s⋆ bound | Direct fires |
|---|---|---|---|---|---|---|
| STATE | 6 | 5.066e0 | 6.576e1 | 0/100 | 5.608e-3 | 0/100 |
| STATE | 10 | 1.100e-1 | 7.065e1 | 0/100 | 5.574e-3 | 0/100 |
| STATE | 14 | 6.601e-3 | 7.051e1 | 0/100 | 5.574e-3 | 0/100 |
| STATE | **15** | 3.296e-3 | 7.05e1 | 0/40 | 5.574e-3 | **40/40** |
| STATE | 16 | 1.647e-3 | 7.051e1 | 0/100 | 5.574e-3 | 100/100 |
| STATE | 18 | 4.115e-4 | 7.051e1 | 0/100 | 5.574e-3 | 100/100 |
| returns | 8 | 3.878e1 | 4.496e1 | 0/100 | 7.216e-1 | 0/100 |
| returns | **10** | 6.153e-1 | 5.257e1 | 0/100 | 7.218e-1 | **100/100** |
| returns | 16 | 7.334e-3 | 5.543e1 | 100/100 | 7.217e-1 | 100/100 |

**Table 11.** The two criteria compared directly.

| Trajectory | Bauer–Fike threshold on r̂ | Direct threshold on r̂ | Ratio | First firing depth (BF → direct) |
|---|---|---|---|---|
| STATE | 1.042e-4 | 5.574e-3 | **53.5×** | ≈ 20–21 → **15** |
| returns | 1.523e-2 | 7.217e-1 | **47.4×** | 16 → **10** |

The direct gate fires five to six bits earlier, a 32- to 64-fold relaxation in
required storage precision for the same conclusion. On the STATE trajectory the
true non-normality penalty is s⋆/(1 − ρ) = 0.759, a factor of 1.32 — not the 70.5
Bauer–Fike charges.

**Falsification.** Theorem 5.4 was checked against the true clean eigenvalues on
every certificate evaluated: 0 violations in 1300 evaluations. Corollary 5.5's
soundness was searched adversarially over 300 fired gates × 61 perturbations each,
including the extremal direction of Proposition 5.6; worst observed ρ(A + E) =
0.999563 < 1, never broken.

**Table 12.** Proposition 5.6 attained, and the real-versus-complex gap. The
extremal perturbation reaches the unit circle exactly in every case, which is what
makes the criterion unimprovable. The Bauer–Fike threshold is the radius (5.3)
would have required.

| Case | s⋆ (lower bound) | ‖E‖ | ρ(A+E) | Bauer–Fike threshold | κ₂(V) | Best **real** ρ at ‖E‖ |
|---|---|---|---|---|---|---|
| random normal | 7.000e-2 | 7.000e-2 | 1.000000 | 7.000e-2 | 1.000 | 1.0000 |
| mild non-normal | 4.639e-2 | 4.639e-2 | 1.000000 | 4.747e-3 | 14.75 | 0.9907 |
| strong non-normal | 2.880e-3 | 3.564e-3 | 1.000000 | 1.180e-3 | 59.31 | 0.9992 |
| state-like | 7.000e-2 | 7.000e-2 | 1.000000 | 7.000e-2 | 1.000 | 1.0000 |

**Three limits, stated.** First, s⋆ is the **complex** distance to instability,
while the actual perturbation A_λ − Â is real. Table 12's final column measures the
consequence: on the two normal cases the extremal direction is attainable by a real
perturbation and there is no gap, while on the non-normal cases the best real
perturbation at the same norm reaches only ρ = 0.9907 and 0.9992. Corollary 5.5 is
therefore sound but slightly conservative for real perturbations on non-normal
operators, and the real structured stability radius [15] is a further improvement
that is **UNMEASURED** here. Second, the lower bound on s⋆ is itself conservative
where the coarse grid's chord is large relative to s⋆ — visible in the
strong-non-normal row, where the certified bound 2.880e-3 sits 19% below the
3.564e-3 the finer extremal search located. The gate uses the lower bound, so this
costs tightness and never soundness. Third, Corollary 5.5 is implemented in the
validator only and is **not** in the distributed package.

**Incidental.** Corollary 5.5 requires no eigendecomposition and therefore has no
ill-conditioned-eigenbasis refusal class at all. It is evaluable on a defective Â,
where Bauer–Fike must refuse.

---

## 6. Rung 4: decisions

The preceding rungs certify intermediate objects. This one certifies a decision:
given a mean-variance quadratic program [16] solved from dithered storage, bound
the clean decision regret R := f(w*) − f(ŵ), the amount by which the chosen
portfolio is worse than the one clean data would have produced.

### 6.1 Construction

**Proposition 6.1 (exact in-set regret).** If the active set is preserved, regret
is exactly the quadratic form (γ/2)·Δwᵀ Σ Δw. Cited, not claimed [17, 18].

**Lemma 6.2 (certified solution radius).**

  ρ̂_w = (ε_μ + γ·ε_Σ·‖ŵ‖)/(γ·σ_lb),  (6.1)

with ε_μ = ‖κ‖₂ where κ_j = Δ_j·√(ln(2p/δ₁)/(2T)) is a Hoeffding vector, and
**ε_Σ = B(δ₂) from Theorem 4.6**. The chain from storage to decision runs through
the second-moment certificate, which is why these rungs are one program and not
four papers.

**Lemma 6.3 (active-set preservation from observed KKT margins).** The active set
is preserved when two observable margin conditions hold:

- **primal:** ρ̂_w is smaller than every inactive constraint's slack;
- **dual:** ĉ_j(‖g‖ + γ(‖Σ̂‖ + ε_Σ)ρ̂_w) < λ̂_j, with ĉ_j from pseudoinverse rows.

Both are computed from observed quantities. The certificate does not assume the
active set; it checks it. This lemma is the rung's claimed contribution.

**Theorem 6.4 (assembly).** With probability at least 1 − (δ₁ + δ₂) over the
injected dither, conditional on the panel, the screens of Lemma 6.3 in the proof's
order certify R ≤ R_in, with every constant computable from stored values.

**Corollary 6.5 (allocation).** R_in depends on the per-column steps, so it admits
a decision-sensitivity water-filling allocation at fixed budget. §8 reports its
measured performance against a pre-registered criterion.

### 6.2 What the referee broke, and how it was repaired

Two majors are published because both pass casual inspection.

**Undeclared constraint qualification.** Lemma 6.3 implicitly assumed linear
independence constraint qualification. The counter-example: three constraints
active at a vertex in ℝ², all with strictly positive multipliers. Every other
screen passes; the active set is rank-deficient, the multipliers are not unique,
and the dual margin therefore means nothing. *Repair:* an observable gate
σ_min([A_eq; A_active]) > tol, with a named `RANK_DEFICIENT_ACTIVE_SET` refusal.
The counter-example is now a test that must fire.

**A formulation that inflicted the failure on itself.** Encoding the budget
constraint Σw = 1 as two inequalities guarantees two active, linearly dependent
rows — so the formulation *created* the rank deficiency the first repair detects.
*Repair:* a native equality block with a free-sign multiplier excluded from every
positivity screen. A test covers a certified solve whose equality multiplier is
genuinely negative (ν = −0.12) without refusing; under the old encoding that case
was indistinguishable from a violated inequality.

A sign error in the Lemma 6.2 stationarity algebra was also corrected, and a
conjectured "escape mode" was **disabled** rather than shipped.

**Implementation note.** The solver is an exact dense primal active-set quadratic
program [19] with the native equality block in every KKT solve, Bland's rule for
anti-cycling, and a hard exactness gate on the stationarity residual and
multiplier signs. It exists because the incumbent sequential-quadratic-programming
path exposes no multipliers and post-normalizes the solution; a certificate whose
dual screen needs multipliers cannot be built on a solver that does not produce
them. The certificate's active set is **all tight inequalities**, not the solver's
independent working set, because a redundant tight row is exactly what should
reach the constraint-qualification gate.

### 6.3 The empirical result

**Zero certified cells at the pre-registered depths b ∈ {6, 8, 10, 12}**, on a
real 40-name × 1499-session panel at γ ∈ {2, 5, 10}. Every cell refuses
(`SINGULAR` at b = 6) or abstains (`ACTIVE_SET_AT_RISK` at b ≥ 8).

**Why, precisely.** A realistic equity covariance has condition number ≈ 170 and
λ_min ≈ 3.2e-5. Since ρ̂_w scales as 1/σ_lb by (6.1), a small λ_min makes the
certified solution radius large — and it must be smaller than every inactive
constraint's slack for Lemma 6.3 to fire. On this panel that does not occur until
roughly 20 bits.

**Table 12.** Extended probe beyond the pre-registered depths, clearly labeled as
such. The machinery is correct and inapplicable at realistic depths, and the probe
is what distinguishes those two statements.

| Measurement | Result | Criterion |
|---|---|---|
| Coverage: realized clean regret ≤ R_in | 3000 / 3000 | ≥ 0.90 |
| Active-set flips | 0 / 3000 | hard zero; Lemma 6.3 never falsified |
| Median R_in/&#124;f(ŵ)&#124; at b = 20 | ≈ 3e-6 | — |
| Median R_in/&#124;f(ŵ)&#124; at b = 24 | ≈ 1.5e-8 | — |

Every refusal is named and observable: `SINGULAR`, `ACTIVE_SET_AT_RISK`,
`RANK_DEFICIENT_ACTIVE_SET`, and solver-consistency failure. None returns a number.

---

## 7. Cross-cutting result I: the conditioning ceiling, revised

The ceiling was observed twice, from independent directions, and originally
reported as a single structural finding: *certifying an intermediate object is far
easier than certifying the decision that depends on it, because decisions turn on
small margins and conditioning amplifies uncertainty past them.*

**Table 13.** The two observations, and their present status.

| Instance | Conditioning quantity | Margin | Original verdict | Present status |
|---|---|---|---|---|
| Contraction gate (§5) | κ₂(V) ≈ 70.5 | 1 − ρ = 7.346e-3 | Ceiling; gate never fires | **Artifact.** Bauer–Fike overstates sensitivity 53×; Corollary 5.5 fires at b = 15 |
| Active-set preservation (§6) | cond(Σ) ≈ 170, λ_min ≈ 3.2e-5 | Inactive slack | Ceiling; zero certified cells | **Stands.** Not resolved by the same device |

The revision matters and the asymmetry is instructive.

**Why the first instance dissolved.** Bauer–Fike is a *lossy* step. It bounds
eigenvalue displacement by an eigenbasis condition number, and the pseudospectral
distance of Theorem 5.4 is the exact answer to the same question. Replacing a
lossy bound with the exact one recovered a factor of 53, and Proposition 5.6
establishes that nothing further is available from an operator-norm radius alone.
The ceiling was in the analysis.

**Why the second has not.** The conditioning in (6.1) is not obviously a lossy
step. A small λ_min genuinely makes the quadratic program's solution sensitive to
perturbation of Σ; the amplification is a property of the optimization problem
rather than of a chosen inequality. Whether an exact analogue exists — the true
distance from ŵ to the nearest solution with a different active set, playing the
role s⋆ plays for the spectrum — is **open and UNMEASURED**. Theorem 5.4's
existence makes it a much more pressing question than it appeared to be, because
the one prior data point suggesting the ceiling was intrinsic has been removed.

**The honest current statement.** The ladder terminates before the decision. It
does **not** terminate at the operator rung, and the claim that it did was an
artifact of the bound. Whether it terminates at the decision rung for structural
reasons, or for the same reason it appeared to terminate one rung earlier, is the
leading open problem of this program.

---

## 8. Cross-cutting result II: allocation futility

Both §4 and §6 admit a bit allocator that optimizes the certificate directly.
Both were constructed, both were evaluated against pre-registered criteria, and
neither was released.

**Table 14.** Certificate-optimal allocation against mean-squared-error-optimal
allocation at matched total budget. Positive is better.

| Program | Case | Improvement over MSE-optimal | Criterion | Decision |
|---|---|---|---|---|
| Second moments (§4) | EQUITY, budget 1200 | **+0.01%** | ≥ 5% | Not released |
| Second moments (§4) | MACRO, budget 192 | **+0.18%** | ≥ 5% | Not released |
| Decisions (§6) | γ = 2 | **−203%** | ≥ 0% | Not released |
| Decisions (§6) | γ = 5 | **−556%** | ≥ 0% | Not released |
| Decisions (§6) | γ = 10 | **−1294%** | ≥ 0% | Not released |

**The criterion discriminates.** In the second-moment case a third allocator using
squared top-principal-component loadings was 27.19% and 78.92% *worse*, so the
comparison is capable of separating allocators. It is reporting that the
sophisticated answer and the simple answer coincide.

**The mechanism.** Mean-squared-error-optimal allocation [20] already very nearly
minimizes the dominant term of both certificates. In the second-moment case the
certificate's weights differ from equal weights only through second-order
structure in Ā_j^abs, and on real panels that structure is weak. In the decision
case the explanation is sharper: MSE-optimal allocation minimizes ε_μ = ‖κ‖₂
*exactly*, and ε_μ is the dominant term of ρ̂_w in (6.1). An allocator optimizing
decision sensitivity is therefore spending bits on the wrong term, which is why it
is not merely no better but dramatically worse.

**Why this is reported rather than released.** A mode delivering a 0.01%
improvement is a control that does nothing, and a control that does nothing is
worse than no control: it is cited, tuned, and credited with outcomes. Two
independently motivated allocation objectives, constructed in separate research
cycles, reached the same conclusion. That is a more useful contribution than a
third attempt.

---

## 9. Limitations and open problems

1. **Every bound here is over storage quantization only**, conditional on a fixed
   data path. None bounds the distance from the clean sample to the population.
   §3.5 measures how much larger that term is; combining them requires the
   decomposition of Paper I and not a sum.
2. **Whether the decision-rung ceiling is intrinsic is open.** Per §7. This is the
   leading open problem.
3. **The real structured stability radius is UNMEASURED.** Per §5.5, Corollary 5.5
   uses the complex distance and is therefore conservative for the real
   perturbation that actually occurs.
4. **Corollary 5.5 is not in the distributed package.** It exists in the validator.
   No shipped code was changed by this paper.
5. **The bounds are conservative by roughly an order of magnitude.** Table 5 shows
   the mean bound at ≈ 10× the mean realized error, which is the characteristic
   behavior of a two-term matrix Bernstein inversion. Coverage of 1.0000 against a
   0.95 floor is a consequence, not an achievement.
6. **The closed-form allocator is out of regime on real panels most of the time.**
   Per Table 7. The geometric-program fallback is what actually runs.
7. **The trajectory results are two trajectories.** §5's conclusions rest on one
   macro state trajectory and one returns trajectory. The 53× and 47× figures of
   Table 11 are properties of those operators' non-normality and should not be
   read as general constants.
8. **Panel drift.** The STATE panel had grown from 1338 to 1340 snapshots between
   the original validation and the measurements of §5.4; ρ(A_λ) moved from 0.99264
   to 0.992654. Reported for exactness rather than because it changes anything.
9. **Assumption 2.3 has no runtime observable.** Dither seed reuse voids every
   concentration bound and is enforced at encode time only. A trajectory stored by
   a non-conforming encoder cannot be detected as such at decode.

---

## 10. Related work

**Dither.** Lemma 2.1 is Schuchman's condition [1], developed by Lipshitz,
Wannamaker and Vanderkooy [2] and Gray and Stockham [3]. Nothing here extends it.

**Matrix concentration.** Theorem 4.6 and the constants of Theorem 5.3 instantiate
the matrix Bernstein inequality [6, 7]; the 1-dependent cross term uses classical
even/odd blocking, for which matrix Freedman [21] would make any split unnecessary.
The inequalities are cited; the computability of their constants from stored values
is what is claimed.

**Empirical distribution bounds.** Theorem 3.3 composes a
Dvoretzky–Kiefer–Wolfowitz band [4] at Massart's tight constant [5].

**Dynamic mode decomposition.** The estimator (5.1) is the ridge variant of the
standard fit [8, 9]. Maity and Goswami [22] treat subtractive-dither DMD and its
asymptotic equivalence to quantization-induced regularization, and leave the
finite-sample statistics of the quantization error explicitly open; that is the
gap §5 closes. A deterministic worst-case treatment under *non*-dithered
quantization is the contrast point: injected dither is precisely what enables a
concentration-rate certificate, and no claim is made on that model.

**Pseudospectra and stability radii.** Theorem 5.4 states that the spectrum lies in
the pseudospectrum [11]; Proposition 5.6 restates the identity between the
minimum of σ_min over the unit circle and the distance to instability [12, 13, 14].
The real structured radius [15] is the improvement §9.3 names as unmeasured. Both
are classical; the contribution is recognizing that the certificate already
delivers exactly the premise these results consume, so the conditioning factor of
(5.3) was never necessary.

**Portfolio optimization under estimation error.** The setting is Markowitz [16];
solution sensitivity is multiparametric quadratic programming [17]; the effect of
estimation error on mean-variance portfolios is a large literature [18, 23]. The
active-set solver follows [19]. Lemma 6.3 — active-set preservation from *observed*
KKT margins under a certified solution radius — is the claimed contribution.

**Bit allocation.** The mean-squared-error-optimal allocation §8 fails to beat is
Huang and Schultheiss [20].

---

## 11. Conclusion

A known error law supports a ladder of certified downstream objects, and the ladder
is real: scalar functionals with a deterministic sandwich holding at certainty one,
a second-moment certificate whose conditioning is worth 22–32× on real panels, an
operator certificate covering in every admissible redraw, and a decision certificate
that is correct and never falsified. Three of the four refuse on real data at
realistic storage depths, and those refusals are the honest content.

The ladder terminates before the decision. It does not terminate at the operator
rung, though it was reported as doing so: the contraction gate's failure was an
artifact of a Bauer–Fike inclusion, and an elementary direct inclusion — provably
optimal given an operator-norm radius — fires five to six bits earlier. The
remaining instance of the conditioning ceiling, at the decision rung, has not been
dissolved by the same device, and whether it can be is now the program's leading
open question rather than a settled structural fact.

Allocation is futile in both places it was attempted, for an identifiable reason:
mean-squared-error-optimal allocation already minimizes the dominant term. Two
independently constructed allocators, neither released, is a more useful result
than a third.

---

## Appendix A — Reproduction

```bash
.venv312/Scripts/python.exe research/papers/run_srcvar.py        # §3, seed 20260720
.venv312/Scripts/python.exe research/papers/run_speccert.py      # §4, ~128 s
.venv312/Scripts/python.exe research/papers/run_ditherdmd.py     # §5.1–5.2
.venv312/Scripts/python.exe research/papers/run_pseudospectral.py # §5.4–5.5 (new)
.venv312/Scripts/python.exe research/papers/run_qpregret.py      # §6
```

**Environment.** Python 3.12; canonical interpreter `.venv312/Scripts/python.exe`.
NumPy only for the mathematics; pandas and SQLite are used to load committed data.
All validators are fully offline and open the history store read-only.

**Seeds and parameters.** §3: seed 20260720, b ∈ {4,6,8,10,12}. §4: δ = 0.05, 500
redraws, both panels. §5.1–5.2: δ = 0.05, 500 redraws, b ∈ {6,8,10}, plus a b = 2
refusal exercise over 50 seeds. §5.4–5.5: δ = 0.05, 100 redraws,
b ∈ {6,8,10,12,14,16,18}, s⋆ by a 4096-point coarse grid with 256× refinement of
the lowest 2% of arcs. §6: γ ∈ {2,5,10}, pre-registered b ∈ {6,8,10,12} plus a
labeled extended probe to b = 24.

**Falsifier tests that must fire.** The constraint-qualification counter-example of
§6.2 must produce `RANK_DEFICIENT_ACTIVE_SET`. The smoothness falsifier of
Proposition 3.5 must produce `SMOOTHNESS_FALSIFIED` on the tick-grid series. At
b = 2 both trajectories of §5 must produce `INSUFFICIENT_REGULARIZATION`.

**Data availability.** The panels are drawn from the program's committed history
store and are not redistributable; construction is deterministic given the store,
and the loaders ship with the validators. The synthetic constructions of §3.2,
§3.3 and §3.4 are fully specified by the stated parameters and reproduce without
the store. §5.4's optimality and soundness searches (Proposition 5.6) are
synthetic and self-contained.

---

## Appendix B — Notation table

| Symbol | Definition | First use |
|---|---|---|
| Δ, Δ_j | Quantization step; per-column where subscripted | §2.1 |
| s_j | Δ_j² | Def. 2.5 |
| e | Storage error y − x | Lemma 2.1 |
| x̄ | Observable surrogate \|y\| + Δ/2 | Def. 2.4 |
| δ | Miscoverage budget | §2.2 |
| T, p | Panel rows, columns | §4 |
| ν̄_obs | Observable matrix-variance proxy | Cor. 4.3 |
| R_obs | Observable range term | Lemma 4.5 |
| ℓ | Log factor ln(2p/δ) or ln(2k/δ_i) | (4.1), §5.1 |
| Ā_j^abs | Per-column surrogate matrix | Lemma 4.4 |
| a_j | Allocation weight λ_max(Ā_j^abs) | Thm. 4.7 |
| B(δ,s) | Second-moment certificate | Thm. 4.6 |
| k, W | State dimension, snapshot count | §5 |
| A_λ, Â | Clean ridge-DMD fit, its debiased estimate | §5 |
| Ĝ, Ĉ | Debiased Gram and cross matrices | §5 |
| m̂ | Observed spectral floor λ_min(Ĝ) + λ | Lemma 5.1 |
| a | ‖Â‖ | Thm. 5.2 |
| c, g | Bounds on ‖E_C‖, ‖E_G‖ | Thm. 5.2 |
| r̂ | Certified operator-norm radius | (5.2) |
| κ₂(V) | Eigenbasis condition number | (5.3) |
| s⋆ | min over the unit circle of σ_min(zI − Â) | (5.4) |
| ρ(·) | Spectral radius | §5.2 |
| σ_min(·) | Smallest singular value | Thm. 5.4 |
| γ | Risk-aversion parameter | §6 |
| ρ̂_w | Certified solution radius | (6.1) |
| ε_μ, ε_Σ | Mean and covariance error terms | Lemma 6.2 |
| σ_lb | Lower bound on λ_min(Σ) | (6.1) |
| ĉ_j | Pseudoinverse row norms in the dual screen | Lemma 6.3 |
| R, R_in | Decision regret, its certified bound | §6 |

### Numbering note

The four constructions were originally documented separately, each numbering its
results from zero, which produced three distinct objects named "Lemma 2" and two
named "Theorem 3". Numbering here is section-scoped and the original ordinal is
preserved as the second component wherever possible, so that Theorem 4.6 is the
former Theorem 6, Theorem 4.7(c) the former Theorem 7c, and Lemmas 6.2 and 6.3 the
former Lemmas 2 and 3 of the decision construction. Theorems 5.4, Corollary 5.5
and Proposition 5.6 are new in this paper and have no antecedent.

---

## References

The identifiers below were recorded from the authors' working bibliography and
should be checked against the primary sources before external publication; this
series has not yet had a bibliographic review.

[1] L. Schuchman. *Dither Signals and Their Effect on Quantization Noise.* IEEE
Transactions on Communication Technology 12(4), 1964.

[2] S. P. Lipshitz, R. A. Wannamaker, J. Vanderkooy. *Quantization and Dither: A
Theoretical Survey.* Journal of the Audio Engineering Society 40(5), 1992.

[3] R. M. Gray, T. G. Stockham. *Dithered Quantizers.* IEEE Transactions on
Information Theory 39(3), 1993.

[4] A. Dvoretzky, J. Kiefer, J. Wolfowitz. *Asymptotic Minimax Character of the
Sample Distribution Function.* Annals of Mathematical Statistics 27(3), 1956.

[5] P. Massart. *The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality.*
Annals of Probability 18(3), 1990.

[6] J. A. Tropp. *User-Friendly Tail Bounds for Sums of Random Matrices.*
Foundations of Computational Mathematics 12(4), 2012.

[7] J. A. Tropp. *An Introduction to Matrix Concentration Inequalities.*
Foundations and Trends in Machine Learning 8(1–2), 2015.

[8] P. J. Schmid. *Dynamic Mode Decomposition of Numerical and Experimental Data.*
Journal of Fluid Mechanics 656, 2010.

[9] J. H. Tu, C. W. Rowley, D. M. Luchtenburg, S. L. Brunton, J. N. Kutz. *On
Dynamic Mode Decomposition: Theory and Applications.* Journal of Computational
Dynamics 1(2), 2014.

[10] F. L. Bauer, C. T. Fike. *Norms and Exclusion Theorems.* Numerische Mathematik
2, 1960.

[11] L. N. Trefethen, M. Embree. *Spectra and Pseudospectra: The Behavior of
Nonnormal Matrices and Operators.* Princeton University Press, 2005.

[12] C. F. Van Loan. *How Near Is a Stable Matrix to an Unstable Matrix?*
Contemporary Mathematics 47, 1985.

[13] R. Byers. *A Bisection Method for Measuring the Distance of a Stable Matrix to
the Unstable Matrices.* SIAM Journal on Scientific and Statistical Computing 9(5),
1988.

[14] D. Hinrichsen, A. J. Pritchard. *Stability Radii of Linear Systems.* Systems
and Control Letters 7(1), 1986.

[15] L. Qiu, B. Bernhardsson, A. Rantzer, E. J. Davison, P. M. Young, J. C. Doyle.
*A Formula for Computation of the Real Stability Radius.* Automatica 31(6), 1995.

[16] H. Markowitz. *Portfolio Selection.* Journal of Finance 7(1), 1952.

[17] A. Bemporad, M. Morari, V. Dua, E. N. Pistikopoulos. *The Explicit Linear
Quadratic Regulator for Constrained Systems.* Automatica 38(1), 2002.

[18] R. Kan, G. Zhou. *Optimal Portfolio Choice with Parameter Uncertainty.*
Journal of Financial and Quantitative Analysis 42(3), 2007.

[19] J. Nocedal, S. J. Wright. *Numerical Optimization.* 2nd ed., Springer, 2006,
§16.3.

[20] J.-Y. Huang, P. M. Schultheiss. *Block Quantization of Correlated Gaussian
Random Variables.* IEEE Transactions on Communication Systems 11(3), 1963.

[21] J. A. Tropp. *Freedman's Inequality for Matrix Martingales.* Electronic
Communications in Probability 16, 2011.

[22] A. Maity, D. Goswami. *Dynamic Mode Decomposition with Quantized Data.* 2024.
arXiv:2404.02014; see also arXiv:2410.02803 and arXiv:2501.07714.

[23] R. O. Michaud. *The Markowitz Optimization Enigma: Is 'Optimized' Optimal?*
Financial Analysts Journal 45(1), 1989.
