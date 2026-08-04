# SpecCert

### A conditional spectral certificate, and a bit allocation that did not earn its keep

**Status:** proof document plus empirical validation on two real panels.
Validator: `research/papers/run_speccert.py`, 128 s, δ = 0.05, 500 dither
redraws, read-only data.

---

## Abstract

Given a data panel stored with per-column subtractive dither at heterogeneous bit
depths, how far can the sample covariance computed from *stored* values be from
the covariance of the clean values, in operator norm?

We give a conditional certificate `B(δ, s)` — a two-term Bernstein inversion
whose per-row surrogate is computed from the stored data itself, rather than from
a data-independent global envelope. On two real panels it is **31.9× and 22.7×
tighter** than the unconditional bound, clearing a pre-registered 10× kill gate.

We then use the certificate as an objective for bit allocation: given a fixed
total bit budget, which columns deserve more depth? The theory gives a
water-filling solution. **On real panels it beat MSE-optimal allocation by 0.01%
and 0.18%, against a pre-registered 5% ship threshold. The mode was not
shipped.** That negative is the more useful half of this paper and §5 explains
why it happens.

## 1. Setting

Two real panels, both read-only from committed history:

| panel | rows T | cols p | transform | column-σ range |
|---|---|---|---|---|
| EQUITY | 1499 | 200 | daily log-returns | 0.0093 – 0.0365 |
| MACRO | 1199 | 32 | daily first-differences | 0.0040 – 1.8475 |

Per-column steps are **heterogeneous**: `κ_j = (max_t|x_{t,j}|)²`, bit depths
`b_j` vary around `b_avg = 6` with no two columns sharing a depth, and
`Δ_j = sqrt(κ_j · 4^{−b_j})`. Unbounded codec, so no clipping.

The MACRO panel is deliberately nastier: its column standard deviations span
three orders of magnitude, which is where a global envelope does worst.

## 2. Gate 1 — Coverage

Theorem 6's probability space: data **fixed**, dither redrawn. For each of 500
redraws we encode the fixed panel, recompute `B(δ, s)` from stored `(Y, Δ)`
alone, and test whether the realised `‖Σ̂ − Σ_clean‖_op ≤ B`.

| panel | covered / K | coverage | floor | mean B | mean op-norm | max op-norm |
|---|---|---|---|---|---|---|
| EQUITY | 500/500 | 1.0000 | 0.95 | 6.374e-04 | 6.328e-05 | 1.024e-04 |
| MACRO | 500/500 | 1.0000 | 0.95 | 0.1167 | 0.01722 | 0.03598 |

**PASS.** Coverage at 1.000 against a 0.95 floor is the expected behaviour of a
conservative two-term Bernstein bound, not a red flag — but note the mean bound
is roughly 10× the mean realised error, which is the price of that conservatism.

## 3. Gate 2 — The pre-registered kill gate

The question this gate asks is whether *conditioning* is doing any work. Ratio =
`B_unconditional / B_conditional`, same Bernstein machinery, same δ; the only
difference is the per-row surrogate `x̄_t = |y_t| + Δ/2` versus a data-independent
global envelope `K·1`.

| panel | B_conditional | B_unconditional | ratio | target |
|---|---|---|---|---|
| EQUITY | 6.377e-04 | 0.02035 | **31.9×** | ≥10× |
| MACRO | 0.11659 | 2.64498 | **22.7×** | ≥10× |

**PASS.** Conditioning is load-bearing. The candidate clears its own kill gate on
both real panels, not just the one required.

This is the paper's positive result. A certificate computed from what you stored
is worth roughly an order of magnitude and a half against one computed from what
you might have stored.

## 4. Gate 3 — Allocation, and its failure

At a matched total budget (`p · b_avg`, `b_avg = 6`, `b_max = 12`, `b_min = 1`),
certificate width `B` under three allocators, all scored by the same machinery:

- **certificate-optimal** — water-fill on weights `a_j = λ_max(Ā_j^abs)`
- **MSE-optimal** — equal weights (Huang–Schultheiss)
- **R3 functional** — weights from squared top-PC loadings

| panel | budget | B_cert | B_mse | B_func | cert vs MSE | cert vs R3 |
|---|---|---|---|---|---|---|
| EQUITY | 1200 | 3.871e-04 | 3.871e-04 | 5.316e-04 | **+0.01%** | +27.19% |
| MACRO | 192 | 9.894e-03 | 9.913e-03 | 4.694e-02 | **+0.18%** | +78.92% |

**Verdict: allocation mode NOT shipped.**

The separation theorem holds *structurally* — the allocations genuinely differ,
certificate-optimal is the minimum on both panels, and the sign of
`B_mse − B_cert` is non-negative as predicted. The direction is right. The
magnitude is nowhere near the 5% secondary gate.

**Why.** MSE-optimal allocation already very nearly minimises the dominant term
of the certificate. The certificate's weights differ from equal weights only
through second-order structure in `Ā_j^abs`, and on real panels that structure is
weak. The R3 functional allocator, by contrast, is 27% and 79% *worse* — so the
gate is discriminating; it is simply telling us that the sophisticated answer and
the simple answer coincide here.

We report this rather than quietly shipping a mode with a 0.01% improvement,
because a knob that does nothing is worse than no knob: people cite it, tune it,
and attribute outcomes to it.

The same finding recurred independently in the QP-Regret work (paper 8), where
decision-sensitivity water-filling was **strictly worse** than MSE-optimal by
−203% to −1294%. Two independent allocation objectives, same conclusion:
MSE-optimal is hard to beat because it already minimises the term that dominates.

## 5. Gate 4 — The regime check is a real gate

Theorem 7c gives two observable checks that must both pass for the closed-form
water-fill to be valid; otherwise a geometric-program fallback ships.

| panel | closed-form admissible | detail |
|---|---|---|
| EQUITY | **1 of 8** budgets (12%) | variance check passes everywhere; **range check fails at every b_avg ≥ 3** |
| MACRO | **0 of 8** (0%) | range check fails at every depth tested |

**Reading.** On real panels the range term routinely fails at moderate and coarse
bit depths, so the allocator correctly refuses the closed form and ships the
geometric-program minimiser instead. This directly implements a referee finding:
*never ship the closed form out of regime.*

A regime check that passed everywhere would be a rubber stamp. This one fails
7/8 and 8/8 of the time on real data, which is the evidence that it is testing
something.

## 6. Honest degradation

- Clipping (`k_clip > 0`) invalidates the surrogate; refuse.
- A column whose Δ is unknown or whose capture law is unrecognised makes the
  panel unusable, not the column optional.
- Out-of-regime allocation falls back rather than approximating.
- The certificate is conditional on the fixed data path. It is a statement about
  the dither, not about sampling.

That last point is the boundary between this paper and the sampling term in the
error budget. `B` says nothing about how far `Σ_clean` is from the population
covariance, and combining the two requires the decomposition described in
paper 1 rather than a sum.

## 7. Prior art

Matrix Bernstein inequalities are Tropp. Bit allocation under an MSE objective is
Huang–Schultheiss. Conditional (data-dependent) concentration is standard in the
empirical-process literature.

The contribution is the specific assembly — a certificate computable from stored
values alone, with an observable regime test governing which allocator ships —
plus two honestly reported negatives: the allocation gain does not clear its gate,
and the closed form is out of regime on real panels most of the time.

## 8. Reproduction

```bash
python research/papers/run_speccert.py
```

Reads committed data read-only. δ = 0.05, 500 redraws, both panels. Runs in
about two minutes.
