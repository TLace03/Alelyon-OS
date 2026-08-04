# Dither-DMD

### A computable spectral certificate for dynamic mode decomposition — and where it is useless

**Status:** proof document plus empirical validation on a real product state
trajectory and a returns trajectory. Gate verdict: PROCEED_REPOSITIONED — the
2-colouring device is **cited as classical blocking**, not claimed. Validator:
`research/papers/run_ditherdmd.py`, δ = 0.05, 500 dither redraws per cell.

---

## Abstract

Dynamic mode decomposition fits a linear operator `A` to a trajectory. If the
trajectory is read from dithered storage, how far can the fitted `A` be from the
one you would have fitted to clean data?

We give an operator-norm certificate `‖Â − A_λ‖₂ ≤ r̂` for debiased ridge-DMD,
holding with probability ≥ 1 − δ over the injected dither on a fixed path,
computable from stored values alone.

**Coverage is 1.000 everywhere the certificate is admissible.** And the
certificate is, for the question people actually want to ask, **useless**: the
downstream contraction gate — does the fitted operator have spectral radius below
1? — **never fires at any tested bit depth**, on a trajectory whose clean
operator is provably contracting. §5 is the paper.

## 1. Setting

**(a) A real product state trajectory.** 14 macro features, weekly, 2000-12-01 →
2026-07-17; k = 14, m = 1338 snapshots. Columns: NFCI, excess bond premium, VIX,
DGS10, curve, industrial production, retail, capacity utilisation, claims, CPI,
oil, copper/gold, dollar, SPX.

- clean full-precision fit: `‖A_λ‖₂ = 1.0493`, spectral radius `ρ = 0.99264`
- **contraction-relevant gap `1 − ρ = 0.00736`**

**(b) 20-coordinate daily log-returns.** k = 20, m = 750 snapshots, 2023-07-20 →
2026-07-17.

- clean fit: `‖A_λ‖₂ = 0.4789`, `ρ = 0.19927`
- gap `1 − ρ = 0.80073`

Ridge parameter λ chosen as the smallest of {0.1, 0.3, 1.0, 3.0} × trace(G)/W
making the clean b = 8 certificate admissible.

## 2. Gate 1 — Coverage

| data | b | ok / refuse | coverage | median error | median r̂ |
|---|---|---|---|---|---|
| state | 6 | 500 / 0 | **1.0000** | 1.378e-02 | 5.094e+00 |
| state | 8 | 500 / 0 | **1.0000** | 3.382e-03 | 5.074e-01 |
| state | 10 | 500 / 0 | **1.0000** | 8.652e-04 | 1.101e-01 |
| returns | 6 | 0 / 500 | *all refused* | — | — |
| returns | 8 | 500 / 0 | **1.0000** | 4.460e-02 | 3.798e+01 |
| returns | 10 | 500 / 0 | **1.0000** | 1.088e-02 | 6.153e-01 |

**PASS** wherever admissible. The 500 refusals at returns/b=6 all carry the named
reason `INSUFFICIENT_REGULARIZATION` — the certified Gram error swamps the
observed spectral floor, so no bound exists and none is emitted.

## 3. Gate 2 — Width honesty

No threshold. An honest table.

| data | b | median r̂ | ‖A_λ‖ | r̂/‖A_λ‖ | 1−ρ gap | verdict |
|---|---|---|---|---|---|---|
| state | 6 | 5.094e+00 | 1.0493 | 4.854 | 0.00736 | **vacuous** (r̂ ≥ ‖A‖) |
| state | 8 | 5.074e-01 | 1.0493 | 0.484 | 0.00736 | partial (r̂ < ‖A‖ but ≫ gap) |
| state | 10 | 1.101e-01 | 1.0493 | 0.105 | 0.00736 | informative about ‖A‖ |
| returns | 8 | 3.798e+01 | 0.4789 | 79.3 | 0.80073 | **vacuous** |
| returns | 10 | 6.153e-01 | 0.4789 | 1.285 | 0.80073 | **vacuous** |

**Reading.** The certificate becomes informative about the *operator norm* around
b ≈ 10. At **every** tested depth it remains orders of magnitude larger than the
`1 − ρ` gap. Bit depth buys operator-norm accuracy. It does not, here, buy
contraction resolution.

## 4. Gate 3 — The contraction gate never fires

The gate certifies contraction iff `ρ_bound = max|λᵢ(Â)| + κ₂(V)·r̂ < 1`
(Bauer–Fike).

| b | median max&#124;λ(Â)&#124; | median κ₂(V) | median r̂ | median ρ_bound | certifies |
|---|---|---|---|---|---|
| 6 | 0.9926 | 71.85 | 5.094e+00 | 3.676e+02 | **0 / 500** |
| 8 | 0.9926 | 72.20 | 5.074e-01 | 3.755e+01 | **0 / 500** |
| 10 | 0.9926 | 70.60 | 1.101e-01 | 8.769e+00 | **0 / 500** |

The clean operator **is** contracting: ρ = 0.99264 < 1. The certificate cannot
establish it at any tested depth, and the reason is a product of two factors:

1. `ρ̂ ≈ 0.993` sits a whisker below the unit circle, leaving 0.00736 of margin.
2. `κ₂(V) ≈ 55–72` — the eigenvector conditioning — **amplifies** r̂ by nearly two
   orders of magnitude before it is compared to that margin.

To certify contraction you would need `r̂ < 0.00736/72 ≈ 1e-4`, roughly a
thousand times tighter than the b = 10 value.

**Zero consistency violations across all redraws.** A false "contracting" would
require `ρ_bound < 1` while clean `ρ ≥ 1`, and was never observed — the gate is
conservative in the safe direction.

## 5. What this result is actually for

A certificate that refuses to conclude what the clean data confirms looks like a
failure. We think it is the paper's contribution, for three reasons.

**It locates the binding constraint precisely.** The obstruction is not
quantisation. It is eigenvector conditioning, `κ₂(V) ≈ 72`, multiplying a bound
that is already conservative. Anyone tempted to solve this by storing more bits
can read off the table that ten bits already gives an informative operator-norm
bound and still misses contraction by three orders of magnitude. **More bits is
not the fix.** A better-conditioned basis, or a certificate that bounds the
spectrum directly rather than routing through Bauer–Fike, might be.

**It is the honest version of a claim that would otherwise be easy to make.**
"Certified contraction from compressed storage" is a saleable sentence. On real
data it is not available, and the version of this system that shipped it would
have been shipping a gate that fires only when the data is easy.

**It generalises.** The same shape appears in QP-Regret (paper 8): a certificate
whose machinery is correct, whose coverage is perfect, and which abstains on
realistic data because a conditioning quantity — there `cond(Σ) ≈ 170` and
`λ_min ≈ 3.2e-5` — makes the margin unreachable. Two independent certificates,
same failure mode: **the intermediate object is well-certified and the decision
is not**, because the decision depends on a small margin that conditioning
amplifies the uncertainty past.

## 6. Gate 4 — Refusal exercise

At 2-bit storage the certificate must refuse rather than emit a number.

| data | b | ok / refuse (50 seeds) | reason |
|---|---|---|---|
| state | 2 | 0 / 50 | `INSUFFICIENT_REGULARIZATION` ×50 |
| returns | 2 | 0 / 50 | `INSUFFICIENT_REGULARIZATION` ×50 |

**PASS.**

## 7. Prior art and positioning

The gate verdict on this cycle was **PROCEED_REPOSITIONED**: a July-2026 result
materialised during the work and the assassination review caught it before proof
effort was wasted. The 2-colouring argument is **cited as classical blocking**,
not claimed as novel.

Incumbents — Maity–Goswami (arXiv:2404.02014, 2410.02803, 2501.07714) — leave the
finite-sample statistics of the quantisation error explicitly open. Our claim is
narrowed accordingly to the **observable-surrogate bootstrap and the assembly**:
computing the concentration constants from stored data rather than from a
data-independent envelope, and composing them into a spectral corollary with
named refusals.

Matrix Bernstein is Tropp. Bauer–Fike is classical. Debiased ridge estimation is
standard. We claim none of them.

## 8. Reproduction

```bash
python research/papers/run_ditherdmd.py
```

All data offline and read-only. Trajectory (a) is the exact state matrix the
product's State Space view uses, entitled through an offline data path.

## 9. What we would want reviewed first

1. Whether a certificate that bounds the spectrum directly — avoiding the
   `κ₂(V)` amplification — is constructible from the same premises. This is the
   difference between a useless gate and a useful one.
2. Whether `INSUFFICIENT_REGULARIZATION` is the right refusal, or whether a
   larger λ should be selected automatically with the bias reported.
3. Whether the conservatism of the two-term Bernstein bound (mean r̂ roughly an
   order of magnitude above realised error) is reducible without losing coverage.
