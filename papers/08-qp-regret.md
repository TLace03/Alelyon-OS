# QP-Regret

### A decision-regret certificate for the mean-variance QP, and zero certified results on real data

**Status:** proof document, **refereed** (one Opus referee, verdict REPAIRABLE →
repaired). Empirical validation on a real 40-name × 1499-session panel.
Validator: `research/papers/run_qpregret.py`. Suite: 2228 passed / 1 skipped /
0 failed.

---

## Abstract

The previous papers certify intermediate objects: a covariance, an operator, a
quantile. This one certifies a **decision**. Given a mean-variance quadratic
program solved from dithered storage, we bound the clean decision regret
`R = f(w*) − f(ŵ)` — how much worse the portfolio you chose is than the one you
would have chosen with clean data.

The bound is computable, refusal-first, and holds at joint confidence
1 − (δ₁ + δ₂).

**On a real panel at the pre-registered bit depths b ∈ {6, 8, 10, 12}, it
certifies nothing. Zero cells. Every one refuses or abstains.** That is the
headline, it is intrinsic rather than a bug, and §4 explains exactly which
property of real equity covariance causes it.

## 1. The construction

**In-set regret is exact.** If the active set is preserved, regret is the
quadratic identity `(γ/2)·Δw′ΣΔw`. This is cited, not claimed
(Bemporad mpQP; Kan–Zhou).

**Lemma 2 — certified solution radius.**

```
ρ̂_w = (ε_μ + γ·ε_Σ·‖ŵ‖) / (γ·σ_lb)
```

with `ε_μ = ‖κ‖₂`, `κ_j = Δ_j·sqrt(ln(2p/δ₁)/(2T))` a Hoeffding vector, and
`ε_Σ = B(δ₂)` from SpecCert (paper 6). The chain from storage to decision runs
through the earlier certificate, which is why these papers are one program.

**Lemma 3 — active-set preservation.** The claimed lemma, verified line by line
in referee review. The active set is preserved when two observable KKT margin
conditions hold:

- **primal:** `ρ̂_w` is smaller than every inactive constraint's slack;
- **dual:** `ĉ_j(‖g‖ + γ(‖Σ̂‖ + ε_Σ)ρ̂_w) < λ̂_j`, with `ĉ_j` from pseudoinverse rows.

Both are computed from **observed** quantities. That is the point: the certificate
does not assume the active set, it checks it.

## 2. What the referee broke, and how it was repaired

The referee verdict was REPAIRABLE, and the two majors are worth publishing
because both are traps that pass casual inspection.

**MAJOR-1 — undeclared LICQ.** The lemma implicitly assumed linear independence
constraint qualification. The killer example: three constraints active at a
vertex in ℝ², all with strictly positive multipliers. Every other screen in the
certificate passes. The active set is rank-deficient and the multipliers are not
unique, so the dual margin means nothing.

*Repair:* an observable LICQ gate — `σ_min([A_eq; A_active]) > tol` — with a
named `RANK_DEFICIENT_ACTIVE_SET` refusal and a solver-consistency guard. The
referee's counter-example is now a test that must fire.

**MAJOR-2 — the budget constraint was self-inflicting LICQ failure.** Encoding
`Σw = 1` as two inequalities (`≤ 1` and `≥ 1`) guarantees two active rows that
are linearly dependent, so the formulation *created* the rank deficiency MAJOR-1
detects.

*Repair:* a **native equality block** with free-sign multiplier `ν` excluded from
every positivity screen. A test covers a certified solve whose equality
multiplier is genuinely negative (`ν = −0.12`) without refusing — which under the
old encoding was indistinguishable from a violated inequality.

Also repaired: a sign error in the Lemma-2 stationarity algebra. Also **disabled**:
an "escape mode" (`E[R] + δ_AS·R_max`) that was labelled CONJECTURED and is not
shipped.

## 3. Implementation notes

`solver.py` is an exact dense primal active-set QP (Nocedal–Wright §16.3) with the
native equality block in every KKT solve, Bland's rule for anti-cycling, and a
hard exactness gate on the stationarity residual and multiplier signs.

It exists because the incumbent SLSQP-based path exposes **no multipliers** and
post-normalises the solution. A certificate whose dual screen needs multipliers
cannot be built on a solver that does not produce them.

`certificate.py` implements Theorem 4 with the screens in the proof's order, and
one deliberate design choice: **the certificate's active set is all tight
inequalities** (slack ≤ tol), *not* the solver's independent working set. A
redundant tight row is exactly the thing that should reach the LICQ gate, and a
working set that has already discarded it would hide the condition the gate
exists to catch.

## 4. The empirical result

**Honest headline: 0 CERTIFIED at b ∈ {6, 8, 10, 12}.**

Every cell refuses (`SINGULAR` at b = 6) or abstains (`ACTIVE_SET_AT_RISK` at
b ≥ 8), on a real 40-name × 1499-session panel at γ ∈ {2, 5, 10}.

**Why, precisely.** A realistic equity covariance has condition number ≈ 170 and
`λ_min ≈ 3.2e-5`. Since `ρ̂_w` scales as `1/σ_lb`, a tiny `λ_min` makes the
certified solution radius large — and it must be *smaller than every inactive
constraint's slack* for Lemma 3 to fire. On this panel that does not happen until
roughly **20 bits**.

This is the same shape as Dither-DMD's contraction abstention (paper 7): the
machinery is correct, coverage is perfect, and a conditioning quantity puts the
required margin out of reach. Twice now, from independent directions, the finding
is that **certifying an intermediate object is much easier than certifying the
decision that depends on it**, because decisions turn on small margins and
conditioning amplifies uncertainty past them.

## 5. The extended probe — the machinery does work

On a clearly-labelled extended probe (b up to 24, beyond the pre-registered
depths):

- **coverage 3000 / 3000** — realised clean regret ≤ `R_in` on every certified
  redraw, against a 0.90 floor
- **active-set-flip gate: 0 flips / 3000** — a hard-zero gate. Lemma 3 was never
  falsified.
- widths tighten monotonically and are tiny where emitted: median
  `R_in/|f(ŵ)| ≈ 3e-6` at b = 20, `≈ 1.5e-8` at b = 24

So the certificate is not wrong. It is correct and inapplicable at realistic
storage depths, and the probe is what distinguishes those two.

## 6. Allocation failed again

Decision-sensitivity water-filling versus MSE-optimal, at matched budget:

| γ | cert vs MSE |
|---|---|
| 2 | **−203%** |
| 5 | **−556%** |
| 10 | **−1294%** |

Strictly, dramatically **worse**. Allocation not shipped as a mode.

This mirrors SpecCert's Gate 3 (paper 6, +0.01% / +0.18%, also not shipped) and
sharpens the explanation: **MSE-optimal already minimises `ε_μ = ‖κ‖₂` exactly**,
and `ε_μ` is the dominant term in `ρ̂_w`. An allocator that optimises decision
sensitivity is spending bits on the wrong term.

Two independent research cycles produced allocation modes that did not clear
their gates. Neither shipped. We regard "we built the sophisticated allocator
twice and it lost to equal weights both times" as a more useful contribution than
a third attempt.

## 7. Honest degradation

Every refusal is named and observable:

- `SINGULAR` — the certified covariance error swamps the spectral floor
- `ACTIVE_SET_AT_RISK` — a KKT margin condition failed
- `RANK_DEFICIENT_ACTIVE_SET` — the LICQ gate fired
- solver-consistency failure — the exactness gate on stationarity residual and
  multiplier signs

None returns a number.

## 8. Prior art

Multiparametric QP and solution sensitivity are Bemporad et al. Estimation-error
effects on mean-variance portfolios are Kan–Zhou, Michaud, and a large literature.
Active-set QP is Nocedal–Wright. LICQ is standard nonlinear programming.

The claimed contribution is **Lemma 3** — active-set preservation from *observed*
KKT margins under a certified solution radius — together with the refusal
discipline the referee's counter-examples forced.

## 9. Reproduction

```bash
python research/papers/run_qpregret.py
```

Real 40-name × 1499-session panel, γ ∈ {2, 5, 10}, pre-registered depths plus a
labelled extended probe. The referee's ℝ² vertex counter-example is in the test
suite and must fire `RANK_DEFICIENT_ACTIVE_SET`.
