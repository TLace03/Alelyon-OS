# Research log — the five open questions

**Date:** 2026-08-04. **Status:** complete; three findings change a paper's thesis.

These are the items the working papers listed as unresolved. Each was closed by
measurement, and each result — including the two that went against the
implementation — is recorded here and carried into the papers rather than
dropped. Scripts named below are runnable offline against committed data.

Evidence conventions: a figure quoted here was produced by the named command on
the date above. `UNMEASURED` means what it says.

---

## R1 — Does a certificate bounding the spectrum directly avoid the κ₂(V) amplification?

*Open question from `07-dither-dmd.md` §9.1.* **Answer: yes, and it is optimal.**

The shipped contraction gate routes the operator-norm certificate through
Bauer–Fike, paying an eigenbasis conditioning factor κ₂(V). On the real STATE
trajectory κ₂(V) ≈ 70.5 against a 1 − ρ margin of 0.00736, and the gate never
fires at any tested bit depth. The question was whether that ceiling is intrinsic
or an artifact of the bound.

**Theorem D1 (direct inclusion).** If ‖Â − A‖₂ ≤ r then every eigenvalue μ of A
satisfies σ_min(μI − Â) ≤ r.

*Proof.* μI − A is singular, so (μI − A)v = 0 for some unit v. Then
(μI − Â)v = (A − Â)v, so σ_min(μI − Â) ≤ ‖(A − Â)v‖ ≤ r. ∎

No diagonalizability, no eigenbasis, no conditioning factor.

**Corollary D2 (contraction gate without κ₂(V)).** Let
s⋆ := min_{|z|=1} σ_min(zI − Â). If ρ(Â) < 1 and r < s⋆ then ρ(A) < 1.

*Proof.* Put A_t := Â + t(A − Â), t ∈ [0,1]. Then ‖A_t − Â‖ ≤ tr ≤ r < s⋆, so by
D1 no eigenvalue of any A_t lies on the unit circle. Eigenvalues are continuous
in t and at t = 0 all lie strictly inside. A continuous eigenvalue path leaving
the open disk must meet the circle. Hence ρ(A) < 1. ∎

Both consume exactly the premise the existing certificate already delivers, so D2
is a drop-in replacement, not a new probabilistic claim. s⋆ is bounded below
rigorously: z ↦ σ_min(zI − Â) is 1-Lipschitz, so over grid arcs of chord c the
true minimum is at least (grid minimum) − c/2.

### Measured (`research/papers/run_pseudospectral.py`, 100 redraws/cell, δ = 0.05)

| trajectory | gate | first bit depth that fires | threshold on r̂ |
|---|---|---|---|
| STATE (k=14, W=1339) | Bauer–Fike | **never** (not at b=18; needs b ≈ 20–21) | 1.04e-4 |
| STATE | direct (D2) | **b = 15** (Δ = 2.44e-4) | 5.574e-3 |
| returns (k=20, W=749) | Bauer–Fike | b = 16 | 1.52e-2 |
| returns | direct (D2) | **b = 10** | 7.217e-1 |

A 5–6 bit reduction, i.e. 32–64× coarser storage for the same conclusion. The
true non-normality penalty on the STATE operator is s⋆/(1−ρ) = 0.759, a factor
1.32 — not the 70.5 Bauer–Fike charges. Bauer–Fike overstates the penalty 53×.

### Optimality and soundness

D2 is unimprovable given only an operator-norm radius. At the minimizing
z⋆ = e^{iθ⋆} with singular vectors u, v, the perturbation E := s⋆uv^H has
‖E‖₂ = s⋆ and makes z⋆ an eigenvalue of Â + E. Constructed explicitly in four
cases; ρ(Â+E) = 1.000000 in every one. So no bound depending only on
‖Â − A‖ ≤ r can accept r ≥ s⋆.

Soundness search: 300 fired gates × 61 adversarial perturbations each (random
complex directions plus the extremal direction). Worst ρ(A+E) observed
0.999563 < 1. Never broken. Theorem D1 falsifier: 0 violations across 1300
certificate evaluations on real data.

### Three honest limits

1. **s⋆ is the *complex* distance to instability.** The actual perturbation
   A_λ − Â is real. Measured by the committed validator: on normal operators the
   extremal direction is attainable by a real perturbation and there is no gap; on
   the non-normal cases the best real perturbation at the same norm reaches only
   ρ = 0.9907 / 0.9992 where the complex extremal reaches 1.0000. D2 is therefore
   sound but slightly conservative for real perturbations on non-normal operators;
   the real structured stability radius is a further improvement and is
   **UNMEASURED**.
2. **The lower bound on s⋆ is conservative where the coarse chord is large
   relative to s⋆.** In the strong-non-normal case the certified bound 2.880e-3
   sits 19% below the 3.564e-3 the finer extremal search located. The gate uses
   the lower bound, so this costs tightness and never soundness.
3. **D2 is not implemented in the shipped package.** It exists in the validator
   only. Nothing in `alelyon.runtime.vector.ditherdmd` was changed.

*Note on figures.* An earlier ad-hoc run of the same searches reported
0.999665 for the soundness worst case and 0.9865 / 0.9802 for the real gap. Those
came from a different random draw and different generated cases. The figures above
are the ones `research/papers/run_pseudospectral.py` reproduces, which is the
script the papers name.

### Incidental

D2 has no `EIGENBASIS_ILL_CONDITIONED` refusal class, because it needs no
eigendecomposition. It is evaluable on defective Â, where Bauer–Fike must refuse.

---

## R2 — Is the per-element margin guard genuinely per-element everywhere?

*Open question from `01-certified-number-envelope.md` §9.1.* **Answer: yes, on
every site reachable in the DSL certified-execution path.**

Two independent guards, both per-element:

- **Deterministic** (`_delta_separated`, `_elementwise_separated`). Literally
  per-element with each element's own Δ: the extremum check is
  `gap_j − (d_j + d_win)/2 > 0` for every competitor j; the element-wise check is
  `|v_i − t| > d_i/2` for every i.
- **Empirical** (`execcert.py`, the `emp_ok` predicate). Per site, compares the
  **minimum** per-element margin against the **maximum** per-element
  perturbation. Since min_i margin_i > 3·max_i pert_i implies
  margin_j > 3·pert_j for every j, this is strictly stronger than a per-element
  check, not weaker.

Evidence: 12 isolated unit cases against the two deterministic guards, including
the aggregate-masking shape the red team originally found (a benign element with
a tiny Δ pinning the series minimum while a different element sits inside its own
Δ/2) — all correct. 5 end-to-end programs; adversarial cases refuse, positive
controls certify at `branch-stable-exact`. A repository-wide search for a margin
or gap quantity combined with an aggregation returns three sites, none of which
aggregates a guard.

---

## R3 — Does `branch-stable` admit a program it should refuse?

*Open question from `01-certified-number-envelope.md` §9.2.*
**Answer: yes. `branch-stable-first-order` is unsound on aggregate decisions when
the storage error is not dithered. `branch-stable-exact` is not affected.**

### The defect

```
certified_run('show sign(mean(series("X")) - c)')
  ->  ok=True, class=branch-stable-first-order, width=0.0
```

while the true decision has the **opposite sign**. A certificate of width zero
over a flipped answer.

**Mechanism.** The empirical guard compares the base margin against the
perturbation observed across K *independent* dither resamples. Independent
resampling moves an n-row aggregate by Θ(Δ/√n). A systematic rounding obeying the
same per-element bound |stored − true| ≤ Δ/2 moves it by Θ(Δ). The fixed
`BRANCH_MARGIN_SAFETY = 3.0` is therefore defeated once √n ≳ 3.

| n | systematic shift | √n | certified? | decision flipped? |
|---|---|---|---|---|
| 25 | 0.0499 | 5.0 | refuses | yes |
| 100 | 0.0499 | 10.0 | **yes** | yes |
| 400 | 0.0499 | 20.0 | **yes** | yes |
| 1600 | 0.0499 | 40.0 | **yes** | yes |

The crossover sits exactly where the mechanism predicts.

### Where the premise sits

Under **genuine subtractive dither the guard is sound**: 0 of 400 draws flip the
decision. The exposure is that `Declared(values, delta=D)` defaults to
`law=None`, which `fetch.py` documents as relative-dither, while the same
module's prose teaches the *interval* reading ("the width of the interval a
stored value could have come from"). The guard needs the independence reading;
nothing checks which one the caller's data satisfies.

This is CLAIMS.md §2 rule 3 — validate against an independently-held invariant,
never against the shape of what the writer emitted — reappearing one level up:
the resampler validates against the writer's declared Δ *under an assumed law*.

### Not affected

`branch-stable-exact` is immune by construction: the deterministic guard uses the
worst-case per-element Δ/2, which is systematic-safe. Randomized search with
adversarially thin margins: **14,833 exact-tier firings, 0 soundness violations**
across sign, min, max and comparison-vs-constant.

### Status

Audit only. Not fixed — AGENTS.md §12 makes an audit read-only. Fix direction:
either the empirical guard's perturbation scale for a decision on an aggregate of
n certified rows must use the worst-case Δ/2 rather than the resample spread, or
`Declared` must require an explicit law and `certified_run` must refuse
aggregate-grounded branch decisions under any law not guaranteeing independence.
Published to the fleet as `28ee148cbefb7d7f`, which **reached nobody** — recorded,
not delivered.

---

## R4 — Does the RSS composite's independence assumption hold?

*Open question from `01-certified-number-envelope.md` §9.3.* **Answer: it is
measurably false, it fails in the conservative direction, and it is the wrong
thing to have been worrying about.**

Measured correlation between the quantization and sampling error components
across replications (AR(1) log-returns, φ = 0.35, T = 300, `mean(returns(·))`):

| Δ | q/s ratio | correlation |
|---|---|---|
| 1e-4 | 1.6e-6 | −0.031 |
| 1e-2 | 1.6e-4 | −0.033 |
| 0.5 | 8.6e-3 | −0.168 |
| 2.0 | 4.8e-2 | −0.343 |
| 8.0 | 3.4e-1 | **−0.388**, 95% CI [−0.457, −0.314] (n = 600) |

The correlation is significantly negative and grows with the quantization/
sampling ratio. **Negative covariance means Var(q+s) < Var(q) + Var(s), so the
root-sum-square overstates the combined width.** The composite is never narrower
than the truth on this evidence. At the ratio the platform actually operates at
(q/s ≈ 1e-4 at 24-bit capture) the correlation is indistinguishable from zero, so
the assumption is harmless where it is used.

**The more useful finding.** The composite's coverage shortfall is not
attributable to the composition at all. At Δ = 0 — with no quantization term in
play — coverage is **0.923 ± 0.011** against a 0.95 target (n = 600). The deficit
belongs to the circular block bootstrap's finite-sample behaviour on serially
correlated data, a known property of the estimator. Adding the quantization term
moves coverage *up* (0.973 at Δ = 8), not down. The open question pointed at the
composition; the measurement points at the sampling term.

---

## R5 — Is the canonical encoding canonical under a second encoder?

*Open question from `02-exact-coordinate-registration.md` §9.1.* **Answer: no.
One confirmed fork and one confirmed ambiguity; a third arm closed clean.**

ADR-0002 records this gap itself — "a cross-implementation decoder remains
UNMEASURED", "the encoding has one implementation". A second encoder was written
from the ADR's stated rules rather than from `canonical.py`, deriving field order
mechanically from `dataclasses.fields()` as the ADR's wording directs.

**1. CONFIRMED FORK.** The ADR says *"Field order is the declared dataclass
order, fixed by the schema version."* `CoordinateSpace.schema_version` is
declared eleventh and encoded **first**. For `morphometry.canonical_space()`:

| encoder | content reference |
|---|---|
| shipped `canonical.py` | `sha256:7b4aad345c23bfa824e8f82313f08ec4…` |
| second encoder, ADR read literally | `sha256:9b39c66dc51b634e6fe06f13f05b9f68…` |

Two commitments for one space. An implementer following the written rule derives
a reference nobody else derives.

**2. CONFIRMED AMBIGUITY.** `CoordinateSpace.declared_bytes` carries no leading
underscore and is excluded by nothing the ADR says, yet is not encoded.
`CoordinateAxis`'s two excluded fields (`_label_membership`, `_declared_bytes`)
*do* carry underscores, so the Python-private convention covers the axis record
and not the space record. An implementer including it forks a third way.

**3. NEGATIVE — arm closed.** The NFC rule does **not** fork. `canonical.py`
never normalizes, but `contracts.py` refuses non-NFC text at construction
(`axis_id must use NFC-normalized Unicode`), so the encoder never receives one.
Refusing rather than silently normalizing is the correct choice and matches the
house refusal doctrine.

**4. POSITIVE CONTROL.** The **axis** encoding reproduced bit-for-bit from the
ADR's rules alone, on every axis tested. The method detects agreement as well as
divergence, so the divergence above is a real signal rather than an artifact of a
sloppy second encoder.

**Classification.** This is a *specification* defect, not an implementation
defect. The shipped bytes are pinned by golden vectors and are the de facto
standard; what is wrong is the written rule that would let a second party
reproduce them. Published to the fleet as `8e6fae80edf0bca3`.

---

## What this changes in the papers

| finding | effect |
|---|---|
| R1 | Paper II's conditioning ceiling is **not** intrinsic at the operator rung. The ceiling was an artifact of Bauer–Fike; a direct, optimal bound fires 5–6 bits earlier. The QP-Regret instance is a different mechanism and survives. |
| R3 | Paper I gains a confirmed unsoundness in the salvage tier, with its exact premise boundary. Load-bearing, not hidden. |
| R4 | Paper I's stated open question is answered *and relocated* — the honest limitation is the bootstrap's coverage, not the composition. |
| R5 | Paper I's canonicality question resolves negative: the encoding is canonical in fact and not in specification. |
| R2 | Confirms the per-element discipline holds. A negative result for the attacker, reported as such. |
