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
3. ~~**D2 is not implemented in the shipped package.** It exists in the validator
   only. Nothing in `alelyon.runtime.vector.ditherdmd` was changed.~~
   **Closed 2026-08-05 — see below.**

### Update, 2026-08-05 — limit 3 closed, and a defect found in closing it

D2 now ships as `alelyon.runtime.vector.ditherdmd.direct_contracting`, with s⋆
exposed separately as `distance_to_instability`. The Bauer–Fike gate was **not**
removed; `run_pseudospectral.py` now imports the shipped implementation instead of
carrying its own, so the sweep validates what ships.

Writing the falsifiers surfaced a defect in the validator this log quotes. Its
"RIGOROUS lower bound" subtracted `sin(h/2)` — half the chord of the *full* grid
arc — where the worst-uncovered point is the arc **midpoint**, at angular distance
h/2 and chord `2·sin(h/4) = sin(h/2)/cos(h/4)`. The smaller constant subtracts too
little, so the lower bound could **exceed** the true s⋆. Exhibited by a normal
operator with an eigenvalue parked at an arc midpoint at radius 1 − 1e-9: the old
constant returns +7.5e-3 against a true s⋆ of 1e-9 on an 8-point grid, and stays
above the truth at 16, 64 and 256 points.

**The figures in this log and in Paper II are unchanged.** At the declared
4096-point grid the overstatement is ≈5.6e-11, and the refined stage that actually
binds is ≈3.4e-18, both far below the reported bounds of 5.574e-3 and 7.217e-1.
What failed was the property the bound was described as having, not the numbers it
produced — recorded here rather than quietly repaired, since this log is the
evidence a reader re-derives the conclusions from. Published to the fleet as
`6f267a5abb32cd5f`, which reached 1 session.

One thing became provable in the move and strengthens R1's answer. For any |z| = 1,
Bauer–Fike gives σ_min(zI − Â) ≥ (1 − ρ(Â))/κ₂(V), so **s⋆ ≥ (1 − ρ(Â))/κ₂(V)**:
every Bauer–Fike firing is a D2 firing. The 53× ratio measured above is therefore a
floor on the improvement rather than a trade between two criteria — which is what
makes shipping both, rather than choosing, the honest arrangement.

A limit that was implicit is now written down: the grid argument is exact, but the
σ_min evaluations are ordinary floating-point SVDs and are not carried in interval
arithmetic. The bound is rigorous *given* those evaluations. End-to-end rigour is
**UNMEASURED**.

#### Re-measured on the shipped code, 2026-08-05

The R1 table above cannot be reproduced exactly: it was taken on a STATE matrix of
k = 14 features × W = 1339 weekly snapshots, and `globals/history.db` has since grown,
so `build_state` now yields k = 11 × 1568. The figures below are therefore a fresh
measurement of the same **property** on the current data, not a re-run of that table.
8 dither redraws per depth, δ = 0.05, read-only load, shipped gates:

| b | median r̂ | median κ₂(V) | BF ρ_bound | BF fires | median s⋆ lb | D2 fires |
|---|---|---|---|---|---|---|
| 10 | 6.159e-2 | 48.03 | 3.950 | 0/8 | 5.996e-3 | 0/8 |
| 12 | 1.509e-2 | 48.11 | 1.718 | 0/8 | 5.986e-3 | 0/8 |
| **14** | 3.753e-3 | 48.07 | 1.173 | 0/8 | 5.985e-3 | **8/8** |
| 16 | 9.371e-4 | 48.05 | 1.038 | 0/8 | 5.986e-3 | 8/8 |
| 18 | 2.342e-4 | 48.05 | 1.004 | **0/8** | 5.986e-3 | 8/8 |

ρ(A_λ) = 0.992584 against a 1 − ρ margin of 0.007416, closely tracking the 0.99264 /
0.00736 the original run saw on the smaller matrix. **Bauer–Fike still never fires,
not even at 18 bits** — its ρ_bound is 1.004 there and it would need b ≈ 19 — while
the direct gate fires from **b = 14**, a 5-bit reduction. The BF threshold on r̂ is
(1 − ρ)/κ₂(V) = 1.543e-4 against s⋆ ≥ 5.985e-3, a factor of **38.8×**; the original
run measured 53.5× on the other matrix. Zero D2 soundness violations: the run asserts
`fired ⇒ ρ(A_λ) < 1` on every evaluation.

This is the Gate 3 abstention in `docs/PROOFS.md` reproduced on current data and then
passed — by changing the bound, not the bit depth or the operator.

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

~~Audit only. Not fixed — AGENTS.md §12 makes an audit read-only.~~ Fix direction:
either the empirical guard's perturbation scale for a decision on an aggregate of
n certified rows must use the worst-case Δ/2 rather than the resample spread, or
`Declared` must require an explicit law and `certified_run` must refuse
aggregate-grounded branch decisions under any law not guaranteeing independence.
Published to the fleet as `28ee148cbefb7d7f`, which **reached nobody** — recorded,
not delivered. **FIXED 2026-08-05 — see below.**

### Fixed, 2026-08-05 — the first fix direction, and the defect was wider than recorded

The guard now takes its perturbation scale from the K dither resamples **and** two
worst-case systematic probes: every element of every input at `+Δ/2`, then at
`−Δ/2`. The margin must clear 3× the largest movement any of them produces, and a
decision that changes under either probe refuses with a reason naming the probe
rather than the resamples — independent dither never finds this case, so blaming
the resamples would send a reader to look at `K`.

**The defect was wider than the table above records.** Re-running the shape at
`0.499Δ` systematic offset, it certifies at **every** n tested, n = 25 included —
not only above the √n ≳ 3 crossover. The recorded n = 25 refusal was a property of
that run's particular margin, not a floor. Measured before the fix, all four
inverted, all four at `branch-stable-first-order` with **width 0.0**:

| n | margin | perturb_scale (dither only) | 3× | Δ/2 | pre-fix | post-fix |
|---|---|---|---|---|---|---|
| 25 | 0.0498 | 1.19674e-2 | 3.59021e-2 | 0.05 | certified −→+, width 0.0 | refused |
| 100 | 0.0498 | 6.0616e-3 | 1.81848e-2 | 0.05 | certified −→+, width 0.0 | refused |
| 400 | 0.0498 | 3.01784e-3 | 9.05353e-3 | 0.05 | certified −→+, width 0.0 | refused |
| 1600 | 0.0498 | 1.58006e-3 | 4.74017e-3 | 0.05 | certified −→+, width 0.0 | refused |

The margin is 0.0498 and Δ/2 is 0.0500: the decision sat inside the bound the
declaration permits at every n, and only the *observed* scale made it look safe.
`perturb_scale` halves as n quadruples — 1.197e-2, 6.06e-3, 3.02e-3, 1.58e-3 —
which is the Δ/√(12n) cancellation the mechanism predicts, measured rather than
argued. The 3× factor never had to be beaten by much: at n = 25 it is already
3.59e-2 against a 4.98e-2 margin.

**Why probes rather than an explicit-law refusal.** The second fix direction —
require a law and refuse under any law not guaranteeing independence — refuses
programs that are genuinely safe, because it never looks at the margin. The probes
measure the bound the declaration actually gives, which is what CLAIMS.md §2 rule 3
asks for, and they are monotonically stricter: adding a probe can only raise
`perturb_scale` and can only clear `stable`, so nothing previously refused becomes
admissible. `branch-stable-exact` is untouched — its deterministic Δ-separation
guard was always worst-case and never consulted the resamples.

**What is still not a theorem.** The probes are the two uniform-sign corners. A
mixed-sign perturbation that moves a non-monotone program further is **UNMEASURED**,
and the tier keeps its `-first-order` name for that reason. The width is also
untouched by design: it remains a conformal order statistic over the dither law,
and a worst-case corner is not a draw from that law, so the probes are excluded
from it. A test asserts the width stays on the dither scale rather than the Δ/2
scale.

**Compatibility.** `branch_sites` is replay-compared, so an envelope issued before
this carries the old, smaller `perturb_scale` and will not match a replay under the
new code. Intended: those envelopes asserted a guard that did not hold. No golden
vector carries a branch tier and no external verification of one is recorded, so
nothing outside this repository is invalidated. SPEC-cne-v0 §7.5 and §8.5 amended.

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

### Followed up, 2026-08-05 — the relocated problem, characterised and gated

R4 relocated the deficit onto the sampling term and stopped. Two things were then
true and neither was written down: the shortfall had been measured at exactly one
(φ, T), and **the term that binds had no coverage harness at all** while the
quantization term that does not bind has had one as a release gate since W4.

Measured against a known population value — prices built as `cumprod(1 + r_t)`
with `r_t` a mean-zero AR(1), so `returns(·)` recovers `r_t` exactly and the truth
is 0 by construction — 300 replications per cell, B = 400, nominal 0.95:

| φ | T | coverage | se |
|---|---|---|---|
| 0.00 | 300 | 0.920 | 0.016 |
| 0.35 | 300 | 0.913 | 0.016 |
| 0.35 | 100 | 0.890 | 0.018 |
| 0.70 | 300 | **0.867** | 0.020 |
| 0.70 | 1000 | 0.907 | 0.017 |

So it is systematic, not a single cell: the shortfall grows with serial
correlation and with small T, and is still present at T = 1000. R4's 0.923 was the
mildest case in the range.

**Block length was tested rather than blamed.** Sweeping `L = mult · T^(1/3)` on
the estimator directly at T = 300:

| φ | mult 1 | 2 | 3 | 4 | 6 |
|---|---|---|---|---|---|
| 0.35 | 0.943 | 0.945 | 0.935 | 0.930 | 0.917 |
| 0.70 | 0.882 | 0.915 | 0.920 | 0.910 | 0.882 |

A longer block buys ≈4 points at φ = 0.70 and **costs** coverage at φ = 0.35, and
no setting reaches 0.95. The rule of thumb is therefore **kept**: changing it would
move the dominant term of every budget in exchange for a trade rather than a fix.
This is a design truth of the percentile bootstrap at finite T, not an INFRA GAP —
so it is stated, not engineered around.

**What shipped.** `Term.extra` now carries `nominal_conf` and a `coverage_status`
naming the measured range, so the calibration travels with the number instead of a
reader assuming the 95% is attained; and
`tests/certification/test_budget_sampling_coverage.py` is the missing gate. It
deliberately does **not** assert coverage ≥ 0.95, which would be false. It pins the
behaviour in both directions — a floor per cell, and an assertion that coverage
does **not** reach nominal, because CLAIMS.md §2 rule 6 says a mutation that only
widens a bound passes every coverage test, so a width falsifier has to be able to
fail upward.

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
