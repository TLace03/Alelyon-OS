# SRC-VaR

### Certified quantiles, VaR and expected shortfall from subtractively-dithered storage

**Status:** proof document plus empirical validation. Refereed. **The headline
claim was retracted in review** — see §6. Validator: `research/papers/run_srcvar.py`,
seed 20260720, fully offline.

---

## Abstract

If numeric data is stored with *subtractive dither* — a known pseudorandom
offset added before quantisation and subtracted after — the resulting error is
uniform on (−Δ/2, Δ/2), independent of the signal, and its distribution is known
exactly rather than assumed. That is a strong enough premise to ask what
downstream statistics can be certified from stored data alone.

This paper answers it for quantiles, Value-at-Risk and Expected Shortfall. It
establishes a deterministic sandwich that holds with certainty 1 and no
distributional assumptions, an exact population smoothing-bias constant, a
finite-sample certified band, and a Richardson two-resolution extrapolation whose
convergence order reveals the smoothness regime.

It also reports, at equal prominence, that the certified population band is
**far wider** than the storage resolution it is built on, that the original
"beats the resolution floor" framing was wrong, and that a real price series on a
coarse tick grid **falsifies the smoothness hypothesis** the population
certificate requires — so the certificate refuses rather than returning a number.

## 1. Setting

Values are stored as `y = Q(x + d) − d` with `d` a known dither uniform on
(−Δ/2, Δ/2). The stored value differs from the clean value by an error that is
uniform, iid across elements, and independent of `x`. Δ is per-column and
committed.

Two targets are distinguished throughout, and conflating them is the error the
retraction corrects:

- the **empirical** quantile of the clean sample you happen to hold;
- the **population** quantile of the distribution the sample came from.

## 2. E1 — the deterministic empirical sandwich

**Claim.** With zero overload, the clean empirical *q*-quantile lies in
`[y_(k) ± Δ/2]` for **every** dither realisation. Certainty 1. No assumptions.

**Result.** Checked at *q* ∈ {0.01, 0.05, 0.95, 0.99} on a real series (XOM),
500 dither redraws at each of *b* ∈ {4, 6, 8, 10, 12}:

| b | Δ | coverage |
|---|---|---|
| 4 | 0.01258 | 100.00% |
| 6 | 0.003146 | 100.00% |
| 8 | 0.0007865 | 100.00% |
| 10 | 0.0001966 | 100.00% |
| 12 | 4.916e-05 | 100.00% |

**PASS** — 100.00% over 10,000 checks. It must be exactly 100%; this is a
deterministic statement, and anything less would falsify it.

This is the strongest result in the paper and the one with the fewest conditions.
It is also the least ambitious: it bounds the error against *your own sample*,
not against the population.

## 3. P1 — exact population smoothing bias

Dithering convolves the population CDF with a uniform kernel, which biases the
population quantile. The bias is bounded by `L·Δ²/24` with `L = sup|f′|`.

Validated on a moment-matched 2-component Gaussian mixture with **known**
L = 1061:

| b | Δ | measured sup&#124;F_Y−F&#124; | bound L·Δ²/24 | ratio |
|---|---|---|---|---|
| 4 | 0.03511 | 0.04724 | 0.05448 | 0.867 |
| 6 | 0.008776 | 0.003373 | 0.003405 | 0.991 |
| 8 | 0.002194 | 0.0002127 | 0.0002128 | 0.999 |
| 10 | 0.0005485 | 1.33e-05 | 1.33e-05 | 1.000 |
| 12 | 0.0001371 | 8.314e-07 | 8.314e-07 | 1.000 |

**PASS** — the constant bounds the measured bias at every depth, and the ratio
approaching 1.000 from below shows the bound is tight rather than merely valid.

## 4. P2 — the certified population band

A DKW-driven finite-sample band, composed with the P1 bias term. Synthetic iid
data, *q* = 0.05, δ = 0.05, n = 5000 per redraw, 400 redraws, b = 8.

- true population quantile: −0.02783
- bands emitted (not refused): 400 / 400
- **coverage 100.00%** against a 93% threshold

**PASS.**

## 5. T3 — Richardson two-resolution extrapolation

Storing at two resolutions and combining as `(4F^{Δ/2} − F^{Δ})/3` cancels the Δ²
term. The residual order reveals the smoothness regime, and the measured log-log
slope against Δ recovers it:

| declared regime | predicted order | measured slope | gate |
|---|---|---|---|
| f″-Lipschitz (C∞ Gaussian mixture) | O(Δ⁴) | **3.989** | ≥3.5 PASS |
| f′-Lipschitz (quadratic B-spline, f″ has jumps) | O(Δ³) | **2.973** | ≥2.8 PASS |

The measured orders match the T3 constants (Δ³/384 and Δ⁴/4608). This is the
result we find most interesting: the *convergence rate is itself a measurement of
the smoothness assumption*, so a caller who declares more smoothness than the
data has can be caught by the slope.

## 6. P4 — the retraction

The original framing claimed the certificate "beats the resolution floor" — that
the certified band could be narrower than Δ. Referee review killed it. The honest
table, reported as plain constant factors with **no order-of-magnitude claim**:

| b | Δ | band `4β/f_lb` | sandwich `Δ` | asymptotic `LΔ²/12f_lb` | band/sandwich |
|---|---|---|---|---|---|
| 4 | 0.03511 | 0.2403 | 0.03511 | 0.08882 | 6.84 |
| 6 | 0.008776 | 0.07372 | 0.008776 | 0.005551 | 8.4 |
| 8 | 0.002194 | 0.06332 | 0.002194 | 0.000347 | 28.9 |
| 10 | 0.0005485 | 0.06266 | 0.0005485 | 2.168e-05 | 114 |
| 12 | 0.0001371 | 0.06262 | 0.0001371 | 1.355e-06 | 457 |

**Reading.** The certified population band is a finite-sample DKW object. Its
width is floored by the sampling term `sqrt(ln(2/δ)/2n)`, which does not shrink
with bit depth at all. So at these sample sizes it is **far wider** than either
the per-sample resolution Δ or the population smoothing scale — and the gap
*grows* with b, reaching 457× at 12 bits, because Δ keeps shrinking while the
band does not.

**The one-sentence conclusion: dither buys the exact checked error model, not a
smaller floor.**

This is the same finding the decomposed error budget reports elsewhere in the
program — sampling binds, quantization does not — arriving here from a completely
different direction. We regard the convergence of those two as the most reliable
thing in the whole research line.

## 7. SMOOTHNESS_FALSIFIED — a refusal on real data

The population certificate requires an L-Lipschitz density. Real prices on a
coarse tick grid do not have one.

Series AITX: 1500 raw closes, 637 distinct values — genuine point masses. At
b = 8, Δ = 0.1027:

- densest 2Δ window mass 0.2533 versus L-Lipschitz threshold 0.1093
- **falsifier fired**
- population band **refused**, reason `SMOOTHNESS_FALSIFIED`
- E1 empirical sandwich still valid, and still contained the clean quantile
  across 200 redraws

**PASS** — the declared hypothesis is refuted by the atoms, the population
certificate refuses rather than fabricating a number, and the assumption-free
result survives.

We consider this the paper's most useful output for a practitioner. The
observable falsifier means the smoothness assumption is not something you declare
and hope about; it is something the data can reject.

## 8. Honest degradation

Every trigger is observable from stored data alone:

- overload (clipping) — invalidates E1, detected by counting boundary codes
- smoothness falsification — the 2Δ-window mass test above
- insufficient sample — the DKW term dominates and the band is reported as such
- unknown or unrecognised capture law — refuse; unknown Δ semantics are unknown,
  not lenient

## 9. Prior art

Subtractive dither and its uniform-independent error model are classical
(Schuchman conditions; Lipshitz–Wannamaker–Vanderkooy). DKW is standard. The
Richardson device is standard numerical analysis. Order statistics of quantiles
under quantisation have been studied.

Our contribution is the **composition with audited finite-sample constants and
named refusal paths**, plus the observation that the extrapolation order is a
usable test of the smoothness declaration. We claim assembly and honesty
discipline, not new theory — and the retracted floor claim is the record of what
happens when that posture slips.

## 10. Reproduction

```bash
python research/papers/run_srcvar.py
```

Reads committed data read-only. Seed 20260720. Bits {4,6,8,10,12}. Fully offline.
Refusals return `ok=False` with a named reason, never a number.
