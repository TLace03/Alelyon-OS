# Certified Computation over Compressed and Committed Data

**A series of five working papers · Alelyon Quantitative Services · 2026-08-04**

These are working papers, not peer-reviewed publications. Each names the
adversarial review it survived, or states that it has not had one. Several report
their headline claim being refuted and what was left standing.

---

## The series

Papers are numbered by dependency, not by date. **Read them in the order below**;
that order is not the numbering, and the reason is given in each entry.

| Order | Paper | Thesis |
|---|---|---|
| 1st | **[IV — What Does Not Transfer](IV-what-does-not-transfer.md)** | Four refutations of structured compression, and the artifact that survived. Reads first because it explains why a *certificate* rather than a compression ratio became this program's object. |
| 2nd | **[II — Certified Statistics from Subtractively Dithered Storage](II-certified-statistics-from-dithered-storage.md)** | A known error law supports a ladder of certified downstream objects. The ladder terminates before the decision — but one rung's apparent ceiling was an artifact of the bound, not of the problem. |
| 3rd | **[I — Replayable Certificates over Computation](I-replayable-certificates.md)** | A signed object should bind the computation, its per-input commitments, and a complete enumeration of what it does not establish. Verification is replay under an out-of-band key, not trust in the issuer. |
| 4th | **[III — Observed versus Declared](III-observed-versus-declared.md)** | Evidence classes are labeled at every read and never merged; where a class is unavailable the result is a named absence, not an estimate. One discipline, two unrelated substrates. |
| 5th | **[V — The Decoder, Not the Weights](V-the-decoder-not-the-weights.md)** | No training procedure can guarantee a model will not invent a figure. A constrained decoder can, because the figure is not in the grammar. |

Paper II is the flagship and the only one carrying theorems. Papers I and II
depend on each other's results and are cross-referenced; III and V are
self-contained.

---

## The honesty contract

Every paper is subordinate to the claim discipline the program operates under, and
each restates it in front matter because a reader of one paper never sees this
page.

1. **Name the certified object.** "Certified" without a noun launders a small
   guarantee over large ones. These certificates bound *storage quantization*,
   which at realistic capture precision is several orders of magnitude below
   sampling error. The decomposed budget says so rather than concealing it.
2. **Revision is detected; invention is not.** A producer who fabricates data at
   capture signs a receipt that verifies perfectly. Verification establishes that
   committed inputs were not revised after commitment; it does not audit a source.
3. **An absent measurement is reported as UNMEASURED.** A blank beside a filled
   field reads as "checked, and fine". Every absent term is named with its reason.
4. **Refusal is a first-class outcome.** Several results here are certificates
   that refuse on real data. That is the system operating correctly.

---

## What these papers report against themselves

The series' distinguishing feature is that the negative results are load-bearing.
The following are stated in the papers at the prominence of results, not in
footnotes.

| Finding | Where |
|---|---|
| The central compression hypothesis was refuted for both target data classes; the ratio is governed by per-element bit width, which commodity formats already deliver | IV §3, §5 |
| A "beats the resolution floor" claim was **retracted** in review; the measured gap grows to 457× at 12 bits | II §3.5 |
| Two independently constructed bit allocators failed their pre-registered criteria (+0.01%/+0.18%, and −203% to −1294%); neither was released | II §8 |
| The decision-regret certificate certifies **zero cells** at the pre-registered bit depths on a real panel | II §6.3 |
| A conditioning ceiling reported as structural is shown to have been an **artifact of the bound** at one of its two observed instances | II §5.3, §7 |
| The first-order branch-stability salvage is shown **unsound** on aggregate decisions under non-dithered error, certifying a flipped decision at width zero | I §9.2 |
| The canonical encoding is canonical in fact and **not in specification**; a second encoder derives a different content reference | I §9.4 |
| A sealed language-model evaluation scored 140/140 by **copying the exemplar**; the win was certification, not reasoning | V §6 |

---

## Research log

[**00-research-log.md**](00-research-log.md) records the five open questions the
earlier drafts listed as unresolved, the measurements that closed them, and the
three findings that changed a paper's thesis. It is the evidence behind Paper I §9
and Paper II §5.4–§5.5, and it includes the arms that closed negative.

---

## Source material

The five papers above consolidate ten earlier single-artifact notes, retained in
this directory as the working record. They are superseded for citation purposes.

| Note | Consolidated into |
|---|---|
| `01-certified-number-envelope.md`, `02-exact-coordinate-registration.md` | Paper I |
| `03-model-morphometry.md`, `04-observed-versus-declared.md` | Paper III |
| `05-src-var.md`, `06-speccert.md`, `07-dither-dmd.md`, `08-qp-regret.md` | Paper II |
| `09-what-does-not-transfer.md` | Paper IV |
| `10-the-decoder-not-the-weights.md` | Paper V |

The consolidation is by thesis rather than by artifact. Three of the series'
strongest results existed only *between* the earlier notes — the conditioning
ceiling appeared in two, allocation futility in two, and "sampling binds,
quantization does not" in three. Split across files those read as coincidences;
merged, they are results.

---

## Reproduction

Each paper carries a reproduction appendix naming its scripts, seeds and
environment. The distributed systems can be exercised directly:

```bash
pip install "alelyon-os[verify,certify,lattice,fleet,workspace,compute]"
alelyon-verify selftest
```

Validators for the mathematical results live in `research/papers/` in the private
monorepo and read only committed data. Expect 3 of 9 golden vectors verified on
the portable fallback and 9 of 9 with the deterministic kernel; a claim of 9 of 9
on a fallback build indicates a fault.

---

## Status of the bibliography

Reference lists were assembled from the authors' working bibliography and have
**not** had a bibliographic review. Identifiers should be checked against primary
sources before external publication. Each paper's reference section carries this
notice.
