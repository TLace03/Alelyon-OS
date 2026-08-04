# Observed versus Declared: A Discipline for Measurement Systems That Must Not Guess

---

## Standing honesty contract

1. **Name the certified object.** Nothing in this paper certifies capability,
   correctness or intent. It measures declared structure and observed occupancy.
2. **Revision is detected; invention is not.** Where a fact originates with the
   party being measured, it is labeled as such and never promoted.
3. **An absent measurement is reported as UNMEASURED.** In this paper the rule is
   load-bearing rather than decorative: `UNATTRIBUTED` and `UNMAPPED` are *values*,
   never blanks.
4. **Refusal is a first-class outcome.** A model whose parameter distribution the
   analytic path cannot compute is refused by name rather than approximated.

---

## Abstract

Two measurement systems are described that share one discipline and no subject
matter. The first measures the structure of a language model — how parameters
distribute across transformer blocks and across functional roles, at what storage
precision — from what a local runtime declares about a model it holds, without
reading a weight, running a forward pass or observing an activation. The second
coordinates concurrent autonomous coding sessions in a shared repository from
records those sessions did not author: the version-control system's own worktree
administration, commit reachability, and the file paths a session has touched. In
each system the evidence separates into two classes that are stored apart, labeled
at every read, and never merged; where a class is unavailable the result is a named
absence rather than an estimate; and where the available class supports only a
weaker conclusion, the weaker conclusion is what is reported. Each system places
its measurements on a canonical coordinate space with an explicit account of where
that space came from, and each under-approximates deliberately in a stated
direction. The paper argues that this pairing — the same discipline demonstrated on
two unrelated substrates — is a stronger claim than either instance, and reports a
field observation in which the coordination system's two most consequential
declared limits were both encountered in ordinary use within a single working
session.

**Keywords:** provenance, evidence classes, named absence, canonical coordinate
spaces, model structure, distributed coordination, advisory protocols,
under-approximation.

---

## 1. Introduction

A measurement system that cannot make a measurement has three options. It can
estimate, and report the estimate as a measurement. It can return a blank. Or it
can return a named absence: a value that says which measurement is missing and
why.

The first is the familiar failure. The second is subtler and, in practice, worse:
a blank beside a filled field reads as *checked, and nothing there*. Only the third
preserves the distinction between a fact and the absence of one, and it requires
the system's type discipline to carry absence as a first-class value.

This paper reports two systems built to that rule, on substrates with nothing in
common. They are presented together because the pairing is the claim. A single
system exhibiting a discipline demonstrates that the discipline is implementable
for that problem. Two unrelated systems exhibiting it — one measuring static
structure from vendor metadata, one measuring dynamic occupancy from version-control
administration — demonstrate that it is a transferable engineering practice.

### 1.1 The shared discipline, stated once

1. **Evidence classes are separated at the type level.** Two kinds of fact never
   occupy one field. Merging them is not discouraged; it is unrepresentable.
2. **Every read is labeled with its class.** A consumer never has to infer which
   kind of evidence produced a value.
3. **An unavailable class produces a named absence.** `UNATTRIBUTED`, `UNMAPPED`,
   a `None` bit width with a stated gap, or a refusal — never a blank and never a
   default.
4. **Measurements are placed on a canonical coordinate space whose provenance is
   explicit.** Two results are comparable when they name the same space, and the
   space reference is how a consumer determines that without inspecting either.
5. **Where approximation is unavoidable, its direction is stated.** Both systems
   under-approximate, and each says which way the error runs and what that makes
   the result unsafe for.

### 1.2 Contributions

1. A morphometry of language model structure computed entirely from declared
   metadata, with two evidence sources that cannot be blended, and a named refusal
   where the weaker source's arithmetic does not apply (§3).
2. A coordination layer for concurrent autonomous sessions built on derived rather
   than self-reported attribution, with the single case where it returns a
   confident wrong answer isolated and printed at every read (§4).
3. A resolver for the repository coordinate space with three outcomes — declared,
   discovered, empty — that never places work into a directory it did not observe
   (§4.3).
4. A field observation in which both of the coordination system's principal
   declared limits were encountered in ordinary use, which is offered as evidence
   that printed limits are read rather than ornamental (§5).
5. An argument that under-approximation is safe for routing and unsafe for gating,
   and that the difference must be stated rather than left to the consumer (§6).

---

## 2. Notation and preliminaries

| Term | Meaning |
|---|---|
| Observed | A fact derived from records the measured party did not author |
| Declared | A fact the measured party states about itself |
| Named absence | A value denoting a specific missing measurement and its reason |
| Coordinate space | An axis system with committed labels, ordering and a content reference |
| Cell | One coordinate on that space |
| Under-approximation | An estimate guaranteed to be no larger than the truth |

**Definition 2.1 (evidence class).** A partition of facts by the identity of the
party that produced the record from which the fact is derived. A fact is *observed*
when that party is not the party being measured, and *declared* otherwise. The
distinction is about authorship of the record, not about accuracy: an observed fact
can be wrong, and a declared fact can be correct.

**Definition 2.2 (named absence).** A value a ∈ A, disjoint from the value domain
V of a field, such that a identifies which measurement is missing and carries a
reason. A field of type V ∪ A cannot represent "missing" as a member of V, which is
what distinguishes this from a sentinel default.

**Definition 2.3 (derivation versus authentication).** A *derivation* infers an
identity from a record's structure under a stated rule. An *authentication*
establishes it against a secret. A derivation can be satisfied by anything able to
produce the structure, and is therefore not an authentication however reliable it
is in practice.

---

## 3. Substrate A: model morphometry

Neuroimaging morphometry measures structure — cortical thickness, regional volume,
surface area — on a canonical coordinate space, so that two brains become
comparable region by region [1, 2, 3]. The same construction applies to language
models.

### 3.1 The coordinate space

Two axes:

- **block** — ordinal; one cell per transformer block, plus cells for the
  pre-block embedding and the post-block head.
- **module** — categorical; the role a tensor plays: attention query, key, value
  and output projections; feed-forward gate, up and down projections;
  normalization; embedding; output.

Every tensor a runtime reports is assigned to exactly one cell by a name-matching
rule, or to an explicitly unassigned bucket. Per-cell measurements are parameter
count, byte footprint, nominal bits per weight, and the set of quantization types
present.

The coordinate space is itself committed: a content reference over the axes, their
ordering and the label dictionary, through the registration core of Paper I. Two
morphometries computed on the same space reference are comparable; two computed on
different references are not, and the reference is how a consumer determines this
without inspecting either.

### 3.2 Two sources that cannot be blended

**Tensor inventory** is the strong source and is *observed* in the sense of
Definition 2.1 relative to the architecture declaration: the runtime publishes each
tensor's name, element type and shape, so per-cell parameter counts are counted and
each tensor's quantization type is its own.

**Declared architecture** is the fallback. The runtime publishes only architecture
fields — block count, embedding length, feed-forward length, head counts — and
per-role parameter counts are computed analytically. This is arithmetic over
declared dimensions and is correct **only for the dense decoder shape it names**.

**A mixture-of-experts model with no tensor inventory is refused by name.** A
sparsely gated model's parameter distribution [9, 10] is not the dense formula's
answer, and returning the dense answer without a flag would be a confident wrong
number wearing the shape of a right one.

The result's `source` field always states which path ran. Mixing them in one result
is not possible — not discouraged, not possible. This is Definition 2.1 enforced at
the type level, and it is the design element §1.1(1) names.

### 3.3 Bit widths are nominal, and say so

The bits-per-weight table is a static record of each format's *published* cost. It
is the format's nominal figure, not a measurement of a file. A type absent from the
table yields a null bit width and a named gap rather than a guess.

This distinction matters because a quantization format's real footprint includes
block scales, zero points and padding, and the nominal figure understates it by a
few percent. Reporting nominal-as-measured would be a small error that compounds
when a consumer sums across a model to predict memory. The system reports both the
nominal figure and, where the runtime publishes byte sizes, the measured footprint,
and leaves the reconciliation visible rather than resolving it silently — a choice
§7 revisits as a limitation.

### 3.4 The boundary is the contribution

**Nothing here reads a weight, runs a forward pass, or observes an activation.**

The objection is immediate: a structural measurement that never reads a weight
cannot say anything about what the model has learned. That is correct, and it is
the point. Three arguments support the boundary.

**It is honest about what is derivable.** Nothing in a tensor inventory carries
information about learned behavior. A tool that measured structure and *implied*
capability would make a claim its inputs cannot support.

**It is inexpensive and safe.** Reading declared metadata requires no accelerator,
no model load and no inference, and touches no weights — so it can run against a
model the caller is not licensed to execute, in an environment with no accelerator,
in milliseconds. A structural comparison across twenty models is a table lookup.

**Structure answers the questions actually asked before deployment.** Where does
this model spend its parameters? Which blocks are stored at lower precision than
their neighbors? Do two checkpoints claiming the same architecture have the same
shape? These are answerable from declared metadata.

---

## 4. Substrate B: coordinating concurrent sessions

When several autonomous coding agents work in one repository at once, each behaving
correctly in isolation, they produce work that cannot be reconciled, because none
could see the others.

The apparent remedy is a registry every agent writes to. It fails twice: an agent
that does not participate is invisible, and an agent that participates dishonestly
is worse than invisible, because its declarations are indistinguishable from
evidence. The problem is a mild instance of the setting in which a participant's
self-report cannot be taken at face value [4].

### 4.1 Attribution is derived, never believed

The version-control system's author and committer fields are freely settable. A
worktree's directory name is chosen by whatever created it. Any session identifier
written into a file is written by the entity claiming that identity. All three are
*the writer describing itself*, so none is used for attribution.

What is used instead: the worktree's administrative path, its head commit, and
reachability from the mainline — records the version-control system keeps for its
own purposes; and the parent-directory convention, which identifies a **tool
family** and nothing finer.

Every derivation carries the rule that produced it in an evidence field, so a
reader can disagree with it. Where no rule fires, the answer is `UNATTRIBUTED` —
a value, per Definition 2.2. **Model identity is not derivable from version-control
records and is not guessed.** Tool family is the honest ceiling for a worktree found
on disk.

### 4.2 One genuine advance, and its exact limit

Some tools place a session identifier in the worktree path. Where that convention
holds, a session is identifiable from a location the version-control system records,
without the agent volunteering anything. That is a real improvement on
self-registration.

It remains a **derivation** and not an authentication (Definition 2.3): a worktree
can be created at any path, so anything able to choose a directory name can adopt a
session-shaped one.

More consequentially, it names the session that **created** the directory, not the
one working in it now. A second session can enter an existing worktree and leave
changes there, and every one of those changes is attributed to the creator. **This
is the single case in which the system reports a confident answer that is wrong**
rather than an honest named absence. It is printed in the limits at every read,
because a limit that is not stated reads as a limit that does not exist.

A second, independent derivation covers what the path cannot: a per-session
transcript file, named by the harness, carrying a working-directory field the
harness wrote. It **over-approximates** occupancy — naming every session that
*started* in a directory rather than the one editing it — and the asymmetry is
deliberate. A mailbox that reaches the intended recipient along with somebody else
is a working mailbox; a mailbox that reaches nobody is not.

### 4.3 The coordinate space belongs to the repository

Work is placed on a (pillar, surface) coordinate, and routing is derived from those
coordinates, so whatever places a path decides who hears a finding about it.

The first implementation hard-coded one repository's directory table. Inside that
repository it was exactly right, and it was *why* attribution could be validated
against an independently held invariant: the table was owner-authored policy the
module did not get to edit.

Distributed as a general library, the same table was wrong in two directions
simultaneously. Every path in an unfamiliar checkout fell through it and read
`UNMAPPED`, so the coordination layer perceived an empty repository; and the table
asserted one private tree's layout as a general vocabulary.

**Table 1.** The resolver's three outcomes. The system reports which one ran.

| Outcome | Source |
|---|---|
| **declared** | A configuration file in the repository root; the repository states its own pillars |
| **discovered** | The repository's own tracked paths — the directories that exist in the checkout the user opened |
| **empty** | Neither is available; everything is `UNMAPPED`, and the operator surface says so rather than presenting an empty table |

Discovery is deliberately simple and explainable: a pillar is a top-level directory;
depth is one by default; a directory whose children are all files is *flat*, so the
file is the unit. The last rule exists because a directory of unrelated scripts
collapsed to a single surface makes one session editing one script read as occupying
all of it, which steers other work away from ground that is genuinely free.

**Nothing is ever placed into a directory that was not observed or declared in the
repository under observation. There is no ambient list.**

### 4.4 A claim is advisory, and that is the design

Territory can be claimed, and a claim is explicitly advisory. Two sessions can hold
one area; the system records the fact, flags it as contested, and refuses nothing.
Reconciliation is between the sessions rather than with the tool.

This is not a limitation awaiting a fix. A lock in this setting is either
unenforceable — nothing prevents an agent editing a file — or a deadlock generator,
when an agent terminates while holding one. Recording the contention and making it
visible is the achievable goal. The same reasoning produces advisory rather than
mandatory locking in operating systems.

### 4.5 Publishing reaches somebody, or it does not

When a finding is published, the system prints **who it reached**, and
`REACHED NOBODY` is a real outcome worded so that it cannot be read as success.

This came from an observed failure. A session claimed an area using a dotted
spelling — legitimate, because pillars contain dots — while routing derives a slashed
form. The claim was accepted, reported as successful, and the session was never
reachable. Four findings published at the real area reported `REACHED NOBODY` while
their author believed the territory had been announced.

The remedy has two halves and the second is the instructive one. Creating a
coordinate is now checked against the vocabulary and refused with a suggestion.
Reading one back is **not** checked, so a bad record can still be inspected and
released. A validator applied symmetrically would have rendered the broken claims
unreadable and therefore unreleasable.

### 4.6 A second axis: discipline is not location

Where a path lives and what discipline touching it requires are different questions,
and neither implies the other. A cryptographer and an interface developer editing
one directory occupy one area and two disciplines.

Disciplines are declared by the repository, in the same file as the pillars — one
repository, one description of it. Each names its trigger kind: **paths**, where the
policy lists the files it governs and the match is exact; and **reachability**, where
the policy names what code must *reach*, which is a property of the import graph
that this module does not compute, so it matches only the anchors named by path.

Reachability-triggered disciplines therefore **under-approximate**: fewer
disciplines than truly apply, never more. That direction is safe for routing a
finding to an interested party and **unsafe for gating a change**, so gating remains
with the policy document, continuous integration and a reviewer. The gap is named as
an infrastructure gap rather than closed with plausible path patterns, because a
guessed pattern reports a discipline as *checked* on ground nobody checked.

---

## 5. Field observation

The coordination system prints its limits at the end of every command. Whether
printed limits are read, or are ornamental, is an empirical question. During the
preparation of this series the system was used in ordinary working conditions, and
**both of its principal declared limits were encountered within a single session.**

**Table 2.** Declared limits and their observed occurrence.

| Declared limit (§4, §7) | Observed |
|---|---|
| "Two sessions sharing one checkout are one path and therefore one derived identity" | Four sessions were live in one checkout. The system **refused to guess**, naming all four candidate identities and stating that selecting one would be a declaration rather than an observation. An explicit identity had to be supplied on the command line, and the result was labeled as declared-but-corroborated. |
| "`REACHED NOBODY` is a real outcome and does not mean the fleet was warned" | A confirmed defect finding was published to the area it concerned and reached **nobody**. It is recorded and was not delivered. |

Two observations follow. First, the shared-checkout case behaved as the discipline
requires: faced with four candidates and no observed basis for choosing, the system
declined to attribute and said why, rather than selecting the most recent. Second,
the `REACHED NOBODY` outcome is not rare. A finding published into an area with no
observable occupant is recorded and undelivered, and the wording is what prevents
the author from believing otherwise. Both are instances of §1.1(3): the named
absence did the work a blank could not.

This is a single observation in one repository and is not a controlled measurement.
It is reported because the alternative — asserting that printed limits are useful —
would be a claim with no evidence behind it.

---

## 6. Discussion: under-approximation must state its direction

Both systems approximate, and both approximate downward.

Morphometry's name-matching rule places a tensor it does not recognize into an
unassigned bucket rather than into a plausible cell, so per-cell counts
under-report and the unassigned bucket carries the difference. Coordination's
reachability-triggered disciplines match only the anchors named by path, so fewer
disciplines are reported than truly apply.

In both cases the direction is the safe one *for the purpose the system serves* and
the unsafe one for a purpose it does not. An under-reported per-cell parameter count
is safe for comparing two models on a shared grid and unsafe for predicting memory.
An under-reported discipline set is safe for routing a finding to an interested
party and unsafe for gating a change.

The generalizable rule is that an approximation's direction is part of its
specification, and a system that states the direction without stating what it makes
the result unsafe *for* has stated half of it. Both systems here name the unsafe
use explicitly, and the coordination system additionally routes the unsafe use
elsewhere — to the policy document, continuous integration and a reviewer — rather
than leaving the consumer to notice.

---

## 7. Limitations and open problems

1. **The name-matching rule is regular-expression based and was written against a
   limited set of architectures.** A tensor it fails to place lands in the
   unassigned bucket, which is visible but not prominent. Its behavior on unfamiliar
   naming conventions is UNMEASURED.
2. **The nominal-versus-measured footprint reconciliation is delegated to the
   caller.** Both figures are reported and neither is preferred. Whether the
   measured footprint should simply replace the nominal figure where the runtime
   publishes byte sizes is unresolved.
3. **The dense analytic formula's coverage of unusual dense architectures is
   UNMEASURED** — shared embeddings, tied output heads, and grouped-query attention
   at unusual ratios [11] are dense but not standard, and the formula's correctness
   for them has not been established.
4. **The discovery heuristic is unvalidated on layouts not yet seen** — in
   particular a monorepo with a single top-level package directory containing
   everything, where "a pillar is a top-level directory" yields one pillar and no
   useful coordinate space.
5. **The transcript-derived identity's over-approximation is uncharacterized.** It
   is deliberate per §4.2, but how wrong it becomes for long-running sessions has
   not been measured.
6. **The observed/declared separation has not met a tool that writes its own
   worktree metadata.** Such a tool would place a self-authored record in a location
   the system treats as observed, blurring the two categories the design rests on.
   No such tool has been encountered; none is known not to exist.
7. **Contention is computed from paths, not semantics.** Two sessions editing
   different functions in one file are reported as contending — a deliberate false
   positive rather than a missed one.
8. **Staleness measures a commit, not a session.** A clean worktree that someone is
   actively reading is indistinguishable from an abandoned one. This is the
   completeness-versus-accuracy tradeoff familiar from failure detection in
   asynchronous systems [5], and it is not soluble by observing harder.
9. **A finding's body is self-reported.** Nothing checks it, and a stale finding is
   indistinguishable from a fresh one except by timestamp.

---

## 8. Related work

**Morphometry.** The construction is borrowed directly from neuroimaging: cortical
thickness estimation on a surface-based coordinate system [1], voxel-based
morphometry [2], and the practice of registering subjects to a canonical template
space so that measurements are comparable across individuals [3]. The transfer is
of method rather than of result.

**Provenance.** The separation of a value from an account of where it came from is
the subject of the data-provenance literature [7] and of provenance interchange
models [8]. The discipline here is narrower and stricter: not merely recording
provenance alongside a value, but making a value's evidence class part of its type
so that two classes cannot occupy one field.

**Coordination without trusted self-report.** The setting in which participants'
self-descriptions cannot be taken at face value is the classical Byzantine
setting [4]; this system does not solve it and does not attempt to, since it
enforces nothing. The staleness question in §7 is the failure-detection
completeness/accuracy tradeoff [5]. The advisory-claim design of §4.4 is the
familiar advisory-locking argument. Causal ordering of distributed events [6] is
adjacent but not used: this system orders nothing and only reports coincidence.

**Model structure.** Sparsely gated mixture-of-experts models [9, 10] are the
architecture class that forces §3.2's refusal, and grouped-query attention [11] is
the dense variation §7.3 names as unmeasured.

---

## 9. Conclusion

The same discipline — evidence classes separated at the type level, labeled at every
read, never merged; absences named rather than blanked; measurements placed on a
coordinate space whose provenance is explicit; approximations whose direction and
unsafe use are both stated — was applied to static model structure and to dynamic
repository occupancy, two problems with no shared subject matter. In each case the
design's value lies mostly in what it refuses to conclude: a mixture-of-experts
model refused rather than approximated by a dense formula, an unattributable
worktree named rather than assigned to its most likely owner, an unmapped path
reported rather than placed in an ambient default.

A field observation during the preparation of this series encountered both of the
coordination system's principal declared limits in ordinary use, and in each the
named absence carried information that a blank would have destroyed. That is a
single observation and is reported as one. It is nonetheless the kind of evidence
the discipline predicts: a system that refuses to guess will visibly refuse, and a
system that guesses will not visibly guess.

---

## Appendix A — Reproduction

Morphometry:

```bash
pip install "alelyon-os[lattice]"
python -c "from alelyon.runtime.vector.lattice import morphometry; print(morphometry.__doc__)"
```

The module accepts a runtime-shaped model description. A tensor inventory produces
the counted path; an architecture-only payload produces the analytic path with the
source field stating so; a mixture-of-experts architecture with no inventory
produces a named refusal. The module performs no network access, no filesystem
access, no clock reads and no user-interface work: the caller supplies the payload,
which is what makes it deterministic, trivially testable, and publishable without a
runtime dependency behind it.

Coordination:

```bash
pip install "alelyon-os[fleet]"
cd /any/git/repo
alelyon-fleet areas        # the coordinate space in force, and where it came from
alelyon-fleet status       # who is working where
alelyon-fleet open-areas   # areas no session is on
```

On a repository that declares nothing, `areas` reports discovery from that
repository's own tracked directories and lists them. No configuration is required,
and no directory outside the checkout it is pointed at is read.

**Environment.** Python 3.12. Both surfaces are offline. The coordination surface
writes to one local database file and is read-only with respect to version control.

**Data availability.** §5's field observation was recorded in a private repository
and the underlying records are not redistributable. The observation's content — four
live sessions in one checkout producing an identity refusal, and a published finding
reporting `REACHED NOBODY` — is reproducible by running two sessions in one checkout
and publishing to an unoccupied area.

---

## Appendix B — Notation table

| Term | Definition | First use |
|---|---|---|
| Observed | Fact derived from records the measured party did not author | Def. 2.1 |
| Declared | Fact the measured party states about itself | Def. 2.1 |
| Named absence | Value denoting a specific missing measurement and its reason | Def. 2.2 |
| `UNATTRIBUTED` | Named absence for session attribution | §4.1 |
| `UNMAPPED` | Named absence for coordinate placement | §4.3 |
| Derivation | Identity inferred from record structure under a stated rule | Def. 2.3 |
| Authentication | Identity established against a secret | Def. 2.3 |
| Pillar, surface | The two axes of the repository coordinate space | §4.3 |
| Block, module | The two axes of the morphometry coordinate space | §3.1 |

---

## References

The identifiers below were recorded from the authors' working bibliography and
should be checked against the primary sources before external publication; this
series has not yet had a bibliographic review.

[1] B. Fischl, A. M. Dale. *Measuring the Thickness of the Human Cerebral Cortex
from Magnetic Resonance Images.* PNAS 97(20), 2000.

[2] J. Ashburner, K. J. Friston. *Voxel-Based Morphometry — The Methods.*
NeuroImage 11(6), 2000.

[3] A. C. Evans, D. L. Collins, S. R. Mills, E. D. Brown, R. L. Kelly,
T. M. Peters. *3D Statistical Neuroanatomical Models from 305 MRI Volumes.* IEEE
Nuclear Science Symposium, 1993.

[4] L. Lamport, R. Shostak, M. Pease. *The Byzantine Generals Problem.* ACM
Transactions on Programming Languages and Systems 4(3), 1982.

[5] T. D. Chandra, S. Toueg. *Unreliable Failure Detectors for Reliable Distributed
Systems.* Journal of the ACM 43(2), 1996.

[6] L. Lamport. *Time, Clocks, and the Ordering of Events in a Distributed System.*
Communications of the ACM 21(7), 1978.

[7] P. Buneman, S. Khanna, W. C. Tan. *Why and Where: A Characterization of Data
Provenance.* ICDT 2001.

[8] W3C. *PROV-DM: The PROV Data Model.* W3C Recommendation, 2013.

[9] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, J. Dean.
*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.*
ICLR 2017. arXiv:1701.06538.

[10] W. Fedus, B. Zoph, N. Shazeer. *Switch Transformers: Scaling to Trillion
Parameter Models with Simple and Efficient Sparsity.* JMLR 23, 2022.
arXiv:2101.03961.

[11] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebrón, S. Sanghai.
*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
Checkpoints.* EMNLP 2023. arXiv:2305.13245.
