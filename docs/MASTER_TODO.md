# Master TODO — current system + dynamic conceptual track

Status: **planning record, not an implementation claim**. Last updated
2026-11-09. This is the single ordered task list. It holds the dynamic-
conceptual direction in parallel without starting implementation or freezing
an architecture.

## 0. Operating rule

Only one milestone is **active** at a time. Everything else is `held`,
`queued`, or `blocked`. A new architecture is not frozen until its gate is
passed; no new scripts or training runs are created merely to explore an idea.

| Lane | State |
|---|---|
| A. Resident system integration and verification | active |
| B. Rounds 023–036 integration | active verification; implementation items tracked in [ROUNDS_023_036_TODO.md](ROUNDS_023_036_TODO.md) |
| C. Dynamic conceptual universe | held; design/questions only |
| D. Canonical/network promotion | blocked by design |
| E. TRIGON-9 Round 042 synthesis | active experimental track; gates in §B2 |

Every write path remains scratch-first, Boolean-gated, generation-CAS checked
and canonical-merge-disabled. The hard file ceiling remains 1 GiB; the 2 GiB
overbootstrap map is never used.

---

## A. Current tasks (active)

These are the existing tasks, in execution order. Detailed acceptance criteria
for the round-specific items are in `ROUNDS_023_036_TODO.md` and
`TOTAL_INTEGRATION_PLAN.md`.

| ID | Task | Acceptance | State |
|---|---|---|---|
| A-01 | Finish the full test-suite run and record exact counts/failures | 1,459 tests, 1 skipped, OK on Termux (611 s) | done |
| A-02 | Re-run the targeted resident/field/cycle/retrieval groups after the latest agent merges | 178+ targeted tests pass; native full-cycle 7/7 | done |
| A-03 | Verify full-cycle modules use all 32,385 coefficients and reject 256-only writes | 32,385/32,385; truncation denied | done |
| A-04 | Verify full-cycle tools persist/reopen all 32,385 words | count/digest equal after reopen | done |
| A-05 | Verify native full-cycle C++ root count and gauge behavior | 32,385 count; 256 refused; gauge invariant | done |
| A-06 | Verify Round 029 raw NaN payload/lane path | all 65,536 payloads; exact gauge | done |
| A-07 | Verify Round 030 native snapshot/audit path | 8 MiB exact; artifact values | done |
| A-08 | Verify Round 031 256-effective-weight vs full-basis distinction | 256 ≠ 32,385; no overclaim | done |
| A-09 | Verify Round 032 retrieval/hyperedge persistence | 50,268 facts bounded; reopen | done |
| A-10 | Verify Round 033 episodic cycle copy | 24-byte cycle exact; raw fails | done |
| A-11 | Verify Round 034 pure copy/abstain fallback | measured 96.7–98%; no 99% claim for fallback | done |
| A-12 | Verify Round 035 two/three-source composition | linked answer; broken abstention; cycle exact | done |
| A-13 | Verify Round 036 contradiction rule and symbolic depth | `opposes` abstains; 4-hop symbolic only | done |
| A-14 | Add `Round033/034/035/036` docs to the index and MANIFEST | paths present; package build includes archives | done |
| A-15 | Re-run packaging tests after new binary/checkpoint archives | wheel 340 files / sdist 867 files; verifier pass; checkpoint/root binaries, corpus JSONL and BPE model excluded | done |
| A-16 | Re-run Alpine/proot suite | 1,459 tests, 1 skipped, OK (654 s) | done |
| A-17 | Confirm no legacy branded filenames/strings remain | repository search for the retired label is empty | done |
| A-18 | Confirm canonical merge stays disabled everywhere | all receipts/flags false | done |
| A-19 | Merge all local feature branches into `main` without losing current integration | no unmerged local branch; conflicts documented | done locally |
| A-20 | Decide whether remote PR branches are in scope; do not merge 104 stale remote refs blindly | scope decision: local branches merged; remote-tracking PR refs require individual review and were not mass-merged | decided |
| A-21 | Commit final docs/tests and merge the remaining worktree delta into main | main contains latest commits; primary worktree clean | done |
| A-22 | Publish branch/remote only after tests and scope decision | explicit operator approval | held |

### A-note on the branch merge

All local branches were merged into `integration/all-branches-20261108`, then
`main`, with current integration preserved on conflicts. Remote-tracking
feature branches were **not** mass-merged: they are historical PR refs, many
far behind and mutually inconsistent. A-20 must decide scope before touching
them.

---

## B. Rounds 023–036 integration (verification + open items)

The detailed table is in [ROUNDS_023_036_TODO.md](ROUNDS_023_036_TODO.md).
The new validated architecture inputs are [ROUND038_ROTATION_HOLONOMY.md](ROUND038_ROTATION_HOLONOMY.md),
[ROUND040_DYNAMIC_MEMORY.md](ROUND040_DYNAMIC_MEMORY.md) and
[NANLM_42_TRIGON.md](NANLM_42_TRIGON.md); their tasks are B-11..B-20 and
T-01..T-18 below.
Summary of what is still genuinely open:

| ID | Open item | Gate |
|---|---|---|
| B-01 | Neural byte-level LM over cycle readout (R033) | held-out BpB vs matched conventional model |
| B-02 | Load/replay supplied R034 checkpoints in NaN | exact 99.0% artifact replay |
| B-03 | Hybrid LM/copy generation (R034) | same retrieval and compute for control |
| B-04 | Learned neural two-hop gate (R035) | 900/900 artifact replay; source-disjoint |
| B-05 | Neural encoder–attention–decoder (R036) | held-out generation; seed stability |
| B-06 | Native R036 prompt audit in resident bridge | 90/90 cycle, 0/90 raw, conventional ties |
| B-07 | Full 32,385-cycle training on corpus | all dimensions; no 256-only fallback |
| B-08 | Continuous corpus/chat → cycle/hyperedge write path | provenance, contradiction, replay |
| B-09 | Free-form generation evaluation with no stored answers | human/teacher evaluation and controls |
| B-10 | Android ARM RSS/cache/energy measurement | measured on target device |
| B-11 | Dynamic fact write API over full-cycle slots (R040) | 32,385 capacity; one generation commit; duplicate idempotent |
| B-12 | Resident add/ask session and bounded typed BFS (R040) | answered path or explicit abstention |
| B-13 | Root + metadata snapshot pair with shared generation/digest (R040) | reopen equality; mismatch denied |
| B-14 | Interactive `add/ask/save/quit` resident CLI (R040) | scripted session test |
| B-15 | Retention after new writes and frozen-reader adaptation (R040) | old + new held-out queries |
| B-16 | SO(2) phase vs SO(3) order-sensitive transport (R038) | order probe; phase near chance, rotation/control comparison |
| B-17 | Conjugation-invariant holonomy trace audit (R038) | trace error <= 1e-6; no frame-gauge claim beyond test |
| B-18 | Four-valued support/refute/unknown/both evidence (R038) | contradiction preserved, not phase-cancelled |
| B-19 | Token-conditioned rotation/attention bridge (R038) | nonzero angle/token gradients; restricted decoder |
| B-20 | Deterministic replay of R038 checkpoints/artifacts | three-seed table and native snapshot audit |
| B-21 | Expose explicit support/refute cycle lanes (R042) | both lanes preserved; no phase cancellation |
| B-22 | Causal delete/restore of a resident pair/cycle (R042) | answer changes/restores with generation CAS |
| B-23 | Persistent add/ask session over raw cycle memory (R042) | save/reload; path or abstention |
| B-24 | Bounded propagation over typed resident edges (R042) | two-to-five-hop positive/negative split |
| B-25 | Four-state `unknown/supports/refutes/both` output (R042) | all four states preserved |
| B-26 | Optional replay of supplied writer/decoder checkpoints (R042) | pytest skip if unavailable; pure path remains |
| B-27 | Matched conventional-memory comparison in NaN (R042) | exact tie documented, no unique claim |
| B-28 | Few-shot phrase adaptation with rehearsal (R042) | old/new phrase classes retained |

These remain open until their gates pass. No item is promoted because a
related artifact reports a number.

**Implemented bounded path (2026-11-09):** B-21, B-22, B-23 and B-25 now have a
symbolic resident implementation in `resident/evidence_lanes.py`,
`resident/evidence_session.py`, the `evidence_lanes()`/`evidence_session()`
facade accessors and the `nanlm_resident_evidence` tool. The lanes preserve
`unknown/supports/refutes/both` independently and support delete/restore with
an in-process generation check. `ConventionalEvidence` is the indexed control
and ties on clean records. B-24 is covered on the symbolic path by the R035
`compose_path` gate, including a new 2–5-hop positive/broken/contradictory
split test; learned propagation, neural four-state speaking, checkpoint replay,
matched-memory compute parity and few-shot adaptation remain open. The
generation check is not a cross-process canonical CAS.


---

## B2. Round 042 TRIGON-9 synthesis (active experimental)

The synthesis design and measured results are in
[NANLM_42_TRIGON.md](NANLM_42_TRIGON.md). It is an experimental reference
architecture, not a frozen replacement for the resident system. The native
trigonometric kernel is raw NaN16; the trainable reference model is host
PyTorch. No live NANNET route or canonical merge is authorized.

| ID | Task | Acceptance | State |
|---|---|---|---|
| T-01 | Write the Round 030–041 synthesis and claim boundary | architecture document with per-round role table | done |
| T-02 | Implement the raw NaN16 triangle/holonomy/sin-cos kernel | 513 pure 0x words; gauge-invariant tests pass | done |
| T-03 | Implement the full 32,385-cycle differentiable field | 256 heads; triangle read; exact gauge audit | done |
| T-04 | Implement L0–L9 complex rotor stack and SO(3) transport | 10 layers; trig attention/state; holonomy trace | done |
| T-05 | Implement 256 sparse heads with direct token-loss path | head readout; ablation measurable | done |
| T-06 | Build the HW4 WordNet matrix mirror and relation probe | 82,192 synsets; typed splits; 45.4% test | done |
| T-07 | Implement bounded NANNET-P architecture evolution | genome bounds; validation + generation-CAS; one accepted change | done (pilot) |
| T-08 | Train a genuine byte-level generative LM | 3.60 bpb, ppl 12.13, 26.97% top-1 on internal test | done (seed 17) |
| T-09 | Matched conventional byte-LM control at equal parameters/compute | same data, tokens, optimizer budget; report cleanly | partial: data/parameter control done, compute match open |
| T-10 | Multi-seed TRIGON-9 replication | three seeds; variance and failure modes | partial: seeds 17 and 29 complete; seed 43 interrupted |
| T-11 | Put an order-sensitive composition task inside the LM | holonomy improves a language/evidence objective, not only the probe | open |
| T-12 | Improve free-generation fluency | human-readable samples; no template/copy claim | open |
| T-13 | Port copy/abstain/compose end-to-end | supported path renders; unsupported path abstains | open |
| T-14 | Continued gradient retention and forgetting test | old + new held-out queries after weight updates | open |
| T-15 | Native N16I tensor/operator ABI for the model hot path | no host substitute; parity receipt | open |
| T-16 | Android ARM RSS/cache/energy measurement | measured on target device | open |
| T-17 | Snapshot pair with shared generation/digest | reopen equality; mismatch denied | open |
| T-18 | Keep canonical merge disabled | every receipt false | done |
| T-19 | Head-usage load balancing and coverage audit | all 255 heads exercised; entropy/usage report; no rich-get-richer collapse | open |

**Current honest reading:** the LM is real but undertrained; memory/head/copy
channels are measurably used but small; trig/holonomy are neutral-to-negative
on byte LM at this budget; the order probe shows the trig algebra is
order-sensitive (100% vs 33.3% phase-only, three seeds); the seed-17 router
collapsed to 4 of 255 heads. The next falsifiable tasks are T-11 (in-LM order
task) and T-19 (head coverage), not another headline number.

---

## C. Dynamic conceptual universe (held design track)

The direction to evaluate later:

> Do not treat the universe as a store for a language model. Treat it as a
> trainable dynamical system whose evolution performs conceptual computation;
> language is a readout of that evolving state.

This is conceptual only. No architecture is frozen, no code is written for it
yet, and it does not displace the current verification track.

### C.1 Five retained ideas

1. **Concepts as transformations.** A concept is partly characterized by how
   it changes neighboring conceptual state, not only by a vector or label.
2. **Relationships as the learning substrate.** Pair cells carry directional
   influence/retention/operator state; heads are sparse activation coordinates.
3. **Cycles as compositional constraints.** A cycle can measure disagreement
   between direct and composed routes; not every cycle should be forced to
   zero because exceptions and legitimate curvature carry information.
4. **Generation as evolving state.** Internal state may take several
   transitions before a token is emitted; token production is a readout.
5. **Large cycle reservoir, sparse head interface.** 32,385 cycle degrees of
   freedom can persist while only a small active set is read/written per token.

### C.2 Ordered conceptual tasks

| ID | Task | Output / gate | State |
|---|---|---|---|
| C-01 | Write a falsifiable position paper: dynamical computation vs. current store+decoder | no code; explicit alternatives and controls | queued |
| C-02 | Define the state vector: which pair/cycle fields are active, retained and read | bounded schema; no architecture freeze | queued |
| C-03 | Define concept-as-operator: parameterization, inverse/identity, composition | small mathematical prototype if approved | queued |
| C-04 | Define a cycle-composition residual for supported paths, allowing nonzero exception cycles | objective and test cases | queued |
| C-05 | Decide whether composition is scalar additive or learned directional transformations | explicit decision with falsification test | queued |
| C-06 | Define multi-timescale dynamics: internal ticks vs token ticks | bounded tick budget, stability test | queued |
| C-07 | Define candidate-state retention/collapse rule | no premature argmax; distribution over interpretations | queued |
| C-08 | Define the language readout from internal state | held-out token prediction protocol | queued |
| C-09 | Define combined loss: language + composition + dynamics | weights tuned only on validation | queued |
| C-10 | Build matched controls: conventional RNN, graph memory, sharpened/ablated cycle | same data/parameters/compute | queued |
| C-11 | Define gauge-invariant cycle features separately from raw pair values | exact invariance audit | queued |
| C-12 | Decide whether the 32,385 cycles are a reservoir, a trainable field, or both | explicit capacity/optimization budget | queued |
| C-13 | Test oversmoothing/oscillation vs internal-tick count | report failures, not only best tick | queued |
| C-14 | Test continual learning and catastrophic forgetting | retention after new corpus/chat | queued |
| C-15 | Test compositional generalization on unseen two/three-hop evidence | child/source-disjoint controls | queued |
| C-16 | Define cross-universe exchange of gauge-invariant relational structures | later; raw weights never exchanged | held |
| C-17 | Keep canonical aggregation blocked until one universe thinks in its own geometry | no multi-universe merge | held |

### C.3 What we explicitly will not do yet

- No new scripts or training runs for this track before C-01/C-02 are accepted.
- No claim that scalar cycles equal a quantum state or a physical superposition.
- No claim that the NaN encoding is uniquely superior; a conventional graph can
  implement the same algebra.
- No immediate aggregation of multiple universes or raw-weight exchange.
- No forcing all cycles to zero; that would erase exception/curvature signals.
- No replacement of the existing temporal/field learners before matched
  controls demonstrate an advantage.
- No canonical merge, promotion, or “self-modification escape.”

### C.4 Strongest falsification tests

1. A matched conventional RNN with the same token budget must not lose to the
   dynamic cycle field on language prediction.
2. A conventional graph with identical gauge-invariant features must not lose
   on composition/retrieval.
3. Cycle-ablated and sharpened variants must be compared, not just the full
   field.
4. A composition residual that permits meaningful curvature must beat both
   “zero all cycles” and “no cycle constraint” on held-out composition.
5. Many internal ticks must be shown to help only where measured; report
   oversmoothing and oscillation.
6. Long-horizon memory and continual learning must be tested, not inferred from
   one episode.
7. Every reported gain must be reproducible on a source-disjoint split and on
   Android ARM.

### C.5 Decision after current verification

After A-01..A-21 and the measurable B gates:

- If the current resident/retrieval/composition system passes its gates, keep
  it as the baseline and evaluate the C track against it.
- If the C track fails its controls, record the failure and do not integrate it.
- If it passes, freeze only the **tested** architecture slice, not the whole
  speculative design.

---

## D. Canonical and network boundary (blocked)

| ID | Task | State |
|---|---|---|
| D-01 | Real NANNET HW2 operator deployment, issuer pin and credentials | blocked: operator work |
| D-02 | Generation-CAS candidate intake with six-runner barrier | blocked: not implemented |
| D-03 | Canonical BS0/MC0 promotion | blocked by design; `merge_enabled=false` |
| D-04 | Cross-universe relational exchange | C-16 only after single-universe gates |

No resident result above authorizes D-01..D-04.

---

## E. Verification commands currently relevant

```bash
# Focused resident/cycle/retrieval/composition groups
python3 -m unittest \
  tests.test_resident_universe tests.test_resident_learning \
  tests.test_full_cycle_memory tests.test_full_cycle_training \
  tests.test_full_cycle_tools tests.test_native_full_cycle \
  tests.test_resident_cycle_observable tests.test_resident_cycle_training \
  tests.test_resident_cycle_persistence tests.test_resident_cycle_snapshot \
  tests.test_resident_mirror_retrieval tests.test_resident_retrieval_hypergraph \
  tests.test_episodic_gate tests.test_composition_gate \
  tests.test_round032_total_integration -q

# Artifact audits / reproductions
python3 scripts/round034_episodic_audit.py
python3 scripts/round035_composition_audit.py
python3 scripts/round036_robustness_audit.py
```

The full suite remains the A-01 gate. No C-track command is to be added until
C-01 and C-02 are accepted.

## F. Change-control checklist

Before changing this list:

- [ ] Is this a measured result, an artifact, a plan, or a hypothesis?
- [ ] Which existing task does it modify or invalidate?
- [ ] Does it require a new architecture freeze? If yes, stop.
- [ ] Does it preserve Boolean foundation, scratch-first writes, generation CAS,
      canonical merge off and the 1 GiB ceiling?
- [ ] Is there a matched conventional control?
- [ ] Is the failure mode reported as clearly as the success?
