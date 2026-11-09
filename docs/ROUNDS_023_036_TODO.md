# Rounds 023–036: total integration TODO

Status: **deep audit of the supplied research archives and the NaNLMind_HERMES
implementation**, 2026-11-08. `done` means code exists and targeted tests pass.
`partial` means the mathematical/execution core is implemented but an external
neural prototype or training corpus is still not end-to-end. `open` means it is
not yet implemented and must not be claimed as working.

The canonical merge gate remains **off**. No item below authorizes BS0/MC0
promotion, a production NANNET cutover, or a 2 GiB overbootstrap copy.

## A. Round-by-round audit

| Round | Supplied result | NaN implementation | Status |
|---|---|---|---|
| 023 | 256 heads, 32,640 mirror pairs, gauge/Hodge/cycle prototype | `resident/cells.py`, `resident/cycle_observable.py`, `nan_geometry.py` | done |
| 024 | growable recursive hyperedge/concept arena | `resident/relations.py`, `resident/cycle_hypergraph.py`, `resident/retrieval_hypergraph.py` | done |
| 025 | WordNet/corpus mapping | corpus trainable files, `scripts/corpus_training.py`, `resident/learning.py` | done for the supplied corpus |
| 026 | resident native next-token prototype | `scripts/resident_learning.py`, `resident/full_cycle_training.py` | partial: native/neural external |
| 027 | semantic relation learning | typed relation/leaf/hyperedge stores | done for structured records; no arbitrary prose parser |
| 028 | shared composition / native hyperedge strengths | `composition_gate.py`, retrieval/hypergraph stores | partial: native strength training is not the same path |
| 029 | raw NaN payload geometry, exact gauge cancellation | `nan_geometry.py`, `resident/nan_lane.py`, `resident/cycle_observable.py` | done |
| 030 | native 8 MiB root training, snapshot, gauge audit | `resident/cycle_native.py`, `resident/cycle_snapshot.py`, `scripts/round030_native_audit.py` | done as optional native bridge |
| 031 | native root in autoregressive LM, 256 effective weights | `full_cycle_memory.py`, `full_cycle_training.py`, `experiments/round031/native_full_cycle.cpp` | full 32,385 basis done; neural LM remains external/open |
| 032 | sparse mirror postings, typed hyperedges, two-hop lookup | `mirror_retrieval.py`, `retrieval_hypergraph.py`, `scripts/round032_retrieval_audit.py` | done for supplied facts |
| 033 | episodic copy generation, native 8 MiB root | `episodic_gate.py`, `EpisodicCycleMemory`, archive audit | copy path done; byte LM remains external |
| 034 | learned copy/abstain gate, hybrid byte mixture | `episodic_gate.py`, pure gate + cycle copy, `scripts/round034_episodic_audit.py` | partial: pure fallback; supplied neural checkpoint not reimplemented |
| 035 | two-/three-source composition, contradiction-free chains | `composition_gate.py`, `scripts/round035_composition_audit.py` | done symbolically; learned neural composer external |
| 036 | encoder–attention–decoder generation, adversarial contradiction | symbolic composition only; `scripts/round036_robustness_audit.py` | partial: neural seq2seq not reimplemented |

### Honest numbers

| Artifact | Value | Class |
|---|---:|---|
| Full cycle basis | 32,385 = C(255,2) | exact |
| Mirror pairs | 32,640 | exact |
| Native root | 8,388,608 B | exact |
| R031 native 16×16 effective weights | 256 | artifact limitation |
| R034 learned neural router | 99.0% accept/abstain | supplied artifact |
| R034 pure NaN fallback | 96.7–98.0% accept/abstain | measured fallback |
| R035 two-hop learned gate | 900/900 | supplied artifact |
| R035 three-hop | 120/120 | supplied artifact |
| R035 resident symbolic gate | 100% on synthetic | measured fallback |
| R036 robust contradiction | 1499/1499 each of 3 kinds | supplied artifact |
| R036 neural seed 17 | 50.0% (constant abstention) | supplied failure |
| R036 native cycle prompts | 90/90 after gauge | supplied artifact |
| R036 raw edge prompts | 0/90 after gauge | supplied artifact |

`R034 pure fallback` is deliberately lower than the supplied neural checkpoint;
do not merge the two numbers into one claim.

## B. Point-by-point integration TODO

| ID | Work item | Files | Acceptance | Failure mode | Status |
|---|---|---|---|---|---|
| R023-01 | Keep 256 head identities immutable | `resident/cells.py` | exhaustive identity tests | deny write | done |
| R023-02 | Keep 32,640 physical mirror pairs and reverse16 mirror | `cells.py`, `cycle_observable.py` | exhaustive mapping | fail closed | done |
| R023-03 | Raw-word cycle read/write | `cycle_observable.py`, `nan_lane.py` | gauge cancels exactly | no float identity | done |
| R023-04 | Hodge/cycle metrics on live rails | `cycle_observable.py` | gauge residual | read-only query | done |
| R024-01 | Typed relation records | `relations.py` | append/read/idempotency | capacity denial | done |
| R024-02 | Hyperedge membership and recursive traversal | `relations.py`, `cycle_hypergraph.py` | bounded two-hop | limit denial | done |
| R024-03 | Concept/leaf identity for every fact | `leaves.py`, `retrieval_hypergraph.py` | content-addressed | collision denial | done |
| R024-04 | Persist recursive graph | universe leaves/relations | reopen equality | no merge | done |
| R025-01 | Deterministic corpus split | `scripts/corpus_training.py` | held-out hash split | no leakage | done |
| R025-02 | Per-source evaluation | `corpus_training.py`, `resident_learning.py` | exact `hits/total` | no ambiguous decimals | done |
| R025-03 | Typed definition/relation extraction | `resident/learning.py` | 2,522 extracted relations | no fact-verified claim | done |
| R026-01 | Resident two-channel learning | `resident/learning.py`, `scripts/resident_learning.py` | NLL falls | no raw-repeat gain | done |
| R026-02 | Best checkpoint/early stop | `corpus_training.py`, deep trainer | epoch 1 selected | later epochs not saved | done |
| R027-01 | Tree edge from head root to leaves | universe/leaves | parent/child audit | no orphan claim | done |
| R027-02 | Relation evidence leaves | universe | subject/predicate/object links | duplicate denial | done |
| R028-01 | Shared composition gate | `composition_gate.py` | learned gate >0.9 synthetic | no free-form claim | done |
| R029-01 | Raw binary64 NaN payload codec | `nan_geometry.py` | all 65,536 payloads | non-NaN denial | done |
| R029-02 | Exact integer gauge store | `nan_geometry.py` | all 32,385 triangles | no float identity | done |
| R029-03 | Native/Python comparison | `scripts/round029_native_compare.py` | 8/8 invariants | host-only timing | done |
| R030-01 | Native cycle bridge | `resident/cycle_native.py` | compile/predict/audit | unavailable skip | done |
| R030-02 | Snapshot NANCYC30 | `cycle_snapshot.py` | exact 8 MiB restore | size/marker denial | done |
| R030-03 | Native gradient audit | `scripts/round030_native_audit.py` | artifact values match | no fake result | done |
| R031-01 | Full 32,385 coefficient storage | `full_cycle_memory.py` | 32,385/32,385 | 256-only denied | done |
| R031-02 | Full-basis trainer | `full_cycle_training.py` | loss decreases on all dims | truncation denied | done |
| R031-03 | Full native C++ 32,385 root | `experiments/round031/native_full_cycle.cpp` | count 32,385; mirror/gauge | 256 mode refused | done |
| R031-04 | Neural autoregressive LM over cycle readout | external PyTorch prototype | held-out BpB matches/improves conventional | not done | open |
| R031-05 | Joint encoder/cycle training | external | gradients through read | not done | open |
| R032-01 | Mirror postings | `mirror_retrieval.py` | 50,268 facts | capacity bound | done |
| R032-02 | Hash comparator | `mirror_retrieval.py` | same members | no speed claim | done |
| R032-03 | Two-hop typed path | `mirror_retrieval.py`, retrieval hypergraph | 2,000/2,000 sample | bounded expansion | done |
| R032-04 | Persist facts/hyperedges | `retrieval_hypergraph.py` | reopen equality | no merge | done |
| R033-01 | Episodic copy memory | `episodic_gate.py` | exact 24-byte cycle copy | raw edge fails | done |
| R033-02 | Learned key encoder | supplied PyTorch checkpoint | 118/120 artifact | external checkpoint | partial |
| R033-03 | Byte-level autoregressive LM | external | BpB measured | not reimplemented | open |
| R034-01 | Copy/abstain gate | `episodic_gate.py` | pure fallback 96.7–98% | no support claim | partial |
| R034-02 | Hybrid LM/copy | external checkpoint | artifact BpB | not reimplemented | open |
| R034-03 | Stress corruption levels | `episodic_gate.py` | reports failures | no overclaim | done |
| R035-01 | Two-source composition | `composition_gate.py` | positive/abstain | broken link denial | done |
| R035-02 | Three-source composition | `composition_gate.py` | 3-hop linked/broken | no answer stored | done |
| R035-03 | Gauge-resistant source read | `CompositionMemory` | cycle exact, raw fails | no unique claim | done |
| R035-04 | Learned neural two-hop gate | supplied PyTorch artifact | 900/900 | external | partial |
| R036-01 | Contradictory relation denial | `composition_gate.py` | `opposes` abstains | no false answer | done |
| R036-02 | Arbitrary positive-depth symbolic chain | `compose_path` | 4-hop synthetic | no neural 4-hop claim | done |
| R036-03 | Neural encoder–attention–decoder | external PyTorch | 38,268 params, held-out gen | not reimplemented | open |
| R036-04 | Adversarial retraining | external checkpoint | 3×1499/1499 | artifact only | partial |
| R036-05 | Native prompt recovery audit | `episodic_gate`/native bridge | 90/90 cycle, 0/90 raw | conventional ties | partial |
| X-01 | Keep canonical merge off | plugin/config/modules | every receipt false | fail closed | done |
| X-02 | Keep 2 GiB map unused | universe | max 1 GiB | reject >1 GiB | done |
| X-03 | Remove stale branding | repo | `grep -ri hyperllm` empty | — | done |
| X-04 | Full no-Torch test path | tests | pure modules pass without torch | skip optional natives | done |
| X-05 | Optional Torch artifact path | scripts | exact checkpoint replay | unavailable skip | open |

## C. Remaining milestones

### M1 — finish resident total facade

- [ ] One `ResidentTotal` entry point exposing `universe`, `cycle`, `full_cycle`,
      `retrieval`, `episodic`, `composition`, `field`, `scratch`.
- [ ] Status merges counts, generation, digest, head entropy, cycle occupancy,
      facts, hyperedges and active scratch layers.
- [ ] All writes route through the scratch/Boolean/CAS gate.

### M2 — finish neural integration

- [ ] Load the supplied R033/R034/R036 checkpoints without making PyTorch a
      required runtime for the pure path.
- [ ] Reproduce R034 99.0% and R036 100% accuracy from the checkpoints.
- [ ] Feed native cycle bytes/readouts into an autoregressive decoder.
- [ ] Report held-out BpB against a conventional model with identical inputs.

### M3 — full 32,385-cycle training

- [ ] Train all 32,385 coefficients on a corpus, not only the 256-to-16×16
      demonstration.
- [ ] Prove no truncation and no 256-only fallback in status.
- [ ] Compare against a conventional 32,385-feature model with identical data.

### M4 — compositional generation with evidence

- [ ] Learn relation extraction from prose into the typed hypergraph.
- [ ] Let the decoder alternate between its own token distribution, copied
      spans and composed facts.
- [ ] Evaluate contradictions, absent evidence, four-hop and held-out books.

### M5 — production gates

- [ ] Real NANNET HW2 operator deployment and issuer pin.
- [ ] Generation-CAS candidate intake plus six-runner barrier (future work).
- [ ] Android ARM cache/RSS/energy benchmark.
- [ ] Full Termux and Alpine test suites; package verification.

## D. Reproduce the current evidence

```bash
python3 -m unittest \
  tests.test_resident_universe tests.test_resident_learning \
  tests.test_full_cycle_memory tests.test_full_cycle_training \
  tests.test_full_cycle_tools tests.test_native_full_cycle \
  tests.test_resident_cycle_observable tests.test_resident_cycle_training \
  tests.test_resident_cycle_persistence tests.test_resident_cycle_snapshot \
  tests.test_resident_mirror_retrieval tests.test_resident_retrieval_hypergraph \
  tests.test_episodic_gate tests.test_composition_gate \
  tests.test_round029_total_integration tests.test_round030_integration \
  tests.test_round031_total_integration tests.test_round032_total_integration \
  -q
python3 scripts/round034_episodic_audit.py
python3 scripts/round035_composition_audit.py
python3 scripts/round036_robustness_audit.py
```

## E. Denial rules

1. Never call a memory lookup “reasoning” unless a composition/abstention test
   proves it.
2. Never call a learned gate “understanding” without held-out, source-disjoint
   evidence.
3. Never attribute a copying gain to NaN geometry when a conventional exact
   index ties.
4. Never enable canonical merge/promotion from any prototype.
5. Never copy the 2 GiB overbootstrap map to the phone resident file.
6. Never hide a failed stress level; report corruption failures.
7. Never merge the 99% neural artifact and the 96.7–98% pure fallback into one
   headline.
