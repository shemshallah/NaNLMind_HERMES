# Quantum-math integration matrix

Snapshot: **2026-11-08**, commit `a14e135`, dirty working tree. This matrix maps
every quantum-inspired, gauge, spectral, cycle and mathematical subsystem to:
where it is actually used, what it produces, which invariants it claims, what
is missing, which tests/scripts back it, and which production TODO item owns
the gap.

It is deliberately conservative. “Quantum” in these modules means classical
linear algebra, classical signal processing, or a quantum-inspired control
calculation. No module in this repository demonstrates physical quantum
hardware, a coherent semantic oracle, a quantum speedup, or a canonical
NaNLMind commit. Where a document claims an integration that the code does not
have, the row says **documented-not-wired**.

## Status legend

- **code-backed** — implementation and at least one test exist; the row is
  still not a production integration claim.
- **documented-not-wired** — the module and tests exist, but no runtime caller
  uses it outside benchmarks/tests.
- **control/null** — the module is a control, a negative result, or an exact
  mathematical audit, not a production semantic engine.
- **documented-only** — the claim exists in docs, but no code path backs it.

**Evidence note.** This document reports static source inspection plus one
targeted run of 22 tests (`test_gauge_memory`, `test_cycle_state`,
`test_phase_memory`, `test_boolean_wave`, `test_temporal_spine`) on 2026-11-08;
all passed. Other test results below are “test file exists / test names
observed”, not a fresh full-suite result. No benchmark was re-run.

## Row index

| Row | Subsystem | Main module | Runtime caller today |
|---|---|---|---|
| QM-01 | Gauge memory | `gauge_memory.py` | none (cycle/semantic helpers, scripts) |
| QM-02 | Cycle space | `cycle_space.py` | none (scripts/tests) |
| QM-03 | Cycle state | `cycle_state.py` | none (tests) |
| QM-04 | Cycle codec | `cycle_codec.py` | none (scripts/tests) |
| QM-05 | Phase memory | `phase_memory.py` | none (scripts/tests) |
| QM-06 | Spectral wave operator | `spectral.py` | none (scripts/tests) |
| QM-07 | Spectral word write | `word_writer.py` | none (scripts/tests) |
| QM-08 | Pair-space quantum walk | `pair_quantum.py` | none (scripts/tests) |
| QM-09 | Robust recovery | `robust_recovery.py` | none (tests) |
| QM-10 | Joint semantic decoder | `semantic_state.py` + script | none (offline script) |
| QM-11 | Semantic acquisition | `semantic_acquisition.py` | none (tests/scripts) |
| QM-12 | Semantic state features | `semantic_state.py` | none (offline script) |
| QM-13 | Temporal spine | `temporal_spine.py` | none (tests) |
| QM-14 | Concept matrix | `concept_matrix.py` | layered learning |
| QM-15 | Boolean wave | `boolean_wave.py` | none (tests/scripts) |
| QM-16 | Head population | `head_population.py` | none (scripts/tests) |
| QM-17 | MTH L0–L9 / root Boolean table | `mth_layers.py` | chat, temporal LM, layered learning |
| QM-18 | MTH 91 kernels / N16I | `mth_catalog.py`, `nan0s_runtime.py` | tool handlers |
| QM-19 | MTH data leaves/hubs | `mth_leaves.py` | tool handlers |
| QM-20 | WordNet HW4 typed graph | `wordnet_store.py`, `wordnet_typed.py`, `dictionary_universe.py` | layered learning/tool |
| QM-21 | Free-chat layered learning | `chat.py`, `layered_learning.py` | chat/tools |
| QM-22 | CLOCKED / head cortex | `clocked_tools.py`, `head_cortex.py` | tools |
| QM-23 | Temporal mirror LM / local runner | `temporal_lm.py`, `universe_chart.py` | native chat |
| QM-24 | Definition/lexical/corpus memory | `definition_decoder.py`, `lexical_memory.py`, `corpus_memory.py` | none (scripts/tests) |
| QM-25 | Cross-resolution memory feedback | `memory.py`, `cross_resolution.py` | memory tools |

---

## QM-01 — Gauge memory

**Module/doc.** `nanlmind_hermes/gauge_memory.py`;
`docs/GAUGE_MEMORY.md`.

**Mathematical role.** Retrieval over divergence-free edge fields on K256 with
an unknown vertex gauge. For each candidate, solve a ridge-regularized
least-squares vertex potential `u` minimizing
`Σ_observed (d_e − (u_j − u_i))²`, then score the residual. The naive control
compares raw edge differences. This is the “strong baseline” from the external
cycle-memory experiments.

**Current integration point.** Imported by `cycle_state.py` and
`semantic_state.py`; also used by `scripts/wordnet_semantic_cycle.py`.
No chat, tool, memory ledger or CLOCKED path calls `retrieve()` or
`gauge_residual()`.

**Inputs/outputs.** `n: int`, `patterns: list[dict[(i,j), float]]`,
`query: dict`, `observed: list[(i,j)]` → prototype index or scalar residual.
No persistence, no source provenance, no semantic labels.

**Invariants/constraints.** Stored patterns are assumed divergence-free; the
gauge fit removes only a vertex potential; the ridge term is `1e-7`; the
method creates no new information. The code is real-valued, not complex.

**Missing wiring.** No L7/L8 relation store, no WordNet edge ingestion, no
concept-matrix edge source, no confidence/evidence model, no tool surface, no
audit record. The document says “L7/L8 relational memory” and “where it fits”;
that is architecture, not an implemented readout.

**Tests/evidence.** `tests/test_gauge_memory.py` has three tests: cycle pattern
has zero divergence; a pure gauge is removed by the fit but inflates naive
distance; gauge retrieval beats naive under erasure. The 22-test targeted run
passed. No end-to-end semantic test.

**Linked production tasks.** `P1-015` (preserve relation semantics),
`P1-016` (evidence-aware promotion), `P2-018` (claim ledger). `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired.** Code-backed as a classical
retrieval baseline; not used by the product. External numbers are not
reproduced here.

---

## QM-02 — Cycle space

**Module/doc.** `nanlmind_hermes/cycle_space.py`; `docs/CYCLE_SPACE.md`,
`docs/ROUND012_015_MTH.md`.

**Mathematical role.** Exact Hodge-style split of the complete K256 pair graph:
32,640 edges, gradient dimension 255, cycle dimension 32,385, 2,763,520
triangles. Functions: `decompose`, `star_decompose`, `reconstruct`,
`fundamental_fluxes`, `triangle_flux`, `triple_coherence`,
`cycle_inner_product`.

**Current integration point.** `scripts/cycle_resonance.py` (NumPy audit) and
tests. No runtime module imports it. `gauge_memory` has an independent
`project_cycle`; the two are not shared.

**Inputs/outputs.** `n: int`, edge-value dict (real or complex) → potential,
gradient dict, cycle dict, fundamental flux dict, coherence scalar.

**Invariants/constraints.** `dim = C(n,2) − (n−1)`; gradient/cycle
orthogonality; exact reconstruction; triangle flux is gauge-invariant;
`triple_coherence` is phase-sensitive and magnitude-blind. Values are
classical; no quantum state.

**Missing wiring.** No persistent cycle store, no typed edge ingestion from
WordNet, no cycle-aware token/answer decoder, no confidence propagation, no
runtime address for a cycle object. The “shared substrate” claim is
mathematical, not a code path.

**Tests/evidence.** `tests/test_cycle_space.py`: dimensions, split/
reconstruction, phase-sensitive triple coherence. `scripts/cycle_resonance.py`
produced the numbers in `docs/CYCLE_SPACE.md`; those were not re-run.

**Linked production tasks.** `P1-019` (coverage map), `P2-018` (claim ledger),
`P2-020` (reproducible benchmarks). `P1-041` (wire/retire registry).

**Claim status.** **code-backed / documented-not-wired.** Exact classical
linear algebra; no semantic or quantum advantage.

---

## QM-03 — Cycle state

**Module/doc.** `nanlmind_hermes/cycle_state.py`; `docs/PHASE_MEMORY.md`,
`docs/ROUND016_MTH.md`.

**Mathematical role.** Persistent complex edge state with a decay factor.
`observe()` blends observations then projects onto the divergence-free cycle
subspace. `apply_diagonal()` applies `P_cycle D P_cycle`; `with_gauge()` returns
the same cycle content in another vertex gauge; `snapshot()/restore()` persists.

**Current integration point.** Tests only. `tests/test_cycle_state.py` imports
it; no chat, memory, WordNet or tool module does.

**Inputs/outputs.** `nodes`, `decay`; observation edge dict; snapshot dict.
Output is an in-process object; no file format is defined outside the test
snapshot helper.

**Invariants/constraints.** Gauge shifts do not change cycle content; decay is
scalar; projection removes only the vertex-gradient component; edge order is
normalized. `apply_diagonal` is classical, not a Hamiltonian simulation.

**Missing wiring.** No durable state file or generation, no source/evidence
binding, no tool to query/update, no integration with `gauge_memory.retrieve`,
no phase/cycle decoder. The docs call it “the memory layer”; that is a design
statement.

**Tests/evidence.** Five tests: observation projects out gradient; gauge
invariance; snapshot/restore; projected diagonal changes cycle state without
gradient; edge normalization. The targeted run passed.

**Linked production tasks.** `P1-015`, `P1-016`, `P2-018`. `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired.** Code-backed prototype; no product
memory path.

---

## QM-04 — Cycle codec

**Module/doc.** `nanlmind_hermes/cycle_codec.py`; `docs/CYCLE_CODEC.md`,
`docs/ROUND012_015_MTH.md`.

**Mathematical role.** Encode a low-dimensional vector `z` as `Q z`, where `Q`
is an orthonormal divergence-free cycle basis on K_n; add a vertex gauge and
edge outliers; decode with gauge least squares or joint Huber IRLS. Also
provides a capacity-matched random linear code control.

**Current integration point.** `scripts/cycle_codec_bench.py` and tests.
No runtime caller; no learned codebook; no WordNet glue.

**Inputs/outputs.** `n`, `columns`, `observed` edge vector, `mask`, optional
weights → decoded `z` and gauge potential. Synthetic only.

**Invariants/constraints.** `Q` columns are orthonormal and divergence-free;
the joint decoder alternates gauge/cycle fits; the random-code control has the
same capacity; the code does not add semantic information. At high
erasure+spikes even the joint decoder degrades.

**Missing wiring.** No learned encoder/decoder, no real definition/answer
dataset in the runtime, no `CycleState` integration, no calibration, no
multi-seed confidence interval, no tool surface.

**Tests/evidence.** `tests/test_cycle_codec.py`: zero divergence, clean
recovery, high-erasure recovery, joint vs non-robust under spikes, random-code
control, finite robust path. Docs report synthetic tables; not re-run.

**Linked production tasks.** `P1-019`, `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **control/null.** The docs explicitly say the ordinary
random code can match the cycle code; no cycle-specific semantic advantage.

---

## QM-05 — Phase memory

**Module/doc.** `nanlmind_hermes/phase_memory.py`; `docs/PHASE_MEMORY.md`,
`docs/ROUND010_011.md`.

**Mathematical role.** Unit phasors per channel. Gauge score
`|Σ conj(q_j) p_j|²` cancels a global phase; naive score
`Re Σ conj(q_j) p_j` does not. Real cosine/sine channels reproduce the same
score. Random phase noise eventually destroys recall.

**Current integration point.** `scripts/phase_memory_bench.py` and tests.
No runtime semantic memory, WordNet classifier or chat path.

**Inputs/outputs.** `dimensions`, pattern list, query phasors, seed,
noise std → scores/accuracy. No persistence.

**Invariants/constraints.** Global phase invariance; real-channel equivalence;
phase noise is the limit; no quantum speedup. `add_noise` normalizes back to
unit phasors.

**Missing wiring.** No persistent phase store, no relation/edge binding, no
integration with `cycle_state` or `gauge_memory`, no production retrieval
tool, no calibration. The document says “memory mechanism”; that is the
intended role, not an implemented product path.

**Tests/evidence.** Five tests: global-phase cancel, real/complex agreement,
gauge beats naive, noise breaks recall, dimension guard. The targeted run
passed. `docs/PHASE_MEMORY.md` reports a 64-pattern sweep; not re-run.

**Linked production tasks.** `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **code-backed / documented-not-wired.** Classical signal
processing; no quantum claim.

---

## QM-06 — Spectral wave operator

**Module/doc.** `nanlmind_hermes/spectral.py`;
`docs/SPECTRAL_ARCHITECTURE.md`, `docs/SPECTRAL_WAVE.md`,
`docs/CONCEPT_SPECTRAL.md`.

**Mathematical role.** Head-to-head Gram matrix `A`; symmetrize and
centrosymmetrize with the reversal `J`; fold into 128×128 `P = B + CJ` and
`M = B − CJ`; graph Laplacian `L = D − A`; inverse-square void potential
`V(r)=λ/r²`; low modes; Fiedler cut; native `reverse8` mirror and complement
`255−i` are kept distinct.

**Current integration point.** `scripts/spectral_experiment.py`,
`scripts/spectral_word_write.py`, `scripts/concept_spectral.py`, tests. No chat,
layered learning, MTH runtime or tool imports `spectral.py`.

**Inputs/outputs.** 256×256 matrix, `k`, iterations → eigenvalues, vectors,
regions, boundary. `fiedler_partition` returns positive/negative/inner/
boundary sets. No persistent object.

**Invariants/constraints.** Centrosymmetry/fold agreement; `J`-invariant inner
set; `L` nonnegative; void potential on the Laplacian, not a positive diagonal
on the coupling; head 0 is the void/NaNNet anchor; native mirror `reverse8` is
not complement `255−i`.

**Missing wiring.** No runtime matrix built from the concept matrix or typed
relations; no low-mode L7 scorer; no head-diversification feedback; no tool.
The docs’ measured status is from offline scripts; the current head population
is rank-limited.

**Tests/evidence.** `tests/test_spectral.py`: power iteration, folded vs full
spectrum, void regions, native vs complement mirror, Fiedler two blocks,
normalized Laplacian, symmetric Gram. Docs report fold error 0.0 and rank
limitations; not re-run.

**Linked production tasks.** `P1-019`, `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired.** The spectral math is code-backed;
the “L7/L8 basis” and “Hermes demo” claims are not wired to runtime.

---

## QM-07 — Spectral word write / WordLeafWriter

**Module/doc.** `nanlmind_hermes/word_writer.py`; `docs/BOOLEAN_WAVE.md`,
`docs/SPECTRAL_WORD.md`.

**Mathematical role.** Letter packets `u` supported outside the void; word
update `ΔA = S + D` with symmetric association `β(u_i u_jᵀ + u_j u_iᵀ)` and
directed order `γ(u_i u_jᵀ − u_j u_iᵀ)`. Then project out inner/void and restore
centrosymmetry. Pair rank/payload packing:
`k(i,j)=i(511−i)/2+j−i−1`, 15-bit rank + 8-bit amplitude + 9-bit phase.

**Current integration point.** `scripts/spectral_word_write.py` and
`tests/test_boolean_wave.py`. No `temporal_lm`, `universe_chart`, chat or MTH
path writes a word leaf into the spectral matrix.

**Inputs/outputs.** `n`, inner set, permutation, letter packets, `β`, `γ` →
updated matrix + word record (couplings, residuals, `void_zero`).

**Invariants/constraints.** Centrosymmetry `J A J = A`; inner set `J`-invariant;
no void support; directed part antisymmetric; rank fits 15 bits; payload is
32-bit NaN16, not IEEE 51-bit.

**Missing wiring.** No temporal-spine handoff, no spectral matrix persisted in
the runtime, no inference from written word couplings, no concept-matrix
source, no tool. `docs/SPECTRAL_WORD.md` says the next step is temporal spine;
that step is missing.

**Tests/evidence.** `tests/test_boolean_wave.py`: exact 16 Boolean functions,
fuzzy non-idempotence, letter packet void rejection, word update preserves
centrosymmetry/void, pair-rank payload round-trip. The targeted run passed.

**Linked production tasks.** `P1-015`, `P2-005`, `P2-018`. `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired.** Code-backed atomic write; not in
the language path.

---

## QM-08 — Pair-space quantum walk / Johnson graph

**Module/doc.** `nanlmind_hermes/pair_quantum.py`; `docs/PAIR_QUANTUM.md`,
`docs/ROUND016_MTH.md`.

**Mathematical role.** Exact spectrum of J(256,2): eigenvalues 508 (×1), 252
(×255), −2 (×32,384); degree 508; revival period π; ideal Grover query count
141 for 32,640 states; classical expected queries 16,320.5. `cycle_space`
supplies the 32,385-dimensional cycle sector.

**Current integration point.** `scripts/pair_quantum_bench.py` and tests. No
runtime scheduler, memory or relation path.

**Inputs/outputs.** `n`, time, valid states → eigenvalues/multiplicities,
return probability, ideal query count/probability. Exact arithmetic.

**Invariants/constraints.** Johnson adjacency definition; spectrum exact;
revival only for the uniform walk; cycle sector stationary under `BᵀB` but not
general noise; Grover assumes a coherent oracle/state preparation that does not
exist.

**Missing wiring.** No semantic oracle, no state preparation, no measurement
model, no error correction, no resource accounting, no runtime use. The
document explicitly disclaims a physical or NaN-specific advantage.

**Tests/evidence.** `tests/test_pair_quantum.py`: K256 spectrum/degree,
revival at π, small J(n,2) spectrum, Grover optimum, small-case bound. Not
re-run in the targeted 22 (it was not in that subset).

**Linked production tasks.** `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **control/null.** Exact classical math and an ideal control;
no quantum implementation.

---

## QM-09 — Robust gauge-flat recovery

**Module/doc.** `nanlmind_hermes/robust_recovery.py`;
`docs/ROBUST_RECOVERY.md`, `docs/ROUND012_015_MTH.md`.

**Mathematical role.** Two-hop median reconstruction of a zero-curvature field
`x[i,j]=θ[j]−θ[i]` under independently corrupted edges; mean and raw controls;
non-flat control.

**Current integration point.** `tests/test_robust_recovery.py` only. No
runtime decoder, no cycle-state recovery, no tool.

**Inputs/outputs.** `n`, edge dict, corruption fraction/seed → recovered edge
dict, MAE summary.

**Invariants/constraints.** Flat field has 255 independent gradient dimensions
(not 32,385 cycle dimensions); the median assumes a zero-curvature field; the
non-flat control must fail; no learned semantic state.

**Missing wiring.** No cycle-space decoder, no typed WordNet edges, no gauge
change model, no runtime use. `docs/ROBUST_RECOVERY.md` is a result report, not
an integration contract.

**Tests/evidence.** Four tests: exact flat field, median beats mean/raw,
non-flat destroyed, orientation antisymmetry. Not re-run in the targeted 22.

**Linked production tasks.** `P1-019`, `P2-018`. `P1-041` (wire/retire registry).

**Claim status.** **control/null.** Classical robust estimation; no semantic
recovery claim.

---

## QM-10 — Joint semantic decoder / semantic state

**Modules/docs.** `nanlmind_hermes/semantic_state.py`,
`scripts/joint_semantic_decoder.py`; `docs/JOINT_SEMANTIC_DECODER.md`,
`docs/ROUND020.md`.

**Mathematical role.** Two feature maps for a controlled novel-term yes/no
question: ordinary hashed ancestry state and cycle-projected ancestry state.
A linear ridge decoder with the same width is trained on disjoint parent
splits; a no-state control is included. `cycle_state()` buckets ancestry
edges to 256 vertices, projects to the cycle subspace and hashes to 64
dimensions.

**Current integration point.** Offline script using NumPy; tests cover the
feature functions. No chat, tool, definition answerer or layered learner calls
`ordinary_state`/`cycle_state`.

**Inputs/outputs.** WordNet-like `nodes`, start offset, depth, dimensions →
64/256-dimensional feature vector; script → accuracy/positive/negative/
hallucination metrics.

**Invariants/constraints.** Same decoder family and width for all variants;
disjoint parent splits; deterministic hashing; ordinary and cycle states are
projections of the same ancestry edges; one split, no multi-seed interval.
The cycle projection is linear/isometric and adds no information.

**Missing wiring.** No prose-derived relations, no autoregressive decoder, no
retrieval-only baseline in the runtime, no direct `CycleState` object, no
calibration/abstention integration, no tool. The docs explicitly say it is a
learned control, not a language generator.

**Tests/evidence.** `tests/test_semantic_state.py`: deterministic normalized
hash, bounded ancestor edges, nonzero ordinary state, finite/nonzero cycle
state. Script results are documented; not re-run.

**Linked production tasks.** `P1-019`, `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **control/null.** Documented result: ordinary state 0.7033,
cycle 0.6953, none 0.6847; no cycle-specific advantage.

---

## QM-11 — Semantic acquisition

**Module/doc.** `nanlmind_hermes/semantic_acquisition.py`;
`docs/SEMANTIC_ACQUISITION.md`, `docs/ROUND019.md`,
`docs/DEFINITION_ACQUISITION.md`, `docs/DEFINITION_QA.md`.

**Mathematical role.** Conservative symbolic acquisition: parse explicit
“a kind/type/species of X that …” or “is a X”, resolve a unique WordNet noun
head, attach a new term, inherit categories through hypernym closure, answer
yes/no/unknown and explain with evidence. No gradient training.

**Current integration point.** Tests and
`scripts/semantic_acquisition_bench.py`; no chat tool, no dictionary tool, no
runtime memory merge. `definition_decoder.py` is a separate retrieval path.

**Inputs/outputs.** Definition text + WordNet graph → concept record, parent
offset, inherited categories; `ask` → yes/unknown + evidence; `explain` →
graph path.

**Invariants/constraints.** One-sense-only head resolution; explicit abstention
when missing/ambiguous; persistent JSON atomic replacement; no invented synset;
inheritance is graph traversal, not prose understanding.

**Missing wiring.** No live WordNet route, no parser for arbitrary prose, no
cycle/decoder state, no tool surface in the 72, no promotion/merge contract, no
multi-hop confidence calibration. The doc’s 400-term result is script-scale.

**Tests/evidence.** `tests/test_semantic_acquisition.py`: conservative parser,
teach/two-hop, unknown abstention, ambiguous head, persistence. Not re-run in
the targeted 22. Benchmark result is documented; not re-run.

**Linked production tasks.** `P1-013` (WordNet route contract), `P1-031`
(depth/address), `P2-020` (reproducible benchmarks). `P1-041` (wire/retire registry).

**Claim status.** **code-backed / documented-not-wired.** Narrow symbolic
capability; not unrestricted definition understanding.

---

## QM-12 — Semantic state features (ordinary vs cycle)

**Module/doc.** `nanlmind_hermes/semantic_state.py`;
`docs/JOINT_SEMANTIC_DECODER.md`, `docs/ROUND020.md`.

**Mathematical role.** `hash_features` (BLAKE2b signed hashing), `ancestor_edges`
(bounded frontier), `ordinary_state` (hashed child/parent counts),
`cycle_state` (bucketed edge dict → `project_cycle` → signed hash). The cycle
map is a deterministic projection, not a trained state.

**Current integration point.** Only `scripts/joint_semantic_decoder.py` and
tests. No definition answerer, chat, CLOCKED or memory path.

**Inputs/outputs.** `nodes`, `start`, `depth`, `dimensions`, `vertices` →
normalized feature list.

**Invariants/constraints.** Deterministic; bounded depth; normalized vector;
cycle state is a projection of the same ancestry edges; hash collisions are
possible; no semantic labels.

**Missing wiring.** The “joint” decoder is offline; no runtime state store, no
tool, no query API, no candidate promotion. The docs already mark it a control.

**Tests/evidence.** Four tests in `tests/test_semantic_state.py`; not re-run in
the targeted 22. Script evidence documented only.

**Linked production tasks.** `P1-019`, `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired / control.**

---

## QM-13 — Temporal spine

**Module/doc.** `nanlmind_hermes/temporal_spine.py`; no `docs/SPINE.md` exists
(the README link is broken). `docs/SPECTRAL_WORD.md` says the next step is to
use the written matrix in the temporal spine.

**Mathematical role.** True rank involution `r ↔ 32639−r` (not complement
`255−i`); pair rank `i(511−i)/2+j−i−1`; per-rank complex coupling `A`; local
update `ε = F − A B`, `A ← A − η ε conj(B)`; mirror `A ← conj(A)`; mirror
residual and error.

**Current integration point.** `tests/test_temporal_spine.py` only. No
`temporal_lm`, `universe_chart`, `spectral`, chat or tool import.

**Inputs/outputs.** Sampled ranks, past/future complex dicts → coupling dict,
training history, mirror residual, status.

**Invariants/constraints.** Rank/pair bijection; `mirror_rank(r)=32639−r`;
mirror-coupling conjugacy; sampled ranks only (default 0..127, not 32,640);
no persistence format; classical.

**Missing wiring.** No document, no runtime frame, no connection to
`TemporalRegister` or `TemporalMirrorLM`, no spectral matrix source, no
evaluation. The README calls it “true rank mirror”; the module exists but the
language path uses FNV/`reverse8`, not this spine.

**Tests/evidence.** Three tests: rank bijection, true mirror is not complement,
local update converges and preserves mirror. The targeted run passed.

**Linked production tasks.** `P2-004` (broken link), `P2-005` (document),
`P2-018`. `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired.** The module is real; the README
integration claim is not backed by runtime code.

---

## QM-14 — Concept matrix

**Module/doc.** `nanlmind_hermes/concept_matrix.py`; `docs/MTH_LAYERS.md`
(“Concept matrix: saturation through superposition”), `docs/CONCEPT_SPECTRAL.md`.

**Mathematical role.** C(256,2)=32,640 unordered pair cells. Every statement
contributes all content-word pairs with a distance weight; typed relations add
a role and `s+`/`s-` rail; a concept leaf identity is the hash of its relation
links plus count; saturation measures occupied cells, density, degree, roles
and top pairs. No one-word-one-cell assignment.

**Current integration point.** `layered_learning.py` (`observe`,
`_load`, `status`) and `scripts/concept_spectral.py`,
`scripts/gemma_corpus_train.py`. It is the one quantum-math-adjacent structure
in a runtime path.

**Inputs/outputs.** Statement string + optional relation dict → pair cells,
concepts, saturation report, leaf signature. `save()` writes JSON; there is no
`load()` in the class (layered learning replays rows instead).

**Invariants/constraints.** Pair key sorted; weights accumulate; roles are
string counts; stopword filtering; no dense 256×256 per-word matrix; persistence
has no schema version validation beyond the outer file.

**Missing wiring.** No dedicated test file; no atomic load/validation; no
relation polarity; no concept-to-head/coupling export for `spectral.py`;
saturation is not used to drive coverage; no query API beyond `leaf`.

**Tests/evidence.** No `tests/test_concept_matrix.py` exists at this snapshot.
Coverage is indirect through `tests/test_layered_learning.py` and
`tests/test_free_chat_learning.py`. `docs/MTH_LAYERS.md` reports 89 statements/
138 concepts/560 pairs; that artifact was not re-run.

**Linked production tasks.** `P1-032` (dedicated test/load), `P1-015`
(relation semantics), `P1-016` (promotion).

**Claim status.** **code-backed runtime / partial integration.** The matrix is
used, but its richest claims (saturation protocol, concept spectral cut) are
not runtime behavior.

---

## QM-15 — Boolean wave

**Module/doc.** `nanlmind_hermes/boolean_wave.py`;
`docs/BOOLEAN_WAVE.md`.

**Mathematical role.** Multilinear extension of all 16 Boolean functions on
`[0,1]`; exact on `{0,1}`; fuzzy logic off the corners; `negate(a)=barrier−a`;
polarity helper. The document rejects Boolean algebra on complex amplitudes and
51-bit IEEE NaN payload claims.

**Current integration point.** `tests/test_boolean_wave.py`. The separate
`word_writer.py` module consumes similar amplitudes in
`scripts/spectral_word_write.py`, but `boolean_wave` itself has no chat, MTH,
layered-learning or tool caller.

**Inputs/outputs.** `mask`, `a`, `b` → 0/1 or float; `verify_exact()` →
correct/total/exact.

**Invariants/constraints.** Exact on 64 boolean cases; `OR(a,a) ≠ a` and
`XOR(a,a) ≠ 0` off `{0,1}`; phase is a separate channel; `polarity` is a
scalar sign, not a vector.

**Missing wiring.** No amplitude/phase state in the runtime, no word-leaf
write into a live matrix, no connection to `MTHHeadMapper` or temporal LM.
`docs/BOOLEAN_WAVE.md` calls the write rule “concrete”; it is a library, not a
runtime write.

**Tests/evidence.** `tests/test_boolean_wave.py` (Boolean exactness, fuzzy
non-idempotence, word writer). The targeted run passed.

**Linked production tasks.** `P1-015`, `P2-018`. `P1-041` (wire/retire registry).

**Claim status.** **code-backed / documented-not-wired.**

---

## QM-16 — Head population

**Module/doc.** `nanlmind_hermes/head_population.py`;
`docs/HEAD_POPULATION.md`, `docs/SPECTRAL_ARCHITECTURE.md`,
`docs/SPECTRAL_WORD.md`.

**Mathematical role.** 256 heads each hold a distribution over the 27-symbol
alphabet (`a–z` + end); a context activates a sparse population via FNV routing;
the population collapses to a next-letter distribution. Head 0 is the
NaNNet/polarity prior; `active`, `balance`, `reward` router, `decorrelate`,
`diversity`, `BigramBaseline`.

**Current integration point.** `scripts/head_population_diagnostics.py`,
`scripts/dictionary_completion.py`, `scripts/spectral_experiment.py`,
`scripts/spectral_word_write.py`, `tests/test_head_population.py`. No
`temporal_lm` or `layered_learning` uses it.

**Inputs/outputs.** Words/prefixes → per-head Counters; context → distribution,
next letter, votes, diversity metrics; matrix vectors for spectral scripts.

**Invariants/constraints.** Head 0 is not trained by default; all 256 heads
remain available; sparse active set per context; no dictionary of words is
stored; a head is not a letter. Rank-1 collapse occurs with `active=255`.

**Missing wiring.** The native chat uses `TemporalMirrorLM` with FNV heads and
its own 256-cell register, not `HeadPopulation`. There is no shared head
distribution with `spectral.py`, no concept matrix coupling, no runtime
`head_population` tool, no online feedback.

**Tests/evidence.** Six tests: superposition not letters; partial-word
training/no dictionary; all 256 activate with head 0 prior; diversity;
long-context balance; bigram baseline. Not re-run in the targeted 22.
`docs/HEAD_POPULATION.md` reports rank/accuracy tables; not re-run.

**Linked production tasks.** `P1-019`, `P2-018`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **documented-not-wired.** The diagnostics are real; the
“head population is the sequence predictor” claim is not the current chat path.

---

## QM-17 — MTH L0–L9, root Boolean table and MTHScorer

**Module/doc.** `nanlmind_hermes/mth_layers.py`;
`docs/MTH_LAYERS.md`, `docs/UNIFIED_TOPOLOGY_AND_LAYERS.md`,
`docs/HERMES_UNIVERSE_MEMORY.md`.

**Mathematical role.** The four NaN face classes (`+s`, `-s`, `+q`, `-q`) form
a 4×4 ordered pair space; each pair addresses a 4-bit Boolean truth table (16
functions of two inputs). L0–L2 verify bit/nibble/byte composition; L3–L6
build phonetic, morpheme, lexical and completion features; L7–L9 build typed
sentence edges, paragraph coherence and causal chains. `MTHScorer` combines
truth, frame, coherence, category error and per-layer signals.
`MTHHeadMapper` maps a relation through a root mask to a head, mirror, value,
rail and leaf tags.

**Current integration point.** `chat.py` (`_native_synthesize` builds the
stack and scorer), `temporal_lm.py` (`self.mth.score` in sentence selection),
`layered_learning.py` (`MTHLayerStack`, `MTHHeadMapper`),
`universe_chart.HeadScratchRegister`, `leaf_algebra.py`, and scripts.
This is the main runtime integration of the mathematical layer stack.

**Inputs/outputs.** Sentences + memory rows → layer metrics, edges,
vocabulary, prefixes; query/candidate → truth/frame/coherence/category error/
layer scores; relation → head/mirror/mask/Boolean name/value/pair/rail/role/
tags/digest.

**Invariants/constraints.** Root table retains all 16 class nibbles; L0–L2
exact; L7 truth requires subject-predicate-object coherence; object-only or
reversed matches do not count; category error penalizes stitched sentences;
`native_verified=False` until a native MTH runner is configured; the head
transform is a local reference, not the native NaN16 implementation.

**Missing wiring.** No native MTH runner is invoked by this module; the root
table is a local reference; L7–L9 features are heuristic; there is no
native-vs-Python parity for the Boolean root; no proof or relation gate. The
docs say “all ten layers are in the deliberation path” — true for
`MTHScorer`, but not evidence of native MTH execution.

**Tests/evidence.** `tests/test_mth_layers.py`: root 4×4/16 functions,
curriculum L0–L9, typed truth vs stitched sentence, every layer in forward/
scoring, head scratch register, structured relation answers, temporal MTH
scores. Not re-run in the targeted 22.

**Linked production tasks.** `P0-030` (native fixture/parity gate), `P1-010`
(typed ABI), `P1-015` (relation semantics), `P1-019` (coverage map).

**Claim status.** **code-backed runtime, partial native claim.** The L0–L9
reference stack is real; native MTH verification is not established by this
module.

---

## QM-18 — MTH 91 kernels, catalog and N16I runtime

**Module/doc.** `nanlmind_hermes/mth_catalog.py`,
`nanlmind_hermes/nan0s_runtime.py`,
`nanlmind_hermes/mth/spec/wave-capabilities.json`;
`docs/MTH_UNIVERSAL_RUNTIME.md`, `docs/MTH_FUNCTION_TREE.md`,
`docs/N16I_UNIFICATION.md`.

**Mathematical role.** The NaNROUTER MTH manifest: 91 kernels in 44 families,
NaN16 source + native fixtures. The portable NaN0S C core assembles with
`nmk.awk` and runs `nan_cli`. Four admitted float32 vector kernels
(dot/cosine/add/sub) have a general-operand N16I path; the other 87 are
fixture-only. `nanlm_mth_dispatch` relocates arbitrary assembled guest images
and runs them under the interpreter-in-NaN16.

**Current integration point.** Tool handlers (`nanlm_mth_catalog`,
`inspect`, `runtime`, `verify`, `execute`, `n16i_selftest`, `dispatch`) and
`mth_leaves` math pairing. Execution is local/derived; it authorizes no
NANNET route, lease, merge or canonical commit.

**Inputs/outputs.** Kernel ID, x/y 256-vectors, timeout, max steps, entry →
status, trap, steps, result, serial/output receipt, execution mode. Catalog →
kernel descriptor pages. Runtime → architecture/compiler/CLI state.

**Invariants/constraints.** Kernel sources/fixtures hashed and manifest-pinned;
91 kernels/44 families; four vector kernels only for `execute`; 87 fixture-only;
N16I is the unified path for the four; direct native is diagnostic/parity only;
workspace/fuel bounded; no Python math substitution; no canonical commit.

**Missing wiring.** 87 kernels lack typed operand ABIs; cross-architecture
builds are only locally exercised on aarch64; the N16I path is not a proof
certificate; `nanlm_mth_dispatch` is generic dispatch, not a typed ABI. The
catalog is not permission.

**Tests/evidence.** `tests/test_mth_runtime.py`: 91 kernels/sources, catalog
pages, architecture/bundled build, native fixture, unified N16I for four
vectors, arbitrary N16I dispatch, differential self-test, plugin tools, missing
compiler failure. `tests/test_n16i_unification.py`: four kernels native-vs-N16I,
tool metadata, leaf add pairing. Not re-run in the targeted 22; prior audit
reported these targeted tests passing. No 91/91 fixture execution artifact in
this audit.

**Linked production tasks.** `P0-030`, `P1-010`, `P1-011`, `P1-025`,
`P1-026`, `P2-031`.

**Claim status.** **code-backed runtime; partial coverage.** Catalog and four
vector ABIs are real; universal arbitrary-input execution is not.

---

## QM-19 — MTH data leaves, hubs and constrained pairing

**Module/doc.** `nanlmind_hermes/mth_leaves.py`;
`docs/MTH_DATA_LEAVES.md`, `docs/MTH_FUNCTION_TREE.md`.

**Mathematical role.** Immutable content-addressed leaves (`leaf:<sha256>`)
with kind `vector|scalar|matrix|relation`, hub ancestry, payload/function/
action relations/identities/constraints/constants, a real NaN16 image, and a
per-kernel function tree. `left::right` pairing is deny-by-default:
`connect` creates a relation; dot/cosine/add/sub execute the admitted N16I
vector kernel, check dimension/finiteness/symmetry/antisymmetry, and create a
new hash leaf.

**Current integration point.** Tool handlers (`nanlm_mth_hub`, `tree`,
`leaf_create`, `leaf_get`, `pair`, `leaf_verify`) and tests. This is a runtime
data layer, not a canonical store.

**Inputs/outputs.** Kind/payload/alias/parent → leaf; left/right/operator →
`status`, hashleaf, new leaf or denial code; tree action map/audit → 91 kernel
leaves/44 family hubs, image verification.

**Invariants/constraints.** No in-place mutation; hubs undeletable; only
admitted vector kinds/operators; input leaves byte-identical; result is new
hash leaf; identity checks before result; denial creates/executes nothing;
execution receipt is local/derived.

**Missing wiring.** No canonical merge; unknown-leaf pairing currently raises
outside the `try` instead of returning `status: denied/unknown_leaf`; audit can
skip a `None` leaf; `_new_leaf` may lose a new alias when content already
exists; `nanlm_mth_hub` returns only a summary; no on-disk image hash check in
all paths. These are audit defects, not features.

**Tests/evidence.** `tests/test_mth_leaves.py`: 9 seed hubs, real NaN image,
idempotent create, connect/math pairing, input preservation, illegal/unknown/
hub denial, audit without payloads, tool handlers. `tests/test_pipeline_integration.py`
adds NANNET-paired tree map/audit 91/91 and pairing. Not re-run in the targeted
22; prior audit reported 6/6 leaf tests and 8/8 pipeline tests.

**Linked production tasks.** `P0-031`, `P1-012`, `P1-029`, `P1-030`,
`P2-026`.

**Claim status.** **code-backed runtime, partial safety contract.**

---

## QM-20 — WordNet HW4 typed graph

**Module/doc.** `nanlmind_hermes/wordnet_typed.py`,
`nanlmind_hermes/wordnet_store.py`,
`nanlmind_hermes/dictionary_universe.py`,
`nanlmind_hermes/wordnet_reasoning.py`;
`docs/WORDNET_GRAPH.md`, `docs/WORDNET_TYPED.md`,
`docs/WORDNET_DIRECTED.md`, `docs/WORDNET_TOPOLOGY.md`,
`docs/WORDNET_ANCESTRY.md`, `docs/WORDNET_SEMANTIC_CYCLE.md`,
`docs/DEFINITION_*`, `session_docs/WORDNET_HW4.md`.

**Mathematical role.** Sparse WordNet 3.1 graph: synsets, lemmas, typed
directed pointers (26 relation types), glosses/examples; hypernym/meronym/
attribute/antonym etc.; definition inheritance via hypernym closure; canonical
HW4 route `nan://GM0:SC0:MC0:BS0:0:HW4:0`; no dense per-word 256×256 matrix.

**Current integration point.** `layered_learning.ingest_word` reads
definitions/hypernyms/typed edges and writes L0–L9 scratch; `dictionary_universe`
exposes structured definitions/relations/addresses; `definition_decoder.py`
provides IDF retrieval/answer; `semantic_acquisition.py` uses ancestry;
`semantic_state.py` uses parents; scripts benchmark WordNet. Chat does not
consult the graph directly except through memory rows.

**Inputs/outputs.** Word/lemma → synset offsets; synset offset → definition,
hypernyms, typed relations, inherited categories, addresses; query/answer →
structured records or graph paths.

**Invariants/constraints.** Sparse addresses; one word leaf points to a synset;
polysemy is superposed synset membership; definitions are lexical evidence, not
execution authority; unknown word returns no fabricated synset; archive counts
are archive-dependent; canonical v2 route preferred, legacy compatibility only.

**Missing wiring.** The code reads a local zip; `session_docs/WORDNET_HW4.md`
says HERMES reads WordNet through NANNET. `depth` in `ingest_word` is validated
and echoed but not used for traversal. `interface()["bidirectional"] = True`
has no write route. `mth_leaves.WORDNET_VECTOR_REF` uses
`::word/vector` instead of `::word/<lemma>/<offset>`. The error text names
`NANLM_WORDNET_ARCHIVE` instead of the `wordnet_archive` setting. No live HW4
route/lease/ACL resolution exists.

**Tests/evidence.** `tests/test_wordnet_tool.py`: canonical route ingest into
10 layers, missing archive fail-closed. `tests/test_layered_learning.py`:
canonical route, 10 layers, lexical rows. `tests/test_wordnet_typed.py`,
`test_wordnet_reasoning.py`, `test_wordnet_store.py`,
`test_definition_decoder.py`, `test_semantic_acquisition.py`. Not re-run in the
targeted 22. Docs report counts 82,192/117,953/84,505 etc.; archive-specific.

**Linked production tasks.** `P1-013`, `P1-014`, `P1-031`, `P2-017`,
`P2-025`, `P2-026`, `P2-027`, `P2-028`.

**Claim status.** **code-backed local read; documented-only live route.**
“Reads WordNet through NANNET” is not backed by code. “Bidirectional” is not
backed.

---

## QM-21 — Free-chat layered learning (L0–L9, 255 heads, 1 GiB scratch)

**Module/doc.** `nanlmind_hermes/chat.py`,
`nanlmind_hermes/layered_learning.py`,
`nanlmind_hermes/scratch_register.py`;
`docs/NANVERSE_LAYERED_LEARNING.md`,
`session_docs/FREE_CHAT_LEARNING.md`,
`docs/PLUGIN_ARCHITECTURE.md`, `docs/HERMES_UNIVERSE_MEMORY.md`.

**Mathematical role.** Every chat turn is projected into L0 bit → L1 nibble →
L2 byte → L3 phonetic → L4 morpheme → L5 lexical → L6 word_completion → L7
sentence → L8 paragraph → L9 narrative. Each layer is distributed across 255
heads (1..255; head 0 is not in this ensemble), written to a 128×8 MiB sparse
scratch page, and summarized into a concept matrix and relation rows.
Conservative promotion groups exact triples by support/confidence.

**Current integration point.** `chat.py` calls `_layered_learning` on every
turn when `chat_learning`; `__init__.py` exposes `nanlm_learning_cycle`,
`learning_status`, `learning_sync`, `wordnet_ingest`, `scratch_register`;
`LayeredLearningCoordinator` owns scratch, stack, mapper, ensemble, concepts,
head register. This is a real runtime path.

**Inputs/outputs.** Text, source, relations, tick → 10 layer records with
scratch addresses/head_top, relation rows, concept saturation, head balance;
status → policy/scratch/ensemble/concepts/head register.

**Invariants/constraints.** 128 pages × 8 MiB logical, sparse; pages 0..119 are
10 layer blocks of 12; page digest verified before read; append-only; 255 heads
1..255; deterministic superposition; policy min_support 3, min_confidence 0.6,
max 64 candidates/100 leaves, approval true, auto_sync false; no canonical
commit.

**Missing wiring / defects.** `_relation` strips polarity and confidence;
`observe` embeds only `relation_list[0]` in every layer leaf; all relation rows
go to `layered_learning.json`, not per-relation scratch leaves; promotion groups
by `(subject,predicate,object)` so support/refute collapse; `sync` can be
reached by more than `nanlm_learning_sync` (current tree gates with
`merge_enabled`); `ConceptMatrix` has no load path; no dedicated concept test;
scratch verification is per page, not a global cross-page manifest.

**Tests/evidence.** `tests/test_layered_learning.py` (10 layers/255 heads,
approval, repeated relation, WordNet route, reopen, bad input),
`tests/test_free_chat_learning.py` (provider/native feed, chat_learning false,
scratch failure resilience), `tests/test_scratch_register.py` (1 GiB logical,
sparse, durability, tamper). Not re-run in the targeted 22 except adjacent
modules; prior audit reported these passing.

**Linked production tasks.** `P0-015`, `P0-017`, `P0-018`, `P1-015`,
`P1-016`, `P1-032`, `P1-033`, `P1-034`, `P2-029`.

**Claim status.** **code-backed runtime / partial semantics.** The 10-layer
scratch path is real; polarity-preserving learning and promotion are not.

---

## QM-22 — CLOCKED and NaN head cortex

**Module/doc.** `nanlmind_hermes/clocked_tools.py`,
`nanlmind_hermes/clocked_mode.py`, `nanlmind_hermes/clocked_client.py`,
`nanlmind_hermes/head_cortex.py`, `nanlmind_hermes/scratch.py`,
`nanlmind_hermes/transform_chain.py`;
`docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md`.

**Mathematical role.** CLOCKED is an external structured-evidence service, not
a quantum module. The Hermes side plans bounded typed endpoint calls, stores
results in an append-only hash-chained scratch, packages evidence, sends it to
the NaNLMind head cortex, and synthesizes an answer through a second head
validation. Head candidates are isolated Boolean-table masks over 256 logical
heads; evaluation compares candidate/baseline relation metrics.

**Current integration point.** Tools `nanlm_clocked_plan/execute/scratch`,
`nanlm_head_process`, `nanlm_clocked_synthesize`,
`nanlm_head_candidate/learn_candidate/candidate_evaluate`. Head cortex is an
HTTP client to a NaNLMind receiving endpoint; no local head state.

**Inputs/outputs.** Question + source descriptors/text + scratch ID + approval
+ plan digest → mode plan, task plan, typed results, evidence atoms, scratch
revision; draft → head response/truth state; candidate request → isolated
candidate response with `promotion_allowed=false`.

**Invariants/constraints.** Planning excludes raw source text; execution
requires `approved=true` and exact `plan_sha256`; fixed endpoint allowlist;
scratch lane writer ownership; CLOCKED outputs are evidence, not ground truth;
candidate cannot mutate canonical state; no M0–M9 full sequence; no six-runner
promotion.

**Missing wiring / claims.** The full M0–M9 transformation sequence is not
demonstrated end-to-end; candidate promotion is absent; head-cortex live
deployment is separate; the service is not a local model. `docs/CLOCKED...`
already marks these outstanding.

**Tests/evidence.** `tests/test_head_cortex_evaluation.py` (client contract,
non-promotion, candidate evaluation), `tests/test_clocked_integration.py`,
`tests/test_pipeline_integration.py`, `scripts/live_smoke.py` (11/11 recorded,
not re-run). No live M0–M9 artifact.

**Linked production tasks.** `P1-035`, `P1-036`, `P0-019`, `P0-024`,
`P2-030`.

**Claim status.** **code-backed partial integration; documented-only full
M0–M9/promotion.**

---

## QM-23 — Temporal mirror LM and local runner

**Module/doc.** `nanlmind_hermes/temporal_lm.py`,
`nanlmind_hermes/universe_chart.py`; `docs/LOCAL_RUNNER.md`,
`docs/LONG_CHAT_DEMO.md`, `docs/FREE_CHAT.md`,
`docs/SPECTRAL_WORD.md` (proposed link).

**Mathematical role.** Word-level predictor over an exposed vocabulary:
trigram/bigram/unigram interpolation, a 256-cell temporal frame updated after
each emitted word, FNV head address + `reverse8` mirror, mirror transition
mass, repetition penalties, beam search. The local runner maps an 8 MiB chart
with a `TemporalRegister` bound to one pair cell's `s+`/`s-` rails and a
`HeadScratchRegister` for relation leaves.

**Current integration point.** `chat.py._native_synthesize` is the default
chat path; `temporal_lm` loads/saves `language/temporal_mirror.json`, calls
`LocalRunner.open`, and uses `MTHScorer`. The runner is not a remote model.

**Inputs/outputs.** Exposure sentence → vocab/counts/pairs/seen; query + allowed
seed → candidate sentences/score/considered/mirror state; chart → verified
tensor bytes, rail read/write, temporal state, head register.

**Invariants/constraints.** Word heads are FNV modulo 256, mirror `reverse8`;
the frame is internal; no NaNNet rail registers written; 8 MiB chart with
SHA/dual-slot; no canonical commit; `MTHScorer` adds truth/frame/coherence.
It does **not** use `HeadPopulation`, `spectral.py`, `cycle_state`,
`temporal_spine`, or `phase_memory`.

**Missing wiring / claims.** The docs and README describe “heads” and
“temporal mirror” in the native model, but the predictor's heads are FNV
buckets, not the `HeadPopulation`/spectral head matrix. The temporal spine is
not wired. The chart is a local derived overlay, not canonical training. The
`temporal_spine` doc link is broken.

**Tests/evidence.** `tests/test_temporal_spine.py` covers the spine, not the
LM. Native chat is covered by `tests/test_chat.py`, `test_free_chat_learning`,
`test_clocked_integration` and long-chat demo docs. `tests/test_spectral.py`
covers the separate spectral operator. No test asserts that
`TemporalMirrorLM` uses `HeadPopulation` or `temporal_spine` because it does
not.

**Linked production tasks.** `P1-033`, `P1-034`, `P1-019`, `P2-004`,
`P2-005`, `P2-018`.

**Claim status.** **code-backed runtime; documented-not-wired for the
spectral/head-population/spine claims.**

---

## QM-24 — Definition, lexical and corpus memory

**Module/doc.** `nanlmind_hermes/definition_decoder.py`,
`nanlmind_hermes/lexical_memory.py`,
`nanlmind_hermes/corpus_memory.py`;
`docs/DEFINITION_QA.md`, `docs/DEFINITION_ACQUISITION.md`,
`docs/CORPUS_MEMORY.md`, `docs/ROUND018.md`, `docs/ROUND020.md`.

**Mathematical role.** IDF-weighted definition retrieval, structured
`answer_what`/`answer_is_a` records, exact typed ancestry, relation readout;
verbatim lexical memory; deterministic text ingestion, SHA-256 dedup,
hash-based 80/10/10 split, IDF token-overlap retrieval. No generation.

**Current integration point.** Scripts
(`definition_qa_bench.py`, `definition_acquisition.py`,
`wordnet_*`) and tests. `chat.py` does not call `DefinitionAnswerer`;
`lexical_memory` is not in the 72 tools; `corpus_memory` is an offline
evidence layer.

**Inputs/outputs.** Definition prefix/offset → ranked synsets/answers/paths;
word/definition → persisted lexical record; files/directories → corpus
manifest/retrieval results.

**Invariants/constraints.** Retrieval overlap is not understanding; exact
ancestry is graph traversal; corpus split is deterministic hash, not
stratified; no dense per-word matrix; no generated prose; private corpus must
not be committed.

**Missing wiring.** No chat tool, no runtime answerer, no joint state/cycle
conditioning, no promotion/merge, no live WordNet route. The Round 018/020
results are offline baselines.

**Tests/evidence.** `tests/test_definition_decoder.py`,
`test_lexical_memory.py`, `test_corpus_memory.py`,
`test_wordnet_reasoning.py`, `test_semantic_acquisition.py`. Not re-run in the
targeted 22. Docs report 2,000-synset/2,400-passage numbers; not re-run.

**Linked production tasks.** `P1-013`, `P1-031`, `P2-020`. `P1-041` (wire/retire registry).

**Claim status.** **code-backed offline; documented-not-wired.** Not a
language decoder.

---

## QM-25 — Cross-resolution memory feedback

**Module/doc.** `nanlmind_hermes/memory.py`,
`nanlmind_hermes/cross_resolution.py`;
`docs/HERMES_UNIVERSE_MEMORY.md`.

**Mathematical role.** Deterministic 10×10 co-occurrence matrix over the ten
resolution coordinates; cross-level polarity agreement/contradiction;
descriptive consistency `(aligned − contradicted)/(aligned + contradicted)`;
n-ary hyperedge/equation records; append-only event ledger; evidence
assessments and reorganization.

**Current integration point.** `tools.py` memory handlers
(`memory_ingest`, `reorganize`, `cross_resolution`, `merge`, `query`,
`assessments`), `chat.py` reflection and `run_curriculum_experiment`; also
`nanlm_memory_cross_resolution`. It is a runtime path, but deterministic and
symbolic.

**Inputs/outputs.** Discovery/event records → SQLite event ledger, projections,
assessment records, 10×10 matrix, equation/hyperedge records; queries →
active records/assessments.

**Invariants/constraints.** Append-only event IDs; immutable discoveries;
contradictions retained; projections rebuildable; no canonical native state;
no learned top-down evaluator; no proof of semantic equivalence; remote merge
is NANNET's boundary, not local.

**Missing wiring.** The cross-resolution score is descriptive, not used as a
gradient; no empirical test that scores predict held-out language improvement;
no authenticated delta transport; no generation/base-root reconciliation;
`nanlm_memory_merge` is a local event union, not canonical merge.

**Tests/evidence.** `tests/test_memory.py`, `tests/test_cross_resolution.py`,
`tests/test_merge_contract.py`, `tests/test_pipeline_integration.py`. Not
re-run in the targeted 22.

**Linked production tasks.** `P0-016`, `P0-017`, `P1-016`, `P2-019`.

**Claim status.** **code-backed runtime; partial semantics.** Deterministic
structural feedback, not a learned semantic engine.

---

# Cross-cutting integration findings

1. **Only nine rows have a runtime caller:** QM-14 concept matrix, QM-17 MTH
   layers, QM-18/19 MTH runtime/leaves, QM-20 WordNet local read, QM-21 layered
   learning, QM-22 CLOCKED/head cortex, QM-23 temporal mirror LM, QM-25 memory
   feedback. The other rows are offline experiments, tests or scripts. This is
   not a defect by itself; the defect is documentation that implies otherwise.
2. **The native chat path is not the spectral/head-population path.** It uses
   `TemporalMirrorLM` with FNV heads and an internal 256-cell frame. The
   spectral matrix, `HeadPopulation`, `cycle_state`, `temporal_spine` and
   `phase_memory` are not called by `temporal_lm` or `chat`.
3. **The only admitted general-operand mathematical execution is four float32
   vector kernels.** Everything else in MTH is fixture verification or
   dispatch. No quantum/gauge/cycle module has a typed native operand ABI.
4. **The only canonical-adjacent claims are negative:** no canonical merge,
   no six-runner barrier, no promotion, no MTH proof certificate, no physical
   quantum result. Keep those negative claims in every status and receipt.
5. **The main production integration risk is naming.** “Head”, “state digest”,
   “merge”, “verify”, “truth” and “native” are overloaded across local derived
   state, native fixture verification, and canonical NaNLMind state. Use the
   scope labels from `P0-021` in every public result.

# Matrix count

- **25 subsystem rows** (`QM-01` … `QM-25`)
- **9 rows with a runtime caller** (QM-14, QM-17, QM-18, QM-19, QM-20,
  QM-21, QM-22, QM-23, QM-25; QM-18/QM-19 are tool-only paths)
- **No row claims physical quantum behavior, a quantum speedup, a canonical
  merge, or a full M0–M9 implementation.**

Linked production checklist: `docs/PRODUCTION_READINESS_TODO.md`.
