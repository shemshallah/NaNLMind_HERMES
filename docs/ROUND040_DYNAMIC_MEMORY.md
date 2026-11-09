# Round 040 — dynamic resident memory value

Status: **architecture input, implementation partially mapped**. This document
adds the validated Round 040 capability to the master architecture and TODO. It
does not freeze the dynamic-conceptual track and does not claim a unique NaN
advantage.

## What Round 040 actually demonstrates

The supplied archive is extracted at `experiments/round040/nan_round040/`.
Its runnable prototype:

- accepts a small controlled grammar of prose statements;
- extracts two entities and a relation/polarity with a trained statistical
  reader;
- writes the accepted relation into an 8,388,608-byte mirror root;
- answers two/three-hop ancestry queries by deterministic graph traversal;
- writes further facts after the initial training and retains the old answers;
- saves and restores the native root and a separate metadata file;
- exposes a local CLI: `add …`, `ask …`, `save`, `quit`.

Measured artifacts, three seeds (17, 29, 43):

| Measurement | Result |
|---|---:|
| Old-question answers before adaptation | 400/400 per seed |
| Old answers after gauge drift | 400/400 per seed |
| New answers after 80 additional chains | 160/160 per seed |
| Conventional indexed control | ties all clean/adapted tests |
| Raw single-edge retrieval after gauge | 281/400, 210/400, 206/400 |
| Native pair audit | pass |
| Snapshot/readout restoration | exact |
| Native root | 8,388,608 B |
| Root cycle slots | 32,385 |

The result is **mutable memory**, not weight-changing continual learning: the
relation reader is frozen during the adaptation phase. The benchmark is a
controlled graph/prose construction, not open-domain extraction. The free
subword generator remains incoherent and is not conditioned on these memories.

## Architectural value

The important contribution is the separation of roles that the master TODO now
adopts:

1. **Reader** turns a statement into typed evidence (entity spans, relation,
   polarity, provenance).
2. **Resident writer** stores the relation state in a stable address and
   increments generation only after validation.
3. **Geometric memory** holds a cycle value; cycle readout is gauge-invariant.
4. **Composer** traverses typed relations and produces a path or abstains.
5. **Readout** renders the path into language.
6. **Snapshot** persists the root plus explicit metadata as a versioned pair.

This is the same shape as the current resident API:

| Round 040 role | Current NaN component | Gap |
|---|---|---|
| controlled relation reader | `resident/learning.py`, `ingest.py` | open-domain extractor not implemented |
| dynamic write | `ResidentUniverse.ingest`, `full_cycle_memory` | no single transactional `add_fact` API yet |
| cycle state | `resident/full_cycle_memory.py` | proven full 32,385 basis |
| traversal/composition | `resident/retrieval_hypergraph.py`, `composition_gate.py` | connect directly to a session |
| local CLI | `scripts/resident_universe_cli.py` | no `add/ask` grammar |
| snapshot | `resident/persistence.py`, `cycle_snapshot.py` | root + metadata pair not transactional |
| language readout | temporal/field learners | not conditioned on dynamic reads |

## Point-by-point integration tasks

| ID | Task | Gate |
|---|---|---|
| R040-01 | Define `DynamicFact(source, relation, polarity, target, provenance, generation)` with bounded strings and stable content ID | schema tests; no unrestricted text |
| R040-02 | Add `add_fact` to the resident facade: validate, write one full-cycle slot, write leaf/relation/hyperedge metadata, commit generation once | no partial write; duplicate idempotent |
| R040-03 | Define cycle-slot allocation across 32,385 cycles, with overflow denial and no 256-only fallback | 32,385 capacity test |
| R040-04 | Implement `ask(source, target)` as bounded typed BFS, returning path + evidence + abstention reason | positive/negative paths |
| R040-05 | Render answers through the existing composer only after path validation | no answer stored in one slot |
| R040-06 | Save/restore root + metadata with a shared generation/digest and crash-consistency rule | reopen equality; mismatch denied |
| R040-07 | Add `session` CLI actions `add`, `ask`, `save`, `quit` over the resident API | scripted session test |
| R040-08 | Preserve polarity/contradiction in the graph; `opposes` must not be traversed as positive ancestry | conflict tests |
| R040-09 | Retention after additional writes (frozen reader) | old + new question suite |
| R040-10 | Matched conventional graph baseline with identical parsed facts | exact tie required for clean tests |
| R040-11 | Gauge audit of the resident root before/after writes | cycle exact; raw edge fails as control |
| R040-12 | Connect dynamic reads to a token-level decoder without making PyTorch mandatory for the pure path | external checkpoint adapter, optional skip |
| R040-13 | Continued gradient training (not just writes) and forgetting test | explicit held-out before/after |
| R040-14 | Open-domain relation reader with uncertainty/provenance; do not treat parser output as verified truth | source-disjoint evaluation |

## What this changes in the current plan

- The dynamic memory track is no longer only speculative: the Round 040
  prototype gives a concrete bounded capability to reproduce.
- It **does not** justify starting the C-track architecture freeze. The C track
  remains held.
- It adds a measurable bridge item: **write-through dynamic facts → traversal →
  language readout**.
- It keeps the conventional control mandatory; Round 040 itself reports a tie.

## Reproduce the supplied artifact

```bash
cd experiments/round040/nan_round040
c++ -O3 -std=c++20 -fPIC -shared native_episode.cpp -o libepisode.so
python experiment.py --seeds 17 29 43
python session.py --new
```

The archive is not yet executed as part of the NaN test suite; its source and
results are integrated as an architecture input, and the implementation tasks
above remain explicit.
