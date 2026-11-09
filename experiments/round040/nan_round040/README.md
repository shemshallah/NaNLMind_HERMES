# NaN Recursive Memory — Round 040: Dynamic episode writes + interactive local universe

## Status / scope

This is a working, **limited-grammar** prose-to-episodic-memory prototype, not a finished fluent generative model. It implements a trained statistical relation-label reader after entity-boundary extraction, direct writes to a resident **8,388,608-byte** native mirror root, native triangle-cycle reads, a conventional identical-facts baseline, multi-hop query composition, continual memory writes, binary snapshot restoration, and an interactive terminal interface. The independently trained Round-039 2,048-subword free-sampling generator is included for a frank fluency check; it is **not yet conditioned on Round-040 episodic memory**.

## Key experimental differences from earlier rounds

- The root changes when novel statements are ingested, instead of merely carrying a fixed 16x16 language-adapter weight matrix.
- 255 usable heads (1–255) and one reserved hub (0). Anchored triangles (0,a,b), `1 <= a < b <= 255`, give **32,385** independent geometric slots. They live inside the 8 MiB object with **32,640** mirror pair cells.
- A learned TF-IDF (1–3 grams) logistic relation reader distinguishes **forward ISA, reverse ISA, negated ISA, and unrelated relation** on 1,200 controlled training sentences. Entity spans are extracted by a restricted grammar, not by a general NER model. All test source concepts are novel nonce entities. This is intentionally easier than open-domain prose extraction.
- The native memory stores a scalar relation state as an anchored circulation `M(0,a)+M(a,b)-M(0,b)`. Every accepted clause writes a new slot; each slot has its concept endpoints and provenance in a separate Python metadata table. Conventional hash/edge traversal receives exactly the same parsed facts.
- Native cycle reads and composition survive strong random node-gauge drift, but the conventional-index control ties all clean and adapted QA tests. No special NaN-bit advantage is established, and metadata is still ordinary Python memory.
- The question answering stage is deterministic graph traversal and rule-rendered output, **not unrestricted neural language generation**.

## Multi-seed benchmark (3 seeds: 17, 29, 43)

Each seed trains a fresh relation reader from 1,200 controlled training sentences. It then writes 200 novel three-edge concept chains, each with a distractor. The 400 test queries include 200 supported three-hop queries plus 200 unsupported queries. Each seed later writes 80 further novel chains and answers 160 additional queries. The test concepts do not appear in relation-reader training.

| Test, per seed | Native | Conventional |
| --- | ---: | ---: |
| Initial old queries | 400/400 | 400/400 |
| Old queries after gauge shift | 400/400 | 400/400 (index unchanged) |
| Old queries after 80 additional chains | 400/400 | 400/400 |
| New queries after additional chains | 160/160 | 160/160 |
| Binary root snapshot restoration | exact | Not a native root |

Raw-edge retrieval after gauge shift (same query set) was 281/400, 210/400, and 206/400. This raw-edge control is not a matched robust conventional memory—it deliberately discards the two additional geometric edges.

**Benchmark caveats:** the supported answers are compositional in a controlled constructed graph, not facts independently discovered from free natural prose. The exact entity spans, relation templates, and multi-hop inference algorithm restrict the benchmark. Continual retention is **writes-only adaptation with frozen reader**, not weight-changing continual gradient training; it does not measure catastrophic forgetting under parameter updates. The model's unrestricted free-generation text remains incoherent. A conventional graph that also calculates the cycle invariants would match the gauge robustness. All reported values can be reproduced from `experiment.py`.

## Interactive local universe

Dependencies: `g++`, Python 3.10+, NumPy, scikit-learn (plus PyTorch and sentencepiece only for free text sampling).

```sh
g++ -O3 -std=c++20 -fPIC -shared native_episode.cpp -o libepisode.so
python experiment.py --seeds 17 29 43
python session.py --new
```

Example interactive conversation (with the supported grammar):

```
add bota is a type of mera.
add mera belongs to the class zena.
ask bota zena
save
quit
```

Rerun `python session.py` without `--new` to restore the native 8 MiB root plus its separate metadata. Two-file snapshots are **not transactional**: a crash between writes could leave them inconsistent; do not use for multiwriter deployments or treat them as a secure file format.

Free-sampling audit using the **frozen and weak** subword generator from Round 039:

```sh
python free_generation.py
```

This correctly samples novel next tokens, but it is independent of `experiment.py` and does not answer the episodic questions. The runnable script and its failed examples are included to keep this distinction explicit.

## What remains before the requested finish line

1. Replace constrained entity-slot detection and sentence-template training with open-domain prose relation/event extraction, including uncertainty and provenance.
2. Connect episodic read operations to a trainable autoregressive language decoder at token level, and backpropagate through dynamic sparse memory reads/writes.
3. Evaluate source-disjoint, human-authored multi-fact QA with paraphrase and distractor shifts; compare equal evidence, memory budget, and optimizer budget against conventional attention and retrieval.
4. Test retention after **continued gradient training** (not just further writes), and nontrivial generated explanations. Report held-out token loss, human-readable samples, factual correctness, abstentions, and memory speed.

The current result is a functional native mutable memory and controlled reasoning prototype, **not general fluent generation or a demonstrated unique geometric advantage**.
