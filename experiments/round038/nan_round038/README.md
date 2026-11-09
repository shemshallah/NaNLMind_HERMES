# NaN Recursive Learning — Round 038

## Research status (2026-11-08)

**Implemented, trained, evaluated and reproduced:** trainable SO(2) phase paths, noncommutative SO(3) Givens transport, a holonomy-feature candidate-path decoder, learned path attention, recurrent token-conditioned phase shifts, a combined model, frozen-angle ablations, scalar and conventional neural baselines, and 8 MiB native mirror-pair memory audit.

**Not established:** an open-domain LLM, arbitrary prose understanding, a physical quantum effect, a fundamental NaN-specific advantage, or superiority over an equally capable conventional graph. The full hybrid is a *trained* but **restricted three-token autoregressive generator** ('evidence', evidence-state, end-of-sequence); sentence entity spans are supplied to the frozen Round 037 writer. The native C++ root stores source features, not the learned neural rotation matrices. Learning updates happen in PyTorch, with gradient flow through learned rotations, attention, and the token decoder.

## Overview

- Starting point: **Round 037 genuine C++20 mirror-pair root**, 256 heads, 32,640 unordered pairs, 8,388,608 bytes total. The new ABI adds independently gauged lanes (used for relation type and adjacency flags) without changing size, reverse-edge antisymmetry or R037 functions. The original R037 `native_cycles.cpp`, source, writer checkpoints, decoder checkpoint and other tests are retained.
- Frozen **Round 037 bidirectional GRU predicate writer** (`writer_17.pt`, `writer_29.pt`, `writer_43.pt`) supplies relation classes for cached synthetic prose patterns with fixed/externally supplied entity spans. No previously unseen sentence syntax is involved in Round 038 evaluation beyond Round 037's known template family.
- **Differentiable learned reasoning**: candidate paths are represented via SO(2) learned phases, SO(3) Givens transformations, normalized path attention and a 3-token GRU decoder. Noncommuting 3D transformations can distinguish path order that commutative 2D phase sums cannot. Query-conditioned rotations depend on the recurrent decoder state at each token step.
- **Holonomy prototype**: loop/closure-trace features of composed 3D rotations are added to the candidate representation. A separate algebraic test verifies the gauge-conjugacy trace invariant of correctly transformed closed loops; the neural prototype has **not** been tested under arbitrary SO(3) frame gauges. The physical memory test instead concerns additive vertex gauges in the 8 MiB C++ root.
- **Four-valued evidence**: support and refutation are kept as independent Boolean/evidence channels, resulting in 'unknown', 'supports', 'refutes', or 'both'. Uncertainty and distractor data are simulated. No refuting evidence is erased by opposite-phase cancellation.

## Experiment A: WordNet-backed fact/evidence classification

Each item uses a real three-edge **WordNet 3.1 noun-hypernym concept chain** (concepts held out by episode), with controlled distractor candidate paths, additional synthetic refutations and uncertainty cues. Four relation states, two structural feature lanes and explicit junction-validity flags are constructed from the candidate episodes, then recovered through the **native C++ 8 MiB root**. Scores are for this controlled evidence task, **not** unrestricted extraction or free-form reasoning from raw natural prose.

- Same source features for all models, no test-label exposure during training; randomized seeded 60/20/20 train/validation/test split.
- Models trained with 420 AdamW updates; checkpoint chosen on validation accuracy (CE tie-break); output decoded autoregressively and checked against four-valued label.
- Strongest conventional control: a permutation-invariant DeepSets encoder with per-edge one-hot features, trainable attention and max-pooling (8,856 parameters). Original conventional GRU path encoder also included.

| Approach | Seed 17 | Seed 29 | Seed 43 |
|---|---:|---:|---:|
| deepsets | 48.04% | 48.89% | 100.00% |
| phase | 74.86% | 100.00% | 100.00% |
| rotation | 100.00% | 72.78% | 100.00% |
| full | 100.00% | 100.00% | 100.00% |

All three seeds scored **100% for the full hybrid**, though DeepSets reached 100% in one seed. Single-seed results cannot establish a consistent advantage by any particular component. Full seed-by-seed results and ablations are in `round038_seed{seed}_results.json`.

## Experiment B: order-sensitive composition probe (synthetic)

Every candidate path has exactly the **same multiset** of three relation symbols and the same continuity flags. Labels depend only on whether a path contains the order `0→1→2` (support), `2→1→0` (refute), both, or neither. This is a deliberately constructed operator-composition probe, **not a WordNet factual inference benchmark**.

A commutative SO(2) sum has identical output on every permutation, so a phase-only system cannot solve the task from those inputs. SO(3) ordered products may distinguish them; so can a sufficiently strong conventional sequence model. Models trained 500 updates, same 60/20/20 split protocol.

| Approach | Seed 17 | Seed 41 | Seed 43 |
|---|---:|---:|---:|
| deepsets | 68.33% | 100.00% | 100.00% |
| phase | 18.33% | 25.56% | 21.67% |
| rotation | 18.33% | 100.00% | 100.00% |
| full | 100.00% | 100.00% | 100.00% |

The full hybrid and the strong conventional DeepSets baseline both reached **100% in two or three seeds**; full hybrid reached 100% in all three. Phase-only remained near chance. SO(3) alone was inconsistent across seeds (one at 18.33%). These outcomes support the value of **order-sensitive transformations**, but they **do not establish a uniquely geometric learning advantage**.

### Full ablation on principal seeds

| Variant | WordNet seed 29 | Order seed 41 | Params (WordNet) |
|---|---:|---:|---:|
| conventional | 48.89% | 73.33% | 11,472 |
| deepsets | 48.89% | 100.00% | 8,856 |
| scalar | 100.00% | 27.22% | 7,800 |
| phase | 100.00% | 25.56% | 6,744 |
| frozen_phase | 48.89% | 25.56% | 6,744 |
| rotation | 72.78% | 100.00% | 6,792 |
| frozen_rotation | 92.78% | 76.11% | 6,792 |
| holonomy | 100.00% | 100.00% | 6,888 |
| dynamic | 100.00% | 99.44% | 6,864 |
| full | 100.00% | 100.00% | 6,960 |


The conventional GRU and unstructured scalar mean baselines are weaker than the DeepSets control on some seeds; no conclusions should be based solely on those weaker baselines. The number of learnable parameters is reported because the models have **comparable but not identical capacity**; exact parameter matching remains future work. The benchmark generator provides connectivity indicators, so success is not evidence that entities and graph junctions were discovered from raw prose.

## Native geometry / math validation

- Native root: **8,388,608 bytes**; **32,640** canonical pair slots and **32,385** available anchored triangle evidence slots.
- Independently gauged numerical lanes: **maximum error 1.5258789e-05** in the separate high-amplitude audit; original R037 baseline audit remains ≈1.81×10⁻⁵.
- After large gauge perturbations, neither raw-edge readout nor its source feature vector matched the unperturbed tensor under the strict audit; triangle-circulation readouts did, within floating-point roundoff.
- Binary root snapshot restore: **exact**. Original Round 037 test still passed 600 independent native test values.
- SO(2) commutator norm: **0.0000000**, vs SO(3) noncommuting rotation norm **1.139924**.
- Conjugation-invariant loop-trace error for mathematically consistent local-frame changes: **1.19e-07**.
- Backpropagated neural-angle gradient norm: **0.0102797**, recurrent token-conditioning gradient: **0.00855878** (both nonzero). These confirm the learned geometric transformation participates in the loss.
- Contradictory evidence is represented as state **3 ('both')** rather than forced to zero.
- Both sets of evaluated model outputs were unchanged after numerical gauge shifts in their native source features, as recorded in the results JSON files.

## Generation and baseline regression

- Trained Round 038 hybrid autoregressively produced the correct restricted evidence-state answer for **180/180** held-out WordNet-backed episodes; twelve examples appear in `generated_answers_038.json`. This output is an evidence-token label rather than free text with entity names.
- The separate, original **Round 037 trained sequence generator** was rerun against its original source, yielding **90/90 semantically correct** restricted-grammar answers and **90/90 unchanged** generated outputs after gauge perturbation. Raw corrupted edge reads recovered zero complete source strings. See `generation_results.json` and `ROUND037_GENERATION_REGRESSION.log`.

## Reproduction

Requires Python 3, PyTorch, NumPy, g++ with C++20, and the separate original WordNet 3.1 dictionary archive `wn3.1.dict.zip` at `/mnt/data/files/wn3.1.dict.zip` or passed explicitly via `--wordnet`. Dependency SHA-256: `1820414c1056652b56717df5794609ef0c91aae033d3fc54ddc9846249294285`. The WordNet archive is **not duplicated** inside this ZIP; it remains a separate input artifact.

```bash
python test_trig_geometry.py
python round038.py --wordnet /path/to/wn3.1.dict.zip --seed 29 --limit 900 --steps 420
python order_probe.py --seed 41 --count 900 --steps 500
python replay_answers.py  # modify its default WordNet path if needed
python test_native.py
python generation_audit.py  # currently requires /mnt/data/wn3.1.dict.zip
```

One per-task model checkpoint is named `trig_wordnet_<mode>_<seed>.pt` or `trig_order_<mode>_<seed>.pt`. Main seeds include all ten variants, while auxiliary seeds cover the strong conventional control, phase, SO(3), and full hybrid. Train/validation/test partitions and class frequencies are in the JSON files. Retraining can vary slightly with PyTorch platform/version and non-deterministic kernels.

## Current interpretation / next falsifiable goal

This is a **publishable partial engineering result**, not a publishable claim of superior open-domain reasoning. It demonstrates a real native-memory-to-trainable-geometric-reasoning-to-token-generation chain, a calibrated structural setting in which phase commutativity fails, and a hybrid with stable performance across three seeds. It is **not yet** the learning breakthrough sought in Round 037, because the writer uses restricted, supplied entity spans, the neural transport is externally trained, the decoded vocabulary is tiny, and a strong non-geometric control can tie the solution. The next experiment must remove those scaffolds and compare against a parameter-/compute-matched graph/attention control on naturally occurring raw prose with multi-hop novel fact retention.

See `CLAIM_LEDGER.md` for the claim boundary and `RESEARCH_LOG.md` for experiment provenance. `README_ROUND037.md` retains the previous README.
