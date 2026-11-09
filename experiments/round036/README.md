# Round 036 — Neural multi-source generation, native cycle-resident evidence

This package is a *controlled generative experiment* advancing from the fixed grammatical renderers in R035. It does **not** establish an open-domain generative language model, autonomous text extraction, or a uniquely NaN-specific advantage.

## Implementation

- `train.py`: parses **real WordNet 3.1 noun synsets**, builds source-child-disjoint train/validation/test splits, and trains a 38,268-parameter neural encoder–attention–autoregressive GRU decoder. Premises are shuffled and expressed in three natural-language relationship phrasings. The decoder predicts entire output word sequences, including multiple well-formed positive phrasings and a learned unsupported-evidence sentence, rather than using a hard-coded answer renderer.
- **Important abstraction:** WordNet entity names are normalized to `e0`,`e1`,... before decoding. This is a slot-language model and **does not learn to generate arbitrary entity names or understand unrestricted prose**. It does learn when to compose the structured information implied by the input *sentences* into an output sequence. The WordNet records provide the underlying valid chains; the grammar variations are synthetic.
- `native_episodic.cpp` is carried forward from Round 035, implementing 256 128-byte heads + 32,640 physical 256-byte mirror-pairs = **8,388,608 bytes**. Individual ASCII source-prompt bytes are encoded as anchored triangle circulations and survive additive vertex-gauge changes. All native source reads for the generation audit go through the **physical root**, not a standalone numpy implementation. An ordinary indexed copy achieves the same retrieval result.
- `audit.py`: checks both neural semantic correctness and that native root gauge perturbations leave the generated output unchanged. Raw single-edge and conventional indexed-storage controls are also evaluated.
- `stress.py`: challenges the baseline with contradictory word choice and unseen four-hop depths. These are important failed tests.
- `robust.py`: adds same-slot contradictory premises (`opposes`) during training and re-trains the end-to-end sequence generator; no explicit label for connection equality is supplied to the model.

## Reproduce

Python 3.10+, `pip install numpy torch`; C++20 compiler (g++ or clang++). `wn3.1.dict.zip` is a **required input**; this package does not duplicate it.

```bash
python train.py /path/to/wn3.1.dict.zip
python audit.py /path/to/wn3.1.dict.zip
python stress.py
python robust.py /path/to/wn3.1.dict.zip
```

`results.json`, `audit_results.json`, `stress_results.json`, `robust_results.json` record the executed evaluations; the `.pt` files are actual trained checkpoints. Tests are deterministic conditional on compatible runtime/libraries. `SHA256SUMS` provides per-artifact checksums.

## Main measured results

### Original model: 2 and 3 hop, held-out source children

- Actual WordNet synsets parsed: **81,499** after name filters. Train **17,986** examples, validation **2,998**, test **2,998**.
- Decoder parameter count **38,268**; train steps **145** per seed.
- Seed 17: semantic accuracy **50.00%**, model collapsed to constant abstention; honest training failure.
- Seed 29: semantic accuracy **100%** on the same *restricted*, role-normalized, held-out test. Exact answer-string agreement **67.01%** because three alternative grammatical positive phrasings are semantically accepted but only one is the reference.
- Held-out next-token teacher-forced loss: seed 17 = **0.20646 nats/token**, seed 29 = **0.09348 nats/token**.

### Native evidence and perturbation

- 90 held-out prompts encoded bytewise as anchored triangle circulations in the 8 MiB root. Following a large random vertex gauge update (standard deviation 80), **90/90** prompts were decoded exactly, **90/90** neural answers unchanged, **0/90** prompts survived if read from a single raw edge. The ordinary conventional memory (no gauge corruption) also recovered all prompts and retained outputs. This demonstrates **exact geometric additive-gauge invariance**, not a unique advantage over all other correct invariant memories.

### Stress results on seed 29, before adversarial retraining

- In-distribution source-disjoint examples: **300/300 semantic correctness**.
- Semantically contradictory premise verbs (`opposes`, not seen during training): **0/300 correct abstentions**.
- Unseen paraphrase wording: **300/300** under this specific substitution, though the model may treat out-of-vocabulary words indiscriminately; **not evidence of authentic paraphrase understanding**.
- Four-hop extrapolation (never trained): **0/150** correct. That is a material generalization failure.

### Robust training track

- Adds 1 adversarial negative per positive, where entity slots align but one predicate is `opposes`, with validation and test groups disjoint by source child.
- Trained for **270 steps** (seed 29), same encoder–attention–GRU style, 29-word normalized vocabulary.
- Held-out semantic accuracies: **1,499/1,499** true connected chains, **1,499/1,499** disconnected chains, **1,499/1,499** relation-conflict chains. These are synthetic premise wording tasks derived from genuine WordNet paths.
- Generalization to arbitrary prose, 4+ hops, non-English text, free-written explanations, and naturally occurring contradictions remains unproven.

## What the geometry actually contributes

For each symbol `v` stored at an anchored cycle `(0,a,b)`, a vertex-gauge change `e(i,j) → e(i,j) + g_i - g_j` leaves `e(0,a)+e(a,b)-e(0,b)=v` unchanged. Native retrieval works after random coordinate shifts and the autoregressive decoder sees the same linguistic input. This property is mathematically guaranteed and could be valuable when independently trained memories use inconsistent local reference coordinates. **An ordinary implementation of the same gauge-invariant graph algorithm would share the property.**

## Next rigorous experiment

1. Train a learned unsupervised or weakly supervised **relation writer** from actual unnormalized WordNet gloss sentences, with uncertain evidence and provenance, not synthetic slot statements.
2. Feed the decoder **multiple source texts after they have been written to native pair/hyperedge memory**, and backpropagate losses to trainable cycle-valued stores (R030/R031 paths).
3. Move from a 28–29 token synthetic vocabulary to a subword or byte-level pretrained decoder, compare held-out bits per byte and free-generation quality with identical regular-memory baselines.
4. Test genuine extrapolation in path length and held-out relation types, and report failures even if in-distribution accuracy is perfect.
