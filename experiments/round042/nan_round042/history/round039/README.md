# NaN Recursive Generative Language — Round 039

## Research status

**Real subword autoregressive free generation, with native gradient-updated cycle memory, but fluency is still poor and no language-model advantage over the matched dense-memory control was observed.**

This round corrects the earlier erroneous repeated-short-document windowing and evaluates the models on natural WordNet 3.1 definition text. Unlike Round 036/037, generation uses a learned 2,048-subword vocabulary and samples tokens directly, with no hardcoded answer phrasing or entity slots.

## Architecture

- Identical 96-channel GRU encoders, 2,048-subword heads, learned memory gates, and trainable 16x16 adapter coefficients.
- **Native:** Each adapter coefficient is a directed three-edge triangle circulation physically stored in a separate anchored triangle within the fixed 8,388,608-byte C++ root, and each gradient writes the corresponding pair lanes. Snapshot/restore and gauge perturbation use native functions.
- **Control:** Same neural network, same 16x16 degrees of freedom, ordinary PyTorch dense parameter tensor; same random seeds, training stream, and optimizer family. **Caveat:** native adapter uses plain gradient steps (`lr=0.02`) whereas dense control uses AdamW (`lr=0.002`), so training optimizers are not strictly matched. Equal representational capacity does not establish equal optimization power.
- Stream training replaces `np.resize`-generated repeated glosses with a single concatenated stream, with BOS/EOS boundary tokens between real definitions. This intentionally allows training windows across unrelated document boundaries; test scores are calculated on independent complete held-out definitions.
- Corpus: 25,000 WordNet training definitions, 1,500 validation and 1,500 test; BPE learned from the training set only. Two seeds, 240 optimizer steps, batch 8, 32-token contexts. Validation checks at steps 1/120/240, best epoch selected on validation.
- Every held-out test metric in `benchmark.json` is aggregated over the first 300 untouched test definitions (7,863 next-token targets), identical for all models.

## Results

| Model | Test perplexity (lower is better) | No memory | Shuffled memory | Test top-1 accuracy |
| --- | ---: | ---: | ---: | ---: |
| Dense control 17 | 251.845 | 263.653 | 272.506 | 0.12998 |
| Native cycle 17 | 255.613 | 259.791 | 260.785 | 0.12972 |
| Dense control 29 | 257.546 | 270.900 | 276.711 | 0.12603 |
| Native cycle 29 | 261.806 | 266.747 | 264.897 | 0.12578 |

The dense control is better by 3.8–4.3 perplexity points. This does NOT prove a geometric language-model quality benefit. The native memory's maximum drift after substantial additive vertex-gauge perturbation is below 1e-6 in each seed. This is an intended mathematical invariant; the plain control is not exposed to coordinate-frame disturbances because it has no such coordinates.

The earlier Round 038 experiments (`train_subword.py`, `train_generate.py` included as sources) ran 1,200 updates on the older windowing protocol and achieved lower perplexity, but the short-definition repetition error motivated the corrected experiment. Their weights are NOT used as initializers here.

Generated samples are in `samples.json` (model produces token by token, but is not reliably grammatical). This is a small architecture and training run; it cannot be claimed as a fluent LLM or general factual reasoner.

## Reproduction

Python 3.10+, `torch`, `numpy`, `sentencepiece`, `g++` C++20. Build the native root library:

```bash
g++ -O3 -std=c++20 -fPIC -shared native_cycle.cpp -o libnative_cycle.so
python train_stream.py --steps 240 --seeds 17 29 --eval-every 120 --batch 8 --context 32 --testdocs 300
python audit_all.py
```

The included `corpus.jsonl` contains WordNet-derived records and train/validation/test splits; `wordnet_bpe.model` contains the frozen train-derived vocabulary. Note the first command regenerates/checkpoints and may take several minutes on CPU. Train/validation/test share some general words but not identical synset source IDs. `results.json` stores seed 29 final in-loop samples because the per-run logger overwrote its top-level JSON on the second run; `benchmark.json` recomputes all four complete paired metrics from saved checkpoints. `train17.log` and `train29.log` preserve both runs.

Native `best_root_17.bin` and `best_root_29.bin` are unversioned host-layout experimental binary snapshots; do not load untrusted files or treat them as transactional durable database formats.

## Next research test

The current adapter is a 256-coefficient learned weight tensor represented via geometry, with no dynamic concept activation or typed evidence composition. Increasing the number of steps alone is not enough. A meaningful benefit requires **content-dependent sparse cycle state** and learned, differentiated memory writes that vary with input episodes. Fair baselines should have identical episode memory, copied evidence, and, separately, the same gauge-invariant feature treatment. Free-text perplexity, held-out multi-hop QA, retention, and performance under gauge drift must be reported independently.
