# Round 020: corpus-ready training and state-conditioned decoding

Round 020 moves from isolated experiments to a corpus/training interface. The
measured result is still a near-tie between the cycle topology and the ordinary
control.

## External result

- 2,400 WordNet-derived passages: 1,934 train / 215 validation / 251 test.
- Two 48-hidden recurrent models, 80 steps, **152,312 parameters** each.
- Plain: held-out perplexity **64.4739**, next-token accuracy **28.22%**
- Triangle-topology cycle transport: perplexity **64.3899**, accuracy **28.29%**
- One seed; the difference is not a meaningful cycle advantage.
- 48 triangle coordinates; 576 nonzero directed couplings; maximum gauge error
  **6.66e-16**.
- Vocabulary transfer: 1,400 old token entries preserved under expansion.
- Retrieval can return irrelevant passages; generated continuation remains weak.

## Hermes additions

### Deterministic corpus store

`nanlmind_hermes/corpus_memory.py`:

- recursive UTF-8 `.txt` ingestion;
- paragraph/window chunking with a token limit;
- exact SHA-256 deduplication;
- deterministic hash-based 80/10/10 train/validation/test split;
- JSON save/load with source paths and snippets;
- IDF-weighted token-overlap retrieval, explicitly not comprehension.

Ingesting the external Round 020 `wordnet_sample.txt` produces the same 2,400
passages, but our deterministic split is 1,927/218/255, not the external
1,934/215/251. The split rule is not claimed to match. This is the
corpus/evidence layer, not a neural language model.

### State-conditioned answer decoder

`scripts/joint_semantic_decoder.py` trains three linear ridge decoders on the
same controlled novel-term category question:

| state | accuracy | positive | negative | hallucination |
|---|---:|---:|---:|---:|
| ordinary ancestry | **0.7033** | 0.7320 | 0.6747 | 0.3253 |
| cycle-projected ancestry | 0.6953 | 0.6373 | 0.7533 | 0.2467 |
| no graph state | 0.6847 | 0.6240 | 0.7453 | 0.2547 |

The ordinary state has the best accuracy; the cycle projection trades positive
recall for fewer false positives. Neither supports a cycle-specific semantic
advantage. This is one deterministic parent split and target sample, without a
multi-seed confidence interval. The symbolic acquisition engine can answer this
controlled task exactly; the decoder is a learned control.

## Next

1. Train the decoder on actual prose-derived relations, not only built-in
   WordNet ancestry.
2. Condition an autoregressive decoder on the corpus/state and compare with a
   retrieval-only and an ordinary-transport control.
3. Evaluate new vocabulary retention, forgetting, calibration, hallucinations
   and held-out multi-hop questions over 5+ seeds.
4. Keep corpus retrieval and generation separate until evidence supports the
   joint claim.
