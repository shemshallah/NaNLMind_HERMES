# NaN Recursive Learning — Round 041

## What this round demonstrates

A real (Linux) 8,388,608-byte native mirror-pair root stores directed edge evidence as anchored triangle circulations. A trained text-direction writer converts a restricted set of WordNet-derived prose patterns to facts. A **trained differentiable propagator** (only three scalar trainable parameters, applying repeated adjacency multiplication) evaluates path reachability from those facts without a supplied proof-status label. A separately trained word-level autoregressive network generates answers and copies unknown source/candidate entity names. Model weights, source corpus, native snapshots and all audit scripts are included.

This is a modular integration, not an end-to-end gradient-trained language reasoner: the writer, propagator, and generator are trained separately, and the generator still receives a predicted proof-status token from the propagator. It is a controlled language with provided entity boundaries; no claim of fully free-form fluent generation, unique NaN-bit advantage, or open-domain reasoning is made.

## Three-seed experimental evidence

| Experiment | Outcome | Source |
|---|---:|---|
| Learned prose writer patterns, 3 seeds | 0 mistakes across 2,200+ evaluated ingested sentences (restricted grammar) | `integrated_summary.json` |
| Old held-out queries, native vs conventional, pre-gauge | 660/660 vs 660/660 | `diff_pipeline_summary.json` |
| Old held-out queries after large vertex-gauge drift | 660/660 | `diff_pipeline_summary.json` |
| Old queries after new writes (retention) | 660/660 | `diff_pipeline_summary.json` |
| New queries after writes | 240/240 | `diff_pipeline_summary.json` |
| Real WordNet 4- and 5-hop chains (native and conventional) | 165/165 each | `deep_native_summary.json` |
| Delete indispensable edge; predicted abstention | 165/165 native and conventional | `deep_native_summary.json` |
| Restore edge; supported answer returns | 165/165 | `deep_native_summary.json` |
| Native causal deletion/restore on 3-hop queries | 240/240 on all intervention stages | `counterfactual_summary.json` |
| Tested 3-to-5-step synthetic graph transfer | 300 positive + 300 broken-negative per depth per seed | `depth_results.json` |
| Full native root snapshot reload | successful in three seeds | `diff_pipeline_summary.json` |

Numbers above concern deterministic/greedy generation under restricted grammatical and entity-role supervision. The conventional indexed-graph control **ties**, and the gauge invariant is an exact geometric construction also implementable without NaN bit patterns.

## What is learned and what is prescribed

- `role_train.py`: autoregressive gated word-level decoder trained via teacher-forced next-token loss. Decoder can emit lexical tokens and copy source/target names. Queries include marked source and target roles. A preliminary ablation (`ablation_results.json`) revealed this decoder relies on the proof-status token, and does not independently infer truth from prose.
- `diff_reasoner.py`: learned scalar edge strength, propagation readout scale, and threshold for multi-step adjacency propagation, trained on positive and negative three-hop WordNet-derived examples. It generalizes to four and five hops when given additional propagation steps; **the adjacency graph comes from parsed evidence, not an unconstrained learned interpretation**. This is a light parametric graph learner, not a deep relational neural network.
- `integrated_audit.py`: trains a lexical direction classifier on a finite list of templated paraphrases. Entity boundaries and the relationship grammar are controlled, not discovered from unrestricted text.
- `native_episode.cpp`: physical resident 8 MiB root, 256 heads and 32,640 mirror-pair records, with sign-reversed directional edge values, triangle cycle read/write, gauge shifts, and a same-platform snapshot interface. This is not a complete NaN16 guest execution engine.
- `differentiable_pipeline.py`: episode-indexed native graph readout fed into learned propagator, which supplies the proof-status input for the token generator. Episode membership lists are host metadata, not trained content-addressable retrieval.
- `counterfactual_audit.py`: causal edge intervention. Generate from a stored WordNet path, remove one essential native fact and regenerate; apply gauge drift; restore fact and regenerate. Repeat against conventional edges.
- `deep_native_audit.py`: finds genuinely longer hypernym paths in the bundled WordNet corpus, tests native four/five-hop generation and one-link-removal abstention against the conventional store. The model is *not* trained end-to-end on these exact paths.

## Reproduction

Tested on CPython 3.13, PyTorch, NumPy, scikit-learn, g++ with C++20. From this directory:

```bash
python -m py_compile *.py
python counterfactual_audit.py
python deep_native_audit.py
python differentiable_pipeline.py
python stress_depth.py
```

`integrated_audit.py` compiles the native shared library on first import if `libepisode.so` is absent. To rerun training (not needed for the evaluations):

```bash
python role_train.py --steps 280 --seeds 17 29 43
python diff_reasoner.py
```

The corpus `corpus.jsonl` is a compact parsed WordNet 3.1 source with explicit held-out split information. Reuse with source/license obligations of WordNet. Native snapshots are full 8 MiB images and intentionally retain only the current round; the .so build product is excluded from the portable package.

## Remaining scientific gaps

1. Train a genuine open-vocabulary, continuous-text autoregressive generator with the native memory **in the loss path**. Current generator produces restricted grammatical answers.
2. Learn unconstrained prose-to-relations; current input patterns and entity parsing are partially prescribed.
3. Learn which paths and hyperedges to attend to, instead of furnishing episode-specific edge lists through host metadata.
4. Compare to capacity- and information-matched attention/memory models on naturally occurring text; current controls show no unique geometry advantage.
5. For claims of continual *learning*, test additional gradient training and interference in learned weights; current retention mainly tests additional **memory writes**.
6. Improve snapshot format with transactionality, checksums, cross-platform encoding and concurrency safety.

The narrow verified result is a functioning learned multi-step reasoning -> token-generation interface backed by a gauge-invariant resident mirror-pair graph, with causal evidence interventions and long-depth generalization in a controlled benchmark.
