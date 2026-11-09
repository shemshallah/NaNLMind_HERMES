# NaN Recursive Learning — Round 042

## Deliverable: one learning, thinking, talking universe

A practical, **restricted-domain neural conceptual dialogue** implementation: a trained relation-phrase writer reads prose with supplied entity boundaries; a custom differentiable bridge writes and reads physical float32 mirror-pair triangles inside a fixed 8,388,608-byte C++20 object; a trainable propagation network combines evidence; and an autoregressive word decoder learns to generate responses conditioned on the **computed memory**, not a pre-supplied answer-status token.

This round adds a usable, **persistent** CLI with 255 locally named cognitive heads plus anchor position 0, new-phrase fine-tuning with rehearsal, and independent support/refutation cycle lanes. The four-state neural speaker learns responses for support, refutation, coexistence of both, and missing evidence.

**Important boundary:** this is not yet a fluent, open-domain language model. The primary learned word decoder has 54 function-word/role tokens plus arbitrary names copied from concept slots. The four-state speaker is also trained on a bounded grammar. Entity boundaries and grammatical relation forms are identified by a constrained parser in interactive mode; WordNet child IDs (not all graph nodes) are split between training and testing. Geometry has not outperformed a conventional matched-memory model on unperturbed inputs.

## Fastest way to use it

Requirements: Linux with `g++ -std=c++20`, Python 3.10+, `torch`, `numpy`. No network access is required with included checkpoints.

```sh
python test_release.py
python converse.py --model adapted_model_17.pt --root sample_universe.bin
```

Once inside the prompt, you can type natural statements/questions without command prefixes, or use `teach`/`ask` explicitly:

```text
nan> quokka is subsumed by mammal
nan> mammal is a kind of animal
nan> what is a quokka?
nan> why is quokka a kind of animal?
nan> quokka is not a kind of animal
nan> is quokka a kind of animal?
nan> gauge
nan> is quokka a kind of animal?
nan> save
nan> quit
```

`sample_universe.bin` and `sample_universe.json` are a matched demonstration snapshot. To start with an empty universe, choose a NEW, nonexistent root path such as `--root local_universe.bin`.

New relationship wording can be learned with **supervised few-shot adaptation**, as demonstrated by:

```sh
python fewshot_writer.py
python retention_audit.py
```

The checkpoint `adapted_model_17.pt` knows three extra phrases: `is subsumed by` (forward), `comprises` (inverse), and `conflicts with` (no taxonomic implication). This is explicit supervised adaptation with rehearsal, not spontaneous acquisition of synonyms from arbitrary chat.

## Major benchmarks (three trained seeds: 17, 29, 43)

| Experiment | Correct | Interpretation |
|---|---:|---|
| WordNet child-split, two-to-four-hop QA | 540 / 540 | New child concepts in a controlled relation grammar |
| Identical tests after gauge drift | 540 / 540 | Triangle circulation preserved inputs and outputs |
| Five-hop tests, beyond two-to-four-hop training | 525 / 540 | Longer-than-trained path inference; 15 failures were broken-chain negatives |
| Same-weights, conventional signed-adjacency bridge | 540 / 540 | Geometry does not win on clean relational prediction |
| Independently retrained conventional bridge | 540 / 540 | Equal architecture and training budget; likewise ties |
| Native physical deletion of necessary pair | 240 / 240 abstentions | Causal: deleting evidence changes generated answers |
| Native pair restoration after deletion | 240 / 240 restored | Causal state restoration |
| Gauge-shifted responses in separate causal audit | 480 / 480 identical | Across 160 questions per seed |
| Permuted node indices in causal audit | 480 / 480 identical | Local symbol/address permutation robust |
| Neural four-state support/refute/both/unknown generation | 480 / 480 | A *second* network, conditioned on two native evidence lanes |
| Four-state responses after gauge drift | 480 / 480 identical | Support and refute channels separately invariant |
| Original writer phrase classes before/after few-shot adaptation | 16 / 16 | No observed class forgetting in this tiny repertoire |
| Newly taught phrase classes after adaptation | 3 / 3 (from 1 / 3) | Limited supervised language adaptation |
| New-phrase source-disjoint QA after writer adaptation | 180 / 180 | Trained answer decoder still works with new writer |

All perfect scores are for **designed, balanced episodes** based on real WordNet hypernym chains, not arbitrary user questions or broad language. The generated answer strings are not automatically supplied by the evaluator. Scoring checks both positive and negative grammatical meaning; the judge was corrected after the initial language-only ablation exposed a false positive in an earlier evaluator.

## Particularly important negative result: token-loss-only training

`language_only_audit.py` removes both writer-direction supervision and the auxiliary proof-reachability loss, retaining only teacher-forced language-token cross entropy. Across 1,000 optimization steps on seed 17, the model collapsed into **always abstaining**: 90/180 correct (0/90 supported, 90/90 unsupported). The original permissive evaluator wrongly counted answers containing "conclusion" as valid positive answers. Both the erroneous historical report and corrected report are preserved as an audit trail. This shows the token loss alone does not reliably bootstrap the latent compositional writer in this setup.

The main joint model includes language cross entropy plus auxiliary supervision for relation direction and reachable-conclusion classification. The *language-only part of the loss* still has a nonzero measured gradient to the prose writer (norms between 6.45e-6 and 3.42e-5 on trained checkpoints). That is a genuine gradient path, but not evidence that the model learns all reasoning from language alone.

## Mathematical construction

There are 256 address vertices, with anchor/gateway position 0 and 255 locally named cognitive head positions 1–255. The physical pair object stores 32,640 unordered pair cells. Among nonzero heads there are 32,385 direct pairs; the remaining 255 connect head 0 to those heads. The complete directed-edge cycle space has dimension 32,385.

For each independent float32 evidence layer `k`, a directed pair field `E[i,j,k]` is antisymmetric. An anchored triangle value is

`F[k;0,a,b] = E[k;0,a] + E[k;a,b] - E[k;0,b]`.

The additive gauge transformation `E'[i,j,k] = E[i,j,k] + g[i] - g[j]` leaves this circulation invariant algebraically. The native bridge writes values by setting the appropriate physical edge coefficient, rereads physical circulation, and exposes a straight-through analytical Jacobian to the PyTorch graph (identity for exact write/read; expected small float32 deviations). `NativeCycle.backward` is not a separately learned backpropagator; it implements the known derivative of the native write/read interface.

The support lane is channel 0; explicit taxonomic refutations are written into channel 1. A supporting path and an explicit refutation can coexist, giving a paraconsistent `both` state. The neural four-state **speaker** learns the surface forms for all four outcomes; the refutation marker itself is provided by controlled fact parsing. This is not yet a full sNaN/qNaN/sign-bit instruction implementation and does not use a physical quantum device.

## Files

- `dynamic_joint.py`: primary native differentiable graph writer, propagation, decoder, WordNet task builder, main training and evaluation.
- `native_episode.cpp`: 8 MiB pair-root with two independently gauge-invariant float lanes, pair identity checks, and experimental full-root snapshots.
- `converse.py`: persistent conversation, concept addressing, evidence retrieval and provenance, learned phrase support and trained four-valued speaker.
- `four_state_language.py`: learned autoregressive support/refutation/both/unknown speaker.
- `fewshot_writer.py`, `retention_audit.py`: supervised new-phrase adaptation, old-phrase rehearsal, held-out QA.
- `causal_audit.py`: physical pair knockout, restoration, node relabeling, writer-language gradient audit.
- `matched_dense.py`: identically trained conventional-memory control.
- `language_only_audit.py`: **failed** token-loss-only experiment.
- `test_release.py`: fast integration test (does not retrain).
- `corpus.jsonl`: WordNet-derived entries and source splits.
- `model_*.pt`, `dense_*.pt`, `four_speaker_*.pt`, `adapted_model_17.pt`, `language_only_model_17.pt`: reproducibility checkpoints.
- `results.json`, `causal_results.json`, `retention_results.json`, `matched_dense_results.json`, `four_speaker_*.json`, `fewshot_writer_17.json`, `language_only_17.json`: numerical evidence.
- `RESEARCH_LOG.md`, `THEORY.md`, `MANIFEST_SHA256.json`: provenance, theorem statement, file hashes.

## Reproduction

```sh
python dynamic_joint.py --steps 650 --seeds 17 29 43 --test 180 --evalevery 130
python matched_dense.py
python causal_audit.py
python four_state_language.py            # speaker seed 17
python fewshot_writer.py
python retention_audit.py
python test_release.py
```

To train other four-state speakers, import `run` and call `run(29,600)` or `run(43,600)`; all checkpoint files are included, so retraining is not needed for the demo. Running `dynamic_joint.py` overwrites `results.json` from scratch, so back up original results if experimenting with smaller training budgets.

### Scientific scope

These experiments demonstrate a new working *implementation*, not a proven statistical advantage over modern language models. A classical conventional graph with the same invariant equations can tie these outputs; native float encoding alone has no established advantage. A major research milestone still remains: fluent open-domain generation, arbitrary prose parsing without supplied entity boundaries, language-only joint acquisition, well-matched natural-corpus evaluation, and evidence of a geometric inductive bias beyond artificially applied gauge drift.

The 8 MiB root is stable in address space while allocated; this implementation does **not** invoke `mlock`. Snapshots are experimental host-layout images with a magic prefix and an associated SHA-256 in the interactive metadata; they do not implement multiwriter transactions or cryptographic security. WordNet source content requires appropriate WordNet attribution/license compliance.
