# WordNet directions, topology and typed operators

This integrates the Round 008/009 WordNet results without adopting their
conclusions wholesale.

## What Round 008/009/010 establish

- Definition/gloss retrieval on WordNet broad noun classes: ~71% final-test
  blend in a controlled setup. Not free-form definition understanding or
  sense identification.
- Round 009 reported directional relation learning with held-out child
  synsets and same-category false parents: symmetric AUC mean **0.6968**,
  directed mean **0.7507**, improvement **+0.0538** across three seeds.
- **Round 010's equal-parameter replication did not reproduce that advantage.**
  Six seeds, 1,041 trainable parameters, ~9,744 train / ~3,248 held-out pairs:
  cosine 0.6512, symmetric 0.6657, ordered 0.6646. The ordered model was
  **0.0011 AUC below symmetric**; positive direction was preferred only 59.4%
  of the time. The Round 009 gain is therefore not robust to model design and
  is not evidence for non-Abelian or cycle learning.
- Both are conventional TF-IDF/linear models; the split is transductive on
  gloss text. No zero-shot new-word evaluation yet.
- Round 011's phase memory establishes gauge-invariant retrieval (100% vs
  41.37% naive) but is exactly reproducible with real sine/cosine channels and
  degrades under independent phase noise. It is a memory mechanism, not a
  quantum effect.

## WordNet topology audit

| graph | vertices | undirected edges | cycle rank |
|---|---:|---:|---:|
| WordNet nouns | 82,192 | 84,505 | **2,314** |
| WordNet verbs | 13,789 | 13,256 | **12** |
| All synsets | 117,791 | 97,761 | **2,326** |
| Complete NaN | 256 | 32,640 | **32,385** |

The full audit reproduces the reported numbers exactly. WordNet is mostly
tree-like; the complete NaN graph is the latent computational substrate. Keep
three things separate:

1. **Which relations are known** — WordNet edges.
2. **Which computational relations are available** — the NaN pair/cycle space.
3. **How meanings compose** — a learned typed operator.

An undirected taxonomy loop is not automatically a contradiction; cycle rank is
not entanglement or semantic quality.

## Implemented in Hermes

- `scripts/wordnet_graph.py`: real WordNet synsets/lemmas/hypernyms; definition
  inheritance probes; HW4:0 route metadata.
- `scripts/wordnet_topology.py`: all-POS graph audit and cycle rank.
- `scripts/wordnet_typed.py`: 117,791 synsets, 378,203 typed pointer edges,
  glosses/examples, 26 relation types; see `docs/WORDNET_TYPED.md`.
- `nanlmind_hermes/dictionary_universe.py`: dedicated HW4 universe, sparse
  synset/word addresses, typed relations, structured definitions, new-word
  leaf insertion, HW4:0 ↔ NaNNet ↔ HW2:0, bidirectional.
- `nanlmind_hermes/relation_operators.py`: 2×2 SU(2) typed relation operators;
  composition order matters; closed-loop trace is vertex-gauge invariant.
- `nanlmind_hermes/phase_memory.py`: gauge-invariant phase retrieval and exact
  real-channel equivalent; see `docs/PHASE_MEMORY.md`.
- `nanlmind_hermes/cycle_state.py`: persistent edge state projected onto the
  divergence-free cycle subspace after every update.
- `nanlmind_hermes/cycle_space.py`: exact gradient/cycle decomposition and
  fundamental triangles.

## Next decisive experiment

Train **matrix-valued relationship operators on the typed graph**; enforce
composition/cycle consistency only where relation semantics justify it
(subtype transitivity, diamonds), never on antonym or derivation cycles.
Compare against a matched model with the constraint disabled, evaluate on
completely unseen synsets, and add a definition decoder. Do not claim an
advantage until it beats the strong conventional baseline (not naive pairs).
The Round 010 replication raises the bar: the ordered model must beat the
equal-parameter symmetric control, not merely cosine.

## Honest boundaries

- Directional gain is real but small and conventional; not proof of a new
  mechanism.
- WordNet edges are typed and directed; scalar antisymmetric phases discard
  semantics.
- Cycle rank is sparse; using the dense NaN substrate is a design choice, not
  a consequence of WordNet.
- No end-to-end generative NaN language model yet.
