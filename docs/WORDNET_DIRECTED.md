# WordNet directions, topology and typed operators

This integrates the Round 008/009 WordNet results without adopting their
conclusions wholesale.

## What Round 008/009 establish

- Definition/gloss retrieval on WordNet broad noun classes: ~71% final-test
  blend in a controlled setup. Not free-form definition understanding or
  sense identification.
- Directional relation learning: held-out child synsets, same-category false
  parents.
  - symmetric AUC mean **0.6968**
  - directed AUC mean **0.7507**
  - improvement **+0.0538** across three seeds
  - correct orientation ~68–73%
  - paired bootstrap interval ~+0.030 to +0.068 AUC (first seed)
- The model is conventional TF-IDF + SVD + logistic regression; it does not
  establish a cycle or non-Abelian advantage.
- The split is transductive on gloss text; not zero-shot new-word evaluation.

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
- `nanlmind_hermes/dictionary_universe.py`: dedicated HW4 universe, sparse
  synset/word addresses, HW4:0 ↔ NaNNet ↔ HW2:0, bidirectional.
- `nanlmind_hermes/relation_operators.py`: 2×2 SU(2) typed relation operators;
  composition order matters; closed-loop trace is vertex-gauge invariant.
- `nanlmind_hermes/cycle_space.py`: exact gradient/cycle decomposition and
  fundamental triangles.

## Next decisive experiment

Train **matrix-valued relationship operators** from WordNet; enforce cycle
consistency through loop products; compare against a matched model with the
cycle loss disabled. Test on completely unseen synsets and a definition
decoder. Do not claim an advantage until it beats the strong conventional
baseline (not naive pairs).

## Honest boundaries

- Directional gain is real but small and conventional; not proof of a new
  mechanism.
- WordNet edges are typed and directed; scalar antisymmetric phases discard
  semantics.
- Cycle rank is sparse; using the dense NaN substrate is a design choice, not
  a consequence of WordNet.
- No end-to-end generative NaN language model yet.
