# WordNet semantic cycle D experiment

A 256-anchor quotient of real WordNet nouns; D is assembled from typed
hypernym, meronym and attribute edges. The cycle-projected graph is a
classical divergence-free projection, not a quantum state.

- noun synsets: **82,192**
- lexicographer classes: **26**
- anchors: **256**
- selected relation weights: `{'hypernym': 0.0, 'meronym': 0.0, 'attribute': 0.0}`

| model | validation | test |
|---|---:|---:|
| principal | 0.2076 | 0.2136 |
| cycle_projected | 0.2076 | 0.2136 |
| random_graph | 0.2076 | 0.2136 |
| ordinary linear ridge | — | 0.2600 |

The validation search selected **zero** relation weights: graph smoothing
did not beat the base cluster labels. A fixed non-zero graph gives a
second, non-cherry-picked view:

- fixed principal test: **0.2009**
- fixed cycle-projected test: **0.2039**
- fixed random-graph test: **0.1931**

## Boundary

- Labels are broad WordNet lexicographer classes, not deep semantics.
- The cycle projection is an isometric/linear operation; it adds no
  information and no cycle-specific advantage is claimed.
- This does not train a language decoder and does not test novel-definition
  reasoning.
- No physical quantum behavior is involved.
