# Definition-to-answer baseline

A WordNet definition prefix retrieves candidate synsets by IDF-weighted
token overlap. Ancestry answers use exact typed hypernym traversal. This
is a structured retrieval/graph decoder, not a generated language model.

- noun synsets available: **82,192**
- sampled synsets: **2,000**
- definition-prefix top-1 (same lexname pool): **0.9700**
- same-category top-1: **1.0000**
- ancestry positives: **300**
- ancestry same-lexname negatives: **300**
- multi-hop ancestry balanced accuracy: **0.9833**
- direct-edge-only balanced accuracy: **0.5000**

## New word

- **sparklebeak** linked to the real `penguin` synset
- definition: short-legged flightless birds of cold southern especially Antarctic regions having webbed feet and wings modified as flippers
- `is a bird`: **True** (one hop)
- `is a animal`: **True** (two hops)
- `is a vehicle`: **False** (negative control)

## Boundary

- Retrieval over supplied WordNet glosses is not learned prose parsing.
- The decoder does not generate free-form answers; it returns structured
  definitions, relation records and ancestry paths.
- The new word is attached by a synset link, not learned from arbitrary
  definition wording.
