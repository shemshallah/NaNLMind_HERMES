# Round 018: lexical LM, persistent glossary, and the missing joint system

Round 018 crossed an engineering milestone and also produced an important null
result.

## External results

- A matched 64-hidden GRU pair trained on 9,600 WordNet glosses, validated on
  1,200 and tested on 1,200:
  - plain held-out perplexity **69.5251**, next-token accuracy **29.5691%**
  - cycle-transport held-out perplexity **69.5419**, next-token accuracy
    **29.5023%**
- The cycle variant is effectively tied with the ordinary control. The external
  archive itself says the transport ablation is not a matched retraining
  control.
- Definition classifier: **70.14%** on 1,373 held-out noun definitions.
- Nearest-gloss retrieval: **44.17%** lexicographer agreement on mixed-POS
  held-out glosses, not noun-only identity accuracy.
- Persistent lexical memory stores a supplied term and definition across
  processes. Recall is retrieval, not conceptual inference.
- Generated continuations are unreliable and contain grammar mistakes.
- The 64D orthogonal transport is not a learned 32,385-dimensional semantic
  field; the cycle-basis audit is separate.

## Hermes additions

### Persistent lexical memory

`nanlmind_hermes/lexical_memory.py` stores verbatim user definitions in an
atomic JSON file and reloads them across processes. It preserves the supplied
text, optional synset address and related records. It does not infer from the
wording.

### Definition-to-answer baseline

`nanlmind_hermes/definition_decoder.py` adds:

- IDF-weighted definition retrieval;
- structured `answer_what` records with definition, hypernyms and inheritance;
- exact typed `answer_is_a` with a shortest path;
- relation readout.

On a 2,000-synset WordNet sample (`docs/DEFINITION_QA.md`):

- definition-prefix top-1 in same-lexname pools: **0.97**
- same-category top-1: **1.00**
- multi-hop ancestry balanced accuracy: **0.9833**
- direct-edge-only balanced accuracy: **0.50**

The new term `sparklebeak` was attached to the real `penguin` synset:

- `is a bird`: true (one hop)
- `is a animal`: true (two hops)
- `is a vehicle`: false

This is graph acquisition from a synset link, not learning from arbitrary prose.

### Typed cycle D experiment

`scripts/wordnet_semantic_cycle.py` builds a 256-anchor quotient of real WordNet
nouns, assembles D from typed hypernym/meronym/attribute edges, and learns
relation weights on validation.

Result: the validation search selected **zero** relation weights. Graph
smoothing did not beat the base cluster labels. A fixed non-zero graph gave:

| model | test accuracy |
|---|---:|
| base cluster labels | ~0.2136 |
| fixed principal D | 0.2009 |
| fixed cycle-projected D | 0.2039 |
| fixed random graph | 0.1931 |
| ordinary linear ridge | **0.2600** |

The cycle projection is a linear isometry and adds no information. No
cycle-specific semantic advantage is established.

### Typed operators

`nanlmind_hermes/typed_cycle.py` provides a small typed relation-operator
registry: relation paths compose in order, reverse paths use the dagger, and
unknown relations get deterministic non-identity operators. This is a
prototype, not a trained edge-to-cycle language layer.

## What this means

The external Round 018 result and the Hermes experiments agree:

1. The cycle/LM interface can be built.
2. It is tied with the ordinary control.
3. Storing definitions and retrieving them is real engineering; it is not the
   same as learning composition or generating reliable prose.
4. The next decisive test is a **joint** system: learn typed edge operators,
   write their outcomes into cycle memory, condition a decoder on retrieved
   concepts, and evaluate novel-definition one- and two-hop questions with
   matched baselines.

## Next

1. Train typed edge operators on WordNet compositional supervision.
2. Condition a decoder on retrieved definitions, with retrieval-only and
   ordinary-transport controls.
3. Evaluate novel definitions with exact-match QA, calibration, hallucination
   checks and five seeds.
4. Keep the quantum/OpenQASM path separate; no physical advantage has been
   demonstrated.
