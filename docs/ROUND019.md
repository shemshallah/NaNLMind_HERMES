# Round 019: controlled definition acquisition and cycle projection

Round 019 demonstrates a narrow but real capability: a newly supplied term can
be attached to a uniquely resolved WordNet parent and inherit its taxonomy
without gradient training. It also demonstrates exact Hodge/cycle projection.
Neither result is unrestricted language understanding.

## External results

- 82,192 noun synsets; 117,953 distinct noun labels.
- 400/400 controlled multi-hop inferences; mean chain length 2.59.
- 398/398 unrelated-category probes abstained rather than guessing.
- Cycle audit: divergence ~2e-15; projected gauge error ~1.4e-15;
  96 encoded coordinates recovered exactly from triangle circulations.
- Definitions are constrained to explicit forms such as “a kind of X that …”.
- The parent sense is selected only when exactly one candidate exists.

## Hermes implementation

`nanlmind_hermes/semantic_acquisition.py` adds:

- conservative `possible_heads` parsing for “a kind/type/species/form/variety
  of X …” and “is a/means a X …”;
- one-sense-only head resolution;
- persistent JSON concept store with atomic replacement;
- inherited-category `ask` returning yes/unknown plus evidence;
- grounded `explain` output;
- explicit abstention on missing/ambiguous evidence.

`scripts/semantic_acquisition_bench.py` reproduces the controlled test on the
real WordNet archive:

| measure | result |
|---|---:|
| noun synsets | 82,192 |
| unique noun labels | 117,953 |
| eligible new terms | 400 |
| head resolution | 0.9975 |
| multi-hop inference | 0.9975 |
| unrelated-category abstentions | 398/398 |
| mean inferred hops | 2.60 |

One of the 400 eligible examples did not resolve under our stricter
one-sense rule; the external run selected a cleaner 400. This is a selection
difference, not evidence of a stronger engine.

The cycle audit matches the external numbers:

- encoded coordinates: 96
- max gauge flux error: ~8.9e-16
- max encoded recovery error: 0
- max cycle divergence: ~2.0e-15
- max projected gauge error: ~1.4e-15

## What this proves

- Persistent symbolic acquisition works when the parent is explicit and unique.
- Graph ancestry can provide one-, two- and three-link inheritance.
- Hodge projection and triangle coordinates are numerically exact.
- Abstention can avoid inventing an unsupported classification.

## What it does not prove

- It is not parsing arbitrary English definitions.
- It does not resolve polysemy or select a sense from context.
- It does not compare as an accuracy number against held-out neural classifiers.
- It does not put the learned state into a differentiable language decoder.
- It does not establish a cycle-specific or quantum learning advantage.

## Next

1. Train a typed edge operator that writes the new definition into a cycle state.
2. Condition an autoregressive decoder on that state and on retrieved glosses.
3. Evaluate novel-definition one/two-hop QA, abstention, calibration and
   hallucination against retrieval-only, ordinary-graph and matched neural
   controls.
4. Separate the physical quantum/OpenQASM track; no QPU result exists.
