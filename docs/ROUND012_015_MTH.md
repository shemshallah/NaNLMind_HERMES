# Rounds 012–015: what they establish and how they map to MTH

These rounds do not demonstrate physical quantum behavior or a native MTH
implementation. They are useful experimental evidence about the **substrate**
that MTH/Hermes already exposes: gauge reference, cycle memory, typed relations
and Robust joint estimation.

## Results

### Round 012 — gauge-flat robust recovery

A zero-curvature field has only 255 independent dimensions. The two-hop median
recovered the exact field through 40% independently corrupted edges, then
degraded gradually. This is classical robust estimation.

- Our 256-node reproduction:
  - 40% corruption: median MAE **0.0**
  - 60%: **0.0021** (external 0.00085)
  - 80%: **0.0978** (external 0.09682)
  - 10% + Gaussian σ=0.05: median **0.00556** (external 0.00541)
  - non-flat control: median MAE **0.7930** (external 0.7877)
- It does **not** recover the 32,385-dimensional non-flat cycle space.
- The non-flat memory-bank experiment retrieved patterns through 60% corruption,
  but the conventional gauge-aligned least-squares control was at least as good.
  No cycle-specific advantage was established.
- WordNet exact ancestry: 2,000/2,000 multi-hop vs 50% direct-edge-only. Our
  reproduction matches: 1,000 positives + 1,000 negatives, multi-hop **1.0**,
  direct **0.5**.

### Round 013 — semantic cycle code under gauge and erasure

A 16-dimensional gloss embedding was encoded in a random divergence-free cycle
basis. Gauge fitting recovered the embedding under erasure, but ordinary least
squares was not robust to gross spikes. The matched ordinary real linear code
performed identically. The codebook was random, not learned; the gloss model was
transductive.

Our `cycle_codec` implementation reproduces the basic mechanism and adds a
blind **joint robust decoder**:

```
y[i,j] = (Q z)[i,j] + phi[j] - phi[i] + outlier
```

The cycle coefficients and all gauge coordinates are fitted together with
Huber IRLS. On synthetic K32 data:

| erasure | spikes | gauge LS cosine | joint robust cosine |
|---:|---:|---:|---:|
| 0% | 0% | 1.000 | 1.000 |
| 0% | 10% | 0.329 | 0.859 |
| 50% | 0% | 0.997 | 1.000 |
| 50% | 10% | 0.410 | 0.776 |
| 90% | 0% | 0.965 | 1.000 |
| 90% | 10% | 0.001 | 0.075 |

This is a synthetic codec result, not learned semantics. At 90% erasure plus
spikes even the joint decoder fails; the external 014 result at 90%/25% used a
different scale/noise setting.

### Round 014 — blind joint gauge + supervised semantic recovery

- 4,500 WordNet glosses for fitting, 16 held-out glosses × 2 seeds per condition.
- 12D supervised latent meanings encoded into 256 nodes.
- At 90% erasure + 25% corruption among remaining edges:
  - cycle ordinary LS: cosine **0.9394**, L2 0.5713
  - cycle joint Huber: cosine **0.9989**, L2 0.0660
  - ordinary random linear code with same decoder: cosine **0.9989**, L2 0.0662
- Category accuracy was **lower** for the robust decoder (0.375 vs 0.438 LS) at
  that condition. Lower latent error is not automatically better semantics.
- The codebook was not learned, and the ordinary code matched the cycle code.
  No NaN- or cycle-specific advantage is established.

### Round 015 — trainable relation operators and contextuality

| variant | direct AUC | 2-hop AUC | implication violation |
|---|---:|---:|---:|
| symmetric | **0.69237** | **0.69437** | 0.010001 |
| ordered, no constraint | 0.66124 | 0.69331 | 0.000750 |
| ordered + transitivity | 0.66122 | 0.69339 | 0.000714 |

The symmetric model is stronger on direct links. The transitivity penalty barely
changes accuracy. The definition-feature ablation drops the symmetric model from
0.69258 to 0.54472 (zeroed) / 0.54226 (shuffled), showing reliance on the
definition feature, not causal understanding.

The Peres–Mermin square verifies the textbook quantum algebra (contextual
witness 6 vs classical bound 4). It is a numerical check, not a physical
experiment. Triangle sums remain gauge-invariant on K256.

## Mapping to MTH

The convergence is structural, not an implementation of the full MTH stack.

| external result | Hermes/MTH counterpart | status |
|---|---|---|
| 256 vertices, 32,640 pairs, 32,385 cycles | `cycle_space.py`, K256 cycle basis | exact shared substrate |
| gauge potential + cycle/semantic field | `gauge_memory.py`, `cycle_state.py`, `cycle_codec.py` | implemented |
| blind joint robust recovery | `cycle_codec.joint_decode` | implemented, synthetic/small-scale |
| ordinary random-code control | `cycle_codec.random_linear_basis` | implemented as a control |
| typed WordNet graph and ancestry | `wordnet_typed.py`, `wordnet_reasoning.py`, L5–L7 | real graph, not a learned LM |
| ordered vs symmetric relation head | `MTHScorer`/L7 typed relation scoring | partial correspondence |
| transitivity penalty | L7 implication/frame consistency | shallow correspondence, not full MTH |
| Peres–Mermin contextuality | not part of MTH | separate quantum algebra; no physical claim |
| native NaN16 MTH | `MTHLayerStack`, NaNROUTER contracts | external rounds do **not** test this |

MTH L0–L2 are the Boolean/nibble/byte substrate; L3–L6 are phonetic, morpheme,
lexical and completion layers; L7–L9 are sentence, paragraph and narrative
relations. The external experiments cover a gauge/cycle memory substrate and a
shallow relation layer. They do not exercise the full L0–L9 curriculum, native
NaN16 kernels, or a generative language model.

So “nearly exactly MTH” is fair as a **structural convergence**: independent
experiments rediscovered gauge + cycle decomposition, robust joint estimation,
typed relation limits, and gauge-invariant loops. It is not independent
validation of MTH's specific Boolean layers, native arithmetic, or language
ability.

## Next

1. Learn the codebook/encoder jointly with the robust decoder; do not use a
   random or transductive code.
2. Compare cycle code, ordinary random linear code, pairwise graph model, and a
   parameter-matched neural baseline on fully unseen synsets.
3. Add a real definition-to-answer decoder and test new-word inference.
4. Integrate the joint robust decoder with the existing `CycleState` and HW4
   dictionary universe; keep the gauge reference recoverable while semantic
   cycle content is learned.
5. Do not claim quantum behavior from numerical contextuality or loop-trace
   checks.
