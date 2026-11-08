# Gauge-aware relational memory

This is the strongest result from the NaN cycle-memory experiments, integrated
into the Hermes architecture as a relational memory/readout layer.

## Why it matters

A pair observation can be contaminated by an arbitrary vertex-gauge field
`u_j − u_i`. A naive edge matcher treats that as real signal. A gauge-aware
matcher solves for the nuisance vertex potential and compares the residual
cycle structure. This is a **strong control**: it uses the same observed edges
and does not create information.

## Method

- Patterns are divergence-free edge fields on K256 (cycle space).
- A query is a stored pattern plus a vertex gauge plus edge noise.
- For each candidate prototype and the observed edge mask:
  1. compute the edge differences `d_e = query_e − prototype_e`;
  2. solve ridge-regularized least squares for the vertex potential `u`
     minimizing `Σ_observed (d_e − (u_j − u_i))²`;
  3. score the residual.
- Naive matching compares `d_e` directly.
- Triangle matching requires all three edges of a sampled triangle to survive;
  under heavy erasure almost no complete triangles remain, which is why it
  collapses while gauge-aware matching does not.

## Implementation

- `nanlmind_hermes/gauge_memory.py`:
  - `project_cycle`, `gradient`, `gauge_residual`, `naive_residual`,
    `retrieve`.
- `tests/test_gauge_memory.py`:
  - cycle patterns have zero divergence;
  - a pure vertex gauge is removed by the gauge fit but inflates naive distance;
  - gauge retrieval identifies the correct prototype under contamination and
    erasure.

## Relation to the external experiments

Round 002 reported 100% gauge-aware retrieval at 99% edge erasure versus
~28% naive pair matching; sampled triangles collapsed to 0% at 99%. The key
conclusion is not that triangles create information; it is that vertex-gauge
handling extracts relational information a naive matcher mishandles.

## Boundaries

- Supervised retrieval from previously stored synthetic patterns, not
  autonomous learning.
- No new information capacity; the gauge-aware model is the strong baseline.
- Classical linear algebra, not quantum behavior.
- The next experiment must compare persistent/phase dynamics against this
  gauge-aware baseline, not against naive pairs.

## Where it fits

- L7/L8 relational memory: store concept/word relation patterns as cycle
  signatures; retrieve under partial/corrupt observations with gauge fitting.
- Head population remains the sequence predictor; the gauge memory is a
  memory/readout layer, not a language model.
- Cycle state persistence and phase dynamics are the next track, using the
  32,385-dimensional cycle basis sparsely.
