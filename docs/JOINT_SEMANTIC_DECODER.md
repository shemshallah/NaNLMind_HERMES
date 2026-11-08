# State-conditioned answer decoder

A controlled yes/no question is asked about a novel term linked to a
unique WordNet parent. Train/validation/test parents are disjoint.
All variants use the same linear ridge decoder and feature width.

- noun synsets: **82,192**
- eligible parents: **5,000**
- test questions: **1,500**

| state | threshold | accuracy | positive | negative | hallucination |
|---|---:|---:|---:|---:|---:|
| ordinary | 0.487 | 0.7033 | 0.7320 | 0.6747 | 0.3253 |
| cycle | 0.524 | 0.6953 | 0.6373 | 0.7533 | 0.2467 |
| none | 0.519 | 0.6847 | 0.6240 | 0.7453 | 0.2547 |

## Boundary

- The symbolic acquisition engine can answer this controlled task exactly;
  this decoder is a learned control, not a language generator.
- The cycle state is a projection of the same ancestry edges, not an
  end-to-end trained semantic field.
- Only one deterministic split and target sample was evaluated; the
  differences are not accompanied by a multi-seed confidence interval.
- No state/Cycle-specific advantage is assumed; the ordinary and no-state
  rows are controls.
