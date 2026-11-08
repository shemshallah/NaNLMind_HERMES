# Coherent spectral architecture: opencode1 + MTH + Hermes

This is the architecture that joins the wave/void model (`opencode1.txt`), the
MTH L0–L9 layers, the 256-head universe, and the Hermes local runner. The
mathematical skeleton is usable; the quantum/noosphere language in the source
is treated as metaphor, not as a physical claim.

## The core correspondence

| wave model | Hermes/MTH |
|---|---|
| 256×256 ordered amplitude matrix `A[m][n]` | 65,536 address registers |
| symmetric part `(A+Aᵀ)/2` | 32,640 unordered pair cells + 256 fixed points |
| antisymmetric part `(A−Aᵀ)/2` | directed `s+` / `s−` rails |
| diagonal `A[m][m]` | 256 heads / self / polarity |
| center void | head 0 = NaNNet + polarity/void anchor |
| barrier ring, 256 antinodes | 256 heads on the relation boundary |
| palindrome `A[m][255−m]` | native mirror pairs / `reverse16` |
| fold to two 128×128 blocks | centrosymmetric fast decomposition |
| projection/measurement | head-population collapse / votes |

## Layer stack

```
L0 bit       16 Boolean functions (root 4×4 class-pair table)
L1 nibble    composition / phase coordinates
L2 byte      byte patterns -> hard-mapped letters
L3 phonetic  letter/feature superposition
L4 morpheme  stems, affixes
L5 lexical   head-population letter distributions
L6 word_completion  prefix -> next-letter collapse
L7 sentence  typed relations; low eigenmodes of L = D - A + V
L8 paragraph spectral resonance / coherence
L9 narrative phase-ordered causal chains
```

- **Letters are hard-mapped** to ASCII; no learning of letter identity.
- **Words are hyperedges** over letter leaves; the dictionary is not stored,
  only the superposition of head distributions.
- **Heads hold all letters**, not one letter each. A context activates a
  population; the population collapses to the next-letter distribution.
- **Head 0 is the void**: NaNNet interface and polarity/zero anchor. It must be
  a calibrated boundary term, not a mass that dominates every collapse.

## Wave operator

The implementation is `nanlmind_hermes/spectral.py`:

1. `A` = head-to-head Gram matrix of letter distributions.
2. Symmetrize; centrosymmetrize with the reversal `J`.
3. Fold `A` into `P = B + CJ` and `M = B − CJ` (two 128×128 blocks).
4. Build the graph Laplacian `L = D − A`; add the void potential
   `V(r) = λ/r²` on a 16×16 head grid.
5. Take **low modes** of `L + V` (the correct wave equation); the uniform zero
   mode is trivial and excluded from tunneling.
6. Use eigenmodes `P` and strengths `Λ` as the L7 relation basis; spectral
   overlap is L8; phase order is L9.

A positive diagonal on the coupling matrix is not the barrier — it attracts the
top mode inward. The barrier belongs on the Laplacian and acts on low modes.

## Measured status (`docs/SPECTRAL_WAVE.md`)

- Full 256×256 and folded 128×128 spectra agree: fold error **0.0**; NumPy
  reference error **0.0**.
- Nonconstant low modes: inner (void) mass **0.001359**; with the void
  potential **0.000792** — a **41.7%** reduction. This is the measurable
  spectral leakage (“tunneling”), not a quantum measurement.
- The coupling matrix is currently **rank 1**: all heads carry nearly the same
  letter distribution. This is the bottleneck, not the eigenmath.
- All 256 heads participate; head 0 is the void/prior. With all heads active
  and head 0 uncalibrated, the collapse is dominated by the prior (`th` votes:
  end 141, e 20).

## Coherent build order

1. **Head diversification** — learn context routing/gating so heads carry
   different distributions; measure rank and entropy.
2. **A from the concept matrix** — use typed pair-cell relations as the
   coupling matrix, not only letter cosine similarity.
3. **Low-mode L7** — typed relation truth from eigenmode projections; void as
   boundary condition.
4. **Hermes demo** — chat/concept matrix/head population/long-chat evidence.
5. **Copy to CLOCKED, Android and runners** only after the Hermes demo passes:
   same chart/register formats, same spectral/void contracts, same validity
   gates.

## Honest boundaries

- “Tunneling” here means nonzero amplitude inside the void barrier; it is a
  spectral property, not quantum tunneling or consciousness.
- The current rank-1 head population is a real limitation; the eigenmode
  structure is only as rich as `A`.
- The source’s noosphere/uploaded-mind claims are not part of the
  implementation.
