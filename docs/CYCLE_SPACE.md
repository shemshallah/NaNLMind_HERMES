# Cycle space and nonuniform mirror-pair resonance

Complete 256-node graph: 32,640 edges; gradient dimension 255; cycle
dimension 32,385; triangles 2,763,520; independent fundamental fluxes
32,385. Classically verified, not quantum hardware.

| measure | uniform | nonuniform mirror |
|---|---:|---:|
| max_cycle_divergence | 0.0 | 3.861502235561494e-13 |
| gradient_cycle_orthogonality | 0.0 | 7.307127125599777e-12 |
| reconstruction_error | 0.0 | 2.482534153247273e-16 |
| triangle_coherence_mean | 1.0 | 0.9044070508430594 |
| triangle_coherence_fraction_above_half | 1.0 | 0.9576347074262775 |
| phase_locking_order | 1.0 | 0.890169224085449 |
| spectral_gap | 256.0 | 192.6755784729074 |
| mirror_residual | 0.0 | 0.0 |
| cycle_norm | 146.935359937627 | 129.57599317462302 |

- nonuniform top eigenvalues: [228.45491757735027, 35.77933910444286, 34.24583991877743, 15.94093479299104, 12.198547016283062, 8.590489957448243, 7.092497846072505, 6.766625981745913]
- uniform top eigenvalues: [256.00000000000006, 6.459788272744712e-14, 5.778247542636387e-16, 3.8612826491253854e-16, 2.833797931155454e-16, 1.5243502401128247e-16, 1.1977783690987349e-16, 1.1092780610713426e-16]
- nonuniform top IPR: [0.004097, 0.006908, 0.00675, 0.007392]
- uniform top IPR: [0.003906, 0.290791, 0.298695, 0.223923]
- energy relative error: 2.207739901964501e-15

## Reading

- Gradient/cycle orthogonality and reconstruction check the Hodge split.
- Nonuniform mirror couplings change the resonance spectrum, localization
  (IPR), phase-locking order, and triangle coherence relative to uniform.
- Magnitudes alone do not determine triangle coherence; the phase field
  does. This is a relational-phase measurement, not quantum evidence.
