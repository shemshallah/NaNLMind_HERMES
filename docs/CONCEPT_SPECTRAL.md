# Concept-graph spectral cut

The letter Gram graph was disconnected (Fiedler value ~0, inner 1,
boundary 0). This matrix is built from the concept relation graph with
a weak ring backbone to keep it connected.

- statements: 89
- concepts: 138; pair cells: 462
- heads used: 111; matrix rank: **256**
- normalized Laplacian Fiedler values: [0.006448, 0.024569]
- inner/positive/negative: 56/56/200
- boundary: **16**
- native mirror fixed points: 16

A nondegenerate second eigenvalue and a boundary > 0 mean the cut is
meaningful. The next step is to write word leaves into this connected
matrix and measure whether their couplings land on the boundary modes.
