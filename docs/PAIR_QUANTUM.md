# K256 pair-address quantum-walk spectrum

The pair-address graph is the Johnson graph J(256,2): unordered pairs
are adjacent when they share a NaN vertex. The adjacency spectrum is
exact and has only three eigenvalues.

| eigenvalue | multiplicity |
|---:|---:|
| 508 | 1 |
| 252 | 255 |
| -2 | 32,384 |

- vertices: **32,640**
- degree: **508**
- exact revival period: **π** (n even)
- return probability at π: **1.000000000000**

## Ideal Grover control

- valid pair states: **32,640** of 32,768 15-qubit states
- best ideal queries: **141**
- target probability: **0.999981016677**
- classical expected queries: **16320.5**

## Boundary

- Three eigenvalues mean the uniform walk has only three frequencies;
  dimensionality alone is not dynamical capacity.
- The cycle subspace is stationary under `B.T B`, but that protects only
  against that particular Hamiltonian, not general noise.
- Grover search assumes a coherent semantic oracle and state preparation
  that do not exist; no physical or NaN-specific advantage is claimed.
