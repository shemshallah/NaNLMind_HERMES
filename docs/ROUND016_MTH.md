# Round 016: pair-space spectrum and MTH implications

Round 016 is classical linear algebra and classical simulation. It is useful
because it constrains the architecture, not because it demonstrates quantum
hardware.

## Exact results

The pair-address graph is J(256,2): two unordered pairs are adjacent when they
share an endpoint. Its spectrum is:

| eigenvalue | multiplicity |
|---:|---:|
| 508 | 1 |
| 252 | 255 |
| -2 | 32,384 |

- degree: 508
- exact return period: π (n even)
- return probability at π: 1
- return at π/2: 0.9688735

The uniform walk therefore has only three distinct frequencies. A 32,640-state
space is not automatically a high-capacity learned dynamics.

## Cycle sector

The oriented incidence matrix `B` gives:

```
R^32640 = im(B^T) ⊕ ker(B)
dimensions: 255 + 32,385
```

`B^T B` annihilates every cycle state. The cycle sector is stationary under
that particular Hamiltonian, but not protected against general noise or
arbitrary perturbations. To make it dynamical, use a cycle-preserving projected
operator such as `P_cycle D P_cycle` or local loop operators.

Hermes implements the state update in `CycleState.apply_diagonal`; it projects
after applying learned edge weights. This is classical, not a physical quantum
evolution.

## Disorder and search

- Random on-site disorder splits the spectrum on J(32,2) but localizes the top
  states (IPR rises from 0.0075 to 0.824 at strength 8). The useful regime is
  selective, learned coupling—not maximum disorder.
- The 15-qubit Grover simulation reaches 99.998% at 141 ideal oracle queries
  versus 16,320.5 classical expected queries. This assumes a coherent semantic
  oracle, state preparation, reflection, and error-free execution. No such
  oracle exists here, and no physical or NaN-specific speedup is claimed.
- A 15-qubit register cannot output 32,640 arbitrary classical values on one
  measurement.

## MTH mapping

| Round 016 object | Hermes/MTH object | status |
|---|---|---|
| K256 pair space | concept matrix / K256 relation space | shared substrate |
| three-level Johnson spectrum | warning about uniform pair dynamics | not MTH validation |
| 255 gradient + 32,385 cycle split | `cycle_space`, `gauge_memory`, `cycle_state` | implemented |
| cycle-preserving `P D P` | `CycleState.apply_diagonal` | classical prototype |
| Grover over 15 qubits | ideal algorithm control | no semantic oracle |
| disorder/localization | learned-coupling design constraint | external simulation |

MTH's L0–L2 Boolean/nibble/byte root and L3–L9 semantic layers are not the
Johnson graph. Round 016 constrains the **substrate** used by MTH memory; it
does not test native MTH kernels, a language model, or physical quantum
behavior.

## Next

1. Learn `D` / relation-aware cycle couplings from typed WordNet and Hermes
   memory; keep the gauge reference recoverable.
2. Benchmark against an equal-parameter ordinary real-valued model and a
   parameter-matched neural baseline.
3. Test semantic prediction, multi-hop composition, and recovery under gauge
   changes jointly.
4. Do not add a quantum-walk claim until a coherent semantic oracle, resource
   accounting, noise model, and hardware experiment exist.
