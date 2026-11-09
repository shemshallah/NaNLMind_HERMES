# Round 038 — rotation, order and holonomy architecture value

Status: **validated architecture input; algebra implemented, neural prototype
external**. The supplied archive is extracted under
`experiments/round038/nan_round038/`.

## What Round 038 establishes

The round compares commutative 2D phase paths with noncommutative 3D rotations
on a deliberately order-sensitive composition probe. The reported values:

| Model | Order seed 17 | Seed 41 | Seed 43 |
|---|---:|---:|---:|
| 2D learned phase | 18.33% | 25.56% | 21.67% |
| 3D learned rotation | 18.33% | 100% | 100% |
| conventional DeepSets | 68.33% | 100% | 100% |
| full geometric hybrid | 100% | 100% | 100% |

The full hybrid also reached 100% on the WordNet-backed restricted task in all
three seeds. Its 6,960 trainable parameters live in external PyTorch
checkpoints; the native 8 MiB root supplies source evidence, not native
parameter training.

Native/geometry checks in the archive:

- root **8,388,608 B**, **32,640** pair cells, **32,385** anchored cycle slots;
- SO(2) commutator norm **0.0**; SO(3) commutator norm **1.1399**;
- holonomy trace error under consistent frame changes **1.19e-7**;
- learned angle gradient norm **0.01028**; token-conditioning gradient **0.00856**;
- snapshot restoration exact; gauge-robust cycle reads, raw-edge failure.

**Honest boundary:** the conventional DeepSets control also reached 100% in some
seeds. This is evidence that order-sensitive transformations are useful in the
constructed task, not evidence of a unique geometric advantage, open-domain
prose understanding, or unrestricted generation. The neural transport is
external and the decoder vocabulary is tiny.

## Value adopted into the architecture

The loop `A → B → C → A` is not only a gauge check. It is a **composition
consistency observable**:

- SO(2) phase sums are commutative and cannot distinguish path order.
- SO(3) ordered products can distinguish order.
- A closed loop's trace is conjugation-invariant; it is the right invariant
  readout when local frames differ.
- Contradictory evidence must remain four-valued (`unknown`, `supports`,
  `refutes`, `both`), not be cancelled by opposite phase.

This supports the master TODO's C-track idea of cycles as compositional
constraints while keeping it held until matched controls pass.

## Implemented in NaNLMind_HERMES

`nanlmind_hermes/resident/rotation_holonomy.py` provides the dependency-free
reference algebra:

- `rotation2`, `compose2`, `commutator_norm2`;
- 3D Givens rotations, ordered products, `commutator_norm3`;
- matrix trace/transpose and conjugation error;
- `closed_loop_holonomy`, `audit_order`.

`tests/test_rotation_holonomy.py` verifies the SO(2) commutation, SO(3)
noncommutation and conjugation-invariant trace.

This is not the trained Round 038 network. It is the stable mathematical layer
that future order-sensitive work must use and compare against.

## Tasks added to the master list

| ID | Task |
|---|---|
| B-16 | SO(2) phase vs SO(3) order-sensitive transport |
| B-17 | conjugation-invariant holonomy trace audit |
| B-18 | four-valued evidence preservation |
| B-19 | token-conditioned rotation/attention bridge |
| B-20 | deterministic replay of the supplied R038 artifacts |

The neural replay and token-conditioned bridge remain open; the algebra and
claim boundaries are now explicit.

## Reproduce the local algebra

```bash
python3 -m unittest tests.test_rotation_holonomy -q
```

The supplied research scripts require PyTorch, NumPy, C++20 and the separate
WordNet dictionary; they are not silently folded into the pure test path.
