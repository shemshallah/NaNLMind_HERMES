# Round 033 integration

Status: **episodic copy path integrated; neural byte LM remains an external
artifact**. The supplied archive is extracted under
`experiments/round033/nan_round033/`.

## What the archive measured

- WordNet noun glosses: 1,800 per split in the byte-LM experiment.
- Byte-level plain vs triangle memory BpB (seed 17 / 29): 3.67070 / 3.67881
  vs 3.67077 / 3.67881. **No predictive gain from the 24-channel cycle readout
  in that experiment.**
- Episodic 12-candidate, two-character corruption: exact 24-byte copy
  **300/300, 296/300, 290/300** for seeds 17, 29, 41.
- Native 8 MiB root audit: **118/120** exact after gauge shift; conventional
  indexed copy ties **118/120**; raw single-edge byte accuracy **94/2880
  (3.26%)**.
- Separate native root audit: 288 stored bytes exact before and after a gauge
  shift of amplitude 6.

These are supplied artifacts. They are not re-labelled as a unique NaN benefit:
the conventional indexed copy ties the selected-record copy path.

## What is implemented in NaNLMind_HERMES

| Capability | Implementation |
|---|---|
| 8 MiB root geometry | `resident/cells.py`, `resident/cycle_observable.py` |
| raw NaN payload/lane layer | `nan_geometry.py`, `resident/nan_lane.py` |
| 12-record episodic copy | `resident/episodic_gate.py` (`EpisodicCycleMemory`) |
| gauge-invariant copy | cycle readout in `episodic_gate.py` |
| raw-edge negative control | `EpisodicCycleMemory.read_raw` |
| learned copy/abstain gate | `resident/episodic_gate.py` |
| archive audit | `scripts/round034_episodic_audit.py` (R034 superset) |

## Not implemented

- The PyTorch byte-level autoregressive generator and its two matched cycle
  readouts.
- End-to-end gradients through the native C++ root from the byte LM.
- A claim of improved free generation. The episodic decoder copies an already
  stored 24-byte record.

## Reproduction

```bash
python3 -m unittest tests.test_episodic_gate -q
python3 scripts/round034_episodic_audit.py
```

## Boundary

`F(0,a,b)=e(0,a)+e(a,b)-e(0,b)` is an exact additive-gauge invariant in the
stored field. That is a mathematical property; it is not evidence that a
language model is more capable, and a conventional implementation of the same
invariant shares it.
