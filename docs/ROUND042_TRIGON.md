# Round 042 — TRIGON-9 architecture input and raw 0xNaN path

Status: **raw 0xNaN16 kernel integrated; host reference and neural checkpoints
are external artifacts**. The archive is extracted under
`experiments/round042/nan_round042/`.

## What Round 042 adds

Round 042 is the first restricted-domain dialogue prototype that combines:

- a learned prose relation writer (supplied entity boundaries);
- a differentiable bridge into the physical 8 MiB mirror-pair root;
- support and refutation as independent cycle lanes;
- learned evidence propagation;
- an autoregressive word decoder whose answer is conditioned on computed
  native memory rather than a pre-supplied proof-status token;
- persistent local universes and supervised few-shot phrase adaptation;
- causal deletion/restoration of a physical pair;
- a conventional matched-memory control.

Archive measurements:

| Experiment | Result |
|---|---:|
| WordNet child-split 2–4-hop QA | 540/540 |
| Same tests after gauge drift | 540/540 |
| Five-hop tests | 525/540 (15 broken-chain negatives) |
| Matched conventional signed adjacency | 540/540 |
| Native physical deletion → abstention | 240/240 |
| Restoration → supported answer | 240/240 |
| Four-state support/refute/both/unknown | 480/480 |
| Gauge-shifted four-state responses | 480/480 identical |
| New phrase classes after few-shot adaptation | 3/3 (from 1/3) |
| Original phrase classes after adaptation | 16/16 |
| New-phrase source-split QA | 180/180 |
| Token-loss-only training | collapsed: 90/180, all positives abstained |

All perfect values are controlled, designed episodes; the conventional graph
ties on clean relations. This is **not** unrestricted fluent generation and
does not prove a unique NaN advantage.

## Raw 0xNaN code

`nanlmind_hermes/nan/trig_cycle.nan` is the pure raw implementation requested
for this round. It contains only `0xXXXX_0xXXXXXXXX_0xXXXX` word lines, with no
Python, comments, C, or hidden host fallback.

- mode 0: exact Q16.16 triangle phase `e(0,a)+e(a,b)-e(0,b)` and Q15 cos/sin;
- mode 1: six-word loop/holonomy sum;
- gauge cancellation is performed in integer arithmetic;
- `nanlmind_hermes/resident/trig_cycle.py` only loads the image into NaNVM0 and
  exposes bounded triangle/holonomy reads.

Measured local checks:

```text
triangle(2000,10000,1808) -> phase=10192, cos_q15=18371, sin_q15=27133
holonomy([1000,2000,3000,4000,5000,6000]) -> phase=12000
gauge_triangle(...) == triangle(...)
```

The raw kernel and wrapper have no canonical merge, no native pointer surface,
and a fixed 100,000-step cap.

## Integration mapping

| Round 042 role | Current component | Gap |
|---|---|---|
| prose relation writer | `resident/ingest.py`, `resident/learning.py` | open-domain writer still external |
| support/refute lanes | `full_cycle_memory.py`, `cycle_observable.py` | two-lane façade not yet exposed in CLI |
| native differentiable bridge | `cycle_native.py`, raw `trig_cycle.nan` | PyTorch bridge external |
| propagation | `composition_gate.py`, `retrieval_hypergraph.py` | learned propagation external |
| four-state answer | `composition_gate.py` | four-state neural speaker external |
| persistent universe | `universe.py` + snapshots | `add/ask` session still TODO B-11..B-15 |
| few-shot phrase adaptation | corpus/ingest | supervised adapter external |
| causal deletion/restoration | full-cycle memory | explicit delete/restore API still TODO |

## Point-by-point tasks

| ID | Task | Acceptance |
|---|---|---|
| B-21 | Expose explicit support/refute cycle lanes in the resident façade | both lanes read/write; no cancellation; four-state test |
| B-22 | Add causal delete/restore of one pair/cycle with generation CAS | deleted evidence changes answer; restore returns it |
| B-23 | Implement persistent `add/ask` session over raw cycle memory | add statement, query path, save/reload equality |
| B-24 | Implement bounded learned/simple propagation over resident typed edges | two-to-five-hop positive/negative split |
| B-25 | Add four-state evidence status to composer/tool output | `unknown/supports/refutes/both` preserved |
| B-26 | Load supplied writer/decoder checkpoints without making torch mandatory | optional replay path; pure path unaffected |
| B-27 | Reproduce matched conventional-memory comparison in NaN | exact tie documented, no unique advantage claim |
| B-28 | Add few-shot phrase class adaptation with rehearsal | old/new phrase classes retained |

## Boundaries

- The raw kernel is a computation primitive, not a language model.
- The host `trigon.py` reference and `.pt` checkpoints are external artifacts.
- The archive's perfect scores are controlled, source-split episodes, not open
  prose understanding.
- Conventional memory ties on clean relations; gauge invariance is the
  demonstrated geometric property.
- No physical quantum behavior, canonical merge or 2 GiB overbootstrap copy.

## Reproduce

```bash
python3 -m unittest tests.test_trig_cycle_nan tests.test_resident_trig_cycle -q
python3 -c 'from nanlmind_hermes.resident.trig_cycle import triangle; print(triangle(2000,10000,1808))'
```

The full archive command sequence is documented in
`experiments/round042/nan_round042/README.md`.
