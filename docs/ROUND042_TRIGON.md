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
for this round. It contains only `NNNN_PPPPPPPP_FFFF` word lines, with no
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

---

## Addendum — TRIGON-9 synthesis (2026-11-09)

The supplied Round 042 archive above is preserved in
`experiments/round042/nan_round042/`. The subsequent synthesis requested after
Rounds 030–041 is documented separately in
[NANLM_42_TRIGON.md](NANLM_42_TRIGON.md) and implemented under
`experiments/round042/` (top-level `trigon.py`, `wordnet_hw4.py`,
`nannet_p.py`, `train.py`, `order_probe.py`).

The synthesis keeps this archive's support/refute lane work and adds:

- a full 32,385-cycle differentiable field with 256 sparse heads;
- L0–L9 complex rotor layers and ordered SO(3) holonomy;
- a byte-level generative LM (3.60 held-out bits/byte, perplexity 12.13 on
  its internal split) with honest malformed-sample disclosure;
- a bounded NANNET-P architecture-evolution gate;
- a 513-word pure raw NaN16 triangle/holonomy/sin-cos kernel.

The archive's controlled dialogue results and the synthesis's language-model
results are separate experiments with separate corpora and evaluation
protocols. Do not merge their numbers into one headline.

---

## Resident support/refute lanes (implemented 2026-11-09)

The four-state lane contour is now implemented in the resident package, not
only described in the archive:

- `resident/evidence_lanes.py` stores each support/refute record in 48 raw
  cycle coefficients, keeps the lanes independent, and returns
  `unknown/supports/refutes/both` without cancellation. Its default namespace
  reserves the final 64 records so the usual composition prefix is untouched;
  `slot_base` can partition another range explicitly.
- `resident/evidence_session.py` adds the bounded
  `X is [not] a [kind of] Y` / `is X a [kind of] Y?` add-ask grammar and
  JSON-safe save/load.
- `resident/universe.py` exposes `evidence_lanes()` and `evidence_session()`;
  the session survives universe reopen.
- `nanlm_resident_evidence` exposes `status/state/ask/tell/save/load` as a
  plugin tool; writes require `training:execute` and `approved=true`.
- `ConventionalEvidence` is the indexed four-state control. It ties the
  geometric path on clean records by design.
- Deletion zeroes the record run and restore rewrites it. Both accept an
  optional generation check; the current check is the in-process field
  generation, not a cross-process canonical CAS.

This closes the bounded B-21, B-22, B-23 and B-25 implementation gates. It does
not add learned propagation, a neural speaker or canonical merge.

