# Resident universe implementation

Status: **implemented and tested** for the local-derived layer. This document
describes the code that exists, not the larger future system.

## What is actually in one file

`nanlmind_hermes/resident/universe.py` opens exactly one sparse memory-mapped
file. The file contains:

| region | implementation | bound on the 256 MiB profile |
|---|---|---:|
| checked header slots | two 4 KiB JSON+digest slots | 64 KiB |
| chart placeholder | reserved for the existing 8 MiB chart | 8 MiB |
| payload arena | text/metadata referenced by leaves | 32 MiB |
| heads + pairs | `cells.py`, 256 heads / 32,640 pair cells | 2 MiB |
| leaf tree | `leaves.py`, fixed 112-byte records + index | ~21 MiB |
| relations + hyperedges | `relations.py`, typed incidence arena | ~42 MiB |
| resonance rows | `resonance.py`, 256-D int16 rows | 16 MiB |
| state journal | 64-byte chained transaction records | 8 MiB |

The default file budget is **256 MiB**. The hard ceiling is **1 GiB**; a
request above 1 GiB is rejected before a file is created. The 2 GiB
overbootstrap map is never opened, copied, or mapped. The file is sparse; cold
pages are not forced resident.

The facade exposes the compatibility section view (all 12 published section
IDs) while the physical layout uses combined regions for heads+pairs and
relations+hyperedges. That avoids pretending a zero-length compatibility ID is
separately addressable.

## Round 23 mapping

The C++ Round 23 prototype uses 128-byte head lanes and 128-byte directional
lanes in a 256-byte pair cell. The Python resident layer keeps the same
mathematical invariants, but stores compact typed state:

- one physical pair cell is shared by `(a,b)` and `(b,a)`;
- the cell has two independently updated directional rails, `s+` and `s-`;
- the mirror is `reverse16(canonical_address)`;
- pair state is Q16.16 weight/amplitude/phase, not untyped floats;
- the canonical pair and its mirror are both readable from the same record.

`ResidentUniverse.directed_weight(a,b)` returns the signed `a -> b` weight.
`cycle_flux(a,b)` computes the Round 23 fundamental circulation relative to
head 0 from the live antisymmetric state. `triangle_flux(a,b,c)` computes a
three-head circulation. These are real derived values; the implementation does
not claim a quantum observable or a performance advantage over native C++.

## Round 24 mapping

`relations.py` provides append-only relations and hyperedges with:

- 32-byte content ids for subject, predicate and object;
- stable relation ids and idempotent re-append;
- a member arena with typed member kinds (`leaf`, `relation`, `pair`, `mirror`);
- all three leaf-role chains (subject, predicate, object);
- bounded chain walks and hyperedge incidence traversal.

`ResidentUniverse.hyperedge_members(id, limit)` returns a bounded incidence
view. A relation ingest creates the relation record, a six-member hyperedge,
leaf-tree evidence leaf, mirror-pair observations, and a resonance row. The
hyperedge is the first recursive grouping layer; parent hyperedges and concept
nodes are the next open work item, not claimed as complete.

## Learning path

`nanlmind_hermes/resident/learning.py` and `scripts/resident_learning.py` add a
measured two-channel loop:

1. the existing `TemporalMirrorLM` learns word n-gram/temporal counts;
2. the resident 256-head pair cells learn directed hashed-token transitions;
3. each accepted sentence is written as a resident leaf with a 256-D feature
   row;
4. held-out evaluation reports temporal, resident and hybrid NLL/perplexity
   separately and per source;
5. the best temporal checkpoint is selected; raw repeated counts are not
   treated as new information.

The free-chat path in `register(ctx)` feeds accepted declarative user exposure
into the same resident file when `resident_learning_enabled` is true. Probe
reads skip the resident writer. Canonical merge remains disabled.

## Reproduce

```bash
python3 -m unittest tests.test_resident_universe tests.test_resident_learning \
  tests.test_resident_tools -v

python3 scripts/resident_learning.py \
  --corpus experiments/corpus/trainable \
  --state-dir /data/data/com.termux/files/usr/tmp/nanlm-resident-full \
  --output docs/RESIDENT_LEARNING.json \
  --size 268435456 --epochs 2 --eval-records 500 --fresh

python3 scripts/resident_universe_cli.py init \
  --path /data/data/com.termux/files/usr/tmp/demo.nlmres --size 16777216
python3 scripts/resident_universe_cli.py status \
  --path /data/data/com.termux/files/usr/tmp/demo.nlmres
```

## Honest boundaries

- The chart placeholder is reserved, not yet the live `Chart8MiB` register.
- The resident feature index is rebuilt from durable leaf payloads on reopen;
  the row bytes are persisted, the Python id map is not.
- The resident pair channel is a coarse 256-head associative model, not a
  replacement for the temporal LM.
- Cycle values are derived and cached only in memory; there is no native
  Hodge benchmark in this layer.
- Export is a candidate bundle only. Intake is generation-CAS staged evidence;
  it never grants canonical authority and never merges BS0/MC0 state.

## Measured corpus run (2026-11-08)

The full local corpus run with explicit definition-relation extraction produced:

| channel | before | epoch 1 |
|---|---:|---:|
| temporal NLL / ppl | 6.7254 / 833.3 | **2.7400 / 15.49** |
| resident pair NLL / ppl | 5.5953 / 269.1 | **3.4160 / 30.45** |
| hybrid NLL / ppl | 5.8347 / 341.96 | **2.5995 / 13.46** |

State: 14,039 train records, 500 held-out records, 19,858 leaves, 2,522
relations, 2,522 hyperedges, 19,602 feature rows, generation 19,709. The selected checkpoint is epoch 1; an identical
second pass changes no metric because the replay guard has already seen the
text. See `docs/RESIDENT_LEARNING_RELATIONS.json` / `.md`.

## 256-head field and scratch-first self-modification

The field layer is implemented in:

- `nanlmind_hermes/head_boolean_field.py` — exact 16-function Boolean base,
  256 independent head states, field coupling and full self-modification checks;
- `nanlmind_hermes/resident/head_field.py` — resident-cell adapter for independent
  head scores and field entropy;
- `nanlmind_hermes/resident/scratch_layers.py` — 256 heads × L0–L9 bounded scratch
  layers with staged-token CAS commit;
- `nanlmind_hermes/resident/selfmod_gate.py` — scratch → Boolean rules → approval
  → generation CAS → one main write;
- `nanlmind_hermes/resident/field_learning.py` — per-sentence independent-head
  activation/coupling learning beside the temporal model;
- `nanlmind_hermes/resident/field_persistence.py` — approved field snapshot and
  versioned per-head scratch persistence;
- `nanlmind_hermes/resident/mth_head_bridge.py` — N16I/MTH Boolean execution per
  head with unavailable-denial instead of fake output;
- `nanlmind_hermes/resident/field_tools.py` — bounded tool handlers;
- `scripts/head_field_benchmark.py` — independent/coupled/single/uniform controls.

Free-chat exposure now activates many heads (measured: 14–30 in short smoke
runs), stages a bounded proposal through the field learner's scratch adapter,
and applies only accepted activation/coupling changes. The 79th plugin tool,
`nanlm_head_field`, supports read-only status/query and explicit
activate/propose/commit. Writes require training scope; activate/commit require
`approved=true`; commit requires the exact generation. Proposal writes go to the
per-head scratch layer `head % 10` before any main write. The legacy Boolean
base is checked before field writes; a failure disables field writes rather than
silently replacing logic.

Benchmark (`--heads 256 --steps 32`): independent mode activates all **256**
heads with **256** distinct activation levels and field entropy **7.8989 bits**
(normalized **0.9874**); the one-head control has exactly one active head. This
is a local Python/Termux measurement, not a native or phone-performance claim.

Measurements are in [HEAD_FIELD_BENCHMARK.md](HEAD_FIELD_BENCHMARK.md) and the
JSON companion.
