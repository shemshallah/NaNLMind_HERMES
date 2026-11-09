# Hermes local CPU runner, 8 MiB chart, and temporal registers

Hermes does not depend on the system RunnerRing for chat compute. At plugin
boot it instantiates its **own local-CPU runner** and holds the local universe
registers in this process:

1. `register(ctx)` boots `LocalRunner.open(ctx)` immediately.
2. If a trained universe exists (`state_dir/universe/native/global_universe.nmg`),
   it is parsed into the documented 8 MiB chart page. If not, a **genesis**
   8 MiB chart is built in memory, so heads can map into registers before any
   Boolean/local training.
3. The runner is cached per state dir for the life of the process. When local
   training produces a new chart, the cached runner rebinds to it and keeps the
   register frame/state.

It neither joins nor drives the system RunnerRing. It shares code and format
with the native stack (chart layout, dual rails, frame sync) and links through
`HW2:0` — the NaN16DE binary translation interface — to the
`BS0:0::HW0:0` runner-ring NaNNet interface. Its route is
`nan://HW2:BS0:MC0:SC0:GM0`.

## Chart mapping

The serialized `NLMU0001` universe page is expanded into the 8 MiB chart:

| region | offset |
|---|---|
| header + counts | `0x000000` |
| address → canonical index | `0x001000` |
| canonical index → address | `0x021000` |
| pair table | `0x031100` |
| address registers (65,536 × 32 B) | `0x041000` |
| descriptors | `0x241000` |
| shared registers (32,896 × 64 B) | `0x3C2800` |
| head table | `0x5C4800` |
| adjacency / dirty maps / checksums | `0x5C6800` |
| head + chart scratch | `0x5EBA10` |
| extension / mapped registers | `0x63BA10` |

The on-disk `.nmg` is opened read-only and never mutated. The in-memory page is
the one heads/chat use; register writes live in the page and in a small
`state_dir/language/temporal_register.json` sidecar. Nothing is sent to the
merge service or Koyeb by the runner.

## Temporal register

One pair cell is bound to a temporal register. Its two rails are real cell
storage:

```text
s+  forward rail   past -> future
s-  reverse rail   future -> past
```

- pair: two active heads (genesis uses `(17, 136)`)
- address: `addr(a,b) = (a << 8) | reverse8(b)`; canonical is `min(addr, reverse16(addr))`
- mirror: `reverse16(canonical)` (the 16-bit analogue of the native mirror link)
- rails: `nan://HW2:BS0:MC0:SC0:GM0::pair/<a>/<b>/s+` and `/s-`
- register slot in the chart extension: magic `NLMREG01`, frame, forward word,
  back word, and a 256-cell temporal state
- commit: write the predicted head to `s+`, read it back, copy to `s-`, advance
  the logical frame; `rollback` restores `s+` from `s-`
- frame meeting follows the native `FrameSync` rule: opposite rail directions
  meet only when their targets coincide

The register state is what the language predictor reads at the start of a turn
and writes back after each committed word: a temporal feedback loop over the
mirror-addressed 256-cell field.

## Language prediction

`TemporalMirrorLM` is a dependency-free, online word-level predictor:

- every exposed sentence trains unigram, bigram and trigram counts plus a
  head/mirror address transition matrix;
- each step scores **every known word**, interpolates the counts, adds the
  temporal/mirror terms and repetition penalties, and keeps several futures;
- output is forced and complete: if no known seed exists, the learner predicts
  from its newly exposed words (or the start token), finishes the sentence,
  then evaluates coverage and open-endedness to decide whether another
  explanation sentence is needed;
- every committed word advances the mapped register frame.

There are no response templates and no retrieval of stored sentences.

## Merge path (not done yet)

The local chart/register is the local universe. The intended system path is:

1. Hermes trains its local universe;
2. merge stores it on Koyeb;
3. Koyeb analyzes it for corruption;
4. per the merge math (not yet implemented) it is integrated into MC0, SC0,
   BS0DB and MC0DB once those exist.

The NaN16DE link through `HW2:0` to `BS0:0::HW0:0` is the interface that path
will use. This repository implements the local runner and register mapping, not
the remote merge/integration.

## Evidence

- `docs/LONG_CHAT_DEMO.md`: logic training, local runner over a trained chart,
  20 exposure sentences, 15-turn conversation, held-out volcano session,
  three-way call with CLOCKED off.
- `python -m unittest discover -s tests`: the current suite, including chart
  mapping, real-rail commits, boot-time genesis, trained-chart rebinding and the
  temporal predictor. The historical baseline at this document's writing was
  272 tests.


## Crash-recoverable LOCAL chart snapshots (N16SNAP1)

At the native temporal model's save boundary, HERMES now checkpoints the
**entire 8 MiB chart** as an unmerged local overlay, using two alternate
generation slots in \`state_dir/language/chart-checkpoints/\`:

- \`chart.0.n16s\` and \`chart.1.n16s\`: full 8,388,608-byte chart bytes
  preceded by an 80-byte header (\`N16SNAP1\`, uint64 generation, 32-byte
  canonical parent root, 32-byte SHA-256 of the chart bytes).
- A named local advisory lock serializes writers. Generation comparison
  prevents an old HERMES process from overwriting a newer snapshot.
- A new slot is written to a distinct 0600 temporary file, fsynced, atomically
  renamed, and followed by a directory fsync. Both generations must not be
  overwritten at once. A corrupt/torn newest slot falls back to an older
  valid generation; a fully corrupt store fails closed instead of resetting.
- A snapshot is eligible for reopening only against the **exact**
  \`global_universe.nmg\` parent-root digest (or its genesis digest), and its
  immutable index, canonical-address list, pair table and chart header must
  match the freshly built chart. The original native .nmg remains unchanged.
- On reopening, the SHA-verified chart's temporal extension restores the
  frame, forward/back register payloads and 256 NaN16 16-bit state values,
  overriding an outdated JSON sidecar.

This module is **host byte I/O and crash-recovery only**, not an independent
Python implementation of NaN16 computation. Guest math and canonical head
training still require real native NaN16 execution and the usual merge
authorization; no local snapshot implies a training or NANNET commit.
The 16-bit chart temporal state is the durable guest representation; an
auxiliary JSON sidecar may retain higher precision for a live process but
is not authoritative after checkpoint recovery. Snapshots are currently
full-page (8 MiB), not incremental, and are created at save boundaries,
not after every individual word.

## HERMES tool: `nanlm_chart_storage`

- `{"action":"inspect"}` returns sanitized snapshot generation, dirty state,
  canonical parent digest and storage protocol without reading head payloads
  into the tool response.
- `{"action":"checkpoint","approved":true}` requires the installation's
  `training:execute` authority and publishes an **unmerged local** 8 MiB
  snapshot. The receipt explicitly states
  `canonical_training_committed=false` and `nannet_merge_issued=false`.
- No `restore` mutation is exposed via the tool. Restoring happens at local
  runner boot only after checking the snapshot and immutable address tables.

The native 255-head COW scratch bank being developed in NaN0S is not yet
attached to this Python filesystem-transport tool. Until its guest-call ABI
is integrated, this tool must not claim that native head-bank mutations
happened.
