# Gemma training series

`scripts/gemma_training_series.py` is a reproducible multi-session experiment
that asks whether Gemma improves at using the real HERMES plugin tools and
whether the plugin's own learning state grows.

The run is local: the real plugin handlers are paired to a loopback NANNET
authority with `universe:read` and `training:execute`. Canonical merge and
promotion are disabled, `nanlm_learning_sync` and `nanlm_memory_merge` are not
on the safe tool surface, and the report makes no proof or canonical-commit
claim.

## Usage

```bash
python3 scripts/gemma_training_series.py \
  --sessions 3 \
  --turns-per-session 8 \
  --url http://127.0.0.1:11434/v1/chat/completions \
  --model gemma4:31b-cloud \
  --state-dir /path/to/persistent/state \
  --output docs/GEMMA_TRAINING_SERIES_RESULTS.json
```

Defaults: 3 sessions, 8 turns per session, the Ollama OpenAI-compatible
endpoint above, model `gemma4:31b-cloud`, and
`docs/GEMMA_TRAINING_SERIES_RESULTS.json` plus the matching `.md` report.
`--sessions` is limited to 1..6 and `--turns-per-session` to 1..20. The state
directory is reused across every session and across reruns unless you choose a
new one. The contract document remains `docs/GEMMA_TRAINING_SERIES.md`; live
results are written to the `_RESULTS` paths.

## Session contract

Stages cycle 1, 2, 3, then repeat for longer series:

1. **Explore** — permanent session context, MTH function tree map/audit, MTH
   catalog, and MTH native runtime.
2. **Data leaves** — create two 256-value vector leaves with the `fill`
   convenience, pair them with `connect` and `add`, and verify a leaf's NaN16
   image.
3. **Free chat and learning state** — two `nanlm_chat` turns (teaching then a
   question), then `nanlm_learning_status` and `nanlm_scratch_register`.
   Inspecting existing leaves is optional.

The prompt contains a required-task checklist, but Gemma chooses the order and
may batch tool calls. The safe tool surface is defined by `SAFE_TOOLS` in the
script; merge/promotion handlers are both absent from it and explicitly denied
by `_execute`.

## Written quiz

The same six-question, no-tools quiz runs before session 1 and after the last
session. It covers session context, the MTH function tree, leaf creation with
`fill`, pair `add`, leaf verification, and chat plus learning status.

Each question is worth one point: half for the required JSON keys and half for
the required keyword groups. Answers must be a JSON object keyed by question id
(a fenced ```json block or an `{"answers": ...}` wrapper is accepted). The
score is deterministic and is recorded with the prompt and raw response.

## Metrics

Per-session metrics:

- total, successful, and failed tool calls;
- distinct tools;
- required-task coverage for that session's stage;
- provider turns used;
- invalid-argument calls (malformed JSON, missing required keys, unknown keys,
  or simple schema type/enum violations);
- failures followed by a later successful call, with the recovering tool.

System-learning snapshots run before and after the series from
`nanlm_learning_status`, `nanlm_status`, and a `nanlm_mth_tree` audit, plus
`nanlm_mth_hub` for the leaf count. They record turns, rows, scratch records,
concept-saturation counts/density, linked kernels, and leaves. The JSON report
contains before, after, and deltas; the Markdown report presents the same table
next to quiz and session metrics.

## Results contract

The JSON report (`nanlm.gemma-training-series/1`) contains:

- configuration and NANNET pairing metadata;
- before/after quiz prompts, answers, and scores;
- one record per session with metrics, required-task status, final summary, and
  a transcript whose tool results are truncated to 4000 characters;
- before/after system-learning snapshots and deltas;
- an aggregate summary with per-stage coverage, session deltas, quiz delta, and
  system deltas;
- explicit boundaries: local loopback NANNET, no canonical merge or promotion,
  no canonical commit, no proof claim.

The Markdown report is a human-readable projection of that JSON, including the
quiz questions and scores, per-session metrics, bounded transcripts, system
before/after/delta, and boundaries. A run only
describes the model, endpoint, prompts, state directory, and tool surface used
for that run. It does not establish general model improvement, canonical
learning, or a proof. No live results are checked into this document; generate
them by running the script.

## Tests

```bash
python3 -m unittest tests.test_gemma_training_series -v
```

The tests use synthetic quiz answers, synthetic call/system metrics, and a
scripted fake provider. They do not contact the live cloud model.

## First live results

The corrected journaled 3-session run is recorded in
[GEMMA_TRAINING_SERIES_RESULTS.md](GEMMA_TRAINING_SERIES_RESULTS.md) and
`.json`. It completed **15/15 tool calls with 0 failures, 12 distinct tools and
100% required-task coverage**. The before/after written quiz improved from
**76.1% to 100.0% (+23.9 points)** after the measured session journal was
carried into the after-quiz and later sessions. The plugin's own derived state
grew: turns 0→2, scratch records 0→20, concepts 0→30, occupied pairs 0→339 and
leaves 144→148. The first, non-journaled run was flat (76.1% → 76.1%) and is
kept as the diagnostic that identified the isolation bug; see
[GEMMA_JOURNAL_QUIZ_PROBE.md](GEMMA_JOURNAL_QUIZ_PROBE.md) and
[GEMMA_TRAINING_DIAGNOSTIC.md](GEMMA_TRAINING_DIAGNOSTIC.md).
