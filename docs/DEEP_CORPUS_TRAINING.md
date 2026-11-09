# Deep corpus training

`scripts/deep_train_corpus.py` trains the native NaNLM temporal student for
multiple deliberate passes over the corpus trainable corpus and evaluates a
fixed held-out sample after every epoch. It reuses `TemporalMirrorLM` and the
token utilities in `nanlmind_hermes.temporal_lm`; it does not implement a new
learner.

Gemma is optional. When requested, it is only a teacher/evaluator through the
existing Ollama OpenAI-compatible chat endpoint. Gemma is **not** the NaNLM
runner, does not write student replies, and never receives held-out answer
text.

This is a local derived-state experiment: no canonical merge, canonical
commit, NANNET route, or proof claim is made.

## Quick start

Run the default deep pass (5 epochs) from the repository root:

```bash
python3 scripts/deep_train_corpus.py \
  --state-dir "$HOME/.local/share/nanlmind-hermes/corpus-deep-training" \
  --epochs 5 \
  --max-eval-records 200 \
  --output docs/CORPUS_DEEP_TRAINING_RESULTS.json
```

Add optional Gemma planning and evaluation probes:

```bash
python3 scripts/deep_train_corpus.py \
  --state-dir "$HOME/.local/share/nanlmind-hermes/corpus-deep-training" \
  --epochs 5 \
  --gemma-plan \
  --gemma-probes \
  --gemma-url http://127.0.0.1:11434/v1/chat/completions \
  --gemma-model gemma4:31b-cloud
```

Important defaults:

- `--trainable-dir` prefers `experiments/corpus/trainable`. If that path is
  absent and the relocated corpus exists at `experiments/corpus/trainable`, the
  relocated path is used. Pass `--trainable-dir` to select a corpus explicitly.
- `--epochs` is optional. Without a plan it defaults to **5**; the accepted
  range is **1..20**. An explicit `--epochs` overrides a Gemma plan.
- `--state-dir` defaults to `$NANLM_CORPUS_STATE_DIR` when set, otherwise
  `~/.local/share/nanlmind-hermes/corpus-deep-training`.
- `--output` defaults to `docs/CORPUS_DEEP_TRAINING_RESULTS.json`; the
  Markdown companion is written beside it with the `.md` suffix.
- `--max-eval-records` defaults to 200. Use `0` to score the whole validation
  split.
- A run resumes from the state already present in `--state-dir`. Use a new or
  empty directory to start from scratch.

## Input schema

Every `*.jsonl` file below the trainable directory is read. A line is one JSON
object (several concatenated objects on one line are also parsed) with schema
`nanlm.trainable.v1`:

```json
{"schema":"nanlm.trainable.v1","record_id":"...","text":"...","stage":"...","source":"...","split":"train","provenance":{}}
```

Validation rules:

- `schema` must be exactly `nanlm.trainable.v1`.
- `record_id` must be a non-empty string and is the deduplication key. The
  first valid occurrence wins; later duplicates are counted and skipped.
- `text` must be a string. Whitespace is normalized and the result must be
  2..4000 characters.
- `split` is case-folded. `train`, `training`, `fit` are training splits;
  `validation`, `valid`, `val`, `dev`, `test`, `heldout`, `hold-out`,
  `holdout` are validation splits.
- Malformed lines, invalid schemas, missing IDs, invalid text, and unknown
  splits are counted rather than silently ignored.

The report includes exact per-file counts for records, accepted rows, train
rows, validation rows, duplicates, rejected rows, and malformed lines, plus
corpus/train/validation SHA-256 digests over sorted record IDs.

### Deterministic validation fallback

Declared validation rows are always used as-is. If the corpus has no validation
rows at all, the loader deterministically moves records with the lowest
`sha256(record_id)` hashes into validation: 5% of the training records, at
least one and always leaving at least one training record. The report records
whether this fallback was used and how many records moved.

The held-out evaluation sample is fixed for the whole run: validation records
are sorted by `sha256("eval:" + record_id)` and the first
`--max-eval-records` records are selected (all when the flag is `0`). The sample
size, record IDs, and digest are written to the report, so every epoch is scored
on the same records.

## Training loop and checkpoints

For each epoch the script:

1. Clears **only** `model.seen`, the replay digest guard, so the same corpus is
   deliberately observed again. Vocabulary, unigram/bigram/trigram counts,
   pair weights, register state, and all other associations are preserved.
2. Calls `TemporalMirrorLM.observe` on every selected training record and
   counts observed versus rejected records. Questions and request-like
   sentences are rejected by the existing learner.
3. Saves the model through `TemporalMirrorLM.save(ctx)`.
4. Scores the fixed held-out sample and records the checkpoint metadata.

The optional Gemma plan can reorder training records by `focus_stages` and cap
`max_records_per_epoch`; it cannot change record text or write replies. The
plan is clamped to 1..20 epochs, 0..number-of-training-records per epoch, and
known stages only.

## Metrics

Held-out next-token evaluation is computed after every epoch and before
training:

- **mean NLL**: mean negative log-likelihood of each target token after the
  first token of each held-out record, using the learner's existing interpolated
  unigram/bigram/trigram probability. Before any vocabulary exists, a fixed
  `1e-6` probability floor is reported so the initial measurement is not
  flattered by smoothing.
- **perplexity**: `exp(mean NLL)`, capped at `exp(50)` for display.
- **vocabulary coverage**: fraction of scored target tokens already present in
  the student vocabulary.
- **deltas**: NLL and perplexity deltas from the pre-training measurement and
  from the previous epoch, plus an explicit `nll_improved` flag.

When `--gemma-probes` is used, completion probes are also scored without
generating student text. Probe metrics report probe count, tokens scored, mean
NLL, perplexity, per-probe rows, and deltas.

These are next-token measures only. They are not a general language
understanding or proof metric.

## Persistence and checkpoint paths

Model files live under the selected state directory:

- `<state-dir>/language/temporal_mirror.json` — temporal student model.
- `<state-dir>/language/temporal_register.json` — temporal register sidecar.
- `<state-dir>/language/chart-checkpoints/chart.0.n16s` or
  `chart.1.n16s` — optional unmerged local 8 MiB chart snapshot written by the
  existing local runner when `TemporalMirrorLM.save()` checkpoints.

Every epoch record contains `state_dir`, `epoch`, `observed`, `rejected`,
vocabulary/bigram/trigram statistics, the model path and SHA-256, register
digest, and the current chart checkpoint generation/path/digest when available.
The final checkpoint metadata is repeated at report top level. A fresh run with
an existing state directory loads the prior model first and reports those
"before" metrics.

The chart snapshot is an unmerged local overlay. This workflow does not touch
or replace the canonical NaNLMind checkpoint.

## Optional Gemma plan and probes

The default endpoint is `http://127.0.0.1:11434/v1/chat/completions` and the
default model is `gemma4:31b-cloud`, matching the repository's existing Ollama
usage. The script sends a strict-JSON request; a fenced JSON object is accepted,
but surrounding prose is not.

### Plan

`--gemma-plan` asks for:

```json
{"rationale":"...","epochs":5,"max_records_per_epoch":0,"focus_stages":[],"notes":"..."}
```

Values are bounded and invalid/missing fields fall back deterministically.
Network failures, invalid JSON, or unusable field types produce a fallback plan
using the ordinary default epoch count and all training records. The report
records `status` (`applied` or `fallback`), the applied plan, clamping, and the
error text.

### Probes

`--gemma-probes` asks for completion probes derived from **training samples
only**:

```json
{"probes":[{"source_record_id":"...","prompt":"...","expected":"..."}]}
```

Each accepted probe must reference a real training record, use an exact prefix
of that record ending at a word boundary, and give the exact next continuation
from the same record. The continuation must also end at a word boundary.
Probes are capped at 32, expected continuations at 8 tokens, and
duplicates/invalid rows are counted and rejected. If no valid probe survives,
the script builds deterministic probes from training records by cutting
normalized text at roughly two-thirds of its words. Probes are scored only with
next-token likelihood; Gemma never writes student replies.

### Held-out isolation

Teacher prompts contain at most 24 training samples, truncated to 400
characters each. Validation/held-out records are never included in plan or
probe prompts, so held-out answer text cannot leak into the teacher context.
The report records `heldout_text_sent: false` and the teacher prompt sample
count/digest.

Gemma is a teacher/evaluator over the existing Ollama endpoint. It is **not**
the NaNLM runner, does not execute NaNLM, and does not merge or admit state.

## Report

The JSON report (schema `nanlm.corpus-deep-training/1`) and its Markdown
companion contain:

- corpus path, file list, per-file counts, exact totals, stage counts, and
  record-ID digests;
- split counts, fallback details, fixed eval-sample IDs and digest;
- requested/effective epochs and the source (`cli`, `default`, `gemma-plan`, or
  `gemma-plan-fallback`);
- state directory, checkpoint paths, and SHA-256 digests;
- optional Gemma plan/probe status and fallback reasons;
- pre-training, per-epoch, and final held-out metrics with deltas;
- boundaries: local derived state, no canonical merge/commit, no proof claim,
  and no student replies written by Gemma.

## Boundaries

- Local, derived temporal state only. There is no canonical merge or commit.
- No proof claim. Held-out NLL/perplexity is a next-token diagnostic.
- Repeated exposure clears only the replay digest guard; it does not reset the
  learner.
- Gemma is a teacher/evaluator only and is not the NaNLM runner.
- The deterministic student path uploads nothing. When `--gemma-plan` or
  `--gemma-probes` is enabled, at most 24 bounded TRAIN samples are sent to the
  configured Ollama endpoint; validation/held-out text is never sent.

## Tests

Tests use tiny synthetic JSONL files and a fake provider only. They do not call
a live model or run long training:

```bash
python3 -m unittest tests.test_deep_train_corpus -v
```
