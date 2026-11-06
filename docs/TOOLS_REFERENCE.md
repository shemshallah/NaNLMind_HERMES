# Tool reference

Version 1.0.0 (2026-11-06). All tools are registered under the `nanlmind` toolset.
Tool schemas reject unknown properties unless stated otherwise.

## Default conversation

### `nanlm_chat`
Open-ended Hermes conversation. This is the default interaction path.

Parameters:
- `message` (string, required, 1 byte–1 MiB)
- `session_id` (string, default `default`, ≤80 safe characters)
- `system_prompt` (string, optional)
- `temperature` (number, 0–2, default 0.7)
- `max_tokens` (integer, 1–8192, default 2048)

Returns: response, provider/model, source digest, transcript count, surface
measurements, reflection proposal count, memory merge counts,
cross-resolution feedback.

### `nanlm_chat_history`
Read the durable transcript.

- `session_id` (string, default `default`)
- `limit` (integer, 1–200, default 40)

## Status and corpus

### `nanlm_status`
No parameters.
Returns universe ID, native checkpoint presence, last native run, memory event
and route counts, and local paths.

### `nanlm_generate_corpus`
Generate a bounded N16TXT/1 corpus using the active Hermes provider.

- `topic` (string, required)
- `records` (integer, 1–64, default 24)

### `nanlm_generate_curriculum`
Generate the selected curriculum stages. Default is all twelve.

- `topic` (string, required)
- `records_per_stage` (integer, 1–64, default 12)

The tool schema does not expose a stage subset; use `nanlm_training_pipeline`
for custom ordering.

## Training

### `nanlm_math_train`
Explicit native 256-head training.

- `corpus_path` (string, required)
- `epochs` (integer, 1–100, default 3)
- `max_samples_per_head` (integer, 1–100000, default 4096)

Requires configured `runner_command`. Writes a resumable topology and emits one
discovery per head.

### `nanlm_train`
Legacy in-memory Context2 bootstrap.

- `corpus_path` (string, required)
- `duration_minutes` (integer, 1–360, optional)

This is a reproducibility run, not a resumable model checkpoint.

### `nanlm_evaluate`
Run topology-byte, Kneser–Ney order-2, and Context2 controls.

- `corpus_path` (string, required)

### `nanlm_curriculum_experiment`
Generate all twelve stages, run native math training, merge discoveries, and
save an experiment manifest.

- `topic` (string, required)
- `records_per_stage` (integer, 1–64, default 12)
- `epochs` (integer, 1–100, default 3)
- `max_samples_per_head` (integer, 1–100000, default 4096)

### `nanlm_logic_session`
Explicit logic-only session: Boolean inverses + deep logic.

- `topic` (string, optional; default `logic`)
- `records_per_stage` (1–64, default 12)
- `epochs` (1–100, default 3)
- `max_samples_per_head` (1–100000, default 4096)

### `nanlm_word_session`
Explicit word session: repeated words + lexical words + word completion.

Parameters are the same as `nanlm_logic_session`; default topic `words`.

### `nanlm_sentence_session`
Explicit sentence/story session: sentence choice + narrative.

Parameters are the same; default topic `sentences`.

### `nanlm_training_pipeline`
Run a preset or custom ordered list of curriculum stages.

- `topic` (string, optional; default `logic and language`)
- `mode` (`pipeline|logic|words|sentences`, default `pipeline`)
- `stages` (array of stage names, optional; overrides preset)
- `records_per_stage` (1–64, default 12)
- `epochs` (1–100, default 3)
- `max_samples_per_head` (1–100000, default 4096)

Unknown stage names are rejected. An empty list is rejected.

## Mathematical memory

### `nanlm_memory_ingest`
Append 1–1000 structured discoveries.

- `discoveries` (array, required, 1–1000)
- `source_digest` (64 hex, required)
- `head` (0–255, optional)
- `layer` (0–31, optional)

### `nanlm_memory_reorganize`
No parameters. Rebuilds L0–L6 projections and assessments without deleting
events.

### `nanlm_memory_cross_resolution`
- `limit` (integer, 1–1000, default 1000)

Computes ten-resolution comparison matrices and derived hyperedges.

### `nanlm_memory_merge`
- `records` (array, 1–1000)
  - each record: `item` object, `source_digest` 64 hex, optional `head` 0–255,
    optional `layer` 0–31

Idempotent union; duplicate events are counted, not overwritten.

### `nanlm_memory_query`
- `subject` (string, optional)
- `head` (0–255, optional)
- `layer` (0–31, optional)
- `limit` (1–1000, default 100)

### `nanlm_memory_assessments`
- `subject` (string, optional)
- `limit` (1–1000, default 100)

Returns support/refute weights and contested status.

## NANNET authority

### `nanlm_auth_init`
No parameters. Creates a local Ed25519 installation identity protected by the
NANNET master key.

### `nanlm_auth_status`
No parameters. Reports identity presence, pairing state, key availability.

### `nanlm_auth_pair`
- `pairing_code` (string, 12–256 characters, required)

Requires `NANNET_AUTH_URL` and pinned `NANNET_ISSUER_PUBLIC_KEY`.

### `nanlm_auth_revoke`
No parameters. Revokes the stored token and removes its encrypted local copy.

## Return conventions

- successful responses are JSON objects;
- failures raise a tool error or return an `error` field;
- a training result includes the corpus digest, command, metrics, native state
  path, discovery count, and memory merge counts;
- a chat result includes the answer before reflection metadata;
- no tool reports a symbolic evidence update as a model-weight update.
