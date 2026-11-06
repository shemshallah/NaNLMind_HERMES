# Tool reference

Version 1.0.1 (2026-11-06). All tools are registered under the `nanlmind` toolset.
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
cross-resolution feedback, and a separate `native_learning` receipt. With
`chat_learning: true`, explicit supported/refuted triples update native heads,
mirror pairs and hyperedges. Neutral/incomplete reflections do not train.

### `nanlm_chat_history`
Read the durable transcript.

- `session_id` (string, default `default`)
- `limit` (integer, 1–200, default 40)

### `nanlm_nan16de_route`
Send one signed raw NaN16 double-endian route envelope to the authenticated
NANNET portal. Requires a paired installation token.

- `route` (string, required, `nan://HW:BS:MC:SC:GM` or explicit `:UM0..254`)
- `kind` (string, default `route`)
- `far_type`, `near_type` (0–3)
- `payload` (32-bit)
- `epoch` (nonnegative)

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

Requires configured `runner_command`. Before language training it runs the
mandatory native full-Boolean gate: 256 heads × all 16 two-input functions ×
4 truth rows (1,024 exact checks), composed half/full adders, majority,
multiplexer and depth-8 circuits, and a word-after-logic probe. A failed gate
aborts the run. It then writes a resumable topology and emits one discovery per
head.

### `nanlm_structured_relation_train`
Train the shared native 256-head topology from caller-supplied typed relations.

- `records` (array, 2–10000): each record contains `split` (`train` or
  `test`), a stable 64-hex `source_digest`, and `item`.
- `item` requires `subject`, `predicate`, `object`, `polarity`
  (`supports` or `refutes`), `confidence`, `evidence`, and
  `provenance.resolution` (bit, nibble, byte, phonetic, morpheme, lexical,
  word_completion, sentence, paragraph or narrative).
- `epochs` (1–100, default 1).

Only train rows mutate state. Test rows are evaluated before and after
training. Every accepted event updates routed heads, three canonical mirror
pairs, temporal links and a provenance-bound hyperedge. Exact replay is a
reported no-op. Metrics are binary relation-polarity metrics, not text
generation accuracy.

### `nanlm_train`
Legacy in-memory Context2 bootstrap.

- `corpus_path` (string, required)
- `duration_minutes` (integer, 1–360, optional)

This is a reproducibility run, not a resumable model checkpoint.

### `nanlm_evaluate`
Run topology-byte, Kneser–Ney order-2, and Context2 controls.

- `corpus_path` (string, required)

### `nanlm_curriculum_experiment`
Generate teacher-labeled typed relations for the selected stages, train the
native relation-polarity objective, evaluate frozen per-resolution challenges
before and after training, merge training evidence, and save an experiment
manifest.

- `topic` (string, required)
- `records_per_stage` (integer, 1–64, default 12)
- `epochs` (integer, 1–100, default 3)
- `max_samples_per_head` (integer, 1–100000, default 4096)

### `nanlm_logic_session`
Explicit logic-only session: Boolean inverses, deep logic and exceptions,
mapped to L0 bit, L1 nibble and L2 byte relation channels. The exact native
full-Boolean foundation gate is mandatory before structured training.

- `topic` (string, optional; default `logic`)
- `records_per_stage` (1–64, default 12)
- `epochs` (1–100, default 3)
- `max_samples_per_head` (1–100000, default 4096)

### `nanlm_word_session`
Explicit word session: phonetic, morphemic, lexical and word-completion relation
channels (L3-L6).

Parameters are the same as `nanlm_logic_session`; default topic `words`.

### `nanlm_sentence_session`
Explicit sentence/story session: sentence, paragraph and narrative relation
channels (L7-L9).

Parameters are the same; default topic `sentences`.

### `nanlm_training_pipeline`
Run a preset or custom ordered list of teacher-generated structured relation
stages. Every populated resolution has an independent frozen held-out set and
before/after metric.

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
