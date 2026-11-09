# Tool reference

Version 1.5.1 (2026-11-08). All 86 tools are registered under the `nanlmind`
toolset. Tool schemas reject unknown properties unless stated otherwise.
Local tools report `authorization.mode: local-unverified` when no NANNET
authority is configured; a configured authority requires a paired token and
fails closed.

## Default conversation

### `nanlm_chat`
Native conversation by default. NaNLMind parses explicit relations from the
input, updates additive memory and native state, synthesizes a reply from the
learned relation set, and writes the turn to the shared scratch window.
No language-model provider is required when `chat_mode: native`.

Parameters:
- `message` (string, required, 1 byte–1 MiB)
- `session_id` (string, default `default`, ≤80 safe characters)
- `system_prompt` (string, optional; provider modes only)
- `temperature` (number, 0–2, default 0.7; provider modes only)
- `max_tokens` (integer, 1–8192, default 2048; provider modes only)

Returns: response, provider/model, source digest, transcript count, surface
measurements, parsed discoveries, memory merge counts, cross-resolution
feedback, `native_learning`, and `native_synthesis`. With `chat_learning: true`,
explicit supported/refuted triples update native heads, mirror pairs and
hyperedges. Native replies are temporal next-word predictions (no response
templates, no retrieval): Hermes' local-CPU runner weighs every known word at
every step, completes the sentence, and may add explanation sentences when
coverage is low or the question is open-ended. Quality is bounded by exposure.
See `docs/LOCAL_RUNNER.md`.

### `nanlm_chat_history`
Read the durable transcript.

- `session_id` (string, default `default`)
- `limit` (integer, 1–200, default 40)

### `nanlm_phone_call`
Three-way call over the shared scratch journal:
`user -> Hermes -> NaNLM -> user + Hermes`. The user may address Hermes or
NaNLM directly and may inject a new message between runs. NaNLM uses native
parsing/synthesis; Hermes uses its configured model as an optional bridge.

The routing kernel is a raw NaN16 program (`nan/phone_call.nan`, 99 words)
executed on the NaNVM0 boundary (`nanvm.py`): validation, addressee routing,
round loops, event-plan emission and turn accounting are 0xNaN, not Python.
Each turn's kernel plan is recorded in the `phone_event` lane with
`kernel: nan16/1`.

- `session_id` (string, default `default`, ≤64 safe characters)
- `action` (`start`, `say`, `run`, `status`, `end`; default `say`)
- `message` (string, required for `start`/`say`)
- `addressee` (`hermes`, `nanlm`, `both`; default `hermes`)
- `rounds` (integer, 1–64, default 1; extra autonomous Hermes ↔ NaNLM turns)
- `clocked` (boolean, default false; persisted on `start`, overridable on
  `say`). **Off by default so a normal call spends no gateway credits.** When
  enabled, Hermes turns run the call window through the CLOCKED gateway
  (extract → grounding → decide) before phrasing the turn. Plan, result and
  evidence lanes share the call scratch window; failures fall back to the
  normal relay.

Returns the call status, turn count, transcript, `scratch_id`, `kernel`, and
the default path. Each participant writes only to its own scratch lane:
`phone_user`, `phone_hermes`, `phone_nanlm`; the kernel plan is written to
`phone_event`, and CLOCKED lanes land in the same journal.

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
and route counts, local paths, and the permanent `session_context` manifest
(session id, generation, document digests, scope `plugin-duration`).

### `nanlm_session_context`
Read the seven permanent plugin-duration documents loaded at registration.

- `document` (optional enum: `permanent-instructions`, `nannet-architecture`,
  `nannet-routing`, `wordnet-hw4`, `free-chat-learning`,
  `mth-universal-runtime`, `mth-data-leaves`)
- `include_text` (boolean, default false; return every document bounded)
- `max_bytes` (integer, 1–16384 per document)

No NANNET identity is required to read the shipped architecture documents.

## Universal native MTH

### `nanlm_mth_catalog`
- `family` (optional string)
- `tags` (optional array of up to 8 strings)
- `offset` (0–1000000, default 0)
- `limit` (1–100, default 32)
- `require_ci` (boolean, default false)

Returns a page of the bundled 91-kernel manifest. Discovery is not permission.

### `nanlm_mth_inspect`
- `kernel_id` (required, `mth.*`)

Returns source/fixture paths and SHA-256 digests, shape, precision, workspace,
status and `execution_mode` (`native_dispatch_abi` for the four admitted vector
kernels, `native_fixture_only` otherwise). The `native_dispatch_abi` label is
catalog metadata; their unified execution path is N16I.

### `nanlm_mth_runtime`
- `build` (boolean, default false; build the portable core if needed)

Returns host architecture, compiler, bundled/external NaN0S core, native CLI
state, catalog health and N16I availability. N16I is the unified
general-operand path for the four admitted vector kernels and runs on the
host-built/bundled core. No kernels are executed.

### `nanlm_mth_verify`
- `kernel_id` (required)
- `max_steps` (1–500000000, optional)
- `timeout` (1–900, default 120)

Assembles and runs the kernel's native NaN16 fixture on the host-built portable
NaN0S core. Returns the exact marker, steps, trap and explicit non-claims. This
remains native fixture verification and is not the general-operand path; it is
local evidence, not a proof certificate or merge.

### `nanlm_mth_execute`
- `kernel_id` (`mth.vector.dot256.f32`, `mth.vector.cosine256.f32`,
  `mth.vector.add256.f32`, `mth.vector.sub256.f32`)
- `x`, `y` (exactly 256 numbers each)
- `timeout` (1–900, default 120)

Executes real operands through the unified N16I general-operand path on the
host-built/bundled portable NaN0S core and returns a float32 scalar or
256-element result. This same path is used by `mth_leaves` math pairing and its
identity re-checks. Direct native execution (`execute_vector_kernel_native`) is
an explicit diagnostic/parity helper, not the default tool path. The receipt is
local/derived and authorizes no NANNET route, lease, merge or canonical commit.
Other kernels have no general arbitrary-input ABI; use `nanlm_mth_verify`.

### `nanlm_mth_n16i_selftest`
- `timeout` (1–900, default 120)

Runs the NaN0S differential gate: the same NaN16 image through the native
diagnostic/parity helper and under the raw-NaN N16I interpreter-in-NaN16;
serial bytes must be identical.

### `nanlm_mth_dispatch`
- `kernel_id` (required)
- `use_fixture` (boolean, default true; false assembles the kernel source)
- `entry` (guest word address, default 0)
- `max_steps` (1–500000000, default 100000000)
- `timeout` (1–900, default 120)

Assembles the bundled kernel/fixture, relocates the arbitrary guest image to a
host base, and dispatches it under N16I with a bounded step budget. Returns the
guest word count, base, mask, steps, trap, serial output and explicit
non-claims. This is generic N16 dispatch, not a typed operand ABI for every
kernel.

## MTH data leaves and hubs

### `nanlm_mth_hub`
No parameters. Returns the immutable hub tree, root/vector/scalar/matrix/
relation/operator/function/WordNet/constraint hubs, pair rules, WordNet HW4
link and function-tree coverage summary.

### `nanlm_mth_tree`
- `action` (`status`, `map`, `audit`; default `status`)
- `verify_images` (boolean, default true for audit)

`map` links all 91 bundled kernels exactly once: one kernel leaf per kernel
under a family hub, with source/fixture digests, operator/data/WordNet links and
immutable ancestry. `audit` checks expected vs linked, duplicates, digest
mismatches, orphan leaves, ancestry and the NaN-executable image of every
kernel leaf. It returns `complete: true` only when all checks pass.

### `nanlm_mth_leaf_create`
- `kind` (`vector`, `scalar`, `matrix`, `relation`)
- `payload` (object; vector requires exactly 256 finite `values`, or a finite
  `fill` repeated across all 256 elements)
- `function_definition`, `action_relations`, `identities`, `constraints`,
  `constants`, `wordnet` (optional metadata)
- `parent`, `alias` (optional)

Creates an immutable content-addressed leaf and a real NaN16 `.nan` image
containing its metadata and payload. Re-creating identical semantic content is
idempotent.

### `nanlm_mth_leaf_get`
- `leaf_id` (required)
- `include_image` (boolean, default false; include exact `.nan` text)

### `nanlm_mth_pair`
- `left`, `right` (leaf IDs or aliases)
- `operator` (`connect`, `dot`, `cosine`, `add`, `sub`; default `connect`)
- `alias` (optional)
- `check_identities` (boolean, default true)

`connect` creates a relation leaf. The four math operators require
`vector::vector`, execute the admitted float32 kernel through the unified N16I
general-operand path, check dimension/finiteness and symmetry/antisymmetry
identities, and create a new hash leaf. Illegal kinds, hub pairing,
unknown/unadmitted operators, failed execution or violated invariants deny and
create/execute nothing. Inputs are never mutated. The execution receipt records
the local N16I path; it authorizes no NANNET route, lease, merge or canonical
commit.

### `nanlm_mth_leaf_verify`
- `leaf_id` (required)

Runs the leaf's real NaN16 image on the Python NaNVM0 boundary and verifies its
magic/output. No native compiler or canonical commit is required.

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
- `layer` (0–9, optional)

### `nanlm_memory_reorganize`
No parameters. Rebuilds L0–L9 projections and assessments without deleting
events.

### `nanlm_memory_cross_resolution`
- `limit` (integer, 1–1000, default 1000)

Computes ten-resolution comparison matrices and derived hyperedges.

### `nanlm_memory_merge`
- `records` (array, 1–1000)
  - each record: `item` object, `source_digest` 64 hex, optional `head` 0–255,
    optional `layer` 0–9

Idempotent union; duplicate events are counted, not overwritten.

### `nanlm_memory_query`
- `subject` (string, optional)
- `head` (0–255, optional)
- `layer` (0–9, optional)
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

Requires the `nannet_auth_url` and `nannet_issuer_public_key` plugin settings plus the masked `NANNET_MASTER_KEY` Hermes secret.

### `nanlm_auth_revoke`
No parameters. Revokes the stored token and removes its encrypted local copy.

## CLOCKED and head-cortex

All CLOCKED tools require a paired NANNET token with `universe:read`.
Candidate creation, learning, and evaluation additionally require
`training:execute` and `approved: true`. CLOCKED results are evidence, never
ground truth; canonical head state is not modified.

### `nanlm_clocked_plan`
Propose a policy-constrained mode and record a bounded multi-source plan
without transmitting source text.

- `question` (string, required, 1–65536 bytes)
- `sources` (array, max 7; each item requires `text`, optionally `id`)
- `scratch_id` (string, default `default`)

### `nanlm_clocked_execute`
Execute a reviewed plan through the scoped CLOCKED gateway.

- `question`, `sources`, `scratch_id` as above
- `approved` (boolean, default `false`) — must be true
- `plan_sha256` (string, required) — exact digest from the plan step
- `max_requests` (integer, 1–16)
- `max_bytes` (integer, 1–1048576)

### `nanlm_clocked_scratch`
Inspect a revisioned scratch or one writer-owned lane.

- `scratch_id` (string, default `default`)
- `lane` (optional enum: `user`, `hermes_plan`, `clocked_request`,
  `clocked_result`, `evidence_atoms`, `conflict_map`, `nan16_package`,
  `head_response`, `hermes_synthesis`, `head_validation`, `final`,
  `learning_proposal`)

### `nanlm_head_process`
Submit a frozen typed evidence package or answer draft to the NaNLMind
head-cortex interface. The plugin never owns canonical head state.

- `question` (string, required)
- `scratch_id` (string, default `default`)
- `purpose` (`EVIDENCE_COGNITION` or `ANSWER_VALIDATION`)
- `draft` (string, optional)

### `nanlm_clocked_synthesize`
Synthesize an answer from revisioned CLOCKED evidence, validate the draft
through NaNLMind heads, and return the composed response.

- `question` (string, required)
- `scratch_id` (string, default `default`)
- `max_tokens` (integer, 1–8192, default 2048)

### `nanlm_head_candidate`
Create an isolated Boolean-mask mutation candidate for one head.

- `mutation_id` (string, safe identifier)
- `head_index` (integer, 0–255)
- `patch_mask` (integer, 1–65535)
- `base_topology_digest` (64-character hex)
- `approved` (boolean, default `false`)
- `scratch_id` (string, default `default`)

### `nanlm_head_learn_candidate`
Build a weak-supervision set from CLOCKED observations and train an isolated
candidate. Never promotes or mutates canonical state.

- `mutation_id`, `base_topology_digest`, `scratch_id` as above
- `approved` (boolean, default `false`)
- `epochs` (integer, default 1)

### `nanlm_head_candidate_evaluate`
Evaluate an isolated candidate against explicit held-out N16REL rows.

- `mutation_id` (string)
- `records` (array, 1–512 N16REL rows)
- `approved` (boolean, default `false`)
- `scratch_id` (string, default `default`)

## Return conventions

- successful responses are JSON objects;
- failures raise a tool error or return an `error` field;
- a training result includes the corpus digest, command, metrics, native state
  path, discovery count, and memory merge counts;
- a chat result includes the answer before reflection metadata;
- no tool reports a symbolic evidence update as a model-weight update.

## Additional registered tool surface

The following 31 tools are registered in `plugin.yaml` and `register(ctx)` but
do not have full parameter sections above. Together with the 48 detailed
sections, they cover all 79 manifest tools. Protected NANNET and CLOCKED tools
still require a paired token and fail closed; `approved: true` is required where
noted.

| Tool | Purpose |
|---|---|
| `nanlm_learning_cycle` | Run one L0-L9 layered learning cycle across the 255-head ensemble and paged scratch, update concepts, and optionally promote with `approved: true`. |
| `nanlm_learning_status` | Inspect the layered learner, 255-head ensemble, paged scratch register, concept matrix and local head register. |
| `nanlm_learning_sync` | Promote conservative scratch relation candidates into the local 8 MiB head register; canonical BS0/MC0 merge remains separate. |
| `nanlm_wordnet_ingest` | Read a word from the sparse HW4 WordNet universe, project definitions and typed hypernym relations into L0-L9 scratch, and optionally promote. |
| `nanlm_scratch_register` | Inspect bounded records in the 1 GiB logical scratch register space (128 x 8 MiB pages). |
| `nanlm_chart_storage` | Inspect or explicitly checkpoint the SHA-verified local 8 MiB NaN16 chart snapshot; never a canonical training or NANNET merge commit. |
| `nanlm_resident_status` | Inspect the bounded one-file mmap resident universe, its generation/digest, leaves, relations, hyperedges, features and pair cells; canonical merge stays disabled. |
| `nanlm_resident_ingest` | Explicitly ingest bounded local evidence into the resident universe with `approved: true`; writes mirror pairs, leaves and hyperedges locally, never canonical state. |
| `nanlm_resident_query` | Read-only exact-link plus resonance/superposition query over the resident universe; probe reads do not mutate the file. |
| `nanlm_head_field` | Inspect or operate the independent 256-head Boolean field. `activate` requires approval; `propose` writes per-head scratch only; `commit` requires approval and exact generation CAS. Canonical merge stays off. |
| `nanlm_chat_train` | Run free-form conversational learning after the mandatory 256-head Boolean foundation and return relation evidence without bypassing canonical merge. |
| `nanlm_tree_inspect` | Inspect the experimental HW0:1-255 head -> math branch -> data leaf hierarchy without claiming a canonical merge. |
| `nanlm_tree_mutate` | Stage one experimental data leaf in the isolated 8 MiB scratch register with optimistic revision and SHA audit; never changes canonical heads. |
| `nanlm_tree_audit` | Inspect SHA-256-logged experimental register mutations without accessing canonical BS0 or MC0 merge gates. |
| `nanlm_native_mth_u32` | Run native NaN16 checked uint32 add/subtract inside a signed leased hardware universe, with overflow status. |
| `nanlm_hw_register` | Explicitly register an HW5-HW255 NANNET hardware node with consent; does not issue a guest universe. |
| `nanlm_hw_status` | Inspect sanitized local hardware-token and lease metadata without revealing the bearer. |
| `nanlm_hw_heartbeat` | Renew an enrolled hardware-node lease with its encrypted bearer while preserving the bound hardware identity. |
| `nanlm_hw_release` | Release a hardware-node lease after server approval; retain the installation key for later explicit re-registration. |
| `nanlm_hw_report` | Send an authenticated bounded device/system issue report through NANNET. |
| `nanlm_hw2_bind` | Bind the permanent fixed HERMES HW2/node-point-3 identity through an admin/fixed-binding NANNET scope; never allocates HW5-HW255 and never returns the bearer. |
| `nanlm_hw2_status` | Inspect sanitized local or remote fixed-HW2 binding metadata without revealing the guest bearer. |
| `nanlm_hw2_report` | Send a bounded report as the fixed HW2 HERMES identity through NANNET. |
| `nanlm_universe_status` | Inspect encrypted local router leases without returning bearer credentials or receipt MACs. |
| `nanlm_universe_open` | Request a distinct router-issued universe through NANNET's `hermes:lease` signed installation scope. |
| `nanlm_universe_resolve` | Live-verify one local head 0..254 against the authenticated router lease; does not execute NaN16 guest code. |
| `nanlm_universe_renew` | Extend a router-issued lease while rotating its receipt revision; older receipts are rejected. |
| `nanlm_universe_release` | Revoke a router-issued universe and erase its local encrypted receipt after router confirmation. |
| `nanlm_universe_dispatch` | Submit one bounded opaque N16T frame to a router-authenticated native guest with stable `dispatch_id` for at-most-once processing. |
| `nanlm_universe_ack` | Confirm the SHA-256 of a completed native N16T response in the durable router journal without implying training. |
| `nanlm_universe_inspect` | Recover a journaled native N16T dispatch without re-running ambiguous work; return the verified frame if completed. |
