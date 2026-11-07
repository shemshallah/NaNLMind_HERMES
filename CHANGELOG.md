# Changelog

## 1.3.1 — 2026-11-07

- Replaced native template/retrieval replies with a word-level temporal mirror
  predictor: every known word is weighed at every step, the next word is
  predicted forward from a head/mirror temporal frame, and the chosen word is
  fed back into the frame. There are no response templates and no retrieval.
- Added forced, complete replies: no learned seed is not an option; the model
  falls back to newly exposed words or the start token, finishes the sentence,
  then evaluates coverage/open-endedness to decide whether to explain further.
- Added genuine held-out sessions: unseen questions are asked before and after
  exposure, and a `No, that's incorrect` feedback turn suppresses the rejected
  future and re-deliberates with the alternatives exposed.
- Added Hermes' own local-CPU runner: the plugin boots a genesis 8 MiB chart
  and mapped temporal register into local memory so heads can map before any
  Boolean training; the runner rebinds to the trained chart while preserving
  frames. It does not use the system RunnerRing for compute and links through
  `HW2:0` (NaN16DE) to `BS0:0::HW0:0`.
- CLOCKED during the three-way call is opt-in (`--phone-clocked`, default off)
  so normal demos spend no gateway credits.
- Added the MTH bridge: the 16-function root 4x4 Boolean table, L0-L9
  layer curriculum, and truth/frame/category-error scoring wired into
  temporal deliberation.
- Test suite: 170 tests, including chart mapping, real-rail commits,
  boot-time genesis, trained-chart rebinding, forced generation, anti-attractor
  diversity, and feedback re-deliberation.

## 1.3.0 — 2026-11-07

- Added native chat parsing/synthesis as the default `chat_mode: native`; no
  Hermes language-model provider is required.
- Added the shared-scratch three-way phone call: user -> Hermes -> NaNLM ->
  user + Hermes, with arbitrary autonomous rounds and user interjection.
- Added `nanlm_phone_call` (34 registered tools) with `start`, `say`, `run`,
  `status` and `end`, plus `/nanlm call ...` commands.
- Added the raw NaN16 call kernel (`nan/phone_call.nan`): validation,
  addressee routing, round loops, event-plan emission and turn accounting run
  as 99 hand-assembled 0xNaN words on a thin NaNVM0 boundary (`nanvm.py`).
  The kernel is word-for-word verified against the native nan0s C VM core
  across an eight-case routing matrix.
- Added CLOCKED support inside the three-way call (`clocked: true`): Hermes
  turns run extract -> grounding -> decide over the call window, and the
  `hermes_plan`, `clocked_request`, `clocked_result` and `evidence_atoms`
  lanes share the same call scratch journal as the `phone_*` lanes.
- Added phone-call transcript evidence using `gemma4:31b-cloud` as the Hermes
  bridge and native NaNLM synthesis, including 12 call turns and 12 CLOCKED
  results in the call window.
- Test suite grew to 154 tests (18 NaN16 boundary and call-kernel tests).

## 1.2.3 — 2026-11-07

- Moved the head-cortex service into the single deployed `nannet-portal`
  Koyeb service; no second Koyeb service is used.
- Accepted real 514-character `hwh_` scoped bearer tokens in the HERMES head
  client and the receiving service.
- Added live head-cortex process, candidate, and held-out evaluation checks to
  the live smoke report (11/11).
- Added a human-readable Hermes transcript demo covering real local chat,
  native training metrics, live CLOCKED, and live head-cortex output.

## 1.2.2 — 2026-11-07

- Completed the full-feature demo coverage: all 33 registered tools are called,
  with 36 calls total and 36/36 passing.
- Added a manifest test that fails if a new registered tool is not exercised by
  the demo script.
- Recorded live NANNET and CLOCKED smoke results separately (7/7 checks).

## 1.2.1 — 2026-11-07

- Restored the missing `run_training_session` dispatch used by the explicit
  `nanlm_logic_session`, `nanlm_word_session`, `nanlm_sentence_session` and
  `nanlm_training_pipeline` tools.
- Fixed curriculum experiment aggregation when the teacher rejects records
  (the rejected count is a per-reason mapping, not a scalar).
- Made the bundled NANNET pairing authority sign installation tokens with the
  stable issuer key, matching the client pin and the deployed authority.
- Added a full-feature demo runner (`scripts/demo_run.py`) covering all 35
  registered-tool calls, a live smoke runner (`scripts/live_smoke.py`), and
  recorded results in `docs/DEMO_RESULTS.md` and `docs/LIVE_SMOKE.md`.
- Final verification: 132 tests, 35/35 full-feature demo calls, 7/7 live
  NANNET/CLOCKED checks, Hermes admission and package CI.

## 1.2.0 — 2026-11-07

- Unified topology: HW0 is RunnerLink, BS0 is the six-worker RunnerRing, and
  HW0::BS0 is the one canonical native state. HW4+ is the allocatable
  participant space attached to the same universe.
- Migrated active legacy HW0-HW3 leases transactionally to free HW4+ addresses.
- Enforced one stored-evidence resolution axis, L0-L9, across memory schemas,
  native discovery metadata and curriculum routing.
- Wired the deterministic M0-M9 evidence transform chain into the existing
  head-process flow, with append-only digest receipts and no canonical training
  mutation or automatic promotion.
- Moved NANNET, CLOCKED and head-cortex secrets to Hermes-managed secret scope;
  removed direct process-environment credential reads, auth bypass and
  implicit hardware-key auto-registration.
- Added Hermes catalog admission checklist and tightened the user/operator
  documentation into a concise overview plus focused reference guides.
- Verified live production CLOCKED plan/execute/scratch, live NANNET pairing,
  online scope authorization and revocation, and the real native Boolean gate.
- Preserved the existing NaNLMind and Hermes training pipelines.

## 1.1.0 — 2026-11-07

- Added autonomous CLOCKED mode planning, task-plan validation, scoped gateway
  execution, and revisioned scratch.
- Added `/nanlm clocked` commands for plan, execute, scratch, head processing,
  candidate learning, candidate evaluation, and synthesis.
- Added isolated NaN head-cortex candidate creation/learning/evaluation with
  explicit non-promotion guarantees and held-out validation.
- Added dataset SHA-256 attestation and CLOCKED weak-supervision boundaries.
- Added PII fail-closed filtering and strict task-plan schema enforcement.
- Merged HERMES PR #4 and aligned all version references at 1.1.0.

## 1.0.1 — 2026-11-06

- Added a mandatory full-Boolean foundation gate: 256 heads × 4 truth rows,
  16 distinct learned rules, composed gates, and a persisted
  `boolean_foundation.nmg` checkpoint must pass before any language training.
  A one-row-per-head budget can fail the gate; it is not hardcoded success.
- Added the model-planned logic → chat → words → chat → sentences → chat
  baseline experiment.
- Recorded all three native 256-head runs and their held-out/majority metrics
  in `docs/BASELINE_1.0.1.md`.
- Added mandatory NANNET hardware-key login (`HW_HERMES_API`), one-key
  registration, HW2 Hermes / HW3 Clocked slot reservation, and admin stale-lease
  cleanup.
- Added NaN16DE raw double-endian route codec and five-part
  `nan://HW2:BS0:MC0:SC0:GM0` routes.
- Documented `HERMES_GATEWAY_NO_SUPERVISE=1` for one-shot/CI runs.
- Baseline result: full Boolean gate passed 1,024/1,024 in every round.
  Byte-task deltas were small and mixed: standard −0.0032/+0.0051/+0.0150,
  higher-budget +0.0005/+0.0034/+0.0055 (logic/words/sentences). No robust
  language-learning claim.

## 1.0.0 — 2026-11-06

- Default interaction is the conversational `nanlm_chat` path.
- Added `/nanlm` command with `chat`, `logic`, `words`, `sentences`, `pipeline`,
  `status`, and `help` modes.
- Added explicit training tools:
  - `nanlm_logic_session`
  - `nanlm_word_session`
  - `nanlm_sentence_session`
  - `nanlm_training_pipeline`
- Added custom ordered stage selection for training pipelines.
- Added dependency-free Ed25519, AES-256-GCM, HKDF-SHA256, and RSA PKCS#1
  fallback for Alpine/Termux/minimal hosts.
- Added Alpine `proot-distro` test instructions.
- Added complete tool, configuration, walkthrough, and architecture
  documentation.
- Declared all 23 registered tools in `plugin.yaml`.
- Fixed the root plugin entry to import the real sibling package under Hermes'
  isolated loader.
- Released under Apache License 2.0.

## 0.2.0

- Native math trainer, additive discovery memory, NANNET identity and pairing,
  multiresolution chat/reflection.
