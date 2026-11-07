# Changelog

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
