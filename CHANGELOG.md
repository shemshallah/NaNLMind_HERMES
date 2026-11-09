# Changelog

## Unreleased — 2026-11-09

- Added the Round 042 TRIGON-9 experimental synthesis: a 1.08M-parameter
  byte-level generative language model coupling the full 32,385-cycle
  gauge-invariant mirror field, 256 sparse heads, L0–L9 complex rotor layers,
  ordered SO(3) transport/holonomy, typed HW4 WordNet relation probe and a
  bounded NANNET-P architecture-evolution gate. Internal held-out test:
  3.6009 / 3.5858 bits/byte, perplexity 12.13 / 12.01, 26.97% / 27.68%
  next-byte top-1 on seeds 17 and 29; free text remains malformed and the
  trig/holonomy ablation is neutral-to-negative on byte LM.
  See `docs/NANLM_42_TRIGON.md`; no live NANNET route or canonical merge.
- Replaced the raw NaN16 trig-cycle image with a 513-word pure 0xNaN image
  (512-entry Q15 quarter-wave table) while preserving the `resident/trig_cycle`
  bridge contract. Added `tests/test_trig_cycle_nan.py` for pure-image format,
  triangle/holonomy gauge invariance and Q15 sin/cos parity, and
  `tests/test_trigon9.py` for the reference model and NANNET-P merge guard.
- Added a matched byte Transformer control on the same data/token budget:
  1,217,731 parameters, 3.7377 test bpb / 13.34 ppl / 24.05% top-1 in 55 s,
  versus TRIGON-9's 3.6009 bpb / 12.13 ppl / 26.97% in 651 s. The control has
  more parameters but is ~12x faster on CPU; no compute-matched claim.
- Added the HW4 WordNet matrix mirror builder (`82,192` noun synsets,
  deterministic LM and relation splits) and the three-seed order probe
  (`100%` ordered SO(3) vs `33.3%` phase-only).
- Added the Round 042 task track T-01..T-18 to the master TODO. The matched
  conventional LM control, multi-seed LM replication, in-LM order task,
  retention, native N16I model ABI and Android ARM measurements remain open.
- Added independent support/refute evidence lanes over the full 32,385-cycle
  basis (`resident/evidence_lanes.py`), a bounded add/ask/save session
  (`resident/evidence_session.py`), the `nanlm_resident_evidence` tool and an
  indexed `ConventionalEvidence` control. Four-state answers preserve
  `unknown/supports/refutes/both`; the evidence namespace reserves the final
  64 record slots and delete/restore uses an in-process generation check, not
  a cross-process canonical CAS.


## 1.5.0 — 2026-11-08

- Added seven permanent plugin-duration session documents (NANNET architecture,
  NANNET routing, WordNet HW4:0, free-chat learning, MTH universal runtime and
  permanent instructions), loaded at registration, hashed, persisted as a
  session manifest, and injected into chat/reflection/curriculum/CLOCKED/
  synthesis calls. User system prompts cannot remove them.
- Added explicit `local-unverified` authority mode when NANNET is not
  configured. A configured NANNET authority still requires a paired token and
  fails closed. Local chat/training no longer requires a NANNET account.
- Free chat now feeds every turn into the shared L0-L9/255-head/paged-scratch
  learner (best effort) and returns a layered-learning receipt. Deterministic
  `X is Y` / `X can Y` relations supplement provider reflection without
  double-counting.
- Canonicalised the WordNet route to
  `nan://GM0:SC0:MC0:BS0:0:HW4:0`; the HW-first form is compatibility only.
- Added the 91-kernel MTH catalog, all kernel/fixture sources, a portable
  NaN0S C-core source snapshot, architecture/compiler autodetection, cached
  native CLI build, native fixture verification, the four admitted float32
  vector operand dispatchers, and the N16I differential self-test.
- Added generic N16 dispatch: `nanlm_mth_dispatch` relocates an assembled
  guest image to a host base and runs it under N16I with a bounded step budget,
  exact serial/output receipt and no typed-ABI overclaim.
- Mapped all 91 MTH kernels into one immutable function leaf each under 44
  family hubs, with source/fixture digests, operator/data/WordNet links, hub
  ancestry audits and NaN-executable image verification. Added
  `docs/MTH_FUNCTION_TREE.md` and its generator.
- Added immutable data leaves with real NaN16 executable images, vector/scalar/
  matrix/relation hubs, a WordNet HW4 vector link, constraints/identities, and
  constrained `left::right` pairing. Illegal/unknown pairings deny without
  creating or executing a leaf; valid native pairings create a new hash leaf
  and preserve input leaves.
- Added `nanlm_session_context`, `nanlm_mth_catalog`, `nanlm_mth_inspect`,
  `nanlm_mth_runtime`, `nanlm_mth_verify`, `nanlm_mth_execute`,
  `nanlm_mth_n16i_selftest`, `nanlm_mth_dispatch`, `nanlm_mth_hub`,
  `nanlm_mth_leaf_create`, `nanlm_mth_leaf_get`, `nanlm_mth_pair` and
  `nanlm_mth_leaf_verify`, `nanlm_mth_tree`, `nanlm_hw2_bind`,
  `nanlm_hw2_status` and `nanlm_hw2_report`.
- Added Round 042 architecture input: pure raw 0xNaN16
  `trig_cycle.nan` triangle/holonomy kernel, `resident/trig_cycle.py` bridge,
  `docs/ROUND042_TRIGON.md`, and tasks B-21..B-28 for support/refute lanes,
  causal delete/restore, persistent add/ask, propagation, four-state output,
  optional checkpoint replay and matched-memory comparison.
- Added Round 038 order/holonomy algebra (`rotation_holonomy.py`) and added
  Round 038/040 architecture inputs to the master TODO: SO(2)/SO(3) order
  sensitivity, conjugation-invariant trace, dynamic fact writes, add/ask
  session, root+metadata snapshot and retention after writes. The neural
  checkpoints and dynamic session remain explicitly open.
- Integrated rounds 023–036 into the resident layer: raw NaN payload geometry,
  the full 32,385-cycle basis, native cycle snapshots/training bridges, sparse
  mirror postings, recursive typed hyperedges, episodic copy/abstain gating,
  two-/three-source composition and contradiction-aware abstention. Added
  `docs/ROUNDS_023_036_TODO.md` with measured/artifact/open status for every
  item. Neural seq2seq checkpoints remain external artifacts and are not
  claimed as reimplemented.
- Added the one-file resident universe (`nanlmind_hermes/resident/universe.py`)
  with checked headers, payload arena, 256-head/32,640-pair cells, leaf tree,
  typed relations/hyperedges, quantized resonance rows, a state-chain journal,
  bounded export/intake and dual-slot checkpoint publication. Default file
  budget 256 MiB; hard ceiling 1 GiB; no 2 GiB overbootstrap copy.
- Added the 256-head Boolean field/self-modification stack:
  `head_boolean_field.py`, resident `head_field.py`, per-head L0-L9
  `scratch_layers.py`, the scratch-first `selfmod_gate.py`, `field_learning.py`,
  `field_persistence.py`, `field_tools.py`, `mth_head_bridge.py` and the
  independent/coupled head benchmark. Every main write requires Boolean-base
  preservation, explicit approval and generation CAS; free chat activates many
  heads and writes scratch layers first.
- Added `nanlm_resident_status`, `nanlm_resident_ingest`,
  `nanlm_resident_query` and `nanlm_head_field` (79 tools total). Accepted
  free-chat exposure feeds the resident pair/leaf store and activates the
  independent 256-head field when `resident_learning_enabled` is true; probe
  reads remain read-only. `nanlm_head_field` stages proposals in per-head
  scratch layers and requires approval plus generation CAS for a main write.
- Added stratified corpus training, deep multi-epoch training, per-source
  held-out metrics, a resident learning loop and generation evaluation. The
  corrected mixed-corpus epoch-1 result is NLL **2.514** / perplexity
  **12.354**; later raw-count passes are discarded by early stop.
- Added the resident candidate export/validation/intake path with generation
  compare-and-swap. Canonical merge remains disabled (`merge_enabled=false`).
- Added live Gemma architecture review and free-chat smoke scripts, the
  architecture-gap ledger, and the MTH universal-runtime documentation.
- Added a live, open-ended Gemma tool-learning harness that starts a loopback
  NANNET authority, pairs the installation with `universe:read` and
  `training:execute`, keeps canonical merge/promotion disabled, executes real
  plugin tools through model tool calls, and writes a transcript/report.
  Initial test: 12/12 calls, 9 tools, 0 failures.
- Ran the journaled 3-session Gemma training series over the unified pipeline:
  15/15 tool calls, 0 failures, 12 distinct tools, 100% required-task coverage;
  before/after quiz **76.1% -> 100.0% (+23.9)** after carrying the measured
  session journal into later sessions and the after-quiz; system state grew
  turns 0->2, scratch 0->20, concepts 0->30, pairs 0->339, leaves 144->148.
  Results: `docs/GEMMA_TRAINING_SERIES_RESULTS.md`. The earlier non-journaled
  run was flat and is retained only as the diagnostic.
- Diagnosed the flat before/after quiz as an isolated-after-quiz bug: the same
  six-question quiz scored 76.1% before, 76.1% without session experience, and
  **100.0%** with the observed session journal in context. See
  `docs/GEMMA_JOURNAL_QUIZ_PROBE.md`.
- Test suite: the full suite passes in Termux and in Alpine/proot with
  `build-base` + `gawk`; the Alpine run is recorded in
  `docs/ALPINE_PROOT_TESTING.md`.

## Unreleased

- Added a 1 GiB logical paged scratch register space (128 x 8 MiB, sparse),
  L0-L9 layer projections, a deterministic 255-head ensemble, conservative
  promotion into the local 8 MiB head register, and sparse HW4 WordNet
  ingestion (`scratch_register.py`, `layered_learning.py`, `wordnet_store.py`).
- Added five HERMES tools: `nanlm_learning_cycle`, `nanlm_learning_status`,
  `nanlm_learning_sync`, `nanlm_wordnet_ingest`, `nanlm_scratch_register`.
- Added deterministic UTF-8 corpus ingestion, exact-passage deduplication,
  hash-based train/validation/test splits, atomic JSON manifests and
  IDF-weighted source retrieval (`nanlmind_hermes/corpus_memory.py`).
- Added sparse ordinary and cycle-projected semantic-state features and a
  controlled novel-term answer-decoder benchmark. The ordinary state scored
  0.7033, cycle projection 0.6953 and the no-state control 0.6847; no
  cycle-specific advantage is claimed.
- Documented the Round 020 corpus-training tie and the next joint-state
  experiment. Historical test suite at that time: 272 tests.

## 1.3.2 — 2026-11-07

- Made the MTH/L0-L9 stack coherent: every layer produces a signal used in
  candidate scoring (structure, phonetic, morpheme, lexical, word completion,
  sentence truth, paragraph coherence, narrative causality).
- Heads now actively use MTH root transforms: each relation maps through a
  root Boolean function to a head and mathematical leaf, organized in a mapped
  head scratch register plus 1 KiB per-head scratchpad leaves.
- Exposed the head register and MTH layers in chat metadata and the long-chat
  session; 172 tests pass.

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
