# Connection audit — NaNLMind HERMES

Snapshot: 2026-11-08, HEAD `a14e135`, working tree dirty. The audit covers the
working tree, not a clean release: `nanlmind_hermes/mth/`,
`nanlmind_hermes/nan0s/`, `nanlmind_hermes/session_docs/`,
`nanlmind_hermes/session_context.py`, the MTH modules and the new integration
tests were untracked at audit time. Concurrent edits were observed during the
audit (notably the N16I unification), so rerun CI after the tree settles.

Point-in-time counts: **72** manifest/registered tools; **7** permanent session
documents; **91** MTH kernels in **44** families; **369** test methods in
**65** test files. The repository still records **344** tests in
`README.md`/`CHANGELOG.md`/`docs/ALPINE_PROOT_TESTING.md`, **312** in
`docs/ARCHITECTURE_GAPS.md`, and **272** in older docs.

Method: static source/doc inspection plus targeted tests only. No full suite,
no cloud provider call, no Gemma rerun, and no long demo run.

## Status table

| Area | Status | Evidence | Main remaining gap |
|---|---|---|---|
| `plugin.yaml` ↔ `register(ctx)` ↔ demo coverage | **connected** | 72 names in both; `tests/test_manifest_registration.py` and `tests/test_plugin_manifest.py` pass; demo has 79 calls over 72 distinct tools | recorded demo result is historical (34 tools/38 calls) |
| Tool reference / count consistency | **partial** | README/TOOLS_REFERENCE claim 72, but TOOLS_REFERENCE has only 48 tool sections; other docs say 33/34/59 | 24 tools undocumented in the canonical reference; catalog capability block is 34/72 |
| Permanent session context + digests | **connected** | 7 packaged documents hashed, persisted, generation/session id stable; `tests/test_session_context.py` passes | docs say six documents; `mth-data-leaves` omitted from the tool enum in TOOLS_REFERENCE |
| Session-context injection | **partial** | injected in chat, reflection, corpus/curriculum, CLOCKED mode/synthesis (`chat.py`, `tools.py`, `clocked_tools.py`) | no test asserts curriculum/CLOCKED/synthesis injection; phone-call provider is not injected |
| Free chat → durable transcript → reflection | **connected** | transcript append paths; provider reflection plus deterministic native parser; `tests/test_free_chat_learning.py` passes; fake-runner test inspected | native update needs `runner_command` (Gemma smoke correctly reports `native_update_failed`) |
| L0–L9 scratch + local promotion | **partial** | 10 layer projections, 255-head ensemble, 1 GiB sparse scratch, policy and approval gate; tests pass | only the first relation is embedded per layer leaf; relation polarity is dropped; promotion can occur through more than `nanlm_learning_sync` |
| Native relation-train path | **partial** | `tools.ingest_conversation_relations` → `_native_relation_train`; fake-runner test inspected | no live native runner in the Gemma smoke; no end-to-end runner test for the tool path |
| Canonical merge gate | **closed** | no merge tool in the 72 specs; `nanlm_status.merge_gate.canonical_merge_enabled=false`; pipeline test asserts it | generation-checked BS0/MC0 validator and six-runner barrier do not exist |
| WordNet HW4 route and ingest | **partial** | canonical/legacy constants; local archive ingest into 10 layers; missing archive fails closed; 2 WordNet tests pass | `depth` is a no-op; implemented path is a local zip read, not a live NANNET HW4 route; `bidirectional` is unbacked |
| MTH catalog + native runtime | **connected** | 91/91 kernels, sources/fixtures present, 44 families, four unified N16I vector kernels; runtime, fixture, N16I and self-test tests pass | 87 kernels still have no typed operand ABI; cross-architecture builds untested; awk probe untested |
| MTH leaves/hubs/tree audit | **partial** | 91/91 function leaves, 91/91 image checks, 53 hubs, constrained pairing and denial tests pass | unknown-leaf pairing raises instead of returning `status: denied`; audit can pass with a dangling alias; hub tool returns only a summary |
| NANNET loopback pairing/scopes | **connected** (local) | real loopback `PairingAuthority` + HTTP, paired scopes `universe:read`/`training:execute`; pipeline 8/8, auth 5/5, pairing 3/3 | `nanlm_curriculum_experiment` bypasses scope checks; no production lease/route; Gemma safe surface still allows local promotion via `approved=true` |
| Gemma artifacts/report | **connected** (evidence read) | 12/12 calls, 0 failures, 9 tools, scopes, canonical merge false; 3-turn smoke; 7,373/1,011 review tokens | no rerun performed; minor artifact label/argument inconsistencies |
| Docs, links, counts, packaging | **partial/open** | current code counts are 72/7/91; stale docs are listed below | README SPINE link broken; new bundles untracked and absent from stale egg-info/build output |

## Evidence by area

### 1. Manifest, registration and demo

- `plugin.yaml:2` declares version `1.5.0`; `plugin.yaml:10-82` declares 72 tools.
  `nanlmind_hermes/__init__.py:599-969` holds the 72 specs and `:980-986`
  registers them. AST comparison reports exact equality: 72 manifest names, 72
  unique registered names.
- `tests/test_manifest_registration.py:46-50` asserts exact manifest/runtime
  equality; `:52-63` asserts version parity with `pyproject.toml`; `:98-102`
  asserts every registered name appears in `scripts/demo_run.py`. All pass.
- `tests/test_plugin_manifest.py:20-27` asserts exact count and set equality;
  `:29-35` asserts every tool has a callable handler and an object schema. All
  pass.
- `scripts/demo_run.py` contains 79 `call(...)` invocations covering all 72
  distinct tool names. This is stronger than the test, which only checks string
  presence. However `docs/DEMO_RESULTS.md:48` is the historical 34-tool/38-call
  run; there is no current 72-tool demo result artifact.
- Doc drift: `docs/TOOLS_REFERENCE.md:3` says all 72 tools are documented, but
  it has 48 `###` sections. The 24 missing tools are:
  `nanlm_learning_cycle`, `nanlm_learning_status`, `nanlm_learning_sync`,
  `nanlm_wordnet_ingest`, `nanlm_scratch_register`, `nanlm_chart_storage`,
  `nanlm_chat_train`, `nanlm_tree_inspect`, `nanlm_tree_mutate`,
  `nanlm_tree_audit`, `nanlm_native_mth_u32`, `nanlm_hw_register`,
  `nanlm_hw_status`, `nanlm_hw_heartbeat`, `nanlm_hw_release`,
  `nanlm_hw_report`, `nanlm_universe_status`, `nanlm_universe_open`,
  `nanlm_universe_resolve`, `nanlm_universe_renew`, `nanlm_universe_release`,
  `nanlm_universe_dispatch`, `nanlm_universe_ack`, `nanlm_universe_inspect`.
  Some appear in focused docs, but not in the reference linked by
  `README.md:163`.
- Additional stale counts: `docs/HERMES_PLUGIN_INSTALL.md:16` says 33 tools;
  `docs/DEMO.md:182` says 33; `docs/HERMES_CATALOG_ADMISSION.md:41` says 34 and
  its capability block (`:89-130`) declares only 34/72;
  `README.md:237-238` says 34 tools and “59 current tools”.

### 2. Permanent session context

- `nanlmind_hermes/session_context.py:31-74` defines seven documents in order:
  permanent instructions, NANNET architecture, NANNET routing, WordNet HW4,
  free-chat learning, MTH universal runtime, MTH data leaves. `:239-241`
  derives a generation from document ids and SHA-256; `:268-312` caches by
  state dir/generation and persists `session/plugin_session_context.json`;
  `:315-326` returns the prompt block or compact instruction block.
- `tests/test_session_context.py:54-85` verifies seven documents, digest lengths,
  persistence, stable generation/session id and stable `started_at`;
  `:129-160` verifies full context in chat and compact rules in reflection;
  `:150-160` verifies a user system prompt cannot remove the permanent block;
  `:192-217` verifies `local-unverified` and configured-authority fail-closed.
  The 9-test file passed.
- Injection sites: `chat.py:671-679` (full block in provider chat),
  `chat.py:216-224` (compact block in reflection), `tools.py:133`, `:474`,
  `:750` (corpus/curriculum), `clocked_tools.py:89`, `:249`, `:272` (CLOCKED
  mode and both synthesis calls). The exact seven documents and digests also
  appear in `docs/GEMMA_TOOL_LEARNING_SESSION.json` and
  `docs/GEMMA_ARCHITECTURE_REVIEW.json`.
- Gaps: `docs/PLUGIN_ARCHITECTURE.md:113`, `CHANGELOG.md:5`,
  `docs/ARCHITECTURE_GAPS.md:11` all say six documents; only the new Gemma
  report says seven. `docs/TOOLS_REFERENCE.md:90-92` omits `mth-data-leaves`
  from the `nanlm_session_context` enum (the schema in `__init__.py:702` has
  it). `phone_call.py:170-193` sends a provider request without the permanent
  context, despite `session_docs/FREE_CHAT_LEARNING.md:82-85` saying it applies
  to every attached provider call. There is no test asserting injection into
  curriculum, CLOCKED planning or CLOCKED synthesis.

### 3. Free chat → transcript → reflection → scratch → promotion → native train

- Transcript: `chat.py:72-96` (safe session path and append), `:599-607`
  (native path), `:735-739` (provider path), `:833-845` (history readback).
- Reflection: `chat.py:180-270` (provider structured reflection, bounded,
  provenance marked unverified); `:292-345` (deterministic native parser for
  `X is Y`/`X has Y`/`X can Y`/`X causes Y`); `:463-486` unions reflection and
  parser records without double counting.
- Layered learner: `chat.py:489-532` builds one observation per turn;
  `layered_learning.py:433-525` writes all ten layer projections across 255
  heads into the 128×8 MiB sparse scratch space;
  `scratch_register.py` implements page headers, append-only records and
  SHA-256 page verification. `tests/test_scratch_register.py:23-102` passes
  (1 GiB logical, sparse allocation, durability, tamper fail-closed).
- Promotion: `layered_learning.py:621-666` groups exact triples, requires
  support/confidence/approval, caps candidates and writes
  `language/layered_learning_sync.jsonl`; `:668-682` allows `approved=true` to
  trigger promotion inside `cycle`.
- Native relation path: `tools.py:1048-1078` forces chat supervision to
  train-only and calls `_native_relation_train` (`:979-1045`), which runs the
  Boolean foundation gate and the native relation CLI when `runner_command` is
  configured. `tests/test_chat.py:270-307` contains the fake-runner test
  and asserts changed heads/mirror pairs/hyperedges and a native state file;
  it was inspected but not rerun in this audit.
  `tests/test_free_chat_learning.py:66-116` verifies 10 layers per turn, shared
  coordinator state, and scratch failure resilience (4 tests pass).
- Canonical merge is closed: `tools.py:110-118` reports
  `canonical_merge_enabled: false`; no merge tool exists among the 72 specs;
  `tests/test_pipeline_integration.py:155-162` asserts the flag while NANNET is
  paired. This is closure by absence, not an implemented validator.
- Gaps: `layered_learning.py:452-469` embeds only `relation_list[0]` in every
  layer leaf; all relation rows are persisted only in
  `language/layered_learning.json` (`:483-510`), not as per-relation scratch
  leaves. `_relation` (`:302-314`) strips polarity and confidence, so a
  supported and a refuted triple with the same surface text collapse to one
  promotion candidate. The docs say only `nanlm_learning_sync` promotes
  (`docs/NANVERSE_LAYERED_LEARNING.md:101-104`,
  `session_docs/FREE_CHAT_LEARNING.md:72-75`), but
  `nanlm_learning_cycle(approved=true)` (`__init__.py:38-45`) and
  `nanlm_wordnet_ingest(approved=true)` (`__init__.py:55-62`) also promote.
  There is no test for either path or for the polarity behavior.

### 4. WordNet HW4

- `dictionary_universe.py:14-17` defines the canonical route
  `nan://GM0:SC0:MC0:BS0:0:HW4:0`, the legacy compatibility envelope
  `nan://HW4:BS0:MC0:SC0:GM0`, the `HW4:0` link and `BS0:0` port.
  `:143-160` returns those values plus plugin-local `::synset`, `::word` and
  `::relation` references.
- `wordnet_store.py:35-65` streams a local WordNet 3.1 zip into a sparse
  synset/lemma index; `:82-91` fails if no archive is found.
  `layered_learning.py:537-599` reads definitions, hypernyms, typed edges and
  inherited categories and calls `observe`, which writes all ten layers.
- `tests/test_wordnet_tool.py:48-88` verifies canonical and legacy routes,
  10 layers, scratch records, `defined_as`, and missing-archive fail-closed.
  `tests/test_layered_learning.py:94-109` verifies the route, 10 layers and
  lexical relation rows. Both passed.
- Gaps: `depth` is validated and echoed (`layered_learning.py:537,542,597`) but
  never used; `DictionaryUniverse.define`/`inherit` use the default depth 8
  (`dictionary_universe.py:40-63,90-107`), so `depth=1` and `depth=8` behave
  the same. The implemented ingest reads a local zip; there is no call from
  `wordnet_store.py` or `ingest_word` to a NANNET route/lease client, so the
  claim that HERMES reads WordNet through NANNET (`session_docs/WORDNET_HW4.md:12`)
  is not backed by this code. `interface()["bidirectional"] = True`
  (`dictionary_universe.py:152`) is also unbacked: there is no write/route path.
  `mth_leaves.py:53` defines `WORDNET_VECTOR_REF` as `...::word/vector`, which
  does not match the documented `::word/<lemma>/<offset>` grammar. The
  missing-archive error mentions `NANLM_WORDNET_ARCHIVE` (`wordnet_store.py:91`)
  while the plugin setting is `wordnet_archive`.

### 5. MTH catalog, portable runtime and N16I

- `mth_catalog.py:30-35` admits four vector kernels; `:136-200` loads the
  bundled manifest, checks source and fixture existence, pins SHA-256 digests
  and validates precision/status/shape; `:234-255` reports health.
  `nanlmind_hermes/mth/spec/wave-capabilities.json` has 91 unique kernels,
  44 families, 91 `ci_verified: true`, 91 `native_verified`; every source and
  fixture path exists.
- `nan0s_runtime.py:75-85` maps host architecture; `:149-202` autodetects
  compiler and probes awk (gawk/mawk preferred, BusyBox HALT mis-emission
  rejected); `:223-286` builds/caches the portable core; `:384-425` runs native
  fixture verification; `:485-624` implements the unified N16I vector operand
  path; `:663-784` implements arbitrary guest dispatch and N16I verification;
  `:819-866` runs the native-vs-N16I differential self-test.
- Targeted tests passed:
  - `MthCatalogTests`: 2/2 (91 kernels, sources/fixtures, catalog pages).
  - `MthRuntimeTests`: 6/6 in 36.2s (architecture/bundled build, native fixture,
    unified N16I for add/sub/dot/cosine, arbitrary N16I dispatch, N16I
    self-test, plugin tools).
  - `tests.test_n16i_unification`: 6/6 in 53.7s (four kernels native-vs-N16I,
    `nanlm_mth_execute` metadata, leaf add pairing records N16I, inputs
    preserved).
- `docs/MTH_UNIVERSAL_RUNTIME.md` and `docs/N16I_UNIFICATION.md` accurately
  describe the unified path and the four-kernel limit. Gaps documented there:
  87 kernels lack a typed operand ABI; cross-architecture builds are only
  exercised on aarch64; the test suite skips only on missing compiler, not on
  incompatible awk; the awk probe itself has no dedicated test.

### 6. MTH data leaves, hubs and function tree

- `mth_leaves.py:41-121` defines nine hubs, kind-to-hub mapping, vector
  identities/constraints and the five operator rules; `:254-332` seeds the
  root/scalar/matrix/relation/operator/WordNet/constraint/function hubs;
  `:494-606` maps every kernel to one family hub and one immutable kernel leaf
  with source/fixture digests, action relations and NaN16 image; `:608-673`
  audits expected/linked/duplicates/digests/orphans/ancestry/images;
  `:683-755` creates content-addressed leaves; `:806-924` implements
  `connect` and native math pairing; `:980-999` provides one cached store per
  state dir.
- `tests.test_mth_leaves.LeafStoreTests`: 6/6 in 3.0s (9 seed hubs, executable
  `.nan` images, idempotent create, connect, illegal/unknown/hub denial,
  91/91 function tree with 91 images, append-only audit without payloads).
  `tests.test_pipeline_integration`: 8/8 in 24.5s, including NANNET-paired
  tree map/audit 91/91 and connect/add pairing with input preservation.
- The Gemma state confirms the same tree: 148 leaves = 53 hubs + 91 kernel
  leaves + 3 vector leaves + 1 relation leaf, 93 aliases.
- Gaps: `pair()` resolves left/right before its `try` block
  (`mth_leaves.py:809-810`), so an unknown leaf raises instead of returning
  `status: denied`/`unknown_leaf` promised by
  `docs/MTH_DATA_LEAVES.md:100-103`. `function_tree_audit` skips
  `leaf is None` (`:630-635`), so a dangling `kernel:` alias can still report
  `complete: true`; it also does not compare the on-disk image hash. `_new_leaf`
  returns an existing leaf before registering a new alias (`:481-492`), so
  creating identical content with a new alias silently loses that alias.
  `nanlm_mth_hub` returns only aggregate status (`status()` at `:778-804`);
  it does not return the hub tree promised by `docs/TOOLS_REFERENCE.md:171-174`.
  Automated denial coverage is limited to `illegal_pairing`,
  `unknown_operator` and `hub_pairing_denied`; `unknown_leaf`,
  `operator_not_admitted`, `native_execution_failed`,
  `invariant_violation` and `identity_violation` have no direct tests.

### 7. NANNET loopback pairing, scopes and local fallback

- `pairing.py:103-138` implements one-time pairing codes with role ceilings
  and stable issuer tokens; `http_api.py:60-108` exposes `/v1/authorize`,
  `/v1/revoke` and `/v1/pair`; `auth_tools.py:67-100` pins the issuer,
  verifies the issued token and stores it encrypted;
  `auth_tools.py:135-175` verifies the local token and performs the online
  authorization call. `tests/test_pairing.py` (3 tests) and
  `tests/test_auth_tools.py` (5 tests) passed.
- `tests/test_pipeline_integration.py:85-143` starts the real loopback
  `PairingAuthority` and HTTP server, pairs with
  `["universe:read", "training:execute"]`, and drives registered handlers.
  All 8 tests pass. `tests/test_session_context.py:192-217` covers
  `local-unverified` without NANNET and configured-authority fail-closed.
- `scripts/gemma_tool_learning_session.py:343-372` performs the same loopback
  pairing; the JSON artifact records mode `nannet`, both scopes,
  `nannet_pipeline_intact: true`, `canonical_merge_enabled: false`,
  `canonical_commit: false`.
- Gaps: `handle_curriculum` and `handle_curriculum_experiment`
  (`__init__.py:150-156`) do not call `_require_scope`, although
  `run_curriculum_experiment` performs native training and writes the shared
  memory (`tools.py:882-904`). With an authority configured, this is a path
  around the paired-token gate. `nanlm_generate_curriculum` also writes a
  corpus manifest without the `corpus:write` check used by
  `nanlm_generate_corpus`. The Gemma “promotion disabled” boundary is not
  enforced: both harnesses include `nanlm_learning_cycle` and
  `nanlm_wordnet_ingest` in `SAFE_TOOLS`
  (`scripts/gemma_tool_learning_session.py:44-66`,
  `scripts/gemma_training_series.py:59-67`), and both tools accept
  `approved=true`, which promotes into the local head register. `_execute`
  only checks the name (`gemma_tool_learning_session.py:147-155`,
  `gemma_training_series.py:532-540`). `tests/test_gemma_training_series.py:207-215`
  checks only that names contain `merge`/`promot`, not the `approved`
  argument. The production lease/route path remains open
  (`README.md:61`, `docs/ARCHITECTURE_GAPS.md:45-47`).

### 8. Gemma live artifacts (read only)

- `docs/GEMMA_TOOL_LEARNING_SESSION.json`: 12 tool calls, 0 failed, 9 distinct
  tools, NANNET mode, scopes `universe:read` + `training:execute`, canonical
  merge/commit false. The Markdown report matches (12/12, 9 tools, 0 failures).
- `docs/GEMMA_FREE_CHAT_SMOKE.json`: 3 turns, 10 layers and 10/20/30 scratch
  records, active heads 177/222/240; `native_learning_status` is
  `native_update_failed` on every turn because no native runner was configured.
- `docs/GEMMA_ARCHITECTURE_REVIEW.json`: 7,373 prompt / 1,011 completion tokens,
  `teacher_review: true`, no canonical commit, no measured training result.
- `docs/GEMMA_INITIAL_TRAINING_REPORT.md` matches the state directory:
  layered turns 2, 20 observations, 227/255 active heads, 148 immutable leaves
  (53 hubs, 91 kernels, 3 vectors, 1 relation), 93 aliases. Minor artifact
  notes: the tool-learning JSON records `nanlm_mth_tree` with
  `verify_images:false`, but map still audits images; the smoke top-level model
  is `gemma4:31b-cloud` while per-turn records say `gemma4:31b`. Neither
  changes the connection conclusion.

### 9. Canonical merge gate

- No handler among the 72 tools performs a canonical merge. `tools.py:110-118`
  reports `canonical_merge_enabled: false`; `chat.py` and
  `layered_learning.py` return `canonical_commit: false`; the MTH and CLOCKED
  boundary dictionaries also return false. `tests/test_pipeline_integration.py`
  asserts the flag.
- This is **closed by absence**, not a wired gate: the generation-checked
  BS0/MC0 validator and six-runner promotion barrier are explicitly open work
  (`docs/ARCHITECTURE_GAPS.md:38-41`,
  `docs/PLUGIN_ARCHITECTURE.md:233-234`).

### 10. Stale docs, broken links and packaging

- Tool counts: `docs/HERMES_PLUGIN_INSTALL.md:16` (33),
  `docs/DEMO.md:182` (33), `docs/HERMES_CATALOG_ADMISSION.md:32-41`
  (version 1.3.2, 272 tests, 34 tools), `README.md:237-238` (34/59).
- Session-document count: `docs/PLUGIN_ARCHITECTURE.md:113`,
  `CHANGELOG.md:5`, `docs/ARCHITECTURE_GAPS.md:11` say six; the code and Gemma
  artifacts have seven. `docs/TOOLS_REFERENCE.md:90-92` omits
  `mth-data-leaves`.
- Test counts: `README.md:214`/`CHANGELOG.md:47`/`docs/ALPINE_PROOT_TESTING.md:42`
  say 344; `docs/ARCHITECTURE_GAPS.md:3` says 312; `docs/LOCAL_RUNNER.md:105`,
  `docs/CORPUS_MEMORY.md:68`, `docs/PHONE_CALL_NAN16.md:88` say 272. Static
  count at snapshot is 369 methods.
- Memory layer bounds: `docs/TOOLS_REFERENCE.md:339,342,353,360` say layer
  0–31 and “L0–L6”; the schemas cap layer at 9
  (`nanlmind_hermes/__init__.py:945-947`), and
  `tests/test_plugin_manifest.py:37-46` enforces 0..9.
- Link audit: one broken local link, `README.md:276` →
  `docs/SPINE.md` (file does not exist).
- Packaging: `nanlmind_hermes/mth/`, `nan0s/` and `session_docs/` are
  untracked; `nanlmind_hermes.egg-info/SOURCES.txt` and `build/lib/` contain
  no MTH/NaN0S/session-doc files. `MANIFEST.in:9` omits `*.nas`, while
  `pyproject.toml:40-54` declares the package data. A clean sdist/wheel build
  must be rerun and inspected before release.

## Prioritized remaining gaps

1. **P0 — NANNET scope bypass.** `nanlm_curriculum_experiment` calls native
   relation training without `_require_scope`; `nanlm_generate_curriculum`
   writes a corpus without `corpus:write`. A configured authority therefore
   does not uniformly fail closed. Add scope checks and handler tests.
2. **P0 — Gemma “promotion disabled” is not enforced.** The safe tool surface
   still includes `nanlm_learning_cycle` and `nanlm_wordnet_ingest`, and both
   accept `approved=true`. Strip/deny `approved=true`, or remove those tools
   from the safe surface, and add a test for the argument, not just the name.
3. **P1 — Tool reference and catalog metadata are incomplete/stale.** 24/72
   tools have no `TOOLS_REFERENCE` section; the catalog capability block
   declares 34/72 and still says version 1.3.2/272 tests. `README.md:163`
   points users at this reference for all schemas/scopes.
4. **P1 — WordNet contract overclaims.** The implemented path is a local zip
   reader, not a live NANNET HW4 route; `depth` is a no-op; `bidirectional:
   True` and the `::word/vector` reference are unbacked. Either implement the
   route/depth/write-back or narrow the contract.
5. **P1 — MTH leaf denial/audit edges.** Unknown-leaf pairing raises instead of
   returning `status: denied`; a dangling `kernel:` alias can still audit as
   `complete`; `nanlm_mth_hub` does not return the advertised hub tree. Add
   tests and fix the contract.
6. **P1 — Promotion contract mismatch.** Docs say only `nanlm_learning_sync`
   promotes, but `nanlm_learning_cycle` and `nanlm_wordnet_ingest` also promote
   with `approved=true`. The layered path also stores only the first relation
   per layer leaf and drops polarity, so promotion cannot distinguish
   support/refute.
7. **P2 — Session-context coverage.** Phone-call provider calls do not receive
   the permanent context; there are no tests for curriculum/CLOCKED/synthesis
   injection. Docs say six documents; code ships seven.
8. **P2 — Counts, links and packaging.** Fix the 33/34/59 tool counts, the
   six/seven document count, 312/344/272 test counts, layer 0–31 text, the
   README SPINE link, and re-run a clean sdist/wheel check with the untracked
   MTH/NaN0S/session-doc bundles.
