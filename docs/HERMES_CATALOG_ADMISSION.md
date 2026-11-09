# Hermes catalog admission checklist — v1.5.1

This checklist tracks the official Hermes plugin catalog submission requirements
for this repository. It distinguishes checks that this plugin repository can
prove from steps that only the upstream catalog maintainer can complete.

## Current status (November 8, 2026)

The old catalog PR is still open and has **CHANGES_REQUESTED**. HERMES v1.5.1
remediates the two explicit blockers on the source branch; the merged SHA,
new upstream pin and final human review are pending until their checks pass.
Historical tests and smoke results below belong to their original tested
versions and are not automatic release validation.

## Plugin-repository checks

| Requirement | Status | Evidence / note |
|---|---|---|
| Current plugin manifest format | PASS | `plugin.yaml` uses manifest v2. |
| Hermes compatibility floor | PASS | `requires_hermes: ">=0.23.0"`; CI validates against Hermes Agent v0.23.0. |
| Python dependency policy | PASS | No mandatory third-party runtime dependencies; `cryptography>=42,<47` is an optional bounded `pyproject.toml` extra, with a dependency-free compatibility path. |
| Secret handling | PASS | NANNET, CLOCKED and head-cortex secrets use `agent.secret_scope.get_secret()`; secret fields are declared in `config_schema`. |
| No process-environment credential reads | PASS | `tests/test_credential_policy.py` scans all plugin Python modules. |
| Manifest/runtime and release-metadata parity | PASS | `tests/test_manifest_registration.py` checks registered tool names, package/manifest versions, secret names, Apache-2.0 license alignment, and build metadata; the admission and release CI jobs pass. |
| No external credential-store scraping | PASS | No reads of `.env`, `auth.json`, browser profiles, vendor CLI stores or keychains. |
| Authorization and approvals | PASS | Protected operations fail closed; candidate learning and CLOCKED execution retain explicit approval gates. No production auth bypass. |
| No implicit hardware auto-registration | PASS | The former auto-registration module is removed; NANNET pairing is explicit. |
| No runtime override/rebinding of Hermes core | PASS | Plugin uses documented tool registration and does not replace Hermes core behavior. |
| No self-updating plugin code | PASS | No updater or mutable code-download path. |
| External I/O disclosure | PASS | README documents NANNET/CLOCKED/head-cortex network endpoints, local native-runner invocation, state files, encrypted identity storage and scratch artifacts. |
| Telemetry | PASS | No telemetry upload path is implemented. |
| Binary payloads | PASS | Repository tree audit found no executables, shared libraries, bytecode, model weights, database dumps or archives. |
| Desktop SDK restrictions | N/A | No Hermes Desktop SDK or Desktop-specific code. |
| Password-manager restrictions | N/A | No password-manager integration or credential scraping. |

## Validation record

### Historical plugin checks (v1.3.2; do not imply current release coverage)

- Historical plugin version: **1.3.2**; proposed resubmission version **1.5.1** in `plugin.yaml` and `pyproject.toml`; license
  **Apache-2.0** in both.
- Unit/integration suite: the current suite passes locally and in CI on Python
  3.10, 3.11 and 3.12 (at least 369 test methods at the current snapshot).
- Hermes Agent v0.23.0 `hermes plugins validate . --install-deps`: **PASS**.
  Manifest, compatibility floor, config schema, required secret names,
  tool/hook/middleware parity, collision scan, security scan and no-core-override
  checks pass.
- Hermes Plugin Doctor `hermes plugins doctor . --ci`: **PASS**. Runtime
  discovery, manifest parsing, import and registration pass; **86 tools, 1 pre_tool_call hook**.
- Release package: wheel and sdist build; Twine metadata checks pass; required
  plugin source/docs are present in the sdist; both distributions install and
  import; SHA-256 checksums generate and verify.
- GitGuardian: no secrets detected.
- MTH bridge: root 4x4 Boolean table verified, L0-L2 composed at 1.0
  accuracy, L3-L9 learned from exposure, and truth/frame/category-error
  scoring integrated into temporal deliberation.
- Local CPU runner: boot-time genesis 8 MiB chart, trained-chart rebinding,
  real pair-cell rails, mapped temporal register, and the `HW2:0` → `BS0:0::HW0:0`
  link are covered by tests and `docs/LOCAL_RUNNER.md`.
- Phone-call routing kernel: raw NaN16 (`nan/phone_call.nan`), run on the
  NaNVM0 boundary with headless device stubs, verified word-for-word against
  the native nan0s C VM core across an eight-case routing matrix; 12
  NaN16 boundary/kernel tests are part of the current suite.

### Live integration checks

| Check | Result | Evidence |
|---|---|---|
| Production CLOCKED gateway | PASS | A scoped `clu_...` participant token executed a versioned `pii` task plan through `ClockedGatewayClient`; the full plan → execute → scratch → evidence-atom flow passed and wrote the expected `hermes_plan`, `clocked_request`, `clocked_result` and `evidence_atoms` lanes. |
| Local loopback NANNET pairing | PASS (local) | `nanlm_auth_init` → `nanlm_auth_pair` → `nanlm_auth_status` → online `training:execute` authorization → `nanlm_auth_revoke` passed against a real loopback authority with `universe:read` and `training:execute`. The issuer key was pinned from `GET /v1/issuer`; canonical merge stayed false. |
| Production NANNET / Koyeb cutover | NOT PERFORMED | The production Koyeb cutover, live lease/route and single-issuer migration remain outstanding. |
| Native runner | PASS | Real `NaNLMind/build/nanlm-universal` run: `bootstrap` completed with a recorded corpus digest and metrics; `byte`, `kn`, and `ctx2` evaluations completed; the Boolean foundation gate passed **1,024/1,024** with all 16 distinct rules, composed gates and a persisted `boolean_foundation.nmg`. |
| Hermes-managed secrets | PASS | Credential-policy tests scan every plugin module; live pairing and CLOCKED calls used the same secret-scope path as production. |
| Historical full plugin demo | PASS (historical) | [DEMO_RESULTS.md](DEMO_RESULTS.md): all 38 calls across the then-registered 34 tools passed with the real native binary, a loopback NANNET authority, and local CLOCKED/head-cortex contract stubs. The current 86-tool surface is covered by `scripts/demo_run.py`, but no 86-tool result report is recorded yet. |
| Demo transcript | PASS (historical) | [HERMES_DEMO.md](HERMES_DEMO.md): native chat, three-way shared-scratch call, native training metrics, live CLOCKED, live head-cortex, and isolated candidate evaluation. |
| Historical live smoke | PASS (historical) | [LIVE_SMOKE.md](LIVE_SMOKE.md): 11/11 checks passed on the single `nannet.apps.mistral.run` service at the recorded time, including head-cortex process and candidate endpoints. This is not evidence of the production Koyeb cutover. |

Live checks are deployment smoke tests, not CI claims. Rows marked historical
record past runs; they are not claims about the current working tree. Exact
commands and the resulting artifacts are recorded in the release notes.

## Upstream catalog submission gates

| Requirement | Status | Required action |
|---|---|---|
| Maintainer-reviewed PR to `NousResearch/hermes-agent` | OPEN / CHANGES REQUESTED | Existing [#150755](https://github.com/NousResearch/hermes-agent/pull/150755) requires current source pin and responses to the native-environment + approval-hook review. Do not open a duplicate catalog PR. |
| Exact 40-character source SHA pin | PENDING | After the HERMES security-fix PR merges and green release CI completes, pin the exact merged 1.5.1 commit in the existing upstream PR. |
| Owner/maintainer-curated submission | READY | Submit from `shemshallah`, the repository owner. |
| `capabilities:` declaration matches tools and environment variables | PREPARED | Update upstream entry to 86 tools, `pre_tool_call` hook and 3 declared secrets; the new pin must match `register(ctx)`. |
| Scanner warnings reviewed | IN CI | Validate v1.5.1 at the final pin and review warnings from the Hermes scanner, especially subprocess and credential interactions. |
| Human review and merge | PENDING | Requires upstream maintainer review; this repository cannot self-approve catalog admission. |

## Prepared upstream catalog capability block (v1.5.1)

Copy the exact capabilities below into the **existing** upstream PR
[#150755](https://github.com/NousResearch/hermes-agent/pull/150755), keeping
the plugin owner, reviewed commit and declared secrets consistent.

```yaml
capabilities:
  provides_tools:
    - nanlm_status
    - nanlm_session_context
    - nanlm_learning_cycle
    - nanlm_learning_status
    - nanlm_learning_sync
    - nanlm_wordnet_ingest
    - nanlm_scratch_register
    - nanlm_mirror_topology
    - nanlm_router_graph_verify
    - nanlm_router_graph_walk
    - nanlm_self_modify
    - nanlm_mth_tools
    - nanlm_teacher_teach
    - nanlm_chart_storage
    - nanlm_resident_status
    - nanlm_resident_ingest
    - nanlm_resident_query
    - nanlm_resident_evidence
    - nanlm_head_field
    - nanlm_chat
    - nanlm_chat_history
    - nanlm_phone_call
    - nanlm_auth_init
    - nanlm_auth_status
    - nanlm_auth_pair
    - nanlm_auth_revoke
    - nanlm_hw_register
    - nanlm_hw_status
    - nanlm_hw_heartbeat
    - nanlm_hw_release
    - nanlm_hw_report
    - nanlm_hw2_bind
    - nanlm_hw2_status
    - nanlm_hw2_report
    - nanlm_universe_status
    - nanlm_universe_open
    - nanlm_universe_resolve
    - nanlm_universe_renew
    - nanlm_universe_release
    - nanlm_universe_dispatch
    - nanlm_universe_ack
    - nanlm_universe_inspect
    - nanlm_native_mth_u32
    - nanlm_mth_catalog
    - nanlm_mth_inspect
    - nanlm_mth_runtime
    - nanlm_mth_verify
    - nanlm_mth_execute
    - nanlm_mth_n16i_selftest
    - nanlm_mth_dispatch
    - nanlm_mth_hub
    - nanlm_mth_tree
    - nanlm_mth_leaf_create
    - nanlm_mth_leaf_get
    - nanlm_mth_pair
    - nanlm_mth_leaf_verify
    - nanlm_chat_train
    - nanlm_tree_inspect
    - nanlm_tree_mutate
    - nanlm_tree_audit
    - nanlm_generate_curriculum
    - nanlm_curriculum_experiment
    - nanlm_logic_session
    - nanlm_word_session
    - nanlm_sentence_session
    - nanlm_training_pipeline
    - nanlm_nan16de_route
    - nanlm_generate_corpus
    - nanlm_train
    - nanlm_math_train
    - nanlm_structured_relation_train
    - nanlm_evaluate
    - nanlm_memory_ingest
    - nanlm_memory_reorganize
    - nanlm_memory_cross_resolution
    - nanlm_memory_merge
    - nanlm_memory_query
    - nanlm_memory_assessments
    - nanlm_clocked_plan
    - nanlm_clocked_execute
    - nanlm_clocked_scratch
    - nanlm_head_process
    - nanlm_clocked_synthesize
    - nanlm_head_candidate
    - nanlm_head_candidate_evaluate
    - nanlm_head_learn_candidate
  provides_hooks:
    - pre_tool_call
  provides_middleware: []
  requires_env:
    - CLOCKED_GATEWAY_TOKEN
    - NANLM_DEVICE_TOKEN
    - NANNET_MASTER_KEY
```

The upstream entry must set `name: nanlmind-hermes`,
`repo: https://github.com/shemshallah/NaNLMind_HERMES`,
`requires_hermes: ">=0.23.0"`, `version: "1.5.0"`, `maintainer: shemshallah`,
`tier: community`, `category: tools`, and a full 40-character SHA for the
reviewed release commit.
