# Hermes catalog admission checklist

This checklist tracks the official Hermes plugin catalog submission requirements
for this repository. It distinguishes checks that this plugin repository can
prove from steps that only the upstream catalog maintainer can complete.

## Plugin-repository checks

| Requirement | Status | Evidence / note |
|---|---|---|
| Current plugin manifest format | PASS | `plugin.yaml` uses manifest v2. |
| Hermes compatibility floor | PASS | `requires_hermes: ">=0.23.0"`; CI validates against Hermes Agent v0.23.0. |
| Python dependency policy | PASS | No mandatory third-party runtime dependencies; `cryptography>=42,<47` is an optional bounded `pyproject.toml` extra, with a dependency-free compatibility path. |
| Secret handling | PASS | NANNET, CLOCKED and head-cortex secrets use `agent.secret_scope.get_secret()`; secret fields are declared in `config_schema`. |
| No process-environment credential reads | PASS | `tests/test_credential_policy.py` scans all plugin Python modules. |
| Manifest/runtime and release-metadata parity | TEST ADDED; CI PENDING | `tests/test_manifest_registration.py` checks registered tool names, package/manifest versions, secret names, Apache-2.0 license alignment, and build metadata. |
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

## Upstream catalog submission gates

| Requirement | Status | Required action |
|---|---|---|
| Maintainer-reviewed PR to `NousResearch/hermes-agent` | PENDING | Submit the catalog entry after this plugin PR is accepted. |
| Exact 40-character source SHA pin | PENDING | Pin the reviewed immutable plugin commit in the upstream catalog; do not use a moving branch or tag. |
| Owner/maintainer-curated submission | PENDING | Confirm repository ownership/maintainer approval in the upstream PR. |
| `capabilities:` declaration matches tools and environment variables | PENDING | In the upstream catalog entry, enumerate the registered tools and the three secret-backed environment names exactly. |
| Scanner warnings reviewed | PENDING | Keep the upstream `hermes plugins validate` output clean; review any caution-level warnings rather than suppressing them. |
| Human review and merge | PENDING | Requires upstream maintainer review; this repository cannot self-approve catalog admission. |

## Prepared upstream catalog capability block

Use this exact capability declaration in the eventual
`NousResearch/hermes-agent/plugin-catalog/nanlmind-hermes.yaml` entry. Keep it
synchronized with `plugin.yaml` and `register(ctx)` at the pinned SHA.

```yaml
capabilities:
  provides_tools:
    - nanlm_status
    - nanlm_chat
    - nanlm_chat_history
    - nanlm_auth_init
    - nanlm_auth_status
    - nanlm_auth_pair
    - nanlm_auth_revoke
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
  provides_hooks: []
  provides_middleware: []
  requires_env:
    - CLOCKED_GATEWAY_TOKEN
    - NANLM_DEVICE_TOKEN
    - NANNET_MASTER_KEY
```

The upstream entry must also set `name: nanlmind-hermes`,
`repo: https://github.com/shemshallah/NaNLMind_HERMES`,
`requires_hermes: ">=0.23.0"`, `version: "1.2.0"`, and a full 40-character
SHA for the reviewed release commit. Use the community tier (the catalog
maintainers control verified/official tiers).

## Validation record

- Plugin version: **1.2.0** in both `plugin.yaml` and `pyproject.toml`.
- Fully validated source/manifest commit: `a99ed96016f844694b444ac63d80909e01f43e10`.
- Python 3.10, 3.11 and 3.12: **128 tests passed on each version** (384 successful test executions).
- Hermes Agent v0.23.0 `hermes plugins validate . --install-deps`: **PASS**. Manifest, compatibility floor, config schema, required secret names, tool/hook/middleware parity, collision scan, security scan and no-core-override checks all passed.
- Hermes Plugin Doctor `hermes plugins doctor . --ci`: **PASS**. Runtime discovery, manifest parsing, import and registration passed; **33 tools, 0 hooks**.
- Release package: wheel and sdist both built; Twine checks passed; required source files were present; wheel and sdist each installed and imported; SHA256 checksums for both distributions were generated and verified.
- Release artifact: [nanlmind-hermes-python-distributions](https://github.com/shemshallah/NaNLMind_HERMES/actions/runs/42346913559/artifacts/13563454894).
- GitGuardian: 121 commits scanned; no secrets detected.
- Live CLOCKED/NANNET/production-native endpoint integration is not exercised by CI and remains a deployment smoke-test gate.
- Upstream catalog admission is still separate: submit the catalog PR with the exact 40-character source SHA, then obtain maintainer review and merge.
