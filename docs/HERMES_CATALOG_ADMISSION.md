# Hermes catalog admission checklist

This checklist tracks the official Hermes plugin catalog submission requirements
for this repository. It distinguishes checks that this plugin repository can
prove from steps that only the upstream catalog maintainer can complete.

## Plugin-repository checks

| Requirement | Status | Evidence / note |
|---|---|---|
| Current plugin manifest format | PASS | `plugin.yaml` uses manifest v2. |
| Hermes compatibility floor | PASS | `requires_hermes: ">=0.23.0"`; CI validates against Hermes Agent v0.23.0. |
| Bounded dependencies | PASS | `cryptography>=42,<47`; no unbounded plugin dependency. |
| Secret handling | PASS | NANNET, CLOCKED and head-cortex secrets use `agent.secret_scope.get_secret()`; secret fields are declared in `config_schema`. |
| No process-environment credential reads | PASS | `tests/test_credential_policy.py` scans all plugin Python modules. |
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

## Validation record

- Plugin version: **1.2.0** in both `plugin.yaml` and `pyproject.toml`.
- Local Python tests: passing on the current branch head.
- Hermes Agent v0.23.0 admission validator and Plugin Doctor: passed on
  `30d264a766d6219c2a23142ffb4b4e97bfbe9b03`; rerun is in progress for later
  topology/state-directory commits.
- A passing local validator is not catalog admission. The upstream catalog PR,
  immutable SHA pin, maintainer review and merge are still required.
