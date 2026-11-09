# Production readiness TODO — NaNLMind HERMES

Snapshot date: **2026-11-08**  
Repository: `/data/data/com.termux/files/home/NaNLMind_HERMES`  
Commit: `a14e135` (`Merge remote-tracking branch 'origin/main' into
feat/nanverse-layered-learning-20261108`) with a dirty working tree.

This document is a prioritized, point-by-point production checklist. It is not
a release note, test report, or proof that any item below is complete. The
working tree changed during this audit; re-run every listed command against the
exact commit that is proposed for release.

## Non-negotiable release constraints

- Canonical BS0/MC0 merge remains **disabled**. No item below authorizes a
  canonical commit until the six-runner barrier, generation CAS, audit trail and
  rollback contract in P0-017…P0-022 exist and pass.
- `merge_enabled` defaults to **false**. Setting it true may enable only local
  derived promotion; it must not be described as canonical merge.
- No module may claim physical quantum behavior, a quantum speedup, a coherent
  semantic oracle, or a proof certificate from a fixture PASS.
- A local test, loopback authority, screenshot or CLI status is **not**
  production evidence. Production evidence must name the deployed service,
  issuer fingerprint, deployment ID, command, output digest and date.

## Snapshot facts used by this checklist

| Fact | Value at snapshot |
|---|---:|
| Registered tools in `plugin.yaml` and `register(ctx)` | 72 |
| Detailed tool sections in `docs/TOOLS_REFERENCE.md` | 48 + a 24-row summary table |
| Permanent session documents in `nanlmind_hermes/session_docs/` | 7 |
| Bundled MTH kernels | 91 |
| MTH family hubs | 44 |
| Test files under `tests/` | 66 |
| Static test methods (grep count, not a run result) | 376 |
| README/CHANGELOG recorded test count | 344 |
| Local runner | 8 MiB chart, mapped temporal register, optional native CLI |

Counts are diagnostic, not acceptance evidence. Use the commands in
`docs/PRODUCTION_READINESS_TODO.md` §“Release evidence commands” and record the
output digest with the release commit.

## Status legend

- **connected** — code, tests, and a documented local/production path exist;
  the stated acceptance still needs a fresh run on the release commit.
- **partial** — some code or local tests exist, but the production contract,
  negative cases, or end-to-end evidence is missing.
- **open** — no implementation or no usable evidence in this repository.

Evidence in this document is a static reading of the working tree plus one
targeted run of 22 quantum-math tests (`test_gauge_memory`, `test_cycle_state`,
`test_phase_memory`, `test_boolean_wave`, `test_temporal_spine`; all passed on
2026-11-08). The full suite, Alpine/Termux suite, production network probes and
Koyeb deployment were **not** re-run for this document. Do not convert a
`connected` label into a production claim without attaching the command output
and artifact digest.

## Priority rule

- **P0** — blocks any production claim, public deployment, or catalog release.
  A P0 item must be closed or explicitly disabled and surfaced in status.
- **P1** — required for a stable production service; may ship behind a
  documented limitation only with operator sign-off.
- **P2** — documentation, reproducibility and cleanup work; does not by itself
  authorize a production claim.

---

# P0 — release-blocking items

## NANNET authority, identity and transport

### P0-001 — Enforce issuer pinning on every trust path

**Why.** A token is only as trustworthy as the key that signed it. Accepting a
pairing response, a stored token, or an online authorize response without the
pinned issuer key permits authority substitution and silent credential
takeover.

**Files/tests.** `nanlmind_hermes/auth_tools.py` (`auth_pair`, `auth_status`,
`require_scope`), `nanlmind_hermes/auth.py` (`verify_token`),
`nanlmind_hermes/pairing.py`, `tests/test_auth_tools.py`,
`tests/test_pairing.py`, `tests/test_session_context.py`,
`scripts/nannet_hw2_connect.py`.

**Acceptance.** Missing pin or a mismatched issuer fails before the token is
written. The pin comparison is constant-time. Local token verification and
online scope authorization both use the pinned key. Tests cover wrong-key
substitution, missing pin, and a token signed by another issuer; no token file
is created on failure.

**Evidence.** `partial` — local mismatch tests exist and the new HW2 probe test
covers issuer substitution; production issuer continuity is recorded only in
`docs/KOYEB_CUTOVER_EVIDENCE.md`.

**Depends on.** None; this is the first trust gate.

### P0-002 — Make authority discovery and pairing-code issuance real or stop documenting them

**Why.** `README.md`, `docs/DEMO.md`, `docs/NANNET_API_HERMES.md` and
`docs/HERMES_CATALOG_ADMISSION.md` describe `GET /v1/issuer` and operator
pairing-code creation. The stdlib server in `nanlmind_hermes/http_api.py`
currently handles `/v1/pair`, `/v1/authorize`, `/v1/revoke`, device routes and
`/health`, but not `/v1/issuer` or `/v1/admin/pairing-codes`. A local operator
cannot follow the documented flow without importing Python objects directly.

**Files/tests.** `nanlmind_hermes/http_api.py`, `nanlmind_hermes/pairing.py`,
`nanlmind_hermes/authority_cli.py`, `scripts/nannet_hw2_connect.py`,
`tests/test_http_api.py`, `docs/NANNET_API_HERMES.md`, `docs/DEMO.md`.

**Acceptance.** Either (a) the local authority exposes an unauthenticated
`GET /v1/issuer` returning only the stable public key, plus an operator-only
pairing-code route protected by an explicit admin credential, with tests; or
(b) every document and script that claims these endpoints is corrected and the
operator flow is documented as CLI-only. The deployed host must expose the same
contract.

**Evidence.** `partial` — deployed service exposes the routes according to
`docs/LIVE_SMOKE.md`; the checked-in stdlib server does not.

**Depends on.** P0-001, P0-004.

### P0-003 — Bind the permanent HERMES endpoint to HW2/node point 3

**Why.** `HW2` is the permanent HERMES hardware endpoint at node point `3`;
`BS0:0` is the shared runner-ring port and `HW0:0` is the hardware root. A
dynamic `HW5–HW255` device lease, a local head index `3`, and node point `3`
are different namespaces. The current `DeviceRegistry` allocates only
`HW5–HW255`; `nannet_device_client.register()` rejects reserved slots; and
`router_universe_client._valid_receipt()` accepts `node_point` values `1..256`
instead of requiring `3` for HERMES.

**Files/tests.** `docs/NANNET_HW2_PRODUCTION.md`,
`scripts/nannet_hw2_connect.py`, `nanlmind_hermes/nannet_device_client.py`,
`nanlmind_hermes/devices.py`, `nanlmind_hermes/router_universe_client.py`,
`tests/test_nannet_hw2_connect.py`, `tests/test_nannet_device_client.py`,
`tests/test_router_universe_client.py`.

**Acceptance.** The authority has a reserved fixed-endpoint provisioning path
for this installation to `HW2:0`/node point `3`. The plugin never relabels a
dynamic lease as HW2. Router receipts for HERMES require
`node_point == 3`, `network_port == "BS0:0"`, `hardware_root == "HW0:0"`,
`heads == 255`, and `nannet_port_head == 255`. Negative tests cover `HW4`,
`HW5`, node point `4`, node point `256`, and a dynamic lease returned for a
fixed-HW2 request.

**Evidence.** `partial` — `docs/NANNET_HW2_PRODUCTION.md` and
`tests/test_nannet_hw2_connect.py` make the limitation explicit; the authority
path and router pinning are not implemented.

**Depends on.** P0-010.

### P0-004 — Make pairing codes one-time, short-lived and role-bounded

**Why.** A pairing code is an installation credential. Reuse, a long TTL, an
unbounded scope set, or a missing role ceiling turns a one-time enrolment into
a durable backdoor.

**Files/tests.** `nanlmind_hermes/pairing.py` (`create_pairing_code`, `pair`),
`nanlmind_hermes/http_api.py`, `nanlmind_hermes/authority_cli.py`,
`tests/test_pairing.py`, `tests/test_http_api.py`.

**Acceptance.** Codes are hashed at rest, expire in 60–3600 seconds, are
consumed transactionally, and are bound to a role whose scopes are checked
against the hard ceiling. Admin codes require an explicit admin request.
Concurrent consumption yields at most one success. Pairing codes, token
bearers, and master keys never appear in logs or responses.

**Evidence.** `connected` locally — `test_pairing.py` covers single use,
expiry, role separation and revocation. There is no HTTP operator route for
code creation (P0-002) and no concurrent-consumption stress test.

**Depends on.** P0-001, P0-008.

### P0-005 — Require signed authorize, revoke and rotate operations

**Why.** Bearer possession alone is not proof of the installation for
state-changing operations. The task contract calls for signed authorize,
revoke and rotate. `auth_tools.auth_revoke` and `require_scope` already sign
`/v1/authorize` and `/v1/revoke`, but the device routes
(`/v1/devices/rotate`, `/v1/devices/heartbeat`, `/v1/devices/release`) accept a
bearer and body without an installation signature.

**Files/tests.** `nanlmind_hermes/auth.py` (`sign_request`, `verify_request`),
`nanlmind_hermes/auth_tools.py`, `nanlmind_hermes/devices.py`,
`nanlmind_hermes/http_api.py`, `nanlmind_hermes/nannet_device_client.py`,
`tests/test_auth.py`, `tests/test_http_api.py`, `tests/test_devices.py`,
`tests/test_nannet_device_client.py`.

**Acceptance.** Every state-changing device route verifies a signed
method/path/timestamp/nonce/body digest or is explicitly documented as a
lower-assurance bearer-only route with a separate threat model. Old bearers
are revoked on rotate/release. Replay, stale timestamp and wrong-body tests
fail closed.

**Evidence.** `partial` — installation authorize/revoke are signed; device
rotate/heartbeat/release are bearer-only.

**Depends on.** P0-007.

### P0-006 — Enforce HTTPS, bare origins and no redirects on every client

**Why.** A token forwarded to a redirect target, a URL-embedded credential, or
plain HTTP leaks the installation identity. Loopback HTTP is acceptable only
for disposable tests.

**Files/tests.** `nanlmind_hermes/auth_tools.py`, `nanlmind_hermes/auth.py`,
`nanlmind_hermes/router_universe_client.py`,
`nanlmind_hermes/nannet_device_client.py`, `nanlmind_hermes/head_cortex.py`,
`nanlmind_hermes/clocked_client.py`, `nanlmind_hermes/http_api.py`,
`tests/test_nannet_device_client.py`, `tests/test_router_universe_client.py`,
`tests/test_head_cortex_evaluation.py`, `tests/test_http_api.py`.

**Acceptance.** HTTPS is required for every non-loopback origin. URLs with
credentials, path, query or fragment are rejected. Redirects are not followed
for authenticated requests. TLS 1.2+ is the minimum for the local server.
Certificate verification is not disabled. Tests cover `http://` non-loopback,
credentials-in-URL, a redirect, and a malformed origin.

**Evidence.** `partial` — device and router clients validate origins and
disable redirects; `auth_tools.auth_pair`/`auth_revoke` validate scheme but do
not reject a path or credentials as thoroughly; the local server TLS is
optional by design.

**Depends on.** P0-001.

### P0-007 — Enforce nonce, clock-skew and body-digest rules everywhere

**Why.** Signed requests are replayable if nonces are not consumed or if a
timestamp can be arbitrarily old/future. Token claims need the same treatment
for `nbf`, `iat` and `exp`.

**Files/tests.** `nanlmind_hermes/auth.py` (`sign_request`, `verify_request`,
`verify_token`, `TokenRegistry`), `nanlmind_hermes/pairing.py`
(`verify_signed_request`), `nanlmind_hermes/http_api.py`,
`tests/test_auth.py`, `tests/test_http_api.py`.

**Acceptance.** Maximum skew is 300 seconds; nonces are 16–128 characters and
consumed exactly once; body SHA-256 is checked before signature acceptance;
stale/future timestamps, nonce replay and body tampering fail closed. The
nonce table has bounded retention and a cleanup test. Token verification
rejects a token whose `nbf` is in the future or whose `exp` has passed.

**Evidence.** `partial` — nonce replay and body tamper are covered locally;
there is no explicit future/stale timestamp test, and `verify_token()` has no
maximum clock-skew parameter for `nbf`/`iat`.

**Depends on.** P0-001.

### P0-008 — Keep key material separate, encrypted and out of the environment

**Why.** NANNET master key, authority signing key, installation private key,
CLOCKED token and head-cortex token are different credentials. Mixing them,
reading environment variables, or leaving readable files defeats the authority
boundary.

**Files/tests.** `nanlmind_hermes/hermes_secrets.py`,
`nanlmind_hermes/auth.py`, `nanlmind_hermes/auth_tools.py`,
`nanlmind_hermes/authority_cli.py`, `nanlmind_hermes/clocked_tools.py`,
`nanlmind_hermes/head_cortex.py`, `plugin.yaml`,
`tests/test_credential_policy.py`, `tests/test_optional_crypto.py`.

**Acceptance.** The installation master key is read only through
`agent.secret_scope.get_secret()` in plugin code. No `.env`, shell or keychain
fallback exists. Installation private key and token envelopes are
AES-256-GCM/HKDF encrypted at rest with mode `0600`. The authority signing key
never leaves the authority DB. CLOCKED and head-cortex tokens are separate
secret names. A missing/incorrect master key fails closed without printing
key material. Rotation is documented (new identity + re-pair, not silent key
replacement).

**Evidence.** `connected` locally — credential policy and encrypted identity
tests exist; production secret injection and rotation are operator work.

**Depends on.** P0-006.

### P0-009 — Make token and lease expiry visible and actionable

**Why.** A paired token that expired silently causes every protected tool to
fail, while a stale lease can route work to the wrong generation. Operators
need a state, not a generic error.

**Files/tests.** `nanlmind_hermes/auth_tools.py` (`auth_status`),
`nanlmind_hermes/devices.py` (`status`, `heartbeat`),
`nanlmind_hermes/router_universe_client.py` (`status`, `_valid_receipt`),
`nanlmind_hermes/__init__.py` (`handle_status`),
`tests/test_auth_tools.py`, `tests/test_devices.py`,
`tests/test_router_universe_client.py`.

**Acceptance.** `nanlm_status` and `nanlm_auth_status` report `valid`,
`expiring`, `expired`, `revoked`, `missing` and `authority_unreachable` as
distinct states; include `expires_at`, remaining seconds and required scopes.
Lease status reports generation, revision, expiry and `locally_expired`.
Renewal and re-pair paths are tested with a fake clock. No state-changing tool
runs against an expired token or lease.

**Evidence.** `partial` — status reports presence and some expiry fields, but
there is no unified health state and no fake-clock renewal test.

**Depends on.** P0-007.

### P0-010 — Keep one issuer and one writable Koyeb writer

**Why.** Two issuers or two writers produce split-brain tokens, leases and
canonical generations. The Koyeb cutover evidence is only useful if the same
issuer fingerprint and single `/data` writer are enforced on every rollout.

**Files/tests.** `docs/KOYEB_CUTOVER_EVIDENCE.md`,
`docs/HERMES_NANNET_KOYEB_MIGRATION.md`, `docs/NANNET_HW2_PRODUCTION.md`,
`scripts/live_smoke.py`, deployment configuration outside this repository,
`tests/test_auth_tools.py`, `tests/test_pipeline_integration.py`.

**Acceptance.** Release record names the Koyeb app/service/deployment ID, the
issuer key fingerprint, the source commit and the `/data` volume ID. A probe
fails if `/v1/issuer` differs from the pin. Rollback is documented as
deployment-image rollback and never replays stale leases. No staging service
mounts the production `/data` volume. A second issuer or second writer is a
release blocker.

**Evidence.** `partial` — a live cutover is documented (2026-11-08), but it is
not reproducible from this repository alone and the local stdlib issuer
discovery route is missing (P0-002).

**Depends on.** P0-001, P0-002.

### P0-011 — Pin every router lease receipt to the HERMES endpoint and head namespace

**Why.** A receipt that is valid but bound to node point `7`, an extra head,
or a different installation is not a HERMES lease. `_valid_receipt()` currently
bounds `node_point` to `1..256` and accepts `receipt_mac` without comparing the
value to a trusted router key.

**Files/tests.** `nanlmind_hermes/router_universe_client.py`
(`_valid_receipt`, `open_universe`, `resolve`, `renew`, `release`),
`tests/test_router_universe_client.py`.

**Acceptance.** Every receipt requires `installation_id`, `node_point == 3`,
`network_port == "BS0:0"`, `hardware_root == "HW0:0"`, `heads == 255`,
`nannet_port_head == 255`, positive `lease_id`/`universe_id`/`generation`/
`revision`, and a future `expires_at`. Renewal must not change lease identity.
Tests reject `node_point=4`, `node_point=256`, `heads=256`, a foreign
installation, a stale generation and an expired receipt.

**Evidence.** `partial` — tests exercise node `3` and reject `0`; the code
still accepts other points and does not verify the MAC locally.

**Depends on.** P0-003, P0-010.

### P0-012 — Make N16T dispatch, ACK and inspection at-most-once and bound

**Why.** Retrying an ambiguous dispatch after a crash can execute guest work
twice. Exposing a response with a wrong digest is worse than failing.

**Files/tests.** `nanlmind_hermes/router_universe_client.py` (`dispatch`,
`acknowledge`, `inspect_dispatch`, `math_u32`),
`tests/test_router_universe_client.py`,
`docs/HERMES_NANNET_LEASE_UNIVERSES.md`.

**Acceptance.** Stable `dispatch_id` is required; request and response SHA-256
are verified; `not_recorded`, `pending_or_ambiguous`, `completed` and
`acknowledged` are distinct; ambiguous work is never auto-retried; response
frames are bound to installation, lease, universe, generation and head;
`model_training_committed` is always false. Tests cover tampered response,
cross-head replay, stale receipt and unknown dispatch.

**Evidence.** `connected` locally — tests cover ack, digest mismatch,
cross-head inspection and native MTH result binding. Hosted router execution
was not available in this audit.

**Depends on.** P0-011.

## Authorization and scope

### P0-013 — Prove scope coverage on every protected handler

**Why.** One unscoped training or write handler is a full authorization
bypass when an authority is configured. The previous audit found
`nanlm_curriculum_experiment` and curriculum generation without scope checks;
the current working tree adds checks, but there is no generated coverage test
that prevents a future handler from omitting one.

**Files/tests.** `nanlmind_hermes/__init__.py` (`register`, all `handle_*`
functions), `nanlmind_hermes/auth_tools.py` (`require_scope`),
`tests/test_manifest_registration.py`, `tests/test_pipeline_integration.py`,
`tests/test_session_context.py`.

**Acceptance.** A checked-in table classifies every registered tool as
`read`, `write`, `identity`, `network` or `local-only`, with the required
scope. A test introspects the registered handlers and fails if a non-exempt
handler has no `_require_scope` call. With a configured authority, an unpaired
or wrong-scope call fails; with no authority, it returns explicit
`local-unverified`. Exemptions are explicit (`nanlm_status`,
`nanlm_session_context`, `nanlm_auth_*`) and justified.

**Evidence.** `partial` — current code scopes training/write handlers and
curriculum, but `handle_chat_history`, `handle_status`, `handle_mth_hub`,
`handle_mth_leaf_get` and `handle_mth_leaf_verify` have no scope check, and
there is no coverage test.

**Depends on.** P0-001.

### P0-014 — Decide and enforce the read surface

**Why.** Transcripts, status paths, leaf payloads and experiment metadata can
contain user content or sensitive operational detail. “Read-only” is not the
same as “public”.

**Files/tests.** `nanlmind_hermes/__init__.py` (`handle_chat_history`,
`handle_status`, `handle_session_context`, `handle_mth_hub`,
`handle_mth_leaf_get`, `handle_mth_leaf_verify`),
`tests/test_session_context.py`, `tests/test_pipeline_integration.py`.

**Acceptance.** When an authority is configured, `nanlm_chat_history` requires
`universe:read`; `nanlm_status` reports only redacted operational data without
that scope; MTH leaf reads require `universe:read` unless the payload is
explicitly public. `nanlm_session_context` remains readable without a paired
identity only because its content is shipped package text. Tests cover each
choice and the no-authority `local-unverified` path.

**Evidence.** `open` for chat history / status / leaf reads; `nanlm_session_context`
is intentionally public.

**Depends on.** P0-013.

### P0-015 — Make `merge_enabled=false` block every promotion path

**Why.** The documented contract says only `nanlm_learning_sync` promotes, but
`nanlm_learning_cycle` and `nanlm_wordnet_ingest` accept `approved=true`.
The current working tree adds a `merge_enabled` gate and returns
`merge_disabled`; the gate and all paths need tests, because this is the
difference between derived scratch and a false merge claim.

**Files/tests.** `nanlmind_hermes/__init__.py` (`_merge_enabled`,
`handle_learning_cycle`, `handle_learning_sync`, `handle_wordnet_ingest`),
`nanlmind_hermes/layered_learning.py` (`cycle`, `synchronize`),
`plugin.yaml` (`merge_enabled`), `tests/test_layered_learning.py`,
`tests/test_pipeline_integration.py`, `scripts/gemma_training_series.py`.

**Acceptance.** With `merge_enabled=false`, all three tools return
`status: merge_disabled`, `promoted: 0` and do not modify
`mth_heads.json` or the sync journal, even with `approved=true`.
`merge_enabled=true` still cannot perform a canonical merge; it only enables
local derived promotion and must report that distinction. Tests assert the
argument, not just the tool name. The Gemma safe-tool surface cannot call any
promotion path.

**Evidence.** `partial` — gate exists in the current tree; no test covers the
three-tool matrix or the safe surface.

**Depends on.** P0-013.

### P0-016 — Keep canonical merge explicitly OFF and name it unambiguously

**Why.** “Merge” is used for the local idempotent event union
(`nanlm_memory_merge`), local head-register promotion, and a nonexistent
canonical BS0/MC0 merge. A user or maintainer can read any of those as a
canonical commit.

**Files/tests.** `nanlmind_hermes/tools.py` (`status`),
`nanlmind_hermes/__init__.py` (`handle_status`),
`nanlmind_hermes/memory.py` (`merge`), `docs/PLUGIN_ARCHITECTURE.md`,
`docs/HERMES_UNIVERSE_MEMORY.md`, `docs/NANVERSE_LAYERED_LEARNING.md`,
`tests/test_pipeline_integration.py`, `tests/test_merge_contract.py`.

**Acceptance.** `nanlm_status` exposes separate keys
`canonical_merge_enabled=false`, `local_promotion_enabled`,
`local_event_union_enabled`; no tool is named or described as canonical merge;
`nanlm_memory_merge` is documented as local event-set union only. The manifest
contains no canonical merge handler. A test asserts the three keys and the
absence of a merge handler.

**Evidence.** `connected` by absence — no canonical merge tool exists and
status reports false; the naming ambiguity remains.

**Depends on.** P0-015.

### P0-017 — Add a promotion/merge audit trail

**Why.** An approval without an immutable record cannot be reviewed,
reproduced, or rolled back. The current local sync writes a JSONL report, but
it lacks actor, base digest, candidate digest, policy version and a stable
event ID.

**Files/tests.** `nanlmind_hermes/layered_learning.py`
(`synchronize`, `language/layered_learning_sync.jsonl`),
`nanlmind_hermes/memory.py`, `nanlmind_hermes/tools.py`,
`tests/test_layered_learning.py`, `tests/test_merge_contract.py`.

**Acceptance.** Every promotion, refusal, dry-run and merge attempt appends an
event with `event_id`, timestamp, actor/tool, `base_generation`, `base_digest`,
`candidate_digest`, policy version, approval flag, result and error code.
Records are append-only, fsync’d, content-addressed and queryable. No secret or
raw token is written. A test replays the journal and reconstructs the promoted
set.

**Evidence.** `partial` — sync JSONL exists; no merge audit, actor, base
generation or replay test.

**Depends on.** P0-015, P0-018.

### P0-018 — Require generation compare-and-swap for promotion

**Why.** Two concurrent promotions can silently overwrite each other. The
chart storage already has a two-slot generation lease; the layered promotion
does not require a base generation.

**Files/tests.** `nanlmind_hermes/chart_storage.py` (`publish` generation
lease), `nanlmind_hermes/layered_learning.py` (`synchronize`),
`nanlmind_hermes/scratch_register.py`, `tests/test_chart_storage.py`,
`tests/test_layered_learning.py`.

**Acceptance.** A promotion request carries `expected_generation` and
`expected_root`. A stale value fails with a stable code before any write.
On success the generation advances exactly once. Concurrent writers: exactly
one succeeds; the other reloads and reports stale. Test uses two coordinators
over one state directory.

**Evidence.** `partial` — chart snapshot CAS exists and is tested; promotion
has no CAS.

**Depends on.** P0-015.

### P0-019 — Keep the six-runner barrier absent and fail closed

**Why.** The canonical promotion path is supposed to require six independent
runner generations and a barrier. No such implementation exists in this
repository. Leaving an open “promotion” API without that guard invites a
future caller to treat local approval as canonical.

**Files/tests.** `docs/PLUGIN_ARCHITECTURE.md`,
`docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md`,
`docs/HERMES_UNIVERSE_MEMORY.md`, `nanlmind_hermes/__init__.py`,
`tests/test_pipeline_integration.py`.

**Acceptance.** No handler performs canonical promotion. Every status and
receipt says `six_runner_barrier: absent` and `canonical_commit: false`.
A future barrier implementation must require six distinct runner signatures,
matching generation, and a rollback pointer before any canonical write.
A test asserts the absence now and a contract test would gate the future
implementation.

**Evidence.** `open` — documented as outstanding, no implementation.

**Depends on.** P0-018.

### P0-020 — Add dry-run, explicit approval and rollback semantics

**Why.** Approval is not a substitute for a reviewable diff. Operators need a
dry-run that performs no write, and a rollback pointer that restores the prior
local state.

**Files/tests.** `nanlmind_hermes/layered_learning.py` (`synchronize`),
`nanlmind_hermes/chart_storage.py`, `nanlmind_hermes/__init__.py`,
`tests/test_layered_learning.py`, `tests/test_chart_storage.py`.

**Acceptance.** `dry_run=true` returns candidate count, would-promote digest
and no filesystem mutation. Promotion requires `approved=true` and a matching
base generation. A rollback record identifies the previous register/chart
snapshot and restores it exactly. A replay of the same request is a no-op with
an identical result digest.

**Evidence.** `partial` — `approved` exists; no dry-run or rollback pointer.

**Depends on.** P0-017, P0-018.

### P0-021 — Separate local derived state from canonical state in every API and document

**Why.** The local 8 MiB head register, 1 GiB scratch, memory ledger and
CLOCKED candidate are derived/local. Canonical NaNLMind heads are a different
owner. A shared word (“merge”, “commit”, “state digest”) hides that boundary.

**Files/tests.** `docs/PLUGIN_ARCHITECTURE.md`,
`docs/NANVERSE_LAYERED_LEARNING.md`, `docs/HERMES_UNIVERSE_MEMORY.md`,
`nanlmind_hermes/__init__.py`, `nanlmind_hermes/chat.py`,
`nanlmind_hermes/layered_learning.py`, `tests/test_pipeline_integration.py`.

**Acceptance.** Every receipt has `state_scope: local-derived` or
`canonical`, and `canonical_commit: false` for all current paths. No field
named `state_digest` is presented as a canonical digest; use
`local_state_digest` and `canonical_state_digest: null`. A test scans result
schemas for ambiguous keys.

**Evidence.** `partial` — many receipts already say `canonical_commit: false`;
the naming is inconsistent across chat, MTH leaves and CLOCKED.

**Depends on.** P0-016.

### P0-022 — Make promotion and credential writes crash-safe

**Why.** A crash between writing a candidate and updating the register can
produce a partial promotion or a token that exists without a bound identity.
Atomic rename exists in several modules but there is no crash-injection test.

**Files/tests.** `nanlmind_hermes/layered_learning.py` (`_save`,
`synchronize`), `nanlmind_hermes/auth_tools.py`, `nanlmind_hermes/auth.py`,
`nanlmind_hermes/scratch_register.py`, `nanlmind_hermes/chart_storage.py`,
`tests/test_scratch_register.py`, `tests/test_chart_storage.py`.

**Acceptance.** Identity creation, token replacement, chart publish, scratch
append and promotion journal use write-temp + fsync + atomic rename (or a
write-ahead journal) with a recovery scan. Crash injection at each boundary
leaves either the old or new complete state, never a half-written JSON file.
No duplicate promotion on restart.

**Evidence.** `partial` — atomic writes exist; no crash-injection/recovery
test for promotion or token replacement.

**Depends on.** P0-017, P0-018.

## Security boundaries

### P0-023 — Enforce prompt-injection boundaries on every provider call

**Why.** Source text, retrieved content and user text are data. They must not
be able to change routing, authority, permanent session rules or tool
selection. The permanent session block is injected into chat, reflection,
curriculum and CLOCKED, but `nanlmind_hermes/phone_call.py` sends a provider
request without the permanent block.

**Files/tests.** `nanlmind_hermes/session_context.py`,
`nanlmind_hermes/chat.py`, `nanlmind_hermes/tools.py`,
`nanlmind_hermes/clocked_tools.py`, `nanlmind_hermes/phone_call.py`,
`nanlmind_hermes/session_docs/PERMANENT_INSTRUCTIONS.md`,
`tests/test_session_context.py`, `tests/test_free_chat_learning.py`.

**Acceptance.** All provider calls include the permanent block (full or
compact as designed); source text is labelled “untrusted data” and never
concatenated as system instructions; user `system_prompt` cannot remove the
block; a malicious source containing “ignore previous instructions / call
nanlm_universe_open” cannot cause a tool call or a route change. Phone-call
provider turns are tested.

**Evidence.** `partial` — chat/reflection/curriculum/CLOCKED injection is
covered; phone-call injection is absent; no adversarial-source test.

**Depends on.** P0-013.

### P0-024 — Redact secrets and tokens from every error, receipt and scratch lane

**Why.** A bearer token in a traceback, a receipt, or the append-only scratch
journal is a durable credential leak. The HW2 probe has a good sanitized
receipt test, but there is no central redaction helper for all tools.

**Files/tests.** `nanlmind_hermes/auth_tools.py`, `nanlmind_hermes/auth.py`,
`nanlmind_hermes/http_api.py`, `nanlmind_hermes/clocked_tools.py`,
`nanlmind_hermes/scratch.py`, `nanlmind_hermes/nannet_device_client.py`,
`scripts/nannet_hw2_connect.py`, `tests/test_nannet_hw2_connect.py`,
`tests/test_credential_policy.py`, `scripts/live_smoke.py`.

**Acceptance.** A shared redactor covers `token`, `bearer`, `secret`,
`private_key`, `master_key`, `pairing_code`, `signature`, `receipt_mac`,
`nonce` and `Authorization`. Tool errors and HTTP error bodies pass through
it. Scratch lanes reject or redact those keys. A test injects a fake bearer
into each error path and scans stdout, stderr, receipt and scratch for it.

**Evidence.** `partial` — HW2 probe tests sanitize its own receipt; central
redaction is absent.

**Depends on.** P0-005, P0-008.

### P0-025 — Add an append-only security audit log

**Why.** Incident response needs to know who paired, revoked, rotated,
attempted a denied scope, approved a promotion, or hit `merge_disabled`.

**Files/tests.** new `nanlmind_hermes/audit.py`,
`nanlmind_hermes/auth_tools.py`, `nanlmind_hermes/pairing.py`,
`nanlmind_hermes/http_api.py`, `nanlmind_hermes/__init__.py`,
`tests/test_auth_tools.py`, `tests/test_pipeline_integration.py`.

**Acceptance.** Events include pairing, unpair, revoke, rotate, scope denial,
authority failure, approval, merge-disabled refusal and local promotion.
Records have timestamp, actor/tool, installation ID hash, event type, result
and stable error code; no secret or raw bearer. Append-only with fsync and
bounded retention. A test inspects the event sequence after a paired demo.

**Evidence.** `open` — no security audit module exists.

**Depends on.** P0-024.

## Packaging and release

### P0-026 — Commit and package every runtime bundle

**Why.** `nanlmind_hermes/mth/`, `nanlmind_hermes/nan0s/`,
`nanlmind_hermes/session_docs/`, `session_context.py`, `mth_catalog.py`,
`mth_leaves.py`, `nan0s_runtime.py` and the new tests were untracked during the
audit. An sdist built from Git without them is not the tested plugin.

**Files/tests.** `MANIFEST.in`, `pyproject.toml` (`package-data`),
`nanlmind_hermes.egg-info/SOURCES.txt`, `build/lib/`,
`tests/test_manifest_registration.py`, `.github/workflows/test.yml`
(`release-package`).

**Acceptance.** All listed files are tracked in Git. `MANIFEST.in` includes
`*.nas`, `.awk`, `.c`, `.h`, `.inc`, `.json` and session docs. A clean
`python -m build` produces wheel and sdist that each contain
`mth/spec/wave-capabilities.json`, 91 `mth/nan16` kernels, `mth/tests`,
`nan0s/core`, `nan0s/tools`, `nan0s/src/tools`, `nan0s/src/tests`,
`nan/*.nan`, 7 `session_docs/*.md`, and the plugin entry files. The stale
`egg-info`/`build/` output is not treated as evidence.

**Evidence.** `partial` — `MANIFEST.in` now includes `*.nas`; the bundles are
still untracked, and `egg-info` contains no MTH files.

**Depends on.** None; packaging track starts here.

### P0-027 — Make release CI inspect the actual archives and install them cleanly

**Why.** Building a wheel is not enough if the MTH binaries, session docs or
package data are missing. The current release job checks a short required-file
list and imports `register`, but does not assert the 91-kernel bundle.

**Files/tests.** `.github/workflows/test.yml` (`release-package`),
`pyproject.toml`, `MANIFEST.in`.

**Acceptance.** CI opens both `.whl` and `.tar.gz`, asserts exact counts for
91 kernels, 44 families and 7 session docs, verifies SHA-256 of
`wave-capabilities.json`, installs each into an empty target with
`--no-deps`, runs `from nanlmind_hermes import register` and a catalog load,
and checks `twine check --strict`. A clean `git archive` build is also tested.

**Evidence.** `partial` — release job exists; missing bundle assertions.

**Depends on.** P0-026.

### P0-028 — Define and test the supported Python and OS matrix

**Why.** `pyproject.toml` requires Python `>=3.10` but classifiers stop at
3.12; CI covers 3.10–3.12; the current host has 3.13 bytecode; Alpine/Termux
evidence is a manually recorded 344-test run. Production support must match
the matrix.

**Files/tests.** `pyproject.toml`, `.github/workflows/ci.yml`,
`.github/workflows/python-tests.yml`, `.github/workflows/test.yml`,
`docs/ALPINE_PROOT_TESTING.md`, `docs/HERMES_PLUGIN_INSTALL.md`.

**Acceptance.** The supported matrix is explicit (for example 3.10–3.13) and
CI runs it. A Termux/aarch64 job and an Alpine/proot job install the package,
build the portable NaN0S core, run the suite, and record skip reasons. A
missing compiler/awk is a skip with a visible status, not a silent pass.
Python 3.13 is added or explicitly declared unsupported.

**Evidence.** `partial` — 3.10–3.12 CI; Termux/Alpine are manual; 3.13
untested.

**Depends on.** P0-027.

## Test evidence and integrity

### P0-029 — Make test and count claims reproducible

**Why.** README says 344 tests; `ARCHITECTURE_GAPS` says 312; older docs say
272; the static snapshot has 376 methods in 66 files. A release cannot claim a
number it cannot reproduce.

**Files/tests.** `README.md`, `CHANGELOG.md`,
`docs/ARCHITECTURE_GAPS.md`, `docs/CONNECTION_AUDIT.md`,
`docs/ALPINE_PROOT_TESTING.md`, `docs/LOCAL_RUNNER.md`,
`docs/CORPUS_MEMORY.md`, `docs/PHONE_CALL_NAN16.md`,
`tests/test_manifest_registration.py`.

**Acceptance.** One generated manifest records `python_version`, test count,
tool count, kernel count, family count and session-doc count. CI fails if a
documented count differs. The release notes link the exact command and output.
Historical counts are marked historical.

**Evidence.** `open` — counts are inconsistent.

**Depends on.** P0-026.

### P0-030 — Gate on all 91 native fixtures and the N16I parity set

**Why.** “91/91 cataloged” is not “91/91 executed”. The tests verify catalog,
tree images and one native fixture plus four vector N16I parity cases. The
remaining 87 kernels have fixture sources but no full per-kernel execution
evidence in this audit.

**Files/tests.** `nanlmind_hermes/nan0s_runtime.py`,
`nanlmind_hermes/mth_catalog.py`, `tests/test_mth_runtime.py`,
`tests/test_n16i_unification.py`, `scripts/mth_function_tree_report.py`,
`nanlmind_hermes/mth/spec/wave-capabilities.json`.

**Acceptance.** CI runs every one of the 91 native fixtures with a bounded
step budget, records PASS/FAIL per kernel, and fails on any missing or trap.
The four admitted vector kernels are compared native-vs-N16I on fixed edge
operands (zeros, equal, opposite, max finite, NaN/Inf rejection). The 87
fixture-only kernels are explicitly reported as “fixture verified, no
general operand ABI”. No Python fallback is used for a claimed native result.

**Evidence.** `partial` — catalog and tree audit exist; one native fixture and
four N16I parity tests were inspected; no 91-fixture CI artifact was found.

**Depends on.** P0-028.

### P0-031 — Fix data-leaf denial, audit and alias invariants

**Why.** `mth_leaves` is a safety layer: an unknown leaf must deny, not raise
outside the `try`; a dangling `kernel:` alias must fail audit; and the hub tool
must return the advertised tree. Those are permission and audit defects, not
cosmetic issues.

**Files/tests.** `nanlmind_hermes/mth_leaves.py` (`pair`,
`function_tree_audit`, `_new_leaf`, `status`),
`tests/test_mth_leaves.py`, `tests/test_pipeline_integration.py`,
`docs/MTH_DATA_LEAVES.md`.

**Acceptance.** `pair()` returns `status: denied` with `unknown_leaf` for an
unresolved input. `function_tree_audit` fails on a `None` leaf, a dangling
alias, a missing on-disk image or a digest mismatch. `nanlm_mth_hub` returns
the hub tree, not only counts. A duplicate-content new alias is retained or
explicitly refused. Tests cover `unknown_leaf`, `operator_not_admitted`,
`native_execution_failed`, `invariant_violation` and `identity_violation`.

**Evidence.** `partial` — tests cover illegal/unknown/hub denial; the code
still resolves inputs outside the try and skips `None` leaves in audit.

**Depends on.** P0-030.

### P0-032 — Automate loopback and production NANNET/router probes

**Why.** The live smoke record is useful but not a repeatable release gate.
A production claim needs a probe that fails on issuer mismatch, wrong node
point, stale lease, scope denial or secret leakage.

**Files/tests.** `scripts/live_smoke.py`, `scripts/nannet_hw2_connect.py`,
`tests/test_pipeline_integration.py`, `tests/test_nannet_hw2_connect.py`,
`tests/test_router_universe_client.py`,
`docs/HERMES_CATALOG_ADMISSION.md`, `docs/LIVE_SMOKE.md`,
`docs/KOYEB_CUTOVER_EVIDENCE.md`.

**Acceptance.** CI runs a real loopback `PairingAuthority` + HTTP server for
pair/authorize/revoke/HW2-probe/router-receipt flows. An opt-in production job
runs the same probe against the pinned host, writes a sanitized receipt with
deployment ID and issuer fingerprint, and fails on any mismatch. The receipt
contains no bearer, key, nonce or pairing code. Loopback evidence is never
labelled production.

**Evidence.** `partial` — loopback tests and a recorded 11/11 live smoke
exist; no automated production gate.

**Depends on.** P0-001–P0-012.

### P0-033 — Add crash-recovery tests for state, token, scratch and chart

**Why.** Production runs restart. The current atomic writes are necessary but
not sufficient evidence that a crash at each boundary recovers.

**Files/tests.** `nanlmind_hermes/auth.py`, `nanlmind_hermes/auth_tools.py`,
`nanlmind_hermes/chart_storage.py`, `nanlmind_hermes/scratch_register.py`,
`nanlmind_hermes/layered_learning.py`,
`tests/test_chart_storage.py`, `tests/test_scratch_register.py`,
`tests/test_auth.py`.

**Acceptance.** A fault-injection harness kills the process after temp write,
after fsync, before rename, after rename and during journal append. On restart:
identity and token are either old-complete or new-complete; chart slot
generation is monotonic; scratch page digest verifies; promotion is applied
once or not at all. The test reports a stable recovery code.

**Evidence.** `partial` — atomic rename and dual-slot chart are implemented;
no crash injection test.

**Depends on.** P0-022.

---

# Recommended P0 execution order

1. **Trust roots:** P0-001, P0-008, P0-010.
2. **Transport and replay:** P0-005, P0-006, P0-007.
3. **Endpoint identity:** P0-002, P0-003, P0-004, P0-011, P0-012.
4. **Authorization:** P0-009, P0-013, P0-014.
5. **Merge-off contract:** P0-015, P0-016, P0-017, P0-018, P0-019, P0-020,
   P0-021, P0-022.
6. **Security and packaging:** P0-023, P0-024, P0-025, P0-026, P0-027,
   P0-028.
7. **Evidence gates:** P0-029, P0-030, P0-031, P0-032, P0-033.

Do not mark a later item complete by testing only its happy path; each P0
acceptance includes a negative case.

---

# P1 — production hardening

Each item is stated as: **why**, **files/tests**, **acceptance**, **evidence**,
**depends on**.

### P1-001 — Add an aggregate diagnostics command

**Why.** Operators need one redacted report for disk, state permissions,
token/lease state, scratch integrity, MTH runtime, WordNet archive and merge
flags.
**Files/tests.** `nanlmind_hermes/tools.py` (`status`),
`nanlmind_hermes/__init__.py` (`handle_status`), `tests/test_pipeline_integration.py`.
**Acceptance.** `nanlm_status` returns schema version, component health,
last error codes, disk free, state path, token/lease expiry, scratch verify,
catalog health and `canonical_merge_enabled=false`; no secrets.
**Evidence.** `partial` — many fields exist, no aggregate health/error model.
**Depends on.** P0-009, P0-025.

### P1-002 — Provide backup and restore for identity, leases and local state

**Why.** Losing `installation.json`, the encrypted token or the state directory
forces re-identification and destroys local learning evidence.
**Files/tests.** `nanlmind_hermes/auth.py`, `auth_tools.py`,
`router_universe_client.py`, `layered_learning.py`, new backup script.
**Acceptance.** Backup includes identity, encrypted token, leases, memory DB,
scratch pages and manifests with SHA-256; restore verifies permissions and
refuses to overwrite a live newer generation. Keys are never included in a
plaintext archive.
**Evidence.** `open` — no backup/restore tool.
**Depends on.** P0-008, P0-018.

### P1-003 — Add structured observability

**Why.** Local logs are absent; failures are Python exceptions. Production
needs request IDs, durations, outcome codes and counters without transcripts.
**Files/tests.** new `nanlmind_hermes/observability.py`, `auth_tools.py`,
`http_api.py`, `clocked_tools.py`, `__init__.py`.
**Acceptance.** Structured JSON events for tool start/end/error, auth scope,
pairing, lease, dispatch, promotion and CLOCKED call; no source text or secret;
configurable sink; tests assert fields and redaction.
**Evidence.** `open`.
**Depends on.** P0-024, P0-025.

### P1-004 — Set rate and size limits on every ingress

**Why.** A local HTTP or tool loop can exhaust memory/CPU or spam a paid
gateway. Existing bounds are per-argument, not per-caller or per-window.
**Files/tests.** `http_api.py`, `clocked_tools.py`, `head_cortex.py`,
`__init__.py`, `tools.py`, `tests/test_http_api.py`.
**Acceptance.** Per-installation request rate, concurrent request cap, body
size, source count, message size, CLOCKED request/byte budgets, and tool
timeouts. Exceeded limits return a stable code and do not call the backend.
**Evidence.** `partial` — many static size limits; no rate limiter.
**Depends on.** P0-013.

### P1-005 — Write a failure-mode runbook

**Why.** Operators need exact behavior for no authority, expired token,
revoked identity, unreachable authority, no runner, missing WordNet archive,
corrupt scratch, full disk, stale lease and CLOCKED outage.
**Files/tests.** `docs/WALKTHROUGHS.md`, `docs/ALPINE_PROOT_TESTING.md`,
`docs/HERMES_PLUGIN_INSTALL.md`, new `docs/OPERATIONS.md`.
**Acceptance.** Each mode has detection, user-visible message, recovery step,
data-loss risk and test reference.
**Evidence.** `partial` — behavior is scattered across code/docs.
**Depends on.** P0-009, P1-001.

### P1-006 — Define authorization latency, caching and retry semantics

**Why.** `require_scope` makes an online call on every protected operation.
This is correct fail-closed behavior but can be slow or unavailable; caching
must not create a bypass.
**Files/tests.** `auth_tools.py`, `__init__.py`, `tests/test_auth_tools.py`.
**Acceptance.** No positive cache without a bounded TTL and revocation check;
negative cache is disallowed; timeout is short and explicit; retry is not
automatic for non-idempotent tools; tests cover authority down and slow.
**Evidence.** `open`.
**Depends on.** P0-007, P0-013.

### P1-007 — Finish the fixed-HW2 versus dynamic-device lifecycle

**Why.** `--register-hw2` cannot bind against the current authority; dynamic
HW5–HW255 leases are a different object. Operators need one documented state
machine.
**Files/tests.** `docs/NANNET_HW2_PRODUCTION.md`,
`nannet_device_client.py`, `devices.py`, `router_universe_client.py`,
`tests/test_nannet_hw2_connect.py`.
**Acceptance.** States: unpaired → paired → fixed-HW2-bound → leased universe;
each transition has an API, receipt and negative test. Dynamic leases cannot
be called HW2. Release is idempotent.
**Evidence.** `partial` — probe and docs; no fixed-HW2 authority path.
**Depends on.** P0-003.

### P1-008 — Make the router lease state machine explicit

**Why.** `open`, `resolve`, `renew`, `release` and local pending state are
implemented, but retry/expiry/ambiguous behavior is not summarized or fully
tested.
**Files/tests.** `router_universe_client.py`,
`tests/test_router_universe_client.py`,
`docs/HERMES_NANNET_LEASE_UNIVERSES.md`.
**Acceptance.** State diagram with `pending`, `open`, `resolved`, `expired`,
`released`, `ambiguous`; request IDs persist across transport errors; renew
advances revision; release erases local receipt only after confirmation.
**Evidence.** `partial` — tests cover pending ID, renew, release, stale.
**Depends on.** P0-011.

### P1-009 — Verify or explicitly trust the router receipt MAC

**Why.** The client checks shape and binding but not the HMAC. That is
documented as trusted delegation, but the trust boundary needs a concrete
verification story (router public key, HMAC key distribution, or NANNET
attestation).
**Files/tests.** `router_universe_client.py`,
`docs/HERMES_NANNET_KOYEB_MIGRATION.md`.
**Acceptance.** Either verify a router signature locally with a pinned key or
document the exact trusted channel and fail if it is absent. A tampered MAC
must not be treated as a valid lease.
**Evidence.** `open` — acknowledged in docs.
**Depends on.** P0-011, P0-012.

### P1-010 — Define typed operand ABIs for the 87 fixture-only kernels

**Why.** `nanlm_mth_execute` only covers dot/cosine/add/sub. The other 87 are
fixture-verifiable, not callable with typed inputs. The function tree should
state this per kernel.
**Files/tests.** `mth_catalog.py`, `nan0s_runtime.py`, `mth/spec/wave-capabilities.json`,
`tests/test_mth_runtime.py`, `docs/MTH_UNIVERSAL_RUNTIME.md`.
**Acceptance.** Each kernel descriptor declares `abi: none|fixture_only|typed`,
input/output shapes, workspace and fuel. `mth_execute` refuses non-typed
kernels with a stable code. A typed ABI adds a differential test.
**Evidence.** `partial` — metadata says fixture-only; no per-kernel ABI.
**Depends on.** P0-030.

### P1-011 — Harden arbitrary N16I dispatch

**Why.** `nanlm_mth_dispatch` runs assembled guest images; arbitrary code
needs clear entry, memory, step and output boundaries.
**Files/tests.** `nan0s_runtime.py` (`verify_kernel_n16i`),
`tests/test_mth_runtime.py`, `docs/MTH_UNIVERSAL_RUNTIME.md`.
**Acceptance.** Entry point, guest base, max steps, output window and trap
codes are explicit; a malformed image fails before execution; a runaway image
hits the step bound; no host Python math is substituted.
**Evidence.** `connected` locally — dispatch test exists; fuzz/malformed-image
coverage is thin.
**Depends on.** P0-030.

### P1-012 — Fix MTH hub/tree inspection and aliases

**Why.** The hub tool returns a summary, not the tree; duplicate content with a
new alias may lose the alias. Operators cannot inspect the advertised object.
**Files/tests.** `mth_leaves.py` (`status`, `_new_leaf`),
`tests/test_mth_leaves.py`, `docs/TOOLS_REFERENCE.md`.
**Acceptance.** Hub output includes root, children, leaf counts and immutable
digests; alias creation is retained or explicitly refused; a dangling alias
fails audit.
**Evidence.** `partial` — known gaps in `docs/CONNECTION_AUDIT.md`.
**Depends on.** P0-031.

### P1-013 — Implement or narrow the WordNet HW4 route contract

**Why.** `session_docs/WORDNET_HW4.md` says HERMES reads WordNet through
NANNET; the code reads a local zip. `bidirectional=True` is unbacked and
`depth` is validated but unused.
**Files/tests.** `dictionary_universe.py`, `wordnet_store.py`,
`layered_learning.py` (`ingest_word`), `tests/test_wordnet_tool.py`.
**Acceptance.** Either implement a signed HW4 read/write route with generation
and ACL, or change the contract to “local archive read, no live route”.
`depth` bounds traversal. `bidirectional` is removed or backed by a write path.
**Evidence.** `partial` — canonical route constants and local ingest tests;
live route and depth are open.
**Depends on.** P0-011.

### P1-014 — Fix WordNet archive configuration and error text

**Why.** `wordnet_store.discover()` says `NANLM_WORDNET_ARCHIVE`, while the
plugin setting is `wordnet_archive`. A user following the error cannot fix it.
**Files/tests.** `wordnet_store.py`, `plugin.yaml`, `README.md`,
`tests/test_wordnet_tool.py`.
**Acceptance.** Error names `wordnet_archive` and the expected path; discovery
order is documented; missing archive fails closed without inventing a word.
**Evidence.** `partial` — fail-closed test exists; message mismatch.
**Depends on.** P1-013.

### P1-015 — Preserve relation polarity and confidence in layered learning

**Why.** `_relation()` strips polarity/confidence, so a supported and a
refuted triple collapse to one promotion candidate. Layer leaves embed only
`relation_list[0]`, so multi-relation turns lose evidence.
**Files/tests.** `layered_learning.py` (`_relation`, `observe`,
`_candidate_rows`), `tests/test_layered_learning.py`,
`tests/test_free_chat_learning.py`.
**Acceptance.** Every relation record retains polarity, confidence, source
digest and resolution; every relation gets its own scratch leaf or a
documented bounded list; promotion groups by polarity and refuses to merge
opposites.
**Evidence.** `partial` — current code drops polarity and only first relation.
**Depends on.** P0-017.

### P1-016 — Make promotion candidates evidence-aware

**Why.** Support count alone can promote repeated noise; contradictions need
explicit handling.
**Files/tests.** `layered_learning.py` (`_candidate_rows`, `synchronize`),
`tests/test_layered_learning.py`.
**Acceptance.** Candidate includes support, refute, source diversity, first/
last tick and contradiction flag; policy can require a minimum source count;
an unresolved contradiction is not promoted as a fact.
**Evidence.** `open`.
**Depends on.** P1-015.

### P1-017 — Inject permanent context into phone-call provider calls

**Why.** `phone_call.py` can send a provider request without the permanent
session block, violating the stated “every attached provider call” contract.
**Files/tests.** `phone_call.py`, `session_context.py`,
`tests/test_session_context.py`.
**Acceptance.** Phone-call provider turns include the compact block; a test
captures the request and asserts the block; failure to load the block is
reported, not silently ignored.
**Evidence.** `open`.
**Depends on.** P0-023.

### P1-018 — Synchronize session-document count, IDs and digests

**Why.** Code ships 7 documents; `CHANGELOG`, `ARCHITECTURE_GAPS` and some
docs say 6. Tool enum and docs must agree.
**Files/tests.** `session_context.py`, `session_docs/`,
`tests/test_session_context.py`, `CHANGELOG.md`,
`docs/ARCHITECTURE_GAPS.md`, `docs/TOOLS_REFERENCE.md`.
**Acceptance.** One source of truth for IDs; docs state 7; the tool enum
includes `mth-data-leaves`; digest test covers every file.
**Evidence.** `partial` — code/test are correct; docs stale.
**Depends on.** P0-029.

### P1-019 — Give every tool a full schema section in the canonical reference

**Why.** `TOOLS_REFERENCE.md` now has 48 detailed sections plus a 24-row table;
the README says all schemas and approval/scope requirements are listed there.
**Files/tests.** `docs/TOOLS_REFERENCE.md`, `plugin.yaml`,
`tests/test_manifest_registration.py`.
**Acceptance.** Every one of the 72 tools has parameters, defaults, return
shape, scope, approval and failure codes; a test extracts names from
`plugin.yaml` and asserts one section per tool.
**Evidence.** `partial` — names covered, schemas not.
**Depends on.** P0-029.

### P1-020 — Automate version and count consistency

**Why.** Version/count drift recurs (1.3.2 vs 1.5.0, 34 vs 72 tools, 272 vs
344 vs 376 tests).
**Files/tests.** `pyproject.toml`, `plugin.yaml`, `README.md`,
`CHANGELOG.md`, `docs/HERMES_CATALOG_ADMISSION.md`,
`tests/test_manifest_registration.py`.
**Acceptance.** CI reads the package/manifest version, tool list, kernel count
and session-doc count and fails on mismatch in release docs.
**Evidence.** `partial` — manifest/version parity test exists; doc counts are
not checked.
**Depends on.** P1-018, P1-019.

### P1-021 — Record a current 72-tool demo artifact

**Why.** The only recorded demo result is the historical 34-tool/38-call run.
`scripts/demo_run.py` has calls for 72 tools but no current result artifact.
**Files/tests.** `scripts/demo_run.py`, `docs/DEMO_RESULTS.md`,
`docs/HERMES_DEMO.md`, `docs/LIVE_SMOKE.md`.
**Acceptance.** Run the demo on the release commit with a real native binary,
loopback NANNET and local CLOCKED stubs; record tool coverage, failures,
durations and artifact digests. Do not label stubs production.
**Evidence.** `partial` — script exists; artifact is historical.
**Depends on.** P0-027, P0-032.

### P1-022 — Make Alpine/Termux installation a tested path

**Why.** The plugin is Android/Termux-first but CI is Ubuntu; the Alpine
record is a manual 344-test run and needs `build-base` + `gawk`.
**Files/tests.** `docs/ALPINE_PROOT_TESTING.md`,
`docs/HERMES_PLUGIN_INSTALL.md`, `nan0s_runtime.py`,
`.github/workflows/*`.
**Acceptance.** Install from wheel/sdist in Termux and Alpine/proot; compile
the portable core; run native fixture and N16I tests; document skip conditions
and exact package names.
**Evidence.** `partial` — manual instructions and recorded result.
**Depends on.** P0-028.

### P1-023 — Add cross-architecture portable-core CI

**Why.** Only aarch64 is exercised. `nan0s_runtime` maps x86_64, x86,
riscv64, ppc64le and s390x but has no evidence there.
**Files/tests.** `nan0s_runtime.py`, `.github/workflows/test.yml`,
`tests/test_mth_runtime.py`.
**Acceptance.** Build and run at least x86_64 and one non-x86 target; run a
native fixture and the N16I differential; publish per-target status.
**Evidence.** `open`.
**Depends on.** P0-028.

### P1-024 — Verify package data byte-for-byte

**Why.** A wheel can contain the manifest but miss kernel fixtures or session
docs; a line-count check is not enough.
**Files/tests.** `pyproject.toml`, `MANIFEST.in`, release CI.
**Acceptance.** Compare SHA-256 of every packaged source, fixture and session
doc against the repository at the release commit; fail on missing/extra.
**Evidence.** `open`.
**Depends on.** P0-026.

### P1-025 — Add a per-family 91-kernel fixture matrix

**Why.** A single aggregate “91/91” can hide one broken family.
**Files/tests.** `scripts/mth_function_tree_report.py`,
`tests/test_mth_runtime.py`, `nan0s_runtime.py`.
**Acceptance.** Report PASS/FAIL/steps/trap per kernel and per family; CI
artifacts retain the matrix; a family with a failure cannot report complete.
**Evidence.** `partial` — tree map exists; per-kernel fixture matrix not
recorded in this audit.
**Depends on.** P0-030.

### P1-026 — Expand vector kernel parity to edge operands

**Why.** Current tests use simple values. Numeric ABI claims need zero,
opposite, large finite, subnormal, NaN and Inf cases.
**Files/tests.** `tests/test_n16i_unification.py`,
`tests/test_mth_runtime.py`, `nan0s_runtime.py`.
**Acceptance.** For dot/cosine/add/sub, native and N16I agree within a stated
tolerance; non-finite input is rejected or propagated per contract; dimension
mismatch denies.
**Evidence.** `partial` — basic parity exists.
**Depends on.** P0-030.

### P1-027 — Build a NANNET loopback test matrix

**Why.** One happy-path channel is not enough for pairing, expiry, revocation,
wrong scope, wrong issuer, replay and device routes.
**Files/tests.** `tests/test_auth_tools.py`, `tests/test_pairing.py`,
`tests/test_http_api.py`, `tests/test_pipeline_integration.py`,
`tests/test_nannet_hw2_connect.py`.
**Acceptance.** Matrix covers pair/authorize/revoke/rotate/release,
wrong scope, expired code/token, revoked identity, replayed nonce, wrong
issuer, TLS rejection and missing secret. Each row has a stable result code.
**Evidence.** `partial` — several tests exist; matrix not unified.
**Depends on.** P0-007.

### P1-028 — Schedule production probes and record their evidence

**Why.** Live smoke is a point-in-time doc; production drift needs periodic
checks.
**Files/tests.** `scripts/live_smoke.py`, `scripts/nannet_hw2_connect.py`,
deployment CI.
**Acceptance.** A scheduled probe checks issuer fingerprint, health, scope,
revoke and router lease; records deployment ID; alerts on failure; never
prints secrets.
**Evidence.** `open`.
**Depends on.** P0-032.

### P1-029 — Add direct tests for every data-leaf denial code

**Why.** The documented denial table is broader than the tests. A denial that
creates a leaf or executes a kernel is a safety failure.
**Files/tests.** `mth_leaves.py`, `tests/test_mth_leaves.py`.
**Acceptance.** Tests assert no new leaf, no native execution and an audit
record for `unknown_leaf`, `hub_pairing_denied`, `illegal_pairing`,
`unknown_operator`, `operator_not_admitted`, `native_execution_failed`,
`invariant_violation` and `identity_violation`.
**Evidence.** `partial` — three codes covered.
**Depends on.** P0-031.

### P1-030 — Verify the on-disk leaf image hash during audit

**Why.** The leaf record stores a NaN image path and digest; audit should
re-read the file and compare, not only count aliases.
**Files/tests.** `mth_leaves.py` (`function_tree_audit`), `tests/test_mth_leaves.py`.
**Acceptance.** Audit fails on a missing, truncated or modified image; a
matching image reports its word count and digest.
**Evidence.** `partial` — audit verifies executable images but not byte hash
in all cases.
**Depends on.** P0-031.

### P1-031 — Test WordNet `depth`, address forms and missing archive

**Why.** `depth` is a no-op and object grammar is plugin-local; tests should
lock the intended behavior.
**Files/tests.** `layered_learning.py` (`ingest_word`),
`dictionary_universe.py`, `tests/test_wordnet_tool.py`,
`tests/test_layered_learning.py`.
**Acceptance.** `depth=1` and `depth=8` differ where the graph allows; the
canonical v2 address is emitted; legacy is accepted only as an explicit
compatibility input; missing archive fails closed.
**Evidence.** `partial` — route and fail-closed tests; depth not.
**Depends on.** P1-013.

### P1-032 — Add a dedicated concept-matrix test and load path

**Why.** `ConceptMatrix` is central to layered learning but has no dedicated
test file; `save()` has no `load()` counterpart.
**Files/tests.** `concept_matrix.py`, `layered_learning.py`,
`tests/test_layered_learning.py`, new `tests/test_concept_matrix.py`.
**Acceptance.** Tests cover pair accumulation, typed roles, leaf signature,
saturation, persistence round-trip and malformed state rejection.
**Evidence.** `open` — no dedicated test.
**Depends on.** P1-015.

### P1-033 — Add free-chat end-to-end tests with fake and live runners

**Why.** `native_update_failed` is correctly reported when no runner exists;
the successful path needs a real/fake runner receipt.
**Files/tests.** `chat.py`, `tools.py`, `tests/test_chat.py`,
`tests/test_free_chat_learning.py`.
**Acceptance.** Fake runner verifies changed heads/mirror pairs/hyperedges;
live runner test records state digests; missing runner never fabricates success.
**Evidence.** `partial` — fake-runner test exists; live runner path not re-run.
**Depends on.** P1-034.

### P1-034 — Prove native relation training end-to-end

**Why.** The native relation path is the only real learning path, but the
Gemma smoke had no runner and reported `native_update_failed`.
**Files/tests.** `tools.py` (`_native_relation_train`), `runner.py`,
`tests/test_chat.py`, `docs/GEMMA_FREE_CHAT_SMOKE.json`.
**Acceptance.** Real `nanlm-universal` run trains, evaluates frozen test set,
changes heads/mirror pairs/hyperedges and reports a new state digest; a second
run resumes the same topology.
**Evidence.** `partial` — fake-runner evidence; no live end-to-end artifact in
this audit.
**Depends on.** P0-030.

### P1-035 — Keep CLOCKED on a separate credential and test the live path

**Why.** CLOCKED uses a participant token, not the upstream API key; the live
smoke only covers `pii` and uses fixtures for other endpoints.
**Files/tests.** `clocked_client.py`, `clocked_tools.py`,
`tests/test_head_cortex_evaluation.py`,
`docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md`.
**Acceptance.** `clu_...` token is required; upstream key is never read;
plan/execute/scratch/head/synthesize live path passes; results are evidence,
not ground truth; no canonical promotion.
**Evidence.** `partial` — live `pii` recorded; other endpoints fixtures.
**Depends on.** P0-024, P0-032.

### P1-036 — Freeze the head-cortex candidate contract

**Why.** Candidate create/learn/evaluate exist, but promotion is absent and
M0–M9 is not fully demonstrated.
**Files/tests.** `head_cortex.py`, `clocked_tools.py`,
`tests/test_head_cortex_evaluation.py`,
`docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md`.
**Acceptance.** Request/response schemas, base digest, split, `promotion_allowed=false`
and `canonical_state_modified=false` are enforced; a six-runner barrier would
be required for any future promote; tests cover stale base and mismatched
digest.
**Evidence.** `partial` — candidate endpoints tested; promotion absent.
**Depends on.** P0-019, P1-009.

### P1-037 — Define a stable error taxonomy

**Why.** Tools raise `ValueError`, `RuntimeError`, `PermissionError` and
module-specific errors. Operators need stable codes, not string matching.
**Files/tests.** all handlers, `docs/TOOLS_REFERENCE.md`, tests.
**Acceptance.** Every public error has `code`, `message`, `retryable` and
`data`; secrets redacted; tests assert codes for auth, scope, lease, MTH,
WordNet, CLOCKED and state errors.
**Evidence.** `open`.
**Depends on.** P0-024.

### P1-038 — Harden state-directory permissions and symlink handling

**Why.** State includes keys, tokens, transcripts and databases. Symlinked
paths or mode `0644` can leak them.
**Files/tests.** `auth.py`, `auth_tools.py`, `nannet_device_client.py`,
`router_universe_client.py`, `layered_learning.py`, `scratch_register.py`,
tests.
**Acceptance.** Directories `0700`, files `0600`, no symlink traversal for
identity/token/scratch paths; a symlink attack test fails closed.
**Evidence.** `partial` — many chmod calls; no symlink test.
**Depends on.** P0-008.

### P1-039 — Test both crypto backends and the no-crypto fallback

**Why.** `cryptography` is optional; Termux/Alpine must import and chat without
it, while auth fails closed.
**Files/tests.** `crypto_compat.py`, `tests/test_optional_crypto.py`,
`tests/test_crypto_compat.py`.
**Acceptance.** Without the extra, chat/training/memory import; auth tools
raise an actionable unavailable error; with the extra, all crypto tests pass.
**Evidence.** `connected` locally — tests exist; Alpine/Termux matrix not
rerun.
**Depends on.** P0-028.

### P1-040 — Audit locks, cleanup and resource bounds

**Why.** Scratch page locks, coordinator cache, leaf cache, SQLite WAL and
HTTP servers need deterministic cleanup; leaked locks cause hangs.
**Files/tests.** `scratch_register.py`, `layered_learning.py`,
`mth_leaves.py`, `memory.py`, `http_api.py`, tests.
**Acceptance.** Locks released on exception; caches reset; servers close;
SQLite connections context-managed; file descriptors bounded; a stress test
repeats open/close 100 times.
**Evidence.** `partial` — caches have reset helpers; no stress test.
**Depends on.** P0-022.

### P1-041 — Decide “wire or retire” for every offline quantum/gauge module

**Why.** `gauge_memory`, `cycle_space`, `cycle_state`, `cycle_codec`,
`phase_memory`, `spectral`, `word_writer`, `pair_quantum`,
`robust_recovery`, `semantic_state`, `semantic_acquisition`, `temporal_spine`,
`boolean_wave`, `head_population`, `definition_decoder`, `lexical_memory` and
`corpus_memory` are documented as architecture pieces, but most have no
runtime caller. Leaving that ambiguity invites false integration claims and
maintenance drift.

**Files/tests.** Each module listed above, its doc, its test file, and
`docs/QUANTUM_MATH_INTEGRATION_MATRIX.md`.

**Acceptance.** A checked-in registry classifies every module as:
`runtime` (has a named caller, test and receipt), `benchmark` (offline, not a
product claim), or `retired` (no code/docs claim). A module marked `runtime`
must be called by a plugin handler, chat path or native runner with a test that
fails if the call disappears. A module marked `benchmark` is removed from
“implemented” language in README/architecture docs. No row remains
“documented-not-wired” without an explicit benchmark/retired label.

**Evidence.** `partial` — individual docs are honest; there is no registry or
caller test. See matrix rows QM-01..QM-16, QM-24.

**Depends on.** P0-021, P1-019.

---

# P2 — documentation, reproducibility and cleanup

### P2-001 — README test count

**Why.** `README.md:214` says 344; static snapshot has 376 methods.
**Files/tests.** `README.md`.
**Acceptance.** Count generated from the release commit; historical counts
marked historical.
**Evidence.** `open`.
**Depends on.** P0-029.

### P2-002 — CHANGELOG session-document count

**Why.** `CHANGELOG.md:5` says six; code ships seven.
**Files/tests.** `CHANGELOG.md`.
**Acceptance.** Seven documents named; IDs match `session_context.DOCUMENT_IDS`.
**Evidence.** `open`.
**Depends on.** P1-018.

### P2-003 — ARCHITECTURE_GAPS session-document count

**Why.** `docs/ARCHITECTURE_GAPS.md:11` says six; code ships seven.
**Files/tests.** `docs/ARCHITECTURE_GAPS.md`.
**Acceptance.** Corrected and linked to the session context test.
**Evidence.** `open`.
**Depends on.** P1-018.

### P2-004 — README temporal-spine link

**Why.** `README.md:279` links `docs/SPINE.md`, which does not exist.
**Files/tests.** `README.md`, `docs/SPECTRAL_WORD.md`.
**Acceptance.** Either add `docs/SPINE.md` or point to a real document and
mark the module as prototype.
**Evidence.** `open`.
**Depends on.** P2-005.

### P2-005 — Document the temporal spine, or remove the claim

**Why.** `nanlmind_hermes/temporal_spine.py` exists and has tests, but no doc;
the README links a missing SPINE.md.
**Files/tests.** `temporal_spine.py`, `tests/test_temporal_spine.py`,
`docs/SPECTRAL_WORD.md`.
**Acceptance.** A doc states true rank mirror `32639-r`, B/F update, mirror
residual and the fact that it is not wired into `temporal_lm`.
**Evidence.** `partial` — code/tests, no doc.
**Depends on.** P2-004.

### P2-006 — TOOLS_REFERENCE layer bounds

**Why.** Lines for memory tools say layer `0–31` and “L0–L6”; schemas and
tests enforce `0..9`.
**Files/tests.** `docs/TOOLS_REFERENCE.md`,
`tests/test_plugin_manifest.py`.
**Acceptance.** Every layer reference says `0..9`/L0–L9.
**Evidence.** `open`.
**Depends on.** P1-019.

### P2-007 — Catalog admission metadata

**Why.** `docs/HERMES_CATALOG_ADMISSION.md` still records version 1.3.2, 34
tools and 272 tests in places while the current manifest is 1.5.0/72.
**Files/tests.** `docs/HERMES_CATALOG_ADMISSION.md`,
`plugin.yaml`, `pyproject.toml`.
**Acceptance.** Version, tool list, test count and SHA placeholders are current
or explicitly historical.
**Evidence.** `partial` — some sections updated; capability block stale.
**Depends on.** P1-020.

### P2-008 — DEMO/install tool counts

**Why.** `docs/DEMO.md` and `docs/HERMES_PLUGIN_INSTALL.md` still mention
33/34/59 tools in historical contexts.
**Files/tests.** `docs/DEMO.md`, `docs/HERMES_PLUGIN_INSTALL.md`.
**Acceptance.** Historical counts labelled; current count is 72.
**Evidence.** `partial`.
**Depends on.** P1-020.

### P2-009 — Device report app version

**Why.** `nannet_device_client._report()` hard-codes `nanlmind-hermes/1.3.2`.
**Files/tests.** `nanlmind_hermes/nannet_device_client.py`.
**Acceptance.** Version read from package metadata or a single constant.
**Evidence.** `open`.
**Depends on.** P1-020.

### P2-010 — Remove or ignore stale build artifacts

**Why.** `nanlmind_hermes.egg-info/SOURCES.txt` and `build/lib/` predate the
MTH/NaN0S/session-doc bundles and can mislead a release audit.
**Files/tests.** `nanlmind_hermes.egg-info/`, `build/`, `.gitignore`.
**Acceptance.** Artifacts regenerated or ignored; release builds from a clean
checkout.
**Evidence.** `partial`.
**Depends on.** P0-026.

### P2-011 — Fix broken local links and anchors

**Why.** Documentation is the operator interface; broken links cause wrong
setup.
**Files/tests.** `README.md`, `docs/*.md`.
**Acceptance.** A link checker passes for local Markdown links and anchors.
**Evidence.** `open`.
**Depends on.** P0-029.

### P2-012 — Reconcile all historical test counts

**Why.** 272/312/344/369/376 appear across docs. Without labels, a reader
cannot tell current from historical.
**Files/tests.** `docs/*.md`, `README.md`, `CHANGELOG.md`.
**Acceptance.** Every count has a date/commit or is generated.
**Evidence.** `open`.
**Depends on.** P0-029.

### P2-013 — MTH function-tree document counts

**Why.** `docs/MTH_FUNCTION_TREE.md` must agree with the manifest and audit.
**Files/tests.** `docs/MTH_FUNCTION_TREE.md`,
`scripts/mth_function_tree_report.py`, `mth/spec/wave-capabilities.json`.
**Acceptance.** 91 kernels, 44 family hubs, 0 missing/orphans; generated from
the manifest.
**Evidence.** `partial`.
**Depends on.** P0-030.

### P2-014 — Add snapshot headers to evidence docs

**Why.** Audits and demo results lack commit/date/command in some files.
**Files/tests.** `docs/GEMMA_*.md`, `docs/DEMO_RESULTS.md`,
`docs/LIVE_SMOKE.md`, `docs/LONG_CHAT_DEMO.md`.
**Acceptance.** Each evidence doc starts with commit, date, command and
artifact digest.
**Evidence.** `partial`.
**Depends on.** P0-032.

### P2-015 — Normalize CHANGELOG structure

**Why.** Released and unreleased entries are interleaved; “Unreleased” appears
after 1.5.0.
**Files/tests.** `CHANGELOG.md`.
**Acceptance.** Keep a Changelog format; released versions newest-first;
Unreleased at top.
**Evidence.** `open`.
**Depends on.** P1-020.

### P2-016 — License and provenance headers

**Why.** Apache-2.0 requires notices for bundled NaN0S/WordNet-derived data.
**Files/tests.** `LICENSE`, `nanlmind_hermes/nan0s/`, `mth/spec/`,
`docs/MTH_UNIVERSAL_RUNTIME.md`.
**Acceptance.** Bundled source provenance, license and modification notes are
present; no incompatible data is shipped.
**Evidence.** `partial`.
**Depends on.** P0-026.

### P2-017 — Mark legacy address forms consistently

**Why.** Some docs show `nan://HW4:BS0:MC0:SC0:GM0` as if canonical.
**Files/tests.** `docs/WORDNET_GRAPH.md`, `docs/WORDNET_TYPED.md`,
`docs/DEFINITION_ACQUISITION.md`, `docs/DEFINITION_QA.md`.
**Acceptance.** Canonical v2 first; legacy labelled compatibility only.
**Evidence.** `partial`.
**Depends on.** P1-013.

### P2-018 — Maintain a quantum-inspired claim ledger

**Why.** Gauge/cycle/phase/spectral/pair-space documents contain measured
classical results and explicit non-claims; a single ledger prevents drift into
quantum marketing.
**Files/tests.** `docs/GAUGE_MEMORY.md`, `docs/CYCLE_*.md`,
`docs/PHASE_MEMORY.md`, `docs/PAIR_QUANTUM.md`,
`docs/SPECTRAL_*.md`, this matrix.
**Acceptance.** Each claim has code, test/script, scale and “no quantum
advantage” status.
**Evidence.** `partial` — individual docs are honest; no ledger.
**Depends on.** P2-019.

### P2-019 — Add a test-coverage map

**Why.** 66 test files and 376 methods are not mapped to production
requirements.
**Files/tests.** `tests/`, `docs/PRODUCTION_READINESS_TODO.md`.
**Acceptance.** Each P0/P1 item links at least one test file or names a new
test to add.
**Evidence.** `open`.
**Depends on.** P0-029.

### P2-020 — Make benchmark scripts reproducible

**Why.** `scripts/*.py` often read a hard-coded Android WordNet path and write
docs; results depend on the archive.
**Files/tests.** `scripts/wordnet_*.py`, `scripts/joint_semantic_decoder.py`,
`scripts/phase_memory_bench.py`, `scripts/cycle_*.py`.
**Acceptance.** CLI argument for archive/output, seed, split and counts;
recorded command line; no overwrite of existing docs by default.
**Evidence.** `partial`.
**Depends on.** P1-013.

### P2-021 — Update the catalog capability block

**Why.** The upstream capability block lists 34 tools; current manifest has 72.
**Files/tests.** `docs/HERMES_CATALOG_ADMISSION.md`.
**Acceptance.** Capability block generated from `plugin.yaml`.
**Evidence.** `open`.
**Depends on.** P1-020.

### P2-022 — Document Termux prerequisites

**Why.** `cryptography`, C compiler and `gawk` are optional/required in
different paths; users hit opaque failures.
**Files/tests.** `docs/HERMES_PLUGIN_INSTALL.md`,
`docs/ALPINE_PROOT_TESTING.md`, `README.md`.
**Acceptance.** Termux packages, Python version, `pkg install` lines and
native build prerequisites are explicit.
**Evidence.** `partial`.
**Depends on.** P1-022.

### P2-023 — Document issuer-key rotation and emergency revocation

**Why.** A changed issuer intentionally breaks the pin; operators need a safe
procedure.
**Files/tests.** `docs/NANNET_API_HERMES.md`,
`docs/NANNET_HW2_PRODUCTION.md`, `SECURITY.md`.
**Acceptance.** Rotation, revocation, re-pair and rollback steps; no
trust-on-first-use.
**Evidence.** `partial`.
**Depends on.** P0-010.

### P2-024 — Publish a NANNET error-code catalog

**Why.** Operators see `token_invalid_scope_or_signature` but not the cause or
recovery.
**Files/tests.** `docs/NANNET_API_HERMES.md`, `http_api.py`,
`auth_tools.py`, `auth.py`.
**Acceptance.** Stable code, HTTP status, cause, recovery and whether the
operation is retryable.
**Evidence.** `open`.
**Depends on.** P1-037.

### P2-025 — Fix the WordNet archive environment name

**Why.** `wordnet_store.py` says `NANLM_WORDNET_ARCHIVE`; the setting is
`wordnet_archive`.
**Files/tests.** `wordnet_store.py`, `plugin.yaml`, `README.md`.
**Acceptance.** One documented name; deprecated alias, if any, logs a warning.
**Evidence.** `open`.
**Depends on.** P1-014.

### P2-026 — Fix the WordNet vector reference grammar

**Why.** `mth_leaves.py` uses `...::word/vector` but the documented grammar is
`::word/<lemma>/<offset>`.
**Files/tests.** `mth_leaves.py`, `docs/MTH_DATA_LEAVES.md`,
`dictionary_universe.py`.
**Acceptance.** Reference includes lemma and offset; a link with no resolved
offset is marked unresolved.
**Evidence.** `open`.
**Depends on.** P1-013.

### P2-027 — Resolve the `bidirectional` claim

**Why.** `DictionaryUniverse.interface()` returns `bidirectional: True` but
there is no write path to HW4.
**Files/tests.** `dictionary_universe.py`, `docs/WORDNET_GRAPH.md`,
`docs/WORDNET_TYPED.md`.
**Acceptance.** Remove the flag or implement a generation-checked write path.
**Evidence.** `open`.
**Depends on.** P1-013.

### P2-028 — Correct the “through NANNET” docstring

**Why.** `dictionary_universe.py` says Hermes reaches HW4 through NaNNet; the
current code reads a local zip.
**Files/tests.** `dictionary_universe.py`, `docs/WORDNET_GRAPH.md`.
**Acceptance.** Docstring distinguishes local archive read from a live route.
**Evidence.** `open`.
**Depends on.** P1-013.

### P2-029 — Correct the promotion path in the free-chat contract

**Why.** `session_docs/FREE_CHAT_LEARNING.md` says only
`nanlm_learning_sync` promotes; the tools also expose `approved` on cycle and
WordNet ingest.
**Files/tests.** `nanlmind_hermes/session_docs/FREE_CHAT_LEARNING.md`,
`nanlmind_hermes/__init__.py`, `docs/NANVERSE_LAYERED_LEARNING.md`.
**Acceptance.** State the exact three paths and the `merge_enabled` gate.
**Evidence.** `partial` — current code gates; docs still say only one path.
**Depends on.** P0-015.

### P2-030 — Use “truth value” rather than “ground truth” for head outputs

**Why.** Boolean-head outputs are predictions over evidence, not independent
facts; wording affects operator trust.
**Files/tests.** `docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md`,
`docs/JOINT_SEMANTIC_DECODER.md`, `docs/MTH_LAYERS.md`.
**Acceptance.** Terminology table: observation, evidence, prediction, truth
state, verification status.
**Evidence.** `partial`.
**Depends on.** P1-036.

### P2-031 — Explain 91 kernels versus 94 manifest entries

**Why.** `docs/HERMES_MTH_UNIVERSAL_TOOLS.md` says 94 entries, 91
CI-verified; the plugin manifest has 91. A reader can think the bundle is
incomplete.
**Files/tests.** `docs/MTH_UNIVERSAL_RUNTIME.md`,
`docs/MTH_FUNCTION_TREE.md`, `mth/spec/wave-capabilities.json`.
**Acceptance.** State that the bundle contains the 91 planning-eligible
kernels and that three local-only entries are excluded by policy.
**Evidence.** `partial`.
**Depends on.** P0-030.

### P2-032 — Align v1/v2 address grammar across repositories

**Why.** `session_docs/NANNET_ROUTING.md` shows a v1 grammar while
`NaNROUTER/spec/address-grammar.md` is v2 apex-layered; both are cited as
canonical.
**Files/tests.** `session_docs/NANNET_ROUTING.md`,
`docs/NANNET_API_HERMES.md`, `NaNROUTER/spec/address-grammar.md`.
**Acceptance.** One canonical v2 form; v1 marked legacy; plugin emits v2.
**Evidence.** `partial`.
**Depends on.** P2-017.

---

# Release evidence commands

Run on the exact release commit in a clean checkout. Save stdout/stderr and
SHA-256 of artifacts; do not paste tokens or pairing codes into the record.

```sh
# Tool / manifest / test discovery
python - <<'PY'
import re, pathlib
init = pathlib.Path("nanlmind_hermes/__init__.py").read_text()
manifest = pathlib.Path("plugin.yaml").read_text()
print("manifest_tools", len(re.findall(r"^\s+- nanlm_", manifest, re.M)))
print("registered_tools", len(re.findall(r'^\s+\("nanlm_', init, re.M)))
PY
grep -rho 'def test_[A-Za-z0-9_]*' tests/*.py | wc -l

# Full suite (after the package data is tracked)
python -m unittest discover -s tests -v

# Packaging
python -m build
python -m twine check --strict dist/*
python - <<'PY'
import glob, tarfile, zipfile
for path in glob.glob("dist/*.whl") + glob.glob("dist/*.tar.gz"):
    print(path)
PY

# MTH catalog/tree
python scripts/mth_function_tree_report.py --help

# Loopback NANNET probe (test/only; no production secret)
python -m unittest tests.test_pipeline_integration tests.test_nannet_hw2_connect -v
```

A release gate passes only when the P0 items are closed, every P0 evidence
item points to a command output and artifact digest, and the status still
reports `canonical_merge_enabled=false` and `six_runner_barrier=absent`.

## P1-042 — Resident typed universe and capability adapters

**Why.** The 8 MiB chart/model/ensemble are process-resident, but relations,
hyperedges, MTH stacks and scratch payloads are queried or rebuilt per call.
That costs latency and keeps the computational primitives out of sync with the
learning state.

**Files.** `docs/RESIDENT_UNIVERSE.md`,
`nanlmind_hermes/{universe_chart,temporal_lm,layered_learning,tools,mth_leaves}.py`,
`nanlmind_hermes/memory.py`.

**Acceptance.** One generation-bound `ResidentUniverse` per state dir with a
compact relation/pair index, bounded memory, paged scratch, append-only writes,
best-checkpoint recovery, and no 2 GiB overbootstrap copy. NaN/N16 and normal-code
adapters use capability-scoped typed frames/JSON, never raw pointers or shell.
Tests cover reload, generation invalidation, memory caps, adapter denial and
parity with current query results.

**Evidence.** `open` — design only.

**Depends on.** P0 merge gate remains closed; P1 memory caps; P1 packaging.

# Count summary

- **P0:** 33 items (`P0-001` … `P0-033`)
- **P1:** 42 items (`P1-001` … `P1-042`)
- **P2:** 32 items (`P2-001` … `P2-032`)
- **Total:** 107 checklist items

See `docs/QUANTUM_MATH_INTEGRATION_MATRIX.md` for the subsystem-by-subsystem
mapping and the linked production IDs.
