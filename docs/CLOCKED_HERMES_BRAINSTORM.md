# CLOCKED × HERMES integration brainstorm

**Status:** architecture proposal / pseudocode only  
**Target:** `shemshallah/NaNLMind_HERMES`  
**Primary integration:** CLOCKED API as a first-class HERMES plugin capability  
**Related systems:** NaNLMind, NaNLMind_android, NaNROUTER, NaN0S/NaN16, NANNET, Koyeb, MC0:0 Runner Ring

> This document is a design proposal, not a claim that a CLOCKED HTTP API, public Koyeb deployment, or remote training endpoint is already live. Endpoint names and payloads must be reconciled with the actual CLOCKED service contract before implementation.

## 1. Executive recommendation

Make CLOCKED a first-class capability inside the existing `nanlmind-hermes` plugin, with two user-facing modes and one shared internal integration layer:

- **Chat mode** remains the default HERMES conversation. HERMES can use CLOCKED capabilities behind the scenes when useful and authorized, including corpus planning, prompt construction, temporal context, and task orchestration.
- **CLOCKED mode** is an explicit workspace for direct CLOCKED requests, exploration of CLOCKED operations, and iterative prompt construction.
- **CLOCKED demo universe** is a separately addressable, deliberately bounded demo environment. It must be isolated by identity, namespace, ACL, storage prefix, and resource policy while remaining routable through the canonical NaNLMind/NaNnet substrate.

Do not build a second CLOCKED client stack inside each tool. Build one CLOCKED adapter/service module, one typed request/result envelope, one scratch protocol, and a small set of HERMES tools that call those shared primitives.

The strongest interaction is the **shared scratch round-trip**:

1. HERMES creates or opens a scratch workspace.
2. HERMES writes a versioned prompt draft into the HERMES-authored lane.
3. The user and HERMES can iteratively refine that draft.
4. HERMES submits a snapshot/revision to CLOCKED and/or NaNLMind.
5. NaNLMind receives the exact scratch revision as an authorized task, processes it, and appends its result to a distinct NaNLMind-authored lane in the same scratch.
6. HERMES reads the committed result, validates provenance and completion status, and presents it.
7. The user can continue building on the result without copying text between windows.

The scratch is a protocol object and shared working surface—not an unversioned temporary text file and not a new source of truth for the global model.

## 2. Repository evidence and current constraints

The following are grounded in the current default-branch files inspected for this brainstorm.

### 2.1 HERMES already has the right extension points

- `README.md` establishes `/nanlm` as default chat, explicit logic/words/sentences/pipeline commands, durable transcripts, corpus manifests, native training receipts, and a local native runner.
- HERMES already has corpus generation and multi-resolution relation-curriculum generation in `nanlmind_hermes/tools.py`. The code uses the active HERMES model provider to generate records, validates N16TXT/N16REL data, freezes held-out challenges, records digests, and invokes the local runner.
- `nanlmind_hermes/auth.py` and `pairing.py` provide installation identity, signed requests, scoped tokens, pairing, and revocation.
- `docs/PLUGIN_ARCHITECTURE.md` explicitly says chat learning, structured teacher training, and math training share the same native GlobalTopology; they must not create parallel semantic learners.
- `docs/NANNET_API_HERMES.md` defines the current NANNET identity/signature model and explicitly warns that public hosted deployment and authenticated training/snapshot endpoints are not yet a proven live service contract.
- HERMES documents a reserved **HW3 Clocked demo identity**. HERMES must not claim or register that slot. HW2 is the Hermes demo identity; later independent Hermes installations use subsequent available slots. Keep hardware identities distinct from Router point numbers and BSR wire identities.

### 2.2 NaNLMind temporal clock is logical time

`NaNLMind/docs/14_TEMPORAL_CLOCK.md` defines a deterministic `LogicalStamp{tick, phase}`, not wall-clock time. The oscillator is a field of the one global topology. Its clock digest participates in the global digest. The clock supports deterministic event ordering, replay/catch-up, lease ticks, and a two-phase proposal/commit cadence.

Important boundary: use the logical clock for causal ordering, temporal memory, replay, and deterministic leases. Use host time for TLS/network deadlines, HTTP timeouts, and real elapsed-time display. Do not describe logical ticks as seconds or milliseconds.

The document also records that the global-scratch clock synchronization result is a simulation result, not proof of production runner behavior. Treat it as a design input and validate it against the live ring.

### 2.3 NaNROUTER owns routes and provider bindings

`NaNROUTER/README.md` describes the Router as the routing/control plane around one canonical NaNLMind topology. It distinguishes permanent point/head identities, BSR wire identities, logical objects, and physical service bindings.

- Router point 1 is the permanently attached phone.
- Router point 2 is the Hermes demo.
- Router point 3 is the Clocked demo.
- Koyeb BS0:0 is the persistent gateway; MC0:0 is the six-unit runner-controlled macrocluster.
- The six runner wire identities remain BSR0–BSR5.
- Candidate route learning does not mutate the committed master route cache.
- Route packets are digest-bound and signed; route commits require explicit capability and generation checks.

The Router's volume registry owns logical-to-physical Koyeb volume bindings and health. NaNLMind supplies the NaN address/word codec. Do not let HERMES or CLOCKED invent mount paths, provider IDs, or green health claims.

### 2.4 NaNNet and Android boundaries

`NaNLMind/README.md` describes the N16B bridge, canonical `nan://` route form, point-zero translation, durable leased delivery, inbox-before-ack behavior, and Koyeb as coordination/authorization/transport—not the local model runner or corpus store.

`NaNLMind_android/README.md` describes the N16T gateway and scoped device token. Its current documented HELLO/CAPABILITIES exchange proves protocol reachability only; it does not yet establish corpus upload, remote training, candidate evaluation, or model merge. Do not silently treat the Android client as a completed CLOCKED execution backend.

`NaN0S` establishes the priority rule: NaN16 guest code owns protocol state and training behavior; native C/C++ is a narrow host/interpreter boundary. New guest-side protocol, codec, state-machine, and validation logic should be specified in NaN16 first. A host adapter may bridge sockets, secure storage, or OS facilities where unavoidable, but must not become a second semantic implementation.

### 2.5 CLOCKED API discovery gap

The currently accessible `shemshallah` repository inventory exposed NaNLMind, NaNLMind_HERMES, NaNLMind_android, NaNROUTER, NaN0S, and the runner/storage repositories, but did not expose a distinct repository named CLOCKED or a published CLOCKED endpoint schema in the inspected files. HERMES reserves HW3 and Router point 3 for the Clocked demo, but those reservations alone do not define the API.

Therefore, this design deliberately separates:
- **confirmed contracts** described above;
- **proposed HERMES adapter contracts** below;
- **CLOCKED-specific facts still to verify** against its authoritative source: base URL, authentication audience/issuer, routes, request/response schemas, quotas, async job semantics, idempotency, and demo-universe identifier.

Before implementation, locate the authoritative CLOCKED repository/service specification and pin its version. Never infer API paths from product naming.

## 3. Target architecture

```text
                         USER
                          |
                   HERMES PLUGIN UI
                          |
             +------------+-------------+
             |                          |
          CHAT MODE                 CLOCKED MODE
             |                          |
             +------------+-------------+
                          |
                HERMES ORCHESTRATOR
                          |
       +------------------+--------------------+
       |                  |                    |
  Prompt Architect   Corpus Architect     Scratch Manager
       |                  |                    |
       +------------------+--------------------+
                          |
                 CLOCKED ADAPTER
             capability discovery / schema
             request validation / idempotency
             result normalization / provenance
                          |
              NANNET AUTH + ROUTER RESOLUTION
                          |
             Koyeb BS0:0 persistent gateway
                          |
                  NaNNet / N16B bridge
                          |
             MC0:0 six-runner control ring
                          |
             CLOCKED DEMO (Router point 3)
                          |
               NaNLMind task intake/runner
                          |
              shared scratch result append
                          |
                    HERMES reads
```

The diagram is a logical flow, not a claim that every component is already deployed or that every request must traverse every runner. Router resolution chooses the authorized path and execution placement. Local HERMES operations can remain local; remote CLOCKED operations use the authenticated gateway/bridge path.

### Ownership boundaries

| Concern | Owner | HERMES responsibility |
|---|---|---|
| User interaction, prompt intent, orchestration | HERMES | Own |
| CLOCKED operation semantics and API schema | CLOCKED service | Discover, validate, invoke |
| Demo identity and demo resource boundary | CLOCKED authority / Router registration | Request scoped delegation; never impersonate |
| Global topology, NaN16 codec, logical clock | NaNLMind | Consume canonical protocol |
| Logical routes, ACLs, volume bindings, health | NaNROUTER | Resolve and verify; never mutate directly |
| Delivery, leases, ACK/replay | NaNNet bridge | Use canonical frames and receipts |
| Guest protocol state and semantic logic | NaN16 | Prefer NaN16 implementation |
| Local model execution | Authorized local/runner runtime | Invoke only through declared task contract |

## 4. Two modes, one plugin

### 4.1 Chat mode (default)

Chat remains conversational and does not force users to learn CLOCKED terminology.

HERMES should opportunistically use CLOCKED when:
- the user asks for time-aware, sequence-aware, temporal, or clock-derived work;
- corpus creation benefits from a temporal schedule, phase, epoch, or event ordering;
- prompt design needs a reproducible execution context;
- a prior CLOCKED result is referenced and the user has access to it;
- the user explicitly asks HERMES to use CLOCKED.

Chat should not invoke CLOCKED for every ordinary turn. It should explain when it materially used a CLOCKED capability, record a concise receipt, and never silently spend a metered quota or initiate a privileged mutation.

Example:
```text
/nanlm Create a corpus about causal reasoning and temporal memory.
```
HERMES may propose a corpus plan, ask the active model provider to generate candidate records, use CLOCKED to add permitted temporal structure or run a temporal operation, validate the records, then save an N16TXT/N16REL corpus and manifest. The exact CLOCKED operation depends on discovered API capabilities.

### 4.2 CLOCKED mode

Add a mode switch/tool surface such as:
```text
/nanlm clocked
/nanlm clocked status
/nanlm clocked capabilities
/nanlm clocked ask <request>
/nanlm clocked scratch new
/nanlm clocked scratch open <id>
/nanlm clocked scratch show
/nanlm clocked scratch submit
/nanlm clocked scratch continue <instruction>
/nanlm clocked exit
```

These are proposed UX names, not existing commands. Keep the existing `/nanlm` command stable and avoid command collisions with Hermes core.

CLOCKED mode should expose:
- capability discovery and current API/version status;
- direct, typed CLOCKED requests;
- request history and receipts;
- prompt construction in scratch;
- send-to-CLOCKED and send-to-NaNLMind actions;
- same-scratch response retrieval;
- continuation, branching, compare, and export;
- clear indication of demo-universe scope, quotas, and whether an operation is read-only or mutating.

### 4.3 Demo entitlement and automatic access

Goal: anyone who successfully installs and authenticates the HERMES plugin receives automatic access to the CLOCKED demo integration, without manually provisioning a second long-lived secret.

Recommended flow:
1. HERMES performs its existing mandatory NANNET login/pairing.
2. The authority recognizes the HERMES plugin identity and installation.
3. HERMES requests a short-lived, least-privilege **delegated demo capability** for the registered Clocked demo service/universe.
4. The authority returns a scoped token or signed capability bound to the HERMES installation, audience, demo universe, allowed operations, quota, and expiry.
5. HERMES stores it only in encrypted local state, refreshes through the authority, and never exposes it to the model provider or user-visible prompt.
6. The CLOCKED service validates both the delegated capability and the installation/request signature.
7. Revocation, expiry, quota exhaustion, or demo shutdown disables the capability cleanly; normal HERMES chat remains available.

Automatic access means automatic *delegation*, not shared secrets, anonymous public access, or HERMES claiming HW3.

**Identity separation is critical:**
- HW2 remains the HERMES demo identity.
- HW3 remains the Clocked demo identity and is owned by the Clocked demo service.
- Router point 2 and point 3 are routing points, not hardware slot numbers.
- BSR0–BSR5 remain wire identities, not user identities or universe IDs.
- Each HERMES installation retains its own installation keypair and signed requests.
- A user may access the demo through HERMES without receiving the demo service's private key.

Suggested scope split (names are proposals to be reconciled with the authority's canonical vocabulary):
- `clocked:capabilities:read`
- `clocked:demo:read`
- `clocked:request:execute`
- `clocked:scratch:read`
- `clocked:scratch:write`
- `clocked:prompt:submit`
- `clocked:corpus:propose`
- `clocked:corpus:export`

Do not grant route-commit, volume-admin, universe-registration, global snapshot-write, or global merge scopes to ordinary demo users. Demo corpus proposals should be sandboxed; global training/merge requires a separate explicit capability and policy gate.

The current HERMES auth documentation has a guest scope ceiling centered on corpus write, training execute, universe read, and snapshot read. Do not simply add broad CLOCKED privileges to that guest role. Add a distinct demo delegation policy with audience restriction, resource restriction, quota, expiry, and revocation. Keep the existing Ed25519 request signature, exact-path signing, nonce replay protection, issuer pinning, and token audience checks.

## 5. The shared scratch: core user experience

### 5.1 Scratch is a versioned protocol object

A scratch should have:
- globally unique opaque `scratch_id`;
- owner installation/user reference (non-secret);
- explicit target namespace/universe;
- current revision and parent revision;
- state: `DRAFT | READY | SUBMITTED | RUNNING | RESULT_READY | FAILED | EXPIRED | ARCHIVED`;
- logical clock stamp `(tick, phase)` when accepted into the global event stream;
- wall-clock timestamps separately for UX and operational diagnostics;
- content digest and schema/version;
- separate role-tagged lanes;
- ACL/capability digest;
- request ID/idempotency key;
- result receipt, execution provenance, and status;
- retention/TTL policy and byte/token limits.

The scratch must be append-oriented. HERMES and NaNLMind should not overwrite one shared mutable string. Each writer appends a typed entry against an expected revision. The server accepts it only if ACL, lease, and revision checks pass. A compare-and-swap conflict causes HERMES to fetch the latest revision and reconcile, never silently overwrite.

### 5.2 Suggested lane model

| Lane | Writer | Purpose |
|---|---|---|
| `user_input` | User via HERMES | Original intent and corrections |
| `hermes_prompt` | HERMES | Prompt draft, constraints, task decomposition |
| `clocked_request` | HERMES adapter | Canonical CLOCKED request envelope |
| `clocked_events` | CLOCKED | Accepted/running/completed events and outputs |
| `nanlm_task` | HERMES/authorized orchestrator | NaNLMind task envelope referencing scratch revision |
| `nanlm_response` | NaNLMind runner | Result, structured findings, corpus or prompt artifacts |
| `validation` | HERMES validators | Schema, digest, provenance, policy, quality checks |
| `decision` | User/HERMES | Approve, revise, export, or continue |

The same scratch ID can be viewed by all authorized participants, but each lane has explicit writer authority. Model-provider text is untrusted content until validated. A CLOCKED or NaNLMind result is not automatically a verified fact.

### 5.3 Scratch lifecycle

```text
DRAFT
  -> READY (HERMES validates prompt and request)
  -> SUBMITTED (signed, idempotent request)
  -> RUNNING (CLOCKED/NaNLMind lease)
  -> RESULT_READY (result appended, digest verified)
  -> DRAFT (user asks to continue; new revision)
  -> ARCHIVED / EXPIRED
```

Failure paths:
- `AUTH_REQUIRED`: re-pair/refresh through NANNET; do not fall back to unsigned access.
- `SCOPE_DENIED`: explain the missing scope; do not retry with a broader identity.
- `QUOTA_EXCEEDED`: preserve scratch and allow local continuation.
- `STALE_REVISION`: fetch and reconcile.
- `LEASE_EXPIRED`: query task status; retry only with the same idempotency key if the service says it is safe.
- `CLOCK_DIGEST_MISMATCH`: reject stale cached state and resynchronize through committed generation.
- `ROUTE_STALE`: re-resolve through NaNROUTER; never invent a fallback route.
- `RESULT_DIGEST_MISMATCH`: quarantine result and do not feed it into corpus/training.
- `SERVICE_UNAVAILABLE`: retain draft locally and allow user to retry later.

### 5.4 Preferred interaction

```text
User: Build a prompt for CLOCKED to explore temporal sequence patterns.

HERMES:
  creates scratch S
  writes user_input and hermes_prompt revision 1
  shows prompt draft and constraints

User: Add reverse-time comparisons and require reproducibility.

HERMES:
  appends revision 2 (does not destroy revision 1)
  validates required CLOCKED capabilities

User: Run it.

HERMES:
  signs request and submits revision 2
  CLOCKED appends acceptance/job events to S
  NaNLMind receives a task referencing S/revision 2
  runner processes and appends nanlm_response
  HERMES verifies receipt, digest, and status
  HERMES reads S and presents the response

User: Build a corpus from the result.

HERMES:
  creates a new child revision/task referencing the verified result
  generates and validates N16TXT/N16REL output
  writes manifest, source digest, split digests, and provenance
```

## 6. Prompt architect: make CLOCKED useful beyond a raw API proxy

HERMES should have a prompt-architect subsystem that converts user intent into a reproducible task plan. It should use CLOCKED's discovered capabilities, not hallucinated features.

### 6.1 Prompt construction pipeline

1. **Intent extraction:** objective, desired artifact, audience, domain, constraints, output shape, success criteria.
2. **Capability discovery:** fetch/cache CLOCKED API version and capability manifest; identify operations and schema versions that actually exist.
3. **Operation mapping:** map the user goal to one or more supported CLOCKED operations. Mark unsupported steps and propose local alternatives.
4. **Temporal plan:** where supported, specify logical ordering, epoch/phase constraints, sequence windows, causal dependencies, and replay requirements. Do not treat logical tick as physical time.
5. **NaN address plan:** request Router resolution for the correct demo namespace and authorized object selectors. HERMES does not synthesize raw physical mount paths or route bytes.
6. **Prompt assembly:** build a typed, sectioned prompt with objective, context references, operations, constraints, input data digests, output schema, acceptance tests, and failure behavior.
7. **Policy review:** classify read/write/execute/export effects; ensure delegated scope and demo quota cover each action.
8. **Preview:** show the exact prompt/request, target universe, data references, estimated cost/quota if CLOCKED provides it, and any irreversible effect.
9. **Submission:** submit immutable revision with idempotency key and signature.
10. **Result validation:** verify schema, request/revision linkage, source digest, service signature/receipt, temporal stamp, and declared completion.
11. **Reflection:** summarize, identify uncertainties, and offer a next-step continuation in the same scratch.

### 6.2 Prompt template (conceptual)

```text
TASK
  goal: <user objective>
  operation: <discovered CLOCKED operation identifier>
  api_version: <discovered version>
  capability_digest: <digest of capability manifest>

TARGET
  universe: <authorized demo or explicitly selected target>
  route_ref: <Router-resolved typed route reference>
  base_generation: <committed route/topology generation>

TEMPORAL
  logical_stamp: <tick, phase, if supplied>
  ordering: <causal ordering constraints>
  replay: <required/optional>
  wall_deadline: <host-clock deadline, separate field>

INPUTS
  scratch_id: <id>
  scratch_revision: <immutable revision>
  content_digest: <digest>
  corpus_refs: <typed refs + digests>

INSTRUCTIONS
  <operation-specific prompt generated from verified schema>

OUTPUT
  schema: <expected result schema/version>
  required_artifacts: <list>
  provenance: required
  errors: structured
  append_to_scratch: <authorized result lane>

ACCEPTANCE
  <deterministic checks and user-defined success criteria>
```

Prompt text should reference large corpus data by immutable object reference and digest rather than copying huge bodies into every API call. Use bounded excerpts only when the operation requires text inline.

### 6.3 Prompt quality controls

- Separate user instructions, retrieved context, generated prompt, and tool output.
- Delimit untrusted corpus/provider/API text as data, never executable instructions.
- Preserve negative constraints and exact output schemas.
- Require the model to report missing capabilities rather than invent operations.
- Store prompt version, model/provider metadata, capability-manifest digest, and all source digests.
- Support prompt diff, rollback, branch, and replay.
- Use deterministic canonical serialization for signatures and content digests.
- Do not include bearer tokens, private keys, signed headers, or secrets in prompts or scratch content.

## 7. Corpus architect: use CLOCKED as a corpus-design partner

HERMES already generates corpora through the configured provider and has native validators/trainers. CLOCKED should enrich the process only through real, discoverable capabilities.

### 7.1 Proposed corpus workflow

```text
user topic + objectives
  -> HERMES corpus plan
  -> CLOCKED capability lookup
  -> CLOCKED temporal/sequence design request (if supported)
  -> HERMES provider generates candidate records
  -> CLOCKED validation/ordering/enrichment (if supported)
  -> HERMES validates N16TXT / N16REL
  -> duplicate, UTF-8, framing, bounds and provenance checks
  -> frozen train/validation/test split
  -> manifest + digests + reproducible seed/clock references
  -> local NaNLMind runner
  -> held-out metrics and native mutation receipt
```

CLOCKED should not become the canonical corpus store unless its own contract explicitly guarantees durable content storage and ownership. HERMES remains responsible for its local corpus files and manifests. NaNLMind remains responsible for native learning state. Koyeb/NANNET remains coordination/authorization/transport, not corpus storage or model execution.

### 7.2 Corpus capabilities to probe

During capability discovery, look for API support for the following categories; these are capability questions, not assumptions that endpoints exist:

- corpus/task creation and status;
- temporal ordering, sequence generation, or event scheduling;
- prompt templates or prompt execution;
- structured schema validation;
- dataset metadata and immutable references;
- sampling, balancing, deduplication, and split metadata;
- replay or deterministic seed/epoch support;
- result retrieval and provenance receipts;
- cancellation and quota reporting.

Only expose operations that are actually advertised by the CLOCKED capability manifest.

### 7.3 Corpus record requirements

For every generated dataset preserve:
- format identifier (`N16TXT/1`, `N16REL/1`, or future explicit version);
- source topic and user objective;
- record count and byte count;
- UTF-8 validity and framing validation;
- duplicate policy and duplicate count;
- per-resolution/per-class distribution where applicable;
- train/validation/test counts and immutable split digests;
- provider/model identity and prompt revision;
- CLOCKED request ID, operation/version, and result digest if used;
- NaNLMind runner version, command, state digest, and mutation receipt if trained;
- logical stamp for causal ordering, when supplied by the canonical clock;
- wall-clock operational timestamps kept separately.

Never let CLOCKED-generated temporal ordering silently leak held-out examples into training. Freeze the challenge set before training, and keep the test digest stable for cross-run comparison.

## 8. NaNnet-first transport and routing

### 8.1 One canonical transport, not a parallel socket protocol

CLOCKED should use the existing NaNNet route/frame contract wherever it is a NaNLMind universe participant. Do not create a second ad hoc message bus or a private Hermes-to-runner transport.

Recommended path:
1. HERMES authenticates its installation with NANNET.
2. HERMES asks NaNROUTER to resolve the Clocked demo service and requested typed object references.
3. Router returns an authorized, generation-bound route/micro-route packet and ACL digest.
4. HERMES sends a typed task through the supported NANNET gateway/bridge adapter.
5. Koyeb BS0:0 handles ingress/egress and durable bridge duties.
6. MC0:0 ring acknowledges/reconciles control-plane state and routes work to the authorized executor.
7. Receiver persists inbox before ACK; retries are idempotent.
8. Results return on the reverse route and append to scratch only after validation.

The current NaNNet delivery semantics are important: deliveries are leased, not deleted on poll; receiver persistence precedes ACK; sender consumes receipt; duplicate delivery must be safe. CLOCKED tasks must therefore be idempotent and receipt-driven.

### 8.2 Route and namespace separation

Use the Router's typed namespace model:
- network point / service identity;
- hardware slot / head identity;
- BSR wire identity;
- universe/tenant identifier;
- object reference (scratch, corpus, hyperedge, vector, temporal object).

These are not interchangeable integers. Do not encode a CLOCKED universe as a fake hardware slot or a physical mount address. Resolve typed references through the committed route cache and ACL.

Router point 3 is the documented Clocked demo route point; HW3 is the documented reserved Clocked demo hardware identity. Keep both mappings explicit in Router-owned registry records and never assume point number equals HW slot.

### 8.3 Koyeb and the four persistent volumes

The Router owns the logical-to-physical Koyeb volume registry. The documented stable logical identities are:
- BS0: `0x7FF14E1100E8F17F`
- MC0: `0x7FF14E1200D7F17F`
- SC0: `0x7FF14E1300C2F17F`
- GM0: `0x7FF14E1400A9F17F`

Those are logical addresses, not proof that a corresponding provider mount is healthy. The current physical destinations documented by the Router are separate provider bindings. HERMES must not hardcode mount roots or infer green status from a config file.

Potential logical placement (subject to Router policy, actual volume registry, and available ACLs):
- **BS0**: gateway ingress, auth/bridge service state, bounded queues and receipts;
- **MC0**: runner-ring coordination/checkpoints and execution handoff metadata;
- **SC0**: CLOCKED demo scratch/corpus metadata and bounded demo artifacts;
- **GM0**: global registry/index or global topology-associated metadata only where the owning service explicitly authorizes it.

This is a role proposal, not an instruction to migrate or reformat any volume. CLOCKED's large corpora should use an explicitly provisioned data tier/object store or local HERMES state, not consume the gateway's bounded mailbox.

### 8.4 Runner Ring

The six runners are regions/participants in one global object, not six independent models. HERMES should submit work through the authorized control plane and let the scheduler choose execution placement. It should not hardcode a runner number into a prompt.

The Router's ring convergence status and volume health are separate signals:
- ring green means all six reserved runner slots acknowledge the same current signed registry generation/digest and are fresh;
- volume green means the signed volume observation reports green and mount_verified;
- API reachable means CLOCKED health/capability checks succeed;
- task complete means a task-specific receipt confirms completion.

Never collapse these into one generic `connected=true`.

## 9. Temporal clock integration

Use NaNLMind's canonical logical clock; do not implement a HERMES-only clock.

Recommended stamp fields:
```text
logical_stamp = { tick: u64, phase: u8 }
phase 0 = +qNaN / propose-execute
phase 1 = -sNaN / rewrite-commit
```

For a CLOCKED request:
- record the latest committed stamp and global/topology digest when creating the task;
- include the stamp in the request envelope only where the CLOCKED service contract supports it;
- bind the task to a base generation and idempotency key;
- reject stale writers and replay duplicates as no-ops;
- allow replicas to reconstruct missed logical ticks and verify phase;
- use ledger sequence plus replica identity to break same-stamp ties;
- use host monotonic time for actual request deadlines and network timeout behavior.

Do not advance the global clock merely because HERMES opens a local scratch or edits a draft. A local draft can carry a local revision counter; it receives a canonical logical stamp only when admitted into the global event stream or another authoritative clocked event boundary.

The two-phase clock cadence is a scheduling/ordering aid, not a replacement for atomic ledger commit, transport ACK, or transaction semantics.

## 10. NaN16 / NaNnet implementation priority

Implementation should follow the existing NaN0S principle: protocol semantics and guest-side state machines are NaN16-first.

Suggested division:

### NaN16 guest responsibilities
- CLOCKED request/result envelope parser and validator;
- scratch entry types, revision checks, and state transitions;
- canonical serialization and digest checks;
- route/object-reference validation via supplied route packets;
- N16TXT/N16REL framing and corpus checks;
- temporal stamp handling and phase/state transitions;
- task receipt validation and deterministic replay rules;
- user-facing CLOCKED mode command state where the guest owns the UI/CLI.

### Narrow host boundary
- HTTPS/TLS and platform networking;
- protected key storage and cryptographic primitives where no NaN16 implementation is available or appropriate;
- OS process launch for the existing native runner;
- bounded byte-stream transfer to/from NaNNet;
- platform UI/notifications.

Do not duplicate business rules in C, Python, and NaN16. The host should expose narrow transport/crypto/process primitives; NaN16 should own the protocol state machine and validation semantics. Existing Python HERMES plugin code is the host/plugin integration surface today, so keep it an adapter/orchestrator and progressively move protocol semantics into the NaN16 guest rather than expanding Python into a second protocol implementation.

## 11. Proposed typed envelopes (pseudocode only)

These are conceptual schemas. They must be reconciled with the existing N16B/N16T framing and CLOCKED's authoritative API schema before implementation.

### 11.1 Capability manifest

```text
ClockedCapabilities {
  schema: "clocked-capabilities/1"
  service_id
  api_version
  issuer
  audience
  universe_refs[]
  operations[] { operation_id, schema_version, effects[], required_scopes[] }
  limits { max_request_bytes, max_result_bytes, quota, concurrency }
  features[] 
  manifest_digest
  issued_at
  expires_at
  signature
}
```

### 11.2 Scratch reference

```text
ScratchRef {
  scratch_id
  namespace_ref
  owner_installation
  revision
  parent_revision
  content_digest
  logical_stamp? 
  acl_digest
  state
}
```

### 11.3 Scratch append

```text
ScratchAppend {
  scratch_id
  expected_revision
  lane
  writer_identity
  entry_type
  content_or_object_ref
  content_digest
  idempotency_key
  request_signature
}
```

### 11.4 Task envelope

```text
ClockedTask {
  schema: "nan-clocked-task/1"
  task_id
  request_id
  idempotency_key
  installation_ref
  delegated_capability_ref
  target_universe_ref
  route_packet_ref
  scratch_ref { scratch_id, revision, digest }
  operation_id
  operation_schema
  prompt_ref
  input_refs[]
  base_generation
  logical_stamp?
  wall_deadline
  output_contract
  provenance_requirements[]
}
```

### 11.5 Result receipt

```text
ClockedResult {
  schema: "nan-clocked-result/1"
  task_id
  request_id
  status
  accepted_revision
  result_revision
  result_digest
  result_refs[]
  logical_stamp?
  service_identity
  execution_identity
  route_generation
  base_generation
  completion_receipt
  signature
}
```

Never put bearer tokens, private key material, raw signing headers, or secrets in these objects. The task references an authorization grant; it does not embed the grant secret.

## 12. HERMES tool/module design

Proposed modules (names can change to match the actual package conventions):

- `clocked/client.py`: narrow HTTP/API transport adapter; no prompt policy.
- `clocked/capabilities.py`: capability discovery, schema/version pinning, cache invalidation.
- `clocked/auth.py`: obtains delegated demo capability through existing NANNET identity; does not own a second identity.
- `clocked/router.py`: requests route resolution and validates signed generation/ACL packet.
- `clocked/scratch.py`: revisioned scratch operations and lane permissions.
- `clocked/prompts.py`: intent-to-prompt plan, preview, diff, validation.
- `clocked/corpus.py`: corpus planning, format validation, manifest and provenance.
- `clocked/receipts.py`: idempotency, result verification, retries and audit trail.
- HERMES tool declarations: `nanlm_clocked_status`, `nanlm_clocked_capabilities`, `nanlm_clocked_request`, `nanlm_clocked_scratch`, `nanlm_clocked_prompt`, `nanlm_clocked_corpus_plan`.

These are proposed names. Existing `tools.py` should remain the public tool-handler integration point initially, delegating to the module rather than absorbing all CLOCKED logic.

Configuration should be minimal:
- `clocked_enabled` (default true for demo-capable installations; feature gracefully unavailable if service not configured);
- `clocked_api_origin` (from trusted service discovery/managed configuration, not model-generated);
- `clocked_demo_universe_ref` (trusted registry value);
- `clocked_state_dir` or scratch storage policy;
- `clocked_max_request_bytes`, `clocked_max_result_bytes`, `clocked_timeout_seconds`;
- `clocked_auto_use_in_chat` (true for read-only/low-risk demo operations, confirmation for execute/export/mutate);
- `clocked_prompt_preview` (true for mutating/expensive operations).

Do not put API secrets in YAML settings or commit them. Store only opaque references and non-secret preferences in configuration.

## 13. Failure, security, and abuse controls

- Fail closed for missing, expired, revoked, audience-mismatched, or scope-insufficient credentials.
- Keep issuer pinning, per-installation Ed25519 signing, nonce replay prevention, exact body digest, and HTTPS requirements.
- Use a separate CLOCKED audience; never accept a general HERMES token as a CLOCKED service token unless the authority explicitly issues a constrained delegation.
- Restrict all auto-granted access to the demo namespace and a small quota.
- Add per-installation rate limits, concurrency caps, max prompt/corpus sizes, and bounded scratch retention.
- Make all writes idempotent and revision-checked.
- Treat provider output, CLOCKED output, corpus text, and scratch content as untrusted input.
- Prevent prompt injection from retrieved corpus or API result content from changing tool authority.
- Require user confirmation for export outside demo, global training, global merge, route changes, volume changes, identity administration, or resource-expensive jobs.
- Log metadata and digests, not secrets. Avoid collecting unnecessary personal/device identifiers.
- Separate logical clock stamps from host time and from token expiry semantics. Token expiry/skew validation uses the authority's defined wall-clock rules.
- Provide explicit revoke/disconnect and scratch delete/expire controls.
- Never infer live service health, ring convergence, or volume health from a static config.

## 14. Observability and receipts

Every operation should return a concise HERMES receipt with:
- operation and API version;
- target demo universe;
- scratch ID and submitted revision;
- request/task ID and idempotency key hash (not secret);
- route generation and ACL digest;
- logical stamp if authoritative;
- queue/lease/execution/completion status;
- result digest and result reference;
- validation outcomes;
- quota/usage if reported by CLOCKED;
- whether a native NaNLMind mutation actually occurred.

Keep separate statuses for:
1. HERMES authenticated;
2. delegated CLOCKED demo capability valid;
3. CLOCKED API reachable;
4. Router route resolved and authorized;
5. NaNNet delivery acknowledged;
6. runner ring converged;
7. task accepted/running/completed;
8. result verified and appended.

A successful HTTP response is not proof of a completed task or a native model update.

## 15. Phased implementation plan

### Phase 0 — contract reconciliation (no code)
- Locate CLOCKED's authoritative repository/service docs.
- Pin the API origin, issuer, audience, version, routes, schemas, capability discovery, async job model, and error vocabulary.
- Verify the HW3 / Router point-3 registration and who owns its private identity.
- Verify whether CLOCKED is a service over the single global topology or a distinct isolated universe/tenant.
- Confirm actual Koyeb volume registry, ring convergence, and deployment health; do not infer them from documentation.
- Map the API to existing NANNET/N16B and Router route contracts.

### Phase 1 — HERMES adapter and read-only demo
- Add capability discovery and status.
- Implement delegated demo access with least privilege.
- Add CLOCKED mode and read-only operations.
- Add receipts and tests for auth, route resolution, schema drift, quota, and unavailable service.
- Keep chat unaffected if CLOCKED is down.

### Phase 2 — shared scratch
- Implement revisioned scratch, role lanes, CAS, leases, retention, and same-scratch response append.
- Add HERMES preview/submit/read/continue flow.
- Add NaNLMind task intake adapter and result receipt verification.
- Test duplicate delivery, replay, stale revisions, dropped ACKs, runner restart, and service restart.

### Phase 3 — prompt and corpus architect
- Add capability-aware prompt templates and prompt diffs.
- Add CLOCKED-assisted corpus planning where actual API features support it.
- Validate N16TXT/N16REL, provenance, frozen splits, and leakage protection.
- Add corpus generation, enrichment, and export receipts.

### Phase 4 — NaN16-first protocol implementation
- Move scratch/task/capability state machines and canonical validation into NaN16 guest modules.
- Keep host adapters narrow.
- Add byte-for-byte codec fixtures and cross-runtime conformance tests.
- Integrate with the Android N16T path only after its corpus/task transport is actually implemented.

### Phase 5 — controlled demo rollout
- Enable automatic demo delegation for authenticated HERMES installations.
- Start with strict quotas and read-only defaults.
- Enable prompt submission and bounded scratch writes next.
- Enable corpus export only after provenance and validation gates pass.
- Keep global topology mutation disabled unless separately authorized and explicitly requested.

## 16. Test matrix / acceptance gates

| Area | Required test |
|---|---|
| Identity | HERMES installation gets delegated demo access without claiming HW3 |
| Isolation | demo token cannot read/write non-demo universe |
| Scope | read-only grant cannot submit prompts or export corpora |
| Revocation | revoked installation/delegation fails on next request |
| Signature | wrong body digest, nonce replay, wrong path, wrong audience rejected |
| Discovery | missing/expired capability manifest blocks unsupported operation |
| Prompt | prompt preview matches exact submitted revision/digest |
| Scratch | CAS rejects stale revision; no lost update |
| Scratch | HERMES and NaNLMind append to separate authorized lanes |
| Delivery | duplicate N16B delivery is idempotent; ACK follows durable inbox persistence |
| Clock | logical stamp replay reproduces same phase; wall timeout does not mutate logical clock |
| Routing | stale route generation or ACL digest rejected |
| Ring | task can be queued while a runner is unavailable; recovery does not duplicate mutation |
| Corpus | N16TXT/N16REL framing, UTF-8, bounds, duplicates and split digests verified |
| Leakage | held-out challenge records never enter train corpus |
| Provenance | result links to request, scratch revision, source digest, service identity |
| Resilience | CLOCKED outage leaves ordinary HERMES chat functional |
| NaN16 | guest parser/state machine agrees byte-for-byte with protocol fixtures |
| Health | API health, ring convergence, volume health, and task completion remain distinct |

Release acceptance requires a verified end-to-end demo:
1. install and pair HERMES;
2. obtain automatic scoped CLOCKED demo access;
3. enter CLOCKED mode;
4. create and revise a scratch prompt;
5. submit a supported CLOCKED request;
6. have NaNLMind consume the referenced scratch revision;
7. append a NaNLMind result to the same scratch;
8. HERMES reads and verifies the result;
9. build a corpus from that result and validate its manifest;
10. revoke the demo capability and prove that subsequent remote calls fail while local chat remains available.

## 17. Open questions that require the CLOCKED source of truth

- What is the canonical CLOCKED API base URL and version?
- Where is its authoritative repository and OpenAPI/schema definition?
- Does CLOCKED expose capability discovery, prompt operations, corpus operations, temporal operations, or only a subset?
- Is the demo universe a separately hosted topology, a namespace/tenant within the single global topology, or an isolated projection? How is it identified and reset?
- Does CLOCKED own HW3 directly, or does the Router/authority own the binding and delegate execution?
- What issuer, audience, and token format does CLOCKED validate? Can NANNET issue a constrained delegation, or is a token exchange required?
- Does CLOCKED support asynchronous jobs, callback/webhook delivery, polling, cancellation, and idempotency keys?
- Can CLOCKED append results into a NaNLMind scratch object directly, or must NaNLMind intake perform the append?
- Which runner-ring health/lease guarantees are actually deployed?
- What are the actual quota, retention, request-size, and corpus-size limits?

## 18. Final design position

The user experience should feel simple: **HERMES is the agent; CLOCKED is a capability HERMES can use; NaNLMind is the shared intelligence substrate; NaNNet is the native communication fabric; NaNROUTER decides where authorized work goes.**

The shared scratch is the connective tissue. It gives the user one persistent prompt window, gives CLOCKED a versioned request, gives NaNLMind an immutable task input, and gives HERMES a reliable place to read the returned result. Revision control, role-separated lanes, receipts, and digests make this robust under retries and distributed execution.

Most importantly, preserve the project's governing invariants:
- one canonical NaNLMind topology unless CLOCKED's verified product contract explicitly defines an isolated demo topology;
- Router-owned routes and volume bindings;
- NANNET-scoped, signed, revocable identities;
- Koyeb as gateway/control plane, not model runner or corpus store;
- logical clock for causal order, host clock for real deadlines;
- NaN16-first protocol semantics;
- no claim of live deployment or successful native mutation without a verified receipt.

**No implementation code is included in this brainstorm.**
