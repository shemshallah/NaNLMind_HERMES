# Autonomous CLOCKED Mode and NaN Head Cortex

**Status:** partial implementation; upstream service deployment and candidate evaluation/promotion remain outstanding  
**Service:** [NaNLMind_CLOCKED](https://github.com/shemshallah/NaNLMind_CLOCKED)  
**Native learner:** [NaNLMind](https://github.com/shemshallah/NaNLMind)  
**Canonical detailed design:** [HERMES Autonomous CLOCKED Cognition and NaN Head Cortex](https://github.com/shemshallah/NaNLMind_CLOCKED/blob/main/docs/HERMES_AUTONOMOUS_CLOCKED_HEAD_CORTEX.md)


## Autonomous mode selection

HERMES now asks its configured model to select among the fixed CLOCKED modes
using the user request and source descriptors (IDs/types, not raw source
bodies). The selected mode is mapped locally to a fixed endpoint set; model
output cannot supply arbitrary endpoints or task dependencies. Invalid,
unavailable or out-of-policy decisions fall back to the deterministic
selector. Any mode that transmits source material still requires explicit
approval before gateway execution.

## 1. Integration objective

HERMES should autonomously decide when an external CLOCKED operation is useful, construct a bounded plan, retrieve several independent CLOCKED outputs, reconcile them into one revisioned scratch, submit the complete typed evidence package to NaNLMind's native M0-M9/head pipeline, and synthesize its answer from the resulting mathematical relation graph.

The answer should receive a second head pass before delivery:

```text
user question + source material
  -> HERMES mode planner
  -> CLOCKED task graph
  -> multi-source typed observations
  -> shared revisioned scratch
  -> NaN16 evidence package
  -> NaNLMind M0-M9 transformations
  -> Boolean-derived relation/head mapping
  -> HERMES answer draft
  -> draft submitted through NaNLMind head validation
  -> HERMES final response
  -> optional candidate learning record
```

This should be an extension of the existing HERMES chat and training loop, not a second conversational agent or independent head universe.

## 1A. Implemented integration surface

The current HERMES branch now includes:
- deterministic autonomous mode selection for local chat, extraction, verification, multi-source comparison, privacy audit and injection/script audit;
- bounded endpoint task plans (maximum 16 calls) with ordered dependencies and explicit approval before source transmission;
- a CLOCKED gateway client that calls only the NaNLMind_CLOCKED participant gateway;
- append-only, hash-chained scratch lanes for source results, extracted evidence atoms, conflicts, NaN16 packages, head traces, drafts and final responses;
- two-pass answer synthesis: HERMES draft -> NaNLMind Boolean-head relation evaluation -> HERMES final;
- a scoped NaNLMind head-candidate endpoint backed by native `HeadView::self_modify_boolean`, producing isolated candidate state without changing canonical topology;
- a held-out candidate-evaluation tool that compares baseline and candidate relation-eval metrics, records accuracy/coverage/balanced-accuracy/utility deltas by resolution, and still cannot promote.

The native candidate path is deliberately candidate-only. It is not yet a complete autonomous learning loop: multi-control scoring, bounded mutation search, reviewer workflow and six-runner generation-barrier promotion remain to be implemented. The current head-cortex inference endpoint is read-only and does not claim its Boolean predictions are ground truth.

## 2. Repository and code ownership

### HERMES owns
- default chat and explicit CLOCKED workspace/mode;
- autonomous mode planner and deterministic plan policy;
- HERMES-facing CLOCKED gateway client (not the upstream SDK);
- revisioned scratch and source reconciliation;
- conversion of endpoint results into a typed NaN16 evidence package;
- NaNLMind head-cortex client and answer synthesis;
- CLI/GUI commands, user approval and budget controls.

### NaNLMind_CLOCKED owns
- Rezolve.ai SDK and upstream API key;
- service gateway and participant credentials;
- CLOCKED endpoint adapters, fan-out, retry/idempotency, credit accounting;
- CLOCKED-specific fixtures, demo data and service tests.

### NaNLMind owns
- authoritative schemas and NaN16 receiving contract;
- M0-M9 native transformations;
- 256 Boolean heads, HeadGate, LogicSessionTrainer and N16REL relation learner;
- evidence consumer, generation compatibility, six-runner ring and canonical commits.

HERMES must not import `clocked_sdk`, store the upstream `CLOCKED_API_KEY`, or call Rezolve directly. It calls the CLOCKED service with a scoped HERMES token. HERMES must not clone six-runner orchestration or create a second canonical head state.

## 3. Existing HERMES code to extend

The current codebase already provides the right attachment points:

| Existing module | Role in the new path |
|---|---|
| `nanlmind_hermes/chat.py` | Keep as default conversational entry, transcript and reflection owner; add mode dispatch before selecting local-only vs hybrid work |
| `nanlmind_hermes/tools.py` | Add domain handlers for mode planning, CLOCKED status/plan/run, scratch and head-cortex calls |
| `nanlmind_hermes/runner.py` | Keep as narrow host-process wrapper for existing native NaNLMind CLI; do not turn it into a distributed runner |
| `nanlmind_hermes/memory.py` | Keep append-only conversational evidence/index and assessments; do not mistake its SQLite projection for canonical native head state |
| `nanlmind_hermes/extract.py` | Reuse typed discovery extraction and validation where compatible |
| `nanlmind_hermes/policy.py` | Extend with CLOCKED operation scopes, byte/cost limits and explicit approval policy |
| `nanlmind_hermes/http_api.py` | Extend only if a local HERMES API surface is needed; never expose privileged CLOCKED credentials |

Proposed new modules (names are design proposals; reconcile with the current plugin API before coding):

- `mode_planner.py`: intent features, mode proposal, budget/policy validation.
- `clocked_client.py`: authenticated client for NaNLMind_CLOCKED gateway only.
- `clocked_mode.py`: validated DAG execution and bounded replanning.
- `scratch.py`: append-only revisioned lanes, ACL and digest handling.
- `evidence_codec.py`: source-specific result normalization and NaN16 package encoding.
- `head_cortex.py`: client for NaNLMind cognition/transform interface; no local duplicate HeadArray.
- `answer_validation.py`: second-pass answer claim validation and bounded correction.
- `mutation_review.py`: inspect and evaluate candidate mutation receipts; no direct canonical mutation.

## 4. Autonomous mode policy

Modes are per-turn execution strategies, not personalities:

- `LOCAL_CHAT`: ordinary conversation, explanation, brainstorming and creative work.
- `CLOCKED_CLASSIFY`: classify intent/task before selecting further work.
- `CLOCKED_EXTRACT`: exact values, fields, dates and spans.
- `CLOCKED_VERIFY`: ground claims against declared sources.
- `CLOCKED_COMPARE`: compare versions/documents and build conflict candidates.
- `CLOCKED_RETRIEVE`: rerank a large source set, then extract/ground selected passages.
- `CLOCKED_GUARDED`: PII/injection/script checks for untrusted inputs.
- `CLOCKED_MULTI_SOURCE`: multiple independent API calls with explicit reconciliation.
- `CLOCKED_PIPELINE`: sequential CLOCKED Pipeline where its one-input semantics fit.
- `NAN_HEAD_REASON`: Boolean/logical mapping without external calls.
- `HYBRID`: CLOCKED observations -> scratch -> NaNLMind heads -> HERMES synthesis.

The model proposes a schema-bound plan. Deterministic policy validates endpoints, source ACLs, input limits, dependency graph, idempotency, quotas, API scopes, scratch revision and user approval. No free-form model text may directly trigger arbitrary calls.

Three execution policies:
- `offline`: never call CLOCKED.
- `approval`: show the plan and wait for approval.
- `budgeted_auto`: execute only within a user-configured credit/request/byte budget.

Default to approval. A planner deciding CLOCKED would help is not the same as permission to spend credits or transmit sensitive data. Local chat should not call CLOCKED.

## 5. Multi-source outputs and shared scratch

CLOCKED Pipeline is a sequential operation over one text input. It is not a general multi-source DAG. For multiple sources, HERMES makes separate typed calls, preserves each result, and merges them only after source-level validation.

Example:

```text
question
  -> guard / privacy scan
  -> source A: extract + ground
  -> source B: extract + ground
  -> source C: extract + ground
  -> compare
  -> claim-level reconciliation
  -> rerank relevant evidence
  -> decide using explicit criteria
  -> append all results to scratch
  -> NaNLMind head processing
```

Each branch preserves request ID, source digest, endpoint/method, schema version, idempotency key, status, cost, latency, retries and exact result reference. A failed branch stays visible; it is never silently omitted.

Scratch lanes:

| Lane | Writer | Contents |
|---|---|---|
| `user` | user | original question, corrections and source refs |
| `hermes_plan` | HERMES | mode, plan graph, budgets and policy result |
| `clocked_request` | HERMES adapter | immutable request snapshots/digests |
| `clocked_result` | CLOCKED service | typed response refs and receipts |
| `evidence_atoms` | normalizer | claims, fields, exact spans, provenance |
| `conflict_map` | reconciler | supports, contradicts, qualifies, duplicates, supersedes |
| `nan16_package` | HERMES | immutable package submitted to NaNLMind |
| `head_response` | NaNLMind | M0-M9 receipts, relation graph, active-head trace, abstentions |
| `hermes_synthesis` | HERMES | answer draft and unresolved issues |
| `head_validation` | NaNLMind | coverage, contradiction and unsupported-claim checks |
| `final` | HERMES | final answer and source references |
| `learning_proposal` | HERMES/learner | candidate record, never auto-admitted |

Each append uses expected revision and parent revision. Writers cannot overwrite one another's lanes. Scratch content is ACL-protected; global NaNLMind telemetry remains content-minimized. Do not store credentials in scratch or prompts.

Preserve native score semantics: Decide probability, Extract span, Grounding verdict, Compare conflict signal and Rerank relevance are different types. Never average them into one generic confidence score. Conflicts remain explicit until source authority, date or direct applicability resolves them.

## 6. NaN head cortex: adapt the head primitive, not the runner ring

The “clone runners and remove runner code” idea should produce a **head cortex facade** in HERMES, backed by NaNLMind's existing head implementation.

Reuse in NaNLMind:
- `HeadArray` / `HeadView` / `HeadRunner`;
- each head's Boolean truth-table learner;
- `HeadGate` relation gates;
- `LogicSessionTrainer` mandatory Boolean foundation;
- `N16REL/1` relation record format and `relation_train.cpp`;
- existing M0-M9 transformations and provenance gates.

Do not copy:
- six-runner ring membership, leader election, heartbeats, leases or sharding;
- distributed handoff and generation barrier;
- Koyeb volume administration;
- canonical checkpoint ownership;
- global route/ACL mutation.

HERMES schedules bounded tasks against the existing 256 logical head IDs and receives results. It does not own the head objects or a competing copy of their state. The HERMES `runner.py` remains only a local process wrapper for the native CLI. The six-runner infrastructure remains in NaNLMind.

### 6.1 Cortex stages

```text
IntentProjector
  -> EvidenceAtomizer
  -> M0 Boolean/truth validity
  -> M1 bytes, UTF-8 spans and offsets
  -> M2 schema types and head routing
  -> M3 deterministic feature projection
  -> M4 temporal ordering/version relations
  -> M5 relation/hyperedge construction
  -> M6 Boolean inference and proof composition
  -> M7 provenance/ACL/route checks
  -> M8 cross-source support/conflict synthesis
  -> M9 calibration, abstention and mutation admission
  -> RelationAssembler
  -> NaNHeadResponse
```

These are task-profile mappings to the existing canonical layers, not a second layer topology. Bind each profile to existing NaNLMind layer IDs and tests before implementation.

### 6.2 Boolean-derived relations

Represent a claim as a typed atom and relation edges such as:
- supports / contradicts;
- implies / requires / excludes;
- qualifies / duplicates;
- precedes / supersedes;
- missing-from / derived-from.

Truth states are `TRUE`, `FALSE`, `UNKNOWN`, and `CONFLICT`. Unknown is not false. Conflict is not resolved by averaging.

Each inferred relation carries premise atom IDs, Boolean gate/rule IDs and versions, M0-M9 path, source provenance, independent verification status, confidence semantics and counterexample-search status. Boolean gates prove a conclusion follows from premises under a rule; they do not prove the premises are true.

The cortex should meta-map relations between relations: recurring proof shapes, common prerequisites, temporal contradictions, missing premises and cross-source dependencies. These become higher-order hyperedges linked back to the original evidence and proof graphs.

## 7. Self-modifying heads

Self-modification should be bounded evolution of Boolean rules, gate selection, relation-feature mappings and routing preferences—not arbitrary executable-code rewriting or direct canonical state mutation during a chat.

Four tiers:
1. **Ephemeral adaptation:** temporary per-turn scratch activations; discarded after the turn.
2. **Candidate mutation:** one head proposes a bounded change to its Boolean table/gate/routing bias in an isolated candidate state.
3. **Cortex experiment:** candidate set tested against truth-table regressions, deep circuits, held-out examples, adversarial counterexamples, shuffled-label controls and no-CLOCKED baselines.
4. **Promotion proposal:** accepted profile is submitted to NaNLMind's evaluator; only the six-runner generation barrier can promote canonical head changes.

A mutation proposal must bind head ID, base digest/generation, before/after truth table, rule/gate ID, evidence refs, counterexamples, expected effect, mutation budget and proposer version. It is rejected if stale, unbounded, provenance-invalid or based on unverified CLOCKED labels as ground truth.

Keep the old head state and rollback pointer. Require deterministic replay, calibration/abstention checks and no regression across existing Boolean and relation tests. CLOCKED agreement alone is not a reward signal: multiple CLOCKED outputs may share one upstream model and are not independent truth sources.

## 8. Two-pass answer cycle

**Pass A — evidence cognition**
1. Freeze the current scratch revision.
2. Encode the user request, evidence atoms, exact spans/refs, source metadata, result types, conflicts and policy metadata into the NaN16 evidence package.
3. Send it to NaNLMind through the existing authorized NANNET route / HW3:0 contract.
4. NaNLMind runs M0-M9 and the Boolean head cortex.
5. Return relation graph, proof refs, active-head trace, uncertainty, conflicts and abstentions.
6. HERMES writes a natural-language draft from the scratch plus head response.

**Pass B — answer validation**
1. Append the draft and claim-to-source map to scratch.
2. Send the draft as a new `ANSWER_VALIDATION` NaN16 input through the same NaNLMind interface.
3. Heads check source coverage, contradiction, unsupported inference, temporal mismatch and uncertainty.
4. HERMES applies only supported corrections.
5. Permit at most one automatic correction pass; then return unresolved disagreement rather than loop indefinitely.
6. Append and deliver the final answer with evidence references.

This ensures the original question/evidence and the synthesized answer both pass through the NaN head pipeline.

## 9. Learning is a separate lane

Every interaction may produce a candidate learning record with source digests, plan version, CLOCKED result refs, head trace and final answer digest. It is not automatically a training example.

Admission requires provenance/license/consent, independent target or human review, split isolation and the native learner's schema. CLOCKED outputs are teacher observations, not gradients or correctness labels. The six-runner ring evaluates and commits accepted candidate generations.

## 10. CLI surface

Use the same domain services from GUI and CLI:

```text
/nanlm mode auto|local|clocked
/nanlm clocked status
/nanlm clocked capabilities
/nanlm clocked budget
/nanlm clocked scratch new|show
/nanlm clocked plan
/nanlm clocked run
/nanlm clocked sources
/nanlm clocked synthesize
/nanlm heads inspect
/nanlm heads trace
/nanlm heads validate
/nanlm heads mutations
/nanlm heads mutation inspect <id>
/nanlm heads mutation test <id>
/nanlm heads mutation accept <id>
/nanlm heads mutation discard <id>
```

Existing `/nanlm` chat and training modes remain unchanged. CLOCKED mode is an added workspace. The GUI and CLI must call the same planner, scratch, service client and head-cortex client.

## 11. Implementation order

1. Keep CLOCKED upstream service code in NaNLMind_CLOCKED; keep only NaNLMind-side schemas/consumer/native learning in NaNLMind.
2. Freeze the NaN16 evidence-package and NaNHeadResponse schemas.
3. Implement scratch revisions and writer lanes.
4. Implement HERMES gateway client and status/capability view.
5. Implement bounded multi-source task graph and typed evidence normalization.
6. Implement autonomous mode proposal plus deterministic policy/budget gate.
7. Implement NaNLMind head-cortex client over the existing HW3:0/NANNET route.
8. Add two-pass answer validation.
9. Add candidate-only head mutation and regression harness.
10. Add training admission and six-runner promotion integration.

## 12. Implementation status

**Implemented first slice (committed and covered by Python CI):**
- Hash-chained append-only HERMES scratch journal with revision conflict checks and lane writer ownership.
- Deterministic mode proposal, bounded task-plan validation and exact-plan approval digest.
- Scoped CLOCKED gateway client and HERMES multi-source orchestration.
- CLOCKED service `POST /v1/tasks` with endpoint allowlisting, dependency ordering, request quotas, idempotency and per-endpoint typed outputs.
- Extraction-to-grounding dependency resolution in the CLOCKED gateway.
- NaN16 evidence-package codec and HERMES head-cortex client.
- Two-pass HERMES synthesis tool: draft -> NaNLMind answer-validation request -> final synthesis.
- NaNLMind native read-only `relation-eval` command and authenticated head-cortex HTTP adapter, with NANNET scope authorization support.
- Native `head-candidate` command and NANNET `training:execute`-authorized HERMES candidate endpoint. It applies `HeadView::self_modify_boolean` to an isolated topology copy, binds the base digest, records before/after digests, and refuses to overwrite canonical state.
- Candidate evaluation compares the candidate and canonical baseline with the same N16REL rows. The caller declares the test split and supplies a canonical dataset digest, which NaNLMind verifies; this binds the submitted records but does not independently attest dataset provenance. Metrics include accuracy, coverage, balanced accuracy, confusion counts and per-resolution results. Stale base/candidate digests are rejected and evaluation cannot promote.
- Shared JSON schemas for evidence packages, scratch entries and head-cortex responses.

**Live-verified in 1.2.1:**
- Production CLOCKED participant token provisioning, gateway configuration,
  task-plan execution, and revisioned scratch writes.
- Production NANNET pairing, pinned issuer discovery, online scope
  authorization, and revocation.
- Local native Boolean gate and runner smoke tests.

**Not yet deployed or end-to-end verified:**
- NANNET-authorized head-cortex service deployment and route registration through the live Router/Koyeb topology.
- Live CLOCKED SDK calls for every direct endpoint method; the live smoke test covers `pii`, while the remaining methods use fixtures and CI fakes.
- Production multi-source scratch transport between devices/services.
- Automated multi-mask search, broader controls, candidate retrieval/review UI and six-runner generation-barrier promotion. Candidate creation, weak-supervision training and held-out baseline comparison are implemented; promotion is not.

The native `relation-eval` path is read-only: it evaluates relation records against an existing serialized topology and does not train or mutate canonical state. The service currently returns Boolean-head relation predictions and provenance; it is not yet a full general-purpose semantic reasoning endpoint.

## 13. Acceptance criteria

- Local chat remains the default and does not call CLOCKED.
- Autonomous planner proposes only schema-valid, budget-bounded plans.
- User approval is required unless budgeted-auto was explicitly configured.
- Multiple CLOCKED branches preserve source identity and native result types.
- Scratch revisions are immutable and concurrent writers cannot overwrite one another.
- Both evidence package and final answer draft are processed by NaNLMind heads.
- Unknown/conflict states survive every layer.
- Head proof graphs preserve premises, gates, source refs and provenance.
- Candidate mutations are isolated, reversible and pass controls.
- HERMES and CLOCKED cannot commit canonical weights or alter route topology.
- The six-runner barrier remains the only canonical promotion path.
- Secrets never enter source, prompt, scratch, telemetry or corpus.
