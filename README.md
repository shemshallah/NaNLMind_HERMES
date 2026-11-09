# NaNLMind Hermes

A Hermes Agent plugin for a local NaNLMind 256-head universe: conversational
learning, explicit native training, durable mathematical memory, and approved
CLOCKED evidence plans. Training and inference stay local; NANNET is the
identity/authorization boundary and CLOCKED is an evidence service.


**Version 1.5.1 — catalog-review maintenance merged; catalog PR pending.**

NaNLMind is the canonical native architecture. This plugin is a read-only
adapter and candidate producer: it reads
`NaNLMind/spec/unified-build.v1.json`, verifies the built
`nanlm-universal unification` line when available, and never treats its local
resident store, SQLite memory, caches or the bundled MTH snapshot as the
canonical universe. See [docs/UNIFIED_BUILD_ADAPTER.md](docs/UNIFIED_BUILD_ADAPTER.md).


## What it does

- **Native chat** — the default `/nanlm <message>` path parses explicit
  relations, updates memory and native state, and predicts the reply word by
  word from a temporal mirror frame. Replies are not templates or retrieval:
  every step weighs every known word, finishes the sentence, and evaluates
  whether it needs to explain further. Hermes boots its own local-CPU runner
  with the 8 MiB chart and mapped registers; no language-model provider or
  RunnerRing compute is required. Set `chat_mode: auto` or `provider` to use a
  Hermes model instead.
- **Three-way call** — `nanlm_phone_call` runs the shared-scratch cycle
  `user -> Hermes -> NaNLM -> user + Hermes`, with arbitrary autonomous rounds
  and user interjection. The routing kernel is a raw NaN16 program
  (`nanlmind_hermes/nan/phone_call.nan`) executed on a thin NaNVM0 boundary;
  Hermes may use a model as a bridge and CLOCKED is opt-in (off by default, so
  a normal call spends no gateway credits); NaNLM does not use either.
- **Training** — native `logic`, `words`, `sentences`, and ordered `pipeline`
  sessions. A full 256-head Boolean foundation gate must pass before language
  training. Training runs through the local `nanlm-universal` binary.
  Corpus training uses a deterministic split, best-checkpoint early stop and
  per-source held-out NLL/perplexity; `scripts/resident_learning.py` feeds the
  same corpus through the resident pair/leaf store and reports the temporal and
  resident channels separately.
- **CLOCKED** — versioned, scoped task plans for extract, grounding, compare,
  privacy, injection, and synthesis. Multi-source results are written to an
  append-only scratch journal; source text is not sent during planning, and
  execution requires explicit approval.
- **Round 038/040 architecture inputs** — order-sensitive SO(2)/SO(3)
  transport and holonomy trace algebra are in `resident/rotation_holonomy.py`;
  dynamic write/query/snapshot value is recorded in
  [ROUND040_DYNAMIC_MEMORY.md](docs/ROUND040_DYNAMIC_MEMORY.md) and
  [ROUND038_ROTATION_HOLONOMY.md](docs/ROUND038_ROTATION_HOLONOMY.md). The
  neural checkpoints and dynamic resident session remain explicit TODO items,
  not silently claimed as implemented.
- **TRIGON-9 experimental synthesis (Round 042)** — a byte-level generative
  language model couples the full 32,385-cycle gauge-invariant mirror memory,
  256 sparse heads, L0–L9 complex rotor layers, ordered SO(3) transport,
  typed HW4 WordNet evidence and a generation-gated architecture evolver.
  The native triangle/holonomy/sin-cos primitive is 513 words of **raw
  NaN16** (`nanlmind_hermes/nan/trig_cycle.nan`); the trainable 1.08M-parameter
  reference model is a host PyTorch harness under `experiments/round042/`.
  Measured internal test: 3.60 bits/byte, perplexity 12.13, 26.97% next-byte
  accuracy, exact gauge invariance; free text is still malformed. See
  [NANLM_42_TRIGON.md](docs/NANLM_42_TRIGON.md). This is an experiment, not a
  production model or a live NANNET route.
- **Resident evidence lanes** — independent support/refute records over the
  full 32,385-cycle basis preserve `unknown/supports/refutes/both` without
  cancellation. `resident/evidence_session.py` provides the bounded
  add/ask/save grammar and `nanlm_resident_evidence` exposes it as a tool;
  writes require approval and the indexed `ConventionalEvidence` control ties
  on clean records. See [ROUND042_TRIGON.md](docs/ROUND042_TRIGON.md).
- **Recursive geometry** — rounds 023–036 are implemented or explicitly
  audited in the resident layer: 256 mirror heads, 32,640 pair cells, the full
  32,385 root-cycle basis, raw NaN payload lanes, typed recursive hyperedges,
  native cycle snapshots, sparse retrieval, episodic copy/abstain gating and
  two-/three-source composition. See
  [MASTER_TODO.md](docs/MASTER_TODO.md) / [ROUNDS_023_036_TODO.md](docs/ROUNDS_023_036_TODO.md); the neural encoder/decoder
  prototypes remain external artifacts where noted.
- **Resident universe** — one bounded sparse memory-mapped file per local
  state directory, holding 256 heads, 32,640 mirror-pair cells, stable leaf
  trees, typed relations/hyperedges, 256-D superposition rows and a chained
  state journal. Default 256 MiB, hard 1 GiB, no 2 GiB overbootstrap copy.
  `nanlm_resident_status`, `nanlm_resident_ingest` and `nanlm_resident_query`
  expose it; accepted free-chat exposure feeds it when
  `resident_learning_enabled` is true. Candidate export/intake is bounded and
  generation-CAS checked; canonical merge stays disabled.
- **Memory** — additive mathematical memory with evidence assessments,
  cross-resolution relations, and merge support.
- **NANNET** — one encrypted installation identity, a pinned issuer key,
  short-lived scoped tokens, signed requests, and fail-closed authorization.
- **Permanent session context** — the NANNET architecture, NANNET routing,
  WordNet HW4:0 and free-chat learning documents ship inside the package. They
  are loaded once at plugin registration, hashed, persisted as a
  plugin-duration session manifest, and injected into attached-provider chat,
  reflection, curriculum, CLOCKED and synthesis calls. A user system prompt
  cannot remove them.
- **Local-first learning** — when no NANNET authority is configured, local
  chat/training tools run in explicitly reported `local-unverified` mode. Once
  an authority URL and pinned issuer key are configured, a paired token is
  mandatory and every protected operation fails closed. Remote router, lease,
  merge and device operations always require NANNET.
- **Universal native MTH runtime** — a compatibility snapshot of all 91 MTH
  NaN16 kernels and fixtures ships with the plugin, together with a portable
  NaN0S C-core source snapshot. The canonical native build and topology remain
  in NaNLMind; this bundle is not a second authority. On
  first use the plugin autodetects the host architecture/compiler, uses a
  configured external NaN0S CLI or compiles the bundled core into its state
  cache, assembles kernels with the bundled POSIX-awk emitter, and runs bounded
  native fixtures. The four admitted float32 vector kernels (dot, cosine, add,
  sub) accept real 256-element operands. The N16I interpreter-in-NaN16
  differential self-test is available, and `nanlm_mth_dispatch` relocates and
  runs arbitrary assembled guest images under N16I. That is generic dispatch,
  not a typed operand ABI for every kernel.
- **Hermes secrets** — NANNET, CLOCKED, and head-cortex credentials are read
  through `agent.secret_scope.get_secret()`.

**BS0:0 is the shared NANNET runner-ring port**; `HW0:0` is the separate hardware root. HW1 is Android, HW2 is HERMES, HW3 is CLOCKED, HW4 is the **permanently reserved WordNet universe**, and HW5–HW255 are allocatable hardware-node slots within BS0. The six runner IDs are internal to BS0, not separate HW allocations. HERMES guest universes are independently leased and isolated; the canonical training ring under BS0 remains shared. The remote universe-lease issuance path is not yet live (see `docs/HERMES_NANNET_LEASE_UNIVERSES.md`).

The plugin does not ship a model, download code, or run a background service.
The optional `cryptography` extra supplies the audited crypto backend; a
dependency-free fallback keeps ordinary chat and training importable on
Termux/Alpine.

## Quick start

```sh
# 1. Install the plugin from its public repository
hermes plugins install shemshallah/NaNLMind_HERMES --yes-deps --enable

# 2. Validate the manifest and registration
hermes plugins doctor /path/to/NaNLMind_HERMES --ci

# 3. Configure the plugin in Hermes Settings → Plugins → NaNLMind Hermes:
#    - state_dir
#    - chat_mode: native (the default; no LLM required)
#    - runner_command: ["/absolute/path/to/NaNLMind/build/nanlm-universal"]
#    - nannet_auth_url and nannet_issuer_public_key
#    - NANNET_MASTER_KEY in the masked secret field
#    - clocked_gateway_url + CLOCKED_GATEWAY_TOKEN for CLOCKED
#    - nanlm_head_cortex_url + NANLM_DEVICE_TOKEN for head-cortex calls

# 4. Create the local identity, then pair with an operator-issued code
#    (run these as Hermes tools)
nanlm_auth_init
nanlm_auth_pair  # pairing_code: "nnh_..."

# 5. Use it: teach a relation, then ask for it back
/nanlm A river is water.
/nanlm What is a river?
/nanlm call start A river is water.
/nanlm call @nanlm What is a river?
/nanlm call run 3
/nanlm logic Boolean inverses and nested implications
/nanlm words common words and next-word choice
/nanlm sentences short stories and sentence continuations
/nanlm pipeline full logic -> language curriculum
/nanlm status
```

For one-shot or CI runs, set `HERMES_GATEWAY_NO_SUPERVISE=1`. Catalog installs
will use `hermes plugins install nanlmind-hermes` after admission.

## Demo path

A complete local demo needs no NANNET or CLOCKED account:

```sh
# Build the native learner and verify the canonical contract once
git clone https://github.com/shemshallah/NaNLMind.git
make -C NaNLMind -j2 nanlm-universal
make -C NaNLMind unified-check

# Configure runner_command to NaNLMind/build/nanlm-universal,
# and nanlmind_root (or NANLMIND_ROOT) to the NaNLMind checkout, then:
/nanlm logic Boolean foundations
/nanlm status
```

With NANNET configured, pair once and repeat the same commands; protected
operations verify the token locally and request online scope authorization.
With CLOCKED configured, the approved plan → execute → scratch → head-process
flow is available through `/nanlm clocked ...`. See the
[three-scenario demo guide](docs/DEMO.md) and the
[CLOCKED reference](docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md).

## Configuration

Settings live under `plugins.entries.nanlmind-hermes.settings`.

| key | type | default | purpose |
|---|---|---|---|
| `state_dir` | str | `~/.local/share/nanlmind-hermes` | one canonical universe, transcripts, memory, scratch, runs |
| `runner_command` | list[str] | `[]` | absolute argv for the local `nanlm-universal` binary |
| `nanlmind_root` | str | `""` | optional NaNLMind checkout root for the canonical `spec/unified-build.v1.json`; `NANLMIND_ROOT` is used when unset |
| `duration_minutes` | int | `30` | wall budget for native bootstrap runs, 1–360 |
| `chat_learning` | bool | `true` | feed every chat turn into L0-L9 layered scratch and, when explicit supported/refuted triples exist, the native relation path |
| `chat_mode` | str | `native` | `native` (NaNLMind parses/synthesizes), `auto` (use provider if present), or `provider` (LLM-authored reply) |
| `wordnet_archive` | str | `""` | optional WordNet 3.1 zip for the sparse HW4:0 dictionary universe; when absent, HW4 ingest fails closed instead of inventing a word |
| `nan0s_root` | str | `""` | optional NaN0S checkout; auto-detected sibling/prebuilt or the bundled portable core is used |
| `nan0s_cc` | str | `""` | optional C compiler for the portable NaN0S core |
| `nan0s_prefer_bundled` | bool | `false` | prefer the bundled portable core over an external NaN0S checkout |
| `merge_enabled` | bool | `false` | explicit promotion/merge option; false returns `merge_disabled` and never promotes or merges until the generation-checked gate exists |
| `nannet_auth_url` | str | `""` | HTTPS NANNET authority (HTTP only for loopback) |
| `nannet_issuer_public_key` | str | `""` | pinned base64url Ed25519 issuer key |
| `clocked_gateway_url` | str | `""` | HTTPS CLOCKED participant gateway |
| `nanlm_head_cortex_url` | str | `""` | HTTPS head-cortex receiving endpoint |

Secrets are declared in `plugin.yaml` and read only through Hermes:
`NANNET_MASTER_KEY`, `CLOCKED_GATEWAY_TOKEN`, `NANLM_DEVICE_TOKEN`.

## Commands and tools

| command | purpose |
|---|---|
| `/nanlm <message>` | chat (default) |
| `/nanlm logic|words|sentences <topic>` | explicit native training session |
| `/nanlm pipeline <topic>` | ordered logic → language curriculum |
| `/nanlm clocked plan|run|scratch|head|learn|evaluate|synthesize` | approved CLOCKED flow |
| `/nanlm route <nan://...>` | signed NaN16DE route envelope |
| `/nanlm status` | state, memory, and native-run status |
| `/nanlm help` | usage |


The manifest and registered schemas are authoritative for all 86 tools.
[docs/TOOLS_REFERENCE.md](docs/TOOLS_REFERENCE.md) documents the main surfaces
and lists the remaining registered tools with one-line purposes. The permanent
session manifest is inspectable with `nanlm_session_context` and
`nanlm_status`; no paired NANNET identity is needed to read the shipped
architecture documents. Local chat/training without NANNET is reported as
`local-unverified`.


## Training

Native training requires `runner_command` pointing at a built
`NaNLMind/build/nanlm-universal`. The adapter runs the real native binary and
records stdout/stderr, corpus SHA-256, metrics, and a reproducibility manifest.
The current native Context2Model bootstrap is in-memory; the manifest is not a
resumable model checkpoint.

Structured sessions begin with the mandatory Boolean foundation gate: 256 heads
× 4 truth rows, 16 distinct learned rules, composed gates, and a persisted
`boolean_foundation.nmg` checkpoint. A failed or undersized gate stops the run.
Language-stage metrics are bounded byte/word task measurements, not a claim of
general language understanding.

See [docs/TRAINING_DURATIONS.md](docs/TRAINING_DURATIONS.md) for budgets and
[docs/WALKTHROUGHS.md](docs/WALKTHROUGHS.md) for complete session examples.

## CLOCKED

Configure `clocked_gateway_url` and the `CLOCKED_GATEWAY_TOKEN` secret. A plan
is generated without transmitting source text; execution requires the exact
`plan_sha256` and an explicit `approved: true`. Results are typed observations,
not ground truth, and are stored in the revisioned scratch journal. Candidate
learning remains isolated and cannot promote or mutate canonical head state.

The token must be a scoped participant credential (`clu_...`), not the
privileged upstream CLOCKED API key. See
[docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md](docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md).

## Catalog security and human approvals

The local native `nanlm-universal` process runs with an explicit, minimal
environment (`PATH`, `HOME`, `LANG` and, when set, `LC_ALL` and
`TMPDIR`). It does **not** inherit Hermes provider keys, profile credentials,
NANNET/CLOCKED tokens, arbitrary `NANLM_*` variables or unrelated shell state.

Model-issued tool calls to `nanlm_clocked_execute`,
`nanlm_head_candidate`, `nanlm_head_learn_candidate` and
`nanlm_head_candidate_evaluate` invoke Hermes's `pre_tool_call` **human
approval gate**; a model-controlled `approved: true` input is not sufficient.
The CLOCKED prompt names the saved plan's SHA-256 and endpoints, without
showing source contents; Hermes controls denial/timeout and unattended
execution. User-typed `/nanlm clocked ... --approved` is a separate explicit
command path, not a model-initiated tool bypass.

**Production recommendation:** install the reviewed optional
`cryptography` backend, for example with
`python -m pip install -e '.[auth]'` from the checked-out plugin source.
The dependency-free Ed25519/AES compatibility implementation is primarily
for constrained test environments and is **not audited for constant-time
operation**. Do not rely on it for high-value credentials.

This plugin's default state directory remains
`~/.local/share/nanlmind-hermes`, not intrinsically profile-isolated.
Configure a unique absolute `state_dir` for each Hermes profile to prevent
cross-profile universe and credential-state sharing. For the model-visible
`nanlm_train`, `nanlm_math_train` and `nanlm_evaluate` tools, a selected
corpus path must resolve to a regular file under `state_dir/corpora`. External
paths, traversal and symlinks escaping that tree are rejected. Import user-
approved corpora there before calling those tools. See
[Catalog admission](docs/HERMES_CATALOG_ADMISSION.md).

## Security boundaries

- BS0:0 has one shared runner-ring NANNET port. Each HERMES guest universe must be explicitly leased and bound to its own NANNET identity; a hardware-node registration alone grants no model-state access. No implicit registration or environment credential reads.
- Endpoint URLs and the pinned issuer key are plugin configuration; secrets are
  Hermes-managed.
- Protected operations fail closed when pairing, scope authorization, or the
  authority is unavailable. Candidate learning and CLOCKED execution retain
  explicit approval gates.
- NANNET/CLOCKED/head-cortex endpoints receive only the data required for the
  requested operation; CLOCKED plans exclude raw source text until execution.

Report security issues privately as described in [SECURITY.md](SECURITY.md).

## Verification


- The full unit/integration suite passes locally; CI runs Python 3.10, 3.11, and 3.12.
- Package verification builds and inspects wheel + sdist archives: 0 missing,
  0 forbidden and 0 oversized members; includes all 91 MTH sources/fixtures,
  the 7 session documents, the NaN0S portable core and both `nan/*.nan` and
  `nan/*.nas` payloads. See `scripts/verify_package.py`.
- `nanlm_mth_tree action=audit` links all **91/91** MTH kernels across 44 family
  hubs with 0 missing/orphans/digest mismatches and 91/91 NaN-executable leaf
  images verified. Alpine native-MTH tests use `build-base` + `gawk`.
- The initial Gemma tool-learning session ran over a **local loopback NANNET
  authority** with a real paired installation (`universe:read`,
  `training:execute`) and canonical merge disabled: **12/12 tool calls
  succeeded, 9 distinct tools used**, including tree audit, vector leaf
  creation, connect/add pairing, leaf verification, two free-chat turns and
  learning-status inspection. See
  [docs/GEMMA_TOOL_LEARNING_SESSION.md](docs/GEMMA_TOOL_LEARNING_SESSION.md).
- Python unit/integration tests run in CI on Python 3.10, 3.11 and 3.12;
  verify current results from the release commit, not historical test counts.

- Hermes Agent v0.23.0 `plugins validate` and `plugins doctor` pass in CI.
- Wheel and source distribution builds, metadata checks, installation/import,
  and SHA-256 checksums pass in CI.
- The phone-call routing kernel is raw NaN16, verified word-for-word against
  the native nan0s C VM core across an eight-case routing matrix; see
  [docs/PHONE_CALL_NAN16.md](docs/PHONE_CALL_NAN16.md).
- The Round 042 trigonometric kernel is 513 pure raw NaN16 words; its triangle
  and closed-loop gauge invariance, Q15 sin/cos readout and pure-image format
  are verified by `tests/test_trig_cycle_nan.py`. The host TRIGON-9 reference
  model is a separate experimental harness, not a native full-model claim; see
  [docs/NANLM_42_TRIGON.md](docs/NANLM_42_TRIGON.md).
- Hermes boots a local-CPU runner over an in-memory 8 MiB chart with real
  `s+`/`s-` pair-cell rails and a mapped temporal register (genesis chart
  before training; rebinds to the trained chart). It links through `HW2:0`
  (NaN16DE) to `BS0:0::HW0:0` without using the RunnerRing for compute; see
  [docs/LOCAL_RUNNER.md](docs/LOCAL_RUNNER.md).

- The historical exhaustive demo passed 38/38 calls across its then-registered
  34 tools. New session-context, free-chat learning, MTH, WordNet and evidence
  tool tests run separately; no claim is made that the historical demo covered
  all 86 current tools.

- A genuine held-out session asked unseen questions before and after new
  exposure, and a "No, that's incorrect" feedback turn triggered a fresh
  deliberation with the rejected answer suppressed; see
  [docs/LONG_CHAT_DEMO.md](docs/LONG_CHAT_DEMO.md).
- A separate transcript demo ran native chat without an LLM provider, the
  three-way shared-scratch call with Gemma as the Hermes bridge and CLOCKED
  evidence inside the call, native
  training, live CLOCKED, and the live head-cortex service. Live smoke passed 11/11 checks
  against the single deployed `nannet.apps.mistral.run` service.
- Exact commits, commands, and evidence are recorded in
  [docs/HERMES_CATALOG_ADMISSION.md](docs/HERMES_CATALOG_ADMISSION.md).

## Documentation

| document | contents |
|---|---|
| [Demo guide](docs/DEMO.md) | offline, NANNET, and CLOCKED scenarios with success signals |
| [Demo transcript](docs/HERMES_DEMO.md) | native chat, three-way call, training, CLOCKED, and live head-cortex output |
| [Phone-call kernel](docs/PHONE_CALL_NAN16.md) | raw NaN16 routing kernel, memory map, routing matrix, C-core parity |
| [TRIGON-9 synthesis](docs/NANLM_42_TRIGON.md) | Round 030–041 synthesis, L0–L9 trig rotors, 32,385 cycles, 256 heads, NANNET-P, HW4 matrix, raw NaN16 kernel, measured limits |
| [Local runner](docs/LOCAL_RUNNER.md) | boot-time 8 MiB chart, mapped temporal register, real rails, HW2:0 link |
| [MTH layers](docs/MTH_LAYERS.md) | root 4x4 Boolean table, L0–L9 curriculum, truth/frame scoring, concept matrix |
| [Free chat baseline](docs/FREE_CHAT.md) | cold direct address, self-contemplation, unknown questions |
| [Universal MTH runtime](docs/MTH_UNIVERSAL_RUNTIME.md) | bundled 91-kernel catalog, architecture autodetect, portable NaN0S core, N16I self-test and arbitrary guest dispatch |
| [MTH data leaves](docs/MTH_DATA_LEAVES.md) | immutable NaN-executable leaves, vector/WordNet hubs, constrained `left::right` pairing and self-modification guards |
| [MTH function tree map](docs/MTH_FUNCTION_TREE.md) | all 91 kernels mapped to family hubs with audit and executable-image evidence |
| [Gemma tool-learning session](docs/GEMMA_TOOL_LEARNING_SESSION.md) | live NANNET-paired open-ended tool-use test with merge disabled |
| [Gemma initial training report](docs/GEMMA_INITIAL_TRAINING_REPORT.md) | aggregated live results, system-state deltas, failure recovery and boundaries |
| [Gemma training series](docs/GEMMA_TRAINING_SERIES.md) | multi-session before/after quiz, tool-use metrics and system-learning deltas |
| [Gemma training series results](docs/GEMMA_TRAINING_SERIES_RESULTS.md) | journaled 3-session result: 15/15 calls, 100% coverage, quiz 76.1% → 100.0% (+23.9), system-state growth |
| [Gemma journal quiz probe](docs/GEMMA_JOURNAL_QUIZ_PROBE.md) | isolates the zero-delta bug: same quiz goes 76.1% → 100% when observed session journal is included |
| [Gemma training diagnostic](docs/GEMMA_TRAINING_DIAGNOSTIC.md) | why the original after-quiz was flat and the corrected journal/discovery design |
| [N16I unification](docs/N16I_UNIFICATION.md) | four vector kernels share the N16I general-operand path; native direct is diagnostic only |
| [NANNET HW2 production](docs/NANNET_HW2_PRODUCTION.md) | operator credential/pairing/signing flow for the permanent HW2/node-point-3 identity |
| [Production readiness TODO](docs/PRODUCTION_READINESS_TODO.md) | 106 point-by-point P0/P1/P2 production and integration gates |
| [Quantum/math integration matrix](docs/QUANTUM_MATH_INTEGRATION_MATRIX.md) | 25-row map of gauge, cycle, phase, spectral, pair-quantum, MTH, WordNet and learning wiring |
| [Quantum bridge](docs/QUANTUM_BRIDGE.md) | deterministic gauge/cycle/phase/pair-spectral/robust-recovery bridge with bounded input |
| [NaNLM learning over time](docs/NANLM_GEMMA_LEARNING_OVER_TIME.md) | Gemma-teacher/native-student held-out probe curve and exact hit vectors |
| [NaNLM novel generation](docs/NANLM_NOVEL_GENERATION.md) | post-training explanation/comparison/counterfactual/story probe with judge |
| [NaNLM accuracy audit](docs/NANLM_ACCURACY_AUDIT.md) | exact per-probe flips; why the denominator-14 decimals look like digit rotations |
| [NaNLM plateau diagnostic](docs/NANLM_PLATEAU_DIAGNOSTIC.md) | per-probe cause categories and curriculum/code ablation plan |
| [Probe immutability](docs/PROBE_IMMUTABILITY.md) | read-only probe fingerprint contract and state-mutation tests |
| [Resident universe](docs/RESIDENT_UNIVERSE.md) | what is actually in RAM today, proposed resident typed universe and primitive/adaptor boundaries |
| [PR integration audit](docs/PR_INTEGRATION_AUDIT.md) | all 58 PRs checked against the unified `main` for ancestry, file retention, feature retention, tests and closed-PR supersession |
| [Architecture gaps](docs/ARCHITECTURE_GAPS.md) | what the Alpine/Gemma test round found and what remains |
| [Gemma bootstrap](docs/GEMMA_BOOTSTRAP.md) | teacher-designed exposure, abstention and known/unknown probes |
| [Saturation session](docs/GEMMA_CORPUS.md) | concept-matrix saturation and organization by chatting |
| [Binary bootstrap](docs/BINARY_BOOTSTRAP.md) | bits → Boolean functions → bytes → letters → words → vectors |
| [Word completion](docs/WORD_COMPLETION.md) | WordNet lemmas, head-population collapse, dictionary in superposition |
| [Spectral architecture](docs/SPECTRAL_ARCHITECTURE.md) | wave/void operator, L0–L9 mapping, Hermes demo, copy path |
| [Head population](docs/HEAD_POPULATION.md) | routing sparsity, void head, rank/specialization diagnostics |
| [Boolean→wave](docs/BOOLEAN_WAVE.md) | multilinear Boolean embedding, atomic word-leaf update, payload packing |
| [Spectral word write](docs/SPECTRAL_WORD.md) | native-J Fiedler cut, word leaf write, eigen/mirror/void measurements |
| [Temporal register](docs/LOCAL_RUNNER.md) | true rank mirror, B/F local update, mirror convergence |
| [Concept spectral](docs/CONCEPT_SPECTRAL.md) | concept-graph matrix, multi-head mapping, normalized cut |
| [Cycle space](docs/CYCLE_SPACE.md) | gradient/cycle split, 32,385 fundamental fluxes, nonuniform mirror resonance |
| [Gauge memory](docs/GAUGE_MEMORY.md) | gauge-aware cycle retrieval, strong baseline, erasure robustness |
| [WordNet graph](docs/WORDNET_GRAPH.md) | real synset/lemma/hypernym graph, definition inheritance, HW4:0 interface |
| [WordNet topology](docs/WORDNET_TOPOLOGY.md) | all-POS cycle-rank audit, sparse WordNet vs dense NaN substrate |
| [WordNet directions](docs/WORDNET_DIRECTED.md) | directional replication audit, typed operators, next cycle-loss test |
| [Typed WordNet](docs/WORDNET_TYPED.md) | 378,203 typed edges, glosses, 26 relation types |
| [Definition acquisition](docs/DEFINITION_ACQUISITION.md) | new word leaf, inherited properties, honest boundaries |
| [Phase memory](docs/PHASE_MEMORY.md) | gauge-invariant phase retrieval, real-channel equivalence, noise limits |
| [Rounds 010–011](docs/ROUND010_011.md) | quantum controls, directional replication, phase-memory and NaN-palindrome limits |
| [Robust recovery](docs/ROBUST_RECOVERY.md) | two-hop median, gauge-flat 40% recovery, non-flat failure |
| [WordNet ancestry](docs/WORDNET_ANCESTRY.md) | 2,000/2,000 exact multi-hop vs 50% direct baseline |
| [Cycle codec](docs/CYCLE_CODEC.md) | gauge-corrected cycle code, joint robust decoder, random-code control |
| [Rounds 012–015 → MTH](docs/ROUND012_015_MTH.md) | robust recovery, blind semantic code, relation limits, MTH mapping |
| [Pair-space spectrum](docs/PAIR_QUANTUM.md) | exact J(256,2) spectrum, revival, cycle stationarity, ideal Grover control |
| [Round 016 → MTH](docs/ROUND016_MTH.md) | three-level warning, cycle-preserving operators, no semantic quantum oracle |
| [Definition QA](docs/DEFINITION_QA.md) | definition retrieval, exact ancestry, one/two-hop new-word probe |
| [Semantic cycle D](docs/WORDNET_SEMANTIC_CYCLE.md) | typed WordNet quotient, learned weights, null cycle/random/linear controls |
| [Round 018](docs/ROUND018.md) | tied recurrent controls, persistent glossary, null semantic-cycle result |
| [Round 019](docs/ROUND019.md) | controlled definition acquisition, abstention, exact Hodge/cycle audit |
| [Corpus memory](docs/CORPUS_MEMORY.md) | deterministic text ingestion, dedup, hash splits, source retrieval |
| [Joint decoder](docs/JOINT_SEMANTIC_DECODER.md) | novel-term yes/no state decoder, ordinary/cycle/no-state controls |
| [Round 020](docs/ROUND020.md) | corpus training tie, 48-channel topology, next joint-state experiment |
| [Long chat session](docs/LONG_CHAT_DEMO.md) | exposure, temporal predictions, held-out session, feedback re-deliberation |
| [Demo results](docs/DEMO_RESULTS.md) | 38/38 exhaustive tool-call results |
| [Live smoke results](docs/LIVE_SMOKE.md) | 11/11 deployed-service checks |
| [Install and validate](docs/HERMES_PLUGIN_INSTALL.md) | install paths, doctor output, upgrade/uninstall |
| [Tools reference](docs/TOOLS_REFERENCE.md) | every tool, schema, scope, and approval gate |
| [Walkthroughs](docs/WALKTHROUGHS.md) | chat, training, memory, CLOCKED, and recovery |
| [Topology and layers](docs/UNIFIED_TOPOLOGY_AND_LAYERS.md) | HW0::BS0, L0–L9, M0–M9 contracts |
| [Unified build adapter](docs/UNIFIED_BUILD_ADAPTER.md) | NaNLMind canonical contract, native verification, HERMES read-only boundary |
| [CLOCKED integration](docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md) | plans, scratch, head process, candidates |
| [NANNET API](docs/NANNET_API_HERMES.md) | identity, pairing, signed requests, scopes |
| [Plugin architecture](docs/PLUGIN_ARCHITECTURE.md) | components, boundaries, invariants |
| [Catalog admission](docs/HERMES_CATALOG_ADMISSION.md) | requirements, evidence, submission gates |
| [Alpine/Termux](docs/ALPINE_PROOT_TESTING.md) | constrained-host testing |

## License

Apache License 2.0. See [LICENSE](LICENSE).
