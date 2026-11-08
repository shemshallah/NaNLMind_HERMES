# NaNLMind Hermes

A Hermes Agent plugin for a local NaNLMind 256-head universe: conversational
learning, explicit native training, durable mathematical memory, and approved
CLOCKED evidence plans. Training and inference stay local; NANNET is the
identity/authorization boundary and CLOCKED is an evidence service.

**Version 1.3.2 — catalog submission pending.**

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
- **CLOCKED** — versioned, scoped task plans for extract, grounding, compare,
  privacy, injection, and synthesis. Multi-source results are written to an
  append-only scratch journal; source text is not sent during planning, and
  execution requires explicit approval.
- **Memory** — additive mathematical memory with evidence assessments,
  cross-resolution relations, and merge support.
- **NANNET** — one encrypted installation identity, a pinned issuer key,
  short-lived scoped tokens, signed requests, and fail-closed authorization.
- **Hermes secrets** — NANNET, CLOCKED, and head-cortex credentials are read
  through `agent.secret_scope.get_secret()`.

The topology is one canonical `HW0::BS0` universe: HW0 is RunnerLink, BS0 owns
the six-worker RunnerRing, HW1 is the phone endpoint, HW2 is Hermes, and HW3 is
CLOCKED. HW4 and above are allocatable participant addresses attached to that
same universe; they are not separate model instances.

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
# Build the native learner once
git clone https://github.com/shemshallah/NaNLMind.git
make -C NaNLMind -j2 nanlm-universal

# Configure runner_command to NaNLMind/build/nanlm-universal, then:
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
| `duration_minutes` | int | `30` | wall budget for native bootstrap runs, 1–360 |
| `chat_learning` | bool | `true` | allow explicit supervised claims to update native heads |
| `chat_mode` | str | `native` | `native` (NaNLMind parses/synthesizes), `auto` (use provider if present), or `provider` (LLM-authored reply) |
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

All 34 registered tools, their schemas, and their approval/scope requirements
are listed in [docs/TOOLS_REFERENCE.md](docs/TOOLS_REFERENCE.md).

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

## Security boundaries

- One installation identity, one `HW0::BS0` universe. No per-endpoint model
  universes, implicit hardware registration, or environment credential reads.
- Endpoint URLs and the pinned issuer key are plugin configuration; secrets are
  Hermes-managed.
- Protected operations fail closed when pairing, scope authorization, or the
  authority is unavailable. Candidate learning and CLOCKED execution retain
  explicit approval gates.
- NANNET/CLOCKED/head-cortex endpoints receive only the data required for the
  requested operation; CLOCKED plans exclude raw source text until execution.

Report security issues privately as described in [SECURITY.md](SECURITY.md).

## Verification

- 230 unit/integration tests pass locally; CI runs Python 3.10, 3.11, and 3.12.
- Hermes Agent v0.23.0 `plugins validate` and `plugins doctor` pass in CI.
- Wheel and source distribution builds, metadata checks, installation/import,
  and SHA-256 checksums pass in CI.
- The phone-call routing kernel is raw NaN16, verified word-for-word against
  the native nan0s C VM core across an eight-case routing matrix; see
  [docs/PHONE_CALL_NAN16.md](docs/PHONE_CALL_NAN16.md).
- Hermes boots a local-CPU runner over an in-memory 8 MiB chart with real
  `s+`/`s-` pair-cell rails and a mapped temporal register (genesis chart
  before training; rebinds to the trained chart). It links through `HW2:0`
  (NaN16DE) to `BS0:0::HW0:0` without using the RunnerRing for compute; see
  [docs/LOCAL_RUNNER.md](docs/LOCAL_RUNNER.md).
- The exhaustive demo passed 38/38 calls across all 34 registered tools.
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
| [Local runner](docs/LOCAL_RUNNER.md) | boot-time 8 MiB chart, mapped temporal register, real rails, HW2:0 link |
| [MTH layers](docs/MTH_LAYERS.md) | root 4x4 Boolean table, L0–L9 curriculum, truth/frame scoring, concept matrix |
| [Free chat baseline](docs/FREE_CHAT.md) | cold direct address, self-contemplation, unknown questions |
| [Gemma bootstrap](docs/GEMMA_BOOTSTRAP.md) | teacher-designed exposure, abstention and known/unknown probes |
| [Saturation session](docs/GEMMA_CORPUS.md) | concept-matrix saturation and organization by chatting |
| [Binary bootstrap](docs/BINARY_BOOTSTRAP.md) | bits → Boolean functions → bytes → letters → words → vectors |
| [Word completion](docs/WORD_COMPLETION.md) | WordNet lemmas, head-population collapse, dictionary in superposition |
| [Spectral architecture](docs/SPECTRAL_ARCHITECTURE.md) | wave/void operator, L0–L9 mapping, Hermes demo, copy path |
| [Head population](docs/HEAD_POPULATION.md) | routing sparsity, void head, rank/specialization diagnostics |
| [Boolean→wave](docs/BOOLEAN_WAVE.md) | multilinear Boolean embedding, atomic word-leaf update, payload packing |
| [Spectral word write](docs/SPECTRAL_WORD.md) | native-J Fiedler cut, word leaf write, eigen/mirror/void measurements |
| [Temporal spine](docs/SPINE.md) | true rank mirror, B/F local update, mirror convergence |
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
| [Long chat session](docs/LONG_CHAT_DEMO.md) | exposure, temporal predictions, held-out session, feedback re-deliberation |
| [Demo results](docs/DEMO_RESULTS.md) | 38/38 exhaustive tool-call results |
| [Live smoke results](docs/LIVE_SMOKE.md) | 11/11 deployed-service checks |
| [Install and validate](docs/HERMES_PLUGIN_INSTALL.md) | install paths, doctor output, upgrade/uninstall |
| [Tools reference](docs/TOOLS_REFERENCE.md) | every tool, schema, scope, and approval gate |
| [Walkthroughs](docs/WALKTHROUGHS.md) | chat, training, memory, CLOCKED, and recovery |
| [Topology and layers](docs/UNIFIED_TOPOLOGY_AND_LAYERS.md) | HW0::BS0, L0–L9, M0–M9 contracts |
| [CLOCKED integration](docs/CLOCKED_AUTONOMOUS_HEAD_CORTEX.md) | plans, scratch, head process, candidates |
| [NANNET API](docs/NANNET_API_HERMES.md) | identity, pairing, signed requests, scopes |
| [Plugin architecture](docs/PLUGIN_ARCHITECTURE.md) | components, boundaries, invariants |
| [Catalog admission](docs/HERMES_CATALOG_ADMISSION.md) | requirements, evidence, submission gates |
| [Alpine/Termux](docs/ALPINE_PROOT_TESTING.md) | constrained-host testing |

## License

Apache License 2.0. See [LICENSE](LICENSE).
