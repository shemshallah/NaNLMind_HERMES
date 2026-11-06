# NaNLMind Hermes

## Quickstart for the impatient

```sh
# Install the plugin from its public repository
hermes plugins install shemshallah/NaNLMind_HERMES --yes-deps --enable

# Validate the manifest and registration
hermes plugins doctor /path/to/NaNLMind_HERMES --ci

# First: NANNET login is mandatory.
# Hermes uses HW_HERMES_API, or auto-registers one hwh_ key.
# Android test nodes use NANLM_DEVICE_TOKEN + NANLM_INTAKE_URL instead.
# For one-shot/CI runs, export HERMES_GATEWAY_NO_SUPERVISE=1.

# Default: chat
/nanlm Hello. What can you do?

# Explicit training modes
/nanlm logic Boolean inverses and nested implications
/nanlm words common words and next-word choice
/nanlm sentences short stories and sentence continuations
/nanlm pipeline full logic -> language curriculum

# Inspect state
/nanlm status
```

## Hermes hardware key

Hermes uses a dedicated `HW_HERMES_API` hardware identity. Priority:

1. `HW_HERMES_API` from the host/system key store;
2. the persisted key in `<state_dir>/hw_hermes_api.json`;
3. automatic registration of exactly one `hwh_` key at
   `POST /v1/hermes/register`.

The key is stored mode `0600`, is separate from Android’s `NANLM_DEVICE_TOKEN`
test namespace, and is tagged `hardware_kind=hermes` at the portal so a later
router can send the identity to the correct kernel. If a system key is present,
Hermes does not register another one.

Slot policy:

```text
HW2  Hermes demo identity
HW3  Clocked demo identity (reserved; Hermes never claims it)
HW4+ lowest free Hermes registration after the demo slot is occupied
```

Two Hermes instances therefore use HW2 (the seeded demo key) and HW4; a third
uses HW5. Android test registration stays in its own namespace and does not
take HW2/HW3.

```sh
export NANNET_AUTH_URL=https://your-nannet-portal
export HW_HERMES_API=hwh_...        # optional; otherwise auto-register
export HERMES_GATEWAY_NO_SUPERVISE=1 # recommended for one-shot/CI runs
```

NANNET login is required for chat and training. Pair first with
`nanlm_auth_init` and `nanlm_auth_pair`, then set `NANNET_AUTH_URL` and the
pinned issuer key. Android-style nodes may provide a scoped
`NANLM_DEVICE_TOKEN` plus `NANLM_INTAKE_URL`; the portal validates the scope.
The plugin fails closed without one of these credentials.

Native training requires the local binary:

```yaml
plugins:
  entries:
    nanlmind-hermes:
      settings:
        runner_command: ["/absolute/path/to/NaNLMind/build/nanlm-universal"]
```

The plugin itself needs no Google API key. It uses the model provider configured
in Hermes (local Ollama, a custom endpoint, or any configured provider).

**Version 1.0.1 — released 2026-11-06**

NaNLMind Hermes connects a Hermes agent to a local NaNLMind 256-head universe.
It provides:

- a default open-ended chat loop with durable conversation history;
- online free-form relation learning from explicit supervised claims in back-and-forth chat;
- a mandatory full-Boolean foundation gate before structured training;
- explicit, ordered logic → words → sentences training sessions;
- a teacher-generated structured relation pipeline with frozen held-out challenges at L0-L9;
- native head, canonical mirror-pair, temporal-link and hyperedge updates in one resumable topology;
- separate pre/post and cross-run metrics for each evaluated resolution;
- additive mathematical memory with evidence assessment and cross-resolution feedback;
- local NANNET identity, pairing, signed requests, and optional authority
  scopes;
- a dependency-free Ed25519/AES-GCM/HKDF path for Alpine, Termux, and minimal
  hosts.

Training runs locally. Koyeb/NANNET is coordination and authorization only;
it is not the model runner or corpus store.

---

## 1. Default behaviour: chat

Plain conversation is the default. The model should use `nanlm_chat` for normal
interaction. With `chat_learning: true` (default), each exchange is reflected
and explicit supported/refuted subject-predicate-object relations are submitted
to the native relation learner. This is a bounded online update, not an automatic
curriculum-generation job. Set `chat_learning: false` to disable native chat
updates while retaining transcripts and symbolic reflection.

Inside a Hermes session:

```text
/nanlm Hello, explain what you remember about penguins.
/nanlm chat Continue the previous discussion.
```

Bare text after `/nanlm` is treated as a chat message:

```text
/nanlm What did we train last time?
```

Chat records:

- a durable JSONL transcript per session;
- provider/model metadata;
- exact UTF-8 bit, nibble, byte, lexical, sentence, and paragraph measurements;
- a structured reflection proposal merged into additive mathematical memory;
- native-learning receipt: accepted relation events, changed heads, mirror-pair
  mutations, hyperedges, checkpoint digests and per-resolution metrics when a
  held-out set is available;
- optional cross-resolution relations.

Reflection is best effort and never blocks the conversational answer. Only
explicit, non-neutral triples with evidence update native state. Neutral or
incomplete reflections remain symbolic evidence; the plugin never claims a
head update unless the native runner confirms it.

---

## 2. Installation

### From the Hermes plugin service / Git repository

```sh
hermes plugins install shemshallah/NaNLMind_HERMES --yes-deps --enable
hermes plugins list
hermes tools
```

Installing from a repository is available immediately. Catalog review and
listing are a separate submission process.

### Local development

```sh
git clone https://github.com/shemshallah/NaNLMind_HERMES.git
hermes plugins install /absolute/path/NaNLMind_HERMES --enable
hermes plugins validate /absolute/path/NaNLMind_HERMES --install-deps
hermes plugins doctor /absolute/path/NaNLMind_HERMES --ci
```

Expected validation:

```text
✓ manifest
✓ loadable
✓ capability probe
✓ declared tools
✓ declared hooks
✓ built-in tool collisions
✓ security scan
✓ no core override
Validation passed.
```

### Minimal Alpine / Termux

The plugin has no required Python dependencies. It uses `cryptography` when
present and otherwise falls back to pure-Python Ed25519, AES-256-GCM, HKDF, and
RSA PKCS#1 verification. See `docs/ALPINE_PROOT_TESTING.md`.

For the audited/accelerated backend:

```sh
pip install "nanlmind-hermes[auth]"
```

The fallback is appropriate for minimal local hosts; use the optional backend
or an HSM for production key custody.

---

## 3. Configuration

Settings live under `plugins.entries.nanlmind-hermes.settings`.

| key | type | default | meaning |
|---|---|---|---|
| `duration_minutes` | int | `30` | wall budget for legacy native bootstrap runs, 1–360 |
| `checkpoint_interval_seconds` | int | `60` | reserved for native checkpoint support |
| `runner_command` | list[str] | `[]` | absolute argv for the local `nanlm-universal` binary |
| `chat_learning` | bool | `true` | allow free-form chat to update native heads/mirror pairs/hyperedges from explicit supervised relations |
| `state_dir` | str | `~/.local/share/nanlmind-hermes` | universe, corpora, transcripts, runs |
| `universe_layers` | int | `7` | logical memory projection layers L0–L9 |

Example:

```yaml
plugins:
  entries:
    nanlmind-hermes:
      enabled: true
      settings:
        state_dir: ~/.local/share/nanlmind-hermes
        universe_layers: 7
        runner_command: ["/opt/NaNLMind/build/nanlm-universal"]
```

The native runner is required for structured training and native chat learning.
Without it, chat still answers and stores reflection evidence, but its
`native_learning` receipt reports that no native update occurred.

---

## 4. Training modes

Training is deliberately command-driven.

### Full pipeline

```text
/nanlm pipeline English logic, common words, and short stories
```

or call `nanlm_training_pipeline`.

The canonical 12-stage order is:

```text
boolean_inverses
deep_logic
story_saturation
exceptions
repetitive_words
phonetic_pairs
morpheme_structure
lexical_words
word_completion
sentence_choice
paragraph_structure
narrative
```

### Logic-only session

```text
/nanlm logic Boolean truth tables and nested implications
```

Uses:

```text
boolean_inverses + deep_logic + exceptions (L0-L2)
```

Tool: `nanlm_logic_session`.

### Word-only session

```text
/nanlm words common words, repetition, and next-word choice
```

Uses:

```text
repetitive_words + phonetic_pairs + morpheme_structure + lexical_words + word_completion (L3-L6)
```

Tool: `nanlm_word_session`.

### Sentence/story session

```text
/nanlm sentences sentence continuations and short narratives
```

Uses:

```text
sentence_choice + paragraph_structure + narrative (L7-L9)
```

Tool: `nanlm_sentence_session`.

### Custom stage order

`nanlm_training_pipeline` accepts an ordered `stages` array:

```json
{
  "topic": "logic and language",
  "stages": ["boolean_inverses", "deep_logic", "word_completion", "sentence_choice"],
  "records_per_stage": 12,
  "epochs": 3,
  "max_samples_per_head": 4096
}
```

Structured training generates N16REL/1 relation records, preserves a frozen
challenge set per topic/resolution, and merges only training evidence into
mathematical memory. Each resolution reports before/after held-out metrics and
a cross-run delta only when its challenge digest is unchanged.

### What native training actually learns

Before structured relation training, Hermes runs the mandatory native
full-Boolean foundation: all 1,024 truth-row checks, all 16 distinct learned
functions, composed circuits and a persisted checkpoint. That checkpoint seeds
the shared `native/global_universe.nmg` once.

The new `relation-train` objective is separate from the legacy byte-bit
`math-train` control. It updates routed heads, canonical mirror-pair cells,
temporal links and provenance-bound hyperedges from typed L0-L9 relations.
Free-form chat does the same online for explicit supported/refuted triples.
The relation trainer uses resolution-specific surface feature vectors; SHA-256 is used only for routing and event identity. It is a compact feature learner, not a pretrained semantic encoder. Neither relation-polarity accuracy nor byte-bit accuracy is text-generation accuracy or independent fact verification.

---

## 5. Commands

| command | behaviour |
|---|---|
| `/nanlm` or `/nanlm chat ...` | default chat |
| `/nanlm logic ...` | logic-only training |
| `/nanlm words ...` | word-only training |
| `/nanlm sentences ...` | sentence/story training |
| `/nanlm pipeline ...` | full staged pipeline |
| `/nanlm status` | universe/native/memory status |
| `/nanlm help` | command help |

The same actions are available as model tools. See
`docs/TOOLS_REFERENCE.md` for every parameter.

---

## 6. Memory model

There are two stores:

1. **Native global topology** — serialized head/cell/scratchpad state, canonical
   mirror pairs, temporal links and hyperedges at
   `universe/native/global_universe.nmg`, resumed by chat and training runs.
2. **Additive discovery ledger** — immutable, content-addressed events, typed
   edges, n-ary hyperedges, and deterministic L0–L9 projections.

A merge is an idempotent set union followed by projection rebuild. Conflicting
claims are retained and become contested; they are not averaged away. The
ledger contains evidence, not neural weights.

Cross-resolution analysis covers ten channels:

```text
bit, nibble, byte, phonetic, morpheme, lexical,
word_completion, sentence, paragraph, narrative
```

It compares aligned evidence, preserves contradictions, and emits auditable
relations. Synthetic controls validate the mechanism; they do not establish
general language understanding.

---

## 7. NANNET identity and pairing

The optional NANNET path provides:

- one local installation Ed25519 identity;
- a master-key-encrypted private-key envelope;
- one-time pairing codes;
- role-bounded scoped tokens;
- signed requests with replay-resistant nonces;
- revocation;
- device lease routes.

Commands/tools:

```text
nanlm_auth_init
nanlm_auth_status
nanlm_auth_pair
nanlm_auth_revoke
```

Chat and training through Hermes require NANNET login and fail closed when
authority credentials are missing or invalid. The test-only
`NANLM_AUTH_TEST_BYPASS=1` switch exists for the local test suite and must not be
used in a live installation. See `docs/NANNET_API_HERMES.md`.

---

## 8. Troubleshooting

| symptom | fix |
|---|---|
| plugin not listed | `hermes plugins list`; enable `nanlmind-hermes` |
| validation import error | ensure `__init__.py` exists; reinstall with `--force` |
| no native training | configure `runner_command` to the local binary |
| `nanlm-universal` not found | build NaNLMind and use an absolute path |
| chat unavailable | configure the Hermes provider with `hermes model` |
| authority errors | check `NANNET_AUTH_URL` and pinned issuer key |
| Alpine `fchdir` error | use `proot-distro login --minimal` |
| state not resumable | native math runner writes the serialized topology; legacy bootstrap does not |

Diagnostics:

```sh
hermes plugins validate /path/to/NaNLMind_HERMES
hermes plugins doctor /path/to/NaNLMind_HERMES --ci
python -m unittest discover -s tests -v
```

---

## 9. Vision and larger system

See [HACKATHON_NOTE.md](HACKATHON_NOTE.md) for the architecture/manifesto: the
c(256,2) universe, layered learning, NANNET, merge model, distributed hierarchy,
and the limits of the experiment.

## 10. Repository map

| path | contents |
|---|---|
| `nanlmind_hermes/__init__.py` | tool registration, `/nanlm` command |
| `nanlmind_hermes/tools.py` | chat, corpus, training, memory, status |
| `nanlmind_hermes/chat.py` | conversational loop and multiresolution reflection |
| `nanlmind_hermes/runner.py` | local native process harness |
| `nanlmind_hermes/memory.py` | additive memory ledger and projections |
| `nanlmind_hermes/auth.py`, `pairing.py` | NANNET identity and signing |
| `nanlmind_hermes/crypto_compat.py` | dependency-free Ed25519/AES-GCM/HKDF |
| `docs/TOOLS_REFERENCE.md` | complete tool/option reference |
| `docs/WALKTHROUGHS.md` | end-to-end usage scenarios |
| `docs/PLUGIN_ARCHITECTURE.md` | design, boundaries, and invariants |
| `docs/NAN16DE.md` | raw NaN16 double-endian route protocol |
| `docs/BASELINE_1.0.1.md` | model-planned logic/words/sentences baseline results |
| `docs/TEACHER_LOOP.md` | teacher → validator → student → feedback protocol |

## 11. Versioning

- `1.0.1` — 2026-11-06: mandatory full-Boolean foundation gate, architecture-aware
  model planning, chat checkpoints after logic/words/sentences rounds, baseline
  metrics, and no-robust-learning conclusion.
- `1.0.0` — 2026-11-06: default chat loop, explicit training modes, full
  pipeline, 24 tools, `/nanlm` command, validation-ready manifest, dependency
  fallback, complete documentation.

License: MIT.
