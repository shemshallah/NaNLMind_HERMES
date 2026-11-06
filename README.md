# NaNLMind Hermes

## Quickstart for the impatient

```sh
# Install the plugin from its public repository
hermes plugins install shemshallah/NaNLMind_HERMES --yes-deps --enable

# Validate the manifest and registration
hermes plugins doctor /path/to/NaNLMind_HERMES --ci

# First: NANNET login is mandatory
# Ask Hermes to run nanlm_auth_init, then complete nanlm_auth_pair.
# Android-style nodes may instead export NANLM_DEVICE_TOKEN + NANLM_INTAKE_URL.

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

**Version 1.0.0 — released 2026-11-06**

NaNLMind Hermes connects a Hermes agent to a local NaNLMind 256-head universe.
It provides:

- a default open-ended chat loop with durable conversation history;
- explicit, ordered logic → words → sentences training sessions;
- a full staged training pipeline;
- native NaNLMind head training with resumable global-topology state;
- additive mathematical memory with evidence assessment and cross-resolution
  feedback;
- local NANNET identity, pairing, signed requests, and optional authority
  scopes;
- a dependency-free Ed25519/AES-GCM/HKDF path for Alpine, Termux, and minimal
  hosts.

Training runs locally. Koyeb/NANNET is coordination and authorization only;
it is not the model runner or corpus store.

---

## 1. Default behaviour: chat

Plain conversation is the default. The model should use `nanlm_chat` for normal
interaction. Training tools are explicit actions, not side effects of chat.

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
- optional cross-resolution relations.

Reflection is best effort and never blocks the conversational answer. It
updates symbolic evidence, not model weights.

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
| `state_dir` | str | `~/.local/share/nanlmind-hermes` | universe, corpora, transcripts, runs |
| `universe_layers` | int | `7` | logical memory projection layers L0–L6 |

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

The native runner is required for `nanlm_math_train`, `nanlm_train`,
`nanlm_evaluate`, and the training pipeline. Chat and memory tools work without
it.

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
boolean_inverses + deep_logic
```

Tool: `nanlm_logic_session`.

### Word-only session

```text
/nanlm words common words, repetition, and next-word choice
```

Uses:

```text
repetitive_words + lexical_words + word_completion
```

Tool: `nanlm_word_session`.

### Sentence/story session

```text
/nanlm sentences sentence continuations and short narratives
```

Uses:

```text
sentence_choice + narrative
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

Every training tool returns a reproducible manifest and merges native head
discoveries into mathematical memory.

### What native training actually learns

`nanlm_math_train` trains the 256 native heads on a bounded byte-bit transition
task. It is real, persistent native training, but not a general language model.
Held-out accuracy measures that task only. Chat reflection is symbolic evidence,
not online gradient descent.

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

1. **Native global topology** — serialized head/cell/scratchpad state at
   `universe/native/global_universe.nmg`, resumed by later native runs.
2. **Additive discovery ledger** — immutable, content-addressed events, typed
   edges, n-ary hyperedges, and deterministic L0–L6 projections.

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

Standalone local chat/training does not require NANNET. Paired mode fails closed
when authority credentials are missing or invalid. See
`docs/NANNET_API_HERMES.md`.

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

## 11. Versioning

- `1.0.0` — 2026-11-06: default chat loop, explicit training modes, full
  pipeline, 24 tools, `/nanlm` command, validation-ready manifest, dependency
  fallback, complete documentation.

License: MIT.
