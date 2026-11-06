# NaNLMind Hermes

Hermes plugin for a local NaNLMind 256-head universe, with NANNET installation identity, authority-side scope checks, native mathematical training, and additive discovery memory.

## Native mathematical trainer

Build the core repository and configure Hermes `runner_command` as an argv array containing the absolute path to `build/nanlm-universal`:

    ["/absolute/path/to/NaNLMind/build/nanlm-universal"]

Use `nanlm_math_train` for the new path. It invokes the native `math-train` command, trains the 256 palindromic heads in the shared global topology, evaluates a held-out corpus split, writes a resumable serialized global topology, validates the 256-record JSONL discovery trace, and merges the discoveries into the additive memory ledger.

This first native trainer is deliberately scoped: it learns per-head Boolean truth-table rules for byte-bit transition tasks. It is a real persistent native training path, but it is not yet a general symbolic mathematics solver, a language model, or the complete multi-layer self-reorganizing trainer. Its held-out accuracy is an evaluation metric, not proof of broad mathematical reasoning.

The state file includes the global topology and serialized universe/head scratchpads. A later run loads that exact state before training. Hermes keeps a separate append-only event ledger so extracted claims and contradictory evidence survive even when native state evolves.

## Other tools

- `nanlm_status` — local universe, latest native math run, checkpoint presence, and memory status.
- `nanlm_generate_corpus` — produce a content-addressed N16TXT/1 corpus.
- `nanlm_train` — run the legacy Context2 bootstrap trainer (its parameters remain in-memory).
- `nanlm_math_train` — run native 256-head training, resume the global topology, validate discoveries, merge, and reorganize.
- `nanlm_evaluate` — compare topology-byte, Kneser-Ney order-2 and Context2 controls.
- `nanlm_memory_ingest`, `nanlm_memory_query`, `nanlm_memory_reorganize`, `nanlm_memory_merge` — manage durable additive mathematical memory.
- `nanlm_auth_init`, `nanlm_auth_pair`, `nanlm_auth_status`, `nanlm_auth_revoke` — manage the local NANNET installation identity.

## What persists

The native global topology is serialized to `universe/native/global_universe.nmg` and resumed on subsequent math-training runs. Structured discoveries are stored separately in a SQLite WAL event ledger with content-addressed IDs, typed graph edges and n-ary hyperedges. Reorganization rebuilds deterministic head/layer projections without deleting evidence. Conflicting hypotheses remain independently auditable.

NANNET is the authorization and delta-coordination boundary. Training and corpus processing run locally; Koyeb is not the native trainer or corpus store.

See `docs/HERMES_UNIVERSE_MEMORY.md` for the memory model and current limitations.
