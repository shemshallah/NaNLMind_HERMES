# NaNLMind Hermes

Hermes plugin for a local NaNLMind 256-head universe, with NANNET installation identity and authority-side scope checks.

## Training is local

The native NaNLMind binary executes on the machine hosting Hermes. Koyeb is the NANNET control plane (pairing, authorization, device leases and small delta merge); it does not receive the corpus or run the native training process.

Build the core repository and configure Hermes runner_command as an argv array containing the absolute path to build/nanlm-universal:

    ["/absolute/path/to/NaNLMind/build/nanlm-universal"]

Available tools:
- nanlm_status — local universe, runner configuration, native run and durable memory status.
- nanlm_generate_corpus — produce a content-addressed N16TXT/1 corpus.
- nanlm_train — invoke the native bootstrap CORPUS EPOCHS CHUNKS trainer with a monotonic deadline and exclusive local training lock.
- nanlm_evaluate — run topology-byte, Kneser-Ney order-2 and Context2 evaluations against the same plain-text corpus.
- nanlm_memory_ingest — append typed mathematical discoveries with source digest, polarity, confidence and provenance.
- nanlm_memory_query — inspect the active head/layer projection.
- nanlm_memory_reorganize — rebuild deterministic projections without mutating the discovery ledger.
- nanlm_memory_merge — idempotently union peer discovery events into the local ledger.
- nanlm_auth_init, nanlm_auth_pair, nanlm_auth_status, nanlm_auth_revoke — manage the local NANNET installation identity.

N16TXT/1 is validated and its framing header is removed before native CLI use. Native bootstrap requires at least 1,000 UTF-8 bytes; byte and baseline evaluations require at least 100 bytes. The harness never pads or duplicates a small corpus.

## What persists

Hermes now has an additive mathematical-memory ledger in `nanlmind_hermes/memory.py`. It stores content-addressed discovery events and typed graph edges in a local SQLite WAL database. Repeated event delivery is idempotent; contradictory hypotheses remain distinct; head/layer projections can be deterministically rebuilt. The event ledger is persistent memory, not conventional neural weights.

The current native bootstrap command still trains Context2Model in process memory and does not serialize its learned parameters. Hermes therefore reports native checkpoint absence honestly. The new memory interface accepts structured discoveries; automatic extraction from native training is not claimed until the core runner emits a stable structured trace/discovery stream.

The core topology defines **256** palindromic heads. Hermes defaults to all 256 and allows an explicit configured head count; it does not silently reinterpret the topology as 255.

See:
- docs/NANNET_API_HERMES.md
- docs/TRAINING_DURATIONS.md
- docs/HERMES_UNIVERSE_MEMORY.md
