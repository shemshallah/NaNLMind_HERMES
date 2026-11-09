# Read-only NaNROUTER L0–L9 graph projection verifier

`RouterGraphPageVerifier(chart)` checks incoming MC0DB graph projections
against the **actual resident 8 MiB chart**. It reads the existing
`N16ROOT1` descriptor and pins the parent root SHA-256, chart generation
and the router's independently reproduced 256-head / 32,640-MP topology
digest. There is no second graph, native head mutation, or new SQLite file.

Provide router `NanLayerGraph.master_graph_page(...)` outputs in ordered
pages starting at offset zero. Each page must have an identical graph
commitment, current chart root and generation, correct canonical routes and
head permissions, and a strictly ascending nonduplicated edge ID sequence.
Every candidate/tombstone record contributes to a streaming state SHA-256
check. Once all pages arrive, the whole-state digest and layer/status counts
must match the committed header. Missing pages, mixed versions and stale
epochs fail closed. A failed verifier cannot be reused.

**Trust boundary:** This validates payload consistency and chart identity
but does not authenticate the remote sender or independently replay the
history hash. `sender_authenticated` and `native_mth_executed` stay false.
Authenticated dispatch, evidence revalidation and MTH execution remain
separate gates before any model update or candidate promotion.

## Hermes plugin entry point

The protected `nanlm_router_graph_verify` tool requires `universe:read` scope
and accepts a complete ordered `pages` array (up to 128 pages per call).
It instantiates the verifier against the already mounted `LocalRunner`
chart and returns success only after the final whole-state digest matches.
Incomplete batches are rejected rather than treated as valid prefixes.
Large snapshots require a future authenticated chunked streaming interface;
this bounded tool performs validation only, never graph admission.

## Independent revision replay (new opt-in read-only verifier)

`RouterGraphHistoryVerifier(chart)` extends the existing streaming state
check with full **version history** verification. Feed the ordinary
`master_graph_page` state pages through `add_state_page` until state
verification completes. Then feed pages produced by
`master_graph_page(..., history_offset=N, history_limit=100)` through
`add_history_page`, starting at offset zero and following
`history_next_offset`. Both phases must share the same state/history
SHA-256 commitment, parent root, chart generation, and topology digest.

The replay independently hashes the complete ordered revision stream using
`nanrouter.nan-layer-history/1\\0`, validates canonical JSON and contiguous
per-edge generations, checks immutable endpoints/proposer and nondecreasing
ticks, enforces lossless tombstones and legal reactivation, and compares each
edge's last historical record to its previously verified present-state record.
Corrupt/mixed/missing pages and stale chart epochs fail closed. It keeps only
an ephemeral verification map for current edge records, releasing each once
its history is compared; this does **not** create another durable graph or
modify the resident chart.

The state-only `RouterGraphPageVerifier` and the existing
`nanlm_router_graph_verify` tool remain backward-compatible. By default
the tool still verifies state only and declares history unreplayed. Supply
a bounded nonempty `history_pages` array together with the existing
`pages` to invoke both phases, requiring *complete* state and history
before returning `history_verified=true`. Callers relying on history must
require that field explicitly; missing or truncated history does not qualify.

**Security boundary remains unchanged:** An entirely self-consistent forged
history is not thereby sender-authenticated. Neither the state verifier nor
the history replay validates immutable MathLeaf evidence from a trusted
source, establishes a native N16T/MTH execution receipt, authorizes a learning
transaction, or promotes a canonical NANNET model. All corresponding status
flags remain false until separate, verifiable gates exist.
