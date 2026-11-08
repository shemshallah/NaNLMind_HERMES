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
