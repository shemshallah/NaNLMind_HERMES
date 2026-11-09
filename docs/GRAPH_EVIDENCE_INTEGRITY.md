# Independent historical evidence-integrity audit

After the signed graph and independent history replay, the next read-only trust boundary is checking that **every historical revision**, including tombstones and reactivations, still points to an immutable existing `MathLeaf` with exactly the committed ID, version and SHA-256 digest.

`audit_graph_revision_evidence(verifier, state_pages, history_pages, resolve_trusted_leaf=...)` independently replays the complete graph state and history against the resident 8 MiB HERMES chart before invoking a **trusted deployment-injected** read-only leaf resolver. It rejects missing/wrong leaves, changed digests, conflicting purported digests of one leaf version, tampered pages, incomplete pagination, and stale chart epoch. It deduplicates leaf lookups by immutable ID/version, while auditing each revision's evidence reference in the aggregate digest.

The resolver must originate from operator-controlled storage (e.g. the existing authorized NaNROUTER `MathStore.get_leaf`); it is **never accepted from peer packet data, a tool request or a sender-supplied public key**. The proof-of-functionality GitHub CI uses actual MC0DB/BS0DB versions and the real HERMES chart; it does not instantiate another model or duplicate graph database.

**What the audit does not prove:** A matching immutable leaf digest is content integrity, not independent origin trust, mathematical correctness, a native execution receipt, anti-replay protection or a learning decision. All these non-authorities remain explicitly false in its report. The module is an opt-in library primitive, not a network write endpoint, a new graph store, or a learning approval gate.

This audit is composable after the existing pinned Ed25519 graph sender verifier and before any future separately authorized native MTH evidence evaluation, but it does not inherit the sender authentication flag or mint new authority.