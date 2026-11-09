# Verified graph navigation (read-only)

`verified_candidate_walk(verifier, state_pages, start=..., target=..., max_hops=...)` is a bounded diagnostic for graph-based language learning research. It consumes the exact current-state page batch independently rehashed against the live HERMES 8 MiB chart, together with a complete verified NaNROUTER revision replay. It **never** builds another persistent graph, creates a new universe, changes chart/head memory, performs host-side math, or admits learning.

- Input nodes are canonical integers 0..32895: heads 0..255 and mirrored-pair cells 256..32895.
- Directed candidate edges are traversed by shortest-hop BFS, deterministically ordered by endpoints, L0–L9 layer, relation and stable edge ID. Non-candidate (retracted) claims are excluded from paths while remaining in the verified revision history.
- Optional L0–L9 layer and relation filters select only existing candidate statements. The output retains evidence IDs, versions, evidence SHA-256 and exact edge generations for independent follow-up validation.
- Simultaneous `supports` and `refutes` claims between the same ordered nodes/layer are **both preserved** as `contested_pairs`, never silently resolved.
- No transitivity, equivalence, mathematical entailment, graph-theorem proof or synthetic words are inferred from a path. A graph path is evidence-backed candidate connectivity only.

Maximum traversal: 8 hops, 1024 nodes and 12,800 matched candidate edges. Callers can set smaller limits; reaching a hard completeness bound fails closed. Replayed and query pages must match a single graph commitment/current chart generation, so post-verification mutation, truncation or mixed graph epochs are rejected. The method holds at most the already bounded request graph in transient memory.

The protected `nanlm_router_graph_walk` plugin tool exposes this **read-only** capability under `universe:read`. It requires ordered, complete `pages` and `history_pages`, a canonical integer `start` node, optional `target`, `layer`, `relation`, and traversal bounds (`max_hops`, `max_nodes`, `max_edges`). The verifier independently replays all revisions against the single mounted 8 MiB chart before evaluating any path. If `attestation` is supplied, the sender's signature is verified using the router identity/key and Ed25519 public key independently pinned in HERMES operator configuration. The returned `sender_authenticated` flag is true only for a valid pinned signature on that exact graph commitment and history; `nonce_replay_protected` remains false. An unsigned request is still verified but cannot claim sender authentication. No graph or chart write endpoint is exposed.

The library-only result keeps `sender_authenticated` false until the caller separately checks the attestation. Both plugin and library results keep `evidence_integrity_verified`, `native_mth_executed`, `learning_admitted`, and `mathematical_truth_promoted` false. A separate trusted immutable MathLeaf audit may validate content integrity, but does not authorize mathematical proof, learning or head writes.

CI independently replays revisions and tests path length, retracted history, contradictory candidates, tampering, truncation, replay requirement, chart epoch mismatch and hard bounds on Python 3.10 and 3.12.
