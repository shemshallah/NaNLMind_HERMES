# Verified graph navigation (read-only)

`verified_candidate_walk(verifier, state_pages, start=..., target=..., max_hops=...)` is a bounded diagnostic for graph-based language learning research. It consumes the exact current-state page batch independently rehashed against the live HERMES 8 MiB chart, together with a complete verified NaNROUTER revision replay. It **never** builds another persistent graph, creates a new universe, changes chart/head memory, performs host-side math, or admits learning.

- Input nodes are canonical integers 0..32895: heads 0..255 and mirrored-pair cells 256..32895.
- Directed candidate edges are traversed by shortest-hop BFS, deterministically ordered by endpoints, L0–L9 layer, relation and stable edge ID. Non-candidate (retracted) claims are excluded from paths while remaining in the verified revision history.
- Optional L0–L9 layer and relation filters select only existing candidate statements. The output retains evidence IDs, versions, evidence SHA-256 and exact edge generations for independent follow-up validation.
- Simultaneous `supports` and `refutes` claims between the same ordered nodes/layer are **both preserved** as `contested_pairs`, never silently resolved.
- No transitivity, equivalence, mathematical entailment, graph-theorem proof or synthetic words are inferred from a path. A graph path is evidence-backed candidate connectivity only.

Maximum traversal: 8 hops, 1024 nodes and 12,800 matched candidate edges. Callers can set smaller limits; reaching a hard completeness bound fails closed. Replayed and query pages must match a single graph commitment/current chart generation, so post-verification mutation, truncation or mixed graph epochs are rejected. The method holds at most the already bounded request graph in transient memory.

This is deliberately a library primitive rather than a new remote write or read tool. It can later serve a signed, authenticated read-only navigation surface without making unverified evidence or native execution claims. Its result keeps `sender_authenticated`, `evidence_integrity_verified`, `native_mth_executed`, `learning_admitted`, and `mathematical_truth_promoted` explicitly false.

CI independently replays revisions and tests path length, retracted history, contradictory candidates, tampering, truncation, replay requirement, chart epoch mismatch and hard bounds on Python 3.10 and 3.12.
