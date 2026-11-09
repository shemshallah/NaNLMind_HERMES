# HERMES signed graph-native result receipt (read-only)

`verify_native_graph_claim_receipt` adds the next boundary after the
history-replayed, candidate-only NaN16 graph probe.

A trusted deployment must provide the active HERMES graph verifier and its
entire state/history page set; a separately pinned NaNROUTER Ed25519 sender
key; a trusted historical MathLeaf resolver; a separately pinned native
worker Ed25519 public key; and trusted expected values for the NaN16 guest
image SHA-256, NMP lease digest, rule epoch, dispatch nonce and exact query.

The verifier **actually repeats** the history-bound stage check, validates the
router signature and audits every historical MathLeaf reference independently,
before checking the exact worker-signed result. The signed body binds:

- Native word / evidence provenance / graph envelope SHA-256 digests
- Current graph commitment, historical digest and chart generation
- Pinned router signature digest and independent MathLeaf audit digest
- Pinned worker identity, native guest-image SHA-256 and rule epoch
- Trusted lease digest, dispatch nonce and exact directed L0-L9 query
- Guest success status 1, four-state mask 0/1/2/3 and issued timestamp

The HERMES helper does **not** compute the relationship in Python; a trusted
worker must first actually run the same staged payload inside the NaN16 guest.
The verifier confirms that a pinned worker signed the claimed result, **not**
independently that its execution took place or was correct. A mere signature
does not attest the underlying VM measurement. Both absent (0) and contested
(3) are retained; contradictions are not silently discarded.

## Non-authorizing security boundary

This interface intentionally returns
`native_execution_independently_verified=false`,
`nonce_replay_protected=false`,
`lease_dispatch_authorized=false`,
`native_admission_verified=false`,
`learning_admitted=false`, and `mathematical_truth_promoted=false`.

It does **not** reserve or consume an NMP lease/nonce. Do not treat the caller's
expected lease digest alone as a lease verification. Before any learning
promotion, implement durable at-most-once dispatch consumption and bind a
verified native execution receipt to actual worker/guest measurement, plus the
separate MathLeaf proof/admission policy. Nothing here mutates the resident
8 MiB chart or canonical model.

Dedicated CI covers both Python 3.10 and 3.12, stale/tampered graph state,
wrong pinned keys, wrong history/evidence/word digests, mismatched chart
generation, guest-image/lease/epoch/query/nonce substitution, expired
receipts, forged native masks, and forbidden `learning_admitted` injection.
