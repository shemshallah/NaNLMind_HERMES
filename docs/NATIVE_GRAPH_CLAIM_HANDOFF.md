# HERMES verified graph -> native NaN16 claim probe (staging only)

`pack_native_graph_claim_input(verifier, state_pages, history_pages)` in
`nanlmind_hermes/router_graph_native_input.py` stages a deterministic
U32 word batch for the NaNROUTER native guest kernel
`mth/nan16/mth_graph_claim_probe.nas` (NaNROUTER PR #107).

The function **independently replays the complete graph history again**
against the existing resident 8 MiB chart, confirms the original verified
commitment, rejects mixed generations, and rejects incomplete/tampered pages.
Only active candidate `supports` and `refutes` records are encoded.
Retractions and unrelated relation types are not silently treated as negative
votes. Every opposing candidate for the same exact directed `(src,dst,layer)`
is retained; no truth or transitivity is inferred.

### Wire format

Each active record consumes four big-endian 32-bit words, in the same
deterministic edge-ID ordering as the verified state pages:

```
[src canonical cell, dst canonical cell, layer 0..9, relation (1 supports | 2 refutes)]
```

The entire snapshot must fit a 1,024-record bound; no partial packet is
returned on overflow. The returned `words` list contains these raw U32
payloads, not instruction opcodes. `word_digest_sha256` is domain-separated
SHA-256 over a 4-byte record count followed by the U32BE payload bytes;
`provenance_sha256` covers sorted canonical JSON of the parallel original
immutable edge/evidence references, and `envelope_sha256` binds the chart,
topology, graph/history commitments, epoch, word count and both digests.

These checksums and the existing graph commitment are **not a sender
signature, an execution receipt, or trusted dispatcher authorization**. The
staging function deliberately returns `sender_authenticated=false`,
`evidence_integrity_verified=false`, `native_mth_executed=false`,
`native_admission_verified=false`, `learning_admitted=false`, and
`mathematical_truth_promoted=false`. External Ed25519 and MathLeaf checks
must remain explicit and independent.

The native guest's exact-match question returns a four-state mask
(0 absent, 1 supports, 2 refutes, 3 contested); the HERMES packing helper
never computes the mask in Python. This avoids promoting an unverified
host-side inference as a native proof.

### Gates still open

1. Bind this immutable staged payload's digest, chart generation, graph
   commitment, trusted sender authentication and audit status to an
   authorized NMP lease with replay protection.
2. Execute the native probe in the same pinned NaN16 guest and produce an
   authenticated receipt for **that exact** frozen payload and rule epoch.
3. Verify the receipt and the independent MathLeaf admission policy before
   any native learning or canonical model promotion.

Dedicated HERMES CI checks synthetic adversarial histories and a **real**
NaNROUTER MC0DB state/history projection. Neither CI job writes chart data
or performs model training.
