# Real native graph result → signed HERMES receipt (CI only)

The CI regression `scripts/verify_real_native_graph_receipt_e2e.py` joins
the existing HERMES replay-verified real NaNROUTER MC0DB graph to an actual
NaN16 guest execution, instead of signing a synthetic mask produced in Python.

It uses one actual HERMES chart and existing trusted MathStore for the entire
test. After constructing opposing candidate edges and a retracted edge in
MC0DB, it verifies complete graph history, a pinned Ed25519 router sender,
and all historical MathLeaf digests. It stages the exact immutable native
`[src,dst,layer,relation]` word snapshot with bounded and domain-separated
payload/provenance digests.

The runner emits a temporary NaN16 fixture using the pinned NaNROUTER native
claim probe, assembles it with pinned NaN0S, and executes the actual guest.
The host reads the **guest serial output**, expecting independently known
fixture outcomes 3 (contested L4 relationship) and 0 (absent L4 relationship).
No Python code implements the relationship operator.

For each actual guest observation a freshly generated **CI-only** worker
Ed25519 key signs a receipt that binds the chart generation, graph history and
commitment, native input and provenance digests, pinned sender attestation,
historical MathLeaf audit digest, exact query, guest-image SHA-256, rule epoch,
nonce and a deliberately untrusted CI lease sentinel. HERMES verifies this
statement using the exact same resident chart and still-open MathStore.
A post-signature forged mask is rejected. No chart mutation occurs.

This proves the CI integration against a genuine NaN16 run but **does not
turn the CI worker key or untrusted lease sentinel into deployment authority**.
It does not attest VM internals independently, durably consume dispatch
nonces, validate an NMP lease, authorize inference as mathematical truth, or
admit training. The receipt verifier must continue reporting all those gates
as false until separately implemented and verified in production.
