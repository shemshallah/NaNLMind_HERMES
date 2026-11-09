# Gemma architecture review (live teacher run)

Generated from the exact permanent session documents shipped with the plugin. This is a teacher review, not a measured training result.

- model: `gemma4:31b-cloud`
- prompt tokens: `7373`
- completion tokens: `1011`

## Reasoning

Review of local learning architecture against permanent session contracts, focusing on the boundary between symbolic evidence, native execution, and canonical state.

## Architecture assessment

- Strict separation between HERMES derived state and BS0 canonical state is maintained.
- MTH immutable leaf model prevents accidental state mutation during mathematical operations.
- WordNet HW4:0 is correctly positioned as read-only lexical evidence, not execution authority.
- The L0-L9 scratch space provides a necessary buffer for evidence before promotion.
- Native execution is correctly limited to 4 admitted kernels for arbitrary operands.
- The merge gate is a critical, non-bypassable boundary for canonical updates.

## Free-chat learning plan

- Online: User/Assistant turns trigger L0-L9 projections into 1 GiB paged scratch.
- Online: Populate concept matrix and relation rows as symbolic evidence.
- Online: Store contradictions as distinct events without coercion.
- Behind Approval: Conservative promotion based on support >= 3 and confidence >= 0.6.
- Behind Approval: Explicit call to nanlm_learning_sync for approved candidates.
- Behind Approval: Native relation-train only for supported/refuted triples.
- Final Gate: Separate canonical merge gate for any BS0/MC0 state change.

## WordNet HW4:0 plan

- Sequence: Word -> Lemma Index -> Synset Offset -> Typed Relation/Gloss.
- Integration: Project WordNet relations into L0-L9 scratch as lexical evidence.
- Constraint: Use WordNet as a source for 'subject/predicate/object' triples, not as a dense matrix.
- Constraint: Treat glosses as graph edges, not as parsed prose understanding.
- Failure: Unknown words must return an honest absence, not an LLM-fabricated synset.
- Failure: Missing wordnet_archive must render HW4 unavailable, not fallback to LLM memory.

## MTH data leaves and pairing plan

- Leaf Identity: Every MTH tool must operate on content-addressed SHA-256 leaves.
- Execution: Use NaN16 images for metadata/payload; no in-place mutation.
- Pairing Logic: 'connect' for relation leaves; 'dot/cosine/add/sub' for mathematical pairing.
- Validation: Check hub ancestry (e.g., hub:mth.vector) before admitting pairing.
- Constraint Check: Verify symmetry/antisymmetry identities before creating result leaf.
- Denial: Illegal pairings (e.g., vector::scalar for dot) must return ERROR without execution.
- Output: Every successful pairing must yield a new immutable hash leaf.

## Missing architecture

- Lack of a live, authenticated NaNROUTER instance to provide real lease/generation IDs.
- Absence of a verified merge-gate implementation for moving HERMES state to BS0.
- Missing differential test results for N16I guest images beyond the basic self-test.
- Undefined mapping between L0-L9 scratch indices and specific 255-head ensemble slots.

## Test plan

- Local: Verify MTH leaf creation and illegal pairing denial (no NANNET needed).
- Local: Run nanlm_mth_verify on 87 fixture kernels to confirm marker matches.
- Local: Simulate L0-L9 projection and conservative promotion logic with mock data.
- Fail Closed: Attempt to route to HW4:0 without a valid router context; must fail.
- Fail Closed: Attempt to mutate a hub leaf; must return error.
- Assertion: Verify that support < 3 prevents promotion to HERMES derived state.

## First experiment

- name: MTH Pairing Constraint Validation
- command/API: `nanlm_mth_execute(left=vector_leaf_A, right=illegal_leaf_B, op='dot')`
- what it would prove: The hub-based pairing constraints and immutability of leaves are enforced.
- what it would not prove: That the native float32 kernel is mathematically accurate or that NANNET routing is active.

### Assertions

- Return value is ERROR
- No new leaf_id is generated
- No mutation of vector_leaf_A occurs

