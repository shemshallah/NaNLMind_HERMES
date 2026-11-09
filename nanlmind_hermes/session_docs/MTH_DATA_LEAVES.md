# MTH data leaves, hubs and constrained pairing (permanent session contract)

Source: plugin `mth_leaves.py`, `mth_catalog.py`, `nan0s_runtime.py`.

## Immutable leaf model

Every data leaf is a content-addressed immutable object with:

- `leaf_id` = `leaf:` + SHA-256 of its canonical content;
- `kind` (`vector`, `scalar`, `matrix`, `relation`);
- one hub ancestry chain (`hub:mth.vector`, `hub:mth.scalar`,
  `hub:mth.matrix`, `hub:mth.relation`) ending at `hub:mth.root`;
- a function definition (data, relation, or admitted native kernel);
- action relations (typed links to inputs, hubs and operators);
- mathematical identities and constraints;
- constants and provenance;
- a real NaN16 executable `.nan` image containing its metadata and payload.

There is no in-place mutation API. New evidence creates a new generation/leaf;
old leaves remain immutable. Hubs cannot be rewritten or deleted.

## Every MTH function is a leaf

`hub:mth.function` is the function root. `nanlm_mth_tree action=map` creates one
immutable kernel leaf for each of the 91 bundled MTH kernels under its family hub
(`hub:mth.function.<family>`), with source/fixture digests, shape, precision,
workspace, marker, status/CI claim, execution mode, and links to the family,
operator, vector/matrix/scalar, WordNet and constraint hubs. `action=audit`
verifies exactly 91 linked kernels, no missing/extra/duplicate aliases, digest
matches, hub ancestry and every leaf's NaN16 image. The current local audit is
91/91 linked across 44 family hubs with 91/91 images verified.

## Root hubs and WordNet

- `hub:mth.vector` defines vector = 256 float32 elements, immutable inputs,
  sparse representation, no dense per-word matrix, and the algebraic
  identities in the permanent vector contract.
- `hub:mth.wordnet` declares the canonical HW4 route
  `nan://GM0:SC0:MC0:BS0:0:HW4:0` and the vector term/object reference.
  WordNet definitions are lexical evidence, never execution authority.
- `hub:mth.constraints` holds deny-by-default invariants.
- `hub:mth.operator` admits only the operators in the hub table.

## Pairing `left::right`

- `connect` creates a relation leaf linking two compatible leaves.
- `dot`, `cosine`, `add`, `sub` require `vector::vector`, execute the admitted
  float32 kernel through the unified N16I general-operand path, check
  constraints and symmetry/antisymmetry identities, and create a new hash leaf.
- The N16I path runs on the host-built/bundled portable NaN0S core.
  `execute_vector_kernel_native` is an explicit diagnostic/parity helper only;
  it is not the default pairing path.
- The result is a new immutable leaf; inputs are never modified.
- Unknown leaves, hub pairing, incompatible kinds, unknown operators,
  unadmitted operators, non-finite values, failed kernel execution or a
  violated identity return an error and create/execute nothing.
- `{VECTOR::VECTOR}` therefore yields a new hash leaf only when the hub rules
  admit the pairing. `vector44::illegalpairing` is an ERROR/denial.

## Self-modification guard

- Hubs and leaves are immutable by construction.
- Pairing is constrained by kind/operator tables and mathematical invariants.
- A result must be a new content-addressed leaf; it cannot replace an input.
- Unknown or unregistered routes/kernels fail closed.
- Execution receipts (N16I or native diagnostic) are local/derived; they do not
  authorize NANNET routes, leases, merges or canonical head/commit updates.
