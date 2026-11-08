# MTH bridge and the L0-L9 layering cognition system

The native **MTH** (NaN-Native Mathematical Substrate) lives in NaNROUTER as
NaN16 source. Hermes does not reimplement native kernels on the host; it
interfaces with the MTH contracts locally and marks `native_verified: false`
until a native MTH runner is configured.

## Root 4x4 Boolean table

The four NaN face classes (`+s`, `-s`, `+q`, `-q`) form a 4x4 ordered pair
space. Each pair addresses one 4-bit Boolean truth table, so the root is the 16
functions of two inputs (`nanlmind_hermes/mth_layers.py: root_table()`).
All 16 class nibbles are retained; no pair is collapsed into its mirror.

## The ten semantic layers

| layer | name | learned signal |
|---:|---|---|
| L0 | bit | the 16 root Boolean functions |
| L1 | nibble | 4-bit composition through the root |
| L2 | byte | 8-bit composition through the root |
| L3 | phonetic | vowel/consonant and grapheme features |
| L4 | morpheme | stems, prefixes, suffixes |
| L5 | lexical | vocabulary and frequency |
| L6 | word_completion | prefix → word prediction |
| L7 | sentence | typed relations with graded, directional truth |
| L8 | paragraph | coherence and sentence linkage |
| L9 | narrative | causal/temporal event chains |

`MTHLayerStack.learn(sentences, rows)` verifies L0-L2 exactly and builds L3-L9
from the exposed corpus and the plugin memory rows. It reports per-layer
accuracy/coverage and the number of typed edges.

## Cognition interface

`MTHScorer` scores a candidate reply with:

- **truth**: object overlap with a learned edge, requiring subject-predicate
  coherence and multiplying the edge degree; reversed and object-only matches
  do not count;
- **frame**: whether the edge predicate matches the question type
  (`what is` → `is`, `can` → `can`, `why/cause` → `causes`);
- **coherence**: topic overlap with the query;
- **category_error**: object overlap from an edge with a different subject
  (this is what penalizes stitched sentences such as
  `Volcano is a flowing body of water can freeze`).

The temporal predictor adds `0.35 * score` to each candidate and records the
MTH fields in its `considered` list, so deliberation shows truth, frame and
category-error evidence next to next-word likelihood.

## Status and boundary

- `native_verified: false`: the host reference intersects MTH, but the native
  NaN16 implementation remains authoritative in NaNROUTER.
- The local runner, 8 MiB chart, mapped register and `HW2:0` NaN16DE link are
  described in `docs/LOCAL_RUNNER.md`.
- `docs/LONG_CHAT_DEMO.md` records the root/nibble/byte accuracies, layer
  summary, held-out session and feedback re-deliberation.

## Heads actively use MTH: the special scratch register

Every relation is transformed through a root Boolean function, not hashed:

```
mask = (sum(ord(c) for c in predicate) + len(subject) + len(object)) mod 16
s = fold(subject, mask)      # byte pairs through the root function
p = fold(predicate, mask)
o = fold(object, mask)
head  = (s ^ (p << 1) ^ (o << 2)) & 0xFF
leaf  = (s << 16) | (p << 8) | o
```

The leaf records its head, mirror (`reverse8`), Boolean function name, layer
(`sentence` or `narrative`, or the memory resolution when present), value and
relation digest.

`HeadScratchRegister` actively organizes the DB tree:

- the **aggregate register** is mapped in the chart extension at
  `HEAD_REGISTER_OFFSET` (magic `NLMHEAD1`) with a 256-head table and leaf
  entries;
- every used head also receives a **1 KiB leaf** in its native head scratchpad
  (magic `NLMLEAF1`) inside the in-memory 8 MiB chart;
- the register is persisted to `state_dir/language/mth_heads.json` and exposed
  in chat metadata as `native_synthesis.heads` (`head_count`, `leaf_count`,
  `head_scratchpads`, masks, layers).

`docs/LONG_CHAT_DEMO.md` records the live head mapping for the session.

## Concept matrix: saturation through superposition

The C(256,2) = 32,640 unordered pair cells are a relation space, not a
one-word-one-cell map. `ConceptMatrix` ingests every statement in superposition:

- every co-occurring concept pair accumulates weight in its cell;
- typed relations add a role and direction (`component_of`, `member_of:chemical`,
  `member_of:geometric`, `opposition`, `causes`, ...);
- a concept's leaf identity is the hash of its relation links, not its spelling;
- saturation measures occupied pair cells, density, average concept degree,
  roles and top pairs.

**First Gemma saturation session** (`docs/GEMMA_CORPUS.md`):

- 89 statements, 138 concepts, 560 occupied pairs = **1.72% density**;
- 87 MTH head leaves after the organization phase;
- the matrix is a real substrate but is sparse: prose alone does not saturate it;
- known-probe accuracy was low (1/12) because the temporal generator was not
  consulting the relation store. That gap is fixed by **structured relation
  candidates**: a yes/no question is answered from a learned edge, e.g.
  `Is copper a chemical element?` → `Copper is a chemical element.`,
  `Is the sky a component of the earth?` → `The sky is a component of the earth.`
  (marked `structured: true`);
- unsupported yes/no questions still confabulate (`Is copper a geometric
  element?`) when no relation or support gate exists. Abstention only appears
  from uncertainty exposure and needs a grounded support gate.

Next: a coverage-guided saturation protocol (target uncovered pair regions),
learned support gating for unsupported questions, and organization snapshots
that compare head→leaf maps before/after chatting.
