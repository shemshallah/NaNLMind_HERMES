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
