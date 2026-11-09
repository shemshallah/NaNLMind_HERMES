# NaNLM accuracy audit

Status: **read-only re-score of an existing artifact. No training run, no new
live result, no proof claim.**

Harness: `scripts/nanlm_accuracy_audit.py`
Tests: `tests/test_nanlm_accuracy_audit.py`

## Artifact and lineage

The audited data is the 6-round direct-run artifact for
`docs/NANLM_GEMMA_LEARNING_OVER_TIME.json`, state directory
`/data/data/com.termux/files/usr/tmp/nanlm-train-direct`, teacher
`gemma4:31b-cloud`, 14 held-out probes.

During the audit a separate fixed re-run replaced the live JSON path with a
different run (`state_dir=/data/data/com.termux/files/usr/tmp/nanlm-train-fixed`).
The direct-run probe snapshot was recovered from the existing independent
plateau diagnostic timeline. Its seven vectors match the direct artifact
exactly. The CLI defaults to the live path, so rerunning it now audits whatever
is currently in that file; pass `--input` to point at the direct-run snapshot
to reproduce the table below.

This distinction matters: the fixed re-run has a different curve and does not
show the 3->4 or 5->6 flips described here. The observations in this document
are about the direct-run artifact named above, not the newer fixed run.

## Recomputed legacy hit vector

Every stage has denominator 14. Integer forms use probe order 0..13 as a binary
string.

| stage | hit vector | integer | hits |
|---|---|---:|---:|
| baseline | `00000000100000` | 32 | 1/14 |
| round 1 | `11100000100000` | 14368 | 4/14 |
| round 2 | `11101110010100` | 15252 | 8/14 |
| round 3 | `11110110101101` | 15789 | 10/14 |
| round 4 | `11101110101101` | 15277 | 10/14 |
| round 5 | `11111110110101` | 16309 | 11/14 |
| round 6 | `11110111111101` | 15869 | 12/14 |

## Per-probe transitions

| transition | net | exact flips |
|---|---:|---|
| baseline -> 1 | +3 | river miss->hit; sky miss->hit; grass miss->hit |
| 1 -> 2 | +4 | cats miss->hit; birds miss->hit; cat miss->hit; ice hit->miss; red+blue miss->hit; sun miss->hit |
| 2 -> 3 | +2 | dogs miss->hit; cats hit->miss; ice miss->hit; red+blue hit->miss; yellow+blue miss->hit; circle miss->hit |
| 3 -> 4 | +0 | **dogs hit->miss; cats miss->hit** |
| 4 -> 5 | +1 | dogs miss->hit; red+blue miss->hit; yellow+blue hit->miss |
| 5 -> 6 | +1 | **cats hit->miss; water freezes miss->hit; yellow+blue miss->hit** |

The recomputed vectors match the stored `hit` flags in every stage. The full
audit JSON records every probe's `same`, `miss->hit`, or `hit->miss` status for
each transition; the table above lists only the changed probes.

## Rounds 3 -> 4: same count, not the same vector

- round 3: `11110110101101`, integer 15789, 10/14
- round 4: `11101110101101`, integer 15277, 10/14
- XOR: `00011000000000`, integer 1536, Hamming distance 2
- flips:
  - `What do dogs do?` hit -> miss:
    `Dogs are animals that bark.` -> `Dogs, and birds are animals.`
  - `What do cats do?` miss -> hit:
    `Cats, and birds are animals that bark.` -> `Cats make a meow sound.`
- net delta: +1 - 1 = 0

This is a dogs/cats swap. The fraction is unchanged because two probes flipped
in opposite directions. It is **not** "no change": the two changed bits are
worth 1024 and 512 in the 14-bit integer form, the XOR is nonzero, and the
Hamming distance is 2. The count equality is a cancellation, not evidence of
stability.

## Rounds 5 -> 6: three flips, not a left bit shift

- round 5: `11111110110101`, integer 16309, 11/14
- round 6: `11110111111101`, integer 15869, 12/14
- XOR: `00001001001000`, integer 584
- changed positions: 4, 7, 10
- flips:
  - `What do cats do?` hit -> miss (position 4):
    `Cats make a meow sound.` -> `Cats, dogs are animals that froze due to cold temperatures cause freezing.`
  - `What happens when water freezes?` miss -> hit (position 7):
    `Water has the ability to flow. ...` -> `Water that froze due to cold temperatures cause freezing turns water into ice. Happens blue and yellow mix to make purple. Freezes a square has the ability to flow.`
  - `What do yellow and blue make?` miss -> hit (position 10):
    `Yellow and blue mix to make purple.` -> `Yellow mix to make purple comes from red and blue create green.`
- arithmetic: 2 miss->hit - 1 hit->miss = **+1**
- left shift of round 5: `11111101101010`; it is not round 6 (Hamming distance 6).

The +1 is the sum of three independent per-probe flips. It is not a bit-shift of
the previous vector, and it is not the fraction producing an arithmetic error.

## Why the decimal strings look like the same digits shifted

The displayed values are exact fractions with denominator 14:

- `11/14 = 0.7 857142857142...`
- `12/14 = 0.8 571428571428...`
- `1/14 = 0.071428571428...`

Both are multiples of `1/14`, and `14 = 2 × 7`. The decimal expansion of `1/7`
is the repeating cycle `142857`; multiplying/rotating through the numerator
produces the same cyclic digits. The two displayed strings are therefore
expected to share/rotate the `85714285714` block. The values are **not equal**:
they differ by exactly `1/14`, and the integer counts differ by one (`11` vs
`12`). This is a property of decimal notation, not an accuracy/bit-shift bug.

To avoid the ambiguity, the live harness now reports exact `hits/total`,
`percent`, the hit vector, and the structured-hit vector; the long float is no
longer the primary accuracy display.

## Actual causes: measurement defects

1. **Keyword substring anywhere.** The training script's scoring is
   `hit = any(word.casefold() in response.casefold() for word in expected)`.
   An expected token in any unrelated clause counts. The clean example is round
   6 yellow+blue: expected `green`; the reply is
   `Yellow mix to make purple comes from red and blue create green.` The
   `green` belongs to a red/blue clause after `comes from`, and the reply also
   says `purple`. It is counted as a hit. Round 6 water-freeze is the same
   pattern: `ice` appears inside a rambling three-sentence reply that also
   discusses purple and a square. `any` also means one of `ice`/`frozen` is
   enough for that probe.
2. **No structured/direct requirement.** The headline `hit` ignores the
   `structured` flag and requires no answer slot. In the direct-run artifact
   every probe's `structured` value is `null`; that is missing measurement, not
   a measured 0% structured rate. The secondary structured rate therefore says
   nothing about this run.
3. **Probes mutated model state in the old artifact.** The direct run predates
   the read-only probe path. Probes were sent through normal `nanlm_chat`
   learning, not `probe=True`. A probe could therefore add its question and
   reply to the learned state, and later rounds were not held out against
   earlier probe exposure. The artifact has no probe/read-only markers and no
   before/after state digests, so it cannot prove otherwise. The current script
   passes `probe=True`; the old artifact remains confounded.
4. **No per-probe state fingerprint or flip ledger.** The old artifact stores
   only a float rate and a boolean `hit` per probe. It has no hit vector, no
   numerator/denominator, no per-probe state fingerprint, and no flip ledger.
   The current script now writes a stage-level before/after digest and a
   derived `flips_from_previous`, but a stage-level digest cannot identify
   which probe changed state. This audit reconstructs flips from stored hit
   flags, which is weaker than a state fingerprint per probe.

The 3->4 and 5->6 behavior is therefore explained by per-probe flips plus
permissive keyword scoring, not by an arithmetic or bit-shift bug in a fraction.

## Corrected measurement model

The harness reports three separate metrics, all offline re-scores of the same
stored replies:

- `keyword_hit`: exact legacy reproduction; expected term anywhere in the
  reply. This is what the stored `hit` means.
- `structured_hit`: `keyword_hit` **and** the recorded `structured` flag is
  true. Null/missing structured values are counted as not measured.
- `direct_exact_hit`: expected term in a bounded direct-answer span. A
  structured candidate is accepted by definition. Otherwise the expected term
  must be at the start of the answer (optionally after an article) or after a
  predicate cue, in the first sentence before connectors such as `comes from`,
  `because`, `due to`, or `which`. This rejects the yellow/blue reply above and
  the rambling water-freeze reply. It is a deterministic heuristic over stored
  text, not a semantic prover.

| stage | keyword | structured | direct exact | denominator |
|---|---:|---:|---:|---:|
| baseline | 1 | 0 (not measured) | 0 | 14 |
| round 1 | 4 | 0 (not measured) | 3 | 14 |
| round 2 | 8 | 0 (not measured) | 8 | 14 |
| round 3 | 10 | 0 (not measured) | 10 | 14 |
| round 4 | 10 | 0 (not measured) | 10 | 14 |
| round 5 | 11 | 0 (not measured) | 11 | 14 |
| round 6 | 12 | 0 (not measured) | 10 | 14 |

Final direct-answer vector: `11110110110101` (10/14, 71.4%). Final keyword
vector: `11110111111101` (12/14, 85.7%). The two missing strict hits are
exactly positions 7 and 10: water-freeze and yellow+blue, the two fragile
keyword hits. This is a corrected readout of the stored artifact, not a new live
accuracy result. The corrected metric does not claim that the student can
answer these probes in a fresh run; the old run's probe mutation and missing
structured flags prevent that inference.

## Reproduce

```bash
# Live/default artifact (audits whatever is at the default path)
python3 scripts/nanlm_accuracy_audit.py

# Direct-run snapshot used for this document
python3 scripts/nanlm_accuracy_audit.py \
  --input /data/data/com.termux/files/usr/tmp/opencode/NANLM_GEMMA_LEARNING_OVER_TIME_direct_snapshot.json \
  --json /tmp/nanlm-accuracy-audit.json \
  --markdown /tmp/nanlm-accuracy-audit.md

python3 -m unittest tests.test_nanlm_accuracy_audit -v
```

## Boundaries

- Read-only re-score; no training, state mutation, relation promotion, or
  package changes.
- No new live result is invented; only the stored direct-run artifact is
  audited.
- The old artifact's probe calls were not read-only, so later rounds may include
  earlier probe text in the learned state.
- The corrected direct-answer metric is a strict offline heuristic, not proof of
  general language learning.
