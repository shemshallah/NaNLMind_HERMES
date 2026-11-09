# NaNLM plateau diagnostic

Status: **read-only diagnostic, no training run, no fixed result claimed.**

This document explains the recorded 71.4% plateau in
`docs/NANLM_GEMMA_LEARNING_OVER_TIME.json` from the actual per-probe replies.
The independent harness is `scripts/nanlm_plateau_analysis.py` with tests in
`tests/test_nanlm_plateau_analysis.py`. It does not modify package code, package
documentation, the run JSON, or the memory ledger.

## Snapshot used here

- Run JSON analyzed (historical snapshot): `docs/NANLM_GEMMA_LEARNING_OVER_TIME.json`
- Run sha256: `a15f233423f90610e3bced2b637327c728823b76aa881e950d315f7fa36b0d88`
- Recorded stages: baseline plus 6 rounds, 14 held-out probes
- Teacher: `gemma4:31b-cloud`, native NaNLM student
- State dir: `/data/data/com.termux/files/usr/tmp/nanlm-train-direct`
- Memory DB: `universe/mathematical_memory.sqlite3`
- DB snapshot during analysis: 29 active relation rows. The DB is live: its
  observation/assessment totals and sha256 changed during this session, so the
  harness records a per-inspection hash and count instead of treating the file
  as fixed.
- The current checkout changed after the run JSON was written: the analyzed run
  file mtime is `15:56:37`, while the package files were already newer when the
  plateau report was generated, and they changed again during the session. The
  harness records the exact `current_code.snapshot` sha256/mtime in JSON.
  Current-code candidate evidence in this document is therefore **not**
  evidence about the code that produced the recorded replies.

Live-run caveat: while this document was being prepared, the run JSON was
replaced by a new live run (state
`/data/data/com.termux/files/usr/tmp/nanlm-train-fixed`; its file hash changes
as each round is appended, and it was still in progress while this was
written). The numeric analysis below is specifically the completed six-round
`a15f23...` snapshot that exhibited the 71.4% plateau. It is not a claim about
the replacement run. Re-run the harness against the current file; it reports
`run_status: in_progress` instead of pretending a partial file is final.

Reproduce:

```bash
python3 scripts/nanlm_plateau_analysis.py \
  --json /tmp/nanlm-plateau.json \
  --markdown /tmp/nanlm-plateau.md
python3 -m unittest tests.test_nanlm_plateau_analysis -v
```

## Observed hit-rate curve

| stage | hits | hit rate |
|---|---:|---:|
| baseline | 1/14 | 7.1% |
| round 1 | 4/14 | 28.6% |
| round 2 | 8/14 | 57.1% |
| **round 3** | **10/14** | **71.4%** |
| **round 4** | **10/14** | **71.4%** |
| round 5 | 11/14 | 78.6% |
| round 6 | 12/14 | 85.7% |

The plateau is the two-round run at rounds 3 and 4: both are exactly 10/14, or
71.4%. `detect_plateaus()` in the harness identifies it automatically from the
recorded hit rates.

Important: the 10 "hits" at the plateau are not all structured answers. At
round 3, 4 of the 10 hits are fragile: the expected token appears somewhere in
free generation, with no structured candidate. At round 4, 5 of the 10 hits are
fragile. Only 6/14 and 5/14 respectively are structured hits. The headline
71.4% therefore overstates what the student can reliably answer.

## Exact per-probe data at the plateau

| probe | expected | round 3 reply | round 4 reply | status |
|---|---|---|---|---|
| What is a river? | water | `River consists of water.` | `River consists of water.` | hit, fragile (no river relation row) |
| What color is the sky? | blue | `Sky possesses a blue color.` | `Sky possesses a blue color.` | hit, fragile (no sky relation row) |
| What color is grass? | green | `Grass is green.` | `Grass is green.` | hit, fragile (no `what color` structured intent) |
| What do dogs do? | bark | `Dogs are animals that bark.` | `Dogs, and birds are animals.` | hit then **miss** |
| What do cats do? | meow | `Cats, and birds are animals that bark.` | `Cats make a meow sound.` | **miss** then hit, fragile |
| What do birds do? | fly | `Birds have the ability to fly.` | `Birds have the ability to fly.` | hit (current code has an action candidate; run-time intent unverified) |
| What is a cat? | animal | `A cat is an animal.` | `A cat is an animal.` | hit |
| What happens when water freezes? | ice, frozen | `Water has the ability to flow. Happens the ability to fly. Freezes the sun creates light.` | `Water has the ability to flow. Happens the sun is a hot star. Freezes the sun is a hot star.` | **miss both rounds** |
| What is ice? | frozen, water | `Ice is frozen water.` | `Ice is frozen water.` | hit |
| What do red and blue make? | purple | `Red and blue make green.` | `Red and blue make green.` | **miss both rounds** |
| What do yellow and blue make? | green | `Yellow and blue make green.` | `Yellow and blue make green.` | hit, fragile (no make/create relation row) |
| What is the sun? | hot, light, star | `The sun is hot.` | `The sun is hot.` | hit |
| What shape has four sides? | square | `Has four sides.` | `Shape with four sides.` | **miss both rounds** |
| What is a circle? | round | `A circle is round.` | `A circle is round.` | hit |

The plateau has three shared misses and one rotating fourth miss:

- Shared: `What happens when water freezes?`
- Shared: `What do red and blue make?`
- Shared: `What shape has four sides?`
- Rotating: `What do cats do?` at round 3, `What do dogs do?` at round 4.

### Shared miss 1: water freezing has no causal answer

By round 3 the ledger contains state facts (`Cold causes freezing`,
`Frozen water is ice`, `Ice is frozen water`), but no causal/event row whose
subject is water freezing and whose object is ice. The teacher paraphrases
(`Freezing water creates ice`, `Freezing turns water into ice`,
`Freezing water becomes ice`) were not extracted by the run-time parser. The
generator filled the gap with `the sun creates light` and `the sun is a hot
star`, i.e. an unrelated causal sentence from another subject.

Primary classification in the harness: `causal_question_missing`. The
`taught_but_not_extracted` evidence flag is false at the final snapshot because
`Ice is frozen water` exists, but the causal-direction row for
`water freezes -> ice` never exists.

### Shared miss 2: red+blue is answered with yellow+blue's object

Both plateau rounds reply `Red and blue make green.` The sibling curriculum
taught yellow+blue green repeatedly, while the red+blue purple rows used
`create` / `mix to make`, which the run-time parser did not turn into relation
rows. The DB has no `(red+blue, make, purple)` row and no
`(yellow+blue, make, green)` row. The generator therefore binds the wrong
subject: the question says red+blue, the answer object is green.

Primary classification: `wrong_subject_interference`. This is a subject-binding
bug, not a missing teacher statement: `Red and blue create purple` appears in
round 2, and `Red and blue mix to make purple` appears in round 5.

### Shared miss 3: the square fact needs reverse lookup

By round 3 the ledger contains three square rows:

- `A square has four sides`
- `Squares have four sides`
- later, `A square is a shape with four sides`

The probe asks for the subject from the property: `What shape has four
sides?` The replies are `Has four sides.` and `Shape with four sides.` They
repeat the object and never select `square`.

Primary classification: `reverse_lookup_missing`. The fact is present in the
forward direction and the teacher is not missing; the observed run had no
inverse object/property -> subject path.

### Rotating fourth miss: cats and dogs

At round 3, `What do cats do?` returns `Cats, and birds are animals that
bark.` The expected `meow` is absent; `bark` comes from the dog fact. The DB has
no cat/meow relation row even though `Cats make a meow sound` was taught in
round 1, because `make` was not extracted then.

At round 4, `What do dogs do?` returns `Dogs, and birds are animals.` The dog
row `Dogs is animals that bark` exists, but the reply drops `bark` and keeps
only the generic hypernym. This is a generation/selection miss, not a
curriculum miss.

## After the plateau: do not call it fixed

Round 5 reaches 11/14 and round 6 reaches 12/14, but the later curve is noisy and
the hit metric is permissive:

- `What shape has four sides?` never passes in any round. It is still a
  reverse-lookup miss at round 6 (`Shape with four sides.`).
- `What do cats do?` regresses at round 6 to
  `Cats, dogs are animals that froze due to cold temperatures cause freezing.`
  It is still a wrong-subject miss.
- Round 6 `What do yellow and blue make?` is counted as a hit because the
  reply contains the token `green`:
  `Yellow mix to make purple comes from red and blue create green.`
  The reply also contains the wrong object `purple`, and the sentence is not a
  clean yellow+blue answer. This is a **fragile/contradictory hit**.
- Round 6 `What happens when water freezes?` is counted as a hit because
  `ice` appears inside a rambling reply that also contains `blue and yellow mix
  to make purple` and `a square has the ability to flow`.

The harness reports `fragile_hit_count` (29 across all stages at this
snapshot) and records the exact reply. A later round is not proof of a fix until
the strict answer-slot metric and the failed probes are re-tested.

## Code bugs vs curriculum gaps

### Code-level causes

The top three code-level causes of the 71.4% plateau, in order, are:

1. **The structured question-intent path could not select answers for the
   plateau's question forms.** Even when the fact was present, the run-time
   path had no usable `what do X do?`, `what do X and Y make?`,
   `what happens when X?`, or `what shape has ...?` selection.
2. **Taught paraphrases did not become relation rows.** `create`, `make`,
   `mix to make`, `becomes`, `turns into`, `consists of`, and `possesses`
   statements were often stored only as observations, so the red+blue,
   yellow+blue, river, sky, and cats probes had no row to select.
3. **No subject-scoped selection or inverse lookup in the observed run.**
   The generator pulled sibling objects (red+blue -> green, cats -> bark,
   water-freezing -> sun/light), and could not invert
   `A square has four sides` to answer `square`.

The detailed code-level list follows.

1. **Question-intent coverage.** At the run time represented by the JSON, the
   structured answer path could not select answers for `what do X do?`,
   `what do X and Y make?`, `what happens when X?`, `what shape has ...?`, or
   `what color is X?`. The harness calls the current
   `mth_layers.structured_candidates()` and records whether a candidate is
   possible **now**; that is deliberately separate from the recorded miss.
2. **Relation extraction gaps.** The run-time parser turned `is/has/can/causes`
   statements into rows but did not consistently turn the teacher's
   `create`, `make`, `mix to make`, `becomes`, `turns into`, `consists of`, or
   `possesses` forms into rows. Evidence from the snapshot:
   - no river row, despite `A river consists of water` and
     `Rivers transport water`;
   - no sky row, despite `The sky possesses a blue color`;
   - no red+blue purple row, despite `Red and blue create purple` and
     `Red and blue mix to make purple`;
   - no yellow+blue green make row, despite repeated `make` / `create` /
     `mix to make` statements;
   - no cat/meow row, despite `Cats make a meow sound`.
3. **No subject-scoped answer selection.** When the structured path returns no
   candidate, the temporal generator can emit an object belonging to a sibling
   subject. The plateau shows this as red+blue -> green, cats -> bark, and
   water-freezing -> sun/light. A usable relation row for one subject is not
   enough if the generator is allowed to answer with another subject's object.
4. **No reverse lookup in the observed run.** The square and ice facts exist in
   the forward direction (`A square has four sides`, `Frozen water is ice`), but
   the replies do not invert them to `square` / `ice` when the question asks
   from the object side. The harness confirms that current code can produce a
   reverse candidate for the square probe; the recorded run did not.
5. **The hit metric is substring-based.** `expected in response` counts wrong
   or contradictory sentences as hits. The round-6 yellow+blue reply is the
   clean example: it contains `green` but answers `purple`. The harness marks
   such hits as fragile.

### Curriculum gaps

- The sibling color-fact blocks are unbalanced. Round 3 teaches yellow+blue
  green but not red+blue purple; round 5 teaches red+blue purple but not
  yellow+blue green. The student then swaps the sibling object.
- The teacher uses many paraphrases with verbs the run-time native parser does
  not extract (`create`, `mix to make`, `becomes`, `turns into`,
  `consists of`, `possesses`). This is a curriculum/parser interface gap: the
  teacher supplied the fact, but the student's extractor did not admit it.
- `Cats make a meow sound` is lesson text, but the native parser did not create
  a `Cats meow` or `Cats make meow` relation row. The cats probe is therefore
  unsupported at the relation layer.
- Fire/red and snow/white are taught but have no held-out probe, so the
  diagnostic cannot measure those facts. That is a coverage gap in the probe
  set, not a plateau cause.

## Memory DB evidence for the plateau probes

The harness opens the SQLite file with `mode=ro` and
`PRAGMA query_only=ON`, records before/after sha256 and size, and verifies the
file is unchanged. It never calls `MemoryStore`, which would create tables/WAL
state. The DB snapshot had 29 active relation rows; 27 are traceable to this
run's teaching statements and 2 are pre-existing `Explain ...` rows.

| probe | relevant rows at snapshot | taught but not extracted | structured candidate possible with current code |
|---|---|---|---|
| water freezes | `Cold causes freezing`; `Frozen water is ice`; `Ice is frozen water`; `Ice is water that froze due to cold`; no `water freezes -> ice` causal row | state rows extracted; no causal-direction row for the `create`/`turn`/`become` paraphrases | yes (current checkout), but not in the recorded run |
| red+blue make | no purple row; only generic/sibling rows | yes (`create`, `mix to make`) | no |
| yellow+blue make | no green make row; `Grass is green` is a different subject | yes (`make`, `create`, `mix to make`) | no |
| shape has four sides | `A square has four sides`; `Squares have four sides`; `A square is a shape with four sides` | no | yes (current checkout), still missed in the run |
| cats do | no meow row; `Cats is animals` only | yes (`Cats make a meow sound`) | no |
| dogs do | `Dogs is animals that bark`; `Dogs is animals` | no | yes (current checkout), but round 4 dropped `bark` |
| sky color | no sky row; only `Grass is green`, `Fire is red`, `Snow is white` | yes (`The sky possesses a blue color`) | no |
| ice | `Ice is frozen water`; `Frozen water is ice` | no | yes (current checkout), run was fixed by round 3 |
| river | `Water has the ability to flow`; no river row | yes (`A river consists of water` was not extracted) | no |

The final snapshot has no relation row with `purple` or a make/create predicate
at all. That is direct evidence that the red+blue / yellow+blue plateau is a
relation-extraction gap, not merely a teacher omission.

## Coverage matrix (target fact -> teaching -> rows -> probe result)

| target fact | probes | taught at plateau | relation rows at plateau | plateau result |
|---|---|---|---|---|
| river and water | `What is a river?` | yes, 3 statements | water-flow row only; no river row | hit, fragile |
| sky/grass/fire/snow colors | sky, grass | yes | grass row only; no sky row | hit, fragile |
| cat/dog/bird animals | cat, dogs | yes | cat/dog/bird hypernym rows | cat hit; dogs miss at round 4 |
| pet actions | dogs, cats, birds | yes | birds-fly row; no meow row; dog-bark row only after round 3 | dogs/cats miss at least once |
| heat, freezing, ice | water freezes, ice | yes | cold/frozen/ice state rows; no causal water-freeze row | water freeze miss; ice hit |
| color mixing | red+blue, yellow+blue | yes | no make/create rows | red+blue miss; yellow+blue fragile hit |
| sun | sun | yes | hot/light/star rows | hit |
| shapes | square, circle | yes from round 3 | square-has-four-sides rows; circle-round row | square miss; circle hit |

This matrix shows the plateau is not a shortage of teaching statements. The
missed probes have 6-18 relevant taught statements by the plateau rounds. The
missing link is usually a relation row or a question-frame binding.

## Ablation plan (planned, not run)

The harness emits the plan as data with `status: planned_not_run`. It also
checks that no example statement contains an exact probe question. No result is
claimed here.

| id | intervention | example teaching (not exact probes) | readout |
|---|---|---|---|
| A1_more_repetitions | repeat each target fact 3x per round with varied position/length | `A river carries water.` / `Water is what a river carries.` / `Water can flow.` | hit curve, fragile hits, interference count |
| A2_explicit_qa_teaching_without_exact_probe | indirect Q/A forms with wording different from the probe | `Q: Which color does the sky have? A: Blue.` / `Q: Which animal says meow? A: A cat.` | exact-probe overlap, candidate rate, hit curve |
| A3_subject_balanced_color_facts | teach sibling color/mixing facts together, balanced | `Red and blue make purple.` / `Yellow and blue make green.` in the same round | sibling swap count, per-probe stability |
| A4_reverse_lookup | teach and index inverse object/property -> subject forms | `Four sides make a square.` / `The round shape is a circle.` | reverse miss count, square probe |
| A5_causal_paraphrase | use extractable causal verbs for the freezing chain | `Freezing water causes ice.` / `Cold causes water to freeze.` / `When water freezes it becomes ice.` | causal miss count, water-freeze probe |

Suggested execution order: A2/A3 first (curriculum balance and question-shaped
exposure), then A4/A5 (new answer paths), then A1 as a stability check. Any
ablation must keep the held-out probe strings out of the teaching statements and
must report both the old substring hit and a strict answer-slot hit.

## Exact fixes and tests needed

1. **Certify the structured intents in the current checkout.**
   `mth_layers.structured_candidates` now contains branches for action,
   make/create, reverse, and `what happens when` questions. Add focused tests
   that feed synthetic relation rows and assert the answer for each branch. Do
   not rely on the recorded run, which predates those branches.
   Tests: `test_structured_candidates_action_intent`,
   `test_structured_candidates_make_intent`,
   `test_structured_candidates_reverse_intent`,
   `test_structured_candidates_causal_intent`.
2. **Extract the teacher's actual paraphrases.**
   Extend `chat._NATIVE_PATTERNS` / the intransitive parser for `make/makes`,
   `create/creates`, `produce/produces`, `mix ... to make`, `become/becomes`,
   `turn/turns into`, `consist/consists of`, `possess/possesses`, and
   `has/have the ability to`. For every taught sentence, assert one row with
   the expected subject/predicate/object.
   Tests: `test_native_parser_extracts_teacher_paraphrases`,
   `test_native_parser_extracts_intransitive_actions`.
3. **Scope generation to the asked subject.**
   When relation rows exist for the question subject, restrict the candidate
   object set to those rows before free generation; never let a sibling
   subject's object win. Regression: alternate red+blue/purple and
   yellow+blue/green in the same round and assert no swap.
   Tests: `test_generation_subject_scope_red_blue_vs_yellow_blue`,
   `test_generation_does_not_answer_cats_with_dogs_bark`.
4. **Add an inverse index / reverse query.**
   Query object or property -> subject for `what shape has four sides?`,
   `what is ice?`, and similar probes. The forward row
   `A square has four sides` must yield `square`; `Frozen water is ice` must
   yield `ice` or `frozen` when the question asks from the object side.
   Tests: `test_reverse_lookup_square`, `test_reverse_lookup_ice`.
5. **Use a strict answer-slot metric beside the substring metric.**
   A hit should be the expected object occupying the answer slot, not a token
   anywhere in a contradictory sentence. Mark the round-6 yellow+blue reply as
   a fragile/contradictory hit. Tests:
   `test_strict_answer_slot_rejects_incidental_expected_token`,
   `test_report_marks_fragile_hits`.
6. **Balance sibling facts in the curriculum.**
   Every round that teaches one color-mixing sibling must teach the other with
   equal repetition and explicit subject labels. Tests:
   `test_curriculum_balances_color_siblings`,
   `test_curriculum_does_not_teach_one_sibling_without_the_other`.
7. **Add causal paraphrases with extractable verbs.**
   Teach and test `Freezing water causes ice`, `Cold causes water to freeze`,
   and `When water freezes it becomes ice`; the causal query must select the
   caused object, not a state row from another subject.
   Test: `test_causal_water_freezing_paraphrases`.
8. **Add probe coverage for fire/red and snow/white.**
   These facts are taught but unprobed, so the 14-probe set cannot measure the
   full color-fact block. Test: `test_probe_set_covers_taught_color_facts`.
9. **Keep the diagnostic test for this harness.**
   `tests/test_nanlm_plateau_analysis.py` verifies all seven causes, the
   read-only DB behavior, 71.4% plateau detection, bounded text output,
   JSON/Markdown generation, and the leak guard on the ablation plan.

## Boundaries

- The 71.4% plateau is a recorded result, not a claim that the current checkout
  will reproduce it.
- The current code snapshot changed after the run; `current_code_recoverable`
  in the JSON is separate from the recorded cause.
- The DB is a final/live ledger, not a historical per-round backup. Per-round
  row availability is approximated from each row's source teaching sentence
  and its first taught round.
- The probe hit metric is the same expected-token substring check used by the
  run script. Fragile hits are reported explicitly.
- No package code, package documentation, run JSON, or training state was
  modified by this diagnostic.
- No ablation was executed and no fixed accuracy is claimed.
