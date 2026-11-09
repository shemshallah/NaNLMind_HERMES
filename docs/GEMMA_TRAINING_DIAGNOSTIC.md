# Gemma training-series diagnostic

The first 3-session run recorded in
[GEMMA_TRAINING_SERIES_RESULTS.md](GEMMA_TRAINING_SERIES_RESULTS.md) finished
all tool tasks, but its before/after written quiz stayed at **76.1% (delta
0.0)**. That flat number is a measurement-design result, not evidence that
Gemma cannot learn from tool use.

## Why the written quiz delta was zero

1. **The after-quiz was isolated.** It was a separate no-tools request issued
   after the sessions. Its prompt contained no session transcript, no tool
   calls and no measured results.
2. **No learning journal was carried forward.** The series wrote a large JSON
   result, but the model never received a compact record of what it had just
   done. Later sessions only saw a short coverage summary.
3. **Both runs answered with generic names.** Before and after answers were
   identical and used labels such as `data_leaf`, `function_tree` and
   `pairing` instead of `nanlm_mth_leaf_create`, `nanlm_mth_tree` and
   `nanlm_mth_pair`. A quiz that tests only prior declarative vocabulary cannot
   detect operational experience that was left out of the prompt.

Meanwhile the plugin's own state changed in that run (turns 0->2, rows 0->1,
scratch records 0->20, concepts 0->34, occupied pairs 0->436, leaves 144->148,
linked kernels 91->91). Those are measured state changes, not a claim of model
weight learning.

## Corrected measurement design

### 1. Bounded session journal

`nanlmind_hermes/gemma_learning_journal.py` stores one JSONL file under the
persistent state directory. `SessionJournal.append_session(...)` records, per
session:

- exact tool names used, with call/failure counts;
- successful, failed and invalid-argument call counts;
- a small set of observed fields extracted from tool results
  (`execution_mode`, `n16i_used`, linked/expected/verified kernel images,
  leaves, turns, rows, scratch records, concept counts and density).

Entries are bounded, large payloads are truncated, and secret-bearing keys and
common token shapes are redacted. `load()` and `tail(limit)` read the retained
window; `build_experience_block(limit=...)` produces a compact prompt block with
the exact tool names and observed values.

### 2. Journal in the next session and the after-quiz

Append after every session, then insert
`build_experience_block(limit=3)` into later session prompts and into the
after-series quiz context. The journal contains observed session data only, not
the quiz rubric. This turns the after-quiz from an isolated recall test into a
test of using available experience.

Keep a separate no-journal after-quiz if retrieval from context must be
distinguished from changes in the model's unaided answers.

### 3. Discovery questions answerable only from observed state

`static_discovery_questions()` asks for values that permanent documentation
cannot supply, including:

- the exact native `execution_mode` and whether N16I was used;
- the leaf-level execution mode for the vector add result;
- linked/expected/verified kernel-image counts;
- leaf count before/after and leaf delta;
- scratch records and scratch delta after teaching;
- concept statements/concepts/occupied pairs after teaching;
- final learning turns/rows and leaf verification fields.

`dynamic_expected(full_series_result)` derives the expected values from the
measured run. `score_discovery_answers(answers, expected)` scores only keys that
the artifact actually measured; unmeasured keys are reported as gaps rather
than inferred. Expected values are operator evidence and must not be inserted
into the model prompt.

### 4. Repeated same-state sessions

The series must repeat at least three sessions against the same persistent state
directory. Compare the first and last session on:

- exact tool-name rate and generic-name rate;
- required-task coverage and turns-to-complete;
- invalid-argument calls and recovered failures;
- discovery-quiz score with the journal present.

Without the first-vs-last comparison, a single successful run cannot distinguish
session experience from prompt wording or model variance.

## Diagnostic command

```bash
python3 scripts/gemma_series_diagnostics.py \
  --input docs/GEMMA_TRAINING_SERIES_RESULTS.json \
  --output /tmp/series-diagnostic.json
```

The script makes no model call and fabricates no answers. It reads a saved
series result, explains the written-quiz delta from prompt/answer evidence,
builds the discovery quiz implied by the observed state, lists measurement
gaps, and writes JSON plus a sibling Markdown report. Use it before the next
live series to confirm what must be measured.

## Boundaries

- One model, endpoint, state directory and local loopback NANNET setup per run.
- The diagnostic reports a saved run; it does not establish general learning.
- No canonical merge, promotion, commit, lease or proof claim.
- The next series must demonstrate the journal effect prospectively; a
  post-hoc observation is not a substitute for that measurement.
