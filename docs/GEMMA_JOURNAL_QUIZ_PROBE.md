# Gemma journal quiz probe

Date: 2026-11-08. Model: `gemma4:31b-cloud`. No tools were available during the
quiz.

## Why this probe exists

The first 3-session series scored the same written quiz (76.1%) before and
after. Inspection showed the after-quiz was an isolated no-tools call with no
session transcript or learning journal. Both runs answered with generic names
such as `data_leaf`, `function_tree` and `pairing`, not the exact tools they had
actually called. That is an experiment-design bug, not evidence that the model
cannot learn from the session.

## Probe

The same six-question quiz and the same scoring rubric were used. The only
change was inserting a compact **observed session journal** as a system message:

- exact tools used in the completed series;
- successful/failed call counts;
- 91/91 linked kernels and 91/91 verified images;
- `nanlm_mth_leaf_create` with `kind=vector` and payload key `fill`;
- `nanlm_mth_pair` operators `connect` and `add`;
- `nanlm_mth_leaf_verify` argument `leaf_id` and success field
  `nan_executable`;
- unified vector execution mode `n16i_guest_interpreter`;
- chat/learning observations: tools, turns, rows, scratch records, concepts,
  occupied pairs and leaves.

No tools were called during the probe. The journal contains only observed
session data, not the quiz rubric.

## Result

| Run | Score |
|---|---:|
| before series | 76.1% |
| after series, no journal (original bug) | 76.1% |
| after series, with observed journal | **100.0%** |

The journal run answered every question with the exact tool names and argument
names, including `nanlm_mth_leaf_create` + `fill`, `nanlm_mth_pair` + `add`,
`nanlm_mth_leaf_verify` + `nan_executable`, and the three chat/status tools.

Artifact: `docs/GEMMA_JOURNAL_QUIZ_PROBE.json`.

## Interpretation

- The model can use session experience when that experience is actually in its
  context. The original flat score was caused by withholding the experience.
- This is an in-context learning result, not a model-weight update and not a
  general-learning proof.
- The corrected training series must carry a bounded session journal into later
  sessions and the after-quiz, and should add discovery questions answerable
  only from observed state.
- The plugin's own state also changed in the original run (turns 0→2, scratch
  0→20, concepts 0→34, pairs 0→436, leaves 144→148); the renderer bug that
  displayed zeros was fixed.

## Boundaries

- One probe, one model, one state directory, one local loopback NANNET setup.
- No canonical merge/promotion, NANNET route/lease or proof claim.
- The next multi-session run must show the journal effect prospectively, not
  only in this post-hoc probe.
