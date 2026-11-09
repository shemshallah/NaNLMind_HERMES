# Gemma initial training/tool-learning report

Date: 2026-11-08. Model: `gemma4:31b-cloud` through the local Ollama
OpenAI-compatible endpoint.

This report aggregates the first live runs. It is evidence about tool use and
the plugin's derived learning state, not a canonical merge, proof or general
language-model benchmark.

## 1. Open-ended tool-learning session (NANNET-paired)

Command:

```sh
python3 scripts/gemma_tool_learning_session.py --turns 12 \
  --state-dir /data/data/com.termux/files/usr/tmp/nanlm-gemma-tool-learning
```

Setup:

- a real loopback `PairingAuthority` and HTTP authority;
- installation paired with `universe:read` + `training:execute`;
- authorization mode in every scoped tool result: `nannet`;
- canonical merge/promotion disabled; no merge or promotion tool was called;
- the model chose and executed real plugin handlers via tool calls.

Result:

- **12/12 tool calls passed**, **0 failures**, **9 distinct tools used**;
- tool sequence: session context, MTH function-tree map/audit, vector catalog,
  runtime, two vector leaf creations, `connect` pairing, `add` pairing, leaf
  verification, two free-chat turns, learning-status inspection.

| # | tool | result |
|---:|---|---|
| 0 | `nanlm_session_context` | 7 permanent documents loaded |
| 1 | `nanlm_mth_tree` (`map`) | 91/91 kernels linked, 91/91 images verified |
| 2 | `nanlm_mth_catalog` (`vector`) | vector family discovered |
| 3 | `nanlm_mth_runtime` | aarch64 host, portable core + gawk detected |
| 4 | `nanlm_mth_leaf_create` `gemma-v1` | immutable vector leaf created |
| 5 | `nanlm_mth_leaf_create` `gemma-v2` | immutable vector leaf created |
| 6 | `nanlm_mth_pair` (`connect`) | new relation hash leaf |
| 7 | `nanlm_mth_pair` (`add`) | new vector hash leaf |
| 8 | `nanlm_mth_leaf_verify` | `nan_executable=true`, output `L` |
| 9 | `nanlm_chat` teaching turn | layered observation returned |
| 10 | `nanlm_chat` question turn | second layered observation |
| 11 | `nanlm_learning_status` | turns/rows/scratch/ensemble inspected |

### What Gemma learned during the session

The immediately preceding 12-turn attempt is the useful learning signal:

- On turn 4 and turn 6 of that earlier run, Gemma supplied short vectors to
  `nanlm_mth_leaf_create`; both calls failed with
  `vector must contain exactly 256 values`.
- The schema and payload were then improved with an explicit
  `{"fill": <finite number>}` form (expanded across exactly 256 elements; no
  silent short-array broadcast).
- On the next full run, Gemma used the fill form and completed **both** leaf
  creations, then proceeded through pairing and verification with no failures.
- It also repeated the less-obvious steps (tree map, runtime, learning-status
  inspection) without prompting between each call.

This is qualitative evidence of tool-use learning; a multi-session series with
before/after quizzes is being added to measure it quantitatively.

### System state after the session

From `state_dir/language/layered_learning.json` and
`state_dir/mth_leaves/index.json`:

- free-chat layered turns: **2**; ensemble observations: **20**;
- active heads: **227 / 255** (coverage 0.890196); max head share 0.015861;
- immutable leaves: **148** total — 53 hubs, 91 kernel leaves, 3 vector leaves,
  1 relation leaf; 93 aliases;
- function tree: 91/91 kernels, 44 family hubs, 0 missing/orphans/digest
  mismatches, 91/91 leaf images executable;
- canonical merge: **not called, not enabled**.

## 2. Free-chat learning smoke (3 turns)

Artifact: `docs/GEMMA_FREE_CHAT_SMOKE.json`.

| turn | user | layers | relations | scratch | active heads | coverage |
|---:|---|---:|---:|---:|---:|---:|
| 1 | A river is water. | 10 | 1 | 10 | 177 | 0.694 |
| 2 | Water can flow, and rivers carry water. | 10 | 1 | 20 | 222 | 0.871 |
| 3 | What is a river? | 10 | 2 | 30 | 240 | 0.941 |

Learning status: turns 3, rows 4, 10 layers per turn. Head coverage rose across
the three turns. The question turn was not treated as a taught fact by the
deterministic parser; reflection may still contribute neutral evidence.

No native relation-train runner was configured, so the native topology update
was not performed. The layered scratch/concept path is real; the native path
correctly reported its limitation rather than fabricating a commit.

## 3. Architecture review

Artifact: `docs/GEMMA_ARCHITECTURE_REVIEW.md` / `.json`.

- 7,373 prompt tokens, 1,011 completion tokens.
- Gemma specifically endorsed the immutable leaf model, hub ancestry checks,
  constrained `left::right` pairing, WordNet HW4 as read-only evidence, and the
  closed canonical merge gate.
- It identified remaining gaps: live authenticated NaNROUTER, merge-gate
  implementation, broader N16I differential coverage, and a clearer L0-L9
  scratch-to-head mapping.

## 4. Multi-session training series (first live run)

Artifacts: `docs/GEMMA_TRAINING_SERIES_RESULTS.md` / `.json`.

Setup: 3 sessions × 8 turns, same persistent state directory, loopback NANNET
paired, canonical merge/promotion disabled.

| Session | Stage | Calls | Failed | Distinct tools | Required coverage | Turns |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | explore | 6 | 0 | 6 | 100% | 6 |
| 2 | data leaves | 5 | 0 | 3 | 100% | 5 |
| 3 | free chat | 4 | 0 | 3 | 100% | 5 |

Aggregate: **15/15 calls, 0 failures, 12 distinct tools, 100% required-task
coverage**. The before/after written quiz was **76.1% → 76.1% (delta 0.0)**.
The first run was flat because it had an experiment-design bug: the after-quiz
was an isolated no-tools call without the session transcript, tool names or
observed results. It answered with generic names such as `data_leaf`,
`function_tree` and `pairing`. A direct probe confirmed the diagnosis: with the
same quiz and no tools, the score goes **76.1% → 100.0%** when the observed
session journal is included. See
[GEMMA_JOURNAL_QUIZ_PROBE.md](GEMMA_JOURNAL_QUIZ_PROBE.md).

The corrected journaled 3-session series then measured **76.1% → 100.0%
(+23.9 points)** with **15/15 tool calls, 0 failures, 12 distinct tools and
100% required-task coverage**. This is in-context transfer from measured session
experience, not a model-weight update or a general-learning proof.

The plugin's derived state grew across the series:

| metric | before | after | delta |
|---|---:|---:|---:|
| turns | 0 | 2 | +2 |
| rows | 0 | 1 | +1 |
| scratch records | 0 | 20 | +20 |
| concepts | 0 | 34 | +34 |
| occupied concept pairs | 0 | 436 | +436 |
| leaves | 144 | 148 | +4 |
| linked kernels | 91 | 91 | 0 |

The +4 leaves are the two session-2 vectors plus their connect and add result
leaves. The function tree was already 91/91 and stayed complete. The add
pairing now uses the unified N16I general-operand path.

## 5. Honest boundaries

- Local loopback NANNET, not production Koyeb.
- No canonical BS0/MC0 merge or promotion.
- No proof certificate; `native_verified`/fixture PASS are contract evidence.
- Only four vector kernels have a general operand dispatcher; all other kernels
  are fixture-verifiable and now also N16I-dispatchable as guest images.
- A multi-session before/after series is needed to quantify whether Gemma's
  tool-use improves and whether system learning compounds across sessions.

## Artifacts

- `docs/GEMMA_TOOL_LEARNING_SESSION.md` / `.json`
- `docs/GEMMA_FREE_CHAT_SMOKE.json`
- `docs/GEMMA_ARCHITECTURE_REVIEW.md` / `.json`
- `docs/MTH_FUNCTION_TREE.md`
- `docs/ARCHITECTURE_GAPS.md`
