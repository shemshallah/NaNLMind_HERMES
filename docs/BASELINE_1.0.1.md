# Baseline experiment 1.0.1

Run date: 2026-11-06
Model: `gemma4:31b-cloud` via Ollama OpenAI-compatible endpoint
Native trainer: local `nanlm-universal` `logic-train` + `math-train`
Mode: model-planned logic → chat → words → chat → sentences → chat

## Boolean foundation gate

Every training round must pass the native full-Boolean gate before language
training starts:

| check | result |
|---|---|
| 256 heads × all 16 two-input functions × 4 truth rows | 1,024 / 1,024 |
| half/full adder, majority, multiplexer, depth-8 composition | 32 / 32 |
| word-after-logic probe | 10 / 10 |
| distinct learned functions | 16 / 16 |
| persisted foundation checkpoint | `boolean_foundation.nmg` |

The gate is enforced by the trainer; a failed or partial foundation aborts the
round. The language result below is the separate byte-bit `math-train` metric,
so it must not be conflated with the Boolean result.

## Standard run

`records=12`, `epochs=1`, `samples/head=128`

| round | model-selected topic | byte held-out | majority baseline | Boolean gate |
|---|---|---:|---:|---|
| logic | L0-L1 Boolean Foundation: 1,024 Truth-Table Exactitude and Depth-8 Circuit Composition | 0.631561 | 0.634766 | 1,024/1,024 |
| words | L2 Lexical Mapping: 1,024-Token Repetition and Word-Completion Probe | 0.575439 | 0.570312 | 1,024/1,024 |
| sentences | L3 Narrative Sequence: 128-Sentence Choice and Temporal Rail Stability | 0.622375 | 0.607422 | 1,024/1,024 |

## Higher-budget run

`records=12`, `epochs=3`, `samples/head=512`

| round | model-selected topic | byte held-out | majority baseline | Boolean gate |
|---|---|---:|---:|---|
| logic | L0-L1 Boolean Foundation: 1,024 Truth-Table Exactitude and Depth-8 Circuit Composition | 0.630570 | 0.630102 | 1,024/1,024 |
| words | L2 Lexical Mapping: 1,024-Token Repetition and Word-Completion Probe | 0.619014 | 0.615663 | 1,024/1,024 |
| sentences | L3 Narrative Sequence: 3-Token Sentence Choice and Temporal Rail Transition | 0.566423 | 0.560924 | 1,024/1,024 |

## Interpretation

- **Full Boolean learning is real and reproducible.** All 1,024 truth-table
  checks, all 32 composed-gate checks, and all 10 word-after-logic probes pass
  before language training is allowed.
- **The byte task still does not show robust language learning.** Standard-run
  deltas are −0.0032, +0.0051, and +0.0150; higher-budget deltas are +0.0005,
  +0.0034, and +0.0055. These are small and inconsistent, and are not
  sufficient to claim general language competence.
- The chat checkpoint receives the gate result, the model plan, and the byte
  metrics. It discusses what the evidence supports rather than claiming the
  round is proof.
- This is a **readiness and reproducibility baseline**, not a claim of AGI or
  general language learning.

## Commands

```sh
export NANNET_AUTH_URL=https://nannet.apps.mistral.run
export HERMES_GATEWAY_NO_SUPERVISE=1
python experiments/baseline_experiment.py \
  --native-command /absolute/path/to/build/nanlm-universal \
  --model-url http://127.0.0.1:11434/v1 \
  --model gemma4:31b-cloud \
  --state-dir experiments/baseline_gate_state \
  --out experiments/baseline_1.0.1.json --records 12

# Higher-budget variant
python experiments/baseline_experiment.py \
  --native-command /absolute/path/to/build/nanlm-universal \
  --model-url http://127.0.0.1:11434/v1 \
  --model gemma4:31b-cloud \
  --state-dir experiments/baseline_gate_high_state \
  --out experiments/baseline_1.0.1_high.json \
  --records 12 --epochs 3 --samples 512
```


## Measurement boundary after resolution-aware trainer

This 1.0.1 table is a historical byte-bit `math-train` baseline. It is not a
word, lexical, sentence, paragraph or narrative score. The near-majority values
and small round-to-round deltas were a central reason to add the separate
`relation-train` objective.

New structured sessions report L0-L9 relation-polarity metrics on frozen
topic-specific challenge sets, including pre-training score, post-training
score, within-run delta, balanced accuracy, majority baseline and cross-run
delta when the test digest matches. Free-form chat reports native mutation
receipts (changed heads, mirror-pair updates, hyperedges and state digests);
it does not fabricate held-out accuracy when no test examples are present.

Do not compare the historical byte metric directly with the new
resolution-specific relation metrics. They are different objectives and
different evaluators.
