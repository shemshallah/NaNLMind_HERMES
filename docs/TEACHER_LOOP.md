# Teacher loop

Hermes chat is the **orchestrator and explainer**. It is not automatically the
student, and its prose is not a weight update.

The useful pattern is:

```text
teacher model (Gemma/Qwen/other)
        |
        | structured examples, not self-graded claims
        v
deterministic validator + held-out split
        |
        | accepted N16TXT / typed records
        v
NaNLMind student (native heads + memory)
        |
        | exact Boolean gate, held-out language metrics
        v
feedback report: errors, deltas, rejected examples
        |
        v
teacher proposes the next small round
```

## 1. Teacher output must be data

Ask the teacher for a schema, not free-form praise:

```json
{
  "task": "boolean_truth",
  "op": 6,
  "a": 1,
  "b": 0,
  "target": 1,
  "reason": "XOR of 1 and 0 is 1"
}
```

For language:

```json
{
  "task": "word_relation",
  "subject": "penguin",
  "predicate": "can",
  "object": "swim",
  "label": true,
  "source": "curriculum"
}
```

The validator must use a deterministic oracle where one exists:

- Boolean labels are checked against all 16 truth tables.
- Arithmetic is recomputed.
- Word relations are checked against an explicit fact store or source corpus.
- A teacher's confidence is metadata, never a pass signal.

Reject invalid labels instead of training on them. The rejection count is part
of the lesson.

## 2. Keep train, validation, and test separate

```text
teacher round 1 -> train.jsonl
teacher round 2 -> validation.jsonl
frozen challenge -> test.jsonl
```

The native `math-train` path already has an internal 80/20 split for its byte
task. That is **not** a substitute for a separate teacher/held-out set. The
Boolean foundation has a different rule: it is a finite coverage gate, so all
1,024 truth rows must be checked, and the population must contain all 16
distinct learned functions. Its failure signal is incomplete observations,
failed heads, or collapsed diversity.

## 3. Feedback must contain errors, not answers

Send the teacher a compact report:

```json
{
  "round": 3,
  "gate": {"passed": true, "distinct_rules": 16},
  "heldout": {"accuracy": 0.63, "majority": 0.63, "bpc": 5.2},
  "errors": [
    {"task": "word_relation", "subject": "penguin", "expected": false,
     "student": "unknown", "count": 7}
  ],
  "rejected": 2,
  "instruction": "Propose a focused round that addresses these errors. Do not repeat the test items."
}
```

Do not send the test answers. Do not let the teacher declare the student
passed. The deterministic gate and held-out evaluator own that decision.

## 4. Stop rules

Stop or roll back when:

- the Boolean gate fails;
- held-out loss/BPC does not improve for a fixed number of rounds;
- validation and test diverge;
- a teacher round largely duplicates previously rejected records;
- the model changes only its prose, not its measured behavior.

## 5. ChatGPT/Hermes wiring

The default `nanlm_chat` path must remain a chat path. A teacher round belongs
behind an explicit command or tool, for example:

```text
/nanlm teach logic 3 rounds
```

That tool would:

1. ask the active teacher for structured examples;
2. validate and deduplicate them;
3. write an N16TXT corpus with a separate held-out set;
4. call the native trainer, which enforces the full-Boolean gate;
5. evaluate and store the result;
6. feed only the error report back to the teacher;
7. stop at the configured budget.

This is a feedback loop over **data and measurements**, not over the teacher's
hidden states. That is why it works across different pretrained providers.

## 6. Current boundary

Today Hermes already has the pieces for a manual loop:

- `nanlm_generate_curriculum` — teacher-generated records;
- `nanlm_training_pipeline` / `nanlm_logic_session` — explicit training;
- `nanlm_math_train` — mandatory full-Boolean gate, then native byte-bit
  training;
- `nanlm_status` — checkpoints and memory;
- `nanlm_chat` — explanation and reflection.

It does **not** yet automatically run the validate → train → evaluate → ask
teacher loop. The next implementation should add that orchestration with a
strict schema and held-out evaluator, not let chat silently train.
