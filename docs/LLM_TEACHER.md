# Provider-universal LLM teacher

The teacher is not Gemma-specific. The same bounded record contract works with
any LLM Hermes already uses.

## Providers

| `teacher_provider` | Transport | Default |
|---|---|---|
| `hermes` | Hermes' configured `ctx.llm.complete` | yes |
| `ollama` | local Ollama `/api/generate` | `gemma4:31b-cloud` |
| `openai` | any `/v1/chat/completions` endpoint | caller-supplied |

`teacher_endpoint` and `teacher_model` override the provider defaults. The
OpenAI-compatible transport reads `LLM_TEACHER_API_KEY` through
`hermes_secrets.get_secret`; no key is written to plugin config or logs.

## Record contract

The teacher returns at most 64 records:

```json
[
  {
    "subject": "copper",
    "predicate": "is",
    "object": "metal",
    "definition": "a reddish metal",
    "layer": 5,
    "confidence": 0.9,
    "evidence": "teacher"
  }
]
```

The parser accepts a bare array or `{"records":[...]}`. Records are validated,
bounded and written as evidence into the L0-L9 scratch learner. They never
commit canonical NaNLMind state.

## Parallel CLI

`scripts/llm_teacher_run.py` runs the same teacher contract as a separate
process:

```sh
python scripts/llm_teacher_run.py --provider ollama \
  --model gemma4:31b-cloud --topic "basic metals" --count 3
```

`--cycles 0 --interval 60` runs it as a bounded parallel teaching loop. Add
`--approve` to apply the local promotion gate after each batch. A verified
local run produced 3 Gemma records and 10 L0-L9 scratch records with
`canonical_training_committed=false`.

## Parallel use

The teacher can run as its own process against the same state directory, or as
the `nanlm_teacher_teach` tool. The runner ring and the teacher are independent:

```text
teacher LLM -> validated records -> HERMES L0-L9 scratch
                                      |
                                      v
runner ring HW5:0 -> NANNET -> active runner / portal
```

The teacher's output is not ground truth. The conservative support/confidence/
approval gate remains between a teacher record and a local head-register leaf.
