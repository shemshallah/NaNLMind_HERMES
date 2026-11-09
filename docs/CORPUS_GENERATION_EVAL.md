# corpus generation evaluation

`scripts/corpus_generation_eval.py` evaluates a **deeply trained native NaNLM
student** on held-out generation and explanation prompts. The student reply is
always produced by the native student path; the optional Gemma judge only labels
an already-written reply.

The report is a diagnostic record. It is not a benchmark of general language
competence and it does not prove that the student understands the prompts.

## Quick start after deep training

The companion deep-training script stores its student here by default:
`~/.local/share/nanlmind-hermes/corpus-deep-training`. Run the evaluation from
the repository root:

```bash
python3 scripts/corpus_generation_eval.py \
  --state-dir "${NANLM_CORPUS_STATE_DIR:-$HOME/.local/share/nanlmind-hermes/corpus-deep-training}" \
  --corpus experiments/corpus/trainable \
  --judge \
  --nll \
  --output docs/CORPUS_GENERATION_EVAL.json
```

That uses the built-in prompt set, asks the optional Gemma judge for rubric
labels, and also computes next-token NLL/perplexity. The JSON report and its
Markdown companion are written atomically:

- `docs/CORPUS_GENERATION_EVAL.json`
- `docs/CORPUS_GENERATION_EVAL.md`

For a deterministic-only run with no judge:

```bash
python3 scripts/corpus_generation_eval.py \
  --state-dir "$HOME/.local/share/nanlmind-hermes/corpus-deep-training" \
  --corpus experiments/corpus/trainable \
  --output docs/CORPUS_GENERATION_EVAL.json
```

With a custom probe file (`probes.jsonl` is an example path; the format is
described below):

```bash
python3 scripts/corpus_generation_eval.py \
  --state-dir "$HOME/.local/share/nanlmind-hermes/corpus-deep-training" \
  --prompts probes.jsonl \
  --corpus experiments/corpus/trainable \
  --judge \
  --output docs/CORPUS_GENERATION_EVAL.json
```

## Inputs

### Student state directory

`--state-dir` must point at the same durable state used by the native student:

- `language/temporal_mirror.json` — the trained `TemporalMirrorLM` counts and
  vocabulary;
- `language/temporal_register.json` — the temporal register sidecar, when
  present;
- `language/layered_learning.json` — the layered learner state, when present;
- `universe/mathematical_memory.sqlite3` — relation/observation memory, when
  present.

The report records which of these files exist, plus temporal vocabulary and
exposure counts when the model file is readable. A missing temporal model is
reported rather than hidden; an untrained or empty state can still run, but the
result is not evidence of trained generation.

### Built-in prompts

If `--prompts` is omitted, the script uses a fixed built-in set covering:

- explanation (two prompts);
- comparison;
- causal explanation;
- counterfactual;
- short story;
- definition (two prompts);
- arithmetic (two prompts).

These prompts are intentionally small and inspectable. They are not a
statistically representative benchmark. For a genuine held-out check, provide
prompts that were not used by training and inspect the corpus-overlap note.

### JSONL or JSON probe file

Each non-empty JSONL line is a JSON string or object. An object may contain:

```json
{
  "id": "explain-river",
  "category": "explanation",
  "prompt": "Explain why a river can carry water.",
  "keyword_groups": [
    ["river"],
    ["water"],
    ["carry", "carries", "flow", "flows"]
  ]
}
```

Rules:

- `prompt` is required; `question` and `message` are accepted aliases.
- `id` and `category` are optional. Missing IDs become `prompt-001`, etc.
  Duplicate IDs are rejected.
- `keyword_groups` is a list of groups; each inner list is a set of acceptable
  alternatives. A plain `keywords` list is treated as one group per keyword.
  `expected` is accepted as another alias.
- A `.json` file may be a list of entries or an object containing a `prompts`,
  `probes`, `rows` or `items` list.
- Concatenated JSON objects on one JSONL line are accepted.

Plain strings are JSON strings, not bare text lines:

```jsonl
{"id":"one","prompt":"What is a river?","keywords":["water","flow"]}
"Tell a short story about a river."
```

### Corpus

`--corpus` is optional. It supplies the reference text for:

- copied-sentence detection;
- corpus-relative novel 3-grams;
- reference excerpts passed to the judge (bounded).

A corpus may be a single `.txt`, `.jsonl`, `.json` or `.md` file, or a directory
searched recursively for those suffixes. `.txt`/`.md` files contribute one
sentence per non-empty line. JSON/JSONL records contribute `text`, `sentence`,
`statement` or `content`, plus common containers such as `statements`,
`sentences`, `records`, `rows`, `train`, `rounds` and `baseline`.

Use the same trainable directory that produced the student when possible; the
checked-in example is `experiments/corpus/trainable`.

The loader does **not** filter splits. If the supplied directory contains
train, validation and test rows, all of them become the reference. For strict
"copied training sentence" semantics, pass a train-only corpus. The report
labels the metric according to the corpus you supplied, not according to an
assumed split.

## How the student is called

The default `NativeChatStudent` registers the plugin on a minimal context and
invokes the `nanlm_chat` tool with:

```json
{"message": "...", "session_id": "corpus-generation-eval", "probe": true}
```

`probe=True` is the read-only evaluation path: it does not merge memory, write
conversation transcripts, update layered scratch state, or save the temporal
model. The script records the returned `probe` and `state_mutated` receipt for
every row.

The native `TemporalMirrorLM` generates the reply. The judge is called only
after the reply exists and is never allowed to replace it. No LLM writes the
student reply.

## Deterministic metrics

All deterministic metrics are computed from the emitted text and, when
supplied, the reference corpus. They are repeatable for the same reply and
corpus.

| metric | computation | interpretation |
|---|---|---|
| `empty` / aggregate `empty_rate` | reply is whitespace-only | Detects no-output cases, not bad answers. |
| `characters` | `len(reply)` | Includes spaces and punctuation. |
| `tokens` | case-folded word/number tokens | Punctuation is not a word token here. |
| `unique_tokens` | distinct word/number tokens | Vocabulary size of this reply. |
| `unique_token_ratio` | unique / total | Higher can mean more varied or simply shorter text. |
| `repeated_token_ratio` | `1 - unique / total` | A crude repetition signal. Common function words can raise it. |
| `max_token_repetition` | highest count of one token | Detects loops and emphatic repetition. |
| `max_trigram_repetition` | highest count of one 3-gram | Detects repeated phrases. |
| `copied_training_sentence` | exact corpus sentence appears in reply | Available only with `--corpus`. |
| `copied_training_sentences` | up to five matching corpus sentences | Display evidence for the copy flag. |
| `novel_3grams` | reply 3-grams not in prompt or corpus | Count of unfamiliar sequences. |
| `novel_3gram_ratio` | novel unique 3-grams / reply unique 3-grams | Surface novelty; not truth or usefulness. |
| `novel_reference` | `prompt_only` or `corpus+prompt` | States what novelty was measured against. |
| `prompt_in_corpus` | exact prompt line exists in corpus | Exact match only; no paraphrase detection. |
| `keyword_coverage` | matched groups / total groups | Presence of required alternatives, not correctness. |
| `keywords_matched` / `keywords_total` | group counts | The denominator is explicit. |
| `missing_keywords` | groups with no matching alternative | Helps inspect what was absent. |
| `next_token_nll` | prompt-conditioned teacher-forced response tokens | Optional; see below. |

### Copied-sentence detection

A corpus sentence counts as copied when its word-token sequence occurs
contiguously inside the reply and it has at least **3 word tokens**. Matching is
word-token based, so whitespace, punctuation and case differences do not hide a
copy.

It does not detect:

- paraphrase or synonym substitution;
- a copied two-word phrase;
- a sentence assembled from corpus fragments;
- copying from training data that was not supplied in `--corpus`.

### Novel 3-grams

The reference set is the union of:

- 3-grams from the prompt itself;
- 3-grams from every supplied corpus sentence, when `--corpus` is present.

A reply 3-gram outside that set is counted as novel. The per-prompt ratio uses
unique reply 3-grams. The aggregate `novel_3gram_ratio_micro` sums novel counts
and divides by summed reply 3-gram counts.

Novelty here means only "not in the provided references". A novel sequence may
be ungrammatical, false, a recombination of known words, or a useful new
explanation. It is not a quality score.

### Keyword coverage

Keyword groups let one expected concept have several acceptable surface forms,
for example `["flow", "flows", "flowing"]`. Matching is case-folded and
word-boundary based. There is no stemming, synonym expansion, or semantic
matching.

Coverage does not check whether:

- the keyword appears in the correct clause;
- the sentence negates the keyword;
- the answer contradicts itself;
- a keyword is used as a name rather than a concept.

A reply can have coverage `1.0` and still be wrong. The per-prompt reply is the
evidence; coverage is only an index into it.

## Optional next-token NLL/perplexity

`--nll` loads `TemporalMirrorLM` from the state directory in read-only mode and
scores the reply token by token:

1. tokenize the prompt and reply with the learner's tokenizer;
2. condition on `[BOS] + prompt tokens + earlier reply tokens`;
3. sum `-log(model._prob(context2, context1, target))` over reply tokens;
4. report mean NLL and `exp(mean NLL)`, capped at `exp(50)` for display.

This calls only the model's probability method. It never calls `observe`,
`generate`, `save`, or a tool that writes state.

NLL/perplexity is a fit measure for this particular student. It is **not**
answer accuracy:

- low NLL can mean the reply is memorized or generic;
- high NLL can mean the reply is novel, surprising, or simply wrong;
- the number depends on tokenization and the model's own sparse counts;
- it does not compare against a baseline or a human judgment.

## Optional Gemma judge

`--judge` sends the prompt, the already-written reply, and up to 40 corpus
excerpts to an Ollama OpenAI-compatible endpoint. Defaults:

- URL: `http://127.0.0.1:11434/v1/chat/completions`
- model: `gemma4:31b-cloud`

Override with `--url` and `--model`. The judge request uses temperature `0.0`
and asks for exactly one JSON object with these keys:

```json
{
  "answers_prompt": 0,
  "coherent": 0,
  "novel_combination": 0,
  "unsupported_claim": 0,
  "notes": "short diagnostic string"
}
```

Fields:

- `answers_prompt` — the reply addresses the requested task;
- `coherent` — the reply is internally readable and relevant, not merely
  fluent;
- `novel_combination` — the reply combines known material in a non-copied way;
- `unsupported_claim` — the reply asserts something the supplied references do
  not support;
- `notes` — short diagnosis, not a score.

The parser accepts fenced JSON, but validates the required keys and binary
values. A malformed rubric becomes an `error` for that row and is counted in
the summary; it is never silently converted into a positive score.

The judge is **diagnostic, not ground truth**. It can be wrong, biased, prompt-
sensitive, or fooled by fluent unsupported text. A judge `unsupported_claim=0`
does not prove the reply is true, and `answers_prompt=1` does not prove the
answer is correct.

## Report contents

The JSON report uses schema `nanlm.corpus-generation-eval/1` and contains:

- `state_dir`, `state` metadata and the prompt/corpus source metadata;
- `judge` configuration and `nll` request/error status;
- `state_probe` before/after fingerprints;
- `rows`, one per prompt:
  - prompt, category, keyword groups;
  - exact student reply;
  - student receipt (`provider`, `model`, `probe`, `state_mutated`, optional
    temporal frame summary);
  - all deterministic metrics;
  - judge rubric or judge error;
- `summary`:
  - empty count/rate, mean length, unique-token and repetition metrics;
  - mean and micro novel 3-gram ratios;
  - copied-sentence count/rate when a corpus was supplied;
  - keyword coverage counts;
  - exact prompt/corpus matches;
  - read-only probe receipt count;
  - judge counts/rates and judge errors;
  - NLL tokens, mean NLL and perplexity when requested;
  - per-category aggregates;
- `boundaries`: read-only probe, no LLM-written reply, judge diagnostic only,
  no held-out proof, no training, no canonical commit, no merge/route/proof
  claim.

The Markdown report contains the same aggregate summary, a per-category table,
every prompt and reply, row metrics, judge output, and the boundary notes.

## Read-only evidence

Before probing, the script hashes the durable learner files it can identify:

- the temporal model and layered/register files under `language/`;
- `universe/mathematical_memory.sqlite3`;
- conversation transcripts;
- paged scratch-register pages;
- native `.nmg` chart files.

After all prompts it hashes them again and records whether the fingerprint is
unchanged. Changed paths are listed. This is supporting evidence for
`probe=True`; it is not a proof about every possible in-process cache or every
file outside the selected set. The per-row `probe=True` and
`state_mutated=False` receipts are reported separately.

## What the metrics do and do not prove

### They can help answer

- Did the student emit anything for the prompts?
- How long and repetitive are the replies?
- Does the reply contain the expected concepts?
- How much of the reply is exact corpus text versus new 3-gram sequences?
- Does the reply fit the student's own next-token distribution?
- How does an optional external judge read the prompt/reply pair?
- Did the read-only probe leave the selected durable state unchanged?

### They do not prove

- **Truth or factual correctness.** No metric checks against a verified answer
  key. Keyword presence and judge labels are not verification.
- **General language understanding.** The built-in set is small, fixed, and
  inspected by humans. It is not a random or stratified benchmark.
- **Held-out status.** The script can report exact prompt/corpus overlap when
  `--corpus` is supplied. It cannot prove that the prompt, a paraphrase, or the
  answer was absent from every training path.
- **Semantic novelty.** Novel 3-grams are only unfamiliar token sequences.
- **Correct keyword use.** Coverage ignores negation, clause position, and
  contradiction.
- **No copying.** The copy detector catches exact sentence fragments above a
  token threshold, not paraphrase or recombination.
- **A good model.** NLL is model-internal fit; a lower number is not a
  correctness result.
- **An objective judge.** The Gemma judge is a diagnostic instrument with the
  usual LLM biases and failure modes.
- **A production or canonical result.** The run performs no training, merge,
  canonical commit, NANNET route, or proof claim.

A flat, noisy or poor result is a valid result. The per-prompt replies and
metrics should be inspected before drawing any conclusion from the aggregate
summary.

## CLI reference

```text
--state-dir PATH   Trained NaNLM state directory (required)
--prompts PATH     JSONL/JSON probe file; built-in prompts if omitted
--corpus PATH      Optional training/reference corpus file or directory
--judge            Ask the optional Gemma judge
--output PATH      JSON report path (Markdown written beside it)
--url URL          Ollama OpenAI-compatible chat completions URL
--model NAME       Judge model name
--nll              Compute TemporalMirrorLM next-token NLL/perplexity
```

Exit behavior:

- a missing state directory or invalid prompt/corpus file exits with an
  argument error;
- a per-prompt student failure is recorded in that row and counted in
  `summary.student_errors`;
- a per-prompt judge failure is recorded in that row and counted in
  `summary.judge.errors`;
- report writing is atomic for both JSON and Markdown.

## Tests

The test module uses a fake native student, a fake judge and local temporary
state; it makes no live cloud call and writes no real training state:

```bash
python3 -m unittest tests.test_corpus_generation_eval -v
```

It covers prompt loading, corpus loading, copy/novelty/keyword/repetition
metrics, NLL math, strict judge parsing, read-only `nanlm_chat` probing,
per-prompt report rows, aggregate summaries, JSON/Markdown output, and boundary
flags.

## Related files

- `scripts/corpus_generation_eval.py` — the evaluator.
- `tests/test_corpus_generation_eval.py` — offline tests.
- `docs/DEEP_CORPUS_TRAINING.md` — the companion deep-training workflow.
- `scripts/deep_train_corpus.py` — produces the state this evaluator reads.
- `docs/NANLM_NOVEL_GENERATION.md` — an earlier, smaller qualitative probe.
- `nanlmind_hermes/temporal_lm.py` — the native `TemporalMirrorLM`.
- `nanlmind_hermes/chat.py` — the `nanlm_chat` native synthesis path.
