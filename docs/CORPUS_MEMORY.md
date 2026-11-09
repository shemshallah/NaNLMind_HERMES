# Corpus memory

`nanlmind_hermes/corpus_memory.py` is the deterministic text/evidence layer
used by the Round 020 work. It is deliberately separate from the language
model: it stores passages, assigns reproducible splits, and returns source
passages for a query. It does not generate text and does not claim semantic
comprehension.

## Ingestion

```python
from nanlmind_hermes.corpus_memory import CorpusStore

store = CorpusStore()
stats = store.ingest(["/path/to/texts"], max_tokens=700)
store.save("/path/to/corpus.json")
```

- Recursively reads UTF-8 `.txt` files from files or directories.
- Splits text on blank-line paragraph boundaries, then into up to
  `max_tokens` whitespace tokens per passage.
- Deduplicates exact passages by SHA-256.
- Stores the source path, passage hash, split and text.
- Uses an atomic JSON replacement for the manifest.

## Splits

The split is derived from the first eight hex digits of the passage SHA-256:

| hash bucket | split |
|---|---|
| 0-7 | train |
| 8 | validation |
| 9 | test |

The same corpus therefore produces the same split on every machine. It is a
deterministic hash split, not a stratified or topic-balanced split, and it is
not guaranteed to reproduce another system's split. Ingesting the Round 020
sample yields 2,400 passages but splits them 1,927/218/255 rather than the
external run's 1,934/215/251.

## Retrieval

```python
results = store.retrieve("a bird with feathers", split="train", limit=3)
for result in results:
    print(result["score"], result["source"], result["text"])
```

Retrieval ranks passages by the sum of inverse-document-frequency weights of
shared content tokens. Stopwords and one-character tokens are ignored. Scores
are not probabilities and retrieval can return irrelevant passages.

## Boundaries

- The store contains source evidence, not learned parameters.
- It has no autoregressive model, checkpoint, or generation path.
- Retrieval overlap is not understanding, entailment, or answer validation.
- Ingested files may contain private text; do not commit real user corpora or
  source manifests containing sensitive paths.
- The tiny 80/10/10 hash split is suitable for controlled retrieval experiments
  only.

## Verification

`tests/test_corpus_memory.py` covers recursive ingestion, exact deduplication,
deterministic hashes, token-window chunking, split filtering, retrieval and
save/load round trips. The historical baseline at this document's writing was
272 tests; use the current suite for the live count.
