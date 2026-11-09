# Corpus training

- corpus: `experiments/corpus/trainable`
- state: `/data/data/com.termux/files/usr/tmp/nanlm-corpus-deep-strat`
- files: 9
- train records: 14039
- held-out records: 1449
- selected checkpoint: epoch 1
- stopped early: True

| epoch | observed | held-out NLL | perplexity | vocab | bigram ctx | trigram ctx | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 13.8155 | 1000000.0 | 0 | 0 | 0 | 0 |
| 1 | 13655 | 2.5140 | 12.3541 | 17962 | 17963 | 60299 | 42.72 |
| 2 | 13655 | 2.5275 | 12.5221 | 17962 | 17963 | 60299 | 39.38 |
| 3 | 13655 | 2.5343 | 12.6074 | 17962 | 17963 | 60299 | 39.48 |

## Selected held-out metrics by source

| source | tokens | NLL | perplexity |
|---|---:|---:|---:|
| `bit_language252.jsonl` | 1839 | 6.7991 | 897.0588 |
| `curriculum.jsonl` | 963 | 3.6760 | 39.4893 |
| `emotions.jsonl` | 3843 | 1.8219 | 6.1836 |
| `hierarchy_curriculum.jsonl` | 11499 | 3.0044 | 20.1734 |
| `perspectives.jsonl` | 6001 | 1.1887 | 3.2828 |
| `self_cultivation.jsonl` | 3609 | 1.4573 | 4.2942 |
| `short_sentences.jsonl` | 754 | 2.2262 | 9.2648 |
| `wordnet_active.jsonl` | 13 | 2.9114 | 18.3818 |

## Boundaries

- Local derived temporal state; no canonical merge.
- Held-out NLL/perplexity is a next-token measure, not general language understanding.
- Repeated passes clear only the replay digest guard; no corpus is uploaded.
