# Corpus training

- corpus: `/storage/emulated/0/download/local-corpus/corpora`
- state: `/data/data/com.termux/files/usr/tmp/hyper-pilot`
- files: 9
- train records: 7552
- held-out records: 448

| epoch | observed | held-out NLL | perplexity | vocab | bigram ctx | trigram ctx | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 13.8155 | 1000000.0 | 0 | 0 | 0 | 0 |
| 1 | 7398 | 5.6609 | 287.396 | 11767 | 11768 | 35979 | 22.56 |
| 2 | 7398 | 5.6821 | 293.5528 | 11767 | 11768 | 35979 | 20.43 |

## Boundaries

- Local derived temporal state; no canonical merge.
- Held-out NLL/perplexity is a next-token measure, not general language understanding.
- Repeated passes clear only the replay digest guard; no corpus is uploaded.
