# Resident corpus learning

- corpus: `experiments/corpus/trainable`
- resident file: `/data/data/com.termux/files/usr/tmp/nanlm-resident-full/resident/universe.nlmres`
- train records: 14039
- held-out records: 500
- selected epoch: 1
- canonical merge: disabled

## Held-out next-token metrics

| pass | temporal NLL | temporal ppl | resident NLL | resident ppl | hybrid NLL | hybrid ppl |
|---:|---:|---:|---:|---:|---:|---:|
| before | 6.7254 | 833.333333 | 5.5953 | 269.14567 | 5.8347 | 341.960308 |
| epoch-1 | 2.7400 | 15.487501 | 3.4229 | 30.656954 | 2.6042 | 13.520929 |
| epoch-2 | 2.7400 | 15.487501 | 3.4229 | 30.656954 | 2.6042 | 13.520929 |

## Resident state

| item | value |
|---|---:|
| leaves | 13911 |
| relations | 0 |
| hyperedges | 0 |
| features | 13655 |
| generation | 13762 |
| digest | `3c8e131ed5fa65002d58e11c653deb4f101545958abcb746322b1d6d8763a8ba` |

## Resident association probe

```json
[
  {
    "resident_probability": 0.18911446,
    "score": 0.14178905,
    "temporal_probability": 0.05389902,
    "word": "."
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12294373,
    "temporal_probability": 5.523e-05,
    "word": "instead"
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12292985,
    "temporal_probability": 1.558e-05,
    "word": "correct"
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12292866,
    "temporal_probability": 1.218e-05,
    "word": "hum"
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12292698,
    "temporal_probability": 7.37e-06,
    "word": "domain"
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12292598,
    "temporal_probability": 4.53e-06,
    "word": "suddenly"
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12292539,
    "temporal_probability": 2.84e-06,
    "word": "raw"
  },
  {
    "resident_probability": 0.18911446,
    "score": 0.12292499,
    "temporal_probability": 1.7e-06,
    "word": "02383482"
  }
]
```

## Boundaries

- Temporal and resident channels are reported separately; the stronger
  temporal baseline cannot hide a flat resident channel.
- The resident pair channel uses 256 hashed heads. Collisions are real and
  are why it is an associative supplement, not the primary LM.
- Per-source metrics are written in the JSON companion.
- No canonical merge, promotion, NANNET upload, or 2 GiB overbootstrap map.
