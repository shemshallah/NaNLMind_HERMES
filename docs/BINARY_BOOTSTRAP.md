# Binary bootstrap: bits to words

Everything is a typed leaf with provenance. 0 and 1 are value leaves and
the 16 Boolean functions are function leaves. Letters are byte patterns;
words are ordered hyperedges over letters; vectors are constructed,
dimension-aware leaves. No free-floating linguistic labels.

## Root Boolean table

- verified: **64/64** function/input combinations
- functions: `BOOL_0` … `BOOL_15`, plus `NEGATE`, `AFFIRM`, `CONCAT`

## Initial mapping matrix tree

| level | contents | count |
|---|---|---:|
| L0 | bit value leaves | 2 |
| L1 | functions | 19 |
| L2 | byte leaves | 256 |
| L3 | printable letter leaves | 95 |
| L4 | word hyperedges | 12 |
| L5 | constructed vector leaves | 1 |

## Bytewise completion

Contiguous split (train 0–191, test 192–255):

```json
{
  "accuracy": 0.375,
  "bits": 512,
  "by_prefix": {
    "0": 0.0,
    "1": 0.0,
    "2": 0.5,
    "3": 0.5,
    "4": 0.5,
    "5": 0.5,
    "6": 0.5,
    "7": 0.5
  }
}
```

Interleaved split (train values not divisible by 4):

```json
{
  "accuracy": 0.375,
  "bits": 512,
  "by_prefix": {
    "0": 0.5,
    "1": 0.5,
    "2": 0.5,
    "3": 0.5,
    "4": 0.5,
    "5": 0.5,
    "6": 0.0,
    "7": 0.0
  }
}
```

## Word leaves as hyperedges

| word | letters | address |
|---|---|---|
| NO | NO | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/6b3c2ea7f8b88064` |
| YES | YES | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/ce9eabcdaaa617e1` |
| ON | ON | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/21c98e9a9eed623d` |
| NOT | NOT | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/923893882a558d23` |
| ONE | ONE | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/b41516fa2cc43e4a` |
| ZERO | ZERO | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/8e2efdb6f2c07abe` |
| BOTH | BOTH | `nan://HW2:BS0:MC0:SC0:GM0::leaf/word/f74a2219d83ef4d3` |

## Typed vector and matrix leaves

```json
{
  "matrix": {
    "dtype": "RATIONAL",
    "rank": 2,
    "shape": [
      2,
      2
    ]
  },
  "vector": {
    "address": "nan://HW2:BS0:MC0:SC0:GM0::leaf/vector/585c6434a6b5b102",
    "dtype": "BOOLEAN",
    "rank": 1,
    "shape": [
      3
    ],
    "value": [
      1,
      0,
      1
    ]
  }
}
```

## Heads and self-modification

```json
{
  "admissions": 804,
  "leaves": 370,
  "transforms": 64,
  "used": 64
}
```

## What this establishes

- 1 and 0 are both values and Boolean-function leaves; NEGATE is a
  registered typed transform, not a language operator.
- Letters are grounded in byte patterns; NO is a hyperedge over N and O.
- Vectors carry rank, shape, dtype, value and provenance; they are built
  by functions, not given as free embeddings.
- The mapping tree is structural: bits -> functions -> bytes -> letters
  -> words -> vectors. The heads admit every leaf and record transforms.
- Bytewise completion is the first sequential skill; it is the same
  prefix-to-next prediction later used on word/temporal rails.

Honest limit: completion accuracy is a frequency baseline over held-out
bytes; it is not yet a learned Boolean circuit per byte. The next stage
is learned completion (Boolean circuits over bit prefixes), then letters
and word hyperedges on the temporal rails.
