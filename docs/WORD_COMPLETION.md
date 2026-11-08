# WordNet 3.1 head-population next-letter completion

Letters are not mapped to heads. Each of the 256 heads holds a
distribution over the whole alphabet (`a`–`z`, end-of-word); a context
activates a population of heads and their distributions collapse into
the next-letter counts. Head 0 is the shared NaNNet/polarity prior.

- lemmas: **20,000** (train 16,096, test 3,904)
- heads with a letter distribution: **256**
- active heads per context: **17**
- letter→head mapping: **none**
- dictionary in superposition: **True**; stored words: **0**

Training streams words and partial words; no dictionary text is kept.

## Accuracy

| model | accuracy | top-3 |
|---|---:|---:|
| head population | 0.1136 | 0.3062 |
| bigram baseline | 0.1903 |  |
| all 256 heads, frequency-weighted (8k) | 0.1167 | 0.3032 |

## All-256-head run

Head 0 is the void/NaNNet prior; heads 1–255 all participate. Training
uses an 8k frequency-weighted curriculum (common words count more;
counts are order-independent).

```json
{
  "examples": {
    "in": {
      "head_votes": {
        "$": 136,
        "e": 20
      },
      "top": [
        [
          "$",
          0.122096
        ],
        [
          "e",
          0.103683
        ],
        [
          "i",
          0.083207
        ],
        [
          "a",
          0.076401
        ],
        [
          "n",
          0.06684
        ],
        [
          "r",
          0.065531
        ]
      ]
    },
    "qu": {
      "head_votes": {
        "$": 144,
        "e": 20
      },
      "top": [
        [
          "$",
          0.12244
        ],
        [
          "e",
          0.103758
        ],
        [
          "i",
          0.083909
        ],
        [
          "a",
          0.076799
        ],
        [
          "n",
          0.067367
        ],
        [
          "r",
          0.066009
        ]
      ]
    },
    "th": {
      "head_votes": {
        "$": 141,
        "e": 20
      },
      "top": [
        [
          "$",
          0.122434
        ],
        [
          "e",
          0.104822
        ],
        [
          "i",
          0.083696
        ],
        [
          "a",
          0.076045
        ],
        [
          "n",
          0.067358
        ],
        [
          "r",
          0.066575
        ]
      ]
    }
  },
  "result": {
    "accuracy": 0.1167,
    "letters": 33444,
    "top3": 0.3032
  },
  "status": {
    "active_per_context": 256,
    "dictionary_in_superposition": true,
    "head_0": "NaNNet + polarity prior",
    "heads": 256,
    "heads_with_distribution": 256,
    "letter_to_head_mapping": false,
    "letters_per_head": 27,
    "observations": 67196,
    "stored_words": 0,
    "superposition_heads": 256
  }
}
```

## Collapse examples

- `^^` → e:0.102, a:0.091, i:0.081, o:0.075, r:0.073, t:0.065, n:0.065, $:0.062
  - head votes: e:5, a:4, t:3, r:2, s:1, o:1, n:1
- `th` → e:0.115, a:0.093, i:0.088, o:0.077, r:0.076, $:0.065, n:0.060, t:0.059
  - head votes: e:12, r:2, t:1, i:1
- `qu` → e:0.104, a:0.096, i:0.089, o:0.078, r:0.069, n:0.067, $:0.065, t:0.056
  - head votes: e:7, a:3, n:2, r:1, o:1, i:1, $:1
- `in` → e:0.108, a:0.093, i:0.082, o:0.075, n:0.066, r:0.066, t:0.062, $:0.057
  - head votes: e:9, n:3, a:2, o:1, t:1, r:1
- `st` → e:0.116, a:0.100, i:0.094, r:0.083, o:0.080, n:0.063, $:0.063, l:0.053
  - head votes: e:11, r:2, a:1, i:1, t:1
- `er` → e:0.107, a:0.095, i:0.089, o:0.077, $:0.074, r:0.070, n:0.059, t:0.057
  - head votes: e:7, a:4, $:2, r:1, t:1

## Head superposition (sample)

The dictionary is stored only as head letter distributions. No word
text is retained.

```json
[
  {
    "head": 0,
    "letters": {
      "e": 0.1036,
      "a": 0.0913,
      "i": 0.0847,
      "o": 0.0774,
      "r": 0.0737,
      "n": 0.068,
      "$": 0.0609,
      "t": 0.0602
    },
    "mass": 264288.0
  },
  {
    "head": 1,
    "letters": {
      "a": 0.105,
      "e": 0.101,
      "t": 0.0804,
      "i": 0.079,
      "o": 0.0769,
      "u": 0.0634,
      "$": 0.062,
      "r": 0.055
    },
    "mass": 23788.0
  },
  {
    "head": 2,
    "letters": {
      "a": 0.1178,
      "i": 0.1132,
      "e": 0.1111,
      "$": 0.099,
      "l": 0.0884,
      "r": 0.0871,
      "o": 0.0818,
      "u": 0.041
    },
    "mass": 8758.0
  },
  {
    "head": 3,
    "letters": {
      "a": 0.1379,
      "i": 0.1241,
      "e": 0.1143,
      "o": 0.0876,
      "r": 0.0564,
      "$": 0.055,
      "l": 0.055,
      "u": 0.0517
    },
    "mass": 25915.0
  },
  {
    "head": 4,
    "letters": {
      "o": 0.1132,
      "a": 0.1051,
      "e": 0.095,
      "i": 0.0944,
      "r": 0.0914,
      "n": 0.0672,
      "$": 0.0568,
      "l": 0.0487
    },
    "mass": 18567.0
  },
  {
    "head": 5,
    "letters": {
      "e": 0.1204,
      "o": 0.0937,
      "i": 0.0897,
      "a": 0.0863,
      "r": 0.0767,
      "t": 0.0678,
      "h": 0.0575,
      "n": 0.056
    },
    "mass": 14951.0
  },
  {
    "head": 6,
    "letters": {
      "n": 0.1288,
      "e": 0.0847,
      "t": 0.0776,
      "a": 0.073,
      "o": 0.0625,
      "p": 0.0591,
      "i": 0.0574,
      "r": 0.057
    },
    "mass": 18551.0
  },
  {
    "head": 7,
    "letters": {
      "s": 0.0824,
      "n": 0.0753,
      "$": 0.0717,
      "r": 0.07,
      "l": 0.0654,
      "a": 0.0641,
      "t": 0.0623,
      "e": 0.0604
    },
    "mass": 18159.0
  }
]
```

## Prefix accuracy

```json
{
  "2": 0.069,
  "3": 0.1129,
  "4": 0.1207,
  "5": 0.1095
}
```

## Honest read

- The population is a hashed mixture: every head holds all letters, so no
  one letter owns a head.
- The head votes are diagnostics of the collapse; the answer is the
  summed distribution, not a single head's choice.
- This is still a statistical model; a learned Boolean/gate mechanism
  should replace the hash activation next.
