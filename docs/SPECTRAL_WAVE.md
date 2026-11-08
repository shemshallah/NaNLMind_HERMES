# Spectral wave operator over the 256x256 address space

Built from the head population's letter distributions: A[h1][h2] is
the cosine Gram weight. The matrix is symmetrized, centrosymmetrized
with the reversal J, folded into two 128x128 blocks, and given the
inverse-square void potential. Head 0 is the void/NaNNet anchor.

- training words: 1,500
- matrix: 256×256
- symmetric: True
- centrosymmetric: True
- eigen reference error: 0.0
- fold error: 0.0

## Top eigenvalues

```json
{
  "base": [
    16.358767,
    0.0,
    0.0,
    0.0
  ],
  "folded": [
    16.358767,
    0.0,
    0.0,
    0.0
  ],
  "numpy": [
    16.358767,
    0.0,
    0.0,
    0.0
  ],
  "void": [
    16.454071,
    1.475718,
    0.485804,
    0.295986
  ]
}
```

## Regions and spectral leakage

```json
{
  "base_coupling": {
    "inner": 0.085938,
    "ring": 0.066406,
    "outer": 0.097656
  },
  "voided_coupling": {
    "inner": 0.827154,
    "ring": 0.070219,
    "outer": 0.102628
  },
  "laplacian_low": {
    "inner": 0.086957,
    "ring": 0.067194,
    "outer": 0.84585
  },
  "void_laplacian_low": {
    "inner": 0.085316,
    "ring": 0.067782,
    "outer": 0.846901
  },
  "rank": 1,
  "void_shift": [
    -0.095304,
    -1.475718,
    -0.485804,
    -0.295986
  ]
}
```

Tunneling = inner-region mass. A nonzero value is spectral leakage
through the void barrier, not a quantum measurement.

## Principal modes (top heads)

### mode 0
```json
[
  {
    "head": 0,
    "amplitude": 0.0625,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 1,
    "amplitude": 0.0625,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 2,
    "amplitude": 0.0625,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 3,
    "amplitude": 0.0625,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 4,
    "amplitude": 0.0625,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 5,
    "amplitude": 0.0625,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  }
]
```
### mode 1
```json
[
  {
    "head": 0,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 1,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 2,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 3,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 4,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 5,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  }
]
```
### mode 2
```json
[
  {
    "head": 0,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 1,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 2,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 3,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 4,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  },
  {
    "head": 5,
    "amplitude": 0.0,
    "region": "outer",
    "letters": [
      [
        "$",
        1500.0
      ],
      [
        "i",
        1147.0
      ],
      [
        "e",
        1068.0
      ]
    ]
  }
]
```

## Reading

- The full 256×256 spectrum and the two 128×128 folded spectra agree,
  confirming the centrosymmetric/palindrome structure.
- The coupling matrix is rank-limited; the graph Laplacian low modes
  are the correct wave operator. The void potential on the Laplacian
  should suppress inner-region mass; the remaining inner mass is
  spectral leakage (tunneling).
- A positive diagonal on the coupling matrix attracts the top mode; it
  is not the barrier. Use L = D - A plus V and low modes.
- The head population is the current coarse A; the next step is to use
  the eigenmodes P and strengths Λ as the MTH L7 relation basis.


## Nonconstant-mode leakage (the meaningful tunneling measure)

The uniform (zero) mode is excluded; it is the trivial constant mode.

```json
{
  "base": {
    "inner": 0.001359,
    "outer": 0.997591,
    "ring": 0.00105
  },
  "void": {
    "inner": 0.000792,
    "outer": 0.997919,
    "ring": 0.001289
  },
  "reduction": 0.4172,
  "rank": 1
}
```

The coupling matrix is rank-1, so the head population is not yet a
rich spectral object. The void barrier reduces inner leakage of the
nonconstant modes from 0.001359 to
0.000792 (a
41.7% reduction): measurable spectral tunneling.

## Architecture status

- Full/folded spectra agree (fold error 0.0); numpy reference
  error 0.0.
- The next bottleneck is rank: all heads currently carry nearly the same
  letter distribution. Diversify head routing / use the relation matrix as A.
- Use the low modes of L = D - A + V as the L7 relation basis; the void is the
  boundary condition, not a positive diagonal on the coupling matrix.
