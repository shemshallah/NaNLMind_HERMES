# Spectral wave operator over the 256x256 address space

Built from the head population's letter distributions: A[h1][h2] is
the cosine Gram weight. The matrix is symmetrized, centrosymmetrized
with the reversal J, folded into two 128x128 blocks, and given the
inverse-square void potential. Head 0 is the void/NaNNet anchor.

- training words: 1,500
- matrix: 256×256
- symmetric: True
- centrosymmetric: True
- eigen reference error: 0.00024203
- fold error: 0.00024203

## Top eigenvalues

```json
{
  "base": [
    17.646824,
    2.219757,
    1.060308,
    0.895898
  ],
  "folded": [
    17.646824,
    2.219757,
    1.060308,
    0.89614
  ],
  "numpy": [
    17.646824,
    2.219757,
    1.060308,
    0.89614
  ],
  "void": [
    17.740519,
    2.372655,
    1.620096,
    1.537442
  ]
}
```

## Regions and spectral leakage

```json
{
  "base_coupling": {
    "inner": 0.355305,
    "ring": 0.257315,
    "outer": 0.38738
  },
  "voided_coupling": {
    "inner": 0.660601,
    "ring": 0.136866,
    "outer": 0.202533
  },
  "laplacian_low": {
    "inner": 0.315488,
    "ring": 0.068705,
    "outer": 0.615807
  },
  "void_laplacian_low": {
    "inner": 0.309497,
    "ring": 0.069108,
    "outer": 0.621394
  },
  "rank": 54,
  "void_shift": [
    -0.093695,
    -0.152897,
    -0.559788,
    -0.641544
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
    "head": 104,
    "amplitude": 0.08043,
    "region": "inner",
    "letters": [
      [
        "i",
        135.0
      ],
      [
        "$",
        102.0
      ],
      [
        "e",
        66.0
      ]
    ]
  },
  {
    "head": 151,
    "amplitude": 0.08043,
    "region": "inner",
    "letters": [
      [
        "$",
        76.0
      ],
      [
        "t",
        21.0
      ],
      [
        "n",
        17.0
      ]
    ]
  },
  {
    "head": 38,
    "amplitude": 0.079714,
    "region": "ring",
    "letters": [
      [
        "$",
        204.0
      ],
      [
        "a",
        75.0
      ],
      [
        "n",
        54.0
      ]
    ]
  },
  {
    "head": 217,
    "amplitude": 0.079714,
    "region": "ring",
    "letters": [
      [
        "i",
        9.0
      ],
      [
        "o",
        1.0
      ],
      [
        "h",
        1.0
      ]
    ]
  },
  {
    "head": 8,
    "amplitude": 0.075209,
    "region": "outer",
    "letters": [
      [
        "i",
        15.0
      ],
      [
        "$",
        8.0
      ],
      [
        "e",
        7.0
      ]
    ]
  },
  {
    "head": 247,
    "amplitude": 0.075209,
    "region": "outer",
    "letters": [
      [
        "$",
        102.0
      ],
      [
        "e",
        68.0
      ],
      [
        "t",
        52.0
      ]
    ]
  }
]
```
### mode 1
```json
[
  {
    "head": 38,
    "amplitude": 0.177033,
    "region": "ring",
    "letters": [
      [
        "$",
        204.0
      ],
      [
        "a",
        75.0
      ],
      [
        "n",
        54.0
      ]
    ]
  },
  {
    "head": 217,
    "amplitude": -0.177033,
    "region": "ring",
    "letters": [
      [
        "i",
        9.0
      ],
      [
        "o",
        1.0
      ],
      [
        "h",
        1.0
      ]
    ]
  },
  {
    "head": 124,
    "amplitude": 0.13875,
    "region": "inner",
    "letters": [
      [
        "n",
        260.0
      ],
      [
        "l",
        220.0
      ],
      [
        "r",
        182.0
      ]
    ]
  },
  {
    "head": 131,
    "amplitude": -0.13875,
    "region": "inner",
    "letters": [
      [
        "o",
        40.0
      ],
      [
        "i",
        9.0
      ],
      [
        "e",
        7.0
      ]
    ]
  },
  {
    "head": 122,
    "amplitude": -0.131742,
    "region": "inner",
    "letters": [
      [
        "i",
        37.0
      ],
      [
        "a",
        33.0
      ],
      [
        "o",
        25.0
      ]
    ]
  },
  {
    "head": 133,
    "amplitude": 0.131742,
    "region": "inner",
    "letters": [
      [
        "t",
        66.0
      ],
      [
        "$",
        65.0
      ],
      [
        "s",
        51.0
      ]
    ]
  }
]
```
### mode 2
```json
[
  {
    "head": 34,
    "amplitude": -0.18174,
    "region": "outer",
    "letters": [
      [
        "$",
        34.0
      ],
      [
        "n",
        20.0
      ],
      [
        "r",
        17.0
      ]
    ]
  },
  {
    "head": 221,
    "amplitude": -0.18174,
    "region": "outer",
    "letters": [
      [
        "$",
        301.0
      ],
      [
        "a",
        106.0
      ],
      [
        "i",
        89.0
      ]
    ]
  },
  {
    "head": 218,
    "amplitude": -0.166601,
    "region": "ring",
    "letters": [
      [
        "$",
        148.0
      ],
      [
        "n",
        61.0
      ],
      [
        "s",
        47.0
      ]
    ]
  },
  {
    "head": 37,
    "amplitude": -0.166601,
    "region": "ring",
    "letters": [
      [
        "$",
        151.0
      ],
      [
        "a",
        57.0
      ],
      [
        "s",
        31.0
      ]
    ]
  },
  {
    "head": 104,
    "amplitude": -0.161937,
    "region": "inner",
    "letters": [
      [
        "i",
        135.0
      ],
      [
        "$",
        102.0
      ],
      [
        "e",
        66.0
      ]
    ]
  },
  {
    "head": 151,
    "amplitude": -0.161937,
    "region": "inner",
    "letters": [
      [
        "$",
        76.0
      ],
      [
        "t",
        21.0
      ],
      [
        "n",
        17.0
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
