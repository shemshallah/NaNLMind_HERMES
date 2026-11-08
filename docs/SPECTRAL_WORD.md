# Word-leaf write into the diversified spectral matrix

The head population uses active=8/context=2 (the best sparse config).
The matrix is centrosymmetrized with the **native reverse8 mirror**, the
void is a **Fiedler spectral cut**, and a three-letter word is written
with the atomic WordLeafWriter rule.

- base rank: **53**; after word write: **57**
- base eigenvalues: [17.712581, 1.665308, 1.553693, 0.962544, 0.799075, 0.558797]
- updated eigenvalues: [17.77763, 1.683073, 1.553685, 1.043025, 0.820832, 0.794217]
- Fiedler values: [-0.0, 0.0]
- inner heads: 1; boundary: 0; ring: 12
- delta norm: 2.086145
- native mirror residual: **0.0**
- void rows max: **0.0**
- projected delta energy: 4.352 (top modes 0.037787)

## Top mode energies

```json
[
  {
    "mode": 3,
    "energy": -0.095867
  },
  {
    "mode": 0,
    "energy": -0.087169
  },
  {
    "mode": 255,
    "energy": 0.084243
  },
  {
    "mode": 195,
    "energy": -0.05694
  },
  {
    "mode": 2,
    "energy": 0.056155
  },
  {
    "mode": 199,
    "energy": 0.052377
  },
  {
    "mode": 69,
    "energy": -0.049206
  },
  {
    "mode": 11,
    "energy": 0.048382
  }
]
```

## Word record

```json
{
  "kind": "word",
  "letters": 3,
  "couplings": [
    {
      "left_support": [
        1,
        2,
        3,
        4
      ],
      "right_support": [
        5,
        6,
        7,
        8
      ]
    },
    {
      "left_support": [
        5,
        6,
        7,
        8
      ],
      "right_support": [
        9,
        10,
        11,
        12
      ]
    }
  ],
  "beta": 0.5,
  "gamma": 0.25,
  "centrosymmetric": true,
  "void_zero": true
}
```

## Reading

- Centrosymmetry and void projection are exact if the residuals are zero.
- The word write is low-rank, so rank is expected to stay nearly stable;
  the meaningful effect is the eigenmode energy distribution.
- The next step is to use this written matrix in the temporal spine and
  measure whether the word couplings survive projection/resonance.
