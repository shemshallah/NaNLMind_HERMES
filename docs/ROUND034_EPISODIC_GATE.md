# Round 034 episodic gate

Source archive SHA-256 verified: **20 files**; missing from archive: run.log, stress.log.

## Artifact result
| seed | router acc | false-answer rate |
|---:|---:|---:|
| 17 | 0.9933 | 0.0050 |
| 29 | 0.9950 | 0.0033 |
| 41 | 0.9817 | 0.0050 |

## Pure resident reproduction
| field | value |
|---|---:|
| router accept accuracy | 0.9800 |
| positive accept | 0.9933 |
| false answer | 0.0167 |
| cycle copy exact | True |
| raw edge exact | False |

## Boundaries
- Pure Python char-key fallback is not the supplied neural encoder; its score differs from the 99.0% artifact.
- Copying a definition with a confidence gate is not free-form generation.
- The conventional indexed control ties the cycle copy path; no NaN-unique advantage is claimed.
- Raw edges are not gauge invariant; cycle reads are.
