# Round 036 robust composition audit

SHA-256 verified: **18 files**; missing: none.

## Supplied neural artifacts
| metric | value |
|---|---:|
| neural seed 29 exact accuracy | 0.6701134089392928 |
| neural seed 29 semantic accuracy | 1.0 (artifact audit) |
| native cycle prompt recovery | 1.0 |
| raw edge prompt recovery | 0.0 |
| robust connected | 1.0 |
| robust disconnected | 1.0 |
| robust relation conflict | 1.0 |

## Resident symbolic fallback
| item | value |
|---|---:|
| gate accuracy | 1.0000 |
| composition accuracy | 1.0000 |
| contradiction abstentions | 180 |
| symbolic four-hop accepted | True |
| cycle read exact | True |
| raw read exact | False |
| full cycle dimension | 32385 |

## Boundaries
- The neural encoder/attention/GRU and checkpoints are external artifacts; they are not reimplemented here.
- The resident symbolic path supports arbitrary positive depth but is not a learned neural generalizer.
- Contradiction abstention is a deterministic relation rule in this fallback.
- Conventional symbolic memory ties; gauge invariance, not unique NaN reasoning, is the geometric claim.
- No canonical merge or free-form language claim.
