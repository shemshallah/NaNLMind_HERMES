# Robust gauge-flat recovery

Two-hop median reconstruction of a 256-node gauge-flat field under
independently corrupted edges. This is classical robust estimation.

| corrupted edges | raw MAE | mean MAE | median MAE |
|---:|---:|---:|---:|
| 10% | 0.2505 | 0.0574 | 0.0000 |
| 20% | 0.5004 | 0.0854 | 0.0000 |
| 40% | 1.0007 | 0.1185 | 0.0000 |
| 60% | 1.5000 | 0.1529 | 0.0021 |
| 80% | 1.9997 | 0.1692 | 0.0978 |

With Gaussian noise σ=0.05 and 10% corruption: raw 0.2859, mean 0.0610, median 0.0056.

Non-flat control at n=256: median MAE 0.7930 — the filter fails, as it must.

## Boundary

- The flat field has 255 independent gradient dimensions; it is not the
  32,385-dimensional cycle memory.
- The decoder assumes zero curvature. It does not recover learned semantic
  cycle states.
- No physical quantum effect is claimed.
