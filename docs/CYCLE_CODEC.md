# Gauge-corrected cycle code

A synthetic semantic vector is encoded into an orthonormal divergence-free
cycle basis, then an unknown vertex gauge, edge erasure, and large spikes
are applied. This is a classical linear code.

| erasure | spikes | gauge cosine | joint robust | uncorrected |
|---:|---:|---:|---:|---:|
| 0% | 0% | 1.0000 | 1.0000 | 1.0000 |
| 0% | 10% | 0.3287 | 0.8594 | 0.3287 |
| 0% | 40% | 0.1888 | 0.3771 | 0.1888 |
| 50% | 0% | 0.9971 | 1.0000 | 0.5072 |
| 50% | 10% | 0.4099 | 0.7762 | 0.3658 |
| 50% | 40% | -0.0500 | 0.2921 | 0.0271 |
| 90% | 0% | 0.9646 | 1.0000 | 0.0548 |
| 90% | 10% | 0.0012 | 0.0750 | -0.0000 |
| 90% | 40% | 0.1323 | 0.2540 | 0.0478 |

## Boundaries

- Gauge fitting recovers the encoded vector under erasure, but the
  plain least-squares fit degrades under large independent spikes.
- The joint robust decoder models cycle coefficients and gauge together
  with Huber reweighting and recovers much more under spikes in this
  synthetic test; this is still not a learned semantic encoder.
- The random codebook is not a learned semantic encoder and adds no
  information; equivalent real linear algebra is sufficient.
- No quantum hardware or physical quantum effect is involved.
