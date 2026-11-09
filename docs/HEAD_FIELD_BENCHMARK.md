# Head-field activation benchmark

Real `nanlmind_hermes.resident.cells.ResidentCells` region over a fresh 2 MiB `bytearray`. The workload is deterministic synthetic activation, not a live model.

## Run

| field | value |
|---|---|
| heads requested / capacity | 256 / 256 |
| steps | 32 |
| seed | 20261108 |
| buffer | 2.0 MiB (2097152 bytes) |
| resident region | 2097152 bytes |
| modes | `independent`, `coupled`, `single_head`, `uniform` |
| total update operations | 40992 |
| total wall (region formatting + modes) | 1373.672 ms |

## Scope and claims

| claim | value |
|---|---|
| real ResidentCells used | yes |
| native code used | no |
| phone/handset performance claim | no |
| canonical merge performed | no |
| live model / training | no |

This is **not** a native-code benchmark and **not** a phone/handset performance measurement or claim.

## Mode summary

| mode | active heads | updated heads | entropy bits | normalized entropy | distinct levels | effective heads | top share | per-head ops | pair ops | update ops/s | wall ms | memory bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `independent` | 256 | 256 | 7.898905 | 0.987363 | 256 | 238.675 | 0.006393 | 8192 | 0 | 62352.5 | 146.443 | 2097152 |
| `coupled` | 256 | 256 | 7.952537 | 0.994067 | 256 | 247.715 | 0.005613 | 8192 | 16384 | 28682.6 | 871.258 | 2097152 |
| `single_head` | 1 | 1 | 0.000000 | 0.000000 | 1 | 1.000 | 1.000000 | 32 | 0 | 71086.8 | 3.619 | 2097152 |
| `uniform` | 256 | 256 | 8.000000 | 1.000000 | 1 | 256.000 | 0.003906 | 8192 | 0 | 85056.4 | 99.770 | 2097152 |

## Active-head distribution

Counts are over all requested heads; buckets are each head's share of the total positive activation. The distribution is reported per mode, not as one aggregate.

| activation share bucket | `independent` | `coupled` | `single_head` | `uniform` |
|---|---:|---:|---:|---:|
| inactive | 0 | 0 | 255 | 0 |
| (0, 0.1%] | 0 | 0 | 0 | 0 |
| (0.1%, 0.25%] | 54 | 24 | 0 | 0 |
| (0.25%, 0.5%] | 133 | 185 | 0 | 256 |
| (0.5%, 1%] | 69 | 47 | 0 | 0 |
| (1%, 2%] | 0 | 0 | 0 | 0 |
| (2%, 5%] | 0 | 0 | 0 | 0 |
| (5%, 10%] | 0 | 0 | 0 | 0 |
| > 10% | 0 | 0 | 1 | 0 |

## Measured vs target

Every target is labelled explicitly; `observed` means the metric is informational rather than a pass/fail threshold.

### `independent` - PASS

| metric | measured | target |
|---|---:|---|
| active head count | 256 | active head count min 2; active head count ideal 256 |
| distinct activation levels | 256 | distinct activation levels min 2 |
| field entropy bits | 7.898905 | observed |
| normalized entropy | 0.987363 | observed |
| top head share | 0.006393 | observed |
| per-head update ops | 8192 | per head updates exact 8192 |
| pair coupling update ops | 0 | pair coupling updates exact 0 |
| address collision rate | 0.000000 | address collision rate max 0.0 |
| rail-key collision rate | 0.000000 | rail collision rate max 0.0 |
| canonical aliasing rate | n/a | observed (intentional mirror sharing) |
| memory bytes | 2097152 | memory bytes max 2097152 |

Checks: `per_head_updates_match`=pass, `pair_coupling_updates_match`=pass, `head_generations_match`=pass, `pair_generations_match`=pass, `memory_bytes_within_buffer`=pass, `address_collisions_zero`=pass, `rail_key_collisions_zero`=pass, `independent_multi_head`=pass, `independent_distinct_levels`=pass, `independent_all_requested_heads_active`=pass

### `coupled` - PASS

| metric | measured | target |
|---|---:|---|
| active head count | 256 | active head count min 2; active head count ideal 256 |
| distinct activation levels | 256 | distinct activation levels min 2 |
| field entropy bits | 7.952537 | observed |
| normalized entropy | 0.994067 | observed |
| top head share | 0.005613 | observed |
| per-head update ops | 8192 | per head updates exact 8192 |
| pair coupling update ops | 16384 | pair coupling updates exact 16384 |
| address collision rate | 0.000000 | address collision rate max 0.0 |
| rail-key collision rate | 0.000000 | rail collision rate max 0.0 |
| canonical aliasing rate | 0.500000 | observed (intentional mirror sharing) |
| memory bytes | 2097152 | memory bytes max 2097152 |

Checks: `per_head_updates_match`=pass, `pair_coupling_updates_match`=pass, `head_generations_match`=pass, `pair_generations_match`=pass, `memory_bytes_within_buffer`=pass, `address_collisions_zero`=pass, `rail_key_collisions_zero`=pass, `coupled_multi_head`=pass, `coupled_pair_updates_present`=pass, `coupled_pair_field_present`=pass, `coupled_all_requested_heads_active`=pass

### `single_head` - PASS

| metric | measured | target |
|---|---:|---|
| active head count | 1 | active head count exact 1 |
| distinct activation levels | 1 | distinct activation levels exact 1 |
| field entropy bits | 0.000000 | observed |
| normalized entropy | 0.000000 | observed |
| top head share | 1.000000 | observed |
| per-head update ops | 32 | per head updates exact 32 |
| pair coupling update ops | 0 | pair coupling updates exact 0 |
| address collision rate | 0.000000 | address collision rate max 0.0 |
| rail-key collision rate | 0.000000 | rail collision rate max 0.0 |
| canonical aliasing rate | n/a | observed (intentional mirror sharing) |
| memory bytes | 2097152 | memory bytes max 2097152 |

Checks: `per_head_updates_match`=pass, `pair_coupling_updates_match`=pass, `head_generations_match`=pass, `pair_generations_match`=pass, `memory_bytes_within_buffer`=pass, `address_collisions_zero`=pass, `rail_key_collisions_zero`=pass, `single_head_exactly_one_active`=pass, `single_head_is_head_zero`=pass, `single_head_one_distinct_level`=pass, `single_head_histogram_total`=pass

### `uniform` - PASS

| metric | measured | target |
|---|---:|---|
| active head count | 256 | active head count exact 256 |
| distinct activation levels | 1 | distinct activation levels exact 1 |
| field entropy bits | 8.000000 | observed |
| normalized entropy | 1.000000 | field entropy normalized min 0.999999 |
| top head share | 0.003906 | observed |
| per-head update ops | 8192 | per head updates exact 8192 |
| pair coupling update ops | 0 | pair coupling updates exact 0 |
| address collision rate | 0.000000 | address collision rate max 0.0 |
| rail-key collision rate | 0.000000 | rail collision rate max 0.0 |
| canonical aliasing rate | n/a | observed (intentional mirror sharing) |
| memory bytes | 2097152 | memory bytes max 2097152 |

Checks: `per_head_updates_match`=pass, `pair_coupling_updates_match`=pass, `head_generations_match`=pass, `pair_generations_match`=pass, `memory_bytes_within_buffer`=pass, `address_collisions_zero`=pass, `rail_key_collisions_zero`=pass, `uniform_all_requested_heads_active`=pass, `uniform_one_distinct_level`=pass, `uniform_entropy_near_max`=pass

## Controls comparison

| comparison | value |
|---|---:|
| independent active heads | 256 |
| single-head control active heads | 1 |
| extra active heads vs single-head control | 255 |
| independent distinct activation levels | 256 |
| single-head distinct levels | 1 |
| independent entropy bits | 7.898905 |
| uniform entropy bits | 8.000000 |
| independent top head share | 0.006393 |
| uniform top head share | 0.003906 |
| coupled pair updates | 16384 |
| coupled unique pair cells | 256 |
| coupled active heads | 256 |
| coupled canonical aliasing rate | 0.500000 |
| coupled rail collision rate | 0.000000 |

## Pair addressing

| metric | `independent` | `coupled` | `single_head` | `uniform` |
|---|---:|---:|---:|---:|
| directed pair keys | 0 | 512 | 0 | 0 |
| unique canonical cells | 0 | 256 | 0 | 0 |
| address collisions | 0 | 0 | 0 | 0 |
| address collision rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| rail-key collisions | 0 | 0 | 0 | 0 |
| rail collision rate | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| canonical aliasing rate | n/a | 0.500000 | n/a | n/a |
| canonical pair coverage | 0.000000 | 0.007843 | 0.000000 | 0.000000 |

Address collisions are checked over distinct directed pair keys; repeated observations are accumulated updates, not collisions. Canonical mirror aliasing is intentional: the two rails preserve the two directions.

## Determinism

- fixed seed: `20261108`
- deterministic fingerprint: `4f2a7cbf56e4d4d2414a88dbc71c8c7da9f3bfb0610097552e4925dc01f68431`
- wall-clock and throughput fields are excluded from the fingerprint; structural and field measurements are included.

## Boundaries

- Measured on host CPython over a 2 MiB in-memory bytearray, not native code, not an mmap file, and not a phone/handset performance claim.
- The workload is deterministic synthetic activation, not model inference, training, semantic quality, or learned behaviour.
- Active heads and entropy describe the injected activation distribution, not semantic capacity or information created by the cell field.
- Canonical pair addresses intentionally share one mirror cell; the s+ and s- rails preserve direction, so zero rail-key collisions is expected.
- Very large --steps values can saturate the fixed-point Q16.16 pair accumulators; generation counts remain exact while amplitude/weight readouts then reflect saturation.
- No canonical merge, promotion, durable user state, or native acceleration is performed by this benchmark.
