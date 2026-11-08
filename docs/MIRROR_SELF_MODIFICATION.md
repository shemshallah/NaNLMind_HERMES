# Mirror topology and Boolean-gated self-modification

HERMES now mounts the same logical addressing tree as NaNROUTER without
copying the physical 8 MiB chart.

## Resident topology

| Plane | HERMES object | Authoritative? |
|---|---|---|
| fixed heads | 256 `HW0:0..255` | logical topology |
| mirror pairs | 32,640 `HW0:MP0..32639` microtables | logical topology |
| canonical cells | 32,896 = 256 + 32,640 | logical topology |
| directional addresses | 65,536 = 256² | logical topology |
| versioned leaves | `hermes_mp_binding` in the existing `mathematical_memory.sqlite3` | derived/phased |
| physical chart | one `Chart8MiB` / `N16ROOT1` | native guest |

`nanlmind_hermes/mirror_topology.py` recomputes the rank/unrank transform and
asserts the exact NaNROUTER topology digest:

```text
13b02d7d45fa00c1a5df9586e630f44033ea49de21b12a14f5350c1f01f90016
```

A mismatch fails closed. The maps are 128 KiB forward and 65,792 bytes inverse,
not a duplicate 8 MiB chart and not 32,640 files.

## Recursive nan:// routes

```text
nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW0:7::L5
nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW0:MP123::L3
nan://GM0:0::SC0:0::MC0:0::BS0:0::HW0:0::HW0:7::HW0:MP123::L3
```

L0–L9 are resolution coordinates, not ten models. HEAD0/NANNET0 can never be a
mutation source or leaf target.

## Versioned leaf bindings

`MirrorLeafBindings` adds two tables to the existing HERMES mathematical memory
SQLite file:

- `hermes_mp_binding` — one active MP → leaf pointer with generation/CAS.
- `hermes_mp_binding_history` — immutable versions.

A binding requires a real endpoint head (1..255), a lowercase SHA-256 leaf
digest, a positive leaf version and the expected generation. Stale writes fail.

## Boolean-gated self-modification

`SelfModificationEngine` combines:

1. `MTHHeadMapper` root Boolean transform (mask, folded value, truth bit);
2. `HeadEnsemble`/`MTHLayerStack` learned context;
3. the 1 GiB paged L0–L9 scratch register;
4. the existing MC0DB-style mirror binding table.

A proposed relation is evaluated through the root truth table. If the Boolean
gate rejects it, the mutation is retained as **negative memory** (polarity -1,
reduced strength) rather than silently promoted. The scratch record preserves
the mask, root function, truth bit, evidence digest, source head and MTH leaf.
Promotion into the local 8 MiB head register still requires the existing
explicit approval gate. Canonical NaNLMind training is never committed here.

## MTH tools

The HERMES facade exposes the same six read/compose names as NaNROUTER:

```text
mth.catalog, mth.inspect, mth.categories,
mth.compose, mth.prepare, mth.route
```

It is a bounded planning facade. Native MTH execution remains the NaNROUTER/
NMP/NaN16 guest path (`nanlm_native_mth_u32` and the NANNET lease).

## Hermes tools

| Tool | Scope |
|---|---|
| `nanlm_mirror_topology` | `universe:read` |
| `nanlm_self_modify` | `training:execute` |
| `nanlm_mth_tools` | `universe:read` |
| `nanlm_teacher_teach` | `training:execute` |
