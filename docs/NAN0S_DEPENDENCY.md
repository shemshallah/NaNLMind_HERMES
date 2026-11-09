# NaN0S dependency contract

`NaNLMind_HERMES` does **not** own a NaN16 VM implementation.

The canonical machine is [`shemshallah/NaN0S`](https://github.com/shemshallah/NaN0S).
HERMES uses it in this order:

1. `nan0s_root` configuration;
2. sibling `../nan0s`, `~/nan0s`, `~/NaN0S` checkout;
3. bundled `nanlmind_hermes/nan0s` snapshot, used only when no checkout exists
   (wheel/offline fallback).

The bundled snapshot is an **exact pinned copy**, recorded in
`nanlmind_hermes/nan0s/SOURCE.json` with the canonical commit and SHA-256 for
every file. Do not edit the bundled files by hand.

## Sync

From a canonical checkout:

```sh
python3 scripts/sync_nan0s.py --source ~/nan0s --write
python3 scripts/sync_nan0s.py --source ~/nan0s --check
python3 -m unittest tests.test_nan0s_sync -v
```

`--check` fails on any local drift or canonical-commit mismatch. The test suite
also compares every manifest hash to the canonical checkout when present.

## ISA unification

NaNVM0 and NaNVM1 are revisions of one machine, not separate interpreters.
The portable C core is the single scalar reference; N16I is the raw-NaN guest
interpreter. New ISA operations are added to:

- `aarch64/android/core/nan16.c` (portable core);
- `aarch64/android/tests/test_core.c` (unit semantics);
- `spec/NANVM_SPEC.md` and `core/OPS.md` (the one ISA document);
- `src/tools/n16i.nas` (guest interpreter dispatch);
- `tools/nmk.awk` (assembler mnemonics);
- `src/prog/nasm0_tables.nas` where the x86-side table is affected.

Then run:

```sh
cd aarch64/android && make -s nan_cli test
cd ../.. && sh tools/ngate_n16i.sh
```

Only after those pass is the snapshot synced into HERMES. No second ISA and no
locally invented opcode are allowed.
