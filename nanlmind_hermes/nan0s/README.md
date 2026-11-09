# Bundled NaN0S portable core

This directory vendors the architecture-neutral parts of
[`shemshallah/NaN0S`](https://github.com/shemshallah/NaN0S) so the HERMES plugin
can build a native NaN16 runner on the user's host without requiring a separate
checkout:

- `core/` — the portable C NaN16/NaNVM1 core used by the Android/aarch64
  conformance bench. It is plain C99 and builds on aarch64, x86_64 and other
  hosts with a C compiler.
- `tools/nan_cli.c` — the headless CLI used to load a `.nan` image, run it for
  a bounded step count and emit serial output / guest words.
- `tools/nmk.awk` — the POSIX-awk NaN16 source emitter used by self-hosted
  builds and CI. It is a host-side assembler seed, not the guest `nasm0`.

Source: NaN0S, Apache-2.0. This is a source snapshot, not a prebuilt binary.
The plugin compiles it into its own state cache on first native MTH use and
records the compiler, architecture and source digest. The guest kernels remain
NaN16 source; only the thin host VM is native per architecture.

The bundled core is used only for bounded local MTH fixture verification and
the four admitted vector-call dispatchers. It does not authorize NANNET routes,
leases, merges or canonical model updates.
