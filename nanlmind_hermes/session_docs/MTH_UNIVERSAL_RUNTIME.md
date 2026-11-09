# MTH universal native runtime (permanent session contract)

Source contract: NaNROUTER `spec/wave-capabilities.json`,
`docs/HERMES_MTH_UNIVERSAL_TOOLS.md`, `docs/MTH_SYSTEM_REGISTRY.md`, and
NaN0S `aarch64/android/core` + `src/tools/n16i.nas`.

## What ships with the plugin

- The **91-kernel MTH manifest** and every NaN16 kernel/fixture source are
  bundled under `nanlmind_hermes/mth/`. They are architecture-neutral NaN16
  source, not prebuilt per-architecture binaries.
- A small **portable NaN0S C-core source snapshot** and the POSIX-awk `nmk.awk`
  assembler seed ship under `nanlmind_hermes/nan0s/`.
- At first native use the plugin detects the host architecture and compiler,
  then either uses a configured external NaN0S `nan_cli`, a prebuilt CLI, or
  compiles the bundled portable core into its own state cache. The compiled CLI
  is cache-keyed by source digest.

## Tool surface

| Tool | Purpose |
|---|---|
| `nanlm_mth_catalog` | Browse all 91 kernels by family/tags/CI status. |
| `nanlm_mth_inspect` | Exact source/fixture digests, shape, precision, workspace and execution mode. |
| `nanlm_mth_runtime` | Host architecture, compiler, core layout, CLI build state, N16I availability. |
| `nanlm_mth_verify` | Assemble and run one kernel's native fixture; native fixture verification only, not the general-operand path. |
| `nanlm_mth_execute` | General-operand execution for the four admitted float32 vector kernels (dot, cosine, add, sub) through the unified N16I path with 256-element inputs. |
| `nanlm_mth_n16i_selftest` | Differential gate: the same NaN16 image through the native diagnostic/parity helper and under the raw-NaN N16I interpreter-in-NaN16; serial bytes must match. |
| `nanlm_mth_dispatch` | Assemble any bundled kernel or fixture and dispatch the arbitrary guest image under N16I with a bounded step budget and exact receipt. |

## Unified execution policy

- The four admitted float32 vector kernels (`dot`, `cosine`, `add`, `sub`)
  use **N16I as the unified general-operand path** for `nanlm_mth_execute`,
  `mth_leaves` math pairing and identity re-checks, and any other tool path.
  Operands are placed in a generated NaN16 driver, relocated, and interpreted
  by N16I running on the host-built/bundled portable NaN0S core.
- `execute_vector_kernel_native` is retained only as an explicit
  diagnostic/parity helper. It is not the default tool path, and its direct
  native status must not be claimed as the general-operand contract.
- Native fixture verification (`nanlm_mth_verify`) remains native. It verifies
  bundled fixtures and is not a general-operand path.
- All execution is local and derived. It authorizes no NANNET route, lease,
  merge or canonical commit.

## Honest limits

- Catalog discovery is not permission and not a proof certificate.
- 87 kernels have native fixture verification but **no general arbitrary-input
  ABI**. Use `nanlm_mth_verify` for them. Do not claim `nanlm_mth_execute`
  supports them.
- The four `mth.vector.{dot,cosine,add,sub}256.f32` kernels have the admitted
  general-operand dispatcher. Their unified tool and pairing path is N16I, not
  direct native execution.
- N16I is the general-operand path for the four admitted vector kernels and
  the guest interpreter-in-NaN16. Its differential self-test is available, and
  `nanlm_mth_dispatch` relocates and runs arbitrary assembled guest images under
  N16I (kernel fixtures and future guest images), with a bounded step budget
  and exact serial/output receipt. This is dispatch, not a typed operand ABI
  for every kernel.
- Execution is local/derived. It does not authorize a NANNET route, lease,
  merge, canonical head update or mathematical proof.
