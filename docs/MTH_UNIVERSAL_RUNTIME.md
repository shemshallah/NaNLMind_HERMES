# Universal native MTH runtime

Status: **implemented and locally exercised on Termux/aarch64.** This is a
local/derived execution surface, not a NANNET route, lease, merge or proof
certificate.

## What ships

| bundle | contents |
|---|---|
| `nanlmind_hermes/mth/spec/wave-capabilities.json` | the 91-kernel NaNROUTER MTH manifest with bundle-relative paths |
| `nanlmind_hermes/mth/nan16/*.nas` | every NaN16 kernel source |
| `nanlmind_hermes/mth/tests/*.nas` | every native fixture |
| `nanlmind_hermes/nan0s/core/*` | portable C NaN16/NaNVM1 core source (Apache-2.0 NaN0S snapshot) |
| `nanlmind_hermes/nan0s/tools/nan_cli.c` | headless native image runner |
| `nanlmind_hermes/nan0s/tools/nmk.awk` | POSIX-awk NaN16 source emitter |
| `nanlmind_hermes/nan0s/src/tools/n16i.nas`, `disasm.nas` | N16I interpreter-in-NaN16 and disassembler |
| `nanlmind_hermes/nan0s/src/tests/n16i_test.nas` | N16I differential self-test driver |

There are no prebuilt architecture-specific binaries in the repository. The
NaN16 kernels are architecture-neutral source; only the thin host VM is native.

## Autodetection and setup

At first native MTH use:

1. `platform.machine()` is mapped to `aarch64`, `x86_64`, `x86`, `riscv64`,
   `ppc64le`, `s390x`, or `portable-c`.
2. A configured `nan0s_root` is used if present.
3. Otherwise the plugin checks a sibling/home `nan0s` checkout, then the bundled
   portable source snapshot. `nan0s_prefer_bundled: true` forces the bundled
   path.
4. A prebuilt external `aarch64/android/nan_cli` is used when executable;
   otherwise the portable core is compiled into
   `state_dir/nan0s/<target>/nan_cli` with `cc`, `gcc` or `clang`.
5. The build is keyed by a SHA-256 of the core/tool sources and records the
   compiler, architecture and timestamp in `build.json`.

No separate NaN0S install is required when a C compiler is available. The
guest kernels always remain NaN16 source. The unified N16I general-operand path
for the four vector kernels runs on this same host-built/bundled core. The
runtime also requires a compatible awk: `gawk` or `mawk`; BusyBox `awk` is
rejected after a probe because it mis-emits HALT words. On Alpine, use:

```sh
apk add --no-cache build-base gawk
```

Use `nanlm_mth_tree action=map` to link all 91 kernels into the immutable
function/data-leaf tree, and `action=audit` to verify coverage, digests, hub
ancestry and every leaf's NaN16 image.

## Tools

| tool | parameters | effect |
|---|---|---|
| `nanlm_mth_catalog` | `family`, `tags`, `offset`, `limit`, `require_ci` | page through all 91 kernels |
| `nanlm_mth_inspect` | `kernel_id` | source/fixture digests, shape, precision, workspace, execution mode |
| `nanlm_mth_runtime` | `build` | architecture, core layout, compiler, native CLI state, N16I availability |
| `nanlm_mth_verify` | `kernel_id`, `max_steps`, `timeout` | assemble and run the kernel's native fixture; native verification only, not the general-operand path |
| `nanlm_mth_execute` | `kernel_id`, `x[256]`, `y[256]`, `timeout` | unified N16I general-operand execution for dot/cosine/add/sub float32 vector kernels |
| `nanlm_mth_n16i_selftest` | `timeout` | native diagnostic/parity helper vs N16I serial-byte differential gate |
| `nanlm_mth_dispatch` | `kernel_id`, `use_fixture`, `entry`, `max_steps`, `timeout` | assemble any bundled kernel/fixture and run the arbitrary guest image under N16I |

All execution tools run through the same `training:execute` local authority
check. With no NANNET authority configured they report `local-unverified`;
with a configured authority they require a paired token.

## Unified execution policy

The four admitted float32 vector kernels — `mth.vector.dot256.f32`,
`mth.vector.cosine256.f32`, `mth.vector.add256.f32` and
`mth.vector.sub256.f32` — use **N16I as the unified general-operand execution
path**. This applies to `nanlm_mth_execute`, `mth_leaves` math pairing and its
identity re-checks, and any other tool path that executes these kernels with
real operands.

Operands are written into a generated NaN16 driver, assembled with the bundled
`nmk.awk`, relocated to the guest window, and run under N16I on the
host-built/bundled portable NaN0S core. The receipt records the N16I path;
results and status come from that execution, not host Python math.

`execute_vector_kernel_native` is retained only as an explicit diagnostic and
parity helper. It is not the default tool path, and direct native execution
must not be claimed as the general-operand contract.

`nanlm_mth_verify` remains native fixture verification. It is not the
general-operand path. Execution is local/derived; it authorizes no NANNET
route, lease, merge or canonical commit.

## Execution modes

| mode | status | scope |
|---|---|---|
| `native_nan0s` | **available** | all 91 native fixtures; `nanlm_mth_verify`; explicit native diagnostic/parity helper for the four vector kernels |
| `n16i_guest_interpreter` | **available** | unified general-operand path for the four vector kernels; arbitrary assembled guest images via `nanlm_mth_dispatch`; self-test differential gate |
| `portable_c_source` | **available** | host build of the core that runs N16I and the native fixtures |
| `python_nanvm0` | limited | phone-call NaNVM0 boundary, not the MTH kernels |

## Existing local evidence

The records below are existing local evidence; this policy change adds no new
live test numbers. The vector numbers come from the native diagnostic/parity
helper, not from a default N16I run.

- catalog: **91/91** kernels, **91** with `ci_verified: true` and
  `status: native_verified` in the shipped manifest.
- native fixture: `mth.k256.laplacian.u16` assembled and ran **5,391 steps**,
  printed `MTH K256 LAPLACIAN PASS`, no trap.
- native diagnostic/parity vector execution with x=2.0, y=3.0 for all 256
  elements:
  - add = **5.0** (first and last elements)
  - sub = **-1.0**
  - dot = **1536.0**
  - cosine = **1.0**
  - direct native status register = **1** for each.
- N16I differential: native **73,982 steps** vs interpreted
  **3,343,783 steps**, **2,640 identical serial bytes**, no trap.
- arbitrary N16I dispatch: the `mth.k256.laplacian.u16` fixture was relocated
  to a host guest base, run under `n16i_run` for **245,464 steps**, produced
  the identical `MTH K256 LAPLACIAN PASS` marker and no trap.

## Honest limits

- A catalog entry is not numerical proof and not permission.
- 87 kernels have native fixtures but no general arbitrary-operand ABI. Use
  `nanlm_mth_verify`; do not claim `nanlm_mth_execute` supports them.
- The four vector kernels are the admitted `mth_execute.nas` program/operator
  allowlist (900/77, 901/78, 902/79, 903/80). Their unified tool and pairing
  path is N16I; direct native execution is only the explicit diagnostic/parity
  helper.
- N16I is the unified general-operand path for the four admitted vector kernels
  and the generic guest-image dispatcher. Outside those four it still has no
  typed operand ABI per kernel; a caller must supply or assemble a guest
  program whose register/memory contract it knows. The runtime reports this
  limit instead of silently substituting Python.
- Execution (N16I or native diagnostic) never authorizes a NANNET route, lease,
  merge, canonical head update or mathematical proof.
- The portable core is a source snapshot of the NaN0S Android/aarch64 C core.
  Cross-architecture builds are expected to work but are only locally exercised
  on aarch64; run `nanlm_mth_verify` on each target before claiming that target.
