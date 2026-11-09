# N16I unification for the admitted vector kernels

This document defines the unified general-operand path for the four admitted
float32 vector kernels:

- `mth.vector.dot256.f32`
- `mth.vector.cosine256.f32`
- `mth.vector.add256.f32`
- `mth.vector.sub256.f32`

All four now share one N16I execution path for real 256-element operands. The
direct native runner remains available as an explicit diagnostic helper, but it
is no longer the public/default execution route.

The acceptance test is `tests/test_n16i_unification.py`. It is the source of
truth for this contract; this document does not embed live run output.

## Two entry points, one operand ABI

| function | execution | reported mode | `n16i_used` | role |
|---|---|---|---|---|
| `nan0s_runtime.execute_vector_kernel_native(ctx, kernel_id, x, y, timeout=...)` | compiled native NaN0S core | `native_nan0s` | `False` | differential/diagnostic helper |
| `nan0s_runtime.execute_vector_kernel(ctx, kernel_id, x, y, timeout=...)` | bundled N16I interpreter guest | `n16i_guest_interpreter` | `True` | unified public path |

Both functions accept the same operands: two lists of exactly 256 numeric
values, interpreted as float32. Both return the kernel status register, the
numeric result (`float` for dot/cosine, 256 values for add/sub), the execution
metadata, and boundary flags. The unified result additionally reports
`native: False` so a caller cannot mistake an interpreted guest run for the
direct native path.

Internally the unified path:

1. validates the kernel id and both 256-element float32 operand lists;
2. emits the same NaN16 driver used by the native helper, including the x, y,
   output, scratch and status buffers;
3. assembles the driver with the bundled emitter and compatible awk;
4. relocates the assembled guest image to a host guest base and enters the
   bundled `n16i.nas` interpreter (`dispatch_image_n16i`);
5. reads the guest result/status words back through the native core's bounded
   serial/peek surface and normalizes them into the result dictionary.

The native helper skips step 4 and runs the assembled driver directly. It
exists so tests and operators can compare both paths with identical operands
and identical guest programs.

## Tool and leaf integration

- `nanlm_mth_execute` calls `execute_vector_kernel`, so its response carries
  `execution_mode: "n16i_guest_interpreter"`, `n16i_used: true` and
  `native: false`. The tool does not silently fall back to the native helper.
- `mth_leaves` math pairing (`add`, `sub`, `dot`, `cosine`) uses the same
  unified function. The result leaf records the execution in
  `function_definition.execution`, including
  `mode: "n16i_guest_interpreter"` and `n16i_used: true`. Input leaves remain
  immutable and unchanged.

The `nanlm_mth_execute` tool still requires the local `training:execute`
authority check. With no NANNET authority configured it reports
`local-unverified`; configuring an authority makes the paired token mandatory.

## Native diagnostic helper

`execute_vector_kernel_native` is intentionally kept. It provides:

- a direct native reference for differential checks against N16I;
- a way to separate kernel/assembler failures from interpreter failures;
- the existing native execution mode for callers that explicitly ask for it.

It is not a fallback for the unified API. A failing N16I path is reported as a
failure rather than being hidden by a native re-run.

## Test coverage

`tests/test_n16i_unification.py` skips when no C compiler is available. The
runtime separately requires a NaN16-emitter-compatible awk (`gawk` or `mawk`);
BusyBox `awk` is rejected by the runtime probe because it mis-emits HALT words.

For every admitted kernel the suite runs native and N16I with `x` set to 256
elements of `2.0` and `y` set to 256 elements of `3.0`, and checks:

| check | requirement |
|---|---|
| status | `status == 1` and `ok` is true on both paths |
| native metadata | `execution_mode == "native_nan0s"`, `n16i_used` false |
| unified metadata | `execution_mode == "n16i_guest_interpreter"`, `n16i_used` true, `native` false |
| numeric expectation | add `5.0`, sub `-1.0`, dot `1536.0`, cosine `1.0` |
| agreement | native and N16I results agree within `1e-5` |

The same module also checks:

- a registered plugin tool context calling `nanlm_mth_execute` for add and
  receiving the unified N16I metadata;
- two vector leaves created from `{"fill": 1.0}` and `{"fill": 2.0}`, paired
  with `add`, producing a result leaf whose
  `function_definition.execution.mode` is `"n16i_guest_interpreter"` and whose
  `n16i_used` flag is true, with both input leaves unchanged.

The tests do not call merge, promotion, lease or NANNET tools.

Run:

```sh
python3 -m unittest tests.test_n16i_unification -v
```

A pass covers these four kernels, these operands and that host's build. It is
not a proof for arbitrary operands or other kernels.

## Honest limits

- **N16I runs on the native core.** The N16I interpreter is a NaN16 guest image
  executed by the same compiled portable NaN0S core. Unification changes which
  guest program runs; it does not create an independent hardware engine, a
  second ISA implementation, or a formally verified interpreter.
- **No canonical commit.** Neither path authorizes or produces a NANNET route,
  lease, merge, canonical head update, training commit or mathematical proof.
  Results are local execution evidence only.
- **Four kernels only.** The other 87 catalogued kernels remain fixture-only or
  lack a general arbitrary-operand ABI. Use `nanlm_mth_verify` for them and do
  not claim `nanlm_mth_execute` support.
- **Float32 and bounded execution.** Operands are float32; the `1e-5` tolerance
  describes the test comparison, not a universal numerical guarantee. N16I runs
  are bounded by step and timeout limits and are slower than direct native
  execution.
- **Environment-dependent.** The tests require a C compiler and a compatible
  awk. Passing them on one host does not by itself establish the same result on
  every target; run the suite on each target before claiming it there.
- **Not a correctness proof.** Agreement between native and N16I for the tested
  operands demonstrates consistency of this execution path, not correctness of
  the kernels for all inputs.
