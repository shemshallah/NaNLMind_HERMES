# Alpine / proot testing on Android

Termux package installation is not required for the HERMES Python test suite.
On an Android/Termux host with `proot-distro`, use a disposable Alpine rootfs.

## Setup

```sh
proot-distro install alpine:3.20
proot-distro login alpine -- /sbin/apk add --no-cache python3 sqlite-libs
# Native MTH tests additionally need a portable-C compiler and a compatible awk.
# Without these, the MTH runtime tests skip and the status reports the setup gap.
proot-distro login alpine -- /sbin/apk add --no-cache build-base gawk
```

The Alpine image may already contain a Python, but the Termux `python3` can
shadow it on PATH. Use the explicit Alpine path `/usr/bin/python3`.

## Run the full suite

```sh
proot-distro login --minimal alpine \
  -b /path/to/NaNLMind_HERMES:/hermes -- \
  /usr/bin/python3 -c '
import os, sys, unittest
os.chdir("/hermes")
sys.path.insert(0, ".")
suite = unittest.TestLoader().discover("tests")
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
'
```

`--minimal` is important on some Android kernels: the default proot binding
profile fails on `fchdir`/`getcwd` with `Function not implemented`.

## Result

On the Termux/Android test host, the full current suite passes:

```text
OK (0 skipped with build-base + gawk)
```

The suite currently contains at least 369 test methods; the exact count is not
pinned here because it changes as tests are added.

With only `python3 sqlite-libs`, the six compiler-backed MTH runtime tests skip
and the native-build-unavailable test still verifies the fail-closed setup
message.

This includes the cryptography-backed and compatibility crypto paths, chat loop,
memory, native runner harness, pairing, auth, HW4+ lease migration, device
routes, HTTP API tests, permanent session-context loading, free-chat layered
learning, canonical WordNet HW4:0 ingestion, the seven permanent session
documents, the bundled 91-kernel MTH catalog (44 families), MTH data leaves and
function-tree audit, portable NaN0S native fixture execution, the four vector
N16I operand dispatchers, and the N16I differential self-test. The production manifest declares
`cryptography>=42,<47`; the compatibility fallback remains available for
isolated constrained-host tests.

The Termux-built `nanlm-universal` binary targets Android/bionic and is not
directly executable inside Alpine/musl. Build the native core inside Alpine or
run the native adapter on the Termux side.
