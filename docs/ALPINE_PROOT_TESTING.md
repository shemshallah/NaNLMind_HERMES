# Alpine / proot testing on Android

Termux package installation is not required for the HERMES Python test suite.
On an Android/Termux host with `proot-distro`, use a disposable Alpine rootfs.

## Setup

```sh
proot-distro install alpine:3.20
proot-distro login alpine -- /sbin/apk add --no-cache python3 sqlite-libs
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

On the Termux/Android test host:

```text
Ran 65 tests in 22.230s
OK
```

This includes the cryptography-backed and compatibility crypto paths, chat loop,
memory, native runner harness, pairing, auth, HW9+ lease migration, device
routes, and HTTP API tests. The production manifest declares
`cryptography>=42,<47`; the compatibility fallback remains available for
isolated constrained-host tests.

The Termux-built `nanlm-universal` binary targets Android/bionic and is not
directly executable inside Alpine/musl. Build the native core inside Alpine or
run the native adapter on the Termux side.
