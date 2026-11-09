# Unified build adapter

NaNLMind is the canonical native architecture. This repository is not a second
build, a second topology, or a second source of truth.

```text
NaNLMind/spec/unified-build.v1.json   canonical build and topology contract
NaNLMind/build/nanlm-universal        canonical native binary
NaNLMind unification                  machine-readable contract line
NaNLMind_HERMES/nanlmind_hermes/unification.py
                                      read-only adapter
```

## What HERMES reads

`unification.status()` looks for the contract at:

1. the explicit `nanlmind_root` contract path supplied by the caller;
2. the sibling checkout `../NaNLMind/spec/unified-build.v1.json` when present;
3. `./NaNLMind/...` and `~/NaNLMind/...` as convenience fallbacks.

The `nanlmind_root` plugin setting takes precedence for `/nanlm status` when it
is non-empty. The adapter never writes the contract or the native state.

When `build/nanlm-universal` exists, the adapter also runs:

```sh
./build/nanlm-universal unification
```

and rejects the build if its heads, triads, role views, pair count, or chart
bytes disagree with the contract. The check is a compatibility gate, not a
claim that every open research gate has passed.

## Three role views

The 256 physical heads support three overlays on one chart:

| view | count | notes |
|---|---:|---|
| router control roles | 250 curriculum heads | head 0 is the NANNET root role; `0,4,5,6,7,8` are BSR control heads |
| MP writer eligibility | 255 heads | `1..255`; existing authorization still applies |
| native triad field | 255 triad members + field head 255 | heads `0..254` vote in 85 triads; head 255 aggregates |

The overlays are not additional charts, heads, universes or permission grants.
A router role cannot reassign the native triad field head, and the NANNET port
is not represented as a native model head.

## Status surface

`/nanlm status` includes `unified_build` with:

- `status`: `green`, `unavailable`, `error`, or a native `rejected` state;
- `canonical_repository` (always `NaNLMind`);
- `contract_path` and `contract_sha256`;
- `topology` and `role_views` from the canonical contract;
- `native` status from the canonical binary, when built;
- `hermes_is_canonical: false`.

If the contract is missing, the adapter reports `unavailable`; it does not
silently substitute hardcoded constants. Tests use a temporary contract and a
fake native executable to verify both acceptance and rejection.

## Scope

This is build/authority unification only. The bundled 91-kernel MTH snapshot,
local resident store, SQLite memory, caches and transcripts remain derived or
compatibility artifacts. They are not the canonical NaNLMind universe, and this
adapter makes no claim of live deployment or fluent-language performance.
