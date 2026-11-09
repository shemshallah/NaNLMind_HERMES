# NANNET routing (permanent session contract)

Source contract: NaNROUTER `spec/address-grammar.md`,
`docs/BS0_HARDWARE_SLOT_CONTRACT.md`, `docs/HERMES_NANNET_KOYEB_MIGRATION.md`.

## Canonical v1 address grammar

```text
address       = "nan://" context ":" bootstrap ":" head-path
context       = gm ":" sc ":" mc
bootstrap     = "BS" uint ":" uint
head-path     = head [selector]
head          = "HW" uint ":" uint
selector      = "::" selector-item { "::" selector-item }
selector-item = head-list | object-ref
object-kind   = "vector" | "hyperedge" | "rule" | "route" | "bin" | "temporal"
```

Examples:

```text
nan://GM0:SC0:MC0:BS0:0:HW0:0
nan://GM0:SC0:MC0:BS0:0:HW2:0
nan://GM0:SC0:MC0:BS0:0:HW4:0
nan://GM0:SC0:MC0:BS0:0:HW0:0::HW0:1,2,3
nan://GM0:SC0:MC0:BS0:0:HW0:0::HW0:1::vector1337
```

## Resolution rules

1. Parse and type-check every segment before lookup.
2. Resolve the root Bootstrap node through the committed master route cache.
3. Resolve each HW selector to a permanent head/node record. Comma lists
   expand to separate explicit records.
4. Resolve typed object refs through association tables.
5. Verify route generation, target generation, permanence, lease and ACL.
6. Reject stale, ambiguous, unregistered or unauthorized paths. Never fall
   back to raw byte offsets or untyped integer interpretation.
7. Return a signed micro-routing packet containing the exact resolved path and
   provenance.

## Fixed routes

| Purpose | Canonical address | Compatibility link |
|---|---|---|
| Bootstrap root | `nan://GM0:SC0:MC0:BS0:0:HW0:0` | `BS0:0` / `HW0:0` |
| Android/phone | `nan://GM0:SC0:MC0:BS0:0:HW1:0` | `HW1:0` |
| HERMES | `nan://GM0:SC0:MC0:BS0:0:HW2:0` | `nan://HW2:BS0:MC0:SC0:GM0` |
| CLOCKED | `nan://GM0:SC0:MC0:BS0:0:HW3:0` | `HW3:0` |
| WordNet universe | `nan://GM0:SC0:MC0:BS0:0:HW4:0` | `nan://HW4:BS0:MC0:SC0:GM0` |
| Runner ring as a whole | `HW5:0` (registered allocation) | six BSR0..BSR5 members |

The older `nan://HWn:BS0:MC0:SC0:GM0` order is a legacy envelope coordinate.
It must be explicitly translated by a versioned endpoint binding. It is never
a second authority and never proof that a route or lease is active.

## WordNet object references

The WordNet universe is sparse. It does not allocate one 256x256 matrix per
word. Plugin-side references are:

```text
nan://GM0:SC0:MC0:BS0:0:HW4:0::synset/<8-digit-offset>
nan://GM0:SC0:MC0:BS0:0:HW4:0::word/<lemma>/<8-digit-offset>
nan://GM0:SC0:MC0:BS0:0:HW4:0::relation/<type>/<source>/<target>
```

The NaNROUTER v1 grammar currently lists only
`vector|hyperedge|rule|route|bin|temporal` object kinds. `synset`, `word` and
`relation` are a WordNet-local typed-reference extension and must be registered
by the router before a live resolution may claim they are canonical. Never
treat these identifiers as raw byte offsets.

## HERMES N16T / NaN16DE boundary

- HERMES links locally through `HW2:0` (NaN16DE) to
  `BS0:0::HW0:0`.
- The generic `/n16t` service and the HERMES delegation use separate native VM
  session pools and separate durable stores.
- A HERMES dispatch is bound to installation, lease id, universe id, native
  generation and head 0..254. A session id is opaque.
- Dispatch is at-most-once with a durable journal. A pending/ambiguous frame is
  not retried automatically.
- Guest execution is not a canonical model-training commit. Only validated
  candidate deltas may enter the separate merge path.

## LLM routing rules

- Use canonical addresses when describing architecture or requesting a route.
- Never invent a lease, universe id, generation, tick, node point, receipt MAC
  or successful route. If a server-bound value is unavailable, report the
  operation as unavailable and fail closed.
- Do not send corpora, secrets or model execution through Koyeb.
- Treat a textual route as a requested path only.
