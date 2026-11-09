# NANNET architecture (permanent session contract)

Source contract: NaNROUTER `README.md`,
`docs/BS0_HARDWARE_SLOT_CONTRACT.md`, `docs/NANNET_NODE_BINDING.md`,
`spec/address-grammar.md`, `spec/point-map.toml`.

## One universe, one network node

- NaNLMind is one computational universe with 256 local heads.
- A universe has exactly one logical NANNET node/port. The node is not a head,
  does not own wave state, and receives no learning ACL.
- The local ring model is `255 heads + 1 NANNET node = 256 positions`.
- Local head indices, node points, hardware-node numbers, BSR worker ids and
  memory-object ids are separate namespaces. Never store them in one untyped
  integer field.

## Bootstrap coordinates

- `BS0:0` is the shared NANNET runner-ring ingress/port.
- `HW0:0` is the separate attached Bootstrap hardware root.
- `nan://GM0:SC0:MC0:BS0:0` is the port/control-plane route.
- `nan://GM0:SC0:MC0:BS0:0:HW0:0` is the canonical Bootstrap root address.
- Six BSR/BSR0..BSR5 worker identities are internal to BS0. They do not consume
  six HW slots and are not separate hardware nodes.
- Point 0 (`BS0:0`, wire BSR0) is the stable runner-ring ingress. It translates
  NaN16 to binary transport, routes to the active runner slot, and translates
  back. It is not a head or a second math owner.

## BS0 hardware allocation

| HW node | Permanent function | Public allocation |
|---|---|---|
| HW0 | Bootstrap hardware root | never |
| HW1 | Android/phone endpoint | no, paired-device only |
| HW2 | HERMES endpoint | no, paired-device only |
| HW3 | CLOCKED endpoint | no, paired-device only |
| HW4 | WordNet universe | never |
| HW5-HW255 | registered hardware nodes | yes, 251 slots |

Other bootstrap classes BS1..BS255 have independent HW0..HW255 namespaces.
An attached 8 MiB universe has its own local heads; its NANNET port is not
`BS0:0` or `HW4`.

## Authority boundary

- NANNET owns registration, installation identity, pairing, scope
  authorization, route admission and merge coordination.
- NaNROUTER owns signed route-cache generations, per-installation HERMES lease
  ledgers, receipt MACs, dispatch journals and node bindings.
- NANLMind owns the canonical learner and its commit gates.
- The HERMES plugin owns only local/derived state and candidate evidence.
- A successful login/pairing is not a universe lease. A hardware registration
  is not model-state access. A requested textual route is not ownership.
- Server-bound values (lease id, universe id, generation, trusted tick, node
  point) must come from authenticated router context, never from request JSON
  or from the attached LLM.
- Unknown, stale, unregistered or unauthorized routes fail closed. There is no
  fallback to raw address interpretation.
- Every runner-ring handoff and point-0 request carries the JAHS handoff seal
  where the NaNROUTER contract requires it.

NANNET is not the model runner, trainer or corpus store. Koyeb is the
persistent ingress/control plane, not a training location.

## HERMES identity

- HERMES uses the permanent `HW2` hardware endpoint and node point 3. Point 3
  is not local head 3.
- A HERMES guest universe is an independently leased universe with its own
  local heads 0..254 and a reserved local NANNET port at position 255.
- Multiple leased HERMES universes may share node point 3 but must never share
  a guest topology, N16T session pool, lease id or receipt generation.
