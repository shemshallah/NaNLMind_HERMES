# HERMES → NANNET login → NaNROUTER per-lease universe

This is a staged integration, **not** a claim that HERMES already receives
a live remote universe lease. The native NaN16DE guest lifecycle is now in
`nanlmind_hermes/nan/hermes_lease.nas`, with regression coverage
`hermes_lease_test.nas` and the GitHub VM gate.

## Intended trust and traffic path

```text
HERMES local plugin / local NaN16DE guest
  → NANNET public Koyeb portal (installation pairing and authorization)
  → NaNROUTER authoritative lease / signed route / object registry
  → distinct HERMES 8 MiB lease universe
  ↔ NANNET network port (one per universe; not a learning head)
  → authorized N16T/N16B traffic + independently validated learning evidence
```

NANNET's established global HERMES identity is
`nan://GM0:SC0:MC0:BS0:HW2`, local bootstrap node point 3. That
stable endpoint address is **not** the ephemeral leased `universe_id`
and neither is any head index. The extended canonical NaNROUTER syntax
is `nan://GM0:SC0:MC0:BS0:0:HW0:0::HW2:24::vector1337`
(see `NaNROUTER/spec/address-grammar.md`).
An address string or an unsigned N16DE envelope is not authority.

Each new authorized HERMES lease gets its **own** universe ID, its own
255 addressable head indices 0..254 and one network port at 255. Lease
generation, deadline, owner and node binding must be checked before
accessing that universe's heads. The HERMES universe is **not** an
alias of NaNLMind's canonical shared 256-head learning state.
NaNLMind's six CPU worker repositories remain six workers for that
other canonical universe, not six independent model owners.

## Existing, usable login mechanisms

The plugin already exposes `nanlm_auth_init`, `nanlm_auth_pair`,
`nanlm_auth_status`, `nanlm_auth_revoke`, and scope-checked calls.
Configure:

- `nannet_auth_url`: HTTPS origin of the **verified** NANNET portal.
  The NaNLMind deployment currently defaults to
  `https://nannet.apps.mistral.run`; an alternative hostname
  (`nannet.app.mistral.com`) must be validated via DNS, TLS and
  issuer key before selecting it.
- `nannet_issuer_public_key`: pin the NANNET pairing issuer
  Ed25519 public key from a separately trusted channel.
- `NANNET_MASTER_KEY`: a **locally generated encryption key** held in
  the Hermes-managed secret store. It is not the portal administrator
  secret, GitHub token, or a Koyeb API token.

`nanlm_auth_pair` sends `POST /v1/pair` and verifies the response
against the pinned issuer. Scope verification calls `POST
/v1/authorize` with an installation-signed request. Those flows are
preexisting host-side credential/HTTPS adapters; they do **not**
instantiate a remote universe, and they are not moved into Python or C
model logic. Guest compute and head state transitions are written in
NaN16DE.

## Native lifecycle

`nhl_open`: create a lease entry after a separately authenticated
and router-admitted universe allocation.

`nhl_resolve`: fetch only a live, owned, generation-matched
local head. Head 255 is the NANNET network port, never a learning head.

`nhl_renew`: extend an existing lease only when it is still live
and the new deadline is strictly later.

`nhl_release`: revoke without allowing a stale generation to revoke
a later lease in the same registry slot.

The registry currently fits **16 concurrent guest leases**, not the
full global router. A supplied owner tag is a guest consistency check,
not a replacement for cryptographic authentication. Every deadline
and lease generation must come from NaNROUTER's trusted context,
rather than untrusted chat/tool input.

## Explicit integration blockers

1. The current guest scope ceiling does not grant
   `universe:register`; the new router allocation must have its own
   narrowly delegated authorization flow rather than giving the
   HERMES guest installation admin privileges.
2. The existing plugin host still manages local canonical state in
   its legacy training paths. Do **not** advertise these as
   router-leased isolated universes until wiring is complete.
3. NaNROUTER's `/v1/nmp/lease` is a worker execution lease, **not**
   a HERMES-universe lease. Router needs durable universe allocation,
   generation/replay protection, a signed receipt, and node binding.
4. The portal currently runs from the NaNLMind Koyeb deployment.
   Move it to NaNROUTER only after a verified database, issuer-key,
   and ingress migration with no concurrent writers.
5. A verified route may transport data; it does not confer the
   privilege to mutate canonical NaNLMind learning state.

See NaNROUTER PR #61 and
`NaNROUTER/docs/HERMES_NANNET_KOYEB_MIGRATION.md` for the
authoritative native handoff and deployment staging contract.
