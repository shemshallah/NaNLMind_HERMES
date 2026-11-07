# NaN16DE: raw NaN16 double-endian transport

NaN16DE is the signed wire envelope for NANNET route messages. It is built from
the actual NaN16 word layout, not a parallel JSON representation.

## Word layout

```text
byte 0..1   far head, big-endian
byte 2..5   payload, little-endian
byte 6..7   near head, little-endian
```

The four legal 0xNaN heads are:

```text
0x7FF1  +sNaN
0xFFF1  -sNaN
0x7FF8  +qNaN
0xFFF8  -qNaN
```

The class nibble is:

```text
opcode = (near_class << 2) | far_class
```

There are exactly 16 opcodes; each is one two-input Boolean truth table with
index `2*a+b`. Example: far `+qNaN` (class 2), near `-sNaN` (class 1) gives
opcode `6` (XOR).

## Route

Hermes uses the five-part route:

```text
nan://HW2:BS0:MC0:SC0:GM0
```

`HW`, `BS`, `MC`, `SC`, `GM` are 0–255. `UM` is optional; when omitted, the
wire encodes it as `255` (the reserved NaN/portal interface slot) and displays
the route without `UM`. Explicit `UM0..UM254` is also accepted.

`UM255` is never a user world route.

## Envelope

```text
"N16DE" | version=1 | route[6] | kind-length | kind |
epoch u64 | timestamp u64 | nonce[16] | installation-id-length |
installation-id | NaN16 word[8] | Ed25519 signature[64]
```

The signature covers SHA-256 of the body. The server verifies:

1. bearer token and installation identity;
2. paired installation Ed25519 public key;
3. signature;
4. ±300-second timestamp window;
5. one-time nonce replay table.

## Hermes identity and shared-universe addressing

Hermes uses an explicitly paired NANNET installation identity. Its private
Ed25519 key and paired token are encrypted at rest under the independently
configured `NANNET_MASTER_KEY` Hermes secret. The plugin retrieves secrets only
through `agent.secret_scope.get_secret()`; it does not read `HW_HERMES_API`,
shell environment credentials, or external credential stores. Pairing is
initiated by `nanlm_auth_init` and `nanlm_auth_pair`; no hardware key is
auto-registered.

The topology is one canonical `HW0::BS0` universe: HW0 is RunnerLink and BS0 is
the six-runner ring. HW1 is the phone endpoint, HW2 the Hermes endpoint and HW3
the CLOCKED endpoint. HW4-HW8 are unallocated. HW9+ are scoped participant
addresses attached to the same universe, never separate model universes.

## Portal route

```http
POST https://<portal>/v1/nan16de/route
Authorization: Bearer <paired token>
Content-Type: application/json

{"envelope":"<raw N16DE hex>"}
```

Response includes the canonical route, raw word, opcode, and body digest.

## Human and binary endpoints

The Koyeb service exposes:

- `GET /portal/` — human-readable portal page;
- `POST /n16t` (compatibility alias `POST /portal`) — binary N16T frames with
  content type `application/vnd.nan16.n16t`;
- `POST /v1/nan16de/route` — signed NaN16DE route envelopes;
- `POST /v1/training/merge` — authenticated training delta merge;
- `GET /healthz` — service health.

The portal URL is configuration, not code. Do not put credentials in the human
portal page or repository.
