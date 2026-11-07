# NANNET_API_HERMES credential protocol (v1)

The protocol separates Hermes provider credentials, GitHub credentials, and one NANNET installation identity per local universe. Never derive encryption keys from provider or GitHub credentials.

## Installation identity

Configure an independent random 256-bit base64url `NANNET_MASTER_KEY` in the masked Hermes plugin-secret field. The plugin reads it only through `agent.secret_scope.get_secret`; it never reads `.env`, shell variables, or another credential store. `nanlm_auth_init` creates an Ed25519 installation keypair and encrypts its private key with HKDF-SHA256 + AES-256-GCM. The master key is never returned or stored beside the identity.

## One-time pairing authority

`PairingAuthority` keeps a stable authority Ed25519 issuer key for installation pairing and derives per-identity keys for short-lived device and admin tokens from the authority master key. Private signing keys remain server-side. Pairing tokens are signed by the stable issuer key, and the authority publishes only its public key at `GET /v1/issuer`; the client pins that key before pairing. Pairing codes default to the guest role and are checked against a hard role scope ceiling; admin codes must be explicitly requested by the authority operator. Codes are hashed at rest, expire in 60–3600 seconds, and are consumed transactionally. Pairing binds the installation UUID to its public key and issues a signed token for 30–86400 seconds. Identity revocation is persisted and invalidates every token for that identity.

Allowed scopes are bounded by role. Guest: corpus:write, training:execute, universe:read, snapshot:read. Admin: all listed scopes, including snapshot:write, universe:register, and installation:admin.

## Pairing HTTP endpoint

The optional stdlib HTTP service exposes POST /v1/pair, POST /v1/authorize, POST /v1/revoke, and admin-only POST /v1/admin/identities/revoke, plus the device enrollment routes POST /v1/devices/register, /heartbeat, /rotate, /release, and /report. Device enrollment requires explicit consent and a versioned terms acknowledgement. The authority stores only a SHA-256 digest of the random installation secret and signed guest bearer tokens; each HW guest bearer is signed by its per-device root-derived guest key, is stored only as a digest, and is rotated/revoked on heartbeat, rotation, and release. The authority does not collect MAC addresses, IMEIs, or serial numbers. Each device lease is a scoped participant identity attached to the single `HW0::BS0` universe. HW0 is the runner link, HW1 is the phone endpoint, HW2 is Hermes, and HW3 is CLOCKED; HW4+ is the allocatable participant space. The legacy `hw_slot` column is retained, and active pre-contract leases in the reserved HW0-HW3 range are transactionally moved to free HW4+ addresses without changing device credentials or lease IDs. Heartbeats extend leases, rotation invalidates the previous bearer, release ends the participant lease, and expiry reclaims abandoned leases. A copied device credential cannot hold two simultaneous leases; a newly generated identity cannot be proven to be the same physical device without platform attestation, so the server deliberately does not pretend MAC-based identity is reliable. The authorize route checks a requested scope against a signed token and persisted revocation state. Revocation accepts the token's bearer credential, records its token ID, and causes subsequent authority-side validation to reject it. It must be deployed behind HTTPS; plain HTTP is accepted only for loopback development. The client pins `nannet_issuer_public_key` in Hermes plugin settings before trusting the returned token. The token is saved encrypted under the installation master key.

For subsequent API requests, send the bearer token and Ed25519 request signature over method, exact path, timestamp, nonce, and SHA-256 body digest. The API must verify token signature, issuer, audience, expiry, scope, installation key, and five-minute clock skew, then atomically consume the nonce. Token revocation is persisted by TokenRegistry.

Payload encryption uses AES-256-GCM with HKDF-SHA256 domain separation and authenticated associated data containing protocol version, installation ID, and object type.

## Demo flow

1. Administrator starts the pairing service; the standalone CLI securely prompts for the authority master key without reading it from an environment variable.
2. Administrator creates a short-lived guest pairing code restricted to demo scopes (or explicitly selects `--role admin` for a trusted administrator installation).
3. Installation configures its own `NANNET_MASTER_KEY` as a Hermes-managed secret and creates its identity.
4. Installation pairs over HTTPS and verifies the issuer against its pinned public key.
5. The demo uses the issued token, then revokes it and demonstrates rejection.

The local authority, stable pairing issuer, domain-separated per-identity
signing keys, role-bounded one-time codes, identity-wide revocation, HTTP
pairing and self-revocation endpoints, client pairing/revocation handlers,
encrypted token storage, and authority-side token validation are implemented.
A hosted authority has been live-verified for pairing, online scope
authorization, and revocation. Head-cortex deployment and authenticated
training/snapshot API endpoints remain separate integration work.

## Signed request enforcement (implemented)

After pairing, the Hermes client signs each authority request using its installation Ed25519 private key. The canonical message is:

    UPPERCASE_METHOD
    EXACT_PATH
    UNIX_TIMESTAMP
    ONE_TIME_NONCE
    SHA256_HEX_OF_EXACT_BODY_BYTES

The request carries X-NANNET-Timestamp, X-NANNET-Nonce, X-NANNET-Body-SHA256 and X-NANNET-Signature alongside the bearer token. The authority looks up the public key bound during pairing, checks the five-minute clock-skew window and body digest, verifies Ed25519, and atomically consumes the nonce. Replayed signed requests are rejected. Hermes uses this for POST /v1/authorize and POST /v1/revoke; the one-time /v1/pair exchange is authenticated by its single-use pairing code and pinned issuer key.

The GitHub Actions BSR OIDC bootstrap is a separate service identity flow and does not use a paired-installation signing key.

## Cluster / Mesh-LLM boundary

The NaNLMind Mesh-LLM Cluster Universe document is an architecture proposal, not a deployed API contract. Its cluster create/simulate/schedule routes and Mesh-LLM management integration are not implemented by this Hermes plugin or the current Koyeb portal. Do not send training corpora or model execution through Koyeb. A future cluster orchestrator may use NANNET-authorized worker agents and Mesh-LLM as a data plane, while semantic updates still enter the single global commit path with base hashes, idempotency and ledger validation.

## Stable issuer pin

Installation pairing tokens are signed by the stable authority issuer key. The
client pins that key as `nannet_issuer_public_key` before pairing and rejects a
response whose issuer key does not match. The operator reads the key from:

    GET /v1/issuer

Per-identity derived keys remain in use for short-lived device and admin
tokens; those tokens are validated online by the authority. The plugin does not
need a per-installation pin for the pairing token.

## CLOCKED API credential separation

The Rezolve.ai CLOCKED API is a separate external service and its official
SDK documents a dedicated CLOCKED API key (normally supplied as
`CLOCKED_API_KEY`). This credential is not a NANNET installation token or a guest participant token.
Never substitute one credential for another or place a CLOCKED key in a
NANNET-signed request payload, prompt, transcript, scratch lane or Git commit.

The proposed HW3:0 gateway may provide a sponsored demo proxy only if the
CLOCKED service owner explicitly authorizes that arrangement. Otherwise users
must configure their own CLOCKED API key through the host secret store. NANNET
pairing by itself does not grant CLOCKED account access or credits.
