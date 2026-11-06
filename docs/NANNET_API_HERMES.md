# NANNET_API_HERMES credential protocol (v1)

The protocol separates Hermes provider credentials, GitHub credentials, and one NANNET installation identity per local universe. Never derive encryption keys from provider or GitHub credentials.

## Installation identity

Provision an independent random 256-bit base64url NANNET_MASTER_KEY through the host OS secret store. nanlm_auth_init creates an Ed25519 installation keypair and encrypts its private key with HKDF-SHA256 + AES-256-GCM. The master key is never returned or stored beside the identity.

## One-time pairing authority

`PairingAuthority` derives a distinct Ed25519 signing key for each identity from the authority master key using separate `identity/admin/v1` and `identity/guest/v1` HKDF domains. Private signing keys remain server-side; only the matching public key is returned during pairing. Pairing codes default to the guest role and are checked against a hard role scope ceiling; admin codes must be explicitly requested by the authority operator. Codes are hashed at rest, expire in 60–3600 seconds, and are consumed transactionally. Pairing binds the installation UUID to its public key and issues a signed token for 30–86400 seconds. Identity revocation is persisted and invalidates every token for that identity.

Allowed scopes are bounded by role. Guest: corpus:write, training:execute, universe:read, snapshot:read. Admin: all listed scopes, including snapshot:write, universe:register, and installation:admin.

## Pairing HTTP endpoint

The optional stdlib HTTP service exposes POST /v1/pair, POST /v1/authorize, POST /v1/revoke, and admin-only POST /v1/admin/identities/revoke, plus the device enrollment routes POST /v1/devices/register, /heartbeat, /rotate, /release, and /report. Device enrollment requires explicit consent and a versioned terms acknowledgement. The authority stores only a SHA-256 digest of the random installation secret and signed guest bearer tokens; each HW guest bearer is signed by its per-device root-derived guest key, is stored only as a digest, and is rotated/revoked on heartbeat, rotation, and release. The authority does not collect MAC addresses, IMEIs, or serial numbers. Each active installation holds one exclusive HW slot. Heartbeats extend leases, rotation invalidates the previous bearer, release returns the slot, and expiry reclaims abandoned slots. A copied device credential cannot hold two simultaneous leases; a newly generated identity cannot be proven to be the same physical device without platform attestation, so the server deliberately does not pretend MAC-based identity is reliable. The authorize route checks a requested scope against a signed token and persisted revocation state. Revocation accepts the token's bearer credential, records its token ID, and causes subsequent authority-side validation to reject it. It must be deployed behind HTTPS; plain HTTP is accepted only for loopback development. The client pins NANNET_ISSUER_PUBLIC_KEY before trusting the returned token. The token is saved encrypted under the installation master key.

For subsequent API requests, send the bearer token and Ed25519 request signature over method, exact path, timestamp, nonce, and SHA-256 body digest. The API must verify token signature, issuer, audience, expiry, scope, installation key, and five-minute clock skew, then atomically consume the nonce. Token revocation is persisted by TokenRegistry.

Payload encryption uses AES-256-GCM with HKDF-SHA256 domain separation and authenticated associated data containing protocol version, installation ID, and object type.

## Demo flow

1. Administrator starts the pairing service with NANNET_AUTHORITY_MASTER_KEY.
2. Administrator creates a short-lived guest pairing code restricted to demo scopes (or explicitly selects `--role admin` for a trusted administrator installation).
3. Installation creates its own NANNET_MASTER_KEY and identity.
4. Installation pairs over HTTPS and verifies the issuer against its pinned public key.
5. The demo uses the issued token, then revokes it and demonstrates rejection.

The local authority, domain-separated per-identity signing keys, role-bounded one-time codes, identity-wide revocation, HTTP pairing and self-revocation endpoints, client pairing/revocation handlers, encrypted token storage, and authority-side token validation are implemented as source components. A public hosted deployment, TLS certificate/operations, and authenticated training/snapshot API endpoints remain deployment/integration work; do not claim a live hosted NANNET service until those pieces are deployed and tested.

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

## Per-installation issuer pin

New tokens are signed by a domain-separated per-identity key. The client must pin the public key for its own installation UUID, not the legacy global signer key. After the client creates its installation identity, the authority operator obtains the matching public key with:

    nannet-authority public-key --role guest --identity INSTALLATION_UUID

The operator communicates this public key to the installation over a trusted channel as NANNET_ISSUER_PUBLIC_KEY. Admin identities use --role admin. The default public-key command without --identity remains the legacy signer key for backwards compatibility only.
