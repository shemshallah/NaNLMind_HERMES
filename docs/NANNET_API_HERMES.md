# NANNET_API_HERMES credential protocol (v1)

The protocol separates Hermes provider credentials, GitHub credentials, and one NANNET installation identity per local universe. Never derive encryption keys from provider or GitHub credentials.

## Installation identity

Provision an independent random 256-bit base64url NANNET_MASTER_KEY through the host OS secret store. nanlm_auth_init creates an Ed25519 installation keypair and encrypts its private key with HKDF-SHA256 + AES-256-GCM. The master key is never returned or stored beside the identity.

## One-time pairing authority

`PairingAuthority` persists its Ed25519 issuer key encrypted in SQLite under a separate authority master key. An administrator creates a one-time pairing code with an explicit scope allowlist. The code is hashed at rest, expires in 60–3600 seconds, and is consumed transactionally. Pairing binds the installation UUID to its public key and issues a signed token for 30–86400 seconds.

Allowed scopes: corpus:write, training:execute, universe:read, snapshot:read, snapshot:write, universe:register, installation:admin.

## Pairing HTTP endpoint

The optional stdlib HTTP service exposes POST /v1/pair. It must be deployed behind HTTPS; plain HTTP is accepted only for loopback development. The client pins NANNET_ISSUER_PUBLIC_KEY before trusting the returned token. The token is saved encrypted under the installation master key.

For subsequent API requests, send the bearer token and Ed25519 request signature over method, exact path, timestamp, nonce, and SHA-256 body digest. The API must verify token signature, issuer, audience, expiry, scope, installation key, and five-minute clock skew, then atomically consume the nonce. Token revocation is persisted by TokenRegistry.

Payload encryption uses AES-256-GCM with HKDF-SHA256 domain separation and authenticated associated data containing protocol version, installation ID, and object type.

## Demo flow

1. Administrator starts the pairing service with NANNET_AUTHORITY_MASTER_KEY.
2. Administrator creates a short-lived pairing code restricted to demo scopes.
3. Installation creates its own NANNET_MASTER_KEY and identity.
4. Installation pairs over HTTPS and verifies the issuer against its pinned public key.
5. The demo uses the issued token, then revokes it and demonstrates rejection.

The local authority, one-time code store, token issuer, HTTP pairing endpoint, and client pairing handler are implemented as source components. A public hosted deployment, TLS certificate/operations, and authenticated training/snapshot API endpoints remain deployment/integration work; do not claim a live hosted NANNET service until those pieces are deployed and tested.
