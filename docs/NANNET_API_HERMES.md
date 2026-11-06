# NANNET_API_HERMES credential protocol (v1)

This is the per-installation authorization layer for NaNLMind Hermes. It is separate from Hermes provider credentials and GitHub credentials.

## Credential domains

1. **Hermes provider credential** belongs to the Hermes host/provider adapter. NaNLMind Hermes does not read, copy, wrap, or encrypt it.
2. **GitHub credential** belongs to the user's GitHub integration or narrowly scoped PAT. It is not a universe credential and is never put in a token.
3. **NANNET installation identity** is one Ed25519 keypair and UUID per local installation/universe. The private key is encrypted at rest with HKDF-SHA256 + AES-256-GCM using an independent 256-bit NANNET_MASTER_KEY.
4. **NANNET_API_HERMES token** is a short-lived Ed25519-signed bearer authorization issued by a separate trusted pairing authority. Claims bind issuer, audience, installation ID, scopes, issue/not-before/expiry times, and a unique token ID.

Never derive the encryption key from a provider API key, GitHub PAT, or bearer token. Do not store those credentials in corpus files, prompts, logs, universe state, or snapshots.

## Installation setup

Generate a fresh independent master key with generate_master_key() and place it in the host OS secret store or a protected environment variable named NANNET_MASTER_KEY. Decode it to bytes before calling create_installation. Do not commit it, print it in CI, or store it beside the encrypted identity. The plugin refuses to overwrite an existing installation identity.

The first implementation deliberately exposes cryptographic primitives and persistent replay/revocation storage, not a pretend hosted service. A pairing authority must authenticate a one-time pairing code out of band, verify the installation public key, issue a scoped token, and retain its signing key outside the installation. The authority's public key is pinned by each client.

## Token scopes

- corpus:write
- training:execute
- universe:read
- snapshot:read
- snapshot:write
- universe:register
- installation:admin

Tokens are Ed25519 signed, audience-restricted, and expire in 30 seconds to 24 hours (recommended lifetime: 15 minutes). The client signs each request over method, path, timestamp, nonce, and SHA-256 body digest. The server must enforce a five-minute clock-skew window and atomically consume each (installation_id, nonce) pair. TokenRegistry provides SQLite-backed nonce uniqueness and token revocation persistence.

## Payload protection

Snapshot/corpus payload encryption uses AES-256-GCM with a per-envelope random salt and nonce and HKDF-SHA256 domain separation. Supply stable associated data (protocol version, installation ID, and object type). Changing ciphertext or associated data fails closed.

## Demo flow and limitations

1. Create one local installation identity and keep its master key outside the repository and state directory.
2. Pair its public key with the demo authority using a one-time code.
3. Receive a short-lived token scoped only to the actions needed by the demo.
4. Sign requests with the installation key; verify token, scope, signature, timestamp, and replay status server-side.
5. Revoke the token and demonstrate rejection.

There is not yet a deployed pairing/authentication server or remote NANNET API. Until one is deployed and its issuer public key is pinned, do not describe local token creation as remote authentication. This repository provides tested client and authority-side primitives for that next service.
