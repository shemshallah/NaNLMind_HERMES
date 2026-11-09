# NANNET → HERMES HW2 production connection

This document describes `scripts/nannet_hw2_connect.py`, a one-shot operator
setup and probe for the permanent HERMES NANNET endpoint. It is not a daemon,
scheduler, trainer, router lease client, or canonical merge path. No live
production authority is contacted by the repository tests, and this document
does not claim that a production HW2 connection is currently live.

## Permanent endpoint identity

| Item | Value |
|---|---|
| NANNET endpoint | `HW2` |
| Canonical address | `nan://GM0:SC0:MC0:BS0:0:HW2:0` |
| Legacy compatibility address | `nan://HW2:BS0:MC0:SC0:GM0` |
| Device-client address form | `nan://GM0:SC0:MC0:BS0:HW2` |
| Node point | `3` |
| NANNET runner-ring port | `BS0:0` |

`HW2` is the permanent HERMES hardware endpoint and maps to node point `3`.
It is **not** local head index `3`. Local head `3` belongs to the model-head
namespace; node points, head indices, hardware-node numbers, and the shared
`BS0:0` runner port are separate namespaces. A successful script run is also
not a NaNROUTER universe lease: the router still owns lease IDs, universe IDs,
generations, route admission, and receipt MACs.

## What the script proves

The script performs this sequence:

1. Validates the authority origin. Production uses HTTPS with normal
   certificate verification. Plain HTTP is accepted only for a loopback
   authority (`127.0.0.0/8`, `::1`, or `localhost`).
2. Loads or creates the local installation identity under
   `<state-dir>/universe/installation.json`. The Ed25519 private key is
   encrypted at rest with the installation master key from
   `--master-key-file`.
3. Loads the pinned issuer Ed25519 public key from `--issuer-key-file` or
   `--issuer-key`. Pairing verifies the authority response against that pin
   before accepting the token.
4. Pairs once through `POST /v1/pair` using `--pairing-code`.
5. Locally verifies the stored token and requires both `universe:read` and
   `training:execute` before making any online authorization request.
6. Optionally asks the existing device client to enroll the permanent HW2
   hardware node (`--register-hw2`).
7. Makes exactly one installation-signed `POST /v1/authorize` request for
   `training:execute`.
8. Prints a sanitized JSON receipt. With `--report PATH`, it also writes the
   same receipt to an owner-only file.

A successful run proves that the configured authority accepted this
installation, that the pinned issuer signed its token, that the token carries
the required guest scopes, and that one signed authorization request succeeded.
It does **not** prove a hosted production deployment, a router universe lease,
HERMES compute access, native model execution, or permission to merge into
canonical BS0/MC0 state. The receipt always reports
`canonical_merge.enabled: false`.

## Required secrets and handling

| Secret | How it is supplied | Rules |
|---|---|---|
| Installation master key | `--master-key-file PATH` | Exactly 32 random bytes, base64url encoded, file mode `0600`. The script rejects group/world-readable files. It never reads `.env` or process environment variables. |
| Issuer pin | `--issuer-key-file PATH` or `--issuer-key KEY` | Public Ed25519 key, 32 bytes base64url. It is public data, but obtain it from a separately trusted channel, never from the same response being verified. |
| Pairing code | `--pairing-code CODE` | One-time, short-lived (60–3600 seconds), and consumed by the authority. Do not paste it into chat, logs, issue trackers, or CI output. |
| Stored token | Written by the script | AES-256-GCM encrypted under the master key in `<state-dir>/universe/nannet_token.json`. No bearer is printed. |

Generate the installation master key locally, for example:

```sh
umask 077
mkdir -p ~/.config/nannet
python3 -c 'from nanlmind_hermes.auth import generate_master_key; print(generate_master_key())' \
  > ~/.config/nannet/hermes-master.key
chmod 600 ~/.config/nannet/hermes-master.key
```

Do not reuse the authority master key as the installation master key. Do not
commit the key file, pairing code, encrypted token, or report to Git. Secrets
are never accepted as positional arguments; unknown/positional arguments are
rejected without echoing their values.

## Production operator steps

1. **Verify the authority origin out of band.** Confirm DNS/TLS for the
   authority host and record the origin, for example
   `https://<verified-authority-host>`. The script refuses plain HTTP for any
   non-loopback host and rejects URLs containing credentials, paths, queries,
   or fragments.
2. **Obtain the issuer public key from a trusted channel.** On the authority
   host, the stable pairing issuer key can be read with
   `nannet-authority public-key` (default guest/legacy identity). Compare a
   short fingerprint with the authority operator over a separate trusted
   channel.
3. **Write the issuer pin file.** Store the base64url public key, for example:

   ```sh
   printf '%s\n' '<issuer-public-key-base64url>' > /secure/path/nannet-issuer.pub
   chmod 644 /secure/path/nannet-issuer.pub
   ```

4. **Create a one-time guest pairing code on the authority.** It must grant at
   least both required scopes:

   ```sh
   nannet-authority create-code \
     --scopes universe:read,training:execute \
     --ttl-seconds 600 \
     --role guest
   ```

   The command prints the pairing code. Keep it out of logs. It can be used
   once. The `hermes:lease` scope is for a separate NaNROUTER universe-lease
   path and is not required by this HW2 connect probe. The current
   `ALLOWED_SCOPES` set has no separate HW2/device scope; the device client's
   bearer is operated with `training:execute`, so the two guest scopes above
   are the smallest set this probe needs.

   If the authority cannot grant both scopes, do not substitute an admin or
   broader code and do not treat a partial token as connected. Ask the
   authority operator to grant the smallest guest set (`universe:read`,
   `training:execute`) or to define a dedicated fixed-HW2 scope, then re-pair.
   The probe fails closed with `error.code: "scope"` and names the missing
   scope(s).

5. **Run the probe from the HERMES installation host:**

   ```sh
   python3 scripts/nannet_hw2_connect.py \
     --authority https://<verified-authority-host> \
     --issuer-key-file /secure/path/nannet-issuer.pub \
     --master-key-file ~/.config/nannet/hermes-master.key \
     --pairing-code '<one-time-code>' \
     --report /secure/path/nannet-hw2-receipt.json
   ```

   Omit `--pairing-code` on later runs to reuse the stored encrypted token
   until it expires. Add `--register-hw2` only if the authority has a real
   fixed-HW2 enrollment path (see the limitation below).

6. **Check the exit code and receipt.** Exit `0` means pairing/authorization
   completed, and `--register-hw2` (if requested) bound HW2. Exit `2` means
   pairing and signed authorization succeeded but the requested HW2
   registration did not bind the permanent endpoint. Exit `1` is a fail-closed
   setup/probe error. A receipt with `ok: false` is not a connection.

## Loopback authority command

Start a disposable real `PairingAuthority` over loopback HTTP in one terminal.
This uses the plugin's real pairing/authorize/device HTTP server and creates a
separate client master key file. It prints paths and the one-time code, but not
the master key value:

```sh
python3 - <<'PY'
import tempfile
from pathlib import Path
from nanlmind_hermes.auth import generate_master_key, _unb64
from nanlmind_hermes.pairing import PairingAuthority
from nanlmind_hermes.http_api import make_server

root = Path(tempfile.mkdtemp(prefix="nannet-hw2-loopback-"))
authority = PairingAuthority(root / "authority.sqlite", _unb64(generate_master_key()))
issuer_path = root / "issuer.pub"
issuer_path.write_text(authority.issuer_public_key() + "\n", encoding="ascii")
master_path = root / "client-master.key"
master_path.write_text(generate_master_key() + "\n", encoding="ascii")
master_path.chmod(0o600)
code = authority.create_pairing_code(
    ["universe:read", "training:execute"], ttl_seconds=600
)["pairing_code"]
server = make_server(authority, "127.0.0.1", 8765)
print("state:", root)
print("issuer-key-file:", issuer_path)
print("master-key-file:", master_path)
print("pairing-code:", code)
try:
    server.serve_forever()
finally:
    server.server_close()
PY
```

In another terminal, run the probe against loopback:

```sh
python3 scripts/nannet_hw2_connect.py \
  --authority http://127.0.0.1:8765 \
  --issuer-key-file '<printed issuer-key-file>' \
  --master-key-file '<printed master-key-file>' \
  --pairing-code '<printed one-time pairing-code>' \
  --report /tmp/nannet-hw2-loopback-receipt.json
```

This is a real loopback HTTP exchange, not a production network connection.
The tests use the same real loopback authority pattern and temporary files.

## Receipt fields

The receipt includes:

- `installation_id` — local installation UUID.
- `issuer.fingerprint_sha256_prefix` — first 16 hex characters of the SHA-256
  of the pinned issuer public key, never the full key.
- `scopes` — scopes on the verified token, which must include
  `training:execute` and `universe:read`.
- `token.valid`, `token.expires_at`, `token.remaining_seconds` — token validity,
  with no token or signature bytes.
- `authorization` — confirms one signed `POST /v1/authorize` request for
  `training:execute`.
- `hw2.address`, `hw2.node_point`, `hw2.network_port` — the permanent endpoint
  identity (`nan://GM0:SC0:MC0:BS0:0:HW2:0`, point `3`, port `BS0:0`), plus the
  registration status when requested.
- `canonical_merge.enabled` — always `false`.

No receipt contains a bearer token, private key, master key, pairing code,
request nonce/signature, or receipt MAC.

## HW2 registration limitation and exact remaining action

The current stdlib authority reserves `HW0`–`HW4` for fixed functions and only
allocates `HW5`–`HW255` through `POST /v1/devices/register`. The existing
`nanlmind_hermes.nannet_device_client.register()` flow correspondingly rejects
any lease in the reserved `HW0`–`HW4` range. Therefore, against the current
stdlib authority, `--register-hw2` cannot honestly succeed:

- If the authority returns a dynamic `HW5`–`HW255` lease, the script releases
  it immediately and reports `hw2.status: "not_hw2"` with the returned address
  and point. It does not relabel that lease as HW2.
- If a future authority returns fixed HW2, the existing device client may
  reject it as a reserved identity; the script reports
  `hw2.status: "blocked_reserved_slot"` rather than claiming success.

In either case the receipt still states the permanent HW2 identity as
`nan://GM0:SC0:MC0:BS0:0:HW2:0`, node point `3`, but registration is not
claimed. The exact remaining operator action is:

> The NANNET authority operator must provision a reserved fixed-endpoint
> binding for this installation to HW2/node point 3 (the paired-device HERMES
> endpoint path), or deploy the dedicated fixed-HW2 enrollment route and update
> the device client to accept only that exact binding. Dynamic
> `/v1/devices/register` does not allocate HW2. Re-run `--register-hw2` after
> that reserved binding exists.

Until then, run the probe without `--register-hw2` to verify pairing, scopes,
and signed authorization only.

## Rotation and revocation

- **Token rotation:** create a new one-time pairing code and re-run with
  `--pairing-code`. The new token replaces the encrypted local token. Revoke
  the old token/identity if it should not remain valid until expiry.
- **Revocation:** from the plugin, call `nanlm_auth_revoke`; it performs a
  signed `POST /v1/revoke` against the pinned authority and erases the local
  encrypted token. An authority administrator can additionally revoke the
  installation identity with the admin-only
  `POST /v1/admin/identities/revoke` route (`role: guest`, the installation
  UUID as `identity`).
- **Installation master-key rotation:** the installation private key is
  encrypted under the current master key, so do not simply replace the key
  file. Revoke the old identity, create a new state directory/identity with a
  newly generated master key, pair it with a new code, then remove the old
  state after confirming revocation.
- **Issuer-key rotation:** a changed authority issuer key intentionally fails
  the pin check. Obtain the new public key through a trusted channel, update
  the pin file, and re-pair. There is no trust-on-first-use fallback.
- **HW device bearer rotation:** the existing device client rotates its bearer
  through `/v1/devices/rotate` or heartbeat and revokes the old bearer. The
  fixed HW2 reserved endpoint must use the authority's reserved-endpoint
  provisioning path, not dynamic HW5–HW255 allocation.
