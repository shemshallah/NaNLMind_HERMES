# NaNLMind Hermes

A Hermes plugin for a local NaNLMind 256-head universe.

See docs/NANNET_API_HERMES.md for the installation identity and authorization protocol.


## Local pairing authority

Install the project, provision an independent 32-byte base64url `NANNET_AUTHORITY_MASTER_KEY` in the authority host's secret store, then:

```sh
nannet-authority public-key
nannet-authority create-code --scopes corpus:write,training:execute,universe:read --ttl-seconds 600
nannet-authority serve --host 127.0.0.1 --port 8765
```

For remote deployment, use a TLS certificate and key and bind only behind a trusted HTTPS deployment boundary. Non-loopback listeners refuse to start without TLS. Provision the installation's separate `NANNET_MASTER_KEY`, pin the authority's public key in `NANNET_ISSUER_PUBLIC_KEY`, set `NANNET_AUTH_URL`, initialize with `nanlm_auth_init`, then pair with `nanlm_auth_pair`. The resulting token is stored encrypted in the installation state directory.

The plugin also exposes `nanlm_auth_status` and `nanlm_auth_revoke`. When an authority URL is configured, corpus generation and training fail closed unless the pinned issuer token has the required scope and the authority confirms it is still active. The `/v1/authorize` endpoint checks current revocation state; `/v1/revoke` invalidates the token immediately at the authority.
