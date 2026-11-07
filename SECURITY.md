# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.2.x | Yes |
| 1.1.x and earlier | No |

## Reporting a vulnerability

Use the repository's private GitHub Security Advisory flow:

<https://github.com/shemshallah/NaNLMind_HERMES/security/advisories/new>

Do not open a public issue containing credentials, pairing codes, private
keys, customer data, or an exploitable proof of concept. If the advisory form
is unavailable, contact the repository owner through GitHub with only a short
summary and a request for a private channel.

Please include:

- affected version or commit;
- Hermes version and platform;
- whether the issue is in plugin code, packaging, or an external NANNET/CLOCKED
  service;
- reproduction steps with secrets and personal data removed;
- impact and any suggested mitigation.

Security issues in Hermes Agent, NANNET, or CLOCKED that are not caused by this
plugin should be reported to their respective repositories.

## Security boundaries

This plugin does not collect telemetry, download code, or read credentials from
environment variables, `.env` files, browser profiles, vendor stores, or
keychains. NANNET, CLOCKED, and head-cortex secrets are read only through
Hermes-managed secret scope. Protected operations fail closed when pairing,
scope authorization, or the authority is unavailable.
