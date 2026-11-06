# Hermes plugin install and validation

The repository is a native `plugin.yaml` + `register(ctx)` plugin.

## Local/development install

```sh
hermes plugins install /path/to/NaNLMind_HERMES --enable
export HERMES_GATEWAY_NO_SUPERVISE=1
hermes plugins validate /path/to/NaNLMind_HERMES --install-deps
hermes plugins doctor /path/to/NaNLMind_HERMES --ci
hermes plugins list
hermes tools
```

The plugin must register the tools declared in `plugin.yaml`. The current
manifest declares 24 tools, including:

```text
nanlm_chat, nanlm_chat_history, nanlm_curriculum_experiment,
nanlm_math_train, nanlm_memory_* and the NANNET auth tools
```

## Repository/catalog install

For a catalog/reviewed install, submit the public repository through the Hermes
plugin catalog process. That is a separate submission/review step; it is not
performed by installing from a local checkout.

The manifest is catalog-oriented:

- `manifest_version: 2`
- `api_version: 1`
- `provides_tools` matches `register(ctx)`
- core dependencies are empty
- optional `[auth]` extra enables the audited `cryptography` backend
- the dependency-free Ed25519/AES-GCM/HKDF fallback keeps the plugin usable on
  minimal Alpine/Termux images

## Alpine test

See `docs/ALPINE_PROOT_TESTING.md` for the no-Termux-package test path.
