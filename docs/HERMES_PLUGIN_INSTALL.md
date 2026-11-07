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
manifest declares 25 tools, including:

```text
nanlm_chat, nanlm_structured_relation_train, nanlm_chat_history,
nanlm_curriculum_experiment, nanlm_math_train, nanlm_memory_* and NANNET auth tools
```

Free-form chat learning is enabled by default through `chat_learning: true`.
It requires the `training:execute` scope when enabled. Set it to false for
chat-only sessions that retain symbolic reflection without native head,
mirror-pair or hyperedge updates.

## Repository/catalog install

For a catalog/reviewed install, submit the public repository through the Hermes
plugin catalog process. That is a separate submission/review step; it is not
performed by installing from a local checkout.

The manifest is catalog-oriented:

- `manifest_version: 2`
- `api_version: 1`
- `provides_tools` matches `register(ctx)`
- the manifest declares bounded `cryptography>=42,<47` for production NANNET identity protection
- NANNET, CLOCKED and head-cortex secrets are declared as masked `config_schema` secrets
- the plugin reads those secrets only through `agent.secret_scope.get_secret()`
- a pure-Python crypto compatibility path remains for constrained development/test hosts

## Alpine test

See `docs/ALPINE_PROOT_TESTING.md` for the no-Termux-package test path.
