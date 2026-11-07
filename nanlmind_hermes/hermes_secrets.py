"""Hermes-managed secret access; never fall back to process environment."""
from __future__ import annotations


def get_secret(name: str) -> str:
    try:
        from agent.secret_scope import get_secret as hermes_get_secret
    except ImportError as exc:
        raise RuntimeError(
            "This operation requires Hermes-managed secrets; run it inside Hermes "
            "and configure the plugin's secret fields."
        ) from exc
    value = hermes_get_secret(name)
    return value if isinstance(value, str) else ""
