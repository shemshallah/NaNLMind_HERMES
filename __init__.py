"""NaNLMind Hermes plugin entry point.

The implementation is a sibling package. Use a relative import so Hermes'
plugin loader keeps the plugin isolated under its generated package name.
"""
from .nanlmind_hermes import register

__all__ = ["register"]
