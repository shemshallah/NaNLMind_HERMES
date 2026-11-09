"""Permanent plugin-duration NANNET/WordNet documents.

These files are package data. They are loaded once at plugin registration,
hashed, persisted as a session manifest, and injected into provider calls so
the attached LLM receives the same architecture contract on every turn.
"""
