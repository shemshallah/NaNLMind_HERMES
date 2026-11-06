# Changelog

## 1.0.0 — 2026-11-06

- Default interaction is the conversational `nanlm_chat` path.
- Added `/nanlm` command with `chat`, `logic`, `words`, `sentences`, `pipeline`,
  `status`, and `help` modes.
- Added explicit training tools:
  - `nanlm_logic_session`
  - `nanlm_word_session`
  - `nanlm_sentence_session`
  - `nanlm_training_pipeline`
- Added custom ordered stage selection for training pipelines.
- Added dependency-free Ed25519, AES-256-GCM, HKDF-SHA256, and RSA PKCS#1
  fallback for Alpine/Termux/minimal hosts.
- Added Alpine `proot-distro` test instructions.
- Added complete tool, configuration, walkthrough, and architecture
  documentation.
- Declared all 23 registered tools in `plugin.yaml`.
- Fixed the root plugin entry to import the real sibling package under Hermes'
  isolated loader.
- Released under MIT.

## 0.2.0

- Native math trainer, additive discovery memory, NANNET identity and pairing,
  multiresolution chat/reflection.
