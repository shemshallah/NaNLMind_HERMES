# Plugin architecture

## Default path

```text
Hermes conversation
  -> nanlm_chat
  -> provider answer
  -> durable transcript
  -> surface measurements
  -> best-effort structured reflection
  -> additive memory merge
  -> optional cross-resolution assessment
```

Chat is intentionally independent of native training. A provider answer is not
treated as a model-weight update, and reflection proposals are evidence, not
learned parameters.

## Explicit training path

```text
/nanlm logic|words|sentences|pipeline
  -> selected curriculum stages
  -> N16TXT/1 corpus
  -> native 256-head math-train
  -> serialized global topology
  -> one discovery per head
  -> validation
  -> memory merge and L0-L6 reorganization
  -> experiment manifest
```

Training is explicit so conversation does not silently start an expensive local
job.

## Stores

| store | owner | contents |
|---|---|---|
| conversation transcripts | plugin state | JSONL user/assistant rows |
| mathematical memory | SQLite WAL | immutable discoveries, edges, hyperedges, assessments |
| native topology | native runner | serialized head/cell/scratchpad state |
| experiment manifests | plugin state | corpus, command, metrics, paths |
| NANNET identity | plugin state | encrypted Ed25519 private key and tokens |

## Memory layers

```text
L0 observation
L1 entity
L2 relation
L3 operator
L4 equation
L5 hypothesis / constraint
L6 contested counterexample / review
```

Projections are rebuildable. The event log is the source of truth. Merge is an
idempotent union; conflicting claims remain visible and are promoted for review.

## Trust boundaries

- The local native process is trusted local code, not a sandbox.
- The model provider is an external network dependency; its answer is untrusted
  text and structured reflection is validated before ingestion.
- NANNET pairing is optional and fail-closed when configured.
- `DEEPREPO_PAT` is not used by this plugin.
- The pure-Python crypto fallback is not constant-time; use the optional audited
  backend or an HSM for production key custody.

## Failure behaviour

| failure | behaviour |
|---|---|
| provider unavailable | chat raises a clear tool error |
| reflection unavailable | chat answer still succeeds |
| native runner missing | training fails before modifying state |
| discovery trace incomplete | native result rejected |
| memory contradiction | retained and assessed, not erased |
| authority unavailable | paired operations fail closed |
| optional UI build failure | CLI/plugin operation remains available |

## Versioning

`1.0.0` is the first documented plugin release. The manifest, `pyproject.toml`,
tool reference, README, and changelog share the same version and release date.
