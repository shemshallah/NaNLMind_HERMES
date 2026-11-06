# NaNLMind Hermes

A Hermes plugin for generating, validating, and training one local NaNLMind universe per installation.

## Operating model

Each installation owns one universe, one hardware endpoint, one bounded runner, and one local state directory. No shared mutable universe or always-on community coordinator is started by default. Another universe is added only through explicit endpoint registration and authorization.

The native topology is NaNLMind's **256 palindromic heads over one universe**. This is not transformer fine-tuning. Its documented replayable learning path uses integer score updates, temporal links, and deterministic state changes.

## Configurable training duration

- Default: **30 minutes** per run.
- Minimum: **1 minute**.
- Maximum: **360 minutes (6 hours)**.
- Set the persistent default in the plugin's Hermes settings as `duration_minutes`.
- Override a single run with the `duration_minutes` argument to `nanlm_train`.
- The runner is supervised against a monotonic wall-clock deadline. At the deadline it receives SIGINT and up to 30 seconds to finish its current bounded unit and atomically checkpoint; TERM/KILL are fallbacks.
- The duration is a ceiling, not a promised epoch count or guarantee of uninterrupted compute.

The plugin also exposes `checkpoint_interval_seconds`, `runner_command`, and `state_dir` in its settings. A typical local state directory is `~/.local/share/nanlmind-hermes`.

## Hermes tools currently registered

- `nanlm_status` — local universe and runner configuration.
- `nanlm_generate_corpus` — uses Hermes' active model to generate bounded N16TXT/1 records.
- `nanlm_train` — validates N16TXT/1 then invokes the configured local runner with the requested duration.
- `nanlm_evaluate` — invokes the configured runner's evaluation mode.

Critique, recall, snapshot exchange, and explicit additional-endpoint registration are follow-on implementation work; they are not represented as complete yet.

## Corpus format

Training input uses the NaN16-owned N16TXT v1 text format. The plugin validates the framing and bounds before handing a corpus to the native runner. Provenance metadata is stored separately from corpus text; corpus records are never treated as commands.

The format enforces UTF-8, exact `N16TXT/1` header, 4096-byte maximum records, 1 MiB maximum corpus, control-character rejection, and deterministic exact-duplicate removal. The NaN16 codec remains authoritative for canonical record-stream encoding.

## Safety and persistence

- One writer per local universe (exclusive lock).
- Monotonic wall-clock deadline, checked by the runner supervisor.
- Checkpoint path is isolated to the installation's state directory.
- A malformed corpus is rejected before runner invocation.
- Corpus SHA-256 and stop reason are recorded in local run status.
- GitHub tokens and Hermes model credentials are never written into the universe, prompts, logs, or snapshots.

## Native runner adapter boundary

The existing NaNLMind core documents the learning substrate but does not yet expose this plugin's subprocess contract. Configure `runner_command` only after building an adapter that accepts:

```text
<runner argv> train --corpus PATH --duration-seconds N
  --checkpoint PATH --checkpoint-interval-seconds N
<runner argv> evaluate --corpus PATH
```

The adapter must handle SIGINT by finishing its current bounded work unit and atomically publishing the checkpoint. Until that adapter exists, training is intentionally disabled rather than pretending the existing CLI supports these flags.

## Development

```sh
python -m unittest discover -s tests -v
```

CI runs the same policy and supervisor tests on pushes and pull requests.
