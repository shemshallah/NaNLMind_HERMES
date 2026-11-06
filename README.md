# NaNLMind Hermes

A Hermes plugin for generating, checking, and training one local NaNLMind universe per installation.

## Operating model

Each installation owns one universe, one hardware endpoint, one bounded runner, and one local state directory. No shared mutable universe or always-on community coordinator is started by default. A second universe is only added through explicit endpoint registration and authorization.

The core topology is the existing NaNLMind **256 palindromic heads over one universe**. This is not a transformer fine-tune. Its replayable learning path uses integer score updates, temporal links, and deterministic state changes.

## Training duration

Every run has a configurable wall-clock budget from 1 through 360 minutes. The default is 30 minutes. The deadline is enforced by a monotonic clock, checked between bounded work units. A run checkpoints on graceful stop and never starts another work unit after its deadline. The duration is a ceiling, not a promise of continuous compute or a target epoch count.

Configuration lives in `~/.config/nanlmind-hermes/config.json` (or `NANLM_CONFIG`). Example:

```json
{
  "training": {
    "duration_minutes": 30,
    "checkpoint_interval_seconds": 60,
    "max_records_per_batch": 128
  },
  "universe": {
    "id": "local-default",
    "state_dir": "~/.local/share/nanlmind-hermes/universe",
    "runner_command": ["./build/nanlm-runner", "train"]
  }
}
```

Set `duration_minutes` to any integer from 1 to 360. A training tool call may override it for that run, within the same limit.

## Hermes integration

Install as a trusted general plugin using the Hermes plugin installer, then enable it. The plugin registers tools for status, corpus generation, critique, training, evaluation, recall, and snapshots. LLM generation uses Hermes' host-owned `ctx.llm` route; provider credentials are not exposed to the plugin.

## Corpus format

Training input uses the NaN16-owned N16TXT v1 format. The plugin validates the framing and bounds before handing the file to the NaN16 codec/runner. Provenance metadata is stored separately from corpus text; corpus records are never treated as commands.

## Safety and persistence

- A run is single-writer per local universe (exclusive lock).
- Deadlines use monotonic time and are enforced in the runner supervisor.
- Checkpoints are written to a temporary file and atomically renamed.
- A failed or malformed corpus never mutates the active universe.
- Snapshots record universe ID, base digest, corpus digest, codec version, runner version, and completion reason.
- GitHub tokens and model credentials are never written into the universe, prompts, logs, or snapshots.

## Current status

Initial plugin scaffold and bounded-run supervisor are under active development. The native runner protocol is deliberately isolated behind an adapter; no unsupported NaNLMind CLI command is assumed.
