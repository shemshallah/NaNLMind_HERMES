# Bounded training duration

The per-installation setting `duration_minutes` accepts integer values 1–360. Default: 30.

- A per-call `duration_minutes` overrides the default for that run only.
- The native runner is supervised against a monotonic deadline.
- At the deadline, the supervisor sends SIGINT and allows up to 30 seconds for the runner to finish its current bounded unit and atomically checkpoint.
- If graceful stop fails, TERM then KILL are used. The run result reports `deadline_forced_stop`; a missing checkpoint is never described as successful.
- An exclusive local lock prevents two training processes from mutating the same universe.
- Wall-clock budget excludes the bounded checkpoint shutdown grace; training work itself does not continue after the deadline signal.

The runner adapter contract is intentionally explicit and must be implemented by the native runner before training is considered end-to-end:

```text
<runner argv> train --corpus PATH --duration-seconds N
  --checkpoint PATH --checkpoint-interval-seconds N
<runner argv> evaluate --corpus PATH
```

The existing NaNLMind repository documents 256 palindromic heads, integer score learning, temporal links, and a measured plain-RNN control. This plugin does not claim transformer fine-tuning or a measured language-model advantage.
