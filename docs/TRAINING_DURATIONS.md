# Hermes local training harness

## Execution boundary

Hermes runs the native NaNLMind executable on the machine hosting the Hermes
plugin. Training does **not** run on Koyeb. Koyeb is the NANNET authority and
small-delta coordination/control plane. Corpus bytes, native process memory,
local universe state and run artifacts remain on the local runner unless a
separate upload/merge action is explicitly implemented.

## Native integration (implemented)

Set Hermes config `runner_command` to an argv array naming the local native
binary, for example:

```json
["/absolute/path/to/NaNLMind/build/nanlm-universal"]
```

The adapter calls the real CLI contract present in NaNLMind:

- `bootstrap CORPUS EPOCHS CHUNKS` — trains the native in-memory Context2
  model and reports held-out BPC/top-1 metrics after each chunk.
- `byte CORPUS` — topology bigram plus independent reference and order-0
  baselines.
- `kn CORPUS 2` — independent Kneser–Ney order-2 baseline.
- `ctx2 CORPUS` — context-2 child-chart layout evaluation.

Training input is strict N16TXT/1 UTF-8. The harness stores a content-addressed
corpus, a run manifest, the complete native training log, exit status, corpus
SHA-256, command, epoch/chunk configuration and metrics. It enforces one writer
per local universe and a monotonic wall-clock budget. Evaluation runs the
native controls against the same corpus and records their outputs.

## Important current native limitation

The native `bootstrap` command builds Context2Model in process memory and
does not currently serialize its learned weights. Therefore Hermes reports
`checkpoint_present: false` and `resume_supported: false`; the manifest/log
is a reproducibility artifact, **not a model checkpoint**. A stopped or finished
run must be replayed from its corpus to reconstruct that in-memory model.
Implementing native Context2Model serialization and resumable snapshots is a
separate core-repository change; Hermes does not fabricate a checkpoint.

## NANNET

When `NANNET_AUTH_URL` and the pinned issuer key are configured, the Hermes
tool entry points require online authority validation for their requested
scopes. The local native process remains local. The portal currently provides
identity/pairing/device lease and head/route-delta merge APIs; it is not a
general remote training-job executor or model-artifact store.

## First local run

1. Build NaNLMind with its documented Makefile and locate
   `build/nanlm-universal`.
2. Configure Hermes `runner_command` with the absolute binary path.
3. Create/pair the local NANNET installation if the authority is configured.
4. Generate or provide a strict N16TXT/1 corpus.
5. Call `nanlm_train` with the corpus path and duration.
6. Call `nanlm_evaluate` to compare topology byte, Kneser–Ney and Context2
   metrics on the same corpus.
7. Inspect `nanlm_status` for the last run and local artifact paths.

Never describe the current Context2 bootstrap as transformer fine-tuning or
claim that its in-memory learned parameters survive process exit.
