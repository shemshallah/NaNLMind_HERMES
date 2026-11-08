# nanrunner-hermes — temporary Ollama/Gemma plugin verification

**Location:** `.github/workflows/nanrunner-hermes.yml` on the
`feat/hermes-per-lease-nan16de-universe` development branch.

**Isolation:** manually triggered GitHub-hosted Ubuntu job named
`nanrunner-hermes`. It receives **only** the
`nanrunner-hermes` GitHub Environment secret, if configured.
No runner repositories `nanlm-cpu-1`–`nanlm-cpu-6` are involved.
There is no persistent self-hosted runner and no cron, push or
pull_request trigger.

**Repository limitation:** `shemshallah/nanrunner-hermes` does not exist.
This workflow lives inside `shemshallah/NaNLMind_HERMES`, not a
separate repository. Moving it to a private standalone repo is possible
after that repo has been explicitly created and connected.

## One-time setup, with no API key in code or chat

1. **Rotate any previously disclosed Ollama key** from the Ollama API
   keys console. Keep the new one out of tickets, source and workflow
   variables that would be visible in logs.
2. In GitHub, navigate to
   **shemshallah/NaNLMind_HERMES → Settings → Environments**.
   Create/select environment **`nanrunner-hermes`**. Under
   **Environment secrets**, add **`OLLAMA_API_KEY`** with the new token.
   Prefer an environment with manual reviewers and deployment-branch
   restrictions if your repository and plan support them. Do **not**
   add a repository-wide secret, and do not set any secret in the six
   CPU worker repositories.
3. The manual `workflow_dispatch` must be available from the
   repository's **default branch** on GitHub. Merge the workflow
   only after tests and review; then open
   **Actions → nanrunner-hermes · isolated Ollama / Gemma plugin test
   → Run workflow**. Select the reviewed branch when offered.
4. Inspect the job summary. A green job means an actual bounded
   Ollama request returned a nonempty `gemma4:31b` completion,
   *and* Hermes Agent plugin validation passed. Failure with
   HTTP 401/403 indicates authentication or model access;
   HTTP 402/429 indicates available credit or throttling.
5. Check your actual **Ollama Usage** and account credits before and
   after the run. The completion API returns token counts, but it
   does **not** prove the usage came out of free credits.

## Scope and safety

- Hosted endpoint: `https://ollama.com/v1/chat/completions`.
- Hosted model ID: `gemma4:31b` (not the local CLI suffix
  `gemma4:31b-cloud`). This naming follows Ollama's API docs.
- Each manually triggered job sends at most one model request:
  short prompt, `max_tokens=24`, `stream=false`.
- A real Hermes Agent installation is performed on the runner using
  the pinned v0.23.0 installation path, followed by
  `hermes plugins validate` and `hermes plugins doctor`.
- No model weights are downloaded to the runner for the cloud test;
  Ollama itself need not be installed for the direct HTTPS API.
- The workflow's `permissions` grants read-only repository access.
  The model credential is **only** made available to the one
  authenticated cloud step through an environment secret.
- An environment is not a cryptographic sandbox from repository
  maintainers. Anyone who can alter a permitted workflow and cause
  an approved job to run can potentially access the environment
  secret. Configure branch rules/review approval and rotate the key
  after testing.
- Do not use untrusted pull request code, user prompts, or build
  scripts in the secret-holding job without a human review.

## Not yet proven

- A live remote NaNROUTER HERMES-universe allocation or Koyeb cutover.
- Actual Gemma **tool invocation inside an active Hermes conversation**;
  plugin doctor/admission and a genuine model completion are separate
  tests. The next test must configure Hermes' provider against the same
  cloud endpoint and show a real plugin tool invocation.
- A specific free-credit balance or a $0 billed event; only the
  authenticated account's usage records can establish that.
