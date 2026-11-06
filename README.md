# NaNLMind Hermes

Hermes plugin for a local NaNLMind 256-head universe, with NANNET installation identity and authority-side scope checks.

## Training is local

The native NaNLMind binary executes on the machine hosting Hermes. Koyeb is the NANNET control plane (pairing, authorization, device leases and small delta merge); it does not receive the corpus or run the native training process.

Build the core repository and configure Hermes runner_command as an argv array containing the absolute path to build/nanlm-universal:

    ["/absolute/path/to/NaNLMind/build/nanlm-universal"]

Available tools:
- nanlm_status — local universe, runner configuration and last run.
- nanlm_generate_corpus — produce a content-addressed N16TXT/1 corpus.
- nanlm_train — invoke the native bootstrap CORPUS EPOCHS CHUNKS trainer with a monotonic deadline and exclusive local training lock.
- nanlm_evaluate — run topology-byte, Kneser-Ney order-2 and Context2 evaluations against the same plain-text corpus.
- nanlm_auth_init, nanlm_auth_pair, nanlm_auth_status, nanlm_auth_revoke — manage the local NANNET installation identity.

N16TXT/1 is validated and its framing header is removed before native CLI use. Native bootstrap requires at least 1,000 UTF-8 bytes; byte and baseline evaluations require at least 100 bytes. The harness never pads or duplicates a small corpus.

## Current native persistence boundary

The native bootstrap command currently trains Context2Model in process memory and reports held-out BPC/top-1 metrics, but it does not serialize learned parameters. Hermes saves the corpus digest, command, configuration, stdout log and run manifest. It explicitly reports checkpoint_present=false and resume_supported=false. These artifacts are not model weights and cannot resume a partially completed run. Native Context2Model serialization remains a core-repository milestone.

## NANNET identity

Provision an independent random 256-bit base64url NANNET_MASTER_KEY in the host secret store. Provision NANNET_AUTH_URL and the pinned NANNET_ISSUER_PUBLIC_KEY for the trusted authority. The installation private key is encrypted at rest with HKDF-SHA256 + AES-256-GCM. Pair using a one-time guest pairing code from the authority operator.

When authority mode is configured, corpus generation, training and evaluation fail closed unless the stored token has the required scope and the authority confirms it online. Paired installation requests to /v1/authorize and /v1/revoke are Ed25519-signed over method, exact path, timestamp, nonce and body digest; the authority consumes each nonce once. The authority master key never leaves the authority host.

See docs/NANNET_API_HERMES.md and docs/TRAINING_DURATIONS.md.
