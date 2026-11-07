# Demo guide

Three reproducible scenarios. Start with the offline demo; add NANNET and
CLOCKED when those services are available. Every command below is also covered
by the test suite or the live smoke record in
[HERMES_CATALOG_ADMISSION.md](HERMES_CATALOG_ADMISSION.md).

## 1. Offline demo: chat and native training

Requirements: Hermes Agent, this plugin, and a built
`NaNLMind/build/nanlm-universal`.

```sh
hermes plugins install shemshallah/NaNLMind_HERMES --yes-deps --enable
hermes plugins doctor /path/to/NaNLMind_HERMES --ci
```

Configure `runner_command` in Hermes:

```yaml
plugins:
  entries:
    nanlmind-hermes:
      enabled: true
      settings:
        state_dir: ~/.local/share/nanlmind-hermes
        runner_command: ["/absolute/path/to/NaNLMind/build/nanlm-universal"]
        chat_mode: native
```

In a Hermes session:

```text
/nanlm A river is water.
/nanlm What is a river?
/nanlm logic Boolean inverses and nested implications
/nanlm status
```

Expected: the first chat stores `A river is water` as a supervised relation
and updates the native state; the second retrieves it as
`I recall these learned relations: A river is water.` The reply and the shared
scratch window are stored under `state_dir`. No language-model provider is
required in `native` mode. The training
reply reports the Boolean foundation result and native run manifest; `status`
shows the one `HW0::BS0` state directory and the last run. A failed Boolean
gate stops before language stages.

## 2. NANNET demo: identity and scoped access

The NANNET authority must expose `GET /v1/issuer` and issue one-time pairing
codes through `POST /v1/admin/pairing-codes` (operator credential). The issuer
public key is the stable key that signs paired installation tokens.

1. Read the authority issuer key:

```sh
curl -fsS https://<nannet-authority>/v1/issuer
```

2. Configure `nannet_auth_url` and pin `nannet_issuer_public_key`. Enter a
   random base64url 256-bit `NANNET_MASTER_KEY` in the masked Hermes secret
   field.

3. Ask the operator for a pairing code:

```sh
curl -fsS -X POST https://<nannet-authority>/v1/admin/pairing-codes \
  -H "Authorization: Bearer <operator-token>" \
  -H "Content-Type: application/json" \
  -d '{"scopes":["universe:read","training:execute"],"ttl_seconds":600}'
```

4. In Hermes:

```text
nanlm_auth_init
nanlm_auth_pair  # pairing_code: "nnh_..."
nanlm_auth_status
/nanlm status
```

Expected: `nanlm_auth_init` creates one encrypted installation identity;
`nanlm_auth_pair` verifies the returned issuer key against the pinned key,
stores the token encrypted, and reports the granted scopes; protected tools
request online scope authorization and fail closed when the authority is
unreachable. `nanlm_auth_revoke` revokes the token and erases the local copy.

## 3. CLOCKED demo: approved evidence plan

Configure `clocked_gateway_url` and the `CLOCKED_GATEWAY_TOKEN` secret. The
token must be a scoped participant credential (`clu_...`), not the upstream
CLOCKED API key. Configure `nanlm_head_cortex_url` and `NANLM_DEVICE_TOKEN`
for head-cortex calls.

Create a request file:

```json
{
  "question": "Compare these two technical sources.",
  "scratch_id": "demo-clocked",
  "sources": [
    {"id": "source-a", "text": "First source text..."},
    {"id": "source-b", "text": "Second source text..."}
  ]
}
```

Plan, review, then execute with the exact plan digest:

```text
/nanlm clocked plan demo.json
/nanlm clocked run demo.json --approved
/nanlm clocked scratch demo-clocked
/nanlm clocked head demo.json
```

Expected: planning does not transmit source text; execution requires the
reviewed `plan_sha256` and explicit `--approved`; typed results and evidence
atoms appear in the append-only scratch journal. CLOCKED results are evidence,
not ground truth, and candidate learning cannot promote or mutate canonical
head state.

## Latest recorded runs

- [HERMES_DEMO.md](HERMES_DEMO.md): human-readable transcript of real local
  chat, native training, live CLOCKED, and live head-cortex output.
- [DEMO_RESULTS.md](DEMO_RESULTS.md): 36/36 tool calls passed against a local
  full-feature demo (real native binary, loopback NANNET authority, local
  CLOCKED contract stubs).
- [LIVE_SMOKE.md](LIVE_SMOKE.md): 11/11 checks passed against the single
  deployed `nannet.apps.mistral.run` service, including head-cortex.

Regenerate the local report with:

```sh
python scripts/demo_run.py --native-binary /path/to/nanlm-universal
```

The live report is opt-in and requires an operator-issued pairing code:

```sh
python scripts/live_smoke.py --nannet-url https://<authority> \
  --pairing-code nnh_... --clocked-url https://<gateway> \
  --clocked-token-file /path/to/clocked-token
```

## What to check

| Area | Success signal |
|---|---|
| Manifest | `hermes plugins validate . --install-deps` passes |
| Registration | `hermes plugins doctor . --ci` reports 33 tools, 0 hooks |
| Chat | reply plus transcript under `state_dir/transcripts` |
| Training | Boolean gate 1,024/1,024 and a native run manifest |
| NANNET | paired token valid, online scope authorization passes, revoke succeeds |
| CLOCKED | plan digest matches, scratch lanes written, no canonical promotion |
| Package | wheel/sdist install and import, SHA-256 checksums verify |
