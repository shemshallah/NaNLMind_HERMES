# Walkthroughs

## 1. Install and validate

```sh
hermes plugins install shemshallah/NaNLMind_HERMES --yes-deps --enable
hermes plugins validate /path/to/NaNLMind_HERMES --install-deps
hermes plugins doctor /path/to/NaNLMind_HERMES --ci
```

Expected:

```text
✓ capability probe
✓ declared tools
✓ security scan
Validation passed.
```

## 2. First conversation and online learning

```text
/nanlm Penguins can swim. They are birds.
```

Every turn writes a transcript and proposes structured evidence. With
`chat_learning: true` (default), explicit supported/refuted subject-predicate-
object relations are sent to the native relation learner. A chat result reports
whether heads, mirror pairs and hyperedges actually changed. Neutral or
incomplete reflections remain symbolic evidence only. Set `chat_learning:
false` to disable native updates while retaining normal chat.

## 3. Configure the native trainer

Build NaNLMind and set:

```yaml
plugins:
  entries:
    nanlmind-hermes:
      settings:
        runner_command: ["/opt/NaNLMind/build/nanlm-universal"]
```

Verify:

```text
/nanlm status
```

The status should show `native_runner_configured: true` and the native binary
path.

## 4. Logic-only training

```text
/nanlm logic Boolean inverses and nested implications
```

This uses:

```text
boolean_inverses
deep_logic
exceptions
```

The exact 1,024-check Boolean foundation gate runs first. The structured teacher
then generates L0 bit, L1 nibble and L2 byte relation records with frozen
held-out challenges. The native relation trainer updates routed heads,
mirror-pair cells and hyperedges.

Inspect the result:

```text
/nanlm status
```

## 5. Word-only training

```text
/nanlm words common words, repetition, and next-word choices
```

This uses:

```text
repetitive_words
phonetic_pairs
morpheme_structure
lexical_words
word_completion
```

The model receives separate L3 phonetic, L4 morpheme, L5 lexical and L6 word
completion tasks, each with its own frozen held-out measurements.

## 6. Sentence/story training

```text
/nanlm sentences short dialogue and narrative continuation
```

This uses:

```text
sentence_choice
paragraph_structure
narrative
```

L7 sentence, L8 paragraph and L9 narrative measurements are kept independent.

## 7. Full pipeline

```text
/nanlm pipeline English logic, words, sentences, and short stories
```

or explicitly set tool parameters:

```json
{
  "topic": "logic and language",
  "mode": "pipeline",
  "records_per_stage": 12,
  "epochs": 3,
  "max_samples_per_head": 4096
}
```

## 8. Custom training order

```json
{
  "topic": "compare logic-first and words-first",
  "stages": [
    "boolean_inverses",
    "deep_logic",
    "word_completion",
    "sentence_choice"
  ],
  "epochs": 2
}
```

The order is preserved. This lets an experiment compare curricula without
changing code.

## 9. Memory inspection

```text
/nanlm status
```

Then use the memory tools:

```json
{"subject": "surface:lexical-profile"}
```

or query assessments to see support/refutation and contested subjects.

## 10. Paired NANNET mode

1. Create the local identity:

```text
nanlm_auth_init
```

2. Provision a one-time pairing code on the authority.
3. Pair:

```text
nanlm_auth_pair {"pairing_code": "nnh_..."}
```

4. Check status:

```text
nanlm_auth_status
```

5. Revoke when finished:

```text
nanlm_auth_revoke
```

Standalone chat and local training do not require this path.

## 11. Alpine / minimal host

```sh
proot-distro install alpine:3.20
proot-distro login alpine -- /sbin/apk add --no-cache python3 sqlite-libs
```

See `docs/ALPINE_PROOT_TESTING.md` for the `--minimal` proot test command.

## 12. Troubleshooting matrix

| observation | interpretation | action |
|---|---|---|
| `nanlm_chat` works, training fails | runner not configured | set `runner_command` |
| validation import error | root entry cannot see package | reinstall current commit |
| memory events grow but chat answer is generic | reflection is not model weights | inspect chat/reflection separately |
| native state exists but no held-out metric | trace/manifest missing | inspect native run output |
| `fchdir`/`getcwd` in Alpine | proot binding profile | use `--minimal` |
| auth unavailable | optional cryptography absent | use fallback or install `[auth]` |
