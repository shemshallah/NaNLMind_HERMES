# Phone-call kernel in raw NaN16 (0xNaN)

The three-way call (`nanlm_phone_call`) is a state machine written primarily
in raw 0xNaN. The routing kernel is a hand-assembled **NaN16 v0.1** word
image; Python is reduced to a marshalling boundary for text and external
services. No C is vendored in this repository.

## Files

| file | role |
|---|---|
| `nanlmind_hermes/nan/phone_call.nan` | the call kernel: 99 raw NaN16 words |
| `nanlmind_hermes/nanvm.py` | NaNVM0 boundary: loads `.nan` images, runs the NaN16 v0.1 core, headless device stubs |
| `nanlmind_hermes/phone_call.py` | text marshalling: runs the kernel, executes the emitted event plan against Hermes/NaNLM adapters, mirrors turns to scratch |

The kernel owns validation, addressee routing, round loops, event-plan
emission, turn accounting and the open/closed state transition. Python only
marshals text between the machine and the participants and calls external
services (LLM bridge, native chat, CLOCKED gateway); external I/O cannot live
inside NaN16 words.

## Word format and predication

Every line is `NNNN_PPPPPPPP_FFFF` — near head, 32-bit payload, far head as
stored in the low word (byte-swapped). The class nibble
`n = (c_near << 2) | c_far` is the predication condition *and*, for `BOOL`,
the truth table: every op except `BOOL` runs only when bit `2Z+L` of `n` is
set. The kernel uses four head pairs:

| heads (near/far) | n | predicate |
|---|---|---|
| `FFF8` / `F8FF` | 15 | ALWAYS |
| `7FF1` / `F8FF` | 3 | NE (`Z=0`) |
| `FFF1` / `F17F` | 4 | EQ (`Z=1`) |
| `7FF1` / `F1FF` | 1 | GT (unsigned `A>B`) |

Only NaNVM0 ops are used (`0x00`–`0x1B`): `BOOL`, `MOVI`, `ADD`, `SUB`, `CMP`,
`LD`, `ST`, `JMP`, `CALL`, `RET`, `NOP`, `HALT`. No NaNVM1 extension op is
emitted. `r13` must be `1` by convention and the machine zeroes all registers,
so the program sets it itself at word 000.

## Memory map

| word | meaning |
|---|---|
| `0000`–`0062` | program image (99 words) |
| `0400`+0 | ACTION: 0=start 1=say 2=run 3=status 4=end |
| `0400`+1 | ADDRESSEE: 0=hermes 1=nanlm 2=both |
| `0400`+2 | ROUNDS: 1–64 |
| `0400`+3 | STATUS: 0=open 1=closed (read/write) |
| `0400`+4 | TURN: running turn counter (read/add) |
| `0400`+5 | HAS_MSG / context available |
| `0400`+6 | EVENT_COUNT (out) |
| `0400`+7 | ERR: 0 ok, 1 closed, 2 bad rounds, 3 bad addressee, 4 message required |
| `0440` | EVENT_BASE: 4 words per event `[speaker][phase][addressee][0]` |

Speakers: 0=user, 1=hermes, 2=nanlm. Phases: 0=user_message,
1=hermes_to_nanlm, 2=nanlm_reply, 3=hermes_followup, 4=hermes_to_user. The
maximum plan is 1 user + 1 forward + 2×63 + 1 relay = 128 events = 512 words.

## Event-plan matrix (as implemented by the kernel)

| action | addressee | plan |
|---|---|---|
| start/say | hermes or both | user, forward, reply, (followup, reply)×(rounds-1), relay |
| start/say | nanlm | user, reply, (followup, reply)×(rounds-1), relay |
| run | hermes | forward, reply, (followup, reply)×(rounds-1), relay |
| status | any | no events; works on a closed call |
| end | any | no events; sets STATUS=1 |

Errors: closed call (say/run/start) → 1; rounds outside 1–64 → 2; addressee
outside 0–2 → 3; no message/context → 4.

## CLOCKED inside the call

`start`/`say` accept `clocked: true` (persisted for the call). Hermes turns in
a clocked call first run the call-window transcript through the CLOCKED
gateway (`extract → grounding → decide` over one aggregated conversation
source), then phrase the turn with that evidence. The `hermes_plan`,
`clocked_request`, `clocked_result` and `evidence_atoms` lanes land in the
same hash-chained scratch journal as `phone_user`, `phone_hermes`,
`phone_nanlm` and `phone_event` — the call window *is* the scratch window
(`scratch_id = call-<session>`). Any CLOCKED failure falls back to the normal
LLM/native relay without dropping the call.

## Verification

- `python -m unittest discover -s tests`: **183 tests pass**; 18 of them are
  NaN16-specific (`tests/test_nanvm.py`, `tests/test_phone_call.py`).
- Cross-machine parity: the identical kernel image and input matrix were run
  on the native nan0s C core (`nan_cli --peek 1024:8 --peek 1088:32`) and on
  `nanvm.py`. Control words (action, status, turn, event count, error) and
  every event word are identical across the routing matrix:
  `start/hermes/r1`, `start/nanlm/r2`, `start/both/r1`, `bad rounds`,
  `run on closed`, `end`, `run open`, `run with empty history`.
- Live transcript demo (v1.3.0): 12 call turns; the call window recorded
  `phone_event: 3`, `hermes_plan: 4`, `clocked_result: 12`,
  `evidence_atoms: 12` (`docs/HERMES_DEMO.md`, `docs/hermes-demo.json`).

Honest limits: `nanvm.py` implements the documented NaNVM0 op range with
headless device stubs; it is a boundary shim, not the native binary. VM1 ops
(0x20+) are out of scope and unused. Text generation is not in the kernel:
the kernel decides *who speaks next and how many turns remain*, while the
adapters produce the words.
