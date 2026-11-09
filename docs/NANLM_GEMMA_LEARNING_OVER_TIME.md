# NaNLM learning over time with Gemma as teacher

- model/teacher: `gemma4:31b-cloud`
- student: NaNLM native mode (no LLM answerer)
- rounds: 4
- state: `/data/data/com.termux/files/usr/tmp/nanlm-train-canonical`
- baseline hit rate: 0.000

## Probe hit rate

| stage | hits | keyword % | structured hits | hit vector | flips | turns | scratch | concepts | pairs | leaves |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| baseline | 0/14 | 0.00% | 0/14 | `00000000000000` | 0 | 0 | 0 | 0 | 0 | 0 |
| round 1 | 8/14 | 57.14% | 3/14 | `11111110100000` | 8 | 12 | 120 | 28 | 38 | 255 |
| round 2 | 12/14 | 85.71% | 9/14 | `11111101011111` | 8 | 24 | 240 | 50 | 102 | 255 |
| round 3 | 14/14 | 100.00% | 11/14 | `11111111111111` | 2 | 36 | 360 | 56 | 116 | 255 |
| round 4 | 14/14 | 100.00% | 11/14 | `11111111111111` | 0 | 48 | 480 | 60 | 134 | 255 |

## Probe detail (final round)

- What is a river? -> `River consists of water.` | expected ['water'] | hit=True
- What color is the sky? -> `Sky is blue mix to make purple.` | expected ['blue'] | hit=True
- What color is grass? -> `Grass is green.` | expected ['green'] | hit=True
- What do dogs do? -> `Dogs can bark.` | expected ['bark'] | hit=True
- What do cats do? -> `Cats can meow.` | expected ['meow'] | hit=True
- What do birds do? -> `Birds can fly.` | expected ['fly'] | hit=True
- What is a cat? -> `A cat is an animal.` | expected ['animal'] | hit=True
- What happens when water freezes? -> `Frozen water is ice.` | expected ['ice', 'frozen'] | hit=True
- What is ice? -> `Ice is water that froze.` | expected ['frozen', 'water'] | hit=True
- What do red and blue make? -> `Red and blue mix to make purple.` | expected ['purple'] | hit=True
- What do yellow and blue make? -> `Yellow and blue mix to make green.` | expected ['green'] | hit=True
- What is the sun? -> `The sun causes light.` | expected ['hot', 'light', 'star'] | hit=True
- What shape has four sides? -> `A square has four sides.` | expected ['square'] | hit=True
- What is a circle? -> `A circle is a round shape.` | expected ['round'] | hit=True

## Boundaries

- Local native student; Gemma designed statements only.
- No canonical merge, NANNET route or proof claim.
- A flat or noisy curve is a real result; inspect per-probe replies.
