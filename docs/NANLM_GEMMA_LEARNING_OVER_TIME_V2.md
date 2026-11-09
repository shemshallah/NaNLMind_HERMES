# NaNLM learning over time with Gemma as teacher

- model/teacher: `gemma4:31b-cloud`
- student: NaNLM native mode (no LLM answerer)
- rounds: 6
- state: `/data/data/com.termux/files/usr/tmp/nanlm-train-v2`
- baseline hit rate: 0.000

## Probe hit rate

| stage | keyword hit rate | structured hit rate | hit vector | flips | delta | turns | scratch | concepts | pairs | leaves |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.000 | 0.000 | `00000000000000` | 0 | +0.000 | 0 | 0 | 0 | 0 | 0 |
| round 1 | 0.429 | 0.143 | `11111000100000` | 6 | +0.429 | 12 | 120 | 22 | 23 | 255 |
| round 2 | 0.857 | 0.571 | `11111110111110` | 6 | +0.857 | 24 | 240 | 47 | 90 | 255 |
| round 3 | 1.000 | 0.786 | `11111111111111` | 2 | +1.000 | 36 | 360 | 53 | 102 | 255 |
| round 4 | 1.000 | 0.786 | `11111111111111` | 0 | +1.000 | 48 | 480 | 54 | 105 | 255 |
| round 5 | 1.000 | 0.786 | `11111111111111` | 0 | +1.000 | 60 | 600 | 55 | 107 | 255 |
| round 6 | 1.000 | 0.786 | `11111111111111` | 0 | +1.000 | 72 | 720 | 58 | 123 | 255 |

## Probe detail (final round)

- What is a river? -> `River consists of water.` | expected ['water'] | hit=True
- What color is the sky? -> `Sky is blue.` | expected ['blue'] | hit=True
- What color is grass? -> `Grass is green.` | expected ['green'] | hit=True
- What do dogs do? -> `Dogs can bark.` | expected ['bark'] | hit=True
- What do cats do? -> `Cats can meow.` | expected ['meow'] | hit=True
- What do birds do? -> `Birds can fly.` | expected ['fly'] | hit=True
- What is a cat? -> `A cat is an animal.` | expected ['animal'] | hit=True
- What happens when water freezes? -> `Ice is water that is frozen.` | expected ['ice', 'frozen'] | hit=True
- What is ice? -> `Ice is water that is frozen.` | expected ['frozen', 'water'] | hit=True
- What do red and blue make? -> `Red mixed with blue makes purple.` | expected ['purple'] | hit=True
- What do yellow and blue make? -> `Yellow mixed with blue makes green.` | expected ['green'] | hit=True
- What is the sun? -> `The sun causes light.` | expected ['hot', 'light', 'star'] | hit=True
- What shape has four sides? -> `A square has four sides.` | expected ['square'] | hit=True
- What is a circle? -> `A circle is a round shape.` | expected ['round'] | hit=True

## Boundaries

- Local native student; Gemma designed statements only.
- No canonical merge, NANNET route or proof claim.
- A flat or noisy curve is a real result; inspect per-probe replies.
