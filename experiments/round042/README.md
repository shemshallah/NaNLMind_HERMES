# Round 042 — TRIGON-9 experimental synthesis

This is the host reference harness for the Round 030–041 synthesis. The native
trigonometric contract is the raw NaN16 image
`nanlmind_hermes/nan/trig_cycle.nan`; the trainable model here is PyTorch and is
not the native runtime.

## Measured run

Seed 17, d_model 80, 10 layers, 4 attention heads, 120 LM steps, 60 relation
steps, 6 NANNET-P generations.

| Metric | Value |
|---|---:|
| Parameters | 1,084,110 |
| Held-out test | 3.6009 bits/byte |
| Held-out perplexity | 12.134 |
| Next-byte top-1 | 26.97% |
| Relation probe | 1,041 / 2,292 (45.42%) |
| Gauge argmax agreement | 100% |
| Max cycle phase error | 1.19e-6 |
| NANNET-P accepted | 1 (head_k 6 → 4) |
| Matched byte Transformer | 3.7377 bpb / 13.34 ppl / 24.05% top-1 (1.22M params, 55 s) |

Ablations and the honest limits are in `docs/NANLM_42_TRIGON.md`. Free samples
are malformed; the model is not fluent.

## Reproduce

```bash
python3 prepare_data.py --wordnet /path/to/wn3.1.dict/dict/data.noun
python3 train.py --steps 120 --relation-steps 60 --arch-generations 6 \
  --d-model 80 --batch 8 --context 48 --seed 17 --out results
python3 order_probe.py
python3 baseline.py --steps 120 --d-model 96 --seed 17 --out results
```

## Files

| file | role |
|---|---|
| `trigon.py` | cycle field, L0–L9 rotor blocks, heads, decoder, relation probe |
| `wordnet_hw4.py` | HW4 local mirror reader, typed relations, prior writer |
| `nannet_p.py` | bounded architecture policy and generation-CAS gate |
| `train.py` | LM/relation training, NANNET-P, ablations, sampling |
| `order_probe.py` | ordered SO(3) vs commutative phase control |
| `prepare_data.py` | compact WordNet 3.1 matrix builder |
| `data/wordnet_hw4.jsonl.gz` | 82,192-synset local mirror |
| `results/` | measured JSON, samples and order probe |

No live NANNET lease, canonical merge or BS0/MC0 promotion is authorized.
