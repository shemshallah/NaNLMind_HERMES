# Round 031 — Resident cycle memory in autoregressive language training

## Main result
This is a *real, measured, native-root-coupled* byte-level recurrent language-model experiment based on the earlier Round 030 C++ root. Two runs on WordNet 3.1, 100 steps each: native root performs worse than a conventional parameterized 16x16 matrix. The 8 MiB root holds **all** effective memory weights, read through anchored triangle circulations and updated through three-edge sparse gradients. A byte GRU is still external; this is not a fully memory-native LM.

All model runs use the WordNet corpus JSONL from Round 025. Definitions are selected from the prior synset-disjoint train/val/test splits. 1,800 noun glosses in each split are selected in original corpus file order; this is *not* a randomized representative WordNet sample and should not be treated as unbiased natural-language generalization. Input windows are sampled within training glosses with short-text wraparound, and evaluation resets recurrent state between documents. The Gutenberg Alice split is reused from Round 022.

## Reproduce
```
c++ -std=c++20 -O3 -fPIC -shared native_cycle.cpp -o libnative_cycle.so
OPENBLAS_NUM_THREADS=1 python train.py --steps 100 --seeds 17 29
OPENBLAS_NUM_THREADS=1 python audit.py
OPENBLAS_NUM_THREADS=1 python transfer.py
OPENBLAS_NUM_THREADS=1 python final_audit.py
```
Requires Python, NumPy, PyTorch, C++20, original Round 025 `corpus.jsonl` and Round 022 Alice files. Paths are currently absolute to these supplied files; edit `load_data` in `train.py` and the Alice paths in `transfer.py` if extracted elsewhere. The archive intentionally does not duplicate prior corpus or checkpoints.

## Tested metrics
See `results.json`, `audit.json`, and `transfer_results.json` for exact records.

WordNet held-out bits per byte (lower is better):

| Seed | ordinary | native cycle |
|---|---:|---:|
| 17 | 3.89467 | 4.00756 |
| 29 | 3.78293 | 3.84761 |

Counts: 13,200 held-out WordNet bytes per evaluated model; 100 training updates; 16-dimensional cycle memory x16 readout. Ordinary model has a conventional 16×16 parameter tensor; native model replaces that tensor with an 8 MiB root and exposes the same 256 effective weights. Neural encoder/decoder and GRU remain trainable in PyTorch; the native root is updated by C++ `nan_apply` after PyTorch computes gradients for its temporary readout. 3-edge distribution of an update is minimum Euclidean norm under the triangle sum constraint.

Gauge shift test: maximum effective-weight deviation below 8e-7, unchanged predicted classes, mirror integrity passes. Switching off cycle memory worsens loss but does not consistently lower accuracy.

Native full 16×16 read, via Python ctypes, ~6-7 microseconds on this Linux host; this is not a batched C++ benchmark and does not measure Android energy. Snapshot reload exact after gradient updates and gauge shift.

WordNet source graph: 117,570 parsed records; 87,655 eligible two-hop directed chains; 1,500 selected chains verified exactly, with no direct parent edge. This is symbolic graph lookup, *not learned composition*, and it is separate from the trained LM.

After a short 30-update Alice transfer, validation Alice BpB improved for all models. Validation WordNet BpB also improved slightly. Untouched Alice test has only 288 bytes, not sufficient for strong generalization claims. Native root post-transfer uses a separate snapshot.

## Architecture limitations
This round demonstrates real direct native-memory training and mathematical gauge invariance, **not** a benefit over the capacity-matched conventional parameter matrix. The root is RAM allocated but not mlocked in this run (`nan_new(0)`). Only 256 effective triangle weights are active; the remaining 32,129 cycle degrees and most pair lanes are unused. The native update currently applies SGD with fixed step size while the neural parameters use AdamW, so matched nominal parameter counts do not imply optimizer-equivalent training. A stricter future experiment should align optimizers and computational budgets, test multiple full books with book-disjoint splits, and incorporate typed hyperedge retrieval into the actual decoder.

No physical quantum effects are claimed. The root stores IEEE float values, not payload-preserving NaN instructions.
