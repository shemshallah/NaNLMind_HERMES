#!/usr/bin/env python3
"""Matched byte-level Transformer control for the Round 042 reference model.

Same corpus, split, token budget, context, optimizer family and evaluation
protocol as ``train.py``.  The control has no cycle field, no trigonometric
rotor bank, no 256-head interface and no NANNET-P.  It is not claimed to be
parameter-identical; the exact counts are reported.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train import PAD, VOCAB, make_batch
from wordnet_hw4 import build_documents, describe, load_records

BASE = Path(__file__).resolve().parent


class ByteTransformer(nn.Module):
    def __init__(self, vocab: int = VOCAB, d_model: int = 96,
                 n_heads: int = 4, n_layers: int = 10,
                 max_context: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_context, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers,
                                            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None):
        b, t = ids.shape
        positions = torch.arange(t, device=ids.device)[None, :]
        x = self.embed(ids) + self.pos(positions)
        mask = torch.triu(
            torch.ones(t, t, dtype=torch.bool, device=ids.device), diagonal=1
        )
        x = self.blocks(x, mask=mask)
        logits = self.head(self.norm(x))
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, self.head.out_features),
                labels.reshape(-1), ignore_index=-100,
            )
        return out


@torch.no_grad()
def evaluate(model: ByteTransformer, docs: list[bytes], limit: int = 36,
             context: int = 160) -> dict:
    model.eval()
    total = 0.0
    tokens = 0
    correct = 0
    for start in range(0, min(limit, len(docs)), 8):
        from train import BOS, EOS
        chunk = [[BOS] + list(doc[:context]) + [EOS]
                 for doc in docs[start:start + 8]]
        width = max(len(row) for row in chunk)
        ids = torch.full((len(chunk), width - 1), PAD, dtype=torch.long)
        labels = torch.full((len(chunk), width - 1), -100, dtype=torch.long)
        for row, seq in enumerate(chunk):
            ids[row, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            labels[row, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        logits = model(ids)["logits"]
        total += float(F.cross_entropy(
            logits.reshape(-1, VOCAB), labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        ))
        tokens += int((labels != -100).sum())
        correct += int((logits.argmax(-1) == labels).logical_and(labels != -100).sum())
    nll = total / max(tokens, 1)
    return {"tokens": tokens, "nats_per_token": nll,
            "bits_per_token": nll / math.log(2),
            "perplexity": math.exp(min(nll, 50.0)),
            "top1": correct / max(tokens, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE / "data" / "wordnet_hw4.jsonl.gz"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default=str(BASE / "results"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = load_records(args.data)
    docs = build_documents(records)
    model = ByteTransformer(d_model=args.d_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3,
                                  weight_decay=0.01)
    rng = random.Random(args.seed)
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = []
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        ids, labels = make_batch(docs["train"], rng, args.batch, args.context)
        optimizer.zero_grad()
        out = model(ids, labels=labels)
        out["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            val = evaluate(model, docs["val"], limit=24)
            history.append({"step": step, "loss": float(out["loss"].item()),
                            **val})
            print("BASE", step, round(float(out["loss"].item()), 3),
                  "val_bpb", round(val["bits_per_token"], 3), flush=True)
            if val["bits_per_token"] < best:
                best = val["bits_per_token"]
                best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    test = evaluate(model, docs["test"], limit=48)
    result = {
        "schema": "nanlm.trigon9-control/1",
        "control": "byte Transformer, no cycle/trig/heads/NANNET-P",
        "config": vars(args),
        "corpus": describe(args.data),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_val_bpb": best,
        "test": test,
        "history": history,
        "seconds": time.time() - start,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_transformer.json").write_text(
        json.dumps(result, indent=2)
    )
    print("TEST", json.dumps(test), flush=True)
    print("DONE", round(result["seconds"], 1), "s", flush=True)


if __name__ == "__main__":
    main()
