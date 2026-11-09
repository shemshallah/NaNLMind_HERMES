#!/usr/bin/env python3
"""Train and falsify the Round 042 TRIGON-9 reference model.

Stages:
  A. typed relation probe over the HW4 WordNet matrix (writes/reads cycles);
  B. byte-level language modelling with the same cycle field;
  C. NANNET-P architecture evolution under a validation + generation-CAS gate;
  D. gauge, ablation and free-sampling evaluation.

The script is intentionally honest about its boundary: the reference model is
PyTorch on the host.  The native contract is the raw NaN16 kernel
``nanlmind_hermes/nan/trig_cycle.nan``; this trainer is not the native runtime.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nannet_p import NannetP
from trigon import (
    CYCLE_COUNT, Genome, RelationTask, TrigonLM, head_for_name,
)
from wordnet_hw4 import (
    HW4_ROUTE, RELATION_LABELS, apply_wordnet_prior, build_documents,
    build_relations, describe, load_records, matrix_digest,
)

BASE = Path(__file__).resolve().parent
PAD, BOS, EOS = 258, 256, 257
VOCAB = 259


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)


def make_batch(docs: list[bytes], rng: random.Random, batch: int, context: int
               ) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for _ in range(batch):
        arr = [BOS] + list(docs[rng.randrange(len(docs))]) + [EOS]
        if len(arr) < context + 1:
            arr = arr + [PAD] * (context + 1 - len(arr))
        start = rng.randrange(max(1, len(arr) - context - 1))
        window = list(arr[start:start + context + 1])
        if len(window) < context + 1:
            window.extend([PAD] * (context + 1 - len(window)))
        x = window[:-1]
        y = [value if value != PAD else -100 for value in window[1:]]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def eval_docs(model: TrigonLM, docs: list[bytes], limit: int = 36,
              context: int = 160) -> dict:
    model.eval()
    total = 0.0
    tokens = 0
    correct = 0
    for start in range(0, min(limit, len(docs)), 8):
        chunk = [[BOS] + list(doc[:context]) + [EOS]
                 for doc in docs[start:start + 8]]
        width = max(len(row) for row in chunk)
        ids = torch.full((len(chunk), width - 1), PAD, dtype=torch.long)
        labels = torch.full((len(chunk), width - 1), -100, dtype=torch.long)
        for row, seq in enumerate(chunk):
            ids[row, :len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long)
            labels[row, :len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long)
        out = model(ids, labels=labels)
        logits = out["logits"]
        total += float(F.cross_entropy(
            logits.reshape(-1, VOCAB), labels.reshape(-1), ignore_index=-100,
            reduction="sum",
        ))
        tokens += int((labels != -100).sum())
        correct += int((logits.argmax(-1) == labels).logical_and(labels != -100).sum())
    nll = total / max(tokens, 1)
    return {
        "tokens": tokens,
        "nats_per_token": nll,
        "bits_per_token": nll / math.log(2),
        "perplexity": math.exp(min(nll, 50.0)),
        "top1": correct / max(tokens, 1),
    }


def name_tensor(name: str, width: int = 16) -> list[int]:
    raw = [BOS] + list(name.encode("utf-8", errors="replace")[:width - 2]) + [EOS]
    raw = raw + [PAD] * (width - len(raw))
    return raw


def relation_batch(relations: list, batch: int, rng: random.Random
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    by_label: dict[int, list] = {label: [] for label in RELATION_LABELS.values()}
    for relation in relations:
        by_label[relation.label].append(relation)
    labels_present = [label for label, rows in by_label.items() if rows]
    left, right, labels, lh, rh = [], [], [], [], []
    for index in range(batch):
        label = labels_present[index % len(labels_present)]
        relation = by_label[label][rng.randrange(len(by_label[label]))]
        left.append(name_tensor(relation.left))
        right.append(name_tensor(relation.right))
        labels.append(relation.label)
        lh.append(head_for_name(relation.left))
        rh.append(head_for_name(relation.right))
    return (
        torch.tensor(left, dtype=torch.long),
        torch.tensor(right, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(lh, dtype=torch.long),
        torch.tensor(rh, dtype=torch.long),
    )


@torch.no_grad()
def eval_relations(model: TrigonLM, task: RelationTask, relations: list,
                   batch: int = 64) -> dict:
    model.eval()
    task.eval()
    correct = 0
    total = 0
    confusion: dict[str, Counter] = {}
    for start in range(0, len(relations), batch):
        rows = relations[start:start + batch]
        left = torch.tensor([name_tensor(row.left) for row in rows], dtype=torch.long)
        right = torch.tensor([name_tensor(row.right) for row in rows], dtype=torch.long)
        labels = torch.tensor([row.label for row in rows], dtype=torch.long)
        lh = torch.tensor([head_for_name(row.left) for row in rows], dtype=torch.long)
        rh = torch.tensor([head_for_name(row.right) for row in rows], dtype=torch.long)
        phase, encoded = task(left, right, model.cycle)
        logits = task.classify(left, right, phase, model.cycle, lh, rh, encoded)
        predicted = logits.argmax(-1)
        correct += int((predicted == labels).sum())
        total += len(rows)
        for truth, guess in zip(labels.tolist(), predicted.tolist()):
            name = [key for key, value in RELATION_LABELS.items() if value == truth][0]
            confusion.setdefault(name, Counter())[
                [key for key, value in RELATION_LABELS.items() if value == guess][0]
            ] += 1
    return {"correct": correct, "total": total,
            "accuracy": correct / max(total, 1), "confusion": confusion}


@torch.no_grad()
def generate(model: TrigonLM, prompt: str, steps: int = 48,
             temperature: float = 0.82, topk: int = 40,
             context: int = 96, seed: int = 7) -> str:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    ids = [BOS] + list(prompt.encode("utf-8", errors="replace"))
    for _ in range(steps):
        window = ids[-context:]
        out = model(torch.tensor([window], dtype=torch.long))
        logits = out["logits"][0, -1] / temperature
        values, index = torch.topk(logits, topk)
        probs = F.softmax(values, dim=0)
        choice = torch.multinomial(probs, 1, generator=generator)
        token = int(index[choice])
        if token == EOS:
            break
        if token not in (PAD, BOS):
            ids.append(token)
    raw = bytes(token for token in ids[1:] if token < 256)
    return raw.decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE / "data" / "wordnet_hw4.jsonl.gz"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--relation-steps", type=int, default=120)
    parser.add_argument("--arch-generations", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default=str(BASE / "results"))
    args = parser.parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data)
    docs = build_documents(records)
    relations = build_relations(records)
    source = describe(args.data)
    source["matrix_digest"] = matrix_digest(records, relations)
    print(json.dumps({"corpus": source,
                      "documents": {k: len(v) for k, v in docs.items()},
                      "relations": {k: len(v) for k, v in relations.items()}},
                     indent=2), flush=True)

    model = TrigonLM(vocab=VOCAB, d_model=args.d_model, n_heads=4, n_layers=10)
    prior = apply_wordnet_prior(model.cycle, relations["train"])
    relation_task = RelationTask(d_model=64)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(relation_task.parameters()),
        lr=1.2e-3, weight_decay=0.01,
    )
    rng = random.Random(args.seed)
    history: list[dict] = []
    start_time = time.time()

    # Stage A: typed relation probe.  This is the HW4 matrix path: each example
    # writes a cycle and reads it back after a content-dependent phase update.
    rel_optimizer = torch.optim.AdamW(
        list(relation_task.parameters()) + list(model.cycle.parameters()),
        lr=2e-3, weight_decay=0.001,
    )
    rel_history = []
    for step in range(1, args.relation_steps + 1):
        relation_task.train()
        left, right, labels, lh, rh = relation_batch(
            relations["train"], 64, rng
        )
        rel_optimizer.zero_grad()
        phase, encoded = relation_task(left, right, model.cycle)
        logits = relation_task.classify(
            left, right, phase, model.cycle, lh, rh, encoded
        )
        counts = Counter(row.label for row in relations["train"])
        weights = torch.tensor(
            [1.0 / max(counts.get(label, 1), 1) for label in range(4)],
            dtype=torch.float32,
        )
        weights = weights / weights.sum() * 4.0
        loss = F.cross_entropy(logits, labels, weight=weights)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(relation_task.parameters()) + list(model.cycle.parameters()), 1.0
        )
        rel_optimizer.step()
        if step == 1 or step % 20 == 0 or step == args.relation_steps:
            val = eval_relations(model, relation_task, relations["val"])
            rel_history.append({"step": step, "loss": float(loss.item()),
                                "val_accuracy": val["accuracy"]})
            print("REL", step, round(float(loss.item()), 4),
                  "val", round(val["accuracy"], 4), flush=True)

    # Stage B: byte language modelling on WordNet glosses.
    best_bpb = float("inf")
    best_nats = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_genome = Genome()
    for step in range(1, args.steps + 1):
        model.train()
        relation_task.train()
        ids, labels = make_batch(docs["train"], rng, args.batch, args.context)
        optimizer.zero_grad()
        out = model(ids, labels=labels)
        loss = out["loss"]
        mem_loss = out.get("mem_loss", torch.tensor(0.0))
        head_loss = out.get("head_loss", torch.tensor(0.0))
        rel_loss = torch.tensor(0.0)
        if step % 4 == 0:
            left, right, rel_labels, lh, rh = relation_batch(
                relations["train"], 32, rng
            )
            phase, encoded = relation_task(left, right, model.cycle)
            rel_logits = relation_task.classify(
                left, right, phase, model.cycle, lh, rh, encoded
            )
            rel_loss = F.cross_entropy(rel_logits, rel_labels)
        total_loss = (loss + 0.15 * mem_loss + 0.10 * head_loss
                      + 0.08 * rel_loss)
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            val = eval_docs(model, docs["val"], limit=24)
            history.append({"step": step, "loss": float(loss.item()),
                            "mem_loss": float(mem_loss.item()),
                            "head_loss": float(head_loss.item()),
                            "rel_loss": float(rel_loss.item()),
                            **val})
            print("LM", step, "loss", round(float(loss.item()), 3),
                  "val_bpb", round(val["bits_per_token"], 3),
                  "top1", round(val["top1"], 3), flush=True)
            if val["bits_per_token"] < best_bpb:
                best_bpb = val["bits_per_token"]
                best_nats = val["nats_per_token"]
                best_state = copy.deepcopy(model.state_dict())
                best_genome = model.genome
                torch.save({
                    "model": best_state,
                    "genome": best_genome.to_dict(),
                    "config": {"d_model": args.d_model, "n_heads": 4,
                               "n_layers": 10, "vocab": VOCAB},
                    "seed": args.seed,
                }, out_dir / f"trigon_best_{args.seed}.pt")
                (out_dir / "progress.json").write_text(json.dumps({
                    "step": step, "best_bpb": best_bpb, "best_nats": best_nats,
                    "history": history, "relation_history": rel_history,
                    "route": HW4_ROUTE, "mode": "local-unverified",
                    "canonical_merge": False,
                }, indent=2, default=str))

    model.load_state_dict(best_state)
    model.set_genome(best_genome)
    print("BEST", best_bpb, "genome", best_genome.digest, flush=True)

    # Stage C: NANNET-P architecture evolution.
    nannet = NannetP(model, seed=args.seed)
    arch_batches = [make_batch(docs["val"], rng, 4, args.context) for _ in range(4)]
    arch_history = []
    for generation in range(args.arch_generations):
        state = nannet.state(
            model, float(best_nats), 0.0, generation
        )
        record = nannet.step(model, state, arch_batches, count=3, margin=1e-3)
        arch_history.append(record)
        print("NANNET-P", generation,
              "base", round(record["baseline_loss"], 4),
              "cand", round(record["candidate_loss"], 4),
              "accepted", record["accepted"], flush=True)
        nannet.save(out_dir / f"nannet_p_{args.seed}.json")
        (out_dir / "progress.json").write_text(json.dumps({
            "step": "arch", "generation": generation, "arch_history": arch_history,
            "best_bpb": best_bpb, "history": history,
            "relation_history": rel_history, "route": HW4_ROUTE,
            "mode": "local-unverified", "canonical_merge": False,
        }, indent=2, default=str))
    model.set_genome(nannet.best_genome)
    best_genome = nannet.best_genome

    # Stage D: final evaluations, gauge and ablations.
    test = eval_docs(model, docs["test"], limit=48)
    relation_test = eval_relations(model, relation_task, relations["test"])
    gauge_before = eval_docs(model, docs["test"], limit=24)
    model.cycle.gauge(scale=2.0)
    gauge_after = eval_docs(model, docs["test"], limit=24)
    model.cycle.clear_gauge()
    gauge_logits = None
    # Compare predictions exactly before/after a second gauge on a fixed batch.
    ids, labels = make_batch(docs["test"], rng, 4, args.context)
    with torch.no_grad():
        before = model(ids)["logits"].argmax(-1)
        model.cycle.gauge(scale=3.0)
        after = model(ids)["logits"].argmax(-1)
        model.cycle.clear_gauge()
    gauge_logits = int((before == after).sum()) / int(before.numel())

    ablations = {}
    for name in ("full", "no_write", "no_memory", "no_trig", "no_holonomy",
                 "no_heads", "no_copy"):
        old_genome = model.genome
        if name == "no_write":
            from dataclasses import replace
            model.set_genome(replace(old_genome, write_rate=0.0))
        elif name == "no_memory":
            model.ablate_memory = True
        elif name == "no_trig":
            model.ablate_trig = True
        elif name == "no_holonomy":
            from dataclasses import replace
            model.set_genome(replace(old_genome, holonomy_gain=0.0))
        elif name == "no_heads":
            model.ablate_heads = True
        elif name == "no_copy":
            from dataclasses import replace
            model.set_genome(replace(old_genome, copy_gain=0.0))
        ablations[name] = eval_docs(model, docs["test"], limit=24)
        model.ablate_memory = False
        model.ablate_trig = False
        model.ablate_heads = False
        model.set_genome(old_genome)
        print("ABLATION", name, round(ablations[name]["bits_per_token"], 4),
              flush=True)

    prompts = ["a person who", "a place where", "an instrument used for",
               "the act of", "a small animal"]
    samples = {prompt: generate(model, prompt, steps=40, seed=args.seed + index)
               for index, prompt in enumerate(prompts)}
    result = {
        "schema": "nanlm.trigon9/1",
        "route": HW4_ROUTE,
        "mode": "local-unverified",
        "live_nannet_route": False,
        "canonical_merge": False,
        "config": vars(args),
        "corpus": source,
        "documents": {key: len(value) for key, value in docs.items()},
        "relations": {key: len(value) for key, value in relations.items()},
        "prior": prior,
        "model": model.parameter_groups(),
        "relation_history": rel_history,
        "lm_history": history,
        "best_val_bpb": best_bpb,
        "best_genome": best_genome.to_dict(),
        "test": test,
        "relation_test": relation_test,
        "gauge": {
            "before": gauge_before,
            "after": gauge_after,
            "argmax_agreement": gauge_logits,
            "max_cycle_error": model.cycle.audit_gauge(
                torch.arange(0, 2048),
                torch.ones(2048, dtype=torch.float32),
            ),
        },
        "ablations": ablations,
        "nannet_p": arch_history,
        "samples": samples,
        "seconds": time.time() - start_time,
    }
    (out_dir / f"trigon_results_{args.seed}.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    (out_dir / f"samples_{args.seed}.json").write_text(
        json.dumps(samples, indent=2)
    )
    print("TEST", json.dumps(test), flush=True)
    print("REL_TEST", json.dumps(relation_test), flush=True)
    print("DONE", round(result["seconds"], 1), "s", flush=True)


if __name__ == "__main__":
    main()
