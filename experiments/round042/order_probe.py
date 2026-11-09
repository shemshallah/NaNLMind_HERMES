#!/usr/bin/env python3
"""Order-sensitive trigonometric probe for the Round 042 synthesis.

Three relation symbols are presented in different orders.  A commutative 2D
phase sum is invariant to permutation; an ordered SO(3) product is not.  The
probe is deliberately small and controlled: it tests the algebra that the
L7-L9 transport layers use, not open-domain language understanding.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanlmind_hermes.resident.rotation_holonomy import (  # noqa: E402
    compose3, givens, ordered_product, trace,
)

SYMBOLS = (0.25, 0.75, 1.25)  # quarter/half-turn angles, arbitrary but fixed
LABELS = {"support": 0, "refute": 1, "mixed": 2}


def phase_features(path: tuple[int, int, int]) -> list[float]:
    """Commutative phase readout: identical for every permutation."""
    theta = sum(SYMBOLS[index] for index in path)
    counts = [path.count(index) / 3.0 for index in range(3)]
    return [math.cos(theta), math.sin(theta), *counts]


def rotation_features(path: tuple[int, int, int]) -> list[float]:
    """Ordered SO(3) product readout: noncommutative and trace-bearing."""
    product = ordered_product([givens(index, SYMBOLS[index]) for index in path])
    values = [product[row][column] for row in range(3) for column in range(3)]
    return [*values, trace(product) / 3.0, sum(values) / 9.0]


def label_for(path: tuple[int, int, int]) -> int:
    if path == (0, 1, 2):
        return LABELS["support"]
    if path == (2, 1, 0):
        return LABELS["refute"]
    return LABELS["mixed"]


def make_data(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    support = (0, 1, 2)
    refute = (2, 1, 0)
    mixed = [(0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)]
    rows = []
    labels = []
    for index in range(count):
        label = index % 3
        path = support if label == 0 else refute if label == 1 else rng.choice(mixed)
        row = rotation_features(path) + phase_features(path)
        rows.append(row)
        labels.append(label)
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels)


def train_probe(seed: int, steps: int = 260) -> dict:
    torch.manual_seed(seed)
    train_x, train_y = make_data(900, seed)
    val_x, val_y = make_data(300, seed + 1)
    test_x, test_y = make_data(300, seed + 2)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        val_acc = float((model(val_x).argmax(-1) == val_y).float().mean())
        test_acc = float((model(test_x).argmax(-1) == test_y).float().mean())
        # Phase-only feature slice is sufficient for a separate linear probe.
        with torch.enable_grad():
            phase_model = nn.Linear(5, 3)
            phase_opt = torch.optim.AdamW(phase_model.parameters(), lr=0.02)
            phase_train = train_x[:, -5:]
            phase_val = val_x[:, -5:]
            phase_test = test_x[:, -5:]
            for _ in range(steps):
                phase_opt.zero_grad()
                F.cross_entropy(phase_model(phase_train), train_y).backward()
                phase_opt.step()
            phase_test_acc = float(
                (phase_model(phase_test).argmax(-1) == test_y).float().mean()
            )
    return {
        "seed": seed,
        "train": 900,
        "val": 300,
        "test": 300,
        "full_test_accuracy": test_acc,
        "phase_only_test_accuracy": phase_test_acc,
        "parameters": sum(p.numel() for p in model.parameters()),
        "phase_parameters": sum(p.numel() for p in phase_model.parameters()),
        "val_accuracy": val_acc,
    }


def main() -> None:
    results = [train_probe(seed) for seed in (17, 29, 43)]
    summary = {
        "schema": "nanlm.trigon-order-probe/1",
        "source": "controlled synthetic order probe",
        "claim_boundary": (
            "The phase-only control is mathematically permutation-invariant. "
            "This does not establish open-domain reasoning or a unique NaN "
            "advantage; a conventional sequence model can also solve it."
        ),
        "results": results,
    }
    out = Path(__file__).resolve().parent / "results" / "order_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
