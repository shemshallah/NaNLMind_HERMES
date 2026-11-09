"""NANNET-P: local architecture-evolution gate for TRIGON-9.

NANNET-P is the experimental *architecture promotion* protocol used by Round
042.  It is deliberately separate from the NANNET identity/authorization
service: in this repository a live NANNET route is not available, so every
record is marked ``local-unverified``.  The route string is the canonical HW4
evidence address, not a claim that a remote lease was issued.

The protocol is a bounded neural architecture search:

1. a policy samples a mutation of the current :class:`Genome`;
2. the candidate is evaluated on held-out validation batches;
3. the candidate is accepted only if loss improves and its parent digest still
   matches (a generation-CAS check);
4. accepted mutations increment the generation and are written to the history.

No candidate can enable canonical merge, rewrite the 1 GiB ceiling, or escape
the bounded gene ranges.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from trigon import Genome, TrigonLM

HW4_ROUTE = "nan://GM0:SC0:MC0:BS0:0:HW0:0::HW4:0"


class ArchPolicy(nn.Module):
    """Diagonal Gaussian policy over 24 continuous mutation coordinates."""

    def __init__(self, state_dim: int = 15, action_dim: int = 24):
        super().__init__()
        self.action_dim = action_dim
        self.body = nn.Sequential(
            nn.Linear(state_dim, 48), nn.Tanh(), nn.Linear(48, 48), nn.Tanh()
        )
        self.mu = nn.Linear(48, action_dim)
        self.log_sigma = nn.Linear(48, action_dim)
        with torch.no_grad():
            self.log_sigma.bias.fill_(-1.0)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(state)
        return self.mu(hidden), self.log_sigma(hidden).clamp(-3.0, 0.5)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_mutation(base: Genome, z: torch.Tensor,
                   generator: torch.Generator) -> Genome:
    z = z.detach().cpu()
    gates = [_clamp(base.layer_gates[i] + float(z[i]) * 0.25, 0.15, 1.5)
             for i in range(10)]
    gains = [_clamp(base.phase_gain[i] + float(z[10 + i]) * 0.25, 0.02, 1.0)
             for i in range(10)]
    write = _clamp(base.write_rate + float(z[20]) * 0.12, 0.01, 0.5)
    decay = _clamp(base.memory_decay + float(z[21]) * 0.03, 0.90, 0.999)
    head_k = int(_clamp(base.head_k + (2 if float(z[22]) > 0.0 else -2), 3, 12))
    order = list(base.order)
    if float(z[23]) > 0.25:
        a = int(torch.randint(0, 9, (1,), generator=generator).item())
        b = a + 1
        order[a], order[b] = order[b], order[a]
    return replace(
        base, layer_gates=tuple(gates), phase_gain=tuple(gains),
        write_rate=write, memory_decay=decay, head_k=head_k,
        order=tuple(order),
    )


class NannetP:
    def __init__(self, model: TrigonLM, lr: float = 3e-4,
                 route: str = HW4_ROUTE, seed: int = 17):
        self.model = model
        self.route = route
        self.generator = torch.Generator().manual_seed(seed)
        self.policy = ArchPolicy()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.generation = 0
        self.history: list[dict] = []
        self.accepted: list[dict] = []
        self.best_genome: Genome = model.genome

    def state(self, model: TrigonLM, loss: float, grad_norm: float,
              step: int) -> torch.Tensor:
        genes = list(model.genome.layer_gates)
        return torch.tensor(
            genes + [float(loss), float(grad_norm), step / 1000.0,
                     float(model.genome.write_rate),
                     float(model.genome.head_k) / 12.0],
            dtype=torch.float32,
        )

    @torch.no_grad()
    def evaluate(self, model: TrigonLM, genome: Genome,
                 batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        old = model.genome
        model.set_genome(genome)
        model.eval()
        total = 0.0
        count = 0
        for ids, labels in batches:
            out = model(ids, labels=labels)
            total += float(out["loss"].item()) * ids.shape[0] * ids.shape[1]
            count += ids.shape[0] * ids.shape[1]
        model.train()
        model.set_genome(old)
        return total / max(count, 1)

    def propose(self, model: TrigonLM, state: torch.Tensor,
                count: int = 3) -> list[tuple[Genome, torch.Tensor]]:
        mu, log_sigma = self.policy(state)
        sigma = torch.exp(log_sigma)
        proposals = []
        for _ in range(count):
            z = mu + sigma * torch.randn_like(mu)
            log_prob = (-0.5 * ((z - mu) / sigma) ** 2
                        - log_sigma - 0.5 * math.log(2 * math.pi)).sum()
            proposals.append((apply_mutation(model.genome, z, self.generator),
                              log_prob))
        return proposals

    def step(self, model: TrigonLM, state: torch.Tensor,
             batches: list[tuple[torch.Tensor, torch.Tensor]],
             count: int = 3, margin: float = 1e-4) -> dict:
        baseline = self.evaluate(model, model.genome, batches)
        parent_digest = model.genome.digest
        proposals = self.propose(model, state, count)
        scored = []
        for genome, log_prob in proposals:
            loss = self.evaluate(model, genome, batches)
            scored.append((loss, genome, log_prob))
        scored.sort(key=lambda item: item[0])
        best_loss, best, log_prob = scored[0]
        reward = (baseline - best_loss) / max(baseline, 1e-6)
        accepted = best_loss < baseline - margin and best.digest != parent_digest
        # Policy-gradient update: the reward is a validation loss improvement,
        # not a training loss or an unverified artifact number.
        self.optimizer.zero_grad()
        policy_loss = -(float(reward) * log_prob)
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        record = {
            "generation": self.generation,
            "parent_digest": parent_digest,
            "candidate_digest": best.digest,
            "baseline_loss": baseline,
            "candidate_loss": best_loss,
            "reward": reward,
            "accepted": bool(accepted),
            "route": self.route,
            "mode": "local-unverified",
            "live_nannet_route": False,
            "canonical_merge": False,
            "genes": best.to_dict(),
        }
        if accepted:
            model.set_genome(best)
            self.best_genome = best
            self.accepted.append(record)
        self.generation += 1
        self.history.append(record)
        return record

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "route": self.route,
            "mode": "local-unverified",
            "live_nannet_route": False,
            "canonical_merge": False,
            "generation": self.generation,
            "best_genome": self.best_genome.to_dict(),
            "accepted": self.accepted,
            "history": self.history,
        }, indent=2))


__all__ = ("NannetP", "ArchPolicy", "apply_mutation", "HW4_ROUTE")
