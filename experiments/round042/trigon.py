"""TRIGON-9: trigonometric recursive invariant generative operator network.

Experimental reference implementation for the Round 042 synthesis.

This module is deliberately a *host-side reference harness*.  The lowest-level
native contract for the same geometry is the raw NaN16 image
``nanlmind_hermes/nan/trig_cycle.nan``; this file exists so the architecture can
be trained and falsified on ordinary hardware.  No claim is made that the
PyTorch reference is the native runtime.

The design keeps four things inseparable:

* the 256-head / 32,385-cycle gauge-invariant mirror memory of Rounds 030-041;
* a trigonometric state (phase + amplitude) in every L0-L9 layer;
* ordered SO(3) transport and closed-loop holonomy for relation order;
* content-dependent sparse writes, so memory changes as language is read.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

HEAD_COUNT = 256
CYCLE_HEADS = 255
CYCLE_COUNT = CYCLE_HEADS * (CYCLE_HEADS - 1) // 2
PAIR_COUNT = HEAD_COUNT * (HEAD_COUNT - 1) // 2
LAYER_NAMES = (
    "L0_bit", "L1_nibble", "L2_byte", "L3_phonetic", "L4_morpheme",
    "L5_lexical", "L6_completion", "L7_sentence", "L8_paragraph",
    "L9_narrative",
)
assert CYCLE_COUNT == 32385
assert PAIR_COUNT == 32640

# Root Boolean functions: class nibble n = (near_class << 2) | far_class.
ROOT_NAMES = (
    "ZERO", "NOR", "NOT_A_AND_B", "NOT_A", "A_AND_NOT_B", "NOT_B", "XOR",
    "NAND", "AND", "XNOR", "B", "A_OR_NOT_B", "A", "NOT_A_OR_B", "OR", "ONE",
)


def bool_class(mask: int, a: int, b: int) -> int:
    return (mask >> ((a << 1) | b)) & 1


def apply_bits(mask: int, left: int, right: int, width: int) -> int:
    out = 0
    for bit in range(width):
        out |= bool_class(mask, (left >> bit) & 1, (right >> bit) & 1) << bit
    return out


def _cycle_index(i: int, j: int) -> int:
    """Canonical index for 1 <= i < j <= 255."""
    if not (1 <= i < j <= 255):
        raise ValueError("cycle heads must satisfy 1 <= i < j <= 255")
    return (i - 1) * (510 - i) // 2 + (j - i - 1)


def _build_pair_tables() -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.full((HEAD_COUNT, HEAD_COUNT), -1, dtype=torch.long)
    sign = torch.zeros((HEAD_COUNT, HEAD_COUNT), dtype=torch.float32)
    for i in range(1, HEAD_COUNT):
        for j in range(1, HEAD_COUNT):
            if i == j:
                continue
            if i < j:
                index[i, j] = _cycle_index(i, j)
                sign[i, j] = 1.0
            else:
                index[i, j] = _cycle_index(j, i)
                sign[i, j] = -1.0
    return index, sign


PAIR_TO_CYCLE, PAIR_SIGN = _build_pair_tables()
CYCLE_I = torch.empty(CYCLE_COUNT, dtype=torch.long)
CYCLE_J = torch.empty(CYCLE_COUNT, dtype=torch.long)
for _i in range(1, CYCLE_HEADS):
    for _j in range(_i + 1, HEAD_COUNT):
        CYCLE_I[_cycle_index(_i, _j)] = _i
        CYCLE_J[_cycle_index(_i, _j)] = _j


# ---------------------------------------------------------------------------
# L0-L6 fixed trigonometric features
# ---------------------------------------------------------------------------
def _root_features(byte: int) -> list[float]:
    low = byte & 0x0F
    high = (byte >> 4) & 0x0F
    values: list[float] = []
    for mask in range(16):
        v = apply_bits(mask, low, high, 4)
        values.append(v / 16.0)
    return values


_ROOT_TABLE = torch.tensor(
    [_root_features(b) for b in range(256)], dtype=torch.float32
)


def byte_trig_features(ids: torch.Tensor) -> torch.Tensor:
    """Fixed L0-L4 features from raw byte identities.

    Returns ``[..., 80]``: 32 root phases, 16 nibble phases, 8 byte phases,
    6 phonetic flags, 12 morpheme phase features, 6 completion context flags.
    """
    byte = (ids % 256).long()
    root = _ROOT_TABLE.to(ids.device)[byte] * (2.0 * math.pi)
    root_cos = torch.cos(root)
    root_sin = torch.sin(root)

    nibble = torch.stack(
        [((byte >> shift) & 0x0F).float() / 16.0 for shift in (0, 4)], dim=-1
    ) * (2.0 * math.pi)
    nibble_cos = torch.cos(nibble)
    nibble_sin = torch.sin(nibble)

    byte_phase = byte.float() / 256.0 * (2.0 * math.pi)
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0], device=ids.device)
    byte_phases = byte_phase.unsqueeze(-1) * freqs
    byte_cos = torch.cos(byte_phases)
    byte_sin = torch.sin(byte_phases)

    is_alpha = ((byte >= 65) & (byte <= 90)) | ((byte >= 97) & (byte <= 122))
    lower = byte | 0x20
    is_vowel = is_alpha & torch.isin(
        lower, torch.tensor([97, 101, 105, 111, 117], device=ids.device)
    )
    is_space = byte == 32
    is_digit = (byte >= 48) & (byte <= 57)
    is_punct = (~is_alpha) & (~is_space) & (~is_digit)
    is_upper = (byte >= 65) & (byte <= 90)
    phonetic = torch.stack(
        [is_alpha, is_vowel, is_space, is_digit, is_punct, is_upper], dim=-1
    ).float()

    # Cheap rolling 3-byte hash phases for L4 morpheme-like units.
    b0 = byte.float()
    b1 = torch.roll(byte, 1, dims=-1).float()
    b2 = torch.roll(byte, 2, dims=-1).float()
    hashes = torch.stack(
        [
            (b0 * 31.0 + b1 * 7.0 + b2) % 61.0 / 61.0,
            (b0 * 17.0 + b1 * 13.0 + b2 * 3.0) % 53.0 / 53.0,
            (b0 * 5.0 + b1 * 29.0 + b2 * 19.0) % 47.0 / 47.0,
        ],
        dim=-1,
    ) * (2.0 * math.pi)
    morph_cos = torch.cos(hashes)
    morph_sin = torch.sin(hashes)

    # L6 completion flags: previous byte class and distance from word boundary.
    prev_space = torch.roll(is_space, 1, dims=-1)
    prev_alpha = torch.roll(is_alpha, 1, dims=-1)
    prev_vowel = torch.roll(is_vowel, 1, dims=-1)
    repeated = torch.roll(byte, 1, dims=-1) == byte
    completion = torch.stack(
        [prev_space, prev_alpha, prev_vowel, repeated, is_space, is_alpha], dim=-1
    ).float()

    return torch.cat(
        [
            root_cos, root_sin, nibble_cos, nibble_sin, byte_cos, byte_sin,
            phonetic, morph_cos, morph_sin, completion,
        ],
        dim=-1,
    )


L0_L6_FEATURES = 32 + 4 + 8 + 6 + 6 + 6


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Genome:
    """Bounded architecture genotype.

    Continuous genes can also be trained by gradient; discrete genes are
    mutated by the NANNET-P controller under a validation + generation-CAS
    gate.
    """

    layer_gates: tuple[float, ...] = (0.85,) * 10
    order: tuple[int, ...] = tuple(range(10))
    head_k: int = 6
    write_rate: float = 0.12
    read_temp: float = 1.0
    holonomy_gain: float = 0.10
    copy_gain: float = 0.15
    phase_gain: tuple[float, ...] = (0.25,) * 10
    memory_decay: float = 0.985

    def __post_init__(self) -> None:
        if len(self.layer_gates) != 10 or len(self.phase_gain) != 10:
            raise ValueError("genome needs exactly ten layer genes")
        if sorted(self.order) != list(range(10)):
            raise ValueError("order must be a permutation of L0-L9")
        if not 3 <= int(self.head_k) <= 12:
            raise ValueError("head_k must be 3..12")
        if not 0.0 <= float(self.write_rate) <= 0.5:
            raise ValueError("write_rate must be 0..0.5")
        if not 0.5 <= float(self.read_temp) <= 2.0:
            raise ValueError("read_temp must be 0.5..2.0")
        if not 0.0 <= float(self.memory_decay) <= 1.0:
            raise ValueError("memory_decay must be 0..1")

    @property
    def digest(self) -> str:
        raw = repr(self).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        return {
            "layer_gates": list(self.layer_gates),
            "order": list(self.order),
            "head_k": int(self.head_k),
            "write_rate": float(self.write_rate),
            "read_temp": float(self.read_temp),
            "holonomy_gain": float(self.holonomy_gain),
            "copy_gain": float(self.copy_gain),
            "phase_gain": list(self.phase_gain),
            "memory_decay": float(self.memory_decay),
            "digest": self.digest,
        }

    def mutated(self, rng: torch.Generator | None = None,
                scale: float = 0.08) -> "Genome":
        """Small bounded mutation; never an unbounded architecture escape."""
        vals = torch.rand(24, generator=rng) * 2.0 - 1.0
        gates = list(self.layer_gates)
        gains = list(self.phase_gain)
        order = list(self.order)
        for index in range(10):
            gates[index] = min(1.5, max(0.15, gates[index] + float(vals[index]) * scale))
            gains[index] = min(1.0, max(0.02, gains[index] + float(vals[10 + index]) * scale))
        k = int(self.head_k)
        if vals[20] > 0.55:
            k = min(12, k + 1)
        elif vals[20] < -0.55:
            k = max(3, k - 1)
        write = min(0.5, max(0.01, self.write_rate + float(vals[21]) * scale * 0.5))
        decay = min(0.999, max(0.90, self.memory_decay + float(vals[22]) * scale * 0.1))
        if vals[23] > 0.7:
            a = int(torch.randint(0, 9, (1,), generator=rng).item())
            b = (a + 1) % 10
            order[a], order[b] = order[b], order[a]
        return replace(
            self, layer_gates=tuple(gates), order=tuple(order), head_k=k,
            write_rate=write, memory_decay=decay, phase_gain=tuple(gains),
        )


# ---------------------------------------------------------------------------
# Gauge-invariant cycle memory
# ---------------------------------------------------------------------------
class CycleField(nn.Module):
    """Anchored triangle memory over 256 heads.

    ``phi(i,j) = e(0,i) + e(i,j) - e(0,j)`` is invariant under the additive
    vertex gauge ``e(i,j) -> e(i,j) + g_i - g_j``.  This is the exact algebra
    used by Rounds 030-041, now exposed as a differentiable fast-weight field.
    """

    def __init__(self, init_scale: float = 0.02):
        super().__init__()
        self.edge_phase = nn.Parameter(torch.randn(HEAD_COUNT, HEAD_COUNT) * init_scale)
        self.edge_logamp = nn.Parameter(torch.zeros(HEAD_COUNT, HEAD_COUNT))
        with torch.no_grad():
            self.edge_phase.fill_diagonal_(0.0)
            self.edge_logamp.fill_diagonal_(0.0)
        self.register_buffer("pair_to_cycle", PAIR_TO_CYCLE.clone(), persistent=False)
        self.register_buffer("pair_sign", PAIR_SIGN.clone(), persistent=False)
        self.register_buffer("cycle_i", CYCLE_I.clone(), persistent=False)
        self.register_buffer("cycle_j", CYCLE_J.clone(), persistent=False)
        self._last_gauge: torch.Tensor | None = None

    def gauge(self, generator: torch.Generator | None = None,
              scale: float = 1.0) -> torch.Tensor:
        """Apply a random additive vertex gauge; returns the g vector."""
        g = torch.randn(HEAD_COUNT, generator=generator) * float(scale)
        with torch.no_grad():
            self.edge_phase.add_(g[:, None] - g[None, :])
        self._last_gauge = g
        return g

    def clear_gauge(self) -> None:
        if self._last_gauge is not None:
            g = self._last_gauge
            with torch.no_grad():
                self.edge_phase.sub_(g[:, None] - g[None, :])
            self._last_gauge = None

    def base_cycle(self, index: torch.Tensor,
                   sign: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return base phase and log-amplitude for cycle indices."""
        i = self.cycle_i.to(index.device)[index]
        j = self.cycle_j.to(index.device)[index]
        phase = self.edge_phase[0, i] + self.edge_phase[i, j] - self.edge_phase[0, j]
        logamp = self.edge_logamp[0, i] + self.edge_logamp[i, j] - self.edge_logamp[0, j]
        return phase * sign, logamp

    def read(self, index: torch.Tensor, sign: torch.Tensor,
             weight: torch.Tensor, delta_phase: torch.Tensor | None = None,
             delta_logamp: torch.Tensor | None = None,
             temperature: float = 1.0, holonomy: bool = True
             ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read sparse cycles.

        ``index`` and ``sign`` are ``[B,T,P]``; ``weight`` is ``[B,T,P]``.
        Returns ``features [B,T,8]`` and the raw phase used for diagnostics.
        """
        base_phase, base_logamp = self.base_cycle(index, sign)
        phase = base_phase
        logamp = base_logamp
        if delta_phase is not None:
            flat_index = index.reshape(index.shape[0], -1)
            dp = torch.gather(delta_phase, 1, flat_index).reshape(index.shape)
            dl = torch.gather(delta_logamp, 1, flat_index).reshape(index.shape)
            phase = phase + dp * sign
            logamp = logamp + dl * sign
        phase = phase / float(temperature)
        logamp = torch.clamp(logamp, -8.0, 4.0)
        amp = torch.exp(logamp)
        w = weight / (weight.sum(dim=-1, keepdim=True) + 1e-8)
        re = (w * amp * torch.cos(phase)).sum(dim=-1)
        im = (w * amp * torch.sin(phase)).sum(dim=-1)
        mag = (w * amp).sum(dim=-1)
        phase_mean = (w * phase).sum(dim=-1)
        features = [re, im, mag, torch.cos(phase_mean), torch.sin(phase_mean)]
        if holonomy:
            # The first three active cycle pairs close a triangle in head
            # space; their phases are already gauge-invariant.
            tri = index[..., :3]
            tri_sign = sign[..., :3]
            a = self.base_cycle(tri[..., 0], tri_sign[..., 0])[0]
            b = self.base_cycle(tri[..., 1], tri_sign[..., 1])[0]
            c = self.base_cycle(tri[..., 2], tri_sign[..., 2])[0]
            if delta_phase is not None:
                flat = tri.reshape(index.shape[0], -1)
                ds = tri_sign.reshape(index.shape[0], -1)
                d = (torch.gather(delta_phase, 1, flat).reshape(tri.shape)
                     * ds.unsqueeze(1))
                a = a + d[..., 0]
                b = b + d[..., 1]
                c = c + d[..., 2]
            theta = a + b + c
            features.extend([torch.cos(theta), torch.sin(theta),
                             (torch.cos(theta) + 1.0) / 2.0])
        return torch.stack(features, dim=-1), phase

    def audit_gauge(self, index: torch.Tensor, sign: torch.Tensor,
                    scale: float = 2.0) -> dict:
        """Exact triangle-invariance audit on a sample of cycle indices."""
        before, _ = self.base_cycle(index, sign)
        g = self.gauge(scale=scale)
        after, _ = self.base_cycle(index, sign)
        self.clear_gauge()
        err = float((before - after).abs().max().item())
        return {"max_cycle_phase_error": err, "gauge_scale": scale,
                "gauge_norm": float(g.norm().item()), "exact": err < 1e-5}


# ---------------------------------------------------------------------------
# Trigonometric attention / rotor state / SO(3) holonomy
# ---------------------------------------------------------------------------
class TrigAttention(nn.Module):
    """Causal attention with a learned cosine phase-alignment bias."""

    def __init__(self, d_model: int, n_heads: int, phase_gain: float = 0.25):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.q_phase = nn.Linear(d_model, n_heads)
        self.k_phase = nn.Linear(d_model, n_heads)
        self.phase_gain = nn.Parameter(torch.full((n_heads,), float(phase_gain)))

    def forward(self, x: torch.Tensor, causal: bool = True,
                phase_scale: float = 1.0) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        content = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_head)
        qp = torch.tanh(self.q_phase(x)).transpose(1, 2) * math.pi
        kp = torch.tanh(self.k_phase(x)).transpose(1, 2) * math.pi
        phase_bias = torch.cos(qp[:, :, :, None] - kp[:, :, None, :])
        gain = (torch.sigmoid(self.phase_gain) * float(phase_scale))[None, :, None, None]
        score = content + gain * phase_bias
        if causal:
            mask = torch.triu(
                torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=1
            )
            score = score.masked_fill(mask[None, None], -1e9)
        attn = F.softmax(score, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, self.d_model)
        return self.out(out)


class TrigState(nn.Module):
    """Cumulative complex rotor state: amplitude * exp(i * cumulative phase)."""

    def __init__(self, d_model: int, n_freq: int = 8):
        super().__init__()
        self.n_freq = n_freq
        self.to_phase = nn.Linear(d_model, n_freq)
        self.to_amp = nn.Linear(d_model, n_freq)
        self.omega = nn.Parameter(torch.randn(n_freq) * 0.02)
        self.out = nn.Linear(2 * n_freq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = torch.cumsum(self.omega[None, None, :] + self.to_phase(x), dim=1)
        amp = F.softplus(self.to_amp(x)) + 1e-4
        z = torch.cat([amp * torch.cos(phase), amp * torch.sin(phase)], dim=-1)
        return self.out(z)


def _givens_batch(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Build a [B,3,3] Givens rotation from axis/angle [B]."""
    b = axis.shape[0]
    c = torch.cos(angle)
    s = torch.sin(angle)
    rotation = torch.eye(3, device=axis.device).expand(b, 3, 3).clone()
    pairs = [(1, 2), (2, 0), (0, 1)]
    for index, (first, second) in enumerate(pairs):
        mask = (axis == index).float()
        diagonal = mask * c + (1.0 - mask)
        off = mask * s
        rotation[:, first, first] = diagonal
        rotation[:, second, second] = diagonal
        rotation[:, first, second] = -off
        rotation[:, second, first] = off
    return rotation


class HolonomyCell(nn.Module):
    """Order-sensitive SO(3) transport; trace is conjugation invariant."""

    def __init__(self, d_model: int):
        super().__init__()
        self.to_angle = nn.Linear(d_model, 1)
        self.to_axis = nn.Linear(d_model, 3)
        self.out = nn.Linear(5, d_model)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        angle = torch.tanh(self.to_angle(x))[..., 0] * math.pi
        axis = self.to_axis(x).argmax(dim=-1)
        eye = torch.eye(3, device=x.device).expand(b, 3, 3).clone()
        state = eye
        traces = []
        for step in range(t):
            r = _givens_batch(axis[:, step], angle[:, step])
            state = state @ r
            traces.append(torch.stack(
                [state[:, 0, 0], state[:, 1, 1], state[:, 2, 2],
                 state[:, 0, 1], state[:, 1, 2]], dim=-1
            ))
        feat = torch.stack(traces, dim=1)
        return torch.sigmoid(self.gain) * self.out(feat)


class TrigBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, layer_index: int):
        super().__init__()
        self.layer_index = layer_index
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.attn = TrigAttention(d_model, n_heads)
        self.trig = TrigState(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.attn_gate = nn.Parameter(torch.tensor(0.7))
        self.trig_gate = nn.Parameter(torch.tensor(0.7))
        self.mlp_gate = nn.Parameter(torch.tensor(0.7))
        self.holonomy = HolonomyCell(d_model) if layer_index >= 7 else None

    def forward(self, x: torch.Tensor, layer_gate: float = 1.0,
                holonomy_gain: float = 0.1,
                phase_scale: float = 1.0) -> torch.Tensor:
        attn = self.attn(self.norm1(x), phase_scale=phase_scale)
        trig = self.trig(self.norm2(x))
        update = (torch.sigmoid(self.attn_gate) * attn
                  + torch.sigmoid(self.trig_gate) * trig)
        x = x + layer_gate * update
        x = x + layer_gate * torch.sigmoid(self.mlp_gate) * self.mlp(self.norm3(x))
        if self.holonomy is not None:
            x = x + layer_gate * float(holonomy_gain) * self.holonomy(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Head router and full model
# ---------------------------------------------------------------------------
class HeadRouter(nn.Module):
    """256 sparse head coordinates; heads are the memory write/read interface."""

    def __init__(self, d_model: int, head_dim: int = 32):
        super().__init__()
        self.head_dim = head_dim
        self.key = nn.Parameter(torch.randn(HEAD_COUNT, head_dim) * 0.02)
        self.phase = nn.Parameter(torch.rand(HEAD_COUNT) * 2.0 * math.pi)
        self.logamp = nn.Parameter(torch.zeros(HEAD_COUNT))
        self.query = nn.Linear(d_model, head_dim)

    def forward(self, h: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self.query(h) @ self.key.T / math.sqrt(self.head_dim)
        # Head 0 is the permanent root/hub; cycle pairs use heads 1..255.
        scores = scores.masked_fill(
            torch.arange(HEAD_COUNT, device=h.device)[None, None, :] == 0, -1e9
        )
        values, index = torch.topk(scores, k=k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(-1, index, True)
        a = F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
        ctx = torch.cat(
            [
                a @ self.key,
                a @ torch.cos(self.phase)[:, None],
                a @ torch.sin(self.phase)[:, None],
                a @ torch.exp(torch.clamp(self.logamp, -4, 4))[:, None],
            ],
            dim=-1,
        )
        return a, index, ctx


class TrigonLM(nn.Module):
    """TRIGON-9 reference language model."""

    def __init__(self, vocab: int = 259, d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 10, genome: Genome | None = None):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.genome = genome or Genome()
        self.embed = nn.Embedding(vocab, d_model)
        self.byte_feat = nn.Linear(L0_L6_FEATURES, d_model)
        self.layers = nn.ModuleList(
            TrigBlock(d_model, n_heads, i) for i in range(n_layers)
        )
        self.router = HeadRouter(d_model)
        self.cycle = CycleField()
        self.write_phase = nn.Linear(d_model + 1, 1)
        self.write_amp = nn.Linear(d_model + 1, 1)
        self.write_gate = nn.Linear(d_model, 1)
        self.mem_norm = nn.LayerNorm(8)
        self.mem_proj = nn.Linear(8 + 35, d_model)
        self.mem_gate = nn.Linear(d_model, d_model)
        self.copy_gate = nn.Linear(d_model, 1)
        self.head = nn.Linear(d_model, vocab)
        self.mem_head = nn.Linear(8, vocab)
        self.head_readout = nn.Linear(HEAD_COUNT, vocab)
        self.mem_head_gate = nn.Parameter(torch.tensor(-2.0))
        self.head_readout_gate = nn.Parameter(torch.tensor(-2.0))
        self.final_norm = nn.LayerNorm(d_model)
        self.ablate_memory = False
        self.ablate_trig = False
        self.ablate_heads = False

    # -- architecture application ---------------------------------------
    def set_genome(self, genome: Genome) -> None:
        self.genome = genome

    def _run_layers(self, h: torch.Tensor) -> torch.Tensor:
        for layer_index in self.genome.order:
            block = self.layers[layer_index]
            gate = self.genome.layer_gates[layer_index]
            if self.ablate_trig and block.trig is not None:
                # Keep the layer but remove its complex-rotor contribution.
                old = block.trig_gate.data.clone()
                block.trig_gate.data.fill_(-10.0)
                h = block(h, layer_gate=gate,
                          holonomy_gain=self.genome.holonomy_gain)
                block.trig_gate.data.copy_(old)
            else:
                h = block(h, layer_gate=gate,
                          holonomy_gain=self.genome.holonomy_gain)
        return h

    # -- pair construction ----------------------------------------------
    def _pair_tensors(self, index: torch.Tensor, head_act: torch.Tensor,
                      h: torch.Tensor, k: int):
        """Build pair indices, signs and weights for every token."""
        b, t, _ = index.shape
        hi = index[:, :, :, None].expand(b, t, k, k)
        hj = index[:, :, None, :].expand(b, t, k, k)
        pair_index = self.cycle.pair_to_cycle.to(h.device)[hi, hj]
        pair_sign = self.cycle.pair_sign.to(h.device)[hi, hj]
        act = torch.gather(head_act, -1, index)
        pair_weight = act[:, :, :, None] * act[:, :, None, :]
        tri = torch.triu(torch.ones(k, k, dtype=torch.bool, device=h.device), 1)
        mask = tri[None, None]
        valid = (pair_index >= 0) & mask
        # Flatten only the upper triangle.
        pair_index = pair_index[valid].reshape(b, t, -1)
        pair_sign = pair_sign[valid].reshape(b, t, -1)
        pair_weight = pair_weight[valid].reshape(b, t, -1)
        return pair_index, pair_sign, pair_weight

    def _memory_sequence(self, h: torch.Tensor, index: torch.Tensor,
                         head_act: torch.Tensor, k: int) -> torch.Tensor:
        b, t, _ = h.shape
        pair_index, pair_sign, pair_weight = self._pair_tensors(
            index, head_act, h, k
        )
        p = pair_index.shape[-1]
        # Content-dependent write phase/amplitude for every pair.
        pair_ctx = torch.cat(
            [
                h[:, :, None, :].expand(b, t, p, self.d_model),
                pair_weight[..., None],
            ],
            dim=-1,
        )
        write_phase = torch.tanh(self.write_phase(pair_ctx))[..., 0] * math.pi
        write_amp = torch.sigmoid(self.write_amp(pair_ctx))[..., 0]
        write_gate = torch.sigmoid(self.write_gate(h))[..., 0]
        rate = float(self.genome.write_rate)
        delta_phase = torch.zeros(b, CYCLE_COUNT, device=h.device)
        delta_logamp = torch.zeros(b, CYCLE_COUNT, device=h.device)
        reads = []
        for step in range(t):
            delta_phase = delta_phase * float(self.genome.memory_decay)
            delta_logamp = delta_logamp * float(self.genome.memory_decay)
            w = rate * write_gate[:, step, None] * pair_weight[:, step]
            phase_inc = (w * write_phase[:, step] * pair_sign[:, step]).reshape(b, -1)
            amp_inc = (w * torch.log(write_amp[:, step] + 1e-4)
                       * pair_sign[:, step]).reshape(b, -1)
            idx_step = pair_index[:, step].reshape(b, -1)
            delta_phase = delta_phase.scatter_add(1, idx_step, phase_inc)
            delta_logamp = delta_logamp.scatter_add(1, idx_step, amp_inc)
            read, _ = self.cycle.read(
                pair_index[:, step][:, None, :],
                pair_sign[:, step][:, None, :],
                pair_weight[:, step][:, None, :],
                delta_phase, delta_logamp,
                temperature=float(self.genome.read_temp),
            )
            reads.append(read[:, 0])
        return torch.stack(reads, dim=1)

    # -- forward --------------------------------------------------------
    def forward(self, ids: torch.Tensor, labels: torch.Tensor | None = None,
                return_aux: bool = False):
        h = self.embed(ids) + self.byte_feat(byte_trig_features(ids))
        h = self._run_layers(h)
        head_act, head_index, head_ctx = self.router(h, self.genome.head_k)
        if self.ablate_memory:
            mem = torch.zeros(h.shape[0], h.shape[1], 8, device=h.device)
        else:
            mem = self._memory_sequence(h, head_index, head_act, self.genome.head_k)
        mem = self.mem_norm(mem)
        mem_ctx = self.mem_proj(torch.cat([mem, head_ctx], dim=-1))
        h = h + torch.sigmoid(self.mem_gate(h)) * mem_ctx
        h = self.final_norm(h)
        logits = self.head(h)
        if not self.ablate_memory:
            logits = logits + torch.sigmoid(self.mem_head_gate) * self.mem_head(mem)
        if not self.ablate_heads:
            logits = logits + torch.sigmoid(self.head_readout_gate) * self.head_readout(head_act)
        # Pointer-copy channel: learned convex mixture with the context.
        copy_score = h @ h.transpose(-1, -2) / math.sqrt(self.d_model)
        mask = torch.triu(
            torch.ones(ids.shape[1], ids.shape[1], dtype=torch.bool, device=ids.device),
            diagonal=1,
        )
        copy_score = copy_score.masked_fill(mask, -1e9)
        copy_dist = F.softmax(copy_score, dim=-1)
        one_hot = F.one_hot(ids, self.vocab).float()
        copy_prob = copy_dist @ one_hot
        copy_gate = torch.sigmoid(self.copy_gate(h)) * float(self.genome.copy_gain)
        logits = logits + copy_gate * torch.log(copy_prob + 1e-6)
        out = {"logits": logits, "mem": mem, "head_act": head_act,
               "head_index": head_index, "copy_gate": copy_gate}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, self.vocab), labels.reshape(-1),
                ignore_index=-100,
            )
            # Auxiliary pressures that force the sparse heads and the dynamic
            # cycle read to carry next-token information instead of remaining
            # decorative residual channels.
            if mem.shape[1] > 1 and not self.ablate_memory:
                out["mem_loss"] = F.cross_entropy(
                    self.mem_head(mem[:, :-1]).reshape(-1, self.vocab),
                    labels[:, 1:].reshape(-1), ignore_index=-100,
                )
            if head_act.shape[1] > 1 and not self.ablate_heads:
                out["head_loss"] = F.cross_entropy(
                    self.head_readout(head_act[:, :-1]).reshape(-1, self.vocab),
                    labels[:, 1:].reshape(-1), ignore_index=-100,
                )
        return out

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Encode text without memory writes, for relation/entity tasks."""
        h = self.embed(ids) + self.byte_feat(byte_trig_features(ids))
        h = self._run_layers(h)
        return h.mean(dim=1)

    def parameter_groups(self) -> dict[str, int]:
        groups = {
            "embedding": sum(p.numel() for p in self.embed.parameters()),
            "byte_features": sum(p.numel() for p in self.byte_feat.parameters()),
            "layers": sum(p.numel() for p in self.layers.parameters()),
            "router": sum(p.numel() for p in self.router.parameters()),
            "cycle": sum(p.numel() for p in self.cycle.parameters()),
            "readout": sum(p.numel() for p in self.head.parameters())
                       + sum(p.numel() for p in self.mem_proj.parameters()),
        }
        groups["total"] = sum(p.numel() for p in self.parameters())
        return groups


# ---------------------------------------------------------------------------
# Auxiliary WordNet relation task: write, then read back, then classify.
# ---------------------------------------------------------------------------
RELATION_NAMES = ("none", "hypernym", "hyponym", "opposes")


def head_for_name(name: str) -> int:
    digest = hashlib.sha256(name.casefold().encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:4], "big") % CYCLE_HEADS


class RelationTask(nn.Module):
    """A controlled typed-evidence probe over the shared cycle field."""

    def __init__(self, d_model: int = 64, n_classes: int = 4):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(259, d_model)
        self.conv = nn.Conv1d(d_model, d_model, 3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.phase_write = nn.Linear(2 * d_model, 1)
        self.classifier = nn.Sequential(
            nn.Linear(4 * d_model + 5, 128), nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids).transpose(1, 2)
        x = F.gelu(self.conv(x)).transpose(1, 2)
        return self.norm(x).mean(dim=1)

    def forward(self, left: torch.Tensor, right: torch.Tensor,
                cycle: CycleField) -> tuple[torch.Tensor, dict]:
        hl = self.encode(left)
        hr = self.encode(right)
        # Entity names are mapped to heads by the same stable hash used by the
        # data layer; the task receives those indices separately.
        pair_phase = torch.tanh(self.phase_write(torch.cat([hl, hr], -1)))[:, 0] * math.pi
        return pair_phase, {"left": hl, "right": hr}

    def classify(self, left: torch.Tensor, right: torch.Tensor,
                 pair_phase: torch.Tensor, cycle: CycleField,
                 left_heads: torch.Tensor, right_heads: torch.Tensor,
                 encoded: dict | None = None) -> torch.Tensor:
        hl = encoded["left"] if encoded else self.encode(left)
        hr = encoded["right"] if encoded else self.encode(right)
        index = cycle.pair_to_cycle.to(left.device)[left_heads, right_heads]
        sign = cycle.pair_sign.to(left.device)[left_heads, right_heads]
        valid = index >= 0
        index = index.clamp_min(0)
        sign = sign * valid.float()
        pair_phase = pair_phase * valid.float()
        b = left.shape[0]
        delta_phase = torch.zeros(b, CYCLE_COUNT, device=left.device)
        delta_amp = torch.zeros(b, CYCLE_COUNT, device=left.device)
        delta_phase = delta_phase.scatter_add(
            1, index[:, None], pair_phase[:, None]
        )
        weight = valid.float()[:, None]
        # Read the just-written cycle without the three-head closure block.
        features, _ = cycle.read(
            index[:, None, None], sign[:, None, None], weight[:, :, None],
            delta_phase, delta_amp, holonomy=False,
        )
        features = features[:, 0]
        pair = torch.cat([hl, hr, hl * hr, (hl - hr).abs(), features], dim=-1)
        return self.classifier(pair)
