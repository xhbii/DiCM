"""Sparse-delta modules: budgeted sparse weight updates on frozen K/V projections.

A module is (support S, values D) on each target weight: W_eff = W + S ⊙ D.
Composition of modules is the sum of their sparse deltas (order-independent; disjoint or
overlapping supports both allowed). A deletion mask is the special case D = -W.

Training uses iterative hard thresholding: after every optimizer step the delta is projected
onto the global top-`budget` entries by magnitude (locked backbone entries are always zero).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn.utils import parametrize


class SparseDeltaParam(nn.Module):
    """Parametrization: weight + own delta (mode a) [+ frozen library delta (mode union)]."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(weight.shape, device=weight.device, dtype=torch.float32))
        self.register_buffer("library", torch.zeros(weight.shape, device=weight.device, dtype=torch.float32))
        self.mode = "a"  # a | b | union | off
        self.enabled = True

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mode == "off":
            return weight
        if self.mode == "a":
            extra = self.delta
        elif self.mode == "b":
            extra = self.library
        else:
            extra = self.delta + self.library
        return weight + extra.to(weight.dtype)


@contextmanager
def sparse_delta_supports(unet: nn.Module, layers: list[str]):
    named = dict(unet.named_modules())
    mods, installed = {}, []
    try:
        for name in layers:
            linear = named[name.removesuffix(".weight")]
            if parametrize.is_parametrized(linear, "weight"):
                raise ValueError(f"already parametrized: {name}")
            p = SparseDeltaParam(linear.weight)
            parametrize.register_parametrization(linear, "weight", p)
            installed.append(linear)
            mods[name] = p
        yield mods
    finally:
        for linear in reversed(installed):
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)


def set_mode(mods, mode: str) -> None:
    if mode not in {"a", "b", "union", "off"}:
        raise ValueError(f"Unknown composition mode: {mode}")
    for m in mods.values():
        m.mode = mode


@contextmanager
def disabled(mods):
    previous = {name: m.enabled for name, m in mods.items()}
    for m in mods.values():
        m.enabled = False
    try:
        yield
    finally:
        for name, m in mods.items():
            m.enabled = previous[name]


def install_library(mods, library: dict[str, dict] | None) -> None:
    """library: name -> {indices, values} (sparse payload) summed over the installed modules."""
    with torch.no_grad():
        for name, m in mods.items():
            m.library.zero_()
            if library is not None and name in library:
                item = library[name]
                m.library.reshape(-1)[item["indices"].to(m.library.device)] = item["values"].to(m.library.device, torch.float32)


def project(mods, budget: int, locks: dict[str, torch.Tensor] | None) -> dict:
    """Keep at most `budget` coordinates globally, including when magnitudes tie."""
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if not mods:
        return dict(nonzero=0, norm=0.0)
    with torch.no_grad():
        if locks is not None:
            for name, m in mods.items():
                m.delta[locks[name]] = 0.0
        flat = torch.cat([m.delta.reshape(-1).abs() for m in mods.values()])
        if not torch.isfinite(flat).all():
            raise ValueError("Cannot project non-finite module values")
        if int((flat > 0).sum()) > budget:
            keep = torch.zeros_like(flat, dtype=torch.bool)
            if budget:
                keep[torch.topk(flat, budget, largest=True).indices] = True
            offset = 0
            for m in mods.values():
                n = m.delta.numel()
                m.delta.reshape(-1).masked_fill_(~keep[offset:offset+n], 0)
                offset += n
        nz = sum(int((m.delta != 0).sum()) for m in mods.values())
        norm = float(torch.sqrt(sum((m.delta.float() ** 2).sum() for m in mods.values())))
    return dict(nonzero=nz, norm=norm)


def export(mods) -> dict[str, dict]:
    out = {}
    with torch.no_grad():
        for name, m in mods.items():
            flat = m.delta.reshape(-1)
            idx = (flat != 0).nonzero(as_tuple=False).flatten()
            out[name] = dict(shape=tuple(m.delta.shape), indices=idx.cpu(), values=flat[idx].detach().cpu().to(torch.float32))
    return out


def payload_count(payload: dict[str, dict]) -> int:
    return sum(int(item["indices"].numel()) for item in payload.values())


def compose(payloads: list[dict[str, dict]]) -> dict[str, dict]:
    """Sum of sparse deltas (union of supports, values added where they overlap)."""
    out = {}
    if not payloads:
        return out
    if any(set(p) != set(payloads[0]) for p in payloads):
        raise ValueError("Modules have different target layers")
    for name in payloads[0]:
        shape = payloads[0][name]["shape"]
        numel = 1
        for s in shape:
            numel *= s
        dense = torch.zeros(numel, dtype=torch.float32)
        for p in payloads:
            if tuple(p[name]["shape"]) != tuple(shape):
                raise ValueError(f"Module shape mismatch: {name}")
            dense.index_add_(0, p[name]["indices"].cpu(), p[name]["values"].to("cpu", torch.float32))
        idx = (dense != 0).nonzero(as_tuple=False).flatten()
        out[name] = dict(shape=shape, indices=idx, values=dense[idx])
    return out


def overlap_count(a: dict[str, dict], b: dict[str, dict]) -> int:
    n = 0
    for name in a:
        sa = set(a[name]["indices"].tolist())
        n += sum(1 for i in b[name]["indices"].tolist() if i in sa)
    return n


def dense_weights(unet: nn.Module, payload: dict[str, dict]) -> dict[str, torch.Tensor]:
    """Materialize W + delta for weight_override (fp32 add, cast back to the parameter dtype)."""
    params = dict(unet.named_parameters())
    out = {}
    with torch.no_grad():
        for name, item in payload.items():
            w = params[name].detach().float().clone()
            w.reshape(-1)[item["indices"].to(w.device)] += item["values"].to(w.device, torch.float32)
            out[name] = w.to(params[name].dtype)
    return out
