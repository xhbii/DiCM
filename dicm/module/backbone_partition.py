"""Coordinate protection derived from retained-prediction sensitivity."""

import torch

def lock_from_scores(scores: dict[str, torch.Tensor], fraction: float) -> dict[str, torch.Tensor]:
    """Global top-`fraction` most sensitive locations become backbone=True (must keep)."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    flats = [(name, tensor.reshape(-1).float()) for name, tensor in scores.items()]
    packed = torch.cat([item[1] for item in flats])
    k = max(1, int(round(fraction * packed.numel())))
    thresh = torch.topk(packed, k, largest=True).values.min()
    return {name: (tensor >= thresh) for name, tensor in scores.items()}

def backbone_stats(locks: dict[str, torch.Tensor]) -> dict:
    locked = sum(int(mask.sum()) for mask in locks.values())
    total = sum(mask.numel() for mask in locks.values())
    return dict(locked=locked, total=total, fraction=locked / max(total, 1))
