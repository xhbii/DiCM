"""Frozen text embeddings, cached latents, and diffusion noise helpers."""

from __future__ import annotations

import torch

def _embed(pipe, prompt: str) -> torch.Tensor:
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return pipe.text_encoder(tok.input_ids.to(pipe.device))[0]

@torch.no_grad()
def cache_prompt_latents(pipe, prompts: list[str], steps: int, seed: int) -> dict[str, torch.Tensor]:
    """Short frozen generations used as on-distribution x0 for training."""
    out = {}
    for i, p in enumerate(prompts):
        gen = torch.Generator(device=pipe.device).manual_seed(seed + i)
        lat = pipe(
            p,
            num_inference_steps=steps,
            guidance_scale=7.5,
            output_type="latent",
            generator=gen,
        ).images
        if not torch.is_tensor(lat):
            lat = torch.as_tensor(lat)
        out[p] = lat.detach()
    return out

def _add_noise(scheduler, x0, noise, t):
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(x0, noise, t)
    # Fallback for schedulers without add_noise.
    alphas = scheduler.alphas_cumprod.to(device=x0.device, dtype=x0.dtype)
    a = alphas[t].view(-1, *([1] * (x0.ndim - 1)))
    return a.sqrt() * x0 + (1.0 - a).sqrt() * noise
