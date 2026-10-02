"""Shared denoiser interface for SD1.5 and SDXL."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SD15_LAYERS = Path(__file__).resolve().parents[1] / "configs/kv_layers_sd15.json"


def _last_content_token(tok) -> int:
    """Index of the final non-special token, which is the concept word itself.

    UCE solves with this vector, not with the end-of-text embedding. The distinction does not
    matter much on SD1.5 but it does on SDXL, where the end-of-text position carries no concept
    information and the resulting edit moves the weights without erasing anything.
    """
    return max(int(tok.attention_mask[0].sum().item()) - 2, 0)


class Backbone:
    key: str = ""
    resolution: int = 512
    latent_channels: int = 4
    steps: int = 20
    guidance: float = 7.5
    dtype = torch.float16

    def __init__(self, device: str = "cuda", dtype=None):
        self.device = device
        if dtype is not None:
            self.dtype = dtype
        self.pipe = self._load()
        for part in self._frozen_parts():
            part.eval().requires_grad_(False)

    # ---- to implement per backbone
    def _load(self):
        raise NotImplementedError

    def _frozen_parts(self):
        raise NotImplementedError

    @property
    def denoiser(self):
        raise NotImplementedError

    def kv_layers(self) -> list[str]:
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, prompt: str) -> dict:
        raise NotImplementedError

    def predict(self, x, t, cond) -> torch.Tensor:
        raise NotImplementedError

    # ---- shared
    @property
    def latent_size(self) -> int:
        return self.resolution // 8

    def scheduler(self):
        return self.pipe.scheduler

    def add_noise(self, x0, noise, t):
        return self.pipe.scheduler.add_noise(x0, noise, t)

    @torch.no_grad()
    def generate(self, prompt: str, seed: int):
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(prompt, num_inference_steps=self.steps, guidance_scale=self.guidance,
                         width=self.resolution, height=self.resolution, generator=gen).images[0]

    @torch.no_grad()
    def cache_latents(self, prompts: list[str], steps: int, seed: int) -> dict:
        """Short frozen generations used as on-distribution x0 for training."""
        out = {}
        for i, p in enumerate(prompts):
            gen = torch.Generator(device=self.device).manual_seed(seed + i)
            lat = self.pipe(p, num_inference_steps=steps, guidance_scale=self.guidance,
                            width=self.resolution, height=self.resolution,
                            output_type="latent", generator=gen).images
            out[p] = (lat if torch.is_tensor(lat) else torch.as_tensor(lat)).detach()
        return out

    def fingerprint(self) -> str:
        """Hash of the denoiser weights, independent of state_dict key order.

        Registering a parametrization re-registers `weight` after `bias`, so a module whose
        projections carry a bias changes key order without changing any value.
        """
        import hashlib

        digest = hashlib.sha256()
        for name, weight in sorted(self.denoiser.state_dict().items()):
            digest.update(name.encode())
            digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @torch.no_grad()
    def kv_context(self, prompt: str) -> torch.Tensor:
        """The vector a concept contributes on the input side of the K/V projections.

        Closed-form editing (UCE) solves in this space, so it has to match the tensor the
        projection actually consumes, which differs per backbone.
        """
        raise NotImplementedError

    def kv_stats(self) -> dict:
        params = dict(self.denoiser.named_parameters())
        layers = self.kv_layers()
        return dict(tensors=len(layers), coordinates=sum(params[n].numel() for n in layers),
                    denoiser_params=sum(p.numel() for p in self.denoiser.parameters()))


class SD15(Backbone):
    key, resolution = "sd15", 512

    def _load(self):
        from dicm.models.sd15_wrapper import load_sd15_pipeline

        return load_sd15_pipeline(device=self.device, dtype=self.dtype)

    def _frozen_parts(self):
        return [self.pipe.unet, self.pipe.text_encoder, self.pipe.vae]

    @property
    def denoiser(self):
        return self.pipe.unet

    def kv_layers(self):
        # the frozen 18-tensor list the SD1.5 results were produced with
        return json.loads(SD15_LAYERS.read_text())["kv"]

    @torch.no_grad()
    def encode(self, prompt):
        tok = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        emb = self.pipe.text_encoder(tok.input_ids.to(self.device))[0]
        return dict(encoder_hidden_states=emb.detach())

    def predict(self, x, t, cond):
        return self.denoiser(x, t, encoder_hidden_states=cond["encoder_hidden_states"]).sample

    @torch.no_grad()
    def kv_context(self, prompt):
        tok = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        h = self.pipe.text_encoder(tok.input_ids.to(self.device))[0][0]
        return h[_last_content_token(tok)].float()


class SDXL(Backbone):
    key, resolution = "sdxl", 1024

    def _load(self):
        from diffusers import DDIMScheduler, StableDiffusionXLPipeline

        from dicm.utils.assets import model_source

        path = model_source("sdxl")
        pipe = StableDiffusionXLPipeline.from_pretrained(str(path), torch_dtype=self.dtype,
                                                         local_files_only=Path(path).is_dir(), add_watermarker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        return pipe.to(self.device)

    def _frozen_parts(self):
        return [self.pipe.unet, self.pipe.text_encoder, self.pipe.text_encoder_2, self.pipe.vae]

    @property
    def denoiser(self):
        return self.pipe.unet

    def kv_layers(self):
        return [n for n, _ in self.denoiser.named_parameters()
                if "attn2" in n and (n.endswith("to_k.weight") or n.endswith("to_v.weight"))]

    @torch.no_grad()
    def encode(self, prompt):
        emb, _, pooled, _ = self.pipe.encode_prompt(prompt=prompt, device=self.device,
                                                    num_images_per_prompt=1, do_classifier_free_guidance=False)
        size = (self.resolution, self.resolution)
        time_ids = torch.tensor([list(size) + [0, 0] + list(size)], device=self.device, dtype=emb.dtype)
        return dict(encoder_hidden_states=emb.detach(),
                    added_cond_kwargs=dict(text_embeds=pooled.detach(), time_ids=time_ids))

    def predict(self, x, t, cond):
        return self.denoiser(x, t, encoder_hidden_states=cond["encoder_hidden_states"],
                             added_cond_kwargs=cond["added_cond_kwargs"]).sample

    @torch.no_grad()
    def kv_context(self, prompt):
        tok = self.pipe.tokenizer(prompt, padding="max_length", max_length=self.pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        emb, _, _, _ = self.pipe.encode_prompt(prompt=prompt, device=self.device,
                                               num_images_per_prompt=1, do_classifier_free_guidance=False)
        return emb[0, _last_content_token(tok)].float()


BACKBONES = {"sd15": SD15, "sdxl": SDXL}


def load_backbone(key: str, device: str = "cuda", dtype=None) -> Backbone:
    """Load a backbone. `dtype=torch.float32` is used for baseline fine-tuning, which
    diverges immediately on fp16 weights because there are no master weights."""
    if key not in BACKBONES:
        raise ValueError(f"unknown backbone {key}; have {sorted(BACKBONES)}")
    return BACKBONES[key](device=device, dtype=dtype)
