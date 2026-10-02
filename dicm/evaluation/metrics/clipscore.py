"""Image-text similarity using CLIP ViT-L/14."""

from __future__ import annotations

import torch

from pathlib import Path
from dicm.utils.assets import model_source


class CLIPScorer:
    """Image-text cosine similarity. score() returns the raw cosine in [-1, 1]."""

    def __init__(self, model_path: str | None = None, device: str = "cuda",
                 scale: float = 1.0):
        from transformers import CLIPModel, CLIPProcessor

        path = model_source("clip", model_path)
        self.device = device
        self.scale = scale
        self.model = CLIPModel.from_pretrained(str(path), local_files_only=Path(path).is_dir()
                                               ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(str(path), local_files_only=Path(path).is_dir())

    @torch.no_grad()
    def score(self, image, text: str) -> float:
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt",
            padding="max_length", truncation=True,
        ).to(self.device)
        out = self.model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((img * txt).sum(dim=-1).item() * self.scale)

    @torch.no_grad()
    def image_embedding(self, image) -> torch.Tensor:
        """L2-normalized CLIP image embedding."""
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inputs)
        if not torch.is_tensor(emb):
            # transformers >= 5 returns the vision output object with
            # pooler_output already passed through visual_projection
            emb = emb.pooler_output
        return (emb / emb.norm(dim=-1, keepdim=True)).squeeze(0).cpu()
