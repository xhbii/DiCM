"""DINOv2 image similarity."""

from __future__ import annotations

import torch

from pathlib import Path
from dicm.utils.assets import model_source


class DINOv2Similarity:
    def __init__(self, model_path: str | None = None, device: str = "cuda"):
        from transformers import AutoImageProcessor, AutoModel

        path = model_source("dino", model_path)
        self.device = device
        self.model = AutoModel.from_pretrained(str(path), local_files_only=Path(path).is_dir()
                                               ).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(str(path),
                                                            local_files_only=Path(path).is_dir())

    @torch.no_grad()
    def embedding(self, image) -> torch.Tensor:
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state[:, 0]
        return (emb / emb.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

    def similarity(self, image_a, image_b) -> float:
        ea, eb = self.embedding(image_a), self.embedding(image_b)
        return float((ea * eb).sum().item())
