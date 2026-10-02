"""Named module selection with scoped installation and exact weight restoration."""
from contextlib import contextmanager
from pathlib import Path
import torch
from dicm.module.sparse_delta import compose, dense_weights

class ModuleLibrary:
    def __init__(self, modules):
        self.modules = dict(modules)

    @classmethod
    def from_directory(cls, directory):
        files = sorted(Path(directory).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt modules in {directory}")
        return cls({p.stem: torch.load(p, map_location="cpu", weights_only=True) for p in files})

    @contextmanager
    def activate(self, unet, concepts):
        """Select each concept once; compose in canonical order and restore on exit.

        Call on the base denoiser. Like other in-place model edits, simultaneous
        activation contexts on the same denoiser require external serialization.
        """
        selected = list(concepts)
        if len(set(selected)) != len(selected):
            raise ValueError("A concept may be selected only once")
        payload = compose([self.modules[c] for c in sorted(selected)])
        params = dict(unet.named_parameters())
        for name, item in payload.items():
            if name not in params or tuple(params[name].shape) != tuple(item["shape"]):
                raise ValueError(f"Module does not match base parameter: {name}")
        weights = dense_weights(unet, payload)
        saved = {name: params[name].detach().clone() for name in weights}
        try:
            with torch.no_grad():
                for name, weight in weights.items():
                    params[name].copy_(weight)
            yield unet
        finally:
            with torch.no_grad():
                for name, weight in saved.items():
                    params[name].copy_(weight)
