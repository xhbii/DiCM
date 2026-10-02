"""Resolve local model snapshots or Hugging Face model identifiers."""
import os
from pathlib import Path

MODELS = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "clip": "openai/clip-vit-large-patch14",
    "dino": "facebook/dinov2-large",
}

def model_source(key, override=None):
    value = str(override or os.environ.get(f"DICM_{key.upper()}_MODEL") or MODELS[key])
    path = Path(value).expanduser()
    if path.is_dir():
        return str(path.resolve())
    if path.is_absolute() or value.startswith((".", "~")):
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    return value
