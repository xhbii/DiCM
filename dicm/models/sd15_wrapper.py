"""Stable Diffusion 1.5 with the DDIM scheduler used by the experiments."""
from pathlib import Path
import torch
from dicm.utils.assets import model_source

SD15_LATENT_SHAPE = (1, 4, 64, 64)

def load_sd15_pipeline(device="cuda", dtype=None, model_path=None):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    source = model_source("sd15", model_path)
    dtype = dtype or (torch.float32 if str(device) == "cpu" else torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained(
        source, torch_dtype=dtype, local_files_only=Path(source).is_dir(),
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)
