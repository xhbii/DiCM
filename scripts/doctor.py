#!/usr/bin/env python3
"""Display dependencies, CUDA availability, and configured assets."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicm.utils.assets import MODELS, model_source

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-cuda', action='store_true')
    a = p.parse_args()
    versions = {}
    missing = []
    for package in ('torch','torchvision','diffusers','transformers','accelerate','safetensors','numpy','Pillow','PyYAML'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    assets = {key: model_source(key) for key in MODELS}
    caption = Path(os.environ.get('DICM_COCO_CAPTIONS', 'data/captions_train2017.json'))
    cuda, gpu = False, None
    if 'torch' in versions:
        import torch
        cuda = torch.cuda.is_available()
        if cuda:
            gpu = torch.cuda.get_device_name(0)
    print(json.dumps(dict(python=sys.version.split()[0], dependencies=versions, missing=missing,
        cuda=cuda, gpu=gpu, models=assets, captions=str(caption), captions_available=caption.is_file()), indent=2))
    if missing or (a.require_cuda and not cuda):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
