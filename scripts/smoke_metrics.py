#!/usr/bin/env python3
"""Check the detector, DINO, and CLIP on a generated smoke-test image."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', default='outputs/smoke/composed.png')
    p.add_argument('--out', default='outputs/smoke/metrics.json')
    a = p.parse_args()
    import math
    import torch
    from PIL import Image
    from dicm.evaluation.metrics.clipscore import CLIPScorer
    from dicm.evaluation.metrics.dinov2 import DINOv2Similarity
    from dicm.module.conflict_bench import Detector
    from dicm.utils.artifacts import write_json
    torch.set_num_threads(4)
    im = Image.open(a.image).convert('RGB')
    clip = CLIPScorer().score(im, 'a cat and a dog beside a wooden table')
    torch.cuda.empty_cache()
    dino = DINOv2Similarity().similarity(im, im)
    torch.cuda.empty_cache()
    detection = Detector('cuda').scores(im, ['cat','dog'])
    assert math.isfinite(clip) and -1 <= clip <= 1
    assert abs(dino-1) < 1e-5
    assert all(0 <= value <= 1 for value in detection.values())
    write_json(a.out, dict(passed=True, clip=clip, dino_self_similarity=dino, detector=detection))
    print(f'Metrics smoke passed: {a.out}')

if __name__ == '__main__':
    main()
