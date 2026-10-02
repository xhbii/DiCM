#!/usr/bin/env python3
"""Train two small real SD1.5 modules and exercise composition on one GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--out', default='outputs/smoke')
    a = p.parse_args()
    if a.steps < 1 or a.budget < 1:
        p.error('steps and budget must be positive')
    import torch
    from dicm.models.sd15_wrapper import load_sd15_pipeline
    from dicm.experiments import train_library as engine
    from dicm.module.conflict_bench import configure, fill, fingerprint, kv_layers
    from dicm.module.latents import _embed, cache_prompt_latents
    from dicm.module.library import ModuleLibrary
    from dicm.module.sparse_delta import payload_count
    from dicm.utils.artifacts import write_json
    if not torch.cuda.is_available():
        raise RuntimeError('The training smoke test requires CUDA')
    torch.set_num_threads(4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pipe = load_sd15_pipeline()
    for part in (pipe.unet, pipe.text_encoder, pipe.vae):
        part.eval().requires_grad_(False)
    configure(['cat', 'dog'])
    engine.TRAIN_CONTEXTS[:] = ['a photo of {a}']
    engine.NEUTRAL[:] = ['car', 'chair']
    retain = ['a small wooden table near a window']
    prompts = [fill(engine.TRAIN_CONTEXTS[0], c) for c in ['cat','dog','car','chair']] + retain
    print('Caching five training latents', flush=True)
    latents = cache_prompt_latents(pipe, prompts, 2, 17)
    with torch.no_grad():
        embeddings = {p: _embed(pipe, p).detach() for p in prompts + ['']}
    layers = kv_layers()
    cfg = dict(engine.CFG, steps=a.steps, budget=a.budget, retain_batch=1,
               retain_weight=8, cohesion_weight=4, lock_draws=1, accept=False,
               subset=True, cohesion_siblings='all')
    base = fingerprint(pipe.unet)
    locks, info = engine.retain_lock(pipe, layers, latents, embeddings, retain, cfg)
    modules, histories = {}, {}
    first_hash = None
    for concept in ['cat', 'dog']:
        library = dict(concepts=list(modules), payloads=list(modules.values())) if modules else None
        bank, history, _ = engine.train_module(pipe, layers, concept, latents, embeddings, retain,
                                              library, locks, cfg, lambda x: print(x, flush=True))
        count = payload_count(bank)
        assert 0 < count <= a.budget, count
        for name, item in bank.items():
            assert not locks[name].cpu().reshape(-1)[item['indices']].any()
            assert torch.isfinite(item['values']).all()
        assert fingerprint(pipe.unet) == base
        modules[concept] = bank
        histories[concept] = history
        module_dir = out / 'modules'
        module_dir.mkdir(exist_ok=True)
        torch.save(bank, module_dir / f'{concept}.pt')
        if concept == 'cat':
            first_hash = hashlib.sha256((module_dir / 'cat.pt').read_bytes()).hexdigest()
    assert hashlib.sha256((module_dir / 'cat.pt').read_bytes()).hexdigest() == first_hash
    library = ModuleLibrary.from_directory(module_dir)
    def render():
        return pipe('a cat and a dog beside a wooden table', num_inference_steps=2,
                    generator=torch.Generator('cuda').manual_seed(101)).images[0]
    with library.activate(pipe.unet, ['cat', 'dog']):
        image = render()
        image.save(out / 'composed.png')
    assert fingerprint(pipe.unet) == base
    with library.activate(pipe.unet, ['dog', 'cat']):
        reverse = render()
    assert image.tobytes() == reverse.tobytes()
    assert fingerprint(pipe.unet) == base
    write_json(out / 'validation.json', dict(passed=True, steps_per_module=a.steps,
        support={k: payload_count(v) for k,v in modules.items()}, locks=info,
        base_fingerprint=base, exact_restore=True, order_equal_images=True,
        frozen_first_module=True, seconds=time.time()-start, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), histories=histories))
    print(f'SMOKE PASSED: {out / "validation.json"}', flush=True)

if __name__ == '__main__':
    main()
