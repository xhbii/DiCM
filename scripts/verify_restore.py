#!/usr/bin/env python3
"""Check base-snapshot assembly and exact unloading for an existing module library."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modules', required=True)
    p.add_argument('--out', default='outputs/restore.json')
    a = p.parse_args()
    import torch
    from dicm.models.sd15_wrapper import load_sd15_pipeline
    from dicm.module.library import ModuleLibrary
    from dicm.module.conflict_bench import fingerprint
    from dicm.module.latents import _embed
    from dicm.utils.artifacts import write_json
    torch.set_num_threads(4)
    pipe = load_sd15_pipeline()
    pipe.unet.requires_grad_(False)
    library = ModuleLibrary.from_directory(a.modules)
    names = sorted(library.modules)
    before = fingerprint(pipe.unet)
    with torch.no_grad():
        emb = _embed(pipe, 'a photo of a cat')
        x = torch.randn((1,4,64,64), device='cuda', dtype=pipe.unet.dtype,
                        generator=torch.Generator('cuda').manual_seed(17))
        t = torch.tensor([500], device='cuda')
        def predict():
            return pipe.unet(x,t,encoder_hidden_states=emb).sample.detach().cpu()
        base_prediction = predict()
        selections = [names[:1], names[:4], names[-2:], names, [], names[:1]]
        snapshots, rows = {}, []
        for selected in selections:
            with library.activate(pipe.unet, selected):
                digest, prediction = fingerprint(pipe.unet), predict()
            assert fingerprint(pipe.unet) == before
            assert torch.equal(predict(), base_prediction)
            with library.activate(pipe.unet, list(reversed(selected))):
                assert fingerprint(pipe.unet) == digest
                assert torch.equal(predict(), prediction)
            key = tuple(sorted(selected))
            if key in snapshots:
                assert snapshots[key] == digest
            snapshots[key] = digest
            rows.append(dict(selected=selected, exact_restore=True, order_equal=True))
        assert fingerprint(pipe.unet) == before
    write_json(a.out, dict(passed=True, base_fingerprint=before, selections=rows))
    print(f'Restore checks passed: {a.out}')

if __name__ == '__main__':
    main()
