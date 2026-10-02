import importlib.util
import json
from pathlib import Path
import pytest
from dicm.experiments.subsets import CONCEPTS8, make_subsets
from dicm.evaluation.report import summarize
from dicm.utils.artifacts import write_json

ROOT = Path(__file__).resolve().parents[1]

def test_subset_plan_is_fixed_unique_and_covers_every_singleton():
    subsets = make_subsets()
    assert subsets == make_subsets()
    assert len(subsets) == len({tuple(s) for s in subsets}) == 39
    assert [sum(len(s)==k for s in subsets) for k in (1,2,4,6,8)] == [8,10,10,10,1]
    assert subsets[:8] == [[c] for c in CONCEPTS8]
    assert all(len(s)==len(set(s)) for s in subsets)


def test_resume_rejects_different_protocol_without_overwriting(tmp_path):
    path = tmp_path/'protocol.json'
    cfg = {'seed':17, 'plan':[(1,8)]}
    write_json(path, cfg)
    write_json(path, cfg)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='Protocol changed'):
        write_json(path, dict(cfg, seed=29))
    assert path.read_bytes() == original


def fixture_results(tmp_path, retain_only=False):
    subsets = make_subsets()
    results = {}
    for selected in subsets:
        results['+'.join(selected)] = dict(selected=selected,
            unselected=[c for c in CONCEPTS8 if c not in selected],
            single={} if retain_only else {c: {'erasure_success':0.9 if c in selected else 0.1} for c in CONCEPTS8},
            retain={'dino':0.8,'clip':0.25})
    write_json(tmp_path/'protocol.json',dict(arm='dicm',subsets=list(results),retain_only=retain_only))
    write_json(tmp_path/'results.json',results)
    return results


def test_report_preserves_worst_member_and_conflict_threshold(tmp_path):
    results = fixture_results(tmp_path)
    tag = next(tag for tag,r in results.items() if len(r['selected'])==2)
    concept = results[tag]['selected'][0]
    results[tag]['single'][concept]['erasure_success'] = 0.7
    write_json(tmp_path/'results.json',results)
    report = summarize(tmp_path)
    k2 = report['by_k'][1]
    assert k2['min_member'] == 0.7
    assert k2['conflicts'] == 1
    assert k2['n'] == k2['conflict_evaluable'] == 10
    assert k2['erasure'] == pytest.approx(0.89)
    assert k2['unselected_retention'] == pytest.approx(0.9)


def test_report_rejects_incomplete_run(tmp_path):
    results = fixture_results(tmp_path)
    results.pop('cat')
    write_json(tmp_path/'results.json',results)
    with pytest.raises(ValueError, match='complete'):
        summarize(tmp_path)


def test_retain_only_does_not_report_zero_erasure_or_false_no_conflict(tmp_path):
    fixture_results(tmp_path, retain_only=True)
    report = summarize(tmp_path)
    assert all(row['erasure'] is None and row['conflict_evaluable']==0 for row in report['by_k'])
    assert all(row['dino']==0.8 for row in report['by_k'])


def test_final_plan_uses_sibling_cohesion_wide_gate_and_reusable_outputs():
    for rq in ['rq1','rq2','rq3','rq4']:
        plan = json.loads((ROOT/'experiments'/f'{rq}.json').read_text())
        main = next(j for j in plan['jobs'] if j['name']=='dicm' and j['stage']=='train')
        args = main['args']
        assert args[args.index('--cohesion-siblings')+1] == 'all'
        assert '--wide-val' in args and '--subset-additivity' in args
        assert '--retrofit' not in args
        assert args[args.index('--out')+1] == 'outputs/main_s{seed}'
        assert args[args.index('--extra-retain')+1] == '72'
        for job in plan['jobs']:
            assert (ROOT/'scripts'/job['script']).is_file()


def test_runner_filters_stages_and_rejects_typos():
    spec = importlib.util.spec_from_file_location('run_rq', ROOT/'scripts/run_rq.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = module.commands('rq4',29,'train',['dicm'])
    assert len(selected)==1
    assert 'outputs/main_s29' in selected[0][1]
    command = module.commands('rq3',29,'train',['global-accept'])[0][1]
    assert command[command.index('--accept-max')+1] == '800'
    with pytest.raises(ValueError, match='Unknown jobs'):
        module.commands('rq1',17,names=['typo'])


def test_base_fingerprint_guard_distinguishes_precision_and_model(tmp_path):
    from dicm.utils.artifacts import verify_base
    verify_base(tmp_path, 'base-a', 'torch.float16')
    verify_base(tmp_path, 'base-a-fp32', 'torch.float32')
    with pytest.raises(ValueError):
        verify_base(tmp_path, 'changed', 'torch.float16')


def test_baseline_training_bootstraps_all_prefixes_without_prior_run(tmp_path, monkeypatch):
    import torch
    from torch import nn
    from types import SimpleNamespace
    from dicm.experiments import train_baselines as engine
    monkeypatch.setattr(engine, 'CONCEPTS', list(engine.CHAIN))
    model = nn.Module()
    model.dtype = torch.float32
    model.attn2 = nn.Linear(2,2,bias=False)
    model.requires_grad_(False)
    pipe = SimpleNamespace(unet=model,text_encoder=nn.Identity(),vae=nn.Identity())
    monkeypatch.setattr(engine, 'load_sd15_pipeline', lambda **kw: pipe)
    def train(pipe, concept, log):
        with torch.no_grad():
            pipe.unet.attn2.weight.add_(0.125)
        return 0.0
    def uce(pipe, concepts, name, log):
        return {'attn2.weight':pipe.unet.attn2.weight.detach().clone() + len(concepts)*0.125}
    monkeypatch.setattr(engine, 'train_esdx', train)
    monkeypatch.setattr(engine, 'run_uce', uce)
    manifest = engine.phase_train(tmp_path, lambda x: None)
    for kind in ('esd','uce'):
        assert all(f'{kind}/{c}' in manifest for c in engine.CHAIN)
        for k in range(2,9):
            item = manifest[f'{kind}/seq_'+'_'.join(engine.CHAIN[:k])]
            delta = torch.load(item['path'], weights_only=True)['attn2.weight']
            assert torch.allclose(delta.float(), torch.full_like(delta.float(), k*0.125), atol=1e-6)
    assert len(manifest) == 37
